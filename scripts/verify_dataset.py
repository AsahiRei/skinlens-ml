"""Verify dataset/{train,valid,test} (made by split_datasetv2.py) against split_manifest.csv: counts table, and no overlap between splits.

Checks (all computed from the files on disk, not trusted from the manifest):
  - manifest rows == files on disk, every file decodes as RGB JPEG with shorter side >= 128
  - no sha256 shared by two files at all
  - no pHash (or h-flipped pHash) within distance 4 across different splits
  - no original_path, leakage group_id or person_id in more than one split
Usage: python -P -E scripts/verify_dataset.py
"""
import csv, hashlib, os, sys
from collections import Counter, defaultdict

import numpy as np
import imagehash
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["acne", "eczema", "oily", "dry", "normal"]
SPLITS = ["train", "valid", "test"]


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "split_manifest.csv"), encoding="utf8")))
    ok = True

    on_disk = {os.path.relpath(os.path.join(r, f), ROOT).replace("\\", "/")
               for s in SPLITS for r, _, fs in os.walk(os.path.join(ROOT, "dataset", s)) for f in fs}
    listed = {r["filepath"] for r in rows}
    if on_disk != listed:
        ok = False
        print(f"FAIL manifest/disk mismatch: {len(on_disk - listed)} unlisted, {len(listed - on_disk)} missing")
    for r in rows:
        parts = r["filepath"].split("/")
        if parts[1] != r["split"] or parts[2] != r["class"] or not parts[3].endswith(f"_{r['class']}.jpg"):
            ok = False; print("FAIL path/label mismatch", r["filepath"])
        sev = r.get("severity")
        if sev is not None and (not parts[3].endswith(f"_{sev}_{r['class']}.jpg")
                                or (sev == "none") != (r["class"] == "normal")
                                or sev not in ("mild", "moderate", "severe", "none")):
            ok = False; print("FAIL severity mismatch", r["filepath"], sev)

    # counts table
    cnt = Counter((r["class"], r["split"]) for r in rows)
    print(f"\n{'class':8s} " + " ".join(f"{s:>14s}" for s in SPLITS) + f" {'total':>7s}")
    for c in CLASSES:
        tot = sum(cnt[(c, s)] for s in SPLITS)
        print(f"{c:8s} " + " ".join(f"{cnt[(c, s)]:6d} ({cnt[(c, s)] / tot:5.1%})" for s in SPLITS) + f" {tot:7d}")
    tot_all = len(rows)
    print(f"{'all':8s} " + " ".join(f"{sum(cnt[(c, s)] for c in CLASSES):6d} ({sum(cnt[(c, s)] for c in CLASSES) / tot_all:5.1%})"
                                    for s in SPLITS) + f" {tot_all:7d}\n")

    # re-hash every output file
    sha, ph, phf, split_of = [], [], [], []
    for r in rows:
        p = os.path.join(ROOT, r["filepath"])
        data = open(p, "rb").read()
        im = Image.open(p)
        if im.format != "JPEG" or im.mode != "RGB" or min(im.size) < 128:
            ok = False; print("FAIL format", r["filepath"], im.format, im.mode, im.size)
        im = im.convert("RGB")
        sha.append(hashlib.sha256(data).hexdigest())
        ph.append(int(str(imagehash.phash(im)), 16))
        phf.append(int(str(imagehash.phash(ImageOps.mirror(im))), 16))
        split_of.append(r["split"])

    dup_sha = [h for h, n in Counter(sha).items() if n > 1]
    print(f"exact duplicate files (sha256): {len(dup_sha)}"); ok &= not dup_sha

    ph = np.array(ph, dtype=np.uint64); phf = np.array(phf, dtype=np.uint64); sp = np.array(split_of)
    pc = lambda x: np.unpackbits(x.astype(np.uint64).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
    cross = 0
    for i in range(len(rows)):
        j = np.arange(i + 1, len(rows))
        j = j[sp[j] != sp[i]]
        d = np.minimum(pc(ph[i] ^ ph[j]), pc(ph[i] ^ phf[j]))
        for jj in j[d <= 4]:
            cross += 1
            if cross <= 5:
                print("  near-dup across splits:", rows[i]["filepath"], rows[jj]["filepath"])
    print(f"near-duplicate pairs across splits (pHash<=4): {cross}"); ok &= cross == 0

    for key in ("original_path", "group_id", "person_id"):
        seen = defaultdict(set)
        for r in rows:
            for v in filter(None, r[key].split(";")):
                seen[v].add(r["split"])
        bad = [v for v, s in seen.items() if len(s) > 1]
        print(f"{key} values in more than one split: {len(bad)}"); ok &= not bad

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
