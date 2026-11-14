"""Cut the collages in dataset/raw (2 or 3 rows x 5 columns, one person per collage) into single-face
tiles and split them train/valid/test BY COLLAGE, so the same person never lands in two splits.

Column order in every collage: Acne, Eczema, Oily, Normal, Dry (checked by eye; 1-15 have 2 rows, 16-20 have 3).
Tiles are found from the white gutters / label strips, not hard-coded coordinates.

Rows are severity levels: 3-row collages are mild / moderate / severe (top to bottom), 2-row collages
(1-15, 33, 34) are mild / severe. Normal has no severity ("none"). Severity is stored in the filename and in
the manifest's "severity" column; there are no severity folders.

Output (dataset/raw is only read; only dataset/{train,valid,test} are deleted and rewritten):
  dataset/{train,valid,test}/{class}/v2_<collage>_r<row>_<severity>_<class>.jpg
  split_manifest.csv       the manifest train.py reads

Usage:
  python -P -E scripts/split_datasetv2.py [--src dataset/raw] [--valid 2 --test 2]
"""
import argparse, csv, glob, hashlib, os, random, shutil, sys

from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dataset")
MANIFEST = os.path.join(ROOT, "split_manifest.csv")
COLUMNS = ["acne", "eczema", "oily", "normal", "dry"]   # left to right in every collage
SEVERITY = {2: ["mild", "severe"], 3: ["mild", "moderate", "severe"]}   # by number of rows, top to bottom
FIELDS = ["filepath", "class", "severity", "split", "source", "original_path", "group_id", "person_id",
          "face_detected", "width", "height", "sha256", "phash"]


def runs(mask, min_len):
    """[start, end) runs of True in a 1-D bool array, at least min_len long."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def grid(row_spans, col_spans):
    return [[(x0, y0, x1, y1) for x0, x1 in col_spans] for y0, y1 in row_spans]


# Collages whose layout the white-gutter detector can't read (white background, borderless tiles or black
# grid lines). Boxes were read off the uniform grid lines and checked by eye.
OVERRIDES = {
    "36": grid([(10, 269), (273, 535), (540, 785)],
               [(35, 282), (284, 531), (533, 780), (782, 1028), (1031, 1275)]),
    "41": grid([(4, 254), (259, 513), (515, 768)],
               [(1, 259), (277, 534), (551, 809), (827, 1084), (1103, 1361)]),
    "43": grid([(43, 318), (320, 624), (626, 943)],
               [(35, 250), (251, 469), (469, 689), (690, 911), (912, 1128)]),
    "44": grid([(43, 318), (320, 625), (626, 943)],
               [(35, 250), (251, 468), (469, 689), (690, 911), (912, 1128)]),
    # white background merges with the 1-2 px gutters (x 274/550/825/1101, y 254/514)
    "56": grid([(0, 254), (255, 514), (515, 768)],
               [(0, 274), (275, 550), (551, 825), (826, 1101), (1102, 1376)]),
}

# Columns to leave out of a collage after a manual / colour check (the other columns are still used).
EXCLUDE = {
    "62": {"normal"},   # normal column has warmer white balance and a beige background (colour shortcut)
}


def find_tiles(img):
    """Return N rows (2 or 3) x 5 boxes (x0, y0, x1, y1) of the photo tiles."""
    a = np.asarray(img.convert("L"), dtype=np.int16)
    white = a > 235
    # a row belongs to a photo when most of it is not white (label strips are mostly white with thin text)
    rows = runs(white.mean(1) < 0.5, 60)
    if len(rows) not in (2, 3):
        raise ValueError(f"expected 2 or 3 photo rows, found {rows}")
    tiles = []
    for y0, y1 in rows:
        cols = runs(white[y0:y1].mean(0) < 0.9, 60)      # gutters are white over the whole row height
        if len(cols) != 5:
            raise ValueError(f"expected 5 photo columns in rows {y0}-{y1}, found {cols}")
        tiles.append([(x0, y0, x1, y1) for x0, x1 in cols])
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(OUT, "raw"))
    ap.add_argument("--valid", type=int, default=2, help="collages (people) in valid")
    ap.add_argument("--test", type=int, default=2, help="collages (people) in test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.jpg")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in files]
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    split_of = {c: "test" for c in order[:args.test]}
    split_of.update({c: "valid" for c in order[args.test:args.test + args.valid]})
    split_of.update({c: "train" for c in order[args.test + args.valid:]})

    for s in ("train", "valid", "test"):            # never touch dataset/raw
        d = os.path.join(OUT, s)
        if os.path.isdir(d):
            shutil.rmtree(d)
    rows, bad = [], []
    for path, cid in zip(files, ids):
        img = Image.open(path).convert("RGB")
        try:
            grid = OVERRIDES[cid] if cid in OVERRIDES else find_tiles(img)
        except ValueError as e:
            bad.append(f"{cid}.jpg: {e}")
            continue
        split = split_of[cid]
        levels = SEVERITY[len(grid)]
        for r, line in enumerate(grid):
            for (x0, y0, x1, y1), cls in zip(line, COLUMNS):
                if cls in EXCLUDE.get(cid, ()):
                    continue
                sev ="none" if cls == "normal" else levels[r]
                # trim 2 px so no gutter / anti-aliased border is left on the tile
                tile = img.crop((x0 + 2, y0 + 2, x1 - 2, y1 - 2))
                rel = f"dataset/{split}/{cls}/v2_{int(cid):02d}_r{r}_{sev}_{cls}.jpg"
                dst = os.path.join(ROOT, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tile.save(dst, quality=95)
                rows.append({"filepath": rel, "class": cls, "severity": sev, "split": split, "source": "datasetv2",
                             "original_path": f"{cid}.jpg r{r}c{COLUMNS.index(cls)}", "group_id": f"v2_{cid}",
                             "person_id": f"v2_{cid}", "face_detected": "", "width": tile.width,
                             "height": tile.height, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                             "phash": ""})
    if bad:
        print("could not parse:\n  " + "\n  ".join(bad))
        sys.exit(1)

    with open(MANIFEST, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for s in ("train", "valid", "test"):
        people = sorted((c for c in ids if split_of[c] == s), key=int)
        counts = {c: sum(r["split"] == s and r["class"] == c for r in rows) for c in COLUMNS}
        print(f"{s:5s} collages {','.join(people):22s} tiles {sum(counts.values()):3d}  {counts}")
    for s in ("train", "valid", "test"):
        sev = Counter(r["severity"] for r in rows if r["split"] == s)
        print(f"{s:5s} severity " + "  ".join(f"{k} {sev[k]}" for k in ("mild", "moderate", "severe", "none")))
    print(f"wrote {len(rows)} tiles to {OUT} and {MANIFEST}")


if __name__ == "__main__":
    main()
