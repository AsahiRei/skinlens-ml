"""Shared constants and the tf.data input pipeline for SkinLens training / export / evaluation.

Preprocessing contract (the app must do exactly the same before calling the TFLite model):
  1. decode the photo as RGB (apply EXIF orientation),
  2. center-crop the largest square,
  3. resize to 224x224 (bilinear),
  4. feed raw pixel values 0..255 (float32 for the float models, uint8 for the int8 model).
Normalisation to [-1, 1] happens inside the model, so the app never has to scale pixels.
"""
import csv, os

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "split_manifest.csv")
MODEL_DIR = os.environ.get("SKINLENS_MODEL_DIR", os.path.join(ROOT, "model_effb0"))   # override for smoke tests
CLASSES = ["acne", "eczema", "oily", "dry", "normal"]   # output index order of the model
SEVERITY = ["mild", "moderate", "severe"]               # output order of the severity head; normal tiles have none (-1)
IMG = 224          # model input side
CACHE_SIDE = 256   # train images are cached at this side, then randomly cropped down to IMG
SEED = 42


def read_split(split, manifests=None, with_severity=False):
    """Return (paths, labels[, severities]) for one split, concatenated over one or more manifests
    (default: split_manifest.csv). Relative manifest paths are resolved against ROOT.
    Severity is the SEVERITY index, or -1 for "none" (normal) / a manifest without a severity column."""
    paths, labels, sevs = [], [], []
    for m in manifests or [MANIFEST]:
        with open(os.path.join(ROOT, m), encoding="utf8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    paths.append(os.path.join(ROOT, row["filepath"]))
                    labels.append(CLASSES.index(row["class"]))
                    sev = row.get("severity") or "none"
                    sevs.append(SEVERITY.index(sev) if sev in SEVERITY else -1)
    return (paths, labels, sevs) if with_severity else (paths, labels)


def center_square(img):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    s = tf.minimum(h, w)
    return tf.image.crop_to_bounding_box(img, (h - s) // 2, (w - s) // 2, s, s)


def load_square(path, side):
    """Decode a JPEG, center-crop the square, resize to side x side. Returns uint8."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = center_square(img)
    img = tf.image.resize(img, (side, side), method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _augment(img, label):
    """img: uint8 CACHE_SIDE square. Zoom-in only (random crop 75-100%), flips, small rotation,
    mild photometric jitter, plus phone-capture variation: small white-balance shift, JPEG
    re-compression, sensor noise. No hue rotation (redness matters for acne/eczema), no blur and
    no down-scaling (fine texture - pores, shine, flaking - separates oily/dry/normal)."""
    img = tf.cast(img, tf.float32)
    frac = tf.random.uniform([], 0.75, 1.0)
    s = tf.cast(frac * CACHE_SIDE, tf.int32)
    img = tf.image.random_crop(img, tf.stack([s, s, 3]))
    img = tf.image.resize(img, (IMG, IMG), method="bilinear", antialias=True)
    img = tf.image.random_flip_left_right(img)
    # white balance: warm/cool shift via opposing red/blue gains (+-4%), green untouched
    t = tf.random.uniform([], -0.04, 0.04)
    img = img * tf.stack([1.0 + t, 1.0, 1.0 - t])
    img = tf.image.random_brightness(img, 0.08 * 255)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.clip_by_value(img, 0, 255)
    # JPEG re-compression (half the images, quality 60-95), like camera / messaging-app output
    img = tf.cond(tf.random.uniform([]) < 0.5,
                  lambda: tf.image.random_jpeg_quality(img / 255.0, 60, 95) * 255.0,
                  lambda: img)
    img.set_shape((IMG, IMG, 3))
    # sensor noise (30% of images, sigma up to 3 grey levels)
    img = tf.cond(tf.random.uniform([]) < 0.3,
                  lambda: img + tf.random.normal(tf.shape(img), 0.0, tf.random.uniform([], 0.0, 3.0)),
                  lambda: img)
    return tf.clip_by_value(img, 0, 255), label


def make_dataset(split, batch=32, training=False, cache=True, manifests=None):
    """Elements are (image, (class, severity)); returns (dataset, class labels, severity labels)."""
    paths, labels, sevs = read_split(split, manifests, with_severity=True)
    ds = tf.data.Dataset.from_tensor_slices((paths, (labels, sevs)))
    side = CACHE_SIDE if training else IMG
    ds = ds.map(lambda p, y: (load_square(p, side), y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        rot = tf.keras.layers.RandomRotation(10 / 360, fill_mode="reflect", seed=SEED)
        ds = ds.batch(batch).map(lambda x, y: (rot(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE), labels, sevs


def macro_f1(y_true, y_pred, n=len(CLASSES)):
    import numpy as np
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)), f1s


def report(y_true, y_pred, title="", classes=CLASSES):
    """Per-class precision/recall/F1 + confusion matrix as text."""
    import numpy as np
    CLASSES = classes
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    mf1, f1s = macro_f1(y_true, y_pred, n)
    lines = [f"== {title}", f"accuracy {np.mean(y_true == y_pred):.4f}   macro-F1 {mf1:.4f}", "",
             f"{'class':8s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'n':>5s}"]
    for c in range(n):
        prec = cm[c, c] / max(cm[:, c].sum(), 1)
        rec = cm[c, c] / max(cm[c].sum(), 1)
        lines.append(f"{CLASSES[c]:8s} {prec:6.3f} {rec:6.3f} {f1s[c]:6.3f} {cm[c].sum():5d}")
    lines += ["", "confusion (rows = true, cols = predicted)", " " * 9 + " ".join(f"{c[:6]:>6s}" for c in CLASSES)]
    for c in range(n):
        lines.append(f"{CLASSES[c]:8s} " + " ".join(f"{v:6d}" for v in cm[c]))
    return "\n".join(lines)
