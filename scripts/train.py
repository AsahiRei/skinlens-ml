"""Train the SkinLens 5-class classifier (acne, eczema, oily, dry, normal) on EfficientNet (EfficientNetB0 by default),
with a second head for severity (mild, moderate, severe) that shares the backbone. The severity head learns only
from non-normal tiles (sample weight 0 for normal); its labels come from the collage row (split_manifest.csv "severity").

Stage 1: frozen ImageNet backbone, train the head.
Stage 2: unfreeze the top of the backbone (BatchNorm stays in inference mode), cosine-decayed LR.
The checkpoint with the best validation score = mean(condition macro-F1, severity macro-F1) is kept. Afterwards a
softmax temperature is fitted per head on the validation set (calibration) and the test set is evaluated once.

Outputs go to one folder per backbone (an existing folder is moved to model_previous/ first):
  model_effb0/        efficientnetb0 on dataset/ (split_manifest.csv), the default run
  model_effv2b0/      efficientnetv2b0 on dataset/
Files in each folder:
  skinlens.keras     logits model (input: float32 RGB 0..255, 224x224), outputs {"logits": [5], "severity": [3]}
  calibration.json   {"temperature": T, "severity_temperature": Ts}; export_tflite.py bakes softmax(logits / T) in
  report.txt         validation and test metrics
  history.csv        per-epoch metrics

Usage (CPU with the defaults: EfficientNetB0 ~65 min):
  python -P -E scripts/train.py                                   # EfficientNetB0, dataset/
  python -P -E scripts/train.py --backbone efficientnetv2b0       # EfficientNetV2B0, dataset/
  After training, export_tflite.py runs automatically into <model folder>/tflite/ (skip with --no-export),
  and a running scripts/serve.py reloads the new model by itself.
  other options: --head-epochs 6 --ft-epochs 30 --unfreeze-from <layer> --model-dir DIR (one backbone only)
"""
import argparse, csv, json, os, shutil, subprocess, sys, time

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skinlens_common import CLASSES, IMG, MODEL_DIR, ROOT, SEED, SEVERITY, macro_f1, make_dataset, report

N = len(CLASSES)
NS = len(SEVERITY)


# backbone -> (constructor, first layer fine-tuned in stage 2 (~top 40% of blocks), needs [-1,1] rescaling here)
BACKBONES = {
    "efficientnetb0": (keras.applications.EfficientNetB0, "block5a_expand_conv", False),     # normalises 0..255 itself
    "efficientnetv2b0": (keras.applications.EfficientNetV2B0, "block5a_expand_conv", False),
}
DIR_SUFFIX = {"efficientnetb0": "_effb0", "efficientnetv2b0": "_effv2b0"}

# manifests relative to ROOT; split_manifest.csv comes from scripts/split_datasetv2.py
MANIFESTS = ["split_manifest.csv"]


def build_model(dropout, backbone="efficientnetb0"):
    ctor, _, rescale = BACKBONES[backbone]
    inp = keras.Input((IMG, IMG, 3), name="image")             # raw RGB 0..255 for every backbone
    x = keras.layers.Rescaling(1 / 127.5, offset=-1, name="to_minus1_1")(inp) if rescale else inp
    base = ctor(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)                                # BN always in inference mode
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    out = keras.layers.Dense(N, name="logits")(x)
    sev = keras.layers.Dense(NS, name="severity")(x)
    return keras.Model(inp, {"logits": out, "severity": sev}, name="skinlens"), base


def sev_scores(sev_true, sev_pred):
    """Severity macro-F1 / accuracy over the tiles that have a severity (non-normal)."""
    m = np.asarray(sev_true) >= 0
    t, p = np.asarray(sev_true)[m], np.asarray(sev_pred)[m]
    mf1, f1s = macro_f1(t, p, NS)
    return mf1, f1s, float(np.mean(t == p)) if len(t) else 0.0


class BestMacroF1(keras.callbacks.Callback):
    """Every epoch: val score = mean(condition macro-F1, severity macro-F1); keep the best weights,
    stop after `patience` epochs without improvement."""

    def __init__(self, val_ds, val_y, val_s, path, patience, log):
        super().__init__()
        self.val_ds, self.val_y, self.path, self.patience, self.log = val_ds, np.array(val_y), path, patience, log
        self.val_s = np.array(val_s)
        self.best, self.bad, self.stage = -1.0, 0, ""

    def on_epoch_end(self, epoch, logs=None):
        out = self.model.predict(self.val_ds, verbose=0)
        logits = out["logits"]
        mf1, f1s = macro_f1(self.val_y, logits.argmax(1))
        acc = float(np.mean(logits.argmax(1) == self.val_y))
        s_mf1, s_f1s, s_acc = sev_scores(self.val_s, out["severity"].argmax(1))
        score = (mf1 + s_mf1) / 2
        row = {"stage": self.stage, "epoch": epoch + 1, "loss": logs.get("loss"),
               "train_acc": logs.get("logits_accuracy", logs.get("accuracy")),
               "train_sev_acc": logs.get("severity_accuracy"),
               "val_acc": acc, "val_macro_f1": mf1, **{f"f1_{c}": f for c, f in zip(CLASSES, f1s)},
               "val_sev_acc": s_acc, "val_sev_macro_f1": s_mf1, **{f"f1_{c}": f for c, f in zip(SEVERITY, s_f1s)},
               "val_score": score}
        self.log.append(row)
        flag = ""
        if score > self.best:
            self.best, self.bad, flag = score, 0, "  * saved"
            self.model.save(self.path)
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.model.stop_training = True
        print(f"[{self.stage}] epoch {epoch + 1}: val acc {acc:.4f}  macro-F1 {mf1:.4f}  "
              + " ".join(f"{c}={f:.2f}" for c, f in zip(CLASSES, f1s))
              + f" | severity acc {s_acc:.4f} macro-F1 {s_mf1:.4f} "
              + " ".join(f"{c}={f:.2f}" for c, f in zip(SEVERITY, s_f1s)) + f" | score {score:.4f}" + flag, flush=True)


def fit_temperature(logits, y):
    """Grid-search the temperature that minimises validation NLL."""
    best_t, best_nll = 1.0, 1e9
    for t in np.arange(0.5, 3.01, 0.01):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        nll = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 2), best_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--ft-epochs", type=int, default=30)
    ap.add_argument("--backbone", nargs="+", default=["efficientnetb0"], choices=sorted(BACKBONES),
                    help="one or more backbones, trained one after the other")
    ap.add_argument("--model-dir", default=None, help="output folder relative to the project (one backbone only)")
    ap.add_argument("--unfreeze-from", default=None, help="first backbone layer to fine-tune (default per backbone)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--severity-weight", type=float, default=1.0, help="loss weight of the severity head")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-export", action="store_true", help="skip the automatic TFLite export after training")
    args = ap.parse_args()
    if args.model_dir and len(args.backbone) > 1:
        ap.error("--model-dir works with a single --backbone")
    manifests = MANIFESTS
    for m in manifests:
        if not os.path.isfile(os.path.join(ROOT, m)):
            sys.exit(f"{m} not found - run scripts/split_datasetv2.py first")

    for backbone in args.backbone:
        if args.model_dir:
            model_dir = os.path.join(ROOT, args.model_dir)
        elif "SKINLENS_MODEL_DIR" in os.environ and len(args.backbone) == 1:
            model_dir = MODEL_DIR                                   # smoke-test override
        else:
            model_dir = os.path.join(ROOT, "model" + DIR_SUFFIX[backbone])
        print(f"\n===== {backbone} on {' + '.join(manifests)} -> {model_dir}", flush=True)
        train_one(backbone, manifests, model_dir, args)
        if not args.no_export:
            export_tflite(model_dir)


def export_tflite(model_dir):
    """Run export_tflite.py on the fresh model; a running serve.py picks up the new files automatically."""
    print(f"\n===== exporting TFLite -> {os.path.join(model_dir, 'tflite')}", flush=True)
    env = dict(os.environ, SKINLENS_MODEL_DIR=model_dir)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_tflite.py")
    r = subprocess.run([sys.executable, "-P", "-E", script], env=env)
    if r.returncode:
        print(f"WARNING: TFLite export failed (exit {r.returncode}); rerun: python -P -E scripts/export_tflite.py", flush=True)


def train_one(backbone, manifests, model_dir, args):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        dst = os.path.join(ROOT, "model_previous", time.strftime("%Y%m%d_%H%M%S_") + os.path.basename(model_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(model_dir, dst)
        print("previous model moved to", dst)
    os.makedirs(model_dir, exist_ok=True)

    train_ds, train_y, train_s = make_dataset("train", args.batch, training=True, manifests=manifests)
    val_ds, val_y, val_s = make_dataset("valid", 64, manifests=manifests)
    val_ds = val_ds.map(lambda x, y: x)                       # predict() only needs the images

    # class weights: square-root inverse frequency, applied as per-sample weights (Keras has no class_weight
    # for multi-output models). The severity head gets weight 0 on normal tiles (no severity label).
    counts = np.bincount(train_y, minlength=N)
    w = (counts.sum() / (N * counts)) ** 0.5
    class_weight = {i: float(v / w.mean()) for i, v in enumerate(w)}
    print("train counts", dict(zip(CLASSES, counts.tolist())), "class weights",
          {CLASSES[i]: round(v, 2) for i, v in class_weight.items()})
    s_counts = np.bincount([s for s in train_s if s >= 0], minlength=NS)
    print("train severity counts", dict(zip(SEVERITY, s_counts.tolist())), "(normal tiles excluded)")
    cw = tf.constant([class_weight[i] for i in range(N)], tf.float32)

    def to_targets(x, y):
        c, s = y
        targets = {"logits": tf.one_hot(c, N), "severity": tf.one_hot(tf.maximum(s, 0), NS)}
        weights = {"logits": tf.gather(cw, c), "severity": tf.cast(s >= 0, tf.float32)}
        return x, targets, weights
    train_ds = train_ds.map(to_targets)

    model, base = build_model(args.dropout, backbone)
    base_name = base.name
    unfreeze_from = args.unfreeze_from or BACKBONES[backbone][1]
    print(f"backbone {backbone} ({base.count_params() / 1e6:.2f} M params), fine-tune from {unfreeze_from}", flush=True)
    loss = {k: keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
            for k in ("logits", "severity")}
    loss_weights = {"logits": 1.0, "severity": args.severity_weight}
    metrics = {"logits": ["accuracy"], "severity": ["accuracy"]}
    best_path = os.path.join(model_dir, "skinlens.keras")
    log = []
    cb = BestMacroF1(val_ds, val_y, val_s, best_path, args.patience, log)
    t0 = time.time()

    # ---- stage 1: head only
    cb.stage = "head"
    model.compile(keras.optimizers.Adam(args.head_lr), loss, loss_weights=loss_weights, metrics=metrics)
    model.fit(train_ds, epochs=args.head_epochs, callbacks=[cb], verbose=2)
    print(f"head stage done in {(time.time() - t0) / 60:.1f} min, best val score {cb.best:.4f}", flush=True)

    # ---- stage 2: fine-tune the top of the backbone (continue from the best head weights)
    model = keras.models.load_model(best_path)
    base = model.get_layer(base_name)
    base.trainable = True
    names = [l.name for l in base.layers]
    start = names.index(unfreeze_from)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
    steps = int(np.ceil(len(train_y) / args.batch)) * args.ft_epochs
    lr = keras.optimizers.schedules.CosineDecay(args.ft_lr, steps, alpha=0.05)
    model.compile(keras.optimizers.Adam(lr), loss, loss_weights=loss_weights, metrics=metrics)
    cb.stage, cb.bad = "finetune", 0
    model.fit(train_ds, epochs=args.ft_epochs, callbacks=[cb], verbose=2)
    print(f"training done in {(time.time() - t0) / 60:.1f} min, best val score {cb.best:.4f}", flush=True)

    with open(os.path.join(model_dir, "history.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)

    # ---- calibration + final evaluation with the best checkpoint
    model = keras.models.load_model(best_path)
    val_out = model.predict(val_ds, verbose=0)
    val_logits = val_out["logits"]
    t, nll = fit_temperature(val_logits, np.array(val_y))
    vm = np.array(val_s) >= 0
    ts, s_nll = fit_temperature(val_out["severity"][vm], np.array(val_s)[vm])
    json.dump({"temperature": t, "val_nll": nll, "severity_temperature": ts, "severity_val_nll": s_nll,
               "classes": CLASSES, "severity": SEVERITY, "backbone": backbone, "manifests": manifests},
              open(os.path.join(model_dir, "calibration.json"), "w"), indent=2)
    test_ds, test_y, test_s = make_dataset("test", 64, manifests=manifests)
    test_ds = test_ds.map(lambda x, y: x)
    test_out = model.predict(test_ds, verbose=0)
    test_logits = test_out["logits"]
    # flip TTA (only reported; the TFLite model runs a single pass)
    flip_out = model.predict(test_ds.map(tf.image.flip_left_right), verbose=0)
    tta_logits = (test_logits + flip_out["logits"]) / 2
    tm = np.array(test_s) >= 0
    sev_t = np.array(test_s)[tm]
    sev_p = test_out["severity"].argmax(1)[tm]
    off2 = int(np.sum(np.abs(sev_t - sev_p) == 2))
    parts = [
        f"backbone {backbone}, data {' + '.join(manifests)}",
        f"temperature {t} (val NLL {nll:.4f}); severity temperature {ts} (val NLL {s_nll:.4f})",
        report(val_y, val_logits.argmax(1), "validation"),
        report(test_y, test_logits.argmax(1), "test"),
        report(test_y, tta_logits.argmax(1), "test + horizontal-flip TTA"),
        report(np.array(val_s)[vm], val_out["severity"].argmax(1)[vm], "severity, validation (non-normal tiles)", SEVERITY),
        report(sev_t, sev_p, "severity, test (non-normal tiles)", SEVERITY)
        + f"\nmild<->severe confusions (off by two levels): {off2}",
    ]
    txt = "\n\n".join(parts)
    open(os.path.join(model_dir, "report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
