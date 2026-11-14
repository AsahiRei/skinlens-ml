"""Convert model_effb0/skinlens.keras to TFLite and verify every converted file against the Keras model.

Each TFLite file has one input and two outputs (tell the outputs apart by their shape, not their index):
  input  "image"  [1, 224, 224, 3]  RGB 0..255   float32 (fp32/fp16 files) or uint8 (int8 file)
  output          [1, 5]            calibrated condition probabilities softmax(logits / T), order = labels.txt
  output          [1, 3]            calibrated severity probabilities (mild, moderate, severe) = severity_labels.txt;
                                    only meaningful when the top condition is not "normal"
Pixel normalisation is inside the model, so the app only center-crops, resizes and copies pixels.

Files written to model_effb0/tflite/:
  skinlens_fp32.tflite   reference, ~9 MB
  skinlens_fp16.tflite   float16 weights, ~4.5 MB, same accuracy as fp32 (recommended default; runs on the GPU delegate)
  skinlens_int8.tflite   full-integer, ~3 MB, fastest on CPU/NNAPI, check its accuracy row in tflite_report.txt
  labels.txt, severity_labels.txt, model_info.json, tflite_report.txt

Usage: python -P -E scripts/export_tflite.py [--calib-images 300]
"""
import argparse, json, os, sys, time

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skinlens_common import CLASSES, IMG, MODEL_DIR, SEED, SEVERITY, load_square, macro_f1, make_dataset, read_split, report

OUT = os.path.join(MODEL_DIR, "tflite")


def serving_model(model, temperature, sev_temperature):
    """logits model -> [condition probabilities [5], severity probabilities [3]], standard layers only (no Lambda)."""
    inp = keras.Input((IMG, IMG, 3), name="image")
    o = model(inp)
    x = keras.layers.Rescaling(1.0 / temperature, name="temperature")(o["logits"])
    out = keras.layers.Softmax(name="probabilities")(x)
    s = keras.layers.Rescaling(1.0 / sev_temperature, name="severity_temperature")(o["severity"])
    sev = keras.layers.Softmax(name="severity_probabilities")(s)
    return keras.Model(inp, [out, sev], name="skinlens_serving")


def split_outputs(it):
    """(condition output detail, severity output detail) of a TFLite interpreter, picked by shape."""
    outs = {int(d["shape"][-1]): d for d in it.get_output_details()}
    return outs[len(CLASSES)], outs[len(SEVERITY)]


def export_saved_model(serve, path):
    """Keras 3 SavedModel export with a fixed [1,224,224,3] signature (static shapes convert cleanly)."""
    serve.export(path, format="tf_saved_model", verbose=False,
                 input_signature=[tf.TensorSpec([1, IMG, IMG, 3], tf.float32, name="image")])


def representative(n):
    paths, labels = read_split("train")
    rng = np.random.default_rng(SEED)
    # stratified sample so every class shapes the activation ranges
    idx = np.concatenate([rng.permutation(np.where(np.array(labels) == c)[0])[: n // len(CLASSES)] for c in range(len(CLASSES))])
    def gen():
        for i in rng.permutation(idx):
            yield [tf.cast(load_square(paths[i], IMG), tf.float32)[None]]
    return gen


def convert(saved_model_dir, kind, calib_n):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if kind == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif kind == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative(calib_n)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8      # app feeds raw uint8 pixels
        conv.inference_output_type = tf.float32   # probabilities come out as float
    return conv.convert()


def run_tflite(path, images):
    it = tf.lite.Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out, sout = split_outputs(it)
    probs, sprobs, times = [], [], []
    for img in images:
        it.set_tensor(inp["index"], img[None].astype(inp["dtype"]))
        t = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - t)
        probs.append(it.get_tensor(out["index"])[0].copy())
        sprobs.append(it.get_tensor(sout["index"])[0].copy())
    return np.array(probs), np.array(sprobs), 1000 * float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    model = keras.models.load_model(os.path.join(MODEL_DIR, "skinlens.keras"))
    cal_path = os.path.join(MODEL_DIR, "calibration.json")
    cal = json.load(open(cal_path)) if os.path.exists(cal_path) else {}
    temperature, sev_temperature = cal.get("temperature", 1.0), cal.get("severity_temperature", 1.0)
    serve = serving_model(model, temperature, sev_temperature)
    sm_dir = os.path.join(OUT, "saved_model")
    export_saved_model(serve, sm_dir)

    test_ds, test_y, test_s = make_dataset("test", 64, cache=False)
    images = np.concatenate([x.numpy() for x, _ in test_ds]).round().astype(np.uint8)
    ref, sref = serve.predict(images.astype(np.float32), batch_size=64, verbose=0)
    sm = np.array(test_s) >= 0                       # severity is scored on non-normal tiles only
    sev_y = np.array(test_s)[sm]

    lines = [f"temperature {temperature}, severity temperature {sev_temperature}",
             report(test_y, ref.argmax(1), "keras (reference)"),
             report(sev_y, sref.argmax(1)[sm], "keras severity (reference, non-normal tiles)", SEVERITY), "",
             f"{'file':24s} {'MB':>6s} {'acc':>7s} {'macroF1':>8s} {'agree':>7s} {'max|dp|':>8s} "
             f"{'sev_acc':>8s} {'sev_agr':>8s} {'ms/img':>7s}"]
    details = []
    for kind in ("fp32", "fp16", "int8"):
        name = f"skinlens_{kind}.tflite"
        path = os.path.join(OUT, name)
        print("converting", name, flush=True)
        open(path, "wb").write(convert(sm_dir, kind, args.calib_images))
        probs, sprobs, ms = run_tflite(path, images)
        pred, spred = probs.argmax(1), sprobs.argmax(1)
        mf1, _ = macro_f1(test_y, pred)
        lines.append(f"{name:24s} {os.path.getsize(path) / 1e6:6.2f} {np.mean(pred == np.array(test_y)):7.4f} {mf1:8.4f} "
                     f"{np.mean(pred == ref.argmax(1)):7.4f} "
                     f"{max(np.abs(probs - ref).max(), np.abs(sprobs - sref).max()):8.4f} "
                     f"{np.mean(spred[sm] == sev_y):8.4f} {np.mean(spred == sref.argmax(1)):8.4f} {ms:7.1f}")
        details.append(report(test_y, pred, name))
        details.append(report(sev_y, spred[sm], name + " severity (non-normal tiles)", SEVERITY))
        print(lines[-1], flush=True)

    open(os.path.join(OUT, "labels.txt"), "w").write("\n".join(CLASSES) + "\n")
    open(os.path.join(OUT, "severity_labels.txt"), "w").write("\n".join(SEVERITY) + "\n")
    info = {
        "classes": CLASSES,
        "severity": SEVERITY,
        "input": {"name": "image", "shape": [1, IMG, IMG, 3], "color": "RGB", "range": "0..255 (no normalisation)",
                  "dtype": {"skinlens_fp32.tflite": "float32", "skinlens_fp16.tflite": "float32", "skinlens_int8.tflite": "uint8"},
                  "preprocess": "apply EXIF rotation, center-crop largest square, resize to 224x224 bilinear"},
        "output": {"shape": [1, len(CLASSES)], "dtype": "float32", "meaning": "calibrated class probabilities (sum to 1)"},
        "severity_output": {"shape": [1, len(SEVERITY)], "dtype": "float32",
                            "meaning": "calibrated severity probabilities (mild, moderate, severe); ignore when the top class is normal",
                            "note": "two outputs: identify them by last dimension (5 = classes, 3 = severity), not by index"},
        "temperature": temperature,
        "severity_temperature": sev_temperature,
        "suggested_min_confidence": 0.5,
    }
    json.dump(info, open(os.path.join(OUT, "model_info.json"), "w"), indent=2)
    txt = "\n".join(lines) + "\n\nagree = top-1 agreement with keras; max|dp| = largest probability difference (both heads); " \
          "sev_acc = severity accuracy on non-normal tiles; sev_agr = severity top-1 agreement with keras; " \
          "ms/img = median CPU latency here (4 threads)\n\n" + "\n\n".join(details) + "\n"
    open(os.path.join(OUT, "tflite_report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
