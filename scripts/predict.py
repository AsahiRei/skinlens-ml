"""Run a SkinLens TFLite model on photos. This is the reference implementation of the app-side code:
preprocessing here is what the Android/iOS/Flutter app has to reproduce.

Usage: python -P -E scripts/predict.py photo1.jpg [photo2.jpg ...] [--model model_effb0/tflite/skinlens_fp16.tflite]
"""
import argparse, json, os

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preprocess(path, side=224):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # 1. upright RGB
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))  # 2. center square
    img = img.resize((side, side), Image.BILINEAR)                     # 3. 224x224
    return np.asarray(img)                                             # 4. uint8 0..255, HWC


def outputs(it, info):
    """(condition output, severity output or None): the two outputs are told apart by their last dimension."""
    by_size = {int(d["shape"][-1]): d for d in it.get_output_details()}
    return by_size[len(info["classes"])], by_size.get(len(info.get("severity", [])))


def severity_of(top, sev_prob, info):
    """(label, probability) of the severity, or None when the condition is normal / the model has no severity head."""
    if sev_prob is None or top == "normal":
        return None
    i = int(np.argmax(sev_prob))
    return info["severity"][i], float(sev_prob[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=os.path.join(ROOT, "model_effb0", "tflite", "skinlens_fp16.tflite"))
    args = ap.parse_args()
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    info = json.load(open(os.path.join(os.path.dirname(args.model), "model_info.json")))
    it = Interpreter(model_path=args.model)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out, sout = outputs(it, info)
    for p in args.images:
        x = preprocess(p)[None].astype(inp["dtype"])    # float32 for fp32/fp16 models, uint8 for int8
        it.set_tensor(inp["index"], x)
        it.invoke()
        prob = it.get_tensor(out["index"])[0]
        sev_prob = it.get_tensor(sout["index"])[0] if sout else None
        order = np.argsort(-prob)
        top = info["classes"][order[0]]
        note = "" if prob[order[0]] >= info["suggested_min_confidence"] else "  (low confidence - ask for a clearer photo)"
        sev = severity_of(top, sev_prob, info)
        sev_txt = f" ({sev[0]} {sev[1]:.2f})" if sev else ""
        print(f"{p}: {top} {prob[order[0]]:.2f}{sev_txt}{note}   |  " +
              "  ".join(f"{info['classes'][i]} {prob[i]:.2f}" for i in order))


if __name__ == "__main__":
    main()
