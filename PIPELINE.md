# SkinLens pipeline: dataset → EfficientNetB0 → TFLite

```
dataset/raw/ (collages) ──split_datasetv2.py──> dataset/{train,valid,test} + split_manifest.csv ──verify_dataset.py──> PASS
                                   │
                              train.py            (EfficientNetB0 + severity head, 2 stages, best val score, temperature calibration, then runs export_tflite.py)
                                   │
                   model_effb0/skinlens.keras + calibration.json + report.txt
                                   │
                          export_tflite.py        (bakes softmax(logits/T) in; fp32 / fp16 / int8; verifies each file)
                                   │
        model_effb0/tflite/skinlens_{fp32,fp16,int8}.tflite + labels.txt + severity_labels.txt + model_info.json + tflite_report.txt
                                   │
                              predict.py          (reference app-side inference)
```

## Commands (system Python 3.12 with TF 2.21; the `.venv` has no TF)

```powershell
python -P -E scripts/split_datasetv2.py         # cut dataset/raw collages into tiles + split by person
python -P -E scripts/verify_dataset.py          # dataset leak/dup check
python -P -E scripts/train.py                   # ~65 min on CPU + export; old model_effb0/ goes to model_previous/<time>/
python -P -E scripts/export_tflite.py           # a few minutes (train.py already runs it; use for a manual re-export)
python -P -E scripts/predict.py selfie.jpg      # sanity check on your own photo
python -P -E scripts/serve.py                   # live browser tester: http://127.0.0.1:8000 (URL / upload / paste / webcam); reloads a new model automatically
```

## Training design

| Choice | Why |
|---|---|
| EfficientNetB0 / 224, ImageNet weights | best test macro-F1 of the backbones tried; fp16 TFLite ≈8 MB, ≈6 ms on CPU (its int8 file lost accuracy: use fp16) |
| Normalisation inside the model (EfficientNet does it itself) | the app feeds raw 0..255 pixels; it can't get the normalisation wrong |
| Stage 1 head only (6 ep, lr 1e-3) → stage 2 fine-tune from `block5a_expand_conv` (30 ep, cosine from 2e-4) | stable start, then adapts the higher-level features to skin texture |
| BatchNorm stays frozen during fine-tuning | small batches on CPU would destabilise BN statistics |
| Center square crop → zoom-in random crops (75–100%), flip, ±10° rotation, mild brightness/contrast/saturation | no zoom-out or down-scaling (oily/dry/normal texture is the signal), no hue shift (redness is the signal for acne/eczema) |
| Label smoothing 0.1 + square-root class weights | noisy web-scraped skin-type labels; eczema has only 466 images |
| Checkpoint on **val macro-F1**, early stop after 6 bad epochs | accuracy alone hides the small eczema class |
| Temperature scaling on val | probabilities mean something, so the app can show "low confidence" |

## Integrating the TFLite model

**Which file:** `skinlens_fp16.tflite` by default (same accuracy as fp32, half the size, GPU delegate friendly).
Use `skinlens_int8.tflite` only if its row in `tflite_report.txt` is close to fp32 and you need the extra CPU speed.

**Contract** (also in `model_info.json`):
- input `[1, 224, 224, 3]`, RGB, values 0..255, **no normalisation**; float32 for fp32/fp16, uint8 for int8
- preprocessing: apply EXIF rotation → center-crop the largest square → resize to 224×224 (bilinear)
- output `[1, 5]` float32 probabilities in the order of `labels.txt`: `acne, eczema, oily, dry, normal`
- output `[1, 3]` float32 severity probabilities in the order of `severity_labels.txt`: `mild, moderate, severe`.
  Show it only when the top class is not `normal`. **The model has two outputs: find each one by its size (5 or 3),
  not by its index**, because the converter does not guarantee the order.

## Severity labels

Severity comes from the collage row, stored in the tile filename (`v2_<collage>_r<row>_<severity>_<class>.jpg`)
and in the `severity` column of `split_manifest.csv`. There are no severity folders. 3-row collages are mild / moderate /
severe from top to bottom; 2-row collages (1-15, 33, 34) are mild / severe; Normal tiles are `none`. The severity head
shares the backbone and is trained only on non-normal tiles. Checkpoints are chosen on
mean(condition macro-F1, severity macro-F1) on validation. The levels are AI-edited, not clinically graded.
- if the top probability is below 0.5, ask the user for a clearer, well-lit, front-facing photo

The model was trained on face close-ups. In the app, guide the user to fill the frame with their face
(for example, a face-outline overlay in the camera). If you add on-device face detection (ML Kit / MediaPipe),
crop a square around the face box with ~20% margin before the steps above.

### Android (Kotlin, `com.google.ai.edge.litert:litert` or `org.tensorflow:tensorflow-lite`)

```kotlin
// assets/: skinlens_fp16.tflite, labels.txt, severity_labels.txt
val interpreter = Interpreter(FileUtil.loadMappedFile(context, "skinlens_fp16.tflite"),
                              Interpreter.Options().setNumThreads(4))
val labels = context.assets.open("labels.txt").bufferedReader().readLines()
val sevLabels = context.assets.open("severity_labels.txt").bufferedReader().readLines()
// two outputs: find them by size, not by index
val classOut = (0 until interpreter.outputTensorCount).first { interpreter.getOutputTensor(it).shape()[1] == labels.size }
val sevOut = (0 until interpreter.outputTensorCount).first { interpreter.getOutputTensor(it).shape()[1] == sevLabels.size }

fun classify(src: Bitmap): Triple<String, Float, String?> {     // src already EXIF-rotated
    val s = minOf(src.width, src.height)
    val square = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    val bmp = Bitmap.createScaledBitmap(square, 224, 224, true)
    val px = IntArray(224 * 224).also { bmp.getPixels(it, 0, 224, 0, 0, 224, 224) }
    val input = ByteBuffer.allocateDirect(4 * 224 * 224 * 3).order(ByteOrder.nativeOrder())
    for (p in px) {                                     // raw 0..255 floats, RGB order
        input.putFloat(((p shr 16) and 0xFF).toFloat())
        input.putFloat(((p shr 8) and 0xFF).toFloat())
        input.putFloat((p and 0xFF).toFloat())
    }
    // int8 model: allocateDirect(224*224*3) and input.put(byte) instead of putFloat
    val out = Array(1) { FloatArray(labels.size) }
    val sev = Array(1) { FloatArray(sevLabels.size) }
    interpreter.runForMultipleInputsOutputs(arrayOf(input.rewind()), mapOf(classOut to out, sevOut to sev))
    val i = out[0].indices.maxBy { out[0][it] }
    val severity = if (labels[i] == "normal") null else sevLabels[sev[0].indices.maxBy { sev[0][it] }]
    return Triple(labels[i], out[0][i], severity)
}
```

### Flutter (`tflite_flutter`)

```dart
final interpreter = await Interpreter.fromAsset('assets/skinlens_fp16.tflite');
// image: package:image, already oriented -> copyResizeCropSquare(image, size: 224)
final input = [List.generate(224, (y) => List.generate(224, (x) {
  final p = img.getPixel(x, y); return [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
}))];
final outs = interpreter.getOutputTensors();
final classIdx = outs.indexWhere((t) => t.shape.last == 5), sevIdx = outs.indexWhere((t) => t.shape.last == 3);
final probs = [List.filled(5, 0.0)], severity = [List.filled(3, 0.0)];
interpreter.runForMultipleInputs([input], {classIdx: probs, sevIdx: severity});
```

### iOS (TensorFlowLiteSwift)
Same contract: 224×224 RGB, 0..255 floats in `Data` (Float32, interleaved RGB), copy into `interpreter.input(at: 0)`,
`invoke()`, then read the output whose shape is [1, 5] (classes) and the one whose shape is [1, 3] (severity).

## Limits to keep in mind
- Single-label: one photo → one of 5 classes. A face that is both acne-prone and oily gets one label.
- Severity is learned from AI-edited rows of the same face, not from clinically graded photos. Few moderate examples
  (2-row collages have none), so "moderate" is the least reliable level.
- 373 of 466 eczema images are body-part DermNet images (see `SOURCES.md`); facial eczema recall is the least reliable number.
- License: ~65% of acne images are non-commercial (ACNE04, CC BY-NC-ND sets). A commercial app needs a model
  retrained on a filtered `split_manifest.csv` (filter on `source`).
- This is a cosmetic/skin-type helper, not a medical diagnosis; say so in the app.
