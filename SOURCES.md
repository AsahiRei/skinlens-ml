# SkinLens dataset sources

Downloaded 2026-11-12. Raw downloads lived in `raw/` (untouched); `dataset/` is built from them by
`scripts/build_dataset.py` and checked by `scripts/verify_dataset.py`.

> **Note (2026-11-12):** `raw/` was deleted to free space. `dataset/` + `split_manifest.csv` are the only copy of the images.
> To rebuild the dataset or add a source, re-download the sources below into `raw/` first, then run `extract_raw.py` and `build_dataset.py`.

"Candidates" = images in the 5 target classes before cleaning; "kept" = images in `dataset/` after cleaning.

| # | Source (id in filenames) | URL | License (as stated by uploader) | Classes taken (original folder → class) | Candidates → kept |
|---|---|---|---|---|---|
| 1 | Oily, Dry and Normal Skin Types Dataset (`shakya`) | https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset | Apache-2.0 | oily, dry, normal | dry 758→426, normal 1274→620, oily 1120→454 |
| 2 | FaceSkinType (`muttaqin`) | https://www.kaggle.com/datasets/muttaqin1113/faceskintype | Apache-2.0 | berjerawat→acne, berminyak→oily, kering→dry, normal | acne 868→781, oily 622→487, dry 625→498, normal 698→553 |
| 3 | skin_type_classification (`akage99`) | https://huggingface.co/datasets/akage99/skin_type_classification | **unknown** (none stated) | oily, dry, normal (combination dropped) | dry 500→422, normal 500→377, oily 500→456 |
| 4 | skintype2 (`prajantha`) | https://huggingface.co/datasets/Prajantha/skintype2 | **none stated** | oily, dry (combination dropped) | 550→0 (every file is a byte-identical copy of akage99) |
| 5 | ACNE04, Layered-Labs rebuild (`acne04`) | https://huggingface.co/datasets/Layered-Labs/ACNE04 (original: https://github.com/xpwu95/LDL) | **ACNE04 academic use only** | all → acne | 1457→1333 |
| 6 | Facial Skin Conditions Classification (`houraatef`) | https://www.kaggle.com/datasets/houraatef/facial-skin-conditions-classification (Roboflow: skinsightproject/facial-skin-classification-crpmo) | Kaggle: CC0-1.0; bundled Roboflow README: CC BY 4.0. Includes ACNE04 copies (`levle*` files), which carry ACNE04's academic-only terms | rows labelled only "Acne" → acne (Skin Redness, blackhead dropped) | 363→219 |
| 7 | Facial Skin Diseases Dataset (`osman`) | https://www.kaggle.com/datasets/osmankagankurnaz/facial-skin-diseases-dataset | Apache-2.0 | all (acne only, TFRecord) → acne | 188→133 |
| 8 | Skin Defects: acne, redness, bags under the eyes, TrainingDataPro (`trainingdatapro`) | https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes | **CC BY-NC-ND 4.0** | acne (redness, bags dropped); 10 people × 3 views | 30→27 |
| 9 | Same dataset on Hugging Face, UniqueData/TrainingDataPro (`uniquedata`) | https://huggingface.co/datasets/UniqueData/dermatology-dataset-acne-redness-and-bags-under-the-eyes | **CC BY-NC-ND 4.0** | acne | 30→0 (identical to #8) |
| 10 | Acne Recognition Dataset, 220 people, IGA 3–4 (`tapakah68`) | https://www.kaggle.com/datasets/tapakah68/skin-problems-34-on-the-iga-scale | **CC BY-NC-ND 4.0** | all → acne; 229 people × 3 views | 687→684 |
| 11 | facial-skin-disease (`shwetakk`) | https://www.kaggle.com/datasets/shwetakk/facial-skin-disease | CC0-1.0 (some images carry DermNet watermarks) | acne, eksim→eczema (herpes, panu, rosacea dropped) | acne 200→99, eczema 198→93 (archive holds two copies of every file) |
| 12 | AICamp 2023 Skin Conditions Dataset (`aicamp`) | https://huggingface.co/datasets/notable12/AICamp-2023-Skin-Conditions-Dataset | Uploader says MIT, but the images are **© DermNet** (watermarked), so MIT cannot be relied on | Eczema, Atopic Dermatitis → eczema (all other 21 classes dropped) | 400→373 |

## Final per-class totals

| class | candidates | kept | of which face-only sources |
|---|---|---|---|
| acne | 3823 | 3276 | all |
| eczema | 598 | 466 | 93 (shwetakk); 373 from aicamp are mostly body parts (hands, feet, limbs) |
| oily | 2492 | 1397 | all |
| dry | 2183 | 1346 | all |
| normal | 2472 | 1550 | all |

## Checked but not used

| Dataset | Why not |
|---|---|
| unidpro/facial-skin-condition-dataset (Kaggle) | CC BY-NC-ND; a 1,200-image sample annotated for many conditions, none mapping cleanly to the 5 classes |
| imtkaggleteam/acne-computer-vision (Kaggle, CC BY 4.0) | Acne lesion detection set; acne is already the largest class |
| "Skin conditions image dataset" (acne/eczema/carcinoma/keratosis/milia/rosacea; Kaggle/HyperAI) named in the brief | Not found by Kaggle search under that name; the known copies are DermNet scrapes, rejected earlier as not face-only |
| amellia/face-skin-disease, mateenzahid/eczema-dataset, killa92, shanmukhadattaboda, manithj | Rejected earlier for SkinLens (DermNet copies / no faces / inconsistent labels / AI-generated / masked faces) |
| Neperl & TheKarna skin-disease-acne-rosacea-normal (HF) | "normal" includes hands/feet; rejected earlier |
| exper1ment/dermnet (HF) | DermNet copy, body parts |

## License summary

- **Commercial use not allowed:** `trainingdatapro`, `tapakah68` (CC BY-NC-ND 4.0), `acne04` (academic only) and the 81 ACNE04 copies inside `houraatef`. Together that is 2,125 of 3,276 acne images (~65%).
- **NoDerivatives (ND):** re-encoded copies of `trainingdatapro` and `tapakah68` must not be redistributed. Train on them internally; don't publish `dataset/`.
- **Unclear rights:** `akage99` (no license), `aicamp` and parts of `shwetakk` (DermNet-copyrighted images under a CC0/MIT label), `shakya`/`muttaqin`/`akage99` (web-scraped photos relabelled under Apache-2.0 / no license; the uploader may not own the images).
- To build a commercial-safe subset, filter `split_manifest.csv` on the `source` column.
