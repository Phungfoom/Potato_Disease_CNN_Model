

**Potato_Disease_CNN_Model**
A CNN model to perform image classification on potatoes to detect diseases. Models’ stability will be tested by adding gateway checks.

Field testing uses held-out **field** leaf images (`test_field_baseline.py`).

Dataset:
https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset?resource=download
Paper:
https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full
United States and Switzerland

Dataset:
https://zenodo.org/records/8286529
Paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12020891/
Mbeya region in the Southern Highlands of Tanzania
November 22, 2022 and April 8, 2023.

Dataset:
https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-healthy-and-late-blight?select=Late+Blight
Paper:
https://www.researchgate.net/publication/398349800_Temporal_Epidemics_of_Potato_Late_Blight_Phytophthora_Infestans_in_Major_Potato_Growing_Areas_of_Ethiopia
Ethiopia
Holeta: June 30, 2022, and June 21, 2023.
Haramaya: July 21, 2022, and July 14, 2023.
Negele Arsi: June 29, 2022, and June 24, 2023.

As well as other diseased leaves from.
[More lab dieseased leaves](https://data.mendeley.com/datasets/rgfhzd5mzw/1)


Dataset:
https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-disease-dataset
Paper:
https://pubmed.ncbi.nlm.nih.gov/38125373/
Central Java, Indonesia
August 2, 2023: For potato farms in Magelang, Central Java
August 15–16, 2023: For potato farms in Wonosobo, Central Java.

**Stress Test**

**Grayscale vs. RGB Analysis:**

Past Research: RGB yields higher results in detecting infected leaves, with the indicators bring highly color dependent. However, over-reliance on chromatic data can lead to lower accuracy in real-world lighting. When these images are converted to gray scale, the accuracy drops significantly. It brings into consideration a model’s capabilities when it comes to lab pictures vs. field pictures. [Comparison of Let Net Architecture, Inception V1 and Inception V3](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1308528/full)

My model: Including grayscale baseline, investigating if color is a 'crutch' for the model. The grayscale test evaluates the models’ capabilities to lighting variance, seeing if the model is only relying on color intensity. RGB and Grayscale will be split into two branches to reproduce results from past research and to train the model to produce more accurate results. [Split RGB and Grayscale]( https://mendel-journal.org/index.php/mendel/article/view/176/175)

Sobel Edge: Grad-CAM to Sobel Edges, focuses on edges of legions rather than whole leaf. 
Past research: Uses 'edge-enhances' CNN (EEDB-CNN) show classification project going up but blur the boundary details during pooling. [ResNet and MobileNet edge feature degradation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12777044/)

My model: Testing Sobel Edge separately, 'Edge-Aware Feature Extraction'. One branch performs standard convolutions for color/texture, and the 2nd branch uses Sobel to maintain high-frequency structural data that pooling usually struggles with. Grad_CAM heatmaps follow contours of the disease rather than the contrast of background noise to identify plants. 
Gaussian Noise: 

Past Research: FL-Efficient Net over previous EfficientNet-BO with focal loss, noise augmentation to ensure models can handle complex settings and images where camera quality is poor.  Research has found that adding Gaussian Noise during training prevents the model from overfitting to better quality lab images. [FL-Efficient Noise]( https://www.researchgate.net/publication/362967440_Research_on_plant_disease_identification_based_on_CNN)

My model: Testing signal-to noise-ratio levels and well as a fast gradient sign method to see if adding invisible pixels confuses the model. This aids in simulating poor camera quality photos to prevent overfitting. 

**Mutli Input CNN:**

**Branch A:** Process raw RGB data to find color/shape patterns. It focuses on color and hue shifts.
**Branch B:** Process the Sobel/Texture maps to focus purely on structural decay. Any irregular edges, fuzz or rings forming on leaves.
These two branches merge at the end to make a final decision.
**Evaluation Metrics:**
Confusion Matrix: Heatmap showing which disease the model confuses
Precision Recall Curves: Tradeoff between being careful/thorough
F1-Score: For unbalanced dataset.

**Goal:** Build a **transparent** CNN for potato disease identification from leaf images: prioritize **interpretable decision records** (probabilities, attribution maps, failure cases, and audit artifacts) over chasing raw accuracy alone. The architecture still uses RGB + grayscale + Sobel branches with optional augmentation so the model is pushed toward **leaf structure** rather than brittle lab-only cues.

**Importance:** AI models for plant disease detection work well in lab environment since real-world background noise is no longer captured on camera. In real world farms, with other variables such as low-quality cameras, messy backgrounds, and varied lighting, models have a harder time classifying images. This model is a more resilient model that can survive real world stressors. 

# Getting started

## Dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use a **TensorFlow-supported** Python (see [tensorflow.org/install](https://www.tensorflow.org/install)).

## Project layout

| Path | Purpose |
|------|--------|
| `data/` | Raw images under `data/<Area>/<class>/...` (gitignored) |
| `processed_data/` | Resized + stratified `train/`, `val/` from `data_preprocessing.py` |
| `processed_data/field_classes/` | Field-test images by class |
| `config.py` | `DATA_PARAMS`, `EPOCHS`, paths, `TRAINING_DOMAIN`, Grad-CAM layer names, attribution defaults; Phase 5 `NEIGHBOR_*`; Phase 6 `COUNTERFACTUAL_*`; Phase 7 `FAILURE_GALLERY_*` |
| `outputs/` | Models, plots, caches, CSVs (gitignored) |
| `outputs/fusion/audit/<run_id>/` | One folder per run: `predictions/`, `metrics/`, `attribution/`, `neighbors/`, `failures/`, `meta/run_info.json` |

## Transparency audit bundle (Phase 0)

Each training run uses `run_id = <timestamp>` and writes **`outputs/fusion/audit/<run_id>/`**:

- **`meta/run_info.json`** — paths to the saved model, plots, validation CSV, and class list.
- **`predictions/val_predictions_*.csv`** — validation set, **master schema** (see `prediction_utils.standard_prediction_column_names`). Phase 1 adds `prob_top1`, `prob_top2`, `margin_top1_top2`, `pred_entropy`, `high_conf_wrong` (threshold documented in `run_info.json` as `high_confidence_wrong_threshold`).
- **`attribution/`** — Grad-CAM figures for that run.

Field evaluation uses the **same CSV schema** under `outputs/fusion/audit/<run_id>/predictions/` (default `run_id` = `field_<timestamp>`, or pass `--audit-run-id` to attach to an existing audit folder).

**Phase 0 “done” for transparency:** audit layout exists, validation + field predictions use one column order, `run_info.json` lists artifacts, attribution outputs live under `attribution/`.

**Schema paper trail:** `prediction_utils.PREDICTION_SCHEMA_VERSION` (e.g. `phase1_v1`) and the column contract live in the module docstring at the top of `prediction_utils.py`. Each `run_info.json` repeats `prediction_schema_version` and `high_confidence_wrong_threshold` so CSV + JSON stay aligned.

## Grad-CAM and attribution (Phase 2)

Figures are written under **`outputs/fusion/audit/<run_id>/attribution/`** (train and field eval). A longer “how to read” guide lives in the module docstring at the top of **`grad_cam_visualizer.py`**.

- **What Grad-CAM shows:** regions that most **raised the score** for the chosen target class. It is **not** a guaranteed leaf mask or disease outline.
- **`--gradcam-target predicted` (default):** explains the **predicted** (argmax) class — good for typical reporting.
- **`--gradcam-target true`:** explains the **true** label — good when the prediction is wrong, to see what would support the correct class.
- **Shallow vs deep RGB:** by default, panels include an early RGB layer (`rgb_conv2`) and a late one (`rgb_conv3_final`). Omit the shallow panel with **`--no-gradcam-shallow`** if you want a compact figure.
- **Sobel panel:** same target class, but gradients for the Sobel branch so you can compare **color/texture** vs **edge** evidence.

Training and field scripts record `gradcam_target` and `gradcam_show_shallow_rgb` in each run’s **`meta/run_info.json`**.

## Occlusion / patch masking (Phase 3)

Optional **sliding-patch occlusion**: gray fill on RGB, recompute `(RGB, gray, gray)` like training, and measure how much the **target class** probability drops at each patch. Brighter regions in the saved figure mean masking there hurt that class’s score (locally important pixels).

- **Training:** add `--occlusion` (see also `--occlusion-samples`, `--occlusion-patch`, `--occlusion-stride`, `--occlusion-fill`, `--occlusion-target`).
- **Field:** `test_field_baseline.py --occlusion` with the same knobs.
- **Outputs:** PNGs `occlusion_<tag>_sample*.png` in **`attribution/`**; summary table **`metrics/occlusion_summary_*.csv`**; **`run_info.json`** gains `occlusion_summary_csv` and an `occlusion` settings block when enabled.

**Note:** Many forward passes per image — expect longer runs than Grad-CAM alone.

## Integrated Gradients (Phase 4)

**Integrated Gradients** attributes importance to **input RGB pixels** by integrating gradients along a straight path from a **baseline** image to the actual input. At each step the pipeline matches training: `gray = rgb_to_grayscale(rgb)` and the model sees `(RGB, gray, gray)`.

- **Baseline (`--ig-baseline`):** fixed to `black` (zeros) for consistency across runs.
- **Target class:** `--ig-target predicted` (default) or `true`, aligned with `--gradcam-target` for cross-method comparison.
- **Training:** `train_model.py --ig` plus `--ig-samples`, `--ig-steps`, `--ig-baseline`, `--ig-target`; omit side-by-side exports with `--no-ig-gradcam-compare`.
- **Field:** `test_field_baseline.py --ig` with the same knobs.
- **Outputs:** `ig_<tag>_sample*.png` and **`ig_vs_gradcam_<tag>_sample*.png`** (original · deep RGB Grad-CAM · IG) under **`attribution/`**; summary **`metrics/ig_summary_*.csv`**; **`run_info.json`** lists `integrated_gradients_summary_csv` and an `integrated_gradients` settings block when enabled.

**Reading IG vs Grad-CAM:** Grad-CAM highlights conv feature maps for the target class; IG sums signed pixel attributions on RGB. Agreement on leaf vs background is reassuring; systematic disagreement is a cue to dig deeper (shortcut features, baseline sensitivity).

## KNN and reports (Phase 5)

Embeddings are **L2-normalized activations** at **`fusion_dense_2`** (config: `NEIGHBOR_EMBEDDING_LAYER`)—the last fused hidden layer before the classifier. **Cosine similarity** is used (equivalent to Euclidean distance on unit vectors).

- **Training:** `train_model.py --neighbors` exports the full **training** split (eval triple, **no augmentation**) to **`neighbors/train_embeddings_<run_id>.npz`** + **`neighbors/train_manifest_<run_id>.csv`**, then runs **k-NN** for the first **`--neighbors-val-samples`** validation images (unshuffled order, same as occlusion/IG). Writes **`neighbor_table_val_*.csv`** and **`neighbor_grid_val_*_q*.png`** under **`neighbors/`**. Override layer with **`--neighbors-embedding-layer`** if you change the fusion model.
- **Field:** after a training run with `--neighbors`, point **`test_field_baseline.py`** at the same model **and** the saved bank: **`--neighbors --neighbors-train-npz <path> --neighbors-train-manifest-csv <path>`** (plus **`--neighbors-top-k`**, **`--neighbors-samples`**, **`--neighbors-grid-figures`** as needed). Produces **`neighbor_table_field_*.csv`** and grids under **`neighbors/`**.
- **`run_info.json`:** `neighbor_train_embeddings_npz`, `neighbor_train_manifest_csv`, `neighbor_val_table_csv` (train path); for field, `neighbor_field_table_csv` and **`neighbors_field`** (paths to the train bank used).

**Caveat:** Class indices must match between the loaded model and the embedding bank (same `num_classes` / label order). Neighbors are **similar in representation space**, not guaranteed “causes” of the prediction.

## Counterfactuals (Phase 6)

**Contrastive example:** among training embeddings, take the **nearest** image whose label **differs** from a chosen reference on the query (default: **predicted** class; optional **true** class via `--counterfactual-exclude`). This surfaces a “close but labeled differently” training case in representation space.

**Minimal mask flip (heuristic):** run the same occlusion grid as Phase 3 on the **initial** predicted class, rank patch locations by how much masking drops that class’s probability, then **cumulatively** mask patches in that order until the **argmax class changes** or a **patch budget** is exhausted. This is a bounded, practical search—not a guaranteed globally minimal edit.

- **Training:** `train_model.py --counterfactuals` (mask flip always). Add **`--neighbors`** in the same run to also write **`metrics/counterfactual_contrastive_*.csv`** using the freshly exported train bank. Tunables: `--counterfactual-samples`, `--counterfactual-contrastive-search-cap`, `--counterfactual-max-mask-patches`, `--counterfactual-patch`, `--counterfactual-stride`, `--counterfactual-fill`, `--counterfactual-exclude`.
- **Field:** `test_field_baseline.py --counterfactuals` writes **`metrics/counterfactual_mask_flip_*.csv`** and **`attribution/counterfactual_mask_flip_*.png`**. For contrastive on field, pass the **same** train **`--neighbors-train-npz`** and **`--neighbors-train-manifest-csv`** paths (you do **not** need `--neighbors` unless you also want the neighbor table/grids).
- **Outputs:** `counterfactual_mask_flip_<split>_<tag>.csv` (per-image: flipped or not, patch count, coords, probs); optional `counterfactual_contrastive_<split>_<tag>.csv`. **`run_info.json`** includes a **`counterfactuals`** or **`counterfactuals_field`** block when enabled.

**Caveat (required reading):** Counterfactuals here are **heuristic tools** for transparency—they are **not** proofs of causality or unique “minimal” edits. Similarity and mask ordering depend on the embedding layer, grid resolution, and patch budget; different choices can yield different narratives.

## Structured reporting (Phase 7)

Every **training** and **field** audit run automatically writes **Phase 7** artifacts into the existing layout:

| Location | Contents |
|----------|----------|
| **`metrics/`** | `per_class_metrics_<split>_<tag>.csv` — precision / recall / F1 / support per class, macro and weighted averages, overall accuracy row |
| | `confusion_matrix_counts_<split>_<tag>.csv` — raw counts (`true_*` rows × `pred_*` columns) |
| | `confusion_matrix_row_norm_<split>_<tag>.csv` — row-normalized confusion (same axes) |
| | `slice_metrics_path_<split>_<tag>.csv` — accuracy and error counts **by path slice** (parent folder of `image_rel_path`, unless you add a `report_slice` or `source` column to predictions later) |
| **`failures/`** | `failure_gallery_<split>_<tag>.png` — top **high-confidence wrong** predictions and **low-margin** cases (errors prioritized; tune caps in `config`: `FAILURE_GALLERY_*`) |

**`run_info.json`** lists relative paths under **`phase7`** for discoverability.

**Slice note:** Without extra metadata columns, “slice” is the **immediate parent directory** of each image path (often the class folder in `train/val/field` layouts). For true area/source slicing, add a **`report_slice`** (or **`source`**) column when building prediction rows.

## Data and shortcut audits

To quantify “background shortcuts” (tray edges, borders, letterboxing), the repo includes a **border-focus metric** computed from **deep RGB Grad-CAM**.

- **Training/val:** run `train_model.py --shortcut-audit` to write `metrics/shortcut_border_focus_val_*.csv`.
- **Compare runs:** train twice (same data), once **without** and once **with** `--anti-background-aug`, and compare the CSVs’ `border_share` / `border_to_center_ratio` (lower is better).
- **Field:** run `test_field_baseline.py --shortcut-audit` to write `metrics/shortcut_border_focus_field_*.csv`.

**Caveat:** this is a heuristic; it flags “heat near borders” but does not prove causality.

## Workflow

1. `python data_preprocessing.py` — builds `processed_data/train`, `val`, manifest.
2. `python train_model.py --stage foundation` — train. Use `--augment` and **`--anti-background-aug`** together to reduce fixed tray/edge shortcuts (often improves Grad-CAM focus on leaves). Optional: `--gradcam-target true`, `--no-gradcam-shallow`, **`--occlusion`** (Phase 3), **`--ig`** (Phase 4), **`--neighbors`** (Phase 5: KNN and reports), **`--counterfactuals`** (Phase 6; combine with **`--neighbors`** for contrastive + mask flip). Writes the audit bundle under `outputs/fusion/audit/<timestamp>/` including **Phase 7** metrics + failure gallery (no extra flag).
3. `python test_field_baseline.py --model_path outputs/fusion/models/potato_leaf_model_*.keras --domain leaf` — field metrics + master-schema CSV, Grad-CAM, and **Phase 7** tables + gallery under `outputs/fusion/audit/field_<timestamp>/` (optional: `--audit-run-id <id>`, same Grad-CAM flags as training: `--gradcam-target`, `--no-gradcam-shallow`, **`--occlusion`**, **`--ig`**, **`--neighbors`** + train NPZ/CSV paths for Phase 5, **`--counterfactuals`** + same NPZ/CSV paths for Phase 6 contrastive).
`python image_view.py` — sample RGB / gray / Sobel panels (`--stage training` or `field`).

# Contributing

Pull requests welcome; keep large data and `outputs/` out of git (see `.gitignore`).

# License

*Add your license if publishing.*

# Authors

# Acknowledgements

README template reference: https://github.com/catiaspsilva/README-template/blob/main/README.md 