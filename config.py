# Data dimensions
DATA_PARAMS = {
    "image_size": (224, 224),
    "batch_size": 8,
    "seed": 123}

DROPOUT_RATE = 0.3

# Sobel branch — trainable kernel warmup.
# Freeze sobel_learn for this many epochs before allowing backprop updates.
# Gives the rest of the network time to stabilise before the Sobel kernels drift
# away from their Gx/Gy initialisation.  Set to 0 to train from epoch 1.
SOBEL_TRAINABLE_WARMUP_EPOCHS = 5
EPOCHS = 30

# Learning-rate schedule (ReduceLROnPlateau)
LR_REDUCE_PATIENCE = 3       # epochs without val_loss improvement before reducing LR
LR_REDUCE_FACTOR = 0.3       # new_lr = lr * factor
LR_REDUCE_MIN_LR = 1e-6     # floor — LR will never drop below this

# Early stopping
EARLY_STOPPING_PATIENCE = 7  # epochs without val_loss improvement before stopping
                              # restore_best_weights=True rolls back to the best checkpoint

# Processed images-root and run outputs (see audit_bundle for audit/<run_id> layout).
BASE_DIR = 'processed_data'
OUTPUT_DIR = 'outputs'
SHARDS_SUBDIR = 'shards'   # relative to BASE_DIR; one NPZ per class per split
# Max RAM (GB) allowed for loading all shards into memory at once.
# If the estimated cost exceeds this, shard loading is skipped and the
# standard JPEG decode pipeline is used instead.  Colab free tier ≈ 12 GB;
# set higher if running on a machine with more RAM.
SHARDS_MAX_RAM_GB = 6.0
# Images per NPZ chunk when building shards. Keeps peak RAM during build to
# ~SHARDS_CHUNK_SIZE × 224 × 224 × 3 × 4 bytes (≈ 0.6 GB at 1000 images).
SHARDS_CHUNK_SIZE = 1000

AUDIT_PATH_SEGMENTS = ("fusion", "audit")
FUSION_MODELS_PATH_SEGMENTS = ("fusion", "models")
FUSION_PLOTS_PATH_SEGMENTS = ("fusion", "plots")

TRAINING_DOMAIN = "leaf"

# RGB branch uses EfficientNetB3 as backbone (sub-model layer name = "efficientnetb3").
# At 224×224 both B0 and B3 output a 7×7 feature map (stride-32). Use 300×300 for 10×10.
# Shallow layer no longer exists; visualize_gradcam_batch gracefully skips it via try/except.
GRADCAM_RGB_SHALLOW_LAYER = "rgb_conv2"       # intentionally absent — shallow panel skipped
GRADCAM_RGB_DEEP_LAYER = "rgb_backbone_output"  # Identity layer after EfficientNetB3 — in the fused graph

GRADCAM_NUM_SAMPLES_DEFAULT = 3
OCCLUSION_NUM_SAMPLES_DEFAULT = 5
OCCLUSION_PATCH_DEFAULT = 24
OCCLUSION_STRIDE_DEFAULT = 12
OCCLUSION_FILL_DEFAULT = 0.5
IG_NUM_SAMPLES_DEFAULT = 3
IG_M_STEPS_DEFAULT = 32

FIELD_CLASSES_SUBDIR = "field_classes"

# Phase 5 — KNN and reports: train embedding bank + k-NN tables/grids (L2-normalized layer output).
NEIGHBOR_EMBEDDING_LAYER = "fusion_dense_2"
NEIGHBOR_TOP_K_DEFAULT = 5
NEIGHBOR_VAL_QUERY_SAMPLES_DEFAULT = 10
NEIGHBOR_FIELD_QUERY_SAMPLES_DEFAULT = 10
NEIGHBOR_GRID_FIGURES_DEFAULT = 3

# Phase 6 — counterfactuals (contrastive k-NN + greedy mask flip to change argmax).
COUNTERFACTUAL_VAL_SAMPLES_DEFAULT = 5
COUNTERFACTUAL_FIELD_SAMPLES_DEFAULT = 5
COUNTERFACTUAL_CONTRASTIVE_SEARCH_CAP_DEFAULT = 200
COUNTERFACTUAL_MASK_FLIP_MAX_PATCHES_DEFAULT = 24

# Phase 7 — structured reporting (failure gallery caps).
FAILURE_GALLERY_MAX_HIGH_CONF_WRONG = 8
FAILURE_GALLERY_MAX_LOW_MARGIN = 8
# Optional pre-filter for low-margin panel (e.g. 0.25); None = rank by margin on all rows.
FAILURE_GALLERY_LOW_MARGIN_THRESHOLD = None

# Data and shortcut audits — Grad-CAM border-focus heuristic.
SHORTCUT_BORDER_FRAC_DEFAULT = 0.18
SHORTCUT_AUDIT_SAMPLES_DEFAULT = 12

# Bumps invalidate tf.data caches when preprocessing or aug changes.
TF_CACHE_PIPELINE_ID = "v2"

# Override cache root directory. None = use outputs/tf_cache/ inside project folder (default).
# On Colab set to "/tmp/spud_cache" to write to local fast disk instead of Google Drive.
import os as _os
TF_CACHE_DIR_OVERRIDE = "/tmp/spud_cache" if _os.path.isdir("/tmp") else None

# Focal Loss gamma — used by focal_loss.py (γ=2 per paper).
# γ=0 reduces to standard cross-entropy; higher values focus more on hard examples.
FOCAL_LOSS_GAMMA = 2.0