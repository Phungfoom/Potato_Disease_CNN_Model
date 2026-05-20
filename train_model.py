import os
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from tqdm import tqdm

import config
from focal_loss import focal_loss
from Potato_bunch_brain import build_combined_model
from audit_bundle import (
    append_runs_manifest,
    ensure_audit_run,
    merge_run_info,
    rel_from_script,
    validate_counterfactual_cli,
    validate_ig_cli,
    validate_neighbors_cli,
    validate_occlusion_cli,
    write_run_info,
)
from counterfactual_audit import run_counterfactual_audit
from grad_cam_visualizer import explain_my_model
from structured_reporting import write_phase7_audit_bundle
from shortcut_audit import run_shortcut_background_audit
from integrated_gradients import run_integrated_gradients_batch
from neighbor_lookup import run_train_neighbor_pipeline
from occlusion_attribution import run_occlusion_val_batch
from prediction_utils import (
    HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
    PREDICTION_SCHEMA_VERSION,
    batch_preds_to_probs,
    build_standard_predictions_dataframe,
)

parser = argparse.ArgumentParser(description="Train Potato ID Fusion Model (leaf-only)")
parser.add_argument('--stage', type=str, default="foundation", choices=['foundation', 'midterm'])
parser.add_argument(
    '--no-shards',
    action='store_true',
    help='Skip NPZ shard loading and use the JPEG image pipeline instead.',
)
parser.add_argument(
    '--no-augment',
    dest='augment',
    action='store_false',
    help='Disable training-only data augmentation (augmentation is ON by default).',
)
parser.add_argument(
    '--validation-freq',
    type=int,
    default=1,
    metavar='N',
    help=(
        'Run Keras validation every N epochs (default: 1 = every epoch). '
        'Use 2+ to speed up training; val_* points in history become sparser.'
    ),
)
parser.add_argument(
    '--anti-background-aug',
    action='store_true',
    help=(
        'Training-only: stronger spatial jitter (translation + random crop + resize) so tray/'
        'letterbox edges do not stay pixel-aligned across epochs—reduces background shortcuts '
        'that make Grad-CAM focus off-leaf. Bump config.TF_CACHE_PIPELINE_ID if you change aug.'
    ),
)
parser.add_argument(
    '--gradcam-target',
    type=str,
    default='predicted',
    choices=['predicted', 'true'],
    help='Grad-CAM class: predicted (argmax) or true label (debug wrong predictions).',
)
parser.add_argument(
    '--no-gradcam-shallow',
    action='store_true',
    help=(
        f'Omit shallow RGB Grad-CAM panel ({config.GRADCAM_RGB_SHALLOW_LAYER}).'
    ),
)
parser.add_argument(
    '--high-conf-threshold',
    type=float,
    default=HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
    metavar='P',
    help=(
        'Rows with prob_top1 >= P and wrong label get high_conf_wrong=True. '
        'Recorded in run_info.json.'
    ),
)
parser.add_argument(
    '--occlusion',
    action='store_true',
    help=(
        'After training: patch occlusion maps + metrics CSV under audit '
        '(attribution/ + metrics/); can be slow (many forward passes per image).'
    ),
)
parser.add_argument(
    '--occlusion-samples',
    type=int,
    default=config.OCCLUSION_NUM_SAMPLES_DEFAULT,
    metavar='N',
    help='Number of validation images (first batch) for occlusion (default: from config).',
)
parser.add_argument(
    '--occlusion-patch',
    type=int,
    default=config.OCCLUSION_PATCH_DEFAULT,
    metavar='PX',
    help='Square patch side in pixels (default: from config).',
)
parser.add_argument(
    '--occlusion-stride',
    type=int,
    default=config.OCCLUSION_STRIDE_DEFAULT,
    metavar='S',
    help='Stride between patch positions (default: from config).',
)
parser.add_argument(
    '--occlusion-fill',
    type=float,
    default=config.OCCLUSION_FILL_DEFAULT,
    metavar='V',
    help='RGB fill value in [0,1] for masked patch (default: from config).',
)
parser.add_argument(
    '--occlusion-target',
    type=str,
    default='predicted',
    choices=['predicted', 'true'],
    help='Which class probability occlusion tracks: predicted or true label.',
)
parser.add_argument(
    '--ig',
    action='store_true',
    help=(
        'After training: Integrated Gradients on RGB + optional IG vs Grad-CAM PNGs '
        '(attribution/ + metrics/); m_steps forward+backward passes per image.'
    ),
)
parser.add_argument(
    '--ig-samples',
    type=int,
    default=config.IG_NUM_SAMPLES_DEFAULT,
    metavar='N',
    help='Val images (first batch, unshuffled order) for IG (default: from config).',
)
parser.add_argument(
    '--ig-steps',
    type=int,
    default=config.IG_M_STEPS_DEFAULT,
    metavar='M',
    help='Riemann steps along path from baseline to input (default: from config).',
)
parser.add_argument(
    '--ig-baseline',
    type=str,
    default='black',
    choices=['black'],
    help='IG reference image baseline (fixed to black for consistency).',
)
parser.add_argument(
    '--ig-target',
    type=str,
    default='predicted',
    choices=['predicted', 'true'],
    help='IG explains gradients toward predicted vs true class logits (same idea as Grad-CAM).',
)
parser.add_argument(
    '--no-ig-gradcam-compare',
    action='store_true',
    help='Skip ig_vs_gradcam_*.png side-by-side exports.',
)
parser.add_argument(
    '--neighbors',
    action='store_true',
    help=(
        'After training: export train embeddings (NPZ+CSV) and val k-NN table + grids '
        'under neighbors/ (uses eval triples, no augmentation).'
    ),
)
parser.add_argument(
    '--neighbors-top-k',
    type=int,
    default=config.NEIGHBOR_TOP_K_DEFAULT,
    metavar='K',
    help='Nearest training images per query (default: from config).',
)
parser.add_argument(
    '--neighbors-val-samples',
    type=int,
    default=config.NEIGHBOR_VAL_QUERY_SAMPLES_DEFAULT,
    metavar='N',
    help='Validation images to run k-NN from (in order, default: from config).',
)
parser.add_argument(
    '--neighbors-grid-figures',
    type=int,
    default=config.NEIGHBOR_GRID_FIGURES_DEFAULT,
    metavar='G',
    help='Max neighbor-grid PNGs to save (one per query index, default: from config).',
)
parser.add_argument(
    '--neighbors-embedding-layer',
    type=str,
    default=config.NEIGHBOR_EMBEDDING_LAYER,
    metavar='NAME',
    help='Fused model layer whose output is L2-normalized for k-NN (default: config).',
)
parser.add_argument(
    '--counterfactuals',
    action='store_true',
    help=(
        'After training: greedy mask-flip search + optional contrastive k-NN (if --neighbors). '
        'Writes metrics/*.csv and attribution/counterfactual_mask_flip_*.png.'
    ),
)
parser.add_argument(
    '--counterfactual-samples',
    type=int,
    default=config.COUNTERFACTUAL_VAL_SAMPLES_DEFAULT,
    metavar='N',
    help='Val images for Phase 6 counterfactuals (default: from config).',
)
parser.add_argument(
    '--counterfactual-contrastive-search-cap',
    type=int,
    default=config.COUNTERFACTUAL_CONTRASTIVE_SEARCH_CAP_DEFAULT,
    metavar='K',
    help='Scan up to K nearest train embeddings for first different-class neighbor.',
)
parser.add_argument(
    '--counterfactual-max-mask-patches',
    type=int,
    default=config.COUNTERFACTUAL_MASK_FLIP_MAX_PATCHES_DEFAULT,
    metavar='M',
    help='Max cumulative occlusion patches to try per image.',
)
parser.add_argument(
    '--counterfactual-patch',
    type=int,
    default=config.OCCLUSION_PATCH_DEFAULT,
    help='Patch size (pixels) for mask-flip search (default: same as occlusion).',
)
parser.add_argument(
    '--counterfactual-stride',
    type=int,
    default=config.OCCLUSION_STRIDE_DEFAULT,
    help='Grid stride for mask-flip search.',
)
parser.add_argument(
    '--counterfactual-fill',
    type=float,
    default=config.OCCLUSION_FILL_DEFAULT,
    help='RGB fill [0,1] for masked regions.',
)
parser.add_argument(
    '--counterfactual-exclude',
    type=str,
    default='pred',
    choices=['pred', 'true'],
    help='Contrastive k-NN: exclude train labels equal to query predicted vs true class.',
)
parser.add_argument(
    '--shortcut-audit',
    action='store_true',
    help=(
        'Data/shortcut audit: compute Grad-CAM border-focus stats on val images and write '
        'metrics/shortcut_border_focus_*.csv (use to compare --anti-background-aug vs baseline).'
    ),
)
parser.add_argument(
    '--shortcut-audit-samples',
    type=int,
    default=config.SHORTCUT_AUDIT_SAMPLES_DEFAULT,
    metavar='N',
    help='Val images for shortcut audit (in order; default: from config).',
)
parser.add_argument(
    '--shortcut-border-frac',
    type=float,
    default=config.SHORTCUT_BORDER_FRAC_DEFAULT,
    metavar='F',
    help='Border thickness fraction in (0,0.49); higher = thicker border band.',
)
parser.add_argument(
    '--base-dir',
    type=str,
    default=config.BASE_DIR,
    help='Root folder for processed images (default: config.BASE_DIR). '
         'Use processed_data_v2 for the 3-way train/val/test split.',
)
parser.add_argument(
    '--mixed-precision',
    action='store_true',
    help=(
        'Enable float16 mixed precision (recommended on T4/V100/A100 Colab GPUs). '
        'Typically 30-50%% faster with half the GPU memory; model weights stay float32.'
    ),
)
args = parser.parse_args()
if args.validation_freq < 1:
    raise SystemExit("--validation-freq must be >= 1")
if not 0.0 < args.high_conf_threshold < 1.0:
    raise SystemExit("--high-conf-threshold must be in (0, 1)")
if args.occlusion:
    try:
        validate_occlusion_cli(
            args.occlusion_samples,
            args.occlusion_patch,
            args.occlusion_stride,
            args.occlusion_fill,
            config.DATA_PARAMS["image_size"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
if args.ig:
    try:
        validate_ig_cli(args.ig_samples, args.ig_steps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
if args.neighbors:
    try:
        validate_neighbors_cli(
            args.neighbors_top_k,
            args.neighbors_val_samples,
            args.neighbors_grid_figures,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
if args.counterfactuals:
    try:
        validate_counterfactual_cli(
            args.counterfactual_samples,
            args.counterfactual_contrastive_search_cap,
            args.counterfactual_max_mask_patches,
            args.counterfactual_patch,
            args.counterfactual_stride,
            args.counterfactual_fill,
            config.DATA_PARAMS["image_size"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
if args.shortcut_audit:
    if args.shortcut_audit_samples < 1:
        raise SystemExit("--shortcut-audit-samples must be >= 1")
    if not 0.0 < float(args.shortcut_border_frac) < 0.49:
        raise SystemExit("--shortcut-border-frac must be in (0, 0.49)")


def _print_device_summary() -> None:
    print(f"[TensorFlow] {tf.__version__}")
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[TensorFlow] GPU count: {len(gpus)}{' — GPU active' if gpus else ' — CPU-only training'}")
    except Exception as exc:  # pragma: no cover
        print(f"[TensorFlow] device list failed: {exc}")


_print_device_summary()

if args.anti_background_aug:
    print(
        "[augment] --anti-background-aug enabled (spatial jitter against fixed tray/edges). "
        "If you changed preprocessing, bump config.TF_CACHE_PIPELINE_ID and clear tf_cache."
    )

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = "leaf"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_audit_paths = ensure_audit_run(script_dir, timestamp)


def _tf_cache_paths(
    *,
    script_dir: str,
    stage: str,
    augment: bool,
    anti_background: bool,
) -> tuple[str, str, str]:
    """Cache prefix paths under outputs/tf_cache/<stage>/{train,val}/."""
    h, w = config.DATA_PARAMS["image_size"]
    bs = config.DATA_PARAMS["batch_size"]
    pipe = getattr(config, "TF_CACHE_PIPELINE_ID", "v1")
    basename = (
        f"spud_{pipe}_bs{bs}_h{h}x{w}_aug{int(augment)}_ab{int(anti_background)}"
    )
    override = getattr(config, "TF_CACHE_DIR_OVERRIDE", None)
    root = os.path.join(override, stage) if override else os.path.join(script_dir, config.OUTPUT_DIR, "tf_cache", stage)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    train_prefix = os.path.join(train_dir, basename)
    val_prefix = os.path.join(val_dir, basename)
    return train_prefix, val_prefix, basename


_train_cache_prefix, _val_cache_prefix, _cache_basename = _tf_cache_paths(
    script_dir=script_dir,
    stage=args.stage,
    augment=bool(args.augment),
    anti_background=bool(args.anti_background_aug),
)
print(
    f"[tf.data cache] stage={args.stage} basename={_cache_basename}\n"
    f"  train -> {_train_cache_prefix}.*\n"
    f"  val   -> {_val_cache_prefix}.*"
)

base = os.path.join(script_dir, args.base_dir)
_shard_base = os.path.join(base, config.SHARDS_SUBDIR)
# Fall back to processed_data_v2 if the default location has no shards
_alt_shard_base = os.path.join(script_dir, "processed_data_v2", config.SHARDS_SUBDIR)
if not os.path.isdir(_shard_base) and os.path.isdir(_alt_shard_base):
    print(f"[shards] Falling back to {_alt_shard_base}")
    _shard_base = _alt_shard_base
_shard_train_dir = os.path.join(_shard_base, "train")

# Derive class_names from shard filenames when shards exist —
# avoids image_dataset_from_directory on runs after preprocessing.
if os.path.isdir(_shard_train_dir):
    # Support both single-file ({class}.npz) and chunked ({class}_c000.npz) shards.
    _seen = set()
    for f in os.listdir(_shard_train_dir):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]  # strip .npz
        if "_c" in name and name.split("_c")[-1].isdigit():
            name = name.rsplit("_c", 1)[0]  # strip _c000 suffix
        _seen.add(name)
    _shard_class_names = sorted(_seen)
else:
    _shard_class_names = []

if _shard_class_names:
    class_names = _shard_class_names
    num_classes = len(class_names)
    train_raw = None
    val_raw = None
    data_dir = _shard_train_dir
else:
    # No shards — build dataset from JPEG files.
    if args.stage == "foundation":
        train_dir = os.path.join(base, "train")
        val_dir = os.path.join(base, "val")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            data_dir = train_dir
            train_raw = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                color_mode="rgb",
                shuffle=True,
                **config.DATA_PARAMS)
            val_raw = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                color_mode="rgb",
                shuffle=False,
                **config.DATA_PARAMS)
        else:
            data_dir = os.path.join(base, f'{DOMAIN}_classes')
            train_raw = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.3,
                subset="training",
                color_mode="rgb",
                shuffle=True,
                **config.DATA_PARAMS)
            val_raw = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.3,
                subset="validation",
                color_mode="rgb",
                shuffle=False,
                **config.DATA_PARAMS)
    else:
        data_dir = os.path.join(base, f"midterm_{DOMAIN}_data")
        train_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            color_mode="rgb",
            shuffle=True,
            **config.DATA_PARAMS)
        val_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            color_mode="rgb",
            shuffle=True,
            **config.DATA_PARAMS)
    class_names = train_raw.class_names
    num_classes = len(class_names)


def _check_shards(shard_split_dir: str, names: list):
    """Validate that all classes have shard files.

    Returns list of (label_idx, [file_paths]) or None if any class is missing.
    Does NOT load pixel data.
    """
    if not os.path.isdir(shard_split_dir):
        return None
    class_files = []
    for idx, cname in enumerate(names):
        single = os.path.join(shard_split_dir, f"{cname}.npz")
        chunks = sorted(
            f for f in os.listdir(shard_split_dir)
            if f.startswith(f"{cname}_c") and f.endswith(".npz")
        )
        if os.path.isfile(single):
            files = [single]
        elif chunks:
            files = [os.path.join(shard_split_dir, c) for c in chunks]
        else:
            return None
        class_files.append((idx, files))
    return class_files


def _get_shard_metadata(class_file_list):
    """Read paths arrays only (not RGB) to build label + path arrays cheaply."""
    lbl_parts, path_parts = [], []
    for label_idx, files in class_file_list:
        for fpath in files:
            with np.load(fpath, allow_pickle=True) as d:
                n = len(d["paths"])
                lbl_parts.append(np.full(n, label_idx, dtype=np.int32))
                path_parts.extend(d["paths"].tolist())
    return np.concatenate(lbl_parts), path_parts


def _dataset_from_shards_lazy(class_file_list, shuffle, prepare_fn, batch_size):
    """Streaming tf.data dataset — loads one NPZ chunk (~0.6 GB) at a time via flat_map.

    Eliminates the SHARDS_MAX_RAM_GB cap by never holding all shards in RAM simultaneously.
    """
    h, w = config.DATA_PARAMS["image_size"]

    all_fpaths = []
    all_labels = []
    total_images = 0
    for label_idx, files in class_file_list:
        for fpath in files:
            all_fpaths.append(fpath.encode())
            all_labels.append(label_idx)
            with np.load(fpath, allow_pickle=True) as _d:
                total_images += len(_d["paths"])

    def _load_chunk(fpath_bytes, label_idx):
        def _np_load(fp, li):
            fp = fp.decode()
            d = np.load(fp, allow_pickle=True)
            rgb = d["rgb"].astype(np.float32)
            labels = np.full(len(rgb), li, dtype=np.int32)
            return rgb, labels

        rgb, labels = tf.numpy_function(
            _np_load, [fpath_bytes, label_idx], [tf.float32, tf.int32]
        )
        rgb.set_shape([None, h, w, 3])
        labels.set_shape([None])
        return tf.data.Dataset.from_tensor_slices((rgb, labels))

    path_ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(all_fpaths), tf.constant(all_labels, dtype=tf.int32))
    )
    if shuffle:
        path_ds = path_ds.shuffle(len(all_fpaths), seed=config.DATA_PARAMS["seed"])

    ds = path_ds.flat_map(_load_chunk)
    if shuffle:
        chunk_size = getattr(config, "SHARDS_CHUNK_SIZE", 1000)
        ds = ds.shuffle(chunk_size * 2, seed=config.DATA_PARAMS["seed"])

    total_batches = (total_images + batch_size - 1) // batch_size
    return (
        ds.map(prepare_fn, num_parallel_calls=2)
        .batch(batch_size)
        .apply(tf.data.experimental.assert_cardinality(total_batches))
        .prefetch(2)
    )


_train_shard_files = _check_shards(os.path.join(_shard_base, "train"), class_names)
_val_shard_files   = _check_shards(os.path.join(_shard_base, "val"),   class_names)
_using_shards = (
    not args.no_shards
    and _train_shard_files is not None
    and _val_shard_files is not None
)

if _using_shards:
    print(f"[shards] NPZ shards found (lazy streaming) — {_shard_base}")
    y_train,    _train_file_paths = _get_shard_metadata(_train_shard_files)
    val_labels, _val_file_paths   = _get_shard_metadata(_val_shard_files)
else:
    print("[shards] No shards found — using JPEG decode pipeline.")
    # train_raw / val_raw are None when a shard directory was detected but the NPZ
    # files are incomplete (e.g. preprocessing was interrupted).  Rebuild from JPEG.
    if train_raw is None or val_raw is None:
        print("[shards] Shard directory exists but NPZs are incomplete — rebuilding JPEG datasets.")
        if args.stage == "foundation":
            _train_dir = os.path.join(base, "train")
            _val_dir   = os.path.join(base, "val")
            if os.path.isdir(_train_dir) and os.path.isdir(_val_dir):
                train_raw = tf.keras.utils.image_dataset_from_directory(
                    _train_dir, color_mode="rgb", shuffle=True, **config.DATA_PARAMS)
                val_raw = tf.keras.utils.image_dataset_from_directory(
                    _val_dir, color_mode="rgb", shuffle=False, **config.DATA_PARAMS)
            else:
                _data_dir = os.path.join(base, f"{DOMAIN}_classes")
                train_raw = tf.keras.utils.image_dataset_from_directory(
                    _data_dir, validation_split=0.3, subset="training",
                    color_mode="rgb", shuffle=True, **config.DATA_PARAMS)
                val_raw = tf.keras.utils.image_dataset_from_directory(
                    _data_dir, validation_split=0.3, subset="validation",
                    color_mode="rgb", shuffle=False, **config.DATA_PARAMS)
        else:
            _data_dir = os.path.join(base, f"midterm_{DOMAIN}_data")
            train_raw = tf.keras.utils.image_dataset_from_directory(
                _data_dir, validation_split=0.3, subset="training",
                color_mode="rgb", shuffle=True, **config.DATA_PARAMS)
            val_raw = tf.keras.utils.image_dataset_from_directory(
                _data_dir, validation_split=0.3, subset="validation",
                color_mode="rgb", shuffle=True, **config.DATA_PARAMS)
        class_names = train_raw.class_names
        num_classes  = len(class_names)
    y_train    = np.concatenate([y for _, y in train_raw], axis=0)
    val_labels = np.concatenate([y for _, y in val_raw],   axis=0)
    _val_file_paths   = [os.path.relpath(p, script_dir).replace("\\", "/") for p in val_raw.file_paths]
    _train_file_paths = [os.path.relpath(p, script_dir).replace("\\", "/") for p in train_raw.file_paths]

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
class_weights_dict = dict(enumerate(weights))

print("Validation images per class:", dict(zip(class_names, np.bincount(val_labels))))

# Guard against stale caches / shard label mismatches.
_ymax_tr  = int(np.max(y_train))    if len(y_train)    else -1
_ymax_val = int(np.max(val_labels)) if len(val_labels) else -1
if _ymax_tr >= num_classes or _ymax_val >= num_classes:
    raise SystemExit(
        "Label index out of range for current class_names. This usually means a stale tf.data cache.\n"
        f"  num_classes={num_classes} class_names={class_names}\n"
        f"  max_label_train={_ymax_tr} max_label_val={_ymax_val}\n"
        "Fix: delete outputs/tf_cache/ (or bump config.TF_CACHE_PIPELINE_ID) and rerun."
    )

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ],
    name="data_augmentation",
)

_anti_background_spatial = tf.keras.Sequential(
    [
        tf.keras.layers.RandomTranslation(height_factor=0.12, width_factor=0.12),
        tf.keras.layers.RandomCrop(200, 200),
        tf.keras.layers.Resizing(*config.DATA_PARAMS["image_size"]),
    ],
    name="anti_background_spatial",
)


def _to_triple(norm_rgb: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gray = tf.image.rgb_to_grayscale(norm_rgb)
    return norm_rgb, gray, gray


def prepare_triple_input_train(image, label):
    norm_image = tf.cast(image, tf.float32) / 255.0
    if args.augment:
        aug = data_augmentation(norm_image, training=True)
        aug = tf.image.random_brightness(aug, max_delta=0.10)
        norm_image = tf.clip_by_value(aug, 0.0, 1.0)
    if args.anti_background_aug:
        norm_image = tf.clip_by_value(
            _anti_background_spatial(norm_image, training=True), 0.0, 1.0
        )
    rgb, gray, sobel = _to_triple(norm_image)
    return (rgb, gray, sobel), label


def prepare_triple_input_eval(image, label):
    norm_image = tf.cast(image, tf.float32) / 255.0
    rgb, gray, sobel = _to_triple(norm_image)
    return (rgb, gray, sobel), label


def _prepare_shard_train(rgb_norm, label):
    """Shard variant of prepare_triple_input_train: rgb_norm already in [0, 1]."""
    if args.augment:
        aug = data_augmentation(rgb_norm, training=True)
        aug = tf.image.random_brightness(aug, max_delta=0.10)
        rgb_norm = tf.clip_by_value(aug, 0.0, 1.0)
    if args.anti_background_aug:
        rgb_norm = tf.clip_by_value(
            _anti_background_spatial(rgb_norm, training=True), 0.0, 1.0
        )
    rgb, gray, sobel = _to_triple(rgb_norm)
    return (rgb, gray, sobel), label


def _prepare_shard_eval(rgb_norm, label):
    """Shard variant of prepare_triple_input_eval: rgb_norm already in [0, 1]."""
    rgb, gray, sobel = _to_triple(rgb_norm)
    return (rgb, gray, sobel), label


if _using_shards:
    train_dataset = _dataset_from_shards_lazy(
        _train_shard_files, shuffle=True,
        prepare_fn=_prepare_shard_train,
        batch_size=config.DATA_PARAMS["batch_size"],
    )
    val_dataset = _dataset_from_shards_lazy(
        _val_shard_files, shuffle=False,
        prepare_fn=_prepare_shard_eval,
        batch_size=config.DATA_PARAMS["batch_size"],
    )
else:
    train_dataset = (
        train_raw.map(prepare_triple_input_train, num_parallel_calls=tf.data.AUTOTUNE)
        .cache(_train_cache_prefix)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .cache(_val_cache_prefix)
        .prefetch(tf.data.AUTOTUNE)
    )


def collect_val_predictions(model, val_ds):
    y_true_chunks, y_pred_chunks, prob_chunks = [], [], []
    for images, labels in tqdm(val_ds, desc="Val predictions (reports/plots)"):
        preds = model.predict_on_batch(images)
        p_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
        probs = batch_preds_to_probs(p_np)
        y_true_chunks.append(labels.numpy())
        y_pred_chunks.append(np.argmax(probs, axis=1))
        prob_chunks.append(probs)
    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    prob_rows = np.vstack(prob_chunks)
    return y_true, y_pred, prob_rows


def plot_training_history(history, plot_dir, domain, stage, timestamp):
    """Plot training and validation accuracy/loss curves side by side."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    h = history.history
    n_train = len(h['accuracy'])
    train_x = range(n_train)

    axes[0].plot(train_x, h['accuracy'], label='Train Acc')
    va = h.get('val_accuracy')
    if va is not None and len(va) == n_train:
        axes[0].plot(train_x, va, label='Val Acc')
    elif va is not None:
        axes[0].plot(range(len(va)), va, label='Val Acc (sparse)', marker='o')
    axes[0].set_title(f'Accuracy: {domain} ({stage})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_x, h['loss'], label='Train Loss')
    vl = h.get('val_loss')
    if vl is not None and len(vl) == n_train:
        axes[1].plot(train_x, vl, label='Val Loss')
    elif vl is not None:
        axes[1].plot(range(len(vl)), vl, label='Val Loss (sparse)', marker='o')
    axes[1].set_title(f'Loss: {domain} ({stage})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'history_{domain}_{stage}_{timestamp}.png'))
    plt.close()

if args.mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[mixed precision] float16 policy active — ~30-50% faster on T4/V100/A100.")

model = build_combined_model(num_classes=num_classes)


def _find_layer(m, name):
    """Recursively find a named layer inside a potentially nested Keras model."""
    for layer in m.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer(layer, name)
            if found is not None:
                return found
    return None


# Freeze sobel_learn before the first compile so the Sobel kernel is excluded
# from the optimizer variable list during warmup epochs.
if config.SOBEL_TRAINABLE_WARMUP_EPOCHS > 0:
    _sobel_layer = _find_layer(model, "sobel_learn")
    if _sobel_layer is not None:
        _sobel_layer.trainable = False
        model.compile(
            optimizer="adam",
            loss=focal_loss(),
            metrics=["accuracy"],
        )
        print(
            f"[SobelWarmup] sobel_learn frozen for first "
            f"{config.SOBEL_TRAINABLE_WARMUP_EPOCHS} epoch(s)."
        )

model_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.FUSION_MODELS_PATH_SEGMENTS
)
plot_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.FUSION_PLOTS_PATH_SEGMENTS
)
attribution_dir = _audit_paths["attribution"]
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

class _SobelWarmupCallback(tf.keras.callbacks.Callback):
    """Unfreeze sobel_learn at ``warmup_epochs`` and recompile.

    The initial freeze + compile happens before model.fit() in the main script
    so the Sobel kernel is excluded from the optimizer from epoch 0.
    This callback handles the unfreeze side only.
    """

    def __init__(self, warmup_epochs: int):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, _logs=None):
        if epoch == self.warmup_epochs:
            layer = _find_layer(self.model, "sobel_learn")
            if layer is not None:
                layer.trainable = True
                lr = float(
                    tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                )
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=focal_loss(),
                    metrics=["accuracy"],
                )
                self.model.make_train_function(force=True)
                print(
                    f"[SobelWarmup] Epoch {epoch}: sobel_learn unfrozen "
                    f"and model recompiled at lr={lr:.2e}."
                )


_checkpoint_path = os.path.join(model_dir, f'potato_{DOMAIN}_model_{timestamp}.keras')
callbacks = [
    _SobelWarmupCallback(warmup_epochs=config.SOBEL_TRAINABLE_WARMUP_EPOCHS),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=_checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        min_lr=config.LR_REDUCE_MIN_LR,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.EPOCHS,
    class_weight=class_weights_dict,
    validation_freq=args.validation_freq,
    callbacks=callbacks,
)

model_path_abs = _checkpoint_path
if not os.path.exists(model_path_abs):
    model.save(model_path_abs)
plot_training_history(history, plot_dir, DOMAIN, args.stage, timestamp)

y_true, y_pred, val_probs = collect_val_predictions(model, val_dataset)

_val_img_rel = _val_file_paths
_val_pred_csv = os.path.join(
    _audit_paths["predictions"],
    f"val_predictions_{DOMAIN}_{args.stage}_{timestamp}.csv",
)
_val_df = build_standard_predictions_dataframe(
    run_id=timestamp,
    split="val",
    domain=DOMAIN,
    stage=args.stage,
    class_names=class_names,
    y_true=y_true,
    y_pred=y_pred,
    probs=val_probs,
    image_rel_paths=_val_img_rel,
    model_rel_path=rel_from_script(script_dir, model_path_abs),
    high_confidence_threshold=args.high_conf_threshold,
)
_val_df.to_csv(_val_pred_csv, index=False)
print(f"[audit] Val predictions (master schema): {_val_pred_csv}")

_p7 = write_phase7_audit_bundle(
    pred_df=_val_df,
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
    metrics_dir=_audit_paths["metrics"],
    failures_dir=_audit_paths["failures"],
    image_lookup_root=script_dir,
    name_tag=f"{DOMAIN}_{args.stage}_{timestamp}",
    split_label=f"val_{DOMAIN}_{args.stage}",
    high_confidence_threshold=args.high_conf_threshold,
)
print(
    f"[audit] Phase 7: {_p7['per_class_metrics_csv']} | "
    f"{_p7.get('failure_gallery_png') or 'no gallery'}"
)

write_run_info(
    _audit_paths["meta"],
    {
        "run_id": timestamp,
        "domain": DOMAIN,
        "stage": args.stage,
        "model_path": rel_from_script(script_dir, model_path_abs),
        "plots_dir": rel_from_script(script_dir, plot_dir),
        "attribution_dir": rel_from_script(script_dir, attribution_dir),
        "val_predictions_csv": rel_from_script(script_dir, _val_pred_csv),
        "class_names": class_names,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "high_confidence_wrong_threshold": args.high_conf_threshold,
        "gradcam_target": args.gradcam_target,
        "gradcam_show_shallow_rgb": not args.no_gradcam_shallow,
        "config_snapshot": {
            "epochs_config": config.EPOCHS,
            "epochs_trained": len(history.history["accuracy"]),
            "batch_size": config.DATA_PARAMS["batch_size"],
            "image_size": list(config.DATA_PARAMS["image_size"]),
            "dropout_rate": config.DROPOUT_RATE,
            "sobel_warmup_epochs": config.SOBEL_TRAINABLE_WARMUP_EPOCHS,
            "lr_reduce_patience": config.LR_REDUCE_PATIENCE,
            "lr_reduce_factor": config.LR_REDUCE_FACTOR,
            "lr_reduce_min_lr": config.LR_REDUCE_MIN_LR,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "augment": args.augment,
            "anti_background_aug": args.anti_background_aug,
            "validation_freq": args.validation_freq,
        },
        "phase7": {
            "per_class_metrics_csv": rel_from_script(
                script_dir, _p7["per_class_metrics_csv"]
            ),
            "confusion_matrix_counts_csv": rel_from_script(
                script_dir, _p7["confusion_matrix_counts_csv"]
            ),
            "confusion_matrix_row_norm_csv": rel_from_script(
                script_dir, _p7["confusion_matrix_row_norm_csv"]
            ),
            "slice_metrics_csv": rel_from_script(script_dir, _p7["slice_metrics_csv"]),
            "failure_gallery_png": rel_from_script(script_dir, _p7["failure_gallery_png"])
            if _p7.get("failure_gallery_png")
            else "",
        },
    },
)
print(f"[audit] Run manifest: {os.path.join(_audit_paths['meta'], 'run_info.json')}")


def save_report_as_image(y_true, y_pred, target_names, plot_dir, timestamp):
    present_classes = np.unique(y_true)
    filtered_names = [target_names[i] for i in present_classes]

    report = classification_report(
        y_true, y_pred, 
        labels=present_classes, 
        target_names=filtered_names, 
        output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    plt.figure(figsize=(8, 5))
    plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')
    plt.axis('off')
    plt.title(f"Results: {DOMAIN}")
    plt.savefig(os.path.join(plot_dir, f'report_{DOMAIN}_{timestamp}.png'), bbox_inches='tight')
    plt.close()


save_report_as_image(y_true, y_pred, class_names, plot_dir, timestamp)

cm = confusion_matrix(y_true, y_pred)

row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_perc = np.divide(
    cm.astype('float'), row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0)

num_classes = len(class_names)
fig_size = max(10, num_classes * 0.8)

plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm_perc * 100,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'Percentage (%)'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix: {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_{DOMAIN}_{timestamp}.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix (Counts): {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_counts_{DOMAIN}_{timestamp}.png'),
            bbox_inches='tight', dpi=300)
plt.close()


def plot_roc_curves(y_true, probs, class_names, plot_dir, domain, timestamp):
    """One-vs-Rest ROC curves for each class plus macro-average."""
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro average (interpolated)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    fig_size = max(7, n * 0.5)
    plt.figure(figsize=(fig_size, fig_size))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, color in zip(range(n), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], color="black", lw=2, linestyle="--",
             label=f"Macro avg (AUC={roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves: {domain.upper()}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"roc_{domain}_{timestamp}.png"),
                bbox_inches="tight", dpi=300)
    plt.close()


_roc_plot_path = os.path.join(plot_dir, f"roc_{DOMAIN}_{timestamp}.png")
plot_roc_curves(y_true, val_probs, class_names, plot_dir, DOMAIN, timestamp)
merge_run_info(
    _audit_paths["meta"],
    {"roc_plot_png": rel_from_script(script_dir, _roc_plot_path)},
)

_manifest_path = append_runs_manifest(
    script_dir,
    {
        "run_id": timestamp,
        "domain": DOMAIN,
        "stage": args.stage,
        "overall_accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "epochs_trained": len(history.history["accuracy"]),
        "sobel_warmup_epochs": config.SOBEL_TRAINABLE_WARMUP_EPOCHS,
        "augment": args.augment,
        "anti_background_aug": args.anti_background_aug,
        "model_path": rel_from_script(script_dir, model_path_abs),
        "roc_plot_png": rel_from_script(script_dir, _roc_plot_path),
        "per_class_metrics_csv": rel_from_script(script_dir, _p7["per_class_metrics_csv"]),
    },
)
print(f"[audit] Runs manifest: {_manifest_path}")


def _val_dataset_and_paths_for_attribution() -> tuple[tf.data.Dataset, list[str]]:
    """Same val split as training, shuffle=False + file_paths for occlusion/IG CSV rows."""
    if _using_shards:
        ds = _dataset_from_shards_lazy(
            _val_shard_files, shuffle=False,
            prepare_fn=_prepare_shard_eval,
            batch_size=config.DATA_PARAMS["batch_size"],
        )
        return ds, _val_file_paths
    base = os.path.join(script_dir, args.base_dir)
    if (
        args.stage == "foundation"
        and os.path.isdir(os.path.join(base, "train"))
        and os.path.isdir(os.path.join(base, "val"))
    ):
        val_dir = os.path.join(base, "val")
        vraw = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            color_mode="rgb",
            shuffle=False,
            **config.DATA_PARAMS,
        )
    elif args.stage == "foundation":
        data_dir = os.path.join(base, f"{DOMAIN}_classes")
        vraw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            color_mode="rgb",
            shuffle=False,
            seed=config.DATA_PARAMS["seed"],
            **config.DATA_PARAMS,
        )
    else:
        data_dir = os.path.join(base, f"midterm_{DOMAIN}_data")
        vraw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            color_mode="rgb",
            shuffle=False,
            seed=config.DATA_PARAMS["seed"],
            **config.DATA_PARAMS,
        )
    ds = vraw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    rels = [
        os.path.relpath(p, script_dir).replace("\\", "/") for p in vraw.file_paths
    ]
    return ds, rels


def _train_dataset_and_paths_for_neighbors() -> tuple[tf.data.Dataset, list[str]]:
    """Unshuffled train split + paths; eval map only (matches val/field inference)."""
    if _using_shards:
        ds = _dataset_from_shards_lazy(
            _train_shard_files, shuffle=False,
            prepare_fn=_prepare_shard_eval,
            batch_size=config.DATA_PARAMS["batch_size"],
        )
        return ds, _train_file_paths
    base = os.path.join(script_dir, args.base_dir)
    if (
        args.stage == "foundation"
        and os.path.isdir(os.path.join(base, "train"))
        and os.path.isdir(os.path.join(base, "val"))
    ):
        train_dir = os.path.join(base, "train")
        traw = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode="rgb",
            shuffle=False,
            **config.DATA_PARAMS,
        )
    elif args.stage == "foundation":
        data_dir = os.path.join(base, f"{DOMAIN}_classes")
        traw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            color_mode="rgb",
            shuffle=False,
            seed=config.DATA_PARAMS["seed"],
            **config.DATA_PARAMS,
        )
    else:
        data_dir = os.path.join(base, f"midterm_{DOMAIN}_data")
        traw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            color_mode="rgb",
            shuffle=False,
            seed=config.DATA_PARAMS["seed"],
            **config.DATA_PARAMS,
        )
    ds = traw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    rels = [
        os.path.relpath(p, script_dir).replace("\\", "/") for p in traw.file_paths
    ]
    return ds, rels


explain_my_model(
    model=model,
    validation_data=val_dataset,
    save_dir=attribution_dir,
    name_tag=timestamp,
    class_names=class_names,
    gradcam_target=args.gradcam_target,
    show_shallow_rgb=not args.no_gradcam_shallow,
)

if args.occlusion:
    _occ_csv = os.path.join(
        _audit_paths["metrics"],
        f"occlusion_summary_{DOMAIN}_{args.stage}_{timestamp}.csv",
    )
    _occ_ds, _occ_paths = _val_dataset_and_paths_for_attribution()
    _occ_written = run_occlusion_val_batch(
        model=model,
        val_dataset=_occ_ds,
        class_names=class_names,
        save_dir=attribution_dir,
        name_tag=timestamp,
        run_id=timestamp,
        image_rel_paths=_occ_paths,
        num_samples=args.occlusion_samples,
        patch_size=args.occlusion_patch,
        stride=args.occlusion_stride,
        fill_value=args.occlusion_fill,
        occlusion_target=args.occlusion_target,
        csv_path=_occ_csv,
    )
    if _occ_written:
        merge_run_info(
            _audit_paths["meta"],
            {
                "occlusion_summary_csv": rel_from_script(script_dir, _occ_csv),
                "occlusion": {
                    "samples": args.occlusion_samples,
                    "patch_size": args.occlusion_patch,
                    "stride": args.occlusion_stride,
                    "fill_value": args.occlusion_fill,
                    "target": args.occlusion_target,
                },
            },
        )
        print(f"[audit] Occlusion: figures in {attribution_dir}, summary {_occ_csv}")

if args.ig:
    _ig_csv = os.path.join(
        _audit_paths["metrics"],
        f"ig_summary_{DOMAIN}_{args.stage}_{timestamp}.csv",
    )
    _ig_ds, _ig_paths = _val_dataset_and_paths_for_attribution()
    _ig_written = run_integrated_gradients_batch(
        model=model,
        val_dataset=_ig_ds,
        class_names=class_names,
        save_dir=attribution_dir,
        name_tag=timestamp,
        run_id=timestamp,
        image_rel_paths=_ig_paths,
        num_samples=args.ig_samples,
        m_steps=args.ig_steps,
        baseline="black",
        ig_target=args.ig_target,
        csv_path=_ig_csv,
        compare_gradcam=not args.no_ig_gradcam_compare,
    )
    if _ig_written:
        merge_run_info(
            _audit_paths["meta"],
            {
                "integrated_gradients_summary_csv": rel_from_script(script_dir, _ig_csv),
                "integrated_gradients": {
                    "samples": args.ig_samples,
                    "m_steps": args.ig_steps,
                    "baseline": "black",
                    "target": args.ig_target,
                    "gradcam_compare": not args.no_ig_gradcam_compare,
                },
            },
        )
        print(f"[audit] Integrated Gradients: {attribution_dir}, summary {_ig_csv}")

if args.neighbors:
    _ntr_ds, _ntr_paths = _train_dataset_and_paths_for_neighbors()
    _nval_ds, _nval_paths = _val_dataset_and_paths_for_attribution()
    _nbr = run_train_neighbor_pipeline(
        full_model=model,
        train_dataset_eval=_ntr_ds,
        train_rel_paths=_ntr_paths,
        val_dataset=_nval_ds,
        val_rel_paths=_nval_paths,
        class_names=class_names,
        neighbors_dir=_audit_paths["neighbors"],
        name_tag=timestamp,
        run_id=timestamp,
        split_label_val="val",
        top_k=args.neighbors_top_k,
        val_query_samples=args.neighbors_val_samples,
        max_grid_figures=args.neighbors_grid_figures,
        project_root=script_dir,
        embedding_layer=args.neighbors_embedding_layer,
    )
    merge_run_info(
        _audit_paths["meta"],
        {
            "neighbor_train_embeddings_npz": rel_from_script(
                script_dir, _nbr["train_embeddings_npz"]
            ),
            "neighbor_train_manifest_csv": rel_from_script(
                script_dir, _nbr["train_manifest_csv"]
            ),
            "neighbor_val_table_csv": rel_from_script(
                script_dir, _nbr["val_neighbor_table_csv"]
            ),
            "neighbors": {
                "embedding_layer": _nbr["embedding_layer"],
                "top_k": args.neighbors_top_k,
                "val_query_samples": args.neighbors_val_samples,
                "grid_figures": args.neighbors_grid_figures,
            },
        },
    )
    print(
        f"[audit] Neighbors: {_audit_paths['neighbors']} "
        f"(train bank + val table {_nbr['val_neighbor_table_csv']})"
    )

if args.counterfactuals:
    _cfd, _cfp = _val_dataset_and_paths_for_attribution()
    _cf_bank = None
    if args.neighbors:
        _npz = np.load(_nbr["train_embeddings_npz"])
        _cf_bank = (
            np.asarray(_npz["embeddings"], dtype=np.float32),
            np.asarray(_npz["y_true"], dtype=np.int32),
            pd.read_csv(_nbr["train_manifest_csv"]),
        )
    _cf_out = run_counterfactual_audit(
        full_model=model,
        val_dataset=_cfd,
        val_rel_paths=_cfp,
        class_names=class_names,
        metrics_dir=_audit_paths["metrics"],
        attribution_dir=_audit_paths["attribution"],
        name_tag=timestamp,
        run_id=timestamp,
        split_label=f"val_{DOMAIN}",
        num_samples=args.counterfactual_samples,
        contrastive_bank=_cf_bank,
        contrastive_search_cap=args.counterfactual_contrastive_search_cap,
        contrastive_exclude=args.counterfactual_exclude,
        patch_size=args.counterfactual_patch,
        stride=args.counterfactual_stride,
        fill_value=args.counterfactual_fill,
        max_mask_patches=args.counterfactual_max_mask_patches,
        embedding_layer=args.neighbors_embedding_layer,
    )
    _payload = {
        "counterfactuals": {
            "mask_flip_summary_csv": "",
            "contrastive_table_csv": "",
            "contrastive_ran": bool(_cf_bank is not None),
            "samples": args.counterfactual_samples,
            "contrastive_search_cap": args.counterfactual_contrastive_search_cap,
            "max_mask_patches": args.counterfactual_max_mask_patches,
            "patch": args.counterfactual_patch,
            "stride": args.counterfactual_stride,
            "fill": args.counterfactual_fill,
            "contrastive_exclude": args.counterfactual_exclude,
        }
    }
    if _cf_out.get("mask_flip_summary_csv"):
        _payload["counterfactuals"]["mask_flip_summary_csv"] = rel_from_script(
            script_dir, _cf_out["mask_flip_summary_csv"]
        )
    if _cf_out.get("contrastive_table_csv"):
        _payload["counterfactuals"]["contrastive_table_csv"] = rel_from_script(
            script_dir, _cf_out["contrastive_table_csv"]
        )
    merge_run_info(_audit_paths["meta"], _payload)
    print(
        f"[audit] Counterfactuals: {_cf_out.get('mask_flip_summary_csv', '')} "
        f"+ { _cf_out.get('contrastive_table_csv') or 'no contrastive (run with --neighbors)'}"
    )

if args.shortcut_audit:
    _sd, _sp = _val_dataset_and_paths_for_attribution()
    _sc_csv = os.path.join(
        _audit_paths["metrics"],
        f"shortcut_border_focus_val_{DOMAIN}_{args.stage}_{timestamp}.csv",
    )
    _sc_written = run_shortcut_background_audit(
        model=model,
        dataset=_sd,
        rel_paths=_sp,
        class_names=class_names,
        run_id=timestamp,
        split_label=f"val_{DOMAIN}",
        save_csv_path=_sc_csv,
        num_samples=args.shortcut_audit_samples,
        border_frac=float(args.shortcut_border_frac),
        gradcam_target=args.gradcam_target,
        gradcam_layer=config.GRADCAM_RGB_DEEP_LAYER,
    )
    if _sc_written:
        merge_run_info(
            _audit_paths["meta"],
            {
                "shortcut_audit": {
                    "border_focus_csv": rel_from_script(script_dir, _sc_written),
                    "samples": args.shortcut_audit_samples,
                    "border_frac": float(args.shortcut_border_frac),
                    "gradcam_target": args.gradcam_target,
                    "gradcam_layer": config.GRADCAM_RGB_DEEP_LAYER,
                    "note": (
                        "Heuristic: Grad-CAM energy near image borders (tray/edge shortcuts). "
                        "Compare runs with and without --anti-background-aug."
                    ),
                }
            },
        )
        print(f"[audit] Shortcut audit: {_sc_written}")