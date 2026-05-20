"""Fine-tune the EfficientNetB3 backbone on a saved fusion model.

Loads a previously trained .keras model, unfreezes the EfficientNetB3 backbone,
and continues training at a low learning rate so the backbone adjusts its features
toward potato disease patterns without destroying its ImageNet knowledge.

Usage:
    python finetune_model.py --model outputs/fusion/models/potato_leaf_model_<timestamp>.keras
"""

import argparse
import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

import config
from focal_loss import focal_loss
from audit_bundle import (
    append_runs_manifest,
    ensure_audit_run,
    merge_run_info,
    rel_from_script,
    write_run_info,
)
from build_sobel_model import (  # noqa: F401 — imports register all @keras_serializable classes
    AlphaWeightedSum,
    SobelKernelInitializer,
    SobelMagnitude,
    sobel_edge_layer,
)
from grad_cam_visualizer import ensure_grad_targets, explain_my_model
from prediction_utils import (
    HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
    PREDICTION_SCHEMA_VERSION,
    batch_preds_to_probs,
    build_standard_predictions_dataframe,
)
from structured_reporting import write_phase7_audit_bundle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to saved .keras model from train_model.py.")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Learning rate for fine-tuning (default: 1e-5).",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Max fine-tuning epochs (early stopping will cut this short).",
)
parser.add_argument(
    "--unfreeze-layers",
    type=int,
    default=None,
    help=(
        "Unfreeze only the last N layers of EfficientNetB3 (default: None = unfreeze all). "
        "Use 30-50 for a more conservative fine-tune."
    ),
)
parser.add_argument(
    "--gradcam-target",
    default="predicted",
    choices=["predicted", "true"],
)
parser.add_argument(
    "--base-dir",
    type=str,
    default=config.BASE_DIR,
)
parser.add_argument(
    "--no-shards",
    action="store_true",
    help="Skip NPZ shard loading and use the JPEG image pipeline instead.",
)
args = parser.parse_args()


def _print_device_summary():
    print(f"[TensorFlow] {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[TensorFlow] GPU count: {len(gpus)}")


_print_device_summary()

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = "leaf"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_audit_paths = ensure_audit_run(script_dir, timestamp)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = tf.keras.models.load_model(
    args.model,
    custom_objects={"sobel_edge_layer": sobel_edge_layer},
)
ensure_grad_targets(model)
print("Model loaded.")

# ── Unfreeze EfficientNetB3 backbone ─────────────────────────────────────────
_backbone_name = next(
    (l.name for l in model.layers if l.name.startswith("efficientnet")), None
)
if _backbone_name is None:
    raise ValueError("No EfficientNet backbone found in model.")
backbone = model.get_layer(_backbone_name)
if args.unfreeze_layers is None:
    backbone.trainable = True
    print(f"EfficientNetB3 fully unfrozen ({len(backbone.layers)} layers).")
else:
    backbone.trainable = True
    for layer in backbone.layers[: -args.unfreeze_layers]:
        layer.trainable = False
    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    print(f"EfficientNetB3 partially unfrozen: {unfrozen}/{len(backbone.layers)} layers trainable.")

trainable_count = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f"Total trainable parameters: {trainable_count:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
    loss=focal_loss(),
    metrics=["accuracy"],
)
print(f"Recompiled with lr={args.lr}")

# ── Load datasets (shards preferred, JPEG fallback) ───────────────────────────
base = os.path.join(script_dir, args.base_dir)
_shard_base = os.path.join(base, config.SHARDS_SUBDIR)
# Fall back to processed_data_v2 if the default location has no shards
_alt_shard_base = os.path.join(script_dir, "processed_data_v2", config.SHARDS_SUBDIR)
if not os.path.isdir(_shard_base) and os.path.isdir(_alt_shard_base):
    print(f"[shards] Falling back to {_alt_shard_base}")
    _shard_base = _alt_shard_base
_shard_train_dir = os.path.join(_shard_base, "train")

if os.path.isdir(_shard_train_dir):
    _seen = set()
    for f in os.listdir(_shard_train_dir):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        if "_c" in name and name.split("_c")[-1].isdigit():
            name = name.rsplit("_c", 1)[0]
        _seen.add(name)
    _shard_class_names = sorted(_seen)
else:
    _shard_class_names = []


def _check_shards(shard_split_dir, names):
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
    lbl_parts, path_parts = [], []
    for label_idx, files in class_file_list:
        for fpath in files:
            with np.load(fpath, allow_pickle=True) as d:
                n = len(d["paths"])
                lbl_parts.append(np.full(n, label_idx, dtype=np.int32))
                path_parts.extend(d["paths"].tolist())
    return np.concatenate(lbl_parts), path_parts


def _dataset_from_shards_lazy(class_file_list, shuffle, prepare_fn, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    h, w = config.DATA_PARAMS["image_size"]
    all_fpaths, all_labels = [], []
    total_images = 0
    for label_idx, files in class_file_list:
        for fpath in files:
            all_fpaths.append(fpath.encode())
            all_labels.append(label_idx)
            with np.load(fpath, allow_pickle=True) as _d:
                total_images += len(_d["paths"])

    def _load_chunk(fpath_bytes, label_idx):
        def _np_load(fp, li):
            d = np.load(fp.decode(), allow_pickle=True)
            rgb = d["rgb"].astype(np.float32)
            return rgb, np.full(len(rgb), li, dtype=np.int32)
        rgb, labels = tf.numpy_function(_np_load, [fpath_bytes, label_idx], [tf.float32, tf.int32])
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
        ds = ds.shuffle(getattr(config, "SHARDS_CHUNK_SIZE", 1000) * 2, seed=config.DATA_PARAMS["seed"])
    total_batches = (total_images + batch_size - 1) // batch_size
    return (
        ds.map(prepare_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .apply(tf.data.experimental.assert_cardinality(total_batches))
        .prefetch(AUTOTUNE)
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


def _to_triple(norm_rgb):
    gray = tf.image.rgb_to_grayscale(norm_rgb)
    return norm_rgb, gray, gray


def _prepare_shard_train(rgb_norm, label):
    aug = data_augmentation(rgb_norm, training=True)
    rgb_norm = tf.clip_by_value(aug, 0.0, 1.0)
    return _to_triple(rgb_norm), label


def _prepare_shard_eval(rgb_norm, label):
    return _to_triple(rgb_norm), label


def prepare_triple_input_train(image, label):
    norm = tf.cast(image, tf.float32) / 255.0
    aug = data_augmentation(norm, training=True)
    norm = tf.clip_by_value(aug, 0.0, 1.0)
    return _to_triple(norm), label


def prepare_triple_input_eval(image, label):
    norm = tf.cast(image, tf.float32) / 255.0
    return _to_triple(norm), label


_train_shard_files = _check_shards(os.path.join(_shard_base, "train"), _shard_class_names) if _shard_class_names else None
_val_shard_files   = _check_shards(os.path.join(_shard_base, "val"),   _shard_class_names) if _shard_class_names else None
_using_shards = (
    not args.no_shards
    and _train_shard_files is not None
    and _val_shard_files is not None
)

if _using_shards:
    print(f"[shards] NPZ shards found (lazy streaming) — {_shard_base}")
    y_train,    _train_paths = _get_shard_metadata(_train_shard_files)
    val_labels, _val_paths   = _get_shard_metadata(_val_shard_files)
    class_names = _shard_class_names

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
    print("[shards] No shards — using JPEG pipeline.")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, color_mode="rgb", shuffle=True, **config.DATA_PARAMS
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir, color_mode="rgb", shuffle=False, **config.DATA_PARAMS
    )
    class_names = train_raw.class_names
    y_train     = np.concatenate([y for _, y in train_raw], axis=0)
    val_labels  = np.concatenate([y for _, y in val_raw],   axis=0)
    _val_paths  = [os.path.relpath(p, script_dir).replace("\\", "/") for p in val_raw.file_paths]

    train_dataset = (
        train_raw.map(prepare_triple_input_train, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(weights))

# ── Fine-tune ─────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]

print(f"\nFine-tuning for up to {args.epochs} epochs at lr={args.lr}...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model_dir = os.path.join(script_dir, config.OUTPUT_DIR, *config.FUSION_MODELS_PATH_SEGMENTS)
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"potato_{DOMAIN}_model_{timestamp}.keras")
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ── Val predictions + reports ─────────────────────────────────────────────────
plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, *config.FUSION_PLOTS_PATH_SEGMENTS)
attribution_dir = _audit_paths["attribution"]
os.makedirs(plot_dir, exist_ok=True)


def collect_val_predictions(m, ds):
    y_true_c, y_pred_c, prob_c = [], [], []
    for images, labels in tqdm(ds, desc="Val predictions"):
        preds = m.predict_on_batch(images)
        p_np  = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
        probs = batch_preds_to_probs(p_np)
        y_true_c.append(labels.numpy())
        y_pred_c.append(np.argmax(probs, axis=1))
        prob_c.append(probs)
    return np.concatenate(y_true_c), np.concatenate(y_pred_c), np.vstack(prob_c)


y_true, y_pred, val_probs = collect_val_predictions(model, val_dataset)

_val_pred_csv = os.path.join(
    _audit_paths["predictions"],
    f"val_predictions_{DOMAIN}_finetune_{timestamp}.csv",
)
_val_df = build_standard_predictions_dataframe(
    run_id=timestamp,
    split="val",
    domain=DOMAIN,
    stage="finetune",
    class_names=class_names,
    y_true=y_true,
    y_pred=y_pred,
    probs=val_probs,
    image_rel_paths=_val_paths,
    model_rel_path=rel_from_script(script_dir, model_path),
    high_confidence_threshold=HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
)
_val_df.to_csv(_val_pred_csv, index=False)

_p7 = write_phase7_audit_bundle(
    pred_df=_val_df,
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
    metrics_dir=_audit_paths["metrics"],
    failures_dir=_audit_paths["failures"],
    image_lookup_root=script_dir,
    name_tag=f"{DOMAIN}_finetune_{timestamp}",
    split_label=f"val_{DOMAIN}_finetune",
    high_confidence_threshold=HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
)

# ROC curves
def plot_roc_curves(y_true, probs, class_names, plot_dir, timestamp):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n)) / n
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    plt.figure(figsize=(max(7, n * 0.5), max(7, n * 0.5)))
    for i, color in zip(range(n), plt.cm.tab10(np.linspace(0, 1, n))):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], "k--", lw=2,
             label=f"Macro avg (AUC={roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {DOMAIN.upper()} (finetune)", fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"roc_{DOMAIN}_finetune_{timestamp}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    return path

_roc_path = plot_roc_curves(y_true, val_probs, class_names, plot_dir, timestamp)

write_run_info(
    _audit_paths["meta"],
    {
        "run_id": timestamp,
        "domain": DOMAIN,
        "stage": "finetune",
        "base_model": args.model,
        "finetune_lr": args.lr,
        "finetune_epochs_ran": len(history.history["loss"]),
        "unfreeze_layers": args.unfreeze_layers,
        "model_path": rel_from_script(script_dir, model_path),
        "val_predictions_csv": rel_from_script(script_dir, _val_pred_csv),
        "roc_plot_png": rel_from_script(script_dir, _roc_path),
        "class_names": class_names,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "phase7": {
            "per_class_metrics_csv": rel_from_script(script_dir, _p7["per_class_metrics_csv"]),
            "confusion_matrix_counts_csv": rel_from_script(script_dir, _p7["confusion_matrix_counts_csv"]),
        },
    },
)

append_runs_manifest(
    script_dir,
    {
        "run_id": timestamp,
        "domain": DOMAIN,
        "stage": "finetune",
        "overall_accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "epochs_trained": len(history.history["loss"]),
        "finetune_lr": args.lr,
        "unfreeze_layers": args.unfreeze_layers,
        "model_path": rel_from_script(script_dir, model_path),
        "roc_plot_png": rel_from_script(script_dir, _roc_path),
    },
)

# Grad-CAM
explain_my_model(
    model=model,
    validation_data=val_dataset,
    save_dir=attribution_dir,
    name_tag=timestamp,
    class_names=class_names,
    gradcam_target=args.gradcam_target,
    num_samples=config.GRADCAM_NUM_SAMPLES_DEFAULT,
)

print(f"\nDone. Fine-tuned model: {model_path}")
