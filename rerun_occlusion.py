"""Rerun occlusion sensitivity on a saved model without retraining.

Usage:
    # First N images from val set (default):
    python rerun_occlusion.py --model outputs/fusion/models/potato_leaf_model_20260408_002451.keras

    # 3 images per class (recommended for reports):
    python rerun_occlusion.py --model outputs/fusion/models/potato_leaf_model_20260408_002451.keras --per-class 3

    # Specific images by val-set index (cohesive with Grad-CAM/IG):
    python rerun_occlusion.py --model outputs/fusion/models/potato_leaf_model_20260408_002451.keras --indices 3 17 42 88 105
"""

import argparse
import os

import numpy as np
import tensorflow as tf

import config
from build_sobel_model import sobel_edge_layer
from occlusion_attribution import run_occlusion_val_batch

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to saved .keras model file.")
parser.add_argument(
    "--samples",
    type=int,
    default=config.OCCLUSION_NUM_SAMPLES_DEFAULT,
    help="Number of validation images to visualize (ignored when --indices or --per-class is set).",
)
parser.add_argument(
    "--per-class",
    type=int,
    default=None,
    metavar="N",
    help="Select N images per class, evenly spread across val set.",
)
parser.add_argument(
    "--indices",
    type=int,
    nargs="+",
    default=None,
    help="Specific val-set indices to visualize (e.g. --indices 3 17 42). "
         "Use the same values across all rerun_* scripts for a cohesive report.",
)
parser.add_argument(
    "--patch",
    type=int,
    default=config.OCCLUSION_PATCH_DEFAULT,
    help="Occlusion patch size in pixels.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=config.OCCLUSION_STRIDE_DEFAULT,
    help="Stride between occlusion patches.",
)
parser.add_argument(
    "--fill",
    type=float,
    default=config.OCCLUSION_FILL_DEFAULT,
    help="Fill value for masked patches (0=black, 0.5=gray, 1=white).",
)
parser.add_argument(
    "--target",
    default="predicted",
    choices=["predicted", "true"],
    help="Which class to explain: the predicted class or the true label.",
)
parser.add_argument(
    "--base-dir",
    type=str,
    default=None,
    help="Root folder for processed images (e.g. processed_data_v2). Overrides config.BASE_DIR.",
)
parser.add_argument(
    "--split",
    type=str,
    default="val",
    choices=["train", "val", "test"],
    help="Which split to run XAI on (default: val).",
)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
print(f"Loading model: {args.model}")
model = tf.keras.models.load_model(
    args.model,
    custom_objects={"sobel_edge_layer": sobel_edge_layer, "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
)
print("Model loaded.")

# Load dataset (shuffle=False so indices are stable)
_base = args.base_dir if args.base_dir is not None else config.BASE_DIR
val_dir = os.path.join(script_dir, _base, args.split)
val_raw = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    color_mode="rgb",
    shuffle=False,
    **config.DATA_PARAMS,
)
class_names = val_raw.class_names
num_classes = len(class_names)


def prepare_triple_input_eval(image, label):
    norm = tf.cast(image, tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(norm)
    return (norm, gray, gray), label


def _build_indexed_dataset(indices):
    """Return (dataset, rel_paths) for a single batch of the images at the given indices."""
    idx_set = set(indices)
    max_idx = max(indices)
    collected = {}
    for i, (img, label) in enumerate(val_raw.unbatch()):
        if i in idx_set:
            collected[i] = (img.numpy(), int(label.numpy()))
        if i > max_idx:
            break
    imgs, labels, paths = [], [], []
    for idx in indices:
        if idx in collected:
            imgs.append(collected[idx][0])
            labels.append(collected[idx][1])
            paths.append(
                os.path.relpath(val_raw.file_paths[idx], script_dir).replace("\\", "/")
            )
    imgs_t = tf.constant(np.stack(imgs), dtype=tf.uint8)
    labels_t = tf.constant(labels, dtype=tf.int32)
    ds = (
        tf.data.Dataset.from_tensor_slices((imgs_t, labels_t))
        .batch(len(imgs))
        .map(prepare_triple_input_eval)
    )
    return ds, paths


def _pick_per_class_indices(n_per_class):
    """Collect n_per_class indices per class, evenly spaced. Prints indices for reuse."""
    class_buckets = {c: [] for c in range(num_classes)}
    for i, (_, label) in enumerate(val_raw.unbatch()):
        c = int(label.numpy())
        class_buckets[c].append(i)

    selected = []
    for c in range(num_classes):
        bucket = class_buckets[c]
        if len(bucket) == 0:
            continue
        if len(bucket) <= n_per_class:
            picks = bucket
        else:
            step = len(bucket) // n_per_class
            picks = [bucket[j * step] for j in range(n_per_class)]
        selected.extend(picks)
        print(f"  {class_names[c]}: indices {picks}")

    selected.sort()
    print(f"\nSelected {len(selected)} indices total: {selected}")
    print("To reuse these exact images across Grad-CAM/IG, run with:")
    print(f"  --indices {' '.join(str(i) for i in selected)}\n")
    return selected


if args.indices:
    print(f"Using specified indices: {args.indices}")
    val_dataset, image_rel_paths = _build_indexed_dataset(args.indices)
    num_samples = len(args.indices)
elif args.per_class:
    print(f"Selecting {args.per_class} images per class ({args.per_class * num_classes} total)...")
    indices = _pick_per_class_indices(args.per_class)
    val_dataset, image_rel_paths = _build_indexed_dataset(indices)
    num_samples = len(indices)
else:
    val_dataset = (
        val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    image_rel_paths = [
        os.path.relpath(p, script_dir).replace("\\", "/") for p in val_raw.file_paths
    ]
    num_samples = args.samples

# Output to same audit run folder as the model
model_name = os.path.splitext(os.path.basename(args.model))[0]
timestamp = model_name.replace("potato_leaf_model_", "").replace("potato_tube_model_", "")
save_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, timestamp, "attribution"
)
metrics_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, timestamp, "metrics"
)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

csv_path = os.path.join(metrics_dir, f"occlusion_summary_{timestamp}.csv")

print(f"Saving occlusion outputs to: {save_dir}")
run_occlusion_val_batch(
    model=model,
    val_dataset=val_dataset,
    class_names=class_names,
    save_dir=save_dir,
    name_tag=timestamp,
    run_id=timestamp,
    image_rel_paths=image_rel_paths,
    num_samples=num_samples,
    patch_size=args.patch,
    stride=args.stride,
    fill_value=args.fill,
    occlusion_target=args.target,
    csv_path=csv_path,
)
print(f"Done. CSV summary: {csv_path}")
