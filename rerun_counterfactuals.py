"""Rerun counterfactual audit on a saved model without retraining.

Optionally loads a pre-built training embedding bank (from rerun_neighbors.py) to
enable contrastive k-NN lookups alongside the mask-flip search.

Usage — mask-flip only (no contrastive bank):
    python rerun_counterfactuals.py --model outputs/fusion/models/potato_leaf_model_20260331_233302.keras

Usage — with contrastive bank from rerun_neighbors.py:
    python rerun_counterfactuals.py \
        --model outputs/fusion/models/potato_leaf_model_20260331_233302.keras \
        --train-npz outputs/fusion/audit/20260331_233302/neighbors/train_embeddings_20260331_233302.npz \
        --train-manifest outputs/fusion/audit/20260331_233302/neighbors/train_manifest_20260331_233302.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from build_sobel_model import sobel_edge_layer
from counterfactual_audit import run_counterfactual_audit

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to saved .keras model file.")
parser.add_argument(
    "--samples",
    type=int,
    default=config.COUNTERFACTUAL_VAL_SAMPLES_DEFAULT,
    help="Number of val images to process.",
)
parser.add_argument(
    "--train-npz",
    default=None,
    help="Path to train_embeddings_*.npz from rerun_neighbors.py (enables contrastive k-NN).",
)
parser.add_argument(
    "--train-manifest",
    default=None,
    help="Path to train_manifest_*.csv from rerun_neighbors.py.",
)
parser.add_argument(
    "--search-cap",
    type=int,
    default=config.COUNTERFACTUAL_CONTRASTIVE_SEARCH_CAP_DEFAULT,
    help="Max neighbors to search for a contrastive match.",
)
parser.add_argument(
    "--max-mask-patches",
    type=int,
    default=config.COUNTERFACTUAL_MASK_FLIP_MAX_PATCHES_DEFAULT,
    help="Max patches to apply in greedy mask-flip search.",
)
parser.add_argument(
    "--patch",
    type=int,
    default=config.OCCLUSION_PATCH_DEFAULT,
    help="Patch size for mask-flip search.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=config.OCCLUSION_STRIDE_DEFAULT,
    help="Stride for mask-flip search.",
)
parser.add_argument(
    "--fill",
    type=float,
    default=config.OCCLUSION_FILL_DEFAULT,
    help="Fill value for masked patches.",
)
parser.add_argument(
    "--embedding-layer",
    default=config.NEIGHBOR_EMBEDDING_LAYER,
    help="Layer name to extract embeddings from (for contrastive k-NN).",
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
    help="Which split to run counterfactuals on (default: val).",
)
parser.add_argument(
    "--indices",
    type=int,
    nargs="+",
    default=None,
    help="Specific dataset indices to run counterfactuals on (e.g. --indices 0 35 70).",
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

# Load dataset
_base = args.base_dir if args.base_dir is not None else config.BASE_DIR
val_dir = os.path.join(script_dir, _base, args.split)
val_raw = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    color_mode="rgb",
    shuffle=False,
    **config.DATA_PARAMS,
)
class_names = val_raw.class_names
val_rel_paths = [
    os.path.relpath(p, script_dir).replace("\\", "/") for p in val_raw.file_paths
]


def prepare_triple_input_eval(image, label):
    norm = tf.cast(image, tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(norm)
    return (norm, gray, gray), label


if args.indices:
    selected = sorted(args.indices)
    indices_set = set(selected)
    imgs, labs = [], []
    for i, (img, lbl) in enumerate(val_raw.unbatch()):
        if i in indices_set:
            imgs.append(img)
            labs.append(lbl)
        if i > max(indices_set):
            break
    val_raw_sub = tf.data.Dataset.from_tensor_slices(
        (tf.stack(imgs), tf.stack(labs))
    ).batch(32)
    val_rel_paths = [val_rel_paths[i] for i in selected if i < len(val_rel_paths)]
    val_dataset = (
        val_raw_sub.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    args.samples = len(selected)
    print(f"Filtering to {len(selected)} specified indices.")
else:
    val_dataset = (
        val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

# Build contrastive bank if npz + manifest provided
contrastive_bank = None
if args.train_npz and args.train_manifest:
    print(f"Loading contrastive bank from: {args.train_npz}")
    npz = np.load(args.train_npz)
    contrastive_bank = (
        np.asarray(npz["embeddings"], dtype=np.float32),
        np.asarray(npz["y_true"], dtype=np.int32),
        pd.read_csv(args.train_manifest),
    )
    print(f"  Bank size: {contrastive_bank[0].shape[0]} training samples")
else:
    print("No contrastive bank provided — mask-flip only (skipping contrastive k-NN).")

# Output to same audit run folder as the model
model_name = os.path.splitext(os.path.basename(args.model))[0]
timestamp = model_name.replace("potato_leaf_model_", "").replace("potato_tube_model_", "")
attribution_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, timestamp, "attribution"
)
metrics_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, timestamp, "metrics"
)

print(f"Saving counterfactual outputs to: {attribution_dir}")
result = run_counterfactual_audit(
    full_model=model,
    val_dataset=val_dataset,
    val_rel_paths=val_rel_paths,
    class_names=class_names,
    metrics_dir=metrics_dir,
    attribution_dir=attribution_dir,
    name_tag=timestamp,
    run_id=timestamp,
    split_label=f"{args.split}_leaf",
    num_samples=args.samples,
    contrastive_bank=contrastive_bank,
    contrastive_search_cap=args.search_cap,
    contrastive_exclude="pred",
    patch_size=args.patch,
    stride=args.stride,
    fill_value=args.fill,
    max_mask_patches=args.max_mask_patches,
    embedding_layer=args.embedding_layer,
)
print("Done.")
if result.get("contrastive_table_csv"):
    print(f"  Contrastive table: {result['contrastive_table_csv']}")
if result.get("mask_flip_summary_csv"):
    print(f"  Mask-flip summary: {result['mask_flip_summary_csv']}")
