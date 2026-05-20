"""Rerun k-NN neighbor lookup on a saved model without retraining.

Builds the training embedding bank and produces a val neighbor table + grid figures.

Usage:
    python rerun_neighbors.py --model outputs/fusion/models/potato_leaf_model_20260331_233302.keras
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from build_sobel_model import sobel_edge_layer
from neighbor_lookup import run_train_neighbor_pipeline, run_val_neighbor_audit, build_embedding_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to saved .keras model file.")
parser.add_argument(
    "--top-k",
    type=int,
    default=config.NEIGHBOR_TOP_K_DEFAULT,
    help="Number of nearest neighbors to retrieve.",
)
parser.add_argument(
    "--val-samples",
    type=int,
    default=config.NEIGHBOR_VAL_QUERY_SAMPLES_DEFAULT,
    help="Number of val images to use as queries.",
)
parser.add_argument(
    "--grid-figures",
    type=int,
    default=config.NEIGHBOR_GRID_FIGURES_DEFAULT,
    help="Number of grid figures to save.",
)
parser.add_argument(
    "--embedding-layer",
    default=config.NEIGHBOR_EMBEDDING_LAYER,
    help="Layer name to extract embeddings from.",
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
    help="Which split to use as queries (default: val).",
)
parser.add_argument(
    "--indices",
    type=int,
    nargs="+",
    default=None,
    help="Specific dataset indices to use as queries (e.g. --indices 0 35 70).",
)
parser.add_argument(
    "--train-npz",
    default=None,
    help="Path to existing train_embeddings_*.npz to skip rebuilding the embedding bank.",
)
parser.add_argument(
    "--train-manifest",
    default=None,
    help="Path to existing train_manifest_*.csv to skip rebuilding the embedding bank.",
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


def prepare_triple_input_eval(image, label):
    norm = tf.cast(image, tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(norm)
    return (norm, gray, gray), label


_base = args.base_dir if args.base_dir is not None else config.BASE_DIR
base_dir = os.path.join(script_dir, _base)

# Load query dataset (val or test)
val_dir = os.path.join(base_dir, args.split)
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
    args.val_samples = len(selected)
    print(f"Filtering to {len(selected)} specified indices.")
else:
    val_dataset = (
        val_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

# Load train dataset (unshuffled, for embedding bank)
train_dir = os.path.join(base_dir, "train")
train_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    color_mode="rgb",
    shuffle=False,
    **config.DATA_PARAMS,
)
train_rel_paths = [
    os.path.relpath(p, script_dir).replace("\\", "/") for p in train_raw.file_paths
]
train_dataset = (
    train_raw.map(prepare_triple_input_eval, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Output to same audit run folder as the model
model_name = os.path.splitext(os.path.basename(args.model))[0]
timestamp = model_name.replace("potato_leaf_model_", "").replace("potato_tube_model_", "")
neighbors_dir = os.path.join(
    script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, timestamp, "neighbors"
)
os.makedirs(neighbors_dir, exist_ok=True)

print(f"Saving neighbor outputs to: {neighbors_dir}")

if args.train_npz and args.train_manifest:
    # Reuse existing embedding bank — skip the 1.5-hour rebuild
    print(f"Loading existing train embeddings from: {args.train_npz}")
    _npz = np.load(args.train_npz)
    train_emb = np.asarray(_npz["embeddings"], dtype=np.float32)
    train_manifest_df = pd.read_csv(args.train_manifest)
    print(f"  Bank size: {train_emb.shape[0]} training samples")

    table_csv, _ = run_val_neighbor_audit(
        full_model=model,
        val_dataset=val_dataset,
        val_rel_paths=val_rel_paths,
        train_embeddings=train_emb,
        train_manifest_df=train_manifest_df,
        class_names=class_names,
        neighbors_dir=neighbors_dir,
        name_tag=timestamp,
        run_id=timestamp,
        split_label=args.split,
        top_k=args.top_k,
        max_queries=args.val_samples,
        max_grid_figures=args.grid_figures,
        embedding_layer=args.embedding_layer,
        project_root=script_dir,
    )
    result = {
        "train_embeddings_npz": args.train_npz,
        "train_manifest_csv": args.train_manifest,
        "val_neighbor_table_csv": table_csv or "",
    }
else:
    result = run_train_neighbor_pipeline(
        full_model=model,
        train_dataset_eval=train_dataset,
        train_rel_paths=train_rel_paths,
        val_dataset=val_dataset,
        val_rel_paths=val_rel_paths,
        class_names=class_names,
        neighbors_dir=neighbors_dir,
        name_tag=timestamp,
        run_id=timestamp,
        split_label_val=args.split,
        top_k=args.top_k,
        val_query_samples=args.val_samples,
        max_grid_figures=args.grid_figures,
        project_root=script_dir,
        embedding_layer=args.embedding_layer,
    )

print(f"Done.")
print(f"  Train embeddings: {result['train_embeddings_npz']}")
print(f"  Train manifest:   {result['train_manifest_csv']}")
print(f"  Val neighbor table: {result['val_neighbor_table_csv']}")
