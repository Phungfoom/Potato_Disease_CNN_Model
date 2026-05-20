"""3-way stratified split preprocessing: train / val / test.

Outputs to processed_data_v2/ — completely separate from processed_data/.
All existing data, models, and audit outputs are untouched.

Split ratios: 60% train / 20% val / 20% test
- val is used during training for early stopping and LR schedule
- test is held out completely until final evaluation

Usage:
    python data_preprocessing_split3.py             # full pipeline
    python data_preprocessing_split3.py --shards-only  # build NPZ shards from existing manifest
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_SIZE = config.DATA_PARAMS["image_size"]
SPLIT_SEED = config.DATA_PARAMS["seed"]
SHARD_CHUNK_SIZE = 2000  # images per chunk — keeps peak RAM under ~300 MB

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

INPUT_DIR  = "data"
OUTPUT_DIR = "processed_data_v2"

EXCLUDED_CLASSES_BY_AREA = {
    "central java, indonesia": {"bacteria", "virus_unknown"},
}


def _pil_resize_with_pad(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    out.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return out


def collect_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """Walk data/<Area>/<class>/ skipping excluded classes (those go to holdout)."""
    image_files: List[Tuple[str, str]] = []
    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue
        if "__MACOSX" in relative_path.split(os.sep):
            continue
        rel_parts = relative_path.split(os.sep)
        area_name  = rel_parts[0] if len(rel_parts) >= 1 else ""
        class_name = rel_parts[1] if len(rel_parts) >= 2 else ""
        excluded = EXCLUDED_CLASSES_BY_AREA.get(area_name.strip().lower(), set())
        if class_name.strip().lower() in excluded:
            continue
        destination_folder = os.path.join(output_dir, relative_path)
        os.makedirs(destination_folder, exist_ok=True)
        for file in files:
            if file.startswith("._"):
                continue
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            full_input = os.path.join(root, file)
            try:
                if os.path.getsize(full_input) == 0:
                    continue
            except OSError:
                continue
            full_output = os.path.join(destination_folder, file)
            image_files.append((full_input, full_output))
    return image_files


def collect_holdout_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """Collect excluded area/class pairs into processed_data_v2/holdout/ for OOD testing."""
    holdout_output_dir = os.path.join(output_dir, "holdout")
    image_files: List[Tuple[str, str]] = []

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue
        if "__MACOSX" in relative_path.split(os.sep):
            continue
        rel_parts = relative_path.split(os.sep)
        area_name  = rel_parts[0] if len(rel_parts) >= 1 else ""
        class_name = rel_parts[1] if len(rel_parts) >= 2 else ""
        excluded = EXCLUDED_CLASSES_BY_AREA.get(area_name.strip().lower(), set())
        if class_name.strip().lower() not in excluded:
            continue
        destination_folder = os.path.join(holdout_output_dir, relative_path)
        os.makedirs(destination_folder, exist_ok=True)
        for file in files:
            if file.startswith("._"):
                continue
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            full_input = os.path.join(root, file)
            try:
                if os.path.getsize(full_input) == 0:
                    continue
            except OSError:
                continue
            full_output = os.path.join(destination_folder, file)
            image_files.append((full_input, full_output))

    return image_files


def resize_and_save_images(image_files: List[Tuple[str, str]]) -> None:
    h, w = TARGET_SIZE
    for input_path, output_path in tqdm(image_files, desc="Resizing"):
        if os.path.isfile(output_path):
            continue
        if not os.path.isfile(input_path):
            continue
        try:
            img = Image.open(input_path).convert("RGB")
            img = _pil_resize_with_pad(img, h, w)
            img.save(output_path, "JPEG", quality=90)
        except Exception as e:
            print(f"Skip corrupt file {input_path}: {e}")


def _area_slug(area: str) -> str:
    s = re.sub(r"[^\w\s-]", "", area)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "unknown"


def assign_stratified_splits_3way(
    rows: List[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> List[str]:
    """Assign train/val/test so each (area, class_name) contributes proportionally."""
    rng = random.Random(seed)
    by_group: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row["area"], row["class_name"])
        by_group[key].append(i)

    split = ["train"] * len(rows)
    for indices in by_group.values():
        rng.shuffle(indices)
        n = len(indices)
        n_val  = max(0, min(n - 2, int(n * val_ratio)))
        n_test = max(0, min(n - n_val - 1, int(n * (1.0 - train_ratio - val_ratio))))
        for idx in indices[:n_val]:
            split[idx] = "val"
        for idx in indices[n_val : n_val + n_test]:
            split[idx] = "test"
    return split


def write_image_manifest(
    image_files: List[Tuple[str, str]],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    for _input, out_path in image_files:
        rel = os.path.relpath(out_path, output_dir).replace("\\", "/")
        parts = rel.split("/")
        area       = parts[0] if len(parts) >= 1 else ""
        class_name = parts[1] if len(parts) >= 2 else ""
        rows.append({"image_rel_path": rel, "area": area, "class_name": class_name})

    splits = assign_stratified_splits_3way(rows)
    for i, s in enumerate(splits):
        rows[i]["split"] = s

    df = pd.DataFrame(rows)
    manifest_path = os.path.join(output_dir, "image_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(df['split'] == s).sum()} images")
    return df


def build_split_dirs(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """Create train/ val/ test/ class subfolders with symlink-free copies."""
    for split_name in ["train", "val", "test"]:
        base = os.path.join(output_dir, split_name)
        subset = manifest_df[manifest_df["split"] == split_name]
        for _, row in tqdm(subset.iterrows(), desc=f"Building {split_name}/", total=len(subset)):
            rel        = row["image_rel_path"]
            class_name = row["class_name"]
            area       = row["area"]
            src = os.path.join(output_dir, rel.replace("/", os.sep))
            if not os.path.isfile(src):
                continue
            slug        = _area_slug(area)
            base_name   = os.path.basename(rel)
            unique_name = f"{slug}_{base_name}"
            dest_dir    = os.path.join(base, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            dst = os.path.join(dest_dir, unique_name)
            if not os.path.isfile(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"Skip {src!r}: {e}")


def build_npz_shards_from_manifest(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """Build chunk-based NPZ shards for all three splits."""
    shard_root = os.path.join(output_dir, config.SHARDS_SUBDIR)
    h, w = TARGET_SIZE

    for split in ("train", "val", "test"):
        subset = manifest_df[manifest_df["split"] == split]
        if subset.empty:
            continue
        shard_split_dir = os.path.join(shard_root, split)
        os.makedirs(shard_split_dir, exist_ok=True)

        for class_name, group in subset.groupby("class_name"):
            rows = list(group.itertuples(index=False))
            n_chunks = max(1, (len(rows) + SHARD_CHUNK_SIZE - 1) // SHARD_CHUNK_SIZE)

            for chunk_idx in range(n_chunks):
                chunk_path = os.path.join(
                    shard_split_dir, f"{class_name}_c{chunk_idx:03d}.npz"
                )
                if os.path.isfile(chunk_path):
                    print(f"  Skip (exists): {chunk_path}")
                    continue

                chunk_rows = rows[chunk_idx * SHARD_CHUNK_SIZE : (chunk_idx + 1) * SHARD_CHUNK_SIZE]
                rgb_list: List[np.ndarray] = []
                path_list: List[str] = []

                for row in tqdm(
                    chunk_rows,
                    desc=f"Sharding {split}/{class_name} chunk {chunk_idx}/{n_chunks-1}",
                    total=len(chunk_rows),
                    leave=True,
                ):
                    img_path = os.path.join(
                        output_dir, row.image_rel_path.replace("/", os.sep)
                    )
                    if not os.path.isfile(img_path):
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = _pil_resize_with_pad(img, h, w)
                        rgb_list.append(np.array(img, dtype=np.float32) / 255.0)
                        path_list.append(row.image_rel_path)
                    except Exception as e:
                        print(f"  Skip {img_path}: {e}")

                if not rgb_list:
                    continue

                tmp_path = chunk_path.replace(".npz", "_tmp.npz")
                np.savez(tmp_path, rgb=np.stack(rgb_list, axis=0), paths=np.array(path_list))
                shutil.move(tmp_path, chunk_path)
                print(f"  Chunk: {chunk_path} ({len(rgb_list)} images)")

    print(f"Shards written to: {shard_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess potato images with 3-way split.")
    parser.add_argument(
        "--shards-only",
        action="store_true",
        help=(
            "Skip image resizing and manifest regeneration. "
            "Read existing image_manifest.csv and build NPZ shards only."
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(script_dir, INPUT_DIR)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    if args.shards_only:
        manifest_path = os.path.join(output_dir, "image_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise SystemExit(
                f"--shards-only requires an existing manifest at:\n  {manifest_path}\n"
                "Run without --shards-only first to generate it."
            )
        print(f"[shards-only] Reading manifest: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        build_npz_shards_from_manifest(manifest_df, output_dir)
        for s in ["train", "val", "test"]:
            print(f"  {s}: {(manifest_df['split'] == s).sum()} images")
        raise SystemExit(0)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split:  {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} train/val/test")

    print("\nCollecting images...")
    image_files = collect_image_paths(input_dir, output_dir)
    print(f"Found {len(image_files)} images.")
    holdout_files = collect_holdout_image_paths(input_dir, output_dir)
    print(f"Found {len(holdout_files)} holdout images.")

    print("\nResizing images...")
    resize_and_save_images(image_files)
    resize_and_save_images(holdout_files)

    print("\nAssigning splits and writing manifest...")
    manifest_df = write_image_manifest(image_files, output_dir)

    print("\nCopying into train/ val/ test/ folders...")
    build_split_dirs(manifest_df, output_dir)

    print("\nBuilding NPZ shards...")
    build_npz_shards_from_manifest(manifest_df, output_dir)

    if holdout_files:
        print(f"Holdout images saved under: {os.path.join(output_dir, 'holdout')}")
    print("\nAll done. processed_data_v2/ is ready.")
