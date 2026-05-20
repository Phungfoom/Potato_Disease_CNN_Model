import os
import random
import re
import shutil
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

import config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_SIZE = config.DATA_PARAMS["image_size"]
VAL_RATIO = 0.3  # Fraction of each (area, class) held out for validation; rest for training
SPLIT_SEED = config.DATA_PARAMS["seed"]
EXCLUDED_CLASSES_BY_AREA = {
    "central java, indonesia": {"bacteria", "virus_unknown"},
}


def collect_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Walk data/<Area>/<class>/ and collect image paths for training/validation.
    Non-image files are ignored. Some area/class pairs can be excluded from
    train/val and handled as holdout-only data.
    """
    image_files: List[Tuple[str, str]] = []

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue
        # Skip macOS zip artifact folders; they can contain fake "._*" files.
        if "__MACOSX" in relative_path.split(os.sep):
            continue

        destination_folder = os.path.join(output_dir, relative_path)
        os.makedirs(destination_folder, exist_ok=True)

        rel_parts = relative_path.split(os.sep)
        area_name = rel_parts[0] if len(rel_parts) >= 1 else ""
        class_name = rel_parts[1] if len(rel_parts) >= 2 else ""
        excluded_classes = EXCLUDED_CLASSES_BY_AREA.get(area_name.strip().lower(), set())
        if class_name.strip().lower() in excluded_classes:
            continue

        for file in files:
            lower = file.lower()
            # Skip macOS resource fork artifacts (often have .jpg extension but aren't images).
            if file.startswith("._"):
                continue
            if lower.endswith(IMAGE_EXTENSIONS):
                full_input_path = os.path.join(root, file)
                # Skip empty files early (common with metadata artifacts).
                try:
                    if os.path.getsize(full_input_path) == 0:
                        continue
                except OSError:
                    continue
                full_output_path = os.path.join(destination_folder, file)
                image_files.append((full_input_path, full_output_path))
    return image_files


def collect_holdout_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Collect excluded area/class image pairs into processed_data/holdout/
    so they can be used for optional out-of-distribution testing.
    """
    holdout_output_dir = os.path.join(output_dir, "holdout")
    image_files: List[Tuple[str, str]] = []

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue
        if "__MACOSX" in relative_path.split(os.sep):
            continue

        rel_parts = relative_path.split(os.sep)
        area_name = rel_parts[0] if len(rel_parts) >= 1 else ""
        class_name = rel_parts[1] if len(rel_parts) >= 2 else ""
        excluded_classes = EXCLUDED_CLASSES_BY_AREA.get(area_name.strip().lower(), set())
        if class_name.strip().lower() not in excluded_classes:
            continue

        destination_folder = os.path.join(holdout_output_dir, relative_path)
        os.makedirs(destination_folder, exist_ok=True)

        for file in files:
            lower = file.lower()
            if file.startswith("._"):
                continue
            if not lower.endswith(IMAGE_EXTENSIONS):
                continue
            full_input_path = os.path.join(root, file)
            try:
                if os.path.getsize(full_input_path) == 0:
                    continue
            except OSError:
                continue
            full_output_path = os.path.join(destination_folder, file)
            image_files.append((full_input_path, full_output_path))

    return image_files


def resize_and_save_images(image_files: Iterable[Tuple[str, str]]) -> None:
    total = len(image_files) if isinstance(image_files, list) else None
    for input_path, output_path in tqdm(image_files, desc="Resizing", total=total):
        try:
            img_bytes = tf.io.read_file(input_path)
            img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

            img_resized = tf.image.resize_with_pad(
                img,
                target_height=TARGET_SIZE[0],
                target_width=TARGET_SIZE[1],
                antialias=True,
            )

            img_uint8 = tf.cast(img_resized, tf.uint8)
            encoded_img = tf.io.encode_jpeg(img_uint8, quality=90)
            tf.io.write_file(output_path, encoded_img)

        except Exception as e:
            print(f"Skip corrupt file {input_path}: {e}")


def _area_slug(area: str) -> str:
    """Safe filename slug from area name (e.g. 'Central Java, Indonesia' -> 'Central_Java_Indonesia')."""
    s = re.sub(r"[^\w\s-]", "", area)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "unknown"


def assign_stratified_splits(
    rows: List[dict],
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> List[str]:
    """
    Assign 'train' or 'val' so each (area, class_name) contributes proportionally.
    That way train and val are both jumbled across areas and classes.
    """
    rng = random.Random(seed)
    by_group: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row["area"], row["class_name"])
        by_group[key].append(i)
    split = ["train"] * len(rows)
    for indices in by_group.values():
        n_val = max(0, min(len(indices) - 1, int(len(indices) * val_ratio)))
        if n_val > 0:
            rng.shuffle(indices)
            for idx in indices[:n_val]:
                split[idx] = "val"
    return split


def write_image_manifest(
    image_files: List[Tuple[str, str]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Write a manifest CSV (image_rel_path, area, class_name, split) so downstream
    scripts can report by region. Assigns stratified train/val split so each area
    and class is represented in both sets (jumbled for training and validation).
    """
    rows = []
    for _, out_path in image_files:
        rel = os.path.relpath(out_path, output_dir)
        rel = rel.replace("\\", "/")
        parts = rel.split("/")
        if len(parts) >= 2:
            area, class_name = parts[0], parts[1]
        else:
            area, class_name = "", ""
        rows.append({"image_rel_path": rel, "area": area, "class_name": class_name})

    splits = assign_stratified_splits(rows, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    for i, s in enumerate(splits):
        rows[i]["split"] = s

    df = pd.DataFrame(rows)
    manifest_path = os.path.join(output_dir, "image_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")
    return df


def build_train_val_dirs(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create processed_data/train/<class>/ and processed_data/val/<class>/ with
    copies of images assigned by the stratified split. Filenames include area
    slug so the same filename in different areas does not clash.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    for split_name, base in [("train", train_dir), ("val", val_dir)]:
        subset = manifest_df[manifest_df["split"] == split_name]
        for _, row in tqdm(
            subset.iterrows(),
            desc=f"Building {split_name}",
            total=len(subset),
        ):
            rel = row["image_rel_path"]
            class_name = row["class_name"]
            area = row["area"]
            src = os.path.join(output_dir, rel.replace("/", os.sep))
            if not os.path.isfile(src):
                continue
            slug = _area_slug(area)
            # Keep original extension; ensure unique name per area
            base_name = os.path.basename(rel)
            unique_name = f"{slug}_{base_name}"
            dest_dir = os.path.join(base, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            dst = os.path.join(dest_dir, unique_name)
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                print(f"Skip copy {src!r} -> {dst!r}: {e}")
    print(f"Train images: {train_dir}")
    print(f"Val images:   {val_dir}")


def _pil_resize_with_pad(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Resize maintaining aspect ratio then pad to (target_h, target_w) with black."""
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    out.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return out


SHARD_CHUNK_SIZE = 2000  # images per chunk file — keeps peak RAM under ~300 MB per chunk


def build_npz_shards_from_manifest(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """
    Build NPZ shards directly from the manifest — no JPEG copy step.

    Large classes are split into chunks of SHARD_CHUNK_SIZE images so that:
      - Peak RAM per write is ~300 MB (not 2+ GB for a full class at once)
      - Each chunk saves independently, so a Colab disconnect only loses the
        current chunk in progress, not the entire class

    Output files:
      {split}/{class}_c000.npz, {class}_c001.npz, ...

    Each chunk contains:
      rgb   — float32 array (N, H, W, 3), normalised to [0, 1]
      paths — str array (N,), rel path from output_dir
    """
    shard_root = os.path.join(output_dir, config.SHARDS_SUBDIR)

    for split in ("train", "val"):
        subset = manifest_df[manifest_df["split"] == split]
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
                        img = _pil_resize_with_pad(img, TARGET_SIZE[0], TARGET_SIZE[1])
                        rgb_list.append(np.array(img, dtype=np.float32) / 255.0)
                        path_list.append(row.image_rel_path)
                    except Exception as e:
                        print(f"Skip shard entry {img_path}: {e}")

                if not rgb_list:
                    continue

                np.savez(
                    chunk_path,
                    rgb=np.stack(rgb_list, axis=0),
                    paths=np.array(path_list),
                )
                print(f"  Chunk: {chunk_path} ({len(rgb_list)} images)")

    print(f"Shards written to: {shard_root}")


def build_npz_shards(output_dir: str) -> None:
    """
    Save per-class NPZ shards from the train/val directory trees.

    Saved under <output_dir>/shards/{split}/{class_name}.npz.
    Each file contains:
      rgb   — float32 array, shape (N, H, W, 3), normalised to [0, 1]
      paths — str array,     shape (N,), rel path from output_dir

    The shard filename matches the class directory name, so train_model.py
    can resolve label indices by enumerating class_names (alphabetical order,
    same as tf.keras.utils.image_dataset_from_directory).
    """
    shard_root = os.path.join(output_dir, config.SHARDS_SUBDIR)

    for split in ("train", "val"):
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        shard_split_dir = os.path.join(shard_root, split)
        os.makedirs(shard_split_dir, exist_ok=True)

        class_dirs = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )

        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            shard_path = os.path.join(shard_split_dir, f"{class_name}.npz")

            img_paths = sorted(
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith("._")
            )
            if not img_paths:
                continue

            rgb_list: List[np.ndarray] = []
            path_list: List[str] = []
            for img_path in tqdm(img_paths, desc=f"Sharding {split}/{class_name}", leave=False):
                try:
                    img_bytes = tf.io.read_file(img_path)
                    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
                    img = tf.image.resize_with_pad(
                        img, TARGET_SIZE[0], TARGET_SIZE[1], antialias=True
                    )
                    img = tf.cast(img, tf.float32) / 255.0
                    rgb_list.append(img.numpy())
                    path_list.append(
                        os.path.relpath(img_path, output_dir).replace("\\", "/")
                    )
                except Exception as e:
                    print(f"Skip shard entry {img_path}: {e}")

            if not rgb_list:
                continue

            chunk_size = getattr(config, "SHARDS_CHUNK_SIZE", 1000)
            if len(rgb_list) <= chunk_size:
                np.savez_compressed(
                    shard_path,
                    rgb=np.stack(rgb_list, axis=0),
                    paths=np.array(path_list),
                )
                print(f"  Shard: {shard_path} ({len(rgb_list)} images)")
            else:
                # Write chunks so no single file exceeds ~chunk_size × image RAM.
                for ci, start in enumerate(range(0, len(rgb_list), chunk_size)):
                    chunk_rgb   = rgb_list[start : start + chunk_size]
                    chunk_paths = path_list[start : start + chunk_size]
                    chunk_path  = os.path.join(
                        shard_split_dir, f"{class_name}_c{ci:03d}.npz"
                    )
                    np.savez_compressed(
                        chunk_path,
                        rgb=np.stack(chunk_rgb, axis=0),
                        paths=np.array(chunk_paths),
                    )
                n_chunks = -(-len(rgb_list) // chunk_size)  # ceiling division
                print(f"  Shard: {class_name} ({len(rgb_list)} images, {n_chunks} chunks)")

    print(f"Shards written to: {shard_root}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess potato disease images.")
    parser.add_argument(
        "--shards-only",
        action="store_true",
        help=(
            "Skip image resizing and manifest regeneration. "
            "Read the existing image_manifest.csv and build NPZ shards only. "
            "Use this when processed_data/ already has correctly resized images."
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, config.BASE_DIR)

    if args.shards_only:
        manifest_path = os.path.join(output_dir, "image_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise SystemExit(
                f"--shards-only requires an existing manifest at:\n  {manifest_path}\n"
                "Run without --shards-only first to generate it."
            )
        print(f"[shards-only] Reading existing manifest: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        build_npz_shards_from_manifest(manifest_df, output_dir)
        counts = manifest_df["split"].value_counts(dropna=False).to_dict()
        print(f"Split counts: {counts}")
        return

    print(f"Scanning input images in '{input_dir}'")
    image_files = collect_image_paths(input_dir, output_dir)
    print(f"Found {len(image_files)} images to process.")
    holdout_image_files = collect_holdout_image_paths(input_dir, output_dir)
    print(f"Found {len(holdout_image_files)} holdout images to process.")

    print("TensorFlow: resizing images.")
    resize_and_save_images(image_files)
    resize_and_save_images(holdout_image_files)

    if image_files:
        manifest_df = write_image_manifest(image_files, output_dir)
        build_npz_shards_from_manifest(manifest_df, output_dir)
        counts = manifest_df["split"].value_counts(dropna=False).to_dict()
        print(f"Split counts: {counts}")
    if holdout_image_files:
        print(f"Holdout images saved under: {os.path.join(output_dir, 'holdout')}")


if __name__ == "__main__":
    main()