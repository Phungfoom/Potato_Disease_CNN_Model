import os
import random
import re
import shutil
from collections import defaultdict
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm
import tensorflow as tf

import config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
METADATA_EXTENSIONS = (".txt", ".md")
TARGET_SIZE = (224, 224)
VAL_RATIO = 0.3  # Fraction of each (area, class) held out for validation; rest for training
SPLIT_SEED = 123


def collect_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Walk data/<Area>/<class>/ and collect image paths; copy area-level metadata
    (.txt, .md) into processed_data so provenance (dataset URL, paper, timeframe)
    stays with the processed images for transparent classification.
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
            elif lower.endswith(METADATA_EXTENSIONS):
                src = os.path.join(root, file)
                dst = os.path.join(destination_folder, file)
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"Skip metadata copy {src!r}: {e}")

    return image_files


def resize_and_save_images(image_files: Iterable[Tuple[str, str]]) -> None:
    # tqdm shows a live X/Y counter when total is known (List/Sequence).
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
    for _input, out_path in image_files:
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


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, config.BASE_DIR)

    print(f"Scanning input images in '{input_dir}'")
    image_files = collect_image_paths(input_dir, output_dir)
    print(f"Found {len(image_files)} images to process.")

    print("TensorFlow: resizing images.")
    resize_and_save_images(image_files)

    if image_files:
        manifest_df = write_image_manifest(image_files, output_dir)
        build_train_val_dirs(manifest_df, output_dir)
        counts = manifest_df["split"].value_counts(dropna=False).to_dict()
        print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()