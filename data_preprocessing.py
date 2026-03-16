import os
from typing import Iterable, List, Tuple

from tqdm import tqdm
import tensorflow as tf

import config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_SIZE = (224, 224)


def collect_image_paths(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    image_files: List[Tuple[str, str]] = []

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == ".":
            continue

        destination_folder = os.path.join(output_dir, relative_path)
        os.makedirs(destination_folder, exist_ok=True)

        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                full_input_path = os.path.join(root, file)
                full_output_path = os.path.join(destination_folder, file)
                image_files.append((full_input_path, full_output_path))

    return image_files


def resize_and_save_images(image_files: Iterable[Tuple[str, str]]) -> None:
    for input_path, output_path in tqdm(image_files, desc="Resizing"):
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


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, config.BASE_DIR)

    print(f"Scanning input images in '{input_dir}'")
    image_files = collect_image_paths(input_dir, output_dir)

    print("TensorFlow: resizing images.")
    resize_and_save_images(image_files)


if __name__ == "__main__":
    main()