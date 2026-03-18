import argparse
import datetime
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from grad_cam_visualizer import visualize_gradcam_batch
from nasa_power_client import fetch_nasa_power_features


def prepare_triple_input_with_path(image, label, path):
    """Return (rgb, gray, sobel), label and path for later NASA join."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    sobel_image = tf.image.sobel_edges(gray_image)
    sobel_image = tf.reduce_sum(tf.abs(sobel_image), axis=-1)
    return (norm_image, gray_image, sobel_image), label, path


def load_field_with_paths(field_dir: str):
    """Load field images and also keep relative file paths."""
    ds = tf.keras.utils.image_dataset_from_directory(
        field_dir,
        color_mode="rgb",
        shuffle=False,
        **config.DATA_PARAMS,
    )
    class_names = ds.class_names
    paths = ds.file_paths
    ds = ds.map(
        lambda img, label: (*prepare_triple_input_with_path(img, label, ""),),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds, class_names, paths


def build_nasa_feature_matrix(
    metadata_df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    """Convert a metadata dataframe with NASA columns into a dense feature matrix."""
    return metadata_df[feature_columns].to_numpy(dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate field images with NASA POWER context (analysis only)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the lab-trained Keras model.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="CSV with columns: image_rel_path, lat, lon, date(YYYY-MM-DD).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    field_dir = os.path.join(script_dir, config.BASE_DIR, "field_classes")

    ds, class_names, file_paths = load_field_with_paths(field_dir)

    metadata = pd.read_csv(args.metadata_csv)
    rel_paths = [
        os.path.relpath(path, start=field_dir).replace("\\", "/") for path in file_paths
    ]
    metadata = metadata.set_index("image_rel_path").reindex(rel_paths)

    feature_records: List[Dict] = []
    dates = []

    for rel, row in metadata.iterrows():
        lat, lon, date_str = row["lat"], row["lon"], row["date"]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        dates.append(date)
        features = fetch_nasa_power_features(lat=lat, lon=lon, date=date)
        if features is None:
            features = {}
        features["image_rel_path"] = rel
        feature_records.append(features)

    nasa_df = pd.DataFrame(feature_records).set_index("image_rel_path")
    joined = metadata.join(nasa_df, how="left")

    env_cols = [c for c in joined.columns if c in ("T2M", "RH2M", "PRECTOT", "ALLSKY_SFC_SW_DWN")]

    bins = pd.qcut(joined["ALLSKY_SFC_SW_DWN"], q=3, labels=["low_light", "mid_light", "high_light"])

    model = tf.keras.models.load_model(args.model_path)

    y_true_all, y_pred_all = [], []
    for (rgb, gray, sobel), labels in ds:
        preds = model.predict_on_batch((rgb, gray, sobel))
        y_true_all.extend(labels.numpy())
        y_pred_all.extend(np.argmax(preds, axis=1))

    joined["y_true"] = y_true_all
    joined["y_pred"] = y_pred_all
    joined["light_bin"] = bins

    summary = joined.groupby("light_bin").apply(
        lambda g: (g["y_true"] == g["y_pred"]).mean()
    )
    print("Accuracy by light condition:")
    print(summary)

    visualize_gradcam_batch(
        model=model,
        validation_data=ds,
        save_dir=os.path.join(script_dir, config.OUTPUT_DIR, "fusion", "plots"),
        name_tag="field_nasa",
        class_names=class_names,
    )


if __name__ == "__main__":
    main()

