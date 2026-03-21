import argparse
import datetime
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from grad_cam_visualizer import visualize_gradcam_batch
from nasa_power_client import fetch_nasa_power_features
from prediction_utils import batch_preds_to_probs, probability_column_names

# Env bin labels for interpretability (how the model behaves under different conditions)
ENV_BIN_LABELS = {
    "light": ["low_light", "mid_light", "high_light"],
    "temp": ["low_temp", "mid_temp", "high_temp"],
    "humidity": ["low_humidity", "mid_humidity", "high_humidity"],
}


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


def _find_class_index(class_names: List[str], fragment: str) -> Optional[int]:
    """Return index of first class whose name contains fragment (e.g. 'late_blight')."""
    fragment = fragment.lower()
    for i, name in enumerate(class_names):
        if fragment in name.lower():
            return i
    return None


def _print_blight_env_analysis(
    joined: pd.DataFrame,
    class_names: List[str],
    blight_idx: int,
) -> None:
    """
    Summarize how env (NASA POWER) relates to correct blight classification.
    Shows whether the model tends to get blight right when conditions match
    known blight-favorable weather (e.g. high humidity, mild temp).
    """
    blight_name = class_names[blight_idx]
    pred_blight = joined["y_pred"] == blight_idx
    true_blight = joined["y_true"] == blight_idx
    correct_blight = pred_blight & true_blight
    wrong_blight = pred_blight & ~true_blight

    print(f"\n--- Interpretability: when does the model classify '{blight_name}'? ---")
    if pred_blight.sum() == 0:
        print("  No predictions of", blight_name)
        return

    env_vars = ["T2M", "RH2M", "PRECTOT", "ALLSKY_SFC_SW_DWN"]
    for col in env_vars:
        if col not in joined.columns or joined[col].isna().all():
            continue
        correct_mean = joined.loc[correct_blight, col].mean()
        wrong_mean = joined.loc[wrong_blight, col].mean()
        all_pred_blight_mean = joined.loc[pred_blight, col].mean()
        print(f"  {col}: mean when pred={blight_name} correct = {correct_mean:.2f}, wrong = {wrong_mean:.2f}, all pred blight = {all_pred_blight_mean:.2f}")

    # Accuracy when the model predicts blight, by humidity bin (blight favors moist conditions)
    if "humidity_bin" in joined.columns and joined["humidity_bin"].notna().any():
        by_h = joined.loc[pred_blight].groupby("humidity_bin", observed=True).agg(
            precision=("correct", "mean"),
            count=("correct", "count"),
        )
        print(f"  When model predicts '{blight_name}', precision by humidity bin:")
        print(by_h.round(4).to_string())


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

    # Bin env vars for stratified analysis (how model performs under different conditions)
    def safe_qcut(series: pd.Series, q: int, labels: List[str]) -> pd.Series:
        out = pd.Series(index=series.index, dtype=object)
        valid = series.notna()
        if valid.sum() < q:
            out.loc[valid] = labels[0]
            return out
        out.loc[valid] = pd.qcut(series.loc[valid], q=q, labels=labels, duplicates="drop")
        return out

    joined["light_bin"] = safe_qcut(joined["ALLSKY_SFC_SW_DWN"], 3, ENV_BIN_LABELS["light"])
    joined["temp_bin"] = safe_qcut(joined["T2M"], 3, ENV_BIN_LABELS["temp"])
    joined["humidity_bin"] = safe_qcut(joined["RH2M"], 3, ENV_BIN_LABELS["humidity"])

    model = tf.keras.models.load_model(args.model_path)

    y_true_all, y_pred_all, prob_all = [], [], []
    for (rgb, gray, sobel), labels in ds:
        preds = model.predict_on_batch((rgb, gray, sobel))
        p_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
        probs = batch_preds_to_probs(p_np)
        y_true_all.extend(labels.numpy())
        y_pred_all.extend(np.argmax(probs, axis=1))
        prob_all.append(probs)

    joined["y_true"] = y_true_all
    joined["y_pred"] = y_pred_all
    probs_stack = np.vstack(prob_all)
    joined["prob_max"] = np.max(probs_stack, axis=1)

    prob_cols = probability_column_names(class_names)
    for j, col in enumerate(prob_cols):
        joined[col] = probs_stack[:, j]
    joined["pred_class"] = [class_names[i] for i in joined["y_pred"]]
    joined["true_class"] = [class_names[i] for i in joined["y_true"]]
    joined["correct"] = joined["y_true"] == joined["y_pred"]

    # --- Accuracy by env condition (does the model do better in certain climates?) ---
    print("\n--- Accuracy by environmental condition (NASA POWER) ---")
    for bin_col, label in [("light_bin", "Light"), ("temp_bin", "Temperature"), ("humidity_bin", "Humidity")]:
        valid = joined[bin_col].notna()
        if valid.sum() == 0:
            continue
        summary = joined.loc[valid].groupby(bin_col, observed=True).agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
        )
        print(f"\n{label} bin:")
        print(summary.round(4).to_string())

    # --- Blight-focused interpretability: when does the model "get" blight? ---
    blight_idx = _find_class_index(class_names, "late_blight")
    if blight_idx is not None:
        _print_blight_env_analysis(joined, class_names, blight_idx)

    # Save detailed results for further analysis and transparency
    out_dir = os.path.join(script_dir, config.OUTPUT_DIR, "fusion", "nasa_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "field_nasa_predictions.csv")
    cols_save = [
        "image_rel_path", "y_true", "y_pred", "true_class", "pred_class", "correct",
        "prob_max", "T2M", "RH2M", "PRECTOT", "ALLSKY_SFC_SW_DWN",
        "light_bin", "temp_bin", "humidity_bin",
    ] + prob_cols
    joined[[c for c in cols_save if c in joined.columns]].to_csv(out_csv, index=False)
    print(f"\nSaved detailed predictions and env: {out_csv}")

    visualize_gradcam_batch(
        model=model,
        validation_data=ds,
        save_dir=os.path.join(script_dir, config.OUTPUT_DIR, "fusion", "plots"),
        name_tag="field_nasa",
        class_names=class_names,
    )


if __name__ == "__main__":
    main()

