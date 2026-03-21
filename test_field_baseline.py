import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from grad_cam_visualizer import visualize_gradcam_batch
from prediction_utils import batch_preds_to_probs, probability_column_names


def prepare_triple_input(image, label):
    """Create (rgb, gray, sobel) triple to match the fusion model inputs."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    sobel_image = tf.image.sobel_edges(gray_image)
    sobel_image = tf.reduce_sum(tf.abs(sobel_image), axis=-1)
    return (norm_image, gray_image, sobel_image), label


def build_field_dataset(field_dir: str) -> tuple[tf.data.Dataset, list[str], list[str]]:
    """Build a tf.data pipeline for field images only; returns file_paths in dataset order."""
    field_raw = tf.keras.utils.image_dataset_from_directory(
        field_dir,
        color_mode="rgb",
        shuffle=False,
        **config.DATA_PARAMS,
    )
    class_names = field_raw.class_names
    file_paths = list(field_raw.file_paths)
    field_spud = field_raw.map(prepare_triple_input).prefetch(tf.data.AUTOTUNE)
    return field_spud, class_names, file_paths


def evaluate_on_field(model_path: str, domain: str = "leaf") -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    field_dir = os.path.join(script_dir, config.BASE_DIR, "field_classes")
    field_spud, class_names, file_paths = build_field_dataset(field_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model = tf.keras.models.load_model(model_path)

    loss, acc = model.evaluate(field_spud, verbose=1)
    print(f"[FIELD] Domain={domain} | loss={loss:.4f} | acc={acc:.4f}")

    y_true, y_pred = [], []
    prob_all: list[np.ndarray] = []
    for images, labels in field_spud:
        preds = model.predict_on_batch(images)
        p_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
        probs = batch_preds_to_probs(p_np)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(probs, axis=1))
        prob_all.append(probs)

    probs_stack = np.vstack(prob_all)
    prob_cols = probability_column_names(class_names)
    rel_paths = [
        os.path.relpath(p, field_dir).replace("\\", "/") for p in file_paths
    ]
    if len(rel_paths) != len(y_true):
        print(
            f"[WARN] file_paths ({len(rel_paths)}) != predictions ({len(y_true)}); "
            "CSV paths may be misaligned."
        )
    results = pd.DataFrame(
        {
            "image_rel_path": rel_paths[: len(y_true)],
            "y_true": y_true,
            "y_pred": y_pred,
            "true_class": [class_names[i] for i in y_true],
            "pred_class": [class_names[i] for i in y_pred],
            "correct": np.array(y_true) == np.array(y_pred),
            "prob_max": np.max(probs_stack, axis=1),
        }
    )
    for j, col in enumerate(prob_cols):
        results[col] = probs_stack[:, j]

    field_eval_dir = os.path.join(script_dir, config.OUTPUT_DIR, "fusion", "field_eval")
    os.makedirs(field_eval_dir, exist_ok=True)
    pred_csv = os.path.join(
        field_eval_dir, f"field_predictions_{domain}_{timestamp}.csv"
    )
    results.to_csv(pred_csv, index=False)
    print(f"[FIELD] Saved per-class probabilities: {pred_csv}")

    plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, "fusion", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    present_classes = np.unique(y_true)
    filtered_names = [class_names[i] for i in present_classes]

    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=filtered_names,
        output_dict=True,
    )
    df = pd.DataFrame(report).transpose().round(2)

    plt.figure(figsize=(8, 5))
    plt.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
    )
    plt.axis("off")
    plt.title(f"Field Results: {domain}")
    plt.savefig(
        os.path.join(plot_dir, f"report_field_{domain}_{timestamp}.png"),
        bbox_inches="tight",
    )
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (Field): {domain}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(
        os.path.join(plot_dir, f"cm_field_{domain}_{timestamp}.png"),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_perc,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix (Percentages, Field): {domain}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(
        os.path.join(plot_dir, f"cm_percent_field_{domain}_{timestamp}.png"),
        bbox_inches="tight",
    )
    plt.close()

    visualize_gradcam_batch(
        model=model,
        validation_data=field_spud,
        save_dir=plot_dir,
        name_tag=f"field_{domain}_{timestamp}",
        class_names=class_names,
    )


if __name__ == "__main__":
    # Example usage:
    # python test_field_baseline.py --model_path outputs/fusion/models/potato_leaf_model_YYYYMMDD_HHMMSS.keras --domain leaf
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate lab-trained model on field images.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved Keras model trained on lab data.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="leaf",
        choices=["leaf"],
        help="Domain label for reporting only.",
    )
    args = parser.parse_args()

    evaluate_on_field(model_path=args.model_path, domain=args.domain)

