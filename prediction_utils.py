"""Model output → probabilities and master prediction CSV layout (see standard_prediction_column_names)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import tensorflow as tf

# Append-only column contract; bump PREDICTION_SCHEMA_VERSION if you add columns.
PREDICTION_SCHEMA_VERSION = "phase1_v1"
HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT = 0.7

META_COLS = [
    "run_id",
    "split",
    "domain",
    "stage",
    "model_rel_path",
    "sample_index",
    "image_rel_path",
]
LABEL_COLS = [
    "y_true",
    "y_pred",
    "true_class",
    "pred_class",
    "correct",
    "prob_max",
]
PHASE1_CONFIDENCE_COLS = [
    "prob_top1",
    "prob_top2",
    "margin_top1_top2",
    "pred_entropy",
    "high_conf_wrong",
]


def vector_to_probs(pred_vector: np.ndarray) -> np.ndarray:
    """Return per-class probabilities (softmax if the vector looks like logits)."""
    v = np.asarray(pred_vector, dtype=np.float64).ravel()
    s = float(np.sum(v))
    m = float(np.max(v))
    if s > 0.99 and m <= 1.0 + 1e-5:
        return v
    return tf.nn.softmax(tf.constant(v, dtype=tf.float32)).numpy()


def batch_preds_to_probs(preds: np.ndarray) -> np.ndarray:
    """Apply :func:`vector_to_probs` to each row of a batch."""
    p = np.asarray(preds)
    out = np.empty_like(p, dtype=np.float64)
    for i in range(len(p)):
        out[i] = vector_to_probs(p[i])
    return out


def probability_column_names(class_names: list[str]) -> list[str]:
    """
    Stable CSV column names for each class, e.g. prob_late_blight, prob_healthy_leaf_images.
    """
    names: list[str] = []
    used: set[str] = set()
    for idx, cname in enumerate(class_names):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(cname).strip()).strip("_").lower()
        if not slug:
            slug = f"class_{idx}"
        col = f"prob_{slug}"
        base = col
        n = 2
        while col in used:
            col = f"{base}_{n}"
            n += 1
        used.add(col)
        names.append(col)
    return names


def standard_prediction_column_names(class_names: list[str]) -> list[str]:
    """Ordered columns for prediction exports."""
    return (
        META_COLS
        + LABEL_COLS
        + PHASE1_CONFIDENCE_COLS
        + probability_column_names(class_names)
    )


def _phase1_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    high_confidence_threshold: float,
) -> dict[str, np.ndarray]:
    """prob_top1, prob_top2, margin, entropy, high_conf_wrong."""
    p = np.asarray(probs, dtype=np.float64)
    n, c = p.shape
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    idx = np.arange(n, dtype=np.int64)
    prob_top1 = p[idx, yp]
    if c < 2:
        prob_top2 = np.zeros(n, dtype=np.float64)
    else:
        prob_top2 = np.sort(p, axis=1)[:, -2]
    margin = prob_top1 - prob_top2
    eps = 1e-12
    pred_entropy = -np.sum(p * np.log(p + eps), axis=1)
    wrong = yt != yp
    high_conf_wrong = (prob_top1 >= float(high_confidence_threshold)) & wrong
    return {
        "prob_top1": prob_top1,
        "prob_top2": prob_top2,
        "margin_top1_top2": margin,
        "pred_entropy": pred_entropy,
        "high_conf_wrong": high_conf_wrong.astype(np.bool_),
    }


def build_standard_predictions_dataframe(
    *,
    run_id: str,
    split: str,
    domain: str,
    stage: str,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    image_rel_paths: list[str] | None = None,
    model_rel_path: str = "",
    high_confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """One row per sample; column order matches standard_prediction_column_names."""
    thr = (
        float(high_confidence_threshold)
        if high_confidence_threshold is not None
        else HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT
    )
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()
    probs_arr = batch_preds_to_probs(np.asarray(probs))
    n = len(y_true_arr)
    if len(y_pred_arr) != n or probs_arr.shape[0] != n:
        raise ValueError("y_true, y_pred, and probs row counts must match")
    if probs_arr.shape[1] != len(class_names):
        raise ValueError("probs width must match len(class_names)")

    prob_cols = probability_column_names(class_names)
    p1 = _phase1_metrics(
        probs_arr, y_true_arr, y_pred_arr, high_confidence_threshold=thr
    )
    rows: dict[str, np.ndarray | list[str] | list[int] | list[bool]] = {
        "run_id": [run_id] * n,
        "split": [split] * n,
        "domain": [domain] * n,
        "stage": [stage] * n,
        "model_rel_path": [model_rel_path] * n,
        "sample_index": list(range(n)),
        "image_rel_path": [],
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "true_class": [class_names[i] for i in y_true_arr],
        "pred_class": [class_names[i] for i in y_pred_arr],
        "correct": (y_true_arr == y_pred_arr).astype(np.bool_),
        "prob_max": np.max(probs_arr, axis=1),
        **{k: p1[k] for k in PHASE1_CONFIDENCE_COLS},
    }
    if image_rel_paths is not None and len(image_rel_paths) == n:
        rows["image_rel_path"] = list(image_rel_paths)
    else:
        rows["image_rel_path"] = [""] * n

    df = pd.DataFrame(rows)
    for j, col in enumerate(prob_cols):
        df[col] = probs_arr[:, j]

    ordered = standard_prediction_column_names(class_names)
    return df[ordered]
