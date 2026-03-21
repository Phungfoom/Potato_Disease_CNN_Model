"""Helpers for turning model outputs into per-class probabilities and safe CSV column names."""

from __future__ import annotations

import re

import numpy as np
import tensorflow as tf


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
