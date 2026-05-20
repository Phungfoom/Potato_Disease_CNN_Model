"""Data and shortcut audits: quantify background/border reliance using Grad-CAM heatmaps.

This is a lightweight diagnostic, not a proof. It estimates whether Grad-CAM energy
concentrates near image borders (tray/edge/letterbox shortcuts) vs the interior.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from grad_cam_visualizer import _resolve_layer_output_tensor, compute_gradcam
from prediction_utils import batch_preds_to_probs


def _border_mask(h: int, w: int, border_frac: float) -> np.ndarray:
    """Boolean mask where True indicates 'border' pixels."""
    bf = float(border_frac)
    bf = max(0.0, min(0.49, bf))
    top = int(round(bf * h))
    left = int(round(bf * w))
    m = np.zeros((h, w), dtype=bool)
    if top > 0:
        m[:top, :] = True
        m[-top:, :] = True
    if left > 0:
        m[:, :left] = True
        m[:, -left:] = True
    return m


def _heatmap_border_stats(heatmap_hw: np.ndarray, border_frac: float) -> dict[str, float]:
    hm = np.asarray(heatmap_hw, dtype=np.float64)
    if hm.ndim != 2 or hm.size == 0:
        return {
            "heatmap_mean": float("nan"),
            "border_mean": float("nan"),
            "center_mean": float("nan"),
            "border_share": float("nan"),
            "border_to_center_ratio": float("nan"),
        }
    h, w = hm.shape
    bm = _border_mask(h, w, border_frac)
    border_vals = hm[bm]
    center_vals = hm[~bm]
    heat_mean = float(np.mean(hm))
    border_mean = float(np.mean(border_vals)) if border_vals.size else float("nan")
    center_mean = float(np.mean(center_vals)) if center_vals.size else float("nan")
    border_sum = float(np.sum(border_vals)) if border_vals.size else 0.0
    total_sum = float(np.sum(hm))
    border_share = float(border_sum / total_sum) if total_sum > 1e-12 else float("nan")
    ratio = (
        float(border_mean / center_mean)
        if center_vals.size and abs(center_mean) > 1e-12
        else float("nan")
    )
    return {
        "heatmap_mean": heat_mean,
        "border_mean": border_mean,
        "center_mean": center_mean,
        "border_share": border_share,
        "border_to_center_ratio": ratio,
    }


def run_shortcut_background_audit(
    *,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    rel_paths: list[str] | None,
    class_names: list[str],
    run_id: str,
    split_label: str,
    save_csv_path: str,
    num_samples: int = 12,
    border_frac: float = config.SHORTCUT_BORDER_FRAC_DEFAULT,
    gradcam_target: str = "predicted",
    gradcam_layer: str = config.GRADCAM_RGB_DEEP_LAYER,
) -> str | None:
    """
    Writes per-sample Grad-CAM border-focus stats to CSV (metrics/).
    Uses RGB deep layer Grad-CAM, because this is where tray/background shortcuts usually show up.
    """
    parent = os.path.dirname(os.path.abspath(save_csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows: list[dict] = []
    taken = 0

    deep_tensor = _resolve_layer_output_tensor(model, gradcam_layer)

    for (rgb_b, gray_b, sobel_b), labels in dataset:
        bs = int(rgb_b.shape[0])
        for i in range(bs):
            if taken >= int(num_samples):
                break
            rgb = rgb_b[i : i + 1]
            gray = gray_b[i : i + 1]
            sobel = sobel_b[i : i + 1]
            inputs = (rgb, gray, sobel)

            true_idx = int(labels[i].numpy())
            pv = model.predict(inputs, verbose=0)
            probs = batch_preds_to_probs(pv)[0]
            pred_idx = int(np.argmax(probs))

            if gradcam_target == "true":
                explain_idx = true_idx
            else:
                explain_idx = None  # predicted class inside compute_gradcam

            heat = compute_gradcam(
                model,
                inputs,
                "RGB",
                class_index=explain_idx,
                target_conv_tensor=deep_tensor,
            )
            stats = _heatmap_border_stats(heat, border_frac)

            rel = ""
            if rel_paths is not None and taken < len(rel_paths):
                rel = rel_paths[taken]

            rows.append(
                {
                    "run_id": run_id,
                    "split": split_label,
                    "sample_index": taken,
                    "image_rel_path": rel,
                    "y_true": true_idx,
                    "y_pred": pred_idx,
                    "true_class": class_names[true_idx],
                    "pred_class": class_names[pred_idx],
                    "prob_top1": float(np.max(probs)),
                    "gradcam_target": gradcam_target,
                    "gradcam_layer": gradcam_layer,
                    "border_frac": float(border_frac),
                    **stats,
                }
            )
            taken += 1
        if taken >= int(num_samples):
            break

    if not rows:
        return None
    pd.DataFrame(rows).to_csv(save_csv_path, index=False)
    return save_csv_path

