"""Integrated Gradients on RGB: path from baseline → input; each step uses (rgb, gray, gray) like training."""

from __future__ import annotations

import os
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config
from grad_cam_visualizer import (
    _class_score_for_gradcam,
    _resolve_layer_output_tensor,
    compute_gradcam,
    overlay_heatmap,
)
from occlusion_attribution import _triple_batch_from_rgb
from prediction_utils import batch_preds_to_probs

BaselineMode = Literal["black", "gray", "mean_color"]
IGTarget = Literal["predicted", "true"]


def _baseline_rgb(rgb: tf.Tensor, mode: str) -> tf.Tensor:
    """rgb: (1, H, W, 3) float32 in [0, 1]."""
    if mode == "black":
        return tf.zeros_like(rgb)
    if mode == "gray":
        return tf.fill(tf.shape(rgb), 0.5)
    if mode == "mean_color":
        mu = tf.reduce_mean(rgb, axis=[1, 2], keepdims=True)
        return tf.ones_like(rgb) * mu
    raise ValueError(f"Unknown baseline: {mode!r}")


def compute_integrated_gradients_rgb(
    model: tf.keras.Model,
    rgb_hw3: np.ndarray,
    *,
    class_index: int,
    baseline: BaselineMode = "black",
    m_steps: int = config.IG_M_STEPS_DEFAULT,
) -> np.ndarray:
    """IG attributions (1,H,W,3); |sum over channels| used as saliency."""
    rgb = tf.constant(np.asarray(rgb_hw3, dtype=np.float32))
    if rgb.ndim == 3:
        rgb = rgb[tf.newaxis, ...]
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    baseline_rgb = _baseline_rgb(rgb, baseline)
    diff = rgb - baseline_rgb
    accum = tf.zeros_like(rgb)

    for i in tqdm(range(m_steps), desc="IG steps", leave=False, unit="step"):
        alpha = tf.constant((float(i) + 0.5) / float(m_steps), dtype=tf.float32)
        rgb_i = baseline_rgb + alpha * diff
        gray_i = tf.image.rgb_to_grayscale(rgb_i)
        with tf.GradientTape() as tape:
            tape.watch(rgb_i)
            preds = model([rgb_i, gray_i, gray_i], training=False)
            score = _class_score_for_gradcam(preds, class_index)
        g = tape.gradient(score, rgb_i)
        if g is not None:
            accum += g

    avg_g = accum / tf.cast(m_steps, tf.float32)
    ig = (rgb - baseline_rgb) * avg_g
    return ig.numpy()


def _ig_magnitude_heatmap(ig_batch: np.ndarray) -> np.ndarray:
    """Per-pixel L1 over RGB, normalized to [0,1]."""
    s = np.abs(ig_batch[0]).sum(axis=-1)
    m = float(np.max(s))
    if m > 1e-8:
        s = s / m
    return s.astype(np.float32)


def save_ig_figure(
    rgb_hw3: np.ndarray,
    ig_batch: np.ndarray,
    save_path: str,
    *,
    title: str,
) -> None:
    rgb = np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0)
    heat = _ig_magnitude_heatmap(ig_batch)
    ov = overlay_heatmap(heat, rgb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Integrated Gradients (|attrib| summed)")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _parent = os.path.dirname(os.path.abspath(save_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ig_vs_gradcam_figure(
    model: tf.keras.Model,
    rgb_hw3: np.ndarray,
    ig_batch: np.ndarray,
    *,
    class_index: int,
    save_path: str,
    title: str,
) -> None:
    """Three-panel figure: input, deep RGB Grad-CAM, IG."""
    rgb = np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0)
    rgb_t = tf.constant(rgb[np.newaxis, ...], dtype=tf.float32)
    gray_t = tf.image.rgb_to_grayscale(rgb_t)
    inputs = (rgb_t, gray_t, gray_t)
    deep = _resolve_layer_output_tensor(model, config.GRADCAM_RGB_DEEP_LAYER)
    heat_g = compute_gradcam(
        model, inputs, "RGB", class_index=class_index, target_conv_tensor=deep
    )
    viz_g = overlay_heatmap(heat_g, rgb)
    heat_i = _ig_magnitude_heatmap(ig_batch)
    viz_i = overlay_heatmap(heat_i, rgb)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(viz_g, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Grad-CAM ({config.GRADCAM_RGB_DEEP_LAYER})")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(viz_i, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Integrated Gradients")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _parent = os.path.dirname(os.path.abspath(save_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_integrated_gradients_batch(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    *,
    class_names: list[str],
    save_dir: str,
    name_tag: str,
    run_id: str,
    image_rel_paths: list[str] | None,
    num_samples: int = config.IG_NUM_SAMPLES_DEFAULT,
    m_steps: int = config.IG_M_STEPS_DEFAULT,
    baseline: BaselineMode = "black",
    ig_target: IGTarget = "predicted",
    csv_path: str | None = None,
    compare_gradcam: bool = True,
) -> str | None:
    """First val batch; writes PNGs and optional summary CSV path.

    Note: this repo standardizes Phase 4 IG to a black baseline for consistency.
    """
    os.makedirs(save_dir, exist_ok=True)
    baseline = "black"
    rows: list[dict] = []
    csv_out = csv_path
    taken = 0

    for (rgb_b, _g, _), labels in val_dataset.take(1):
        bs = int(rgb_b.shape[0])
        n = min(num_samples, bs)
        for i in range(n):
            rgb = rgb_b[i].numpy()
            true_idx = int(labels[i].numpy())
            pv = model.predict(_triple_batch_from_rgb(rgb[np.newaxis, ...]), verbose=0)
            probs = batch_preds_to_probs(pv)[0]
            pred_idx = int(np.argmax(probs))
            explain_idx = true_idx if ig_target == "true" else pred_idx

            ig = compute_integrated_gradients_rgb(
                model,
                rgb,
                class_index=explain_idx,
                baseline=baseline,
                m_steps=m_steps,
            )
            mean_abs = float(np.mean(np.abs(ig)))

            rel_img = ""
            if image_rel_paths is not None and taken < len(image_rel_paths):
                rel_img = image_rel_paths[taken]

            tgt = class_names[explain_idx]
            ttl = (
                f"{name_tag} | IG target={ig_target} ({tgt}) | "
                f"true={class_names[true_idx]} pred={class_names[pred_idx]} | "
                f"baseline={baseline}"
            )
            fname = f"ig_{name_tag}_sample{taken}.png"
            fpath = os.path.join(save_dir, fname)
            save_ig_figure(rgb, ig, fpath, title=ttl)

            cmp_name = f"ig_vs_gradcam_{name_tag}_sample{taken}.png"
            cmp_path = os.path.join(save_dir, cmp_name)
            if compare_gradcam:
                save_ig_vs_gradcam_figure(
                    model,
                    rgb,
                    ig,
                    class_index=explain_idx,
                    save_path=cmp_path,
                    title=ttl,
                )

            rows.append(
                {
                    "run_id": run_id,
                    "sample_index": taken,
                    "image_rel_path": rel_img,
                    "ig_target": ig_target,
                    "target_class_index": explain_idx,
                    "target_class_name": tgt,
                    "y_true": true_idx,
                    "y_pred": pred_idx,
                    "ig_baseline": baseline,
                    "ig_m_steps": m_steps,
                    "mean_abs_ig": mean_abs,
                    "figure_filename": fname,
                    "compare_filename": cmp_name if compare_gradcam else "",
                }
            )
            taken += 1
        break

    if csv_out and rows:
        pd.DataFrame(rows).to_csv(csv_out, index=False)

    return csv_out if rows else None
