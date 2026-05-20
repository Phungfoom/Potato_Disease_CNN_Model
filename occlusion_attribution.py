"""Sliding patch on RGB; heatmap = drop in target-class prob when patch is masked (vs baseline)."""

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
from prediction_utils import batch_preds_to_probs

OcclusionTarget = Literal["predicted", "true"]


def _triple_batch_from_rgb(rgb_batch01: np.ndarray) -> list[np.ndarray]:
    """(B,H,W,3) in [0,1] → [rgb, gray, gray] numpy batch."""
    t = tf.constant(rgb_batch01, dtype=tf.float32)
    gray = tf.image.rgb_to_grayscale(t)
    return [t.numpy(), gray.numpy(), gray.numpy()]


def _apply_square_patch(
    rgb: np.ndarray, r0: int, c0: int, patch: int, fill: float
) -> np.ndarray:
    out = np.array(rgb, copy=True, dtype=np.float32)
    ph = min(patch, out.shape[0] - r0)
    pw = min(patch, out.shape[1] - c0)
    if ph > 0 and pw > 0:
        out[r0 : r0 + ph, c0 : c0 + pw, :] = fill
    return out


def compute_occlusion_map(
    model: tf.keras.Model,
    rgb_hw3: np.ndarray,
    *,
    class_index: int,
    patch_size: int = config.OCCLUSION_PATCH_DEFAULT,
    stride: int = config.OCCLUSION_STRIDE_DEFAULT,
    fill_value: float = config.OCCLUSION_FILL_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Grid cells: baseline P(class) − P(class) with patch masked; larger ⇒ masking hurts more."""
    rgb = np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0)
    h, w = rgb.shape[0], rgb.shape[1]
    ps = int(patch_size)
    st = int(stride)
    if ps < 1 or st < 1 or ps > h or ps > w:
        raise ValueError("patch_size / stride invalid for image size")

    base = model.predict(_triple_batch_from_rgb(rgb[np.newaxis, ...]), verbose=0)
    probs_b = batch_preds_to_probs(base)
    baseline_p = float(probs_b[0][class_index])

    rows = list(range(0, h - ps + 1, st))
    cols = list(range(0, w - ps + 1, st))
    gh, gw = len(rows), len(cols)
    grid = np.zeros((gh, gw), dtype=np.float32)

    total_cells = gh * gw
    pbar = tqdm(
        total=total_cells,
        desc="Occlusion patches",
        leave=False,
        unit="patch",
    )
    try:
        for gi, r0 in enumerate(rows):
            for gj, c0 in enumerate(cols):
                masked = _apply_square_patch(rgb, r0, c0, ps, fill_value)
                out = model.predict(
                    _triple_batch_from_rgb(masked[np.newaxis, ...]), verbose=0
                )
                p = float(batch_preds_to_probs(out)[0][class_index])
                grid[gi, gj] = baseline_p - p
                pbar.update(1)
    finally:
        pbar.close()

    return grid, probs_b[0], baseline_p, (gh, gw)


def _upsample_grid_to_image(grid: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """Nearest-neighbor upscale grid to H×W."""
    h, w = int(image_h), int(image_w)
    return cv2.resize(
        np.asarray(grid, dtype=np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def save_occlusion_figure(
    rgb_hw3: np.ndarray,
    importance_grid: np.ndarray,
    save_path: str,
    *,
    title: str,
    class_names: list[str],
    class_index: int,
) -> None:
    """RGB + ΔP heatmap (grid upsampled)."""
    rgb = np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0)
    up = _upsample_grid_to_image(importance_grid, rgb.shape[0], rgb.shape[1])
    umin = float(np.min(up)) if up.size else 0.0
    umax = float(np.max(up)) if up.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    if umin < -1e-8:
        lim = max(abs(umin), abs(umax), 1e-6)
        im = axes[1].imshow(up, cmap="RdBu_r", vmin=-lim, vmax=lim)
        cbar_label = "ΔP (target class, blue=mask helps, red=mask hurts)"
    else:
        vmax_vis = max(float(np.percentile(up, 99)), umax, 1e-6)
        im = axes[1].imshow(up, cmap="hot", vmin=0.0, vmax=vmax_vis)
        cbar_label = "ΔP (target class, higher=mask hurts more)"

    axes[1].imshow(rgb, alpha=0.35)
    axes[1].set_title(f"Occlusion ΔP ({class_names[class_index]})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label=cbar_label)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _parent = os.path.dirname(os.path.abspath(save_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_occlusion_val_batch(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    *,
    class_names: list[str],
    save_dir: str,
    name_tag: str,
    run_id: str,
    image_rel_paths: list[str] | None,
    num_samples: int = config.OCCLUSION_NUM_SAMPLES_DEFAULT,
    patch_size: int = config.OCCLUSION_PATCH_DEFAULT,
    stride: int = config.OCCLUSION_STRIDE_DEFAULT,
    fill_value: float = config.OCCLUSION_FILL_DEFAULT,
    occlusion_target: OcclusionTarget = "predicted",
    csv_path: str | None = None,
) -> str | None:
    """First batch only; image_rel_paths[k] must match batch order. Returns csv_path or None."""
    os.makedirs(save_dir, exist_ok=True)
    rows_out: list[dict] = []
    csv_out = csv_path

    taken = 0
    for (rgb_b, _gray_b, _), labels in val_dataset.take(1):
        bs = int(rgb_b.shape[0])
        n = min(num_samples, bs)
        for i in range(n):
            rgb = rgb_b[i].numpy()
            true_idx = int(labels[i].numpy())
            pred_vec = model.predict(
                _triple_batch_from_rgb(rgb[np.newaxis, ...]), verbose=0
            )
            pred_probs = batch_preds_to_probs(pred_vec)[0]
            pred_idx = int(np.argmax(pred_probs))

            if occlusion_target == "true":
                explain_idx = true_idx
            else:
                explain_idx = pred_idx

            grid, _bpv, baseline_p, _gsh = compute_occlusion_map(
                model,
                rgb,
                class_index=explain_idx,
                patch_size=patch_size,
                stride=stride,
                fill_value=fill_value,
            )
            rel_img = ""
            if image_rel_paths is not None and taken < len(image_rel_paths):
                rel_img = image_rel_paths[taken]

            fname = f"occlusion_{name_tag}_sample{taken}.png"
            fpath = os.path.join(save_dir, fname)
            tgt_name = class_names[explain_idx]
            title = (
                f"{name_tag} | target={occlusion_target} ({tgt_name}) | "
                f"true={class_names[true_idx]} pred={class_names[pred_idx]}"
            )
            save_occlusion_figure(
                rgb,
                grid,
                fpath,
                title=title,
                class_names=class_names,
                class_index=explain_idx,
            )

            rows_out.append(
                {
                    "run_id": run_id,
                    "sample_index": taken,
                    "image_rel_path": rel_img,
                    "occlusion_target": occlusion_target,
                    "target_class_index": explain_idx,
                    "target_class_name": tgt_name,
                    "y_true": true_idx,
                    "y_pred": pred_idx,
                    "patch_size": patch_size,
                    "stride": stride,
                    "fill_value": fill_value,
                    "baseline_prob_target": baseline_p,
                    "mean_importance": float(np.mean(grid)),
                    "max_importance": float(np.max(grid)),
                    "std_importance": float(np.std(grid)),
                    "grid_h": grid.shape[0],
                    "grid_w": grid.shape[1],
                    "figure_filename": fname,
                }
            )
            taken += 1
        break

    if csv_out and rows_out:
        pd.DataFrame(rows_out).to_csv(csv_out, index=False)

    return csv_out if rows_out else None
