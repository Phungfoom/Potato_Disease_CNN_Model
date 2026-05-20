"""Phase 6 — Practical counterfactuals: contrastive k-NN + greedy patch masking to flip prediction."""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

import config
from neighbor_lookup import _gather_queries, _cosine_sim_from_l2_dist, build_embedding_model
from occlusion_attribution import (
    _apply_square_patch,
    _triple_batch_from_rgb,
    compute_occlusion_map,
)
from prediction_utils import batch_preds_to_probs

ContrastiveExclude = Literal["pred", "true"]


def _nearest_contrastive_index(
    *,
    train_embeddings: np.ndarray,
    train_y: np.ndarray,
    query_emb: np.ndarray,
    exclude_class: int,
    search_cap: int = 200,
) -> tuple[int | None, int | None, float | None, float | None]:
    """
    Search up to ``search_cap`` nearest neighbors of ``query_emb`` in ``train_embeddings``
    and return the first training index with ``train_y[index] != exclude_class``.
    Returns (train_index, rank_1based, dist, cos_sim).
    """
    k = min(search_cap, len(train_embeddings))
    nn_index = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)
    dist, idx = nn_index.kneighbors(query_emb.reshape(1, -1))
    dist_row = dist[0]
    idx_row = idx[0]
    for rank, (d, j) in enumerate(zip(dist_row, idx_row), start=1):
        j = int(j)
        if int(train_y[j]) != int(exclude_class):
            sim = float(_cosine_sim_from_l2_dist(np.array([d]))[0])
            return j, rank, float(d), sim
    return None, None, None, None


def greedy_mask_flip_search(
    model: tf.keras.Model,
    rgb_hw3: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    fill_value: float,
    max_patches: int,
) -> dict:
    """
    Sort occlusion cells by how much masking that cell drops prob of the *initial* argmax class,
    then greedily add patches (cumulative) until argmax changes or ``max_patches`` is reached.
    """
    rgb = np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0)
    h, w = rgb.shape[0], rgb.shape[1]
    ps = int(patch_size)
    st = int(stride)

    base = model.predict(_triple_batch_from_rgb(rgb[np.newaxis, ...]), verbose=0)
    p0 = batch_preds_to_probs(base)[0]
    c0 = int(np.argmax(p0))

    grid, _, _baseline_p, _gsh = compute_occlusion_map(
        model,
        rgb,
        class_index=c0,
        patch_size=ps,
        stride=st,
        fill_value=fill_value,
    )
    rows = list(range(0, h - ps + 1, st))
    cols = list(range(0, w - ps + 1, st))
    gh, gw = len(rows), len(cols)
    order: list[tuple[float, int, int]] = []
    for gi in range(gh):
        for gj in range(gw):
            order.append((float(grid[gi, gj]), gi, gj))
    order.sort(key=lambda t: t[0], reverse=True)

    masked = np.array(rgb, copy=True, dtype=np.float32)
    applied: list[tuple[int, int]] = []
    limit = min(int(max_patches), len(order))
    for k in range(limit):
        _, gi, gj = order[k]
        r0, c0pix = rows[gi], cols[gj]
        masked = _apply_square_patch(masked, r0, c0pix, ps, fill_value)
        applied.append((r0, c0pix))
        pred = model.predict(_triple_batch_from_rgb(masked[np.newaxis, ...]), verbose=0)
        pk = batch_preds_to_probs(pred)[0]
        ck = int(np.argmax(pk))
        if ck != c0:
            return {
                "initial_pred": c0,
                "initial_prob_top1": float(p0[c0]),
                "flipped": True,
                "n_patches": k + 1,
                "new_pred": ck,
                "new_prob_top1": float(pk[ck]),
                "masked_rgb": masked,
                "applied_patch_corners_hw": applied,
            }

    pred_f = model.predict(_triple_batch_from_rgb(masked[np.newaxis, ...]), verbose=0)
    pf = batch_preds_to_probs(pred_f)[0]
    cf = int(np.argmax(pf))
    return {
        "initial_pred": c0,
        "initial_prob_top1": float(p0[c0]),
        "flipped": False,
        "n_patches": limit,
        "new_pred": cf,
        "new_prob_top1": float(pf[cf]),
        "masked_rgb": masked,
        "applied_patch_corners_hw": applied,
    }


def _save_mask_flip_figure(
    rgb_orig: np.ndarray,
    result: dict,
    *,
    save_path: str,
    title: str,
    patch_size: int,
    class_names: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.clip(rgb_orig, 0.0, 1.0))
    axes[0].set_title(f"Original (pred: {class_names[result['initial_pred']]})")
    axes[0].axis("off")

    ax1 = axes[1]
    ax1.imshow(np.clip(result["masked_rgb"], 0.0, 1.0))
    ps = int(patch_size)
    for r0, c0 in result.get("applied_patch_corners_hw", []):
        rect = mpatches.Rectangle(
            (c0, r0),
            ps,
            ps,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax1.add_patch(rect)
    flip_note = "flipped" if result["flipped"] else "no flip (bounded)"
    ax1.set_title(
        f"After {result['n_patches']} patch(es) | {flip_note} → {class_names[result['new_pred']]}"
    )
    ax1.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_counterfactual_audit(
    *,
    full_model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    val_rel_paths: list[str],
    class_names: list[str],
    metrics_dir: str,
    attribution_dir: str,
    name_tag: str,
    run_id: str,
    split_label: str,
    num_samples: int,
    contrastive_bank: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None,
    contrastive_search_cap: int,
    contrastive_exclude: ContrastiveExclude,
    patch_size: int,
    stride: int,
    fill_value: float,
    max_mask_patches: int,
    embedding_layer: str | None = None,
) -> dict[str, str | None]:
    """
    Writes contrastive CSV (if ``contrastive_bank`` set) and mask-flip CSV + figures under audit paths.
    Returns keys: contrastive_table_csv, mask_flip_summary_csv (paths or None).
    """
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(attribution_dir, exist_ok=True)
    layer = embedding_layer or config.NEIGHBOR_EMBEDDING_LAYER

    contrastive_csv_path: str | None = None
    if contrastive_bank is not None:
        train_emb, train_y, manifest_df = contrastive_bank
        if len(manifest_df) != len(train_emb):
            raise ValueError("contrastive manifest length must match embedding rows")
        embed_model = build_embedding_model(full_model, layer)
        q_emb, q_y, q_pred, q_paths = _gather_queries(
            embed_model,
            full_model,
            val_dataset,
            val_rel_paths,
            num_samples,
        )
        rows_c: list[dict] = []
        for qi in range(len(q_emb)):
            excl = int(q_pred[qi]) if contrastive_exclude == "pred" else int(q_y[qi])
            tidx, rank, dist, sim = _nearest_contrastive_index(
                train_embeddings=train_emb,
                train_y=train_y,
                query_emb=q_emb[qi],
                exclude_class=excl,
                search_cap=contrastive_search_cap,
            )
            row_m = manifest_df.iloc[int(tidx)] if tidx is not None else None
            rows_c.append(
                {
                    "run_id": run_id,
                    "split": split_label,
                    "query_index": qi,
                    "query_image_rel_path": q_paths[qi] if qi < len(q_paths) else "",
                    "query_y_true": int(q_y[qi]),
                    "query_y_pred": int(q_pred[qi]),
                    "contrastive_exclude_relative_to": contrastive_exclude,
                    "excluded_class_index": excl,
                    "contrastive_neighbor_found": tidx is not None,
                    "contrastive_neighbor_rank": rank if rank is not None else "",
                    "contrastive_train_index": int(tidx) if tidx is not None else "",
                    "contrastive_image_rel_path": (
                        str(row_m["image_rel_path"]) if row_m is not None else ""
                    ),
                    "contrastive_y_true": int(row_m["y_true"]) if row_m is not None else "",
                    "contrastive_class": str(row_m["true_class"]) if row_m is not None else "",
                    "euclidean_distance": dist if dist is not None else "",
                    "cosine_similarity": sim if sim is not None else "",
                }
            )
        contrastive_csv_path = os.path.join(
            metrics_dir, f"counterfactual_contrastive_{split_label}_{name_tag}.csv"
        )
        pd.DataFrame(rows_c).to_csv(contrastive_csv_path, index=False)

    rows_m: list[dict] = []
    taken_m = 0
    for (rgb_b, _g, _), labels in val_dataset:
        bs = int(rgb_b.shape[0])
        for i in range(bs):
            if taken_m >= int(num_samples):
                break
            rgb = rgb_b[i].numpy()
            true_idx = int(labels[i].numpy())
            rel_img = val_rel_paths[taken_m] if taken_m < len(val_rel_paths) else ""

            result = greedy_mask_flip_search(
                full_model,
                rgb,
                patch_size=patch_size,
                stride=stride,
                fill_value=fill_value,
                max_patches=max_mask_patches,
            )
            pred_before = int(result["initial_pred"])

            qi = taken_m
            fig_name = f"counterfactual_mask_flip_{split_label}_{name_tag}_q{qi}.png"
            fig_path = os.path.join(attribution_dir, fig_name)
            _save_mask_flip_figure(
                rgb,
                result,
                save_path=fig_path,
                patch_size=patch_size,
                class_names=class_names,
                title=(
                    f"{name_tag} | q{qi} | true={class_names[true_idx]} "
                    f"pred_in={class_names[pred_before]}"
                ),
            )

            corners = result.get("applied_patch_corners_hw", [])
            coord_str = ";".join(f"{r},{c}" for r, c in corners)

            rows_m.append(
                {
                    "run_id": run_id,
                    "split": split_label,
                    "query_index": qi,
                    "image_rel_path": rel_img,
                    "y_true": true_idx,
                    "baseline_pred": result["initial_pred"],
                    "baseline_pred_prob": result["initial_prob_top1"],
                    "prediction_flipped": result["flipped"],
                    "n_patches_applied": result["n_patches"],
                    "patch_grid_corners_rc": coord_str,
                    "pred_after_masking": result["new_pred"],
                    "prob_top1_after": result["new_prob_top1"],
                    "patch_size": patch_size,
                    "stride": stride,
                    "fill_value": fill_value,
                    "max_mask_patches_budget": max_mask_patches,
                    "figure_filename": fig_name,
                    "method_note": (
                        "Heuristic greedy mask using occlusion ranking on initial predicted class; "
                        "not a minimal edit in a formal sense."
                    ),
                }
            )
            taken_m += 1
        if taken_m >= int(num_samples):
            break

    mask_csv = os.path.join(
        metrics_dir, f"counterfactual_mask_flip_{split_label}_{name_tag}.csv"
    )
    if rows_m:
        pd.DataFrame(rows_m).to_csv(mask_csv, index=False)

    return {
        "contrastive_table_csv": contrastive_csv_path,
        "mask_flip_summary_csv": mask_csv if rows_m else None,
    }
