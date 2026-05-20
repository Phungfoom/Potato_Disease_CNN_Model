"""Phase 7 — Structured reporting: per-class metrics, confusion tables, path-based slices, failure gallery."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config


def path_source_slice(rel_path: str) -> str:
    """
    Heuristic slice key from a relative image path: last directory segment before the filename.
    Example: processed_data/val/Healthy/img.jpg -> Healthy; Late_Blight/x.png -> Late_Blight.
    Use for grouped metrics when no external metadata column exists.
    """
    norm = rel_path.replace("\\", "/").strip()
    if not norm:
        return "_empty"
    parent = os.path.dirname(norm)
    if not parent or parent == ".":
        return "_root"
    return parent.replace("\\", "/").split("/")[-1] or "_root"


def export_per_class_metrics_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: str,
    *,
    split_label: str,
) -> str:
    """Precision / recall / F1 / support per class; macro and weighted averages as extra rows."""
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    labels = np.arange(len(class_names), dtype=int)
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict] = []
    for name in class_names:
        block = rep.get(name)
        if isinstance(block, dict):
            rows.append(
                {
                    "split": split_label,
                    "row_type": "class",
                    "name": name,
                    "precision": float(block["precision"]),
                    "recall": float(block["recall"]),
                    "f1": float(block["f1-score"]),
                    "support": int(block["support"]),
                    "accuracy": np.nan,
                }
            )
    for tag in ("macro avg", "weighted avg"):
        block = rep.get(tag)
        if isinstance(block, dict):
            rows.append(
                {
                    "split": split_label,
                    "row_type": tag.replace(" ", "_"),
                    "name": tag,
                    "precision": float(block["precision"]),
                    "recall": float(block["recall"]),
                    "f1": float(block["f1-score"]),
                    "support": int(block["support"]),
                    "accuracy": np.nan,
                }
            )
    acc = rep.get("accuracy")
    try:
        acc_f = float(acc) if acc is not None else None
    except (TypeError, ValueError):
        acc_f = None
    if acc_f is not None:
        rows.append(
            {
                "split": split_label,
                "row_type": "overall_accuracy",
                "name": "accuracy",
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "support": int(len(y_true)),
                "accuracy": acc_f,
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def export_confusion_matrix_csvs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_counts: str,
    out_row_norm: str,
    *,
    split_label: str,
) -> tuple[str, str]:
    """Write raw counts and row-normalized confusion matrices (same label order as model)."""
    labels = np.arange(len(class_names), dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_rn = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    idx = [f"true_{c}" for c in class_names]
    cols = [f"pred_{c}" for c in class_names]
    for path, mat in ((out_counts, cm), (out_row_norm, cm_rn)):
        par = os.path.dirname(os.path.abspath(path))
        if par:
            os.makedirs(par, exist_ok=True)
        df = pd.DataFrame(mat, index=idx, columns=cols)
        df.insert(0, "split", split_label)
        df.to_csv(path)
    return out_counts, out_row_norm


def export_slice_metrics_csv(
    pred_df: pd.DataFrame,
    out_path: str,
    *,
    split_label: str,
) -> str:
    """
    Group by path-derived slice (parent folder of image). If you add a column ``report_slice``
    or ``source`` to predictions later, this function will prefer it over the path heuristic.
    """
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if pred_df.empty:
        cols = [
            "split",
            "slice_key",
            "n_samples",
            "accuracy",
            "mean_margin_top1_top2",
            "n_errors",
            "slice_note",
        ]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        return out_path
    work = pred_df.copy()
    if "report_slice" in work.columns:
        work["_slice"] = work["report_slice"].astype(str)
    elif "source" in work.columns:
        work["_slice"] = work["source"].astype(str)
    else:
        work["_slice"] = work["image_rel_path"].astype(str).map(path_source_slice)
    rows: list[dict] = []
    for slice_v, g in work.groupby("_slice", dropna=False):
        yt = g["y_true"].to_numpy(dtype=np.int64)
        yp = g["y_pred"].to_numpy(dtype=np.int64)
        n = len(g)
        acc = float(np.mean(yt == yp)) if n else 0.0
        margin_mean = (
            float(g["margin_top1_top2"].mean())
            if "margin_top1_top2" in g.columns
            else float("nan")
        )
        rows.append(
            {
                "split": split_label,
                "slice_key": str(slice_v),
                "n_samples": n,
                "accuracy": acc,
                "mean_margin_top1_top2": margin_mean,
                "n_errors": int(np.sum(yt != yp)),
                "slice_note": (
                    "path_parent_folder unless predictions include report_slice or source"
                ),
            }
        )
    pd.DataFrame(rows).sort_values("slice_key").to_csv(out_path, index=False)
    return out_path


def _load_rgb_for_gallery(abs_path: str, image_size: tuple[int, int]) -> np.ndarray:
    if not os.path.isfile(abs_path):
        return np.zeros((*image_size, 3), dtype=np.float32)
    img = tf.keras.utils.load_img(abs_path, target_size=image_size)
    return np.asarray(tf.keras.utils.img_to_array(img), dtype=np.float32) / 255.0


def build_failure_gallery_png(
    pred_df: pd.DataFrame,
    *,
    image_lookup_root: str,
    failures_dir: str,
    filename: str,
    class_names: list[str],
    high_confidence_threshold: float,
    max_high_conf_wrong: int,
    max_low_margin: int,
    low_margin_threshold: float | None,
) -> str | None:
    """
    One figure: top row = high-confidence wrong (sorted by prob_top1 desc).
    Bottom row = lowest-margin cases (ambiguous); prefer errors, then fill with lowest margins overall.
    """
    if pred_df.empty:
        return None
    os.makedirs(failures_dir, exist_ok=True)
    out_path = os.path.join(failures_dir, filename)
    h, w = config.DATA_PARAMS["image_size"]

    need = {"high_conf_wrong", "prob_top1", "margin_top1_top2", "image_rel_path", "y_true", "y_pred"}
    if not need.issubset(pred_df.columns):
        return None

    hcw = pred_df[pred_df["high_conf_wrong"] == True].copy()  # noqa: E712
    if len(hcw):
        hcw = hcw.sort_values("prob_top1", ascending=False).reset_index(drop=True).head(
            max_high_conf_wrong
        )

    work_margin = pred_df
    if low_margin_threshold is not None and "margin_top1_top2" in work_margin.columns:
        m = work_margin["margin_top1_top2"] <= low_margin_threshold
        work_margin = work_margin[m]
    wrong_first = work_margin[work_margin["y_true"] != work_margin["y_pred"]].sort_values(
        "margin_top1_top2", ascending=True
    )
    rest = work_margin.sort_values("margin_top1_top2", ascending=True)
    low_ids: list[int] = []
    seen: set[int] = set()
    for df_part in (wrong_first, rest):
        for i in df_part.index:
            if i in seen:
                continue
            seen.add(i)
            low_ids.append(i)
            if len(low_ids) >= max_low_margin:
                break
        if len(low_ids) >= max_low_margin:
            break
    if low_ids:
        low_rows = pred_df.loc[low_ids].reset_index(drop=True)
    else:
        low_rows = (
            pred_df.sort_values("margin_top1_top2", ascending=True)
            .head(max_low_margin)
            .reset_index(drop=True)
        )

    n_h = len(hcw)
    n_l = len(low_rows)
    ncols = max(n_h, n_l, 1)
    fig_h = 5.2 * 2
    fig_w = min(3.2 * ncols, 28)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(ncols):
        ax = axes[0, j]
        if j < len(hcw):
            row = hcw.iloc[j]
            rel = str(row["image_rel_path"])
            abs_p = os.path.join(image_lookup_root, rel.replace("/", os.sep))
            im = _load_rgb_for_gallery(abs_p, (h, w))
            ax.imshow(np.clip(im, 0, 1))
            t_idx = int(row["y_true"])
            p_idx = int(row["y_pred"])
            ax.set_title(
                f"HC wrong\nT:{class_names[t_idx]} P:{class_names[p_idx]}\n"
                f"p={row['prob_top1']:.2f}",
                fontsize=8,
            )
        else:
            ax.axis("off")
        ax.axis("off")

    for j in range(ncols):
        ax = axes[1, j]
        if j < len(low_rows):
            row = low_rows.iloc[j]
            rel = str(row["image_rel_path"])
            abs_p = os.path.join(image_lookup_root, rel.replace("/", os.sep))
            im = _load_rgb_for_gallery(abs_p, (h, w))
            ax.imshow(np.clip(im, 0, 1))
            t_idx = int(row["y_true"])
            p_idx = int(row["y_pred"])
            mk = "WR" if t_idx != p_idx else "OK"
            ax.set_title(
                f"Low margin ({mk})\nT:{class_names[t_idx]} P:{class_names[p_idx]}\n"
                f"m={row['margin_top1_top2']:.3f}",
                fontsize=8,
            )
        else:
            ax.axis("off")
        ax.axis("off")

    thr_note = f"high_conf_wrong uses prob_top1>={high_confidence_threshold}"
    fig.suptitle(
        f"Failure gallery | {thr_note} | row2 = lowest margin (errors preferred)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_phase7_audit_bundle(
    *,
    pred_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    metrics_dir: str,
    failures_dir: str,
    image_lookup_root: str,
    name_tag: str,
    split_label: str,
    high_confidence_threshold: float,
) -> dict[str, str | None]:
    """
    Phase 7 exports under existing audit folders (metrics/, failures/).
    ``image_lookup_root``: directory that ``pred_df['image_rel_path']`` is relative to.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)
    n_samples = int(len(np.asarray(y_true).ravel()))
    pc = os.path.join(metrics_dir, f"per_class_metrics_{split_label}_{name_tag}.csv")
    cm_c = os.path.join(
        metrics_dir, f"confusion_matrix_counts_{split_label}_{name_tag}.csv"
    )
    cm_n = os.path.join(
        metrics_dir, f"confusion_matrix_row_norm_{split_label}_{name_tag}.csv"
    )
    sl = os.path.join(metrics_dir, f"slice_metrics_path_{split_label}_{name_tag}.csv")
    gal_name = f"failure_gallery_{split_label}_{name_tag}.png"

    if n_samples == 0:
        export_slice_metrics_csv(pred_df, sl, split_label=split_label)
        pd.DataFrame([{"note": "no samples; phase 7 metrics skipped"}]).to_csv(
            pc, index=False
        )
        pd.DataFrame([{"note": "no samples"}]).to_csv(cm_c, index=False)
        pd.DataFrame([{"note": "no samples"}]).to_csv(cm_n, index=False)
        return {
            "per_class_metrics_csv": pc,
            "confusion_matrix_counts_csv": cm_c,
            "confusion_matrix_row_norm_csv": cm_n,
            "slice_metrics_csv": sl,
            "failure_gallery_png": None,
        }

    export_per_class_metrics_csv(
        y_true, y_pred, class_names, pc, split_label=split_label
    )
    export_confusion_matrix_csvs(
        y_true, y_pred, class_names, cm_c, cm_n, split_label=split_label
    )
    export_slice_metrics_csv(pred_df, sl, split_label=split_label)
    gal = build_failure_gallery_png(
        pred_df,
        image_lookup_root=image_lookup_root,
        failures_dir=failures_dir,
        filename=gal_name,
        class_names=class_names,
        high_confidence_threshold=high_confidence_threshold,
        max_high_conf_wrong=config.FAILURE_GALLERY_MAX_HIGH_CONF_WRONG,
        max_low_margin=config.FAILURE_GALLERY_MAX_LOW_MARGIN,
        low_margin_threshold=config.FAILURE_GALLERY_LOW_MARGIN_THRESHOLD,
    )
    return {
        "per_class_metrics_csv": pc,
        "confusion_matrix_counts_csv": cm_c,
        "confusion_matrix_row_norm_csv": cm_n,
        "slice_metrics_csv": sl,
        "failure_gallery_png": gal,
    }
