from __future__ import annotations

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from build_sobel_model import sobel_edge_layer
from focal_loss import SparseCategoricalFocalLoss
from audit_bundle import (
    ensure_audit_run,
    merge_run_info,
    rel_from_script,
    validate_counterfactual_cli,
    validate_ig_cli,
    validate_neighbors_cli,
    validate_occlusion_cli,
    write_run_info,
)
from counterfactual_audit import run_counterfactual_audit
from grad_cam_visualizer import visualize_gradcam_batch
from structured_reporting import write_phase7_audit_bundle
from integrated_gradients import run_integrated_gradients_batch
from neighbor_lookup import run_field_neighbor_pipeline
from occlusion_attribution import run_occlusion_val_batch
from shortcut_audit import run_shortcut_background_audit
from prediction_utils import (
    HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
    PREDICTION_SCHEMA_VERSION,
    batch_preds_to_probs,
    build_standard_predictions_dataframe,
)


def prepare_triple_input(image, label):
    """(RGB, gray, gray) in [0,1], same as train_model eval."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    return (norm_image, gray_image, gray_image), label


def build_field_dataset(field_dir: str) -> tuple[tf.data.Dataset, list[str], list[str]]:
    """Field images, shuffle=False; returns dataset, class_names, file_paths."""
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


def evaluate_on_field(
    model_path: str,
    domain: str = config.TRAINING_DOMAIN,
    audit_run_id: str | None = None,
    base_dir: str | None = None,
    split: str = "val",
    high_confidence_threshold: float | None = None,
    *,
    gradcam_target: str = "predicted",
    show_shallow_rgb: bool = True,
    run_occlusion: bool = False,
    occlusion_samples: int = config.OCCLUSION_NUM_SAMPLES_DEFAULT,
    occlusion_patch: int = config.OCCLUSION_PATCH_DEFAULT,
    occlusion_stride: int = config.OCCLUSION_STRIDE_DEFAULT,
    occlusion_fill: float = config.OCCLUSION_FILL_DEFAULT,
    occlusion_target: str = "predicted",
    run_ig: bool = False,
    ig_samples: int = config.IG_NUM_SAMPLES_DEFAULT,
    ig_steps: int = config.IG_M_STEPS_DEFAULT,
    ig_baseline: str = "black",
    ig_target: str = "predicted",
    ig_gradcam_compare: bool = True,
    run_neighbors: bool = False,
    neighbors_train_npz: str | None = None,
    neighbors_train_manifest_csv: str | None = None,
    neighbors_top_k: int = config.NEIGHBOR_TOP_K_DEFAULT,
    neighbors_samples: int = config.NEIGHBOR_FIELD_QUERY_SAMPLES_DEFAULT,
    neighbors_grid_figures: int = config.NEIGHBOR_GRID_FIGURES_DEFAULT,
    neighbors_embedding_layer: str | None = None,
    run_counterfactuals: bool = False,
    counterfactual_samples: int = config.COUNTERFACTUAL_FIELD_SAMPLES_DEFAULT,
    counterfactual_contrastive_search_cap: int = config.COUNTERFACTUAL_CONTRASTIVE_SEARCH_CAP_DEFAULT,
    counterfactual_max_mask_patches: int = config.COUNTERFACTUAL_MASK_FLIP_MAX_PATCHES_DEFAULT,
    counterfactual_patch: int = config.OCCLUSION_PATCH_DEFAULT,
    counterfactual_stride: int = config.OCCLUSION_STRIDE_DEFAULT,
    counterfactual_fill: float = config.OCCLUSION_FILL_DEFAULT,
    counterfactual_exclude: str = "pred",
    run_shortcut_audit: bool = False,
    shortcut_audit_samples: int = config.SHORTCUT_AUDIT_SAMPLES_DEFAULT,
    shortcut_border_frac: float = config.SHORTCUT_BORDER_FRAC_DEFAULT,
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    _base = base_dir if base_dir is not None else config.BASE_DIR
    if split in ("val", "test", "train"):
        field_dir = os.path.join(script_dir, _base, split)
    else:
        field_dir = os.path.join(script_dir, _base, config.FIELD_CLASSES_SUBDIR)
    field_spud, class_names, file_paths = build_field_dataset(field_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = audit_run_id or f"field_{timestamp}"
    audit_paths = ensure_audit_run(script_dir, run_id)
    conf_thr = (
        float(high_confidence_threshold)
        if high_confidence_threshold is not None
        else HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT
    )
    _custom_objects = {
        "sobel_edge_layer": sobel_edge_layer,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
    }
    try:
        model = tf.keras.models.load_model(
            model_path, custom_objects=_custom_objects, safe_mode=False
        )
    except TypeError:
        model = tf.keras.models.load_model(model_path, custom_objects=_custom_objects)

    if run_occlusion:
        validate_occlusion_cli(
            occlusion_samples,
            occlusion_patch,
            occlusion_stride,
            occlusion_fill,
            config.DATA_PARAMS["image_size"],
        )

    if run_ig:
        validate_ig_cli(ig_samples, ig_steps)

    if run_neighbors:
        if not neighbors_train_npz or not neighbors_train_manifest_csv:
            raise ValueError(
                "run_neighbors requires neighbors_train_npz and neighbors_train_manifest_csv "
                "(from a training run with train_model.py --neighbors)."
            )
        validate_neighbors_cli(
            neighbors_top_k, neighbors_samples, neighbors_grid_figures
        )
    if run_shortcut_audit:
        if shortcut_audit_samples < 1:
            raise ValueError("shortcut_audit_samples must be >= 1")
        if not 0.0 < float(shortcut_border_frac) < 0.49:
            raise ValueError("shortcut_border_frac must be in (0, 0.49)")

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
    rel_paths = [
        os.path.relpath(p, field_dir).replace("\\", "/") for p in file_paths
    ]
    if len(rel_paths) != len(y_true):
        print(
            f"[WARN] file_paths ({len(rel_paths)}) != predictions ({len(y_true)}); "
            "CSV paths may be misaligned."
        )
    model_abs = os.path.abspath(model_path)
    results = build_standard_predictions_dataframe(
        run_id=run_id,
        split="field",
        domain=domain,
        stage="field_eval",
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs_stack,
        image_rel_paths=rel_paths[: len(y_true)],
        model_rel_path=rel_from_script(script_dir, model_abs),
        high_confidence_threshold=conf_thr,
    )
    pred_csv = os.path.join(
        audit_paths["predictions"],
        f"field_predictions_{domain}_{timestamp}.csv",
    )
    results.to_csv(pred_csv, index=False)
    print(f"[FIELD] Saved per-class probabilities (master schema): {pred_csv}")

    if run_shortcut_audit:
        _sc_csv = os.path.join(
            audit_paths["metrics"],
            f"shortcut_border_focus_field_{domain}_{timestamp}.csv",
        )
        _sc_written = run_shortcut_background_audit(
            model=model,
            dataset=field_spud,
            rel_paths=rel_paths,
            class_names=class_names,
            run_id=run_id,
            split_label=f"field_{domain}",
            save_csv_path=_sc_csv,
            num_samples=shortcut_audit_samples,
            border_frac=float(shortcut_border_frac),
            gradcam_target=gradcam_target,
            gradcam_layer=config.GRADCAM_RGB_DEEP_LAYER,
        )
        if _sc_written:
            merge_run_info(
                audit_paths["meta"],
                {
                    "shortcut_audit": {
                        "border_focus_csv": rel_from_script(script_dir, _sc_written),
                        "samples": shortcut_audit_samples,
                        "border_frac": float(shortcut_border_frac),
                        "gradcam_target": gradcam_target,
                        "gradcam_layer": config.GRADCAM_RGB_DEEP_LAYER,
                    }
                },
            )
            print(f"[FIELD] Shortcut audit: {_sc_written}")

    _p7_field = write_phase7_audit_bundle(
        pred_df=results,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        class_names=class_names,
        metrics_dir=audit_paths["metrics"],
        failures_dir=audit_paths["failures"],
        image_lookup_root=field_dir,
        name_tag=f"{domain}_{timestamp}",
        split_label=f"field_{domain}",
        high_confidence_threshold=conf_thr,
    )
    print(
        f"[FIELD] Phase 7: {_p7_field['per_class_metrics_csv']} | "
        f"{_p7_field.get('failure_gallery_png') or 'no gallery'}"
    )

    write_run_info(
        audit_paths["meta"],
        {
            "run_id": run_id,
            "domain": domain,
            "stage": "field_eval",
            "model_path": rel_from_script(script_dir, model_abs),
            "field_predictions_csv": rel_from_script(script_dir, pred_csv),
            "class_names": class_names,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "high_confidence_wrong_threshold": conf_thr,
            "gradcam_target": gradcam_target,
            "gradcam_show_shallow_rgb": show_shallow_rgb,
            "phase7": {
                "per_class_metrics_csv": rel_from_script(
                    script_dir, _p7_field["per_class_metrics_csv"]
                ),
                "confusion_matrix_counts_csv": rel_from_script(
                    script_dir, _p7_field["confusion_matrix_counts_csv"]
                ),
                "confusion_matrix_row_norm_csv": rel_from_script(
                    script_dir, _p7_field["confusion_matrix_row_norm_csv"]
                ),
                "slice_metrics_csv": rel_from_script(
                    script_dir, _p7_field["slice_metrics_csv"]
                ),
                "failure_gallery_png": rel_from_script(
                    script_dir, _p7_field["failure_gallery_png"]
                )
                if _p7_field.get("failure_gallery_png")
                else "",
            },
        },
    )

    plot_dir = os.path.join(
        script_dir, config.OUTPUT_DIR, *config.FUSION_PLOTS_PATH_SEGMENTS
    )
    attribution_dir = audit_paths["attribution"]
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
        save_dir=attribution_dir,
        name_tag=f"field_{domain}_{timestamp}",
        class_names=class_names,
        gradcam_target=gradcam_target,
        show_shallow_rgb=show_shallow_rgb,
    )

    if run_occlusion:
        _tag = f"field_{domain}_{timestamp}"
        _occ_csv = os.path.join(
            audit_paths["metrics"],
            f"occlusion_summary_field_{domain}_{timestamp}.csv",
        )
        _occ_done = run_occlusion_val_batch(
            model=model,
            val_dataset=field_spud,
            class_names=class_names,
            save_dir=attribution_dir,
            name_tag=_tag,
            run_id=run_id,
            image_rel_paths=rel_paths,
            num_samples=occlusion_samples,
            patch_size=occlusion_patch,
            stride=occlusion_stride,
            fill_value=occlusion_fill,
            occlusion_target=occlusion_target,
            csv_path=_occ_csv,
        )
        if _occ_done:
            merge_run_info(
                audit_paths["meta"],
                {
                    "occlusion_summary_csv": rel_from_script(script_dir, _occ_csv),
                    "occlusion": {
                        "samples": occlusion_samples,
                        "patch_size": occlusion_patch,
                        "stride": occlusion_stride,
                        "fill_value": occlusion_fill,
                        "target": occlusion_target,
                    },
                },
            )
            print(f"[FIELD] Occlusion: {attribution_dir} + {_occ_csv}")

    if run_ig:
        _tag = f"field_{domain}_{timestamp}"
        _ig_csv = os.path.join(
            audit_paths["metrics"],
            f"ig_summary_field_{domain}_{timestamp}.csv",
        )
        _ig_done = run_integrated_gradients_batch(
            model=model,
            val_dataset=field_spud,
            class_names=class_names,
            save_dir=attribution_dir,
            name_tag=_tag,
            run_id=run_id,
            image_rel_paths=rel_paths,
            num_samples=ig_samples,
            m_steps=ig_steps,
            baseline="black",
            ig_target=ig_target,
            csv_path=_ig_csv,
            compare_gradcam=ig_gradcam_compare,
        )
        if _ig_done:
            merge_run_info(
                audit_paths["meta"],
                {
                    "integrated_gradients_summary_csv": rel_from_script(script_dir, _ig_csv),
                    "integrated_gradients": {
                        "samples": ig_samples,
                        "m_steps": ig_steps,
                        "baseline": "black",
                        "target": ig_target,
                        "gradcam_compare": ig_gradcam_compare,
                    },
                },
            )
            print(f"[FIELD] Integrated Gradients: {attribution_dir} + {_ig_csv}")

    if run_neighbors:
        assert neighbors_train_npz is not None and neighbors_train_manifest_csv is not None
        npz_abs = (
            neighbors_train_npz
            if os.path.isabs(neighbors_train_npz)
            else os.path.join(script_dir, neighbors_train_npz)
        )
        csv_abs = (
            neighbors_train_manifest_csv
            if os.path.isabs(neighbors_train_manifest_csv)
            else os.path.join(script_dir, neighbors_train_manifest_csv)
        )
        _tag = f"field_{domain}_{timestamp}"
        _nb = run_field_neighbor_pipeline(
            full_model=model,
            field_dataset=field_spud,
            field_rel_paths=rel_paths,
            field_images_dir=field_dir,
            train_embeddings_npz=npz_abs,
            train_manifest_csv=csv_abs,
            class_names=class_names,
            neighbors_dir=audit_paths["neighbors"],
            name_tag=_tag,
            run_id=run_id,
            top_k=neighbors_top_k,
            max_queries=neighbors_samples,
            max_grid_figures=neighbors_grid_figures,
            project_root=script_dir,
            embedding_layer=neighbors_embedding_layer,
        )
        merge_run_info(
            audit_paths["meta"],
            {
                "neighbor_field_table_csv": rel_from_script(
                    script_dir, _nb["field_neighbor_table_csv"]
                ),
                "neighbors_field": {
                    "train_embeddings_npz": rel_from_script(script_dir, npz_abs),
                    "train_manifest_csv": rel_from_script(script_dir, csv_abs),
                    "embedding_layer": _nb["embedding_layer"],
                    "top_k": neighbors_top_k,
                    "field_query_samples": neighbors_samples,
                    "grid_figures": neighbors_grid_figures,
                },
            },
        )
        print(f"[FIELD] Neighbors: {audit_paths['neighbors']} + {_nb['field_neighbor_table_csv']}")

    if run_counterfactuals:
        validate_counterfactual_cli(
            counterfactual_samples,
            counterfactual_contrastive_search_cap,
            counterfactual_max_mask_patches,
            counterfactual_patch,
            counterfactual_stride,
            counterfactual_fill,
            config.DATA_PARAMS["image_size"],
        )
        _cf_bank = None
        if neighbors_train_npz and neighbors_train_manifest_csv:
            _npz_cf = (
                neighbors_train_npz
                if os.path.isabs(neighbors_train_npz)
                else os.path.join(script_dir, neighbors_train_npz)
            )
            _csv_cf = (
                neighbors_train_manifest_csv
                if os.path.isabs(neighbors_train_manifest_csv)
                else os.path.join(script_dir, neighbors_train_manifest_csv)
            )
            _zd = np.load(_npz_cf)
            _cf_bank = (
                np.asarray(_zd["embeddings"], dtype=np.float32),
                np.asarray(_zd["y_true"], dtype=np.int32),
                pd.read_csv(_csv_cf),
            )
        _tag_cf = f"field_{domain}_{timestamp}"
        _cf_out = run_counterfactual_audit(
            full_model=model,
            val_dataset=field_spud,
            val_rel_paths=rel_paths,
            class_names=class_names,
            metrics_dir=audit_paths["metrics"],
            attribution_dir=attribution_dir,
            name_tag=_tag_cf,
            run_id=run_id,
            split_label=f"field_{domain}",
            num_samples=counterfactual_samples,
            contrastive_bank=_cf_bank,
            contrastive_search_cap=counterfactual_contrastive_search_cap,
            contrastive_exclude=counterfactual_exclude,
            patch_size=counterfactual_patch,
            stride=counterfactual_stride,
            fill_value=counterfactual_fill,
            max_mask_patches=counterfactual_max_mask_patches,
            embedding_layer=neighbors_embedding_layer or config.NEIGHBOR_EMBEDDING_LAYER,
        )
        _cf_meta = {
            "counterfactuals_field": {
                "mask_flip_summary_csv": "",
                "contrastive_table_csv": "",
                "contrastive_ran": _cf_bank is not None,
                "samples": counterfactual_samples,
                "contrastive_exclude": counterfactual_exclude,
            }
        }
        if _cf_out.get("mask_flip_summary_csv"):
            _cf_meta["counterfactuals_field"]["mask_flip_summary_csv"] = rel_from_script(
                script_dir, _cf_out["mask_flip_summary_csv"]
            )
        if _cf_out.get("contrastive_table_csv"):
            _cf_meta["counterfactuals_field"]["contrastive_table_csv"] = rel_from_script(
                script_dir, _cf_out["contrastive_table_csv"]
            )
        merge_run_info(audit_paths["meta"], _cf_meta)
        print(
            f"[FIELD] Counterfactuals: {_cf_out.get('mask_flip_summary_csv', '')} "
            f"+ {_cf_out.get('contrastive_table_csv') or 'mask-flip only (add train NPZ/CSV for contrastive)'}"
        )


if __name__ == "__main__":
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
        default=config.TRAINING_DOMAIN,
        choices=["leaf"],
        help="Domain label for reporting only.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Root folder for processed images (e.g. processed_data_v2). Overrides config.BASE_DIR.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split subfolder to evaluate on (default: test).",
    )
    parser.add_argument(
        "--audit-run-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Optional audit folder id under outputs/fusion/audit/<id>/. "
            "Default: field_<timestamp>."
        ),
    )
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=HIGH_CONFIDENCE_WRONG_THRESHOLD_DEFAULT,
        metavar="P",
        help="Same as train_model: threshold for high_conf_wrong column (0–1).",
    )
    parser.add_argument(
        "--gradcam-target",
        type=str,
        default="predicted",
        choices=["predicted", "true"],
        help="Grad-CAM target class: predicted (argmax) or true label (error analysis).",
    )
    parser.add_argument(
        "--no-gradcam-shallow",
        action="store_true",
        help=(
            f"Omit shallow RGB Grad-CAM panel ({config.GRADCAM_RGB_SHALLOW_LAYER}), "
            "same as train_model."
        ),
    )
    parser.add_argument(
        "--occlusion",
        action="store_true",
        help="Run patch occlusion maps + metrics CSV (slow); same idea as train_model --occlusion.",
    )
    parser.add_argument(
        "--occlusion-samples",
        type=int,
        default=config.OCCLUSION_NUM_SAMPLES_DEFAULT,
        metavar="N",
        help="Field images from first batch for occlusion (default: from config).",
    )
    parser.add_argument(
        "--occlusion-patch",
        type=int,
        default=config.OCCLUSION_PATCH_DEFAULT,
        help="Occlusion patch size (pixels, default: from config).",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=config.OCCLUSION_STRIDE_DEFAULT,
        help="Occlusion stride (default: from config).",
    )
    parser.add_argument(
        "--occlusion-fill",
        type=float,
        default=config.OCCLUSION_FILL_DEFAULT,
        help="Patch fill in [0,1] (default: from config).",
    )
    parser.add_argument(
        "--occlusion-target",
        type=str,
        default="predicted",
        choices=["predicted", "true"],
        help="Occlusion explains drop in prob for predicted vs true class.",
    )
    parser.add_argument(
        "--ig",
        action="store_true",
        help="Run Integrated Gradients + optional IG vs Grad-CAM (slow); same idea as train_model --ig.",
    )
    parser.add_argument(
        "--ig-samples",
        type=int,
        default=config.IG_NUM_SAMPLES_DEFAULT,
        metavar="N",
        help="Field images from first batch for IG (default: from config).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=config.IG_M_STEPS_DEFAULT,
        metavar="M",
        help="IG Riemann steps (default: from config).",
    )
    parser.add_argument(
        "--ig-baseline",
        type=str,
        default="black",
        choices=["black"],
        help="IG baseline (fixed to black for consistency).",
    )
    parser.add_argument(
        "--ig-target",
        type=str,
        default="predicted",
        choices=["predicted", "true"],
        help="IG target class: predicted or true label.",
    )
    parser.add_argument(
        "--no-ig-gradcam-compare",
        action="store_true",
        help="Skip ig_vs_gradcam_*.png side-by-side exports.",
    )
    parser.add_argument(
        "--neighbors",
        action="store_true",
        help="k-NN vs training embeddings; requires --neighbors-train-npz and --neighbors-train-manifest-csv.",
    )
    parser.add_argument(
        "--neighbors-train-npz",
        type=str,
        default=None,
        metavar="PATH",
        help="train_embeddings_<run>.npz from a train_model --neighbors run.",
    )
    parser.add_argument(
        "--neighbors-train-manifest-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="train_manifest_<run>.csv matching the NPZ.",
    )
    parser.add_argument(
        "--neighbors-top-k",
        type=int,
        default=config.NEIGHBOR_TOP_K_DEFAULT,
        help="Neighbors per field query (default: from config).",
    )
    parser.add_argument(
        "--neighbors-samples",
        type=int,
        default=config.NEIGHBOR_FIELD_QUERY_SAMPLES_DEFAULT,
        help="Field images to query (in directory order, default: from config).",
    )
    parser.add_argument(
        "--neighbors-grid-figures",
        type=int,
        default=config.NEIGHBOR_GRID_FIGURES_DEFAULT,
        help="Max neighbor-grid PNGs (default: from config).",
    )
    parser.add_argument(
        "--neighbors-embedding-layer",
        type=str,
        default=None,
        help="Override embedding layer (default: same as training export / config).",
    )
    parser.add_argument(
        "--shortcut-audit",
        action="store_true",
        help="Data/shortcut audit: write Grad-CAM border-focus metrics CSV under metrics/.",
    )
    parser.add_argument(
        "--shortcut-audit-samples",
        type=int,
        default=config.SHORTCUT_AUDIT_SAMPLES_DEFAULT,
        metavar="N",
        help="Field images for shortcut audit (in order; default: from config).",
    )
    parser.add_argument(
        "--shortcut-border-frac",
        type=float,
        default=config.SHORTCUT_BORDER_FRAC_DEFAULT,
        metavar="F",
        help="Border thickness fraction in (0,0.49).",
    )
    parser.add_argument(
        "--counterfactuals",
        action="store_true",
        help=(
            "Phase 6: greedy mask-flip + optional contrastive k-NN when train NPZ/CSV paths are set."
        ),
    )
    parser.add_argument(
        "--counterfactual-samples",
        type=int,
        default=config.COUNTERFACTUAL_FIELD_SAMPLES_DEFAULT,
        metavar="N",
        help="Field images for counterfactuals (default: from config).",
    )
    parser.add_argument(
        "--counterfactual-contrastive-search-cap",
        type=int,
        default=config.COUNTERFACTUAL_CONTRASTIVE_SEARCH_CAP_DEFAULT,
        metavar="K",
        help="Nearest-neighbor scan cap for different-class contrastive (default: config).",
    )
    parser.add_argument(
        "--counterfactual-max-mask-patches",
        type=int,
        default=config.COUNTERFACTUAL_MASK_FLIP_MAX_PATCHES_DEFAULT,
        metavar="M",
        help="Max cumulative patches per mask-flip attempt.",
    )
    parser.add_argument(
        "--counterfactual-patch",
        type=int,
        default=config.OCCLUSION_PATCH_DEFAULT,
        help="Mask-flip patch size (pixels).",
    )
    parser.add_argument(
        "--counterfactual-stride",
        type=int,
        default=config.OCCLUSION_STRIDE_DEFAULT,
        help="Mask-flip grid stride.",
    )
    parser.add_argument(
        "--counterfactual-fill",
        type=float,
        default=config.OCCLUSION_FILL_DEFAULT,
        help="Fill [0,1] for masked regions.",
    )
    parser.add_argument(
        "--counterfactual-exclude",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Contrastive k-NN excludes train labels matching query predicted vs true class.",
    )
    args = parser.parse_args()
    if not 0.0 < args.high_conf_threshold < 1.0:
        raise SystemExit("--high-conf-threshold must be in (0, 1)")
    if args.occlusion:
        try:
            validate_occlusion_cli(
                args.occlusion_samples,
                args.occlusion_patch,
                args.occlusion_stride,
                args.occlusion_fill,
                config.DATA_PARAMS["image_size"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if args.ig:
        try:
            validate_ig_cli(args.ig_samples, args.ig_steps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if args.neighbors:
        if not args.neighbors_train_npz or not args.neighbors_train_manifest_csv:
            raise SystemExit(
                "--neighbors requires --neighbors-train-npz and --neighbors-train-manifest-csv"
            )
        try:
            validate_neighbors_cli(
                args.neighbors_top_k, args.neighbors_samples, args.neighbors_grid_figures
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if args.counterfactuals:
        try:
            validate_counterfactual_cli(
                args.counterfactual_samples,
                args.counterfactual_contrastive_search_cap,
                args.counterfactual_max_mask_patches,
                args.counterfactual_patch,
                args.counterfactual_stride,
                args.counterfactual_fill,
                config.DATA_PARAMS["image_size"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if args.shortcut_audit:
        if args.shortcut_audit_samples < 1:
            raise SystemExit("--shortcut-audit-samples must be >= 1")
        if not 0.0 < float(args.shortcut_border_frac) < 0.49:
            raise SystemExit("--shortcut-border-frac must be in (0, 0.49)")

    evaluate_on_field(
        model_path=args.model_path,
        domain=args.domain,
        audit_run_id=args.audit_run_id,
        base_dir=args.base_dir,
        split=args.split,
        high_confidence_threshold=args.high_conf_threshold,
        gradcam_target=args.gradcam_target,
        show_shallow_rgb=not args.no_gradcam_shallow,
        run_occlusion=args.occlusion,
        occlusion_samples=args.occlusion_samples,
        occlusion_patch=args.occlusion_patch,
        occlusion_stride=args.occlusion_stride,
        occlusion_fill=args.occlusion_fill,
        occlusion_target=args.occlusion_target,
        run_ig=args.ig,
        ig_samples=args.ig_samples,
        ig_steps=args.ig_steps,
        ig_baseline=args.ig_baseline,
        ig_target=args.ig_target,
        ig_gradcam_compare=not args.no_ig_gradcam_compare,
        run_neighbors=args.neighbors,
        neighbors_train_npz=args.neighbors_train_npz,
        neighbors_train_manifest_csv=args.neighbors_train_manifest_csv,
        neighbors_top_k=args.neighbors_top_k,
        neighbors_samples=args.neighbors_samples,
        neighbors_grid_figures=args.neighbors_grid_figures,
        neighbors_embedding_layer=args.neighbors_embedding_layer
        or config.NEIGHBOR_EMBEDDING_LAYER,
        run_shortcut_audit=args.shortcut_audit,
        shortcut_audit_samples=args.shortcut_audit_samples,
        shortcut_border_frac=args.shortcut_border_frac,
    )

