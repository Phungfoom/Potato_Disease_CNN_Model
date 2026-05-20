"""Standard layout for transparency / audit outputs under outputs/fusion/audit/<run_id>/."""

from __future__ import annotations

import json
import os
from typing import Any

import config


def audit_run_root(script_dir: str, run_id: str) -> str:
    """Absolute path: <project>/outputs/fusion/audit/<run_id>/."""
    return os.path.join(
        script_dir, config.OUTPUT_DIR, *config.AUDIT_PATH_SEGMENTS, run_id
    )


def validate_occlusion_cli(
    samples: int,
    patch: int,
    stride: int,
    fill: float,
    image_size: tuple[int, int],
) -> None:
    """Raise ValueError if occlusion flags are invalid (train + field share this)."""
    ih, iw = image_size
    if samples < 1:
        raise ValueError("--occlusion-samples must be >= 1")
    if patch < 1 or stride < 1:
        raise ValueError("--occlusion-patch and --occlusion-stride must be >= 1")
    if patch > ih or patch > iw:
        raise ValueError("--occlusion-patch must be <= image size from config")
    if not 0.0 <= fill <= 1.0:
        raise ValueError("--occlusion-fill must be in [0, 1]")


def validate_ig_cli(samples: int, steps: int) -> None:
    """Raise ValueError if Integrated Gradients flags are invalid."""
    if samples < 1:
        raise ValueError("--ig-samples must be >= 1")
    if steps < 1:
        raise ValueError("--ig-steps must be >= 1")


def validate_counterfactual_cli(
    samples: int,
    search_cap: int,
    max_mask_patches: int,
    patch: int,
    stride: int,
    fill: float,
    image_size: tuple[int, int],
) -> None:
    if samples < 1:
        raise ValueError("--counterfactual-samples must be >= 1")
    if search_cap < 1:
        raise ValueError("--counterfactual-contrastive-search-cap must be >= 1")
    if max_mask_patches < 1:
        raise ValueError("--counterfactual-max-mask-patches must be >= 1")
    validate_occlusion_cli(1, patch, stride, fill, image_size)


def validate_neighbors_cli(top_k: int, query_samples: int, grid_figures: int) -> None:
    if top_k < 1:
        raise ValueError("--neighbors-top-k must be >= 1")
    if query_samples < 1:
        raise ValueError("--neighbors-val-samples / --neighbors-samples must be >= 1")
    if grid_figures < 0:
        raise ValueError("--neighbors-grid-figures must be >= 0")


def ensure_audit_run(script_dir: str, run_id: str) -> dict[str, str]:
    """
    Create outputs/.../audit/<run_id>/ with standard subfolders:
    predictions (master CSV), metrics (tables), attribution (maps/Grad-CAM),
    neighbors (Phase 5), failures (Phase 7 gallery PNGs), meta (run_info.json).
    """
    root = audit_run_root(script_dir, run_id)
    paths = {
        "root": root,
        "predictions": os.path.join(root, "predictions"),
        "metrics": os.path.join(root, "metrics"),
        "attribution": os.path.join(root, "attribution"),
        "neighbors": os.path.join(root, "neighbors"),
        "failures": os.path.join(root, "failures"),
        "meta": os.path.join(root, "meta"),
    }
    for key in ("predictions", "metrics", "attribution", "neighbors", "failures", "meta"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def write_run_info(meta_dir: str, payload: dict[str, Any]) -> str:
    path = os.path.join(meta_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def merge_run_info(meta_dir: str, updates: dict[str, Any]) -> str:
    """Shallow-merge keys into run_info.json (creates file if missing)."""
    path = os.path.join(meta_dir, "run_info.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    else:
        data = {}
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def rel_from_script(script_dir: str, abs_path: str) -> str:
    """Path relative to project root, POSIX-style."""
    return os.path.relpath(abs_path, script_dir).replace("\\", "/")


def append_runs_manifest(script_dir: str, row: dict[str, Any]) -> str:
    """Append one summary row to outputs/fusion/runs_manifest.csv.

    Creates the file with a header on first call; appends on subsequent calls.
    Each row captures the core metrics and settings for one training run so
    runs can be compared without opening individual audit folders.
    """
    import csv

    manifest_path = os.path.join(
        script_dir, config.OUTPUT_DIR, config.AUDIT_PATH_SEGMENTS[0], "runs_manifest.csv"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    write_header = not os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return manifest_path
