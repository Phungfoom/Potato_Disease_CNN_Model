"""Phase 5 — KNN and reports: L2-normalized fused-layer embeddings, k-NN, CSV tables, grid PNGs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import config
from prediction_utils import batch_preds_to_probs


def build_embedding_model(
    full_model: tf.keras.Model, layer_name: str | None = None
) -> tf.keras.Model:
    """Map triple inputs → L2-normalized activations at ``layer_name`` (default: config)."""
    name = layer_name or config.NEIGHBOR_EMBEDDING_LAYER
    layer = full_model.get_layer(name)
    raw = layer.output
    normed = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(tf.cast(x, tf.float32), axis=-1),
        name="l2_normalize",
    )(raw)
    return tf.keras.Model(
        inputs=full_model.inputs, outputs=normed, name="embedding_extractor"
    )


def _predict_embed_batch(embed_model: tf.keras.Model, images) -> np.ndarray:
    out = embed_model.predict_on_batch(images)
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out, dtype=np.float32)


def export_train_embeddings(
    embed_model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    train_rel_paths: list[str],
    *,
    npz_path: str,
    manifest_csv_path: str,
    class_names: list[str],
) -> tuple[int, int]:
    """
    Write float32 [N,D] embeddings + int32 labels to npz; paths + names in CSV.
    Returns (n_samples, dim).
    """
    emb_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for images, labels in tqdm(train_dataset, desc="Train embeddings"):
        e = _predict_embed_batch(embed_model, images)
        emb_chunks.append(e)
        y_chunks.append(np.asarray(labels.numpy(), dtype=np.int32))

    if not emb_chunks:
        raise ValueError("train_dataset produced no batches for embedding export")
    embeddings = np.vstack(emb_chunks).astype(np.float32)
    y_true = np.concatenate(y_chunks, axis=0)
    n, dim = embeddings.shape

    if n != len(train_rel_paths):
        raise ValueError(
            f"train_rel_paths ({len(train_rel_paths)}) != embedded rows ({n})"
        )

    for p in (npz_path, manifest_csv_path):
        parent = os.path.dirname(os.path.abspath(p))
        if parent:
            os.makedirs(parent, exist_ok=True)
    np.savez_compressed(npz_path, embeddings=embeddings, y_true=y_true)

    rows = [
        {
            "sample_index": i,
            "image_rel_path": train_rel_paths[i],
            "y_true": int(y_true[i]),
            "true_class": class_names[int(y_true[i])],
        }
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(manifest_csv_path, index=False)
    return n, dim


def _cosine_sim_from_l2_dist(dist: np.ndarray) -> np.ndarray:
    """Unit vectors: ||a-b||^2 = 2 - 2 cos θ → cos = 1 - d^2/2."""
    return np.clip(1.0 - (dist**2) / 2.0, -1.0, 1.0)


def _gather_queries(
    embed_model: tf.keras.Model,
    classify_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    rel_paths: list[str],
    max_queries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """First ``max_queries`` samples from ``dataset`` (in order)."""
    q_emb: list[np.ndarray] = []
    q_y: list[int] = []
    q_pred: list[int] = []
    q_paths: list[str] = []
    taken = 0
    for images, labels in dataset:
        bs = int(images[0].shape[0])
        e = _predict_embed_batch(embed_model, images)
        pv = classify_model.predict_on_batch(images)
        p_np = pv.numpy() if hasattr(pv, "numpy") else np.asarray(pv)
        probs = batch_preds_to_probs(p_np)
        pred = np.argmax(probs, axis=1)
        lab = labels.numpy()
        for i in range(bs):
            if taken >= max_queries:
                break
            if taken < len(rel_paths):
                q_paths.append(rel_paths[taken])
            else:
                q_paths.append("")
            q_emb.append(e[i])
            q_y.append(int(lab[i]))
            q_pred.append(int(pred[i]))
            taken += 1
        if taken >= max_queries:
            break
    if not q_emb:
        raise ValueError("No query samples collected for neighbor search")
    return (
        np.stack(q_emb, axis=0),
        np.asarray(q_y, dtype=np.int32),
        np.asarray(q_pred, dtype=np.int32),
        q_paths,
    )


def build_neighbor_index(train_embeddings: np.ndarray, top_k: int) -> NearestNeighbors:
    nn = NearestNeighbors(
        n_neighbors=min(top_k, len(train_embeddings)),
        algorithm="brute",
        metric="euclidean",
    )
    nn.fit(train_embeddings)
    return nn


def run_val_neighbor_audit(
    *,
    full_model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    val_rel_paths: list[str],
    train_embeddings: np.ndarray,
    train_manifest_df: pd.DataFrame,
    class_names: list[str],
    neighbors_dir: str,
    name_tag: str,
    run_id: str,
    split_label: str,
    top_k: int,
    max_queries: int,
    max_grid_figures: int,
    embedding_layer: str,
    project_root: str,
    query_path_prefix: str | None = None,
) -> tuple[str | None, str | None]:
    """
    k-NN from normalized train bank; CSV table + up to ``max_grid_figures`` PNG grids.
    Returns (table_csv_path, None) or (path, path) — second reserved for future.
    """
    embed_model = build_embedding_model(full_model, embedding_layer)
    top_k = min(top_k, len(train_embeddings))
    if top_k < 1:
        raise ValueError("top_k must be >= 1 and train set non-empty")
    if len(train_manifest_df) != len(train_embeddings):
        raise ValueError("train manifest length must match embedding row count")

    nn = build_neighbor_index(train_embeddings, top_k)
    q_emb, q_y, q_pred, q_paths = _gather_queries(
        embed_model,
        full_model,
        val_dataset,
        val_rel_paths,
        max_queries,
    )
    dist, idx = nn.kneighbors(q_emb, n_neighbors=top_k)
    sim = _cosine_sim_from_l2_dist(dist)

    table_rows: list[dict] = []
    for qi in range(len(q_emb)):
        for rank in range(top_k):
            ni = int(idx[qi, rank])
            row = train_manifest_df.iloc[ni]
            table_rows.append(
                {
                    "run_id": run_id,
                    "split": split_label,
                    "query_index": qi,
                    "query_image_rel_path": q_paths[qi] if qi < len(q_paths) else "",
                    "query_y_true": int(q_y[qi]),
                    "query_y_pred": int(q_pred[qi]),
                    "query_true_class": class_names[int(q_y[qi])],
                    "query_pred_class": class_names[int(q_pred[qi])],
                    "neighbor_rank": rank + 1,
                    "neighbor_train_index": ni,
                    "neighbor_image_rel_path": row["image_rel_path"],
                    "neighbor_y_true": int(row["y_true"]),
                    "neighbor_class": row["true_class"],
                    "cosine_similarity": float(sim[qi, rank]),
                }
            )

    os.makedirs(neighbors_dir, exist_ok=True)
    table_csv = os.path.join(
        neighbors_dir, f"neighbor_table_{split_label}_{name_tag}.csv"
    )
    pd.DataFrame(table_rows).to_csv(table_csv, index=False)

    def _load_rgb_train(path_rel: str) -> np.ndarray:
        p = os.path.join(project_root, path_rel.replace("/", os.sep))
        if not os.path.isfile(p):
            return np.zeros((64, 64, 3), dtype=np.float32)
        img = tf.keras.utils.load_img(p, target_size=config.DATA_PARAMS["image_size"])
        return np.asarray(tf.keras.utils.img_to_array(img), dtype=np.float32) / 255.0

    def _load_rgb_query(path_rel: str) -> np.ndarray:
        root = query_path_prefix if query_path_prefix is not None else project_root
        p = os.path.join(root, path_rel.replace("/", os.sep))
        if not os.path.isfile(p):
            return np.zeros((64, 64, 3), dtype=np.float32)
        img = tf.keras.utils.load_img(p, target_size=config.DATA_PARAMS["image_size"])
        return np.asarray(tf.keras.utils.img_to_array(img), dtype=np.float32) / 255.0

    n_grids = min(max_grid_figures, len(q_emb))
    for qi in range(n_grids):
        fig, axes = plt.subplots(1, 1 + top_k, figsize=(3 * (1 + top_k), 3))
        qim = (
            _load_rgb_query(q_paths[qi])
            if qi < len(q_paths)
            else np.zeros((64, 64, 3), dtype=np.float32)
        )
        axes[0].imshow(np.clip(qim, 0, 1))
        axes[0].set_title(f"Query\n{class_names[int(q_pred[qi])]}")
        axes[0].axis("off")
        for r in range(top_k):
            ni = int(idx[qi, r])
            rel = str(train_manifest_df.iloc[ni]["image_rel_path"])
            nim = _load_rgb_train(rel)
            axes[r + 1].imshow(np.clip(nim, 0, 1))
            axes[r + 1].set_title(
                f"#{r + 1} sim={sim[qi, r]:.3f}\n{train_manifest_df.iloc[ni]['true_class']}"
            )
            axes[r + 1].axis("off")
        fig.suptitle(f"k-NN ({embedding_layer}) | {split_label} query {qi}", fontsize=10)
        plt.tight_layout()
        gpath = os.path.join(
            neighbors_dir, f"neighbor_grid_{split_label}_{name_tag}_q{qi}.png"
        )
        plt.savefig(gpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return table_csv, None


def run_train_neighbor_pipeline(
    *,
    full_model: tf.keras.Model,
    train_dataset_eval: tf.data.Dataset,
    train_rel_paths: list[str],
    val_dataset: tf.data.Dataset,
    val_rel_paths: list[str],
    class_names: list[str],
    neighbors_dir: str,
    name_tag: str,
    run_id: str,
    split_label_val: str,
    top_k: int,
    val_query_samples: int,
    max_grid_figures: int,
    project_root: str,
    embedding_layer: str | None = None,
) -> dict[str, str]:
    """Export train bank + val neighbor table/grids. Returns paths for run_info."""
    layer = embedding_layer or config.NEIGHBOR_EMBEDDING_LAYER
    embed_model = build_embedding_model(full_model, layer)
    npz_path = os.path.join(neighbors_dir, f"train_embeddings_{name_tag}.npz")
    manifest_path = os.path.join(neighbors_dir, f"train_manifest_{name_tag}.csv")
    export_train_embeddings(
        embed_model,
        train_dataset_eval,
        train_rel_paths,
        npz_path=npz_path,
        manifest_csv_path=manifest_path,
        class_names=class_names,
    )
    data = np.load(npz_path)
    train_emb = np.asarray(data["embeddings"], dtype=np.float32)
    manifest_df = pd.read_csv(manifest_path)
    table_csv, _ = run_val_neighbor_audit(
        full_model=full_model,
        val_dataset=val_dataset,
        val_rel_paths=val_rel_paths,
        train_embeddings=train_emb,
        train_manifest_df=manifest_df,
        class_names=class_names,
        neighbors_dir=neighbors_dir,
        name_tag=name_tag,
        run_id=run_id,
        split_label=split_label_val,
        top_k=top_k,
        max_queries=val_query_samples,
        max_grid_figures=max_grid_figures,
        embedding_layer=layer,
        project_root=project_root,
        query_path_prefix=None,
    )
    results = {
        "train_embeddings_npz": npz_path,
        "train_manifest_csv": manifest_path,
        "val_neighbor_table_csv": table_csv or "",
        "embedding_layer": layer,
    }

    # Per-branch banks: RGB, Grayscale, Sobel — no retraining needed,
    # just re-extract from each branch's pooling layer.
    branch_layers = {
        "rgb":   "rgb_global_pool",
        "gray":  "gray_global_pool",
        "sobel": "sobel_global_pool",
    }
    for branch, blayer in branch_layers.items():
        try:
            full_model.get_layer(blayer)
        except ValueError:
            print(f"[neighbors] Branch layer '{blayer}' not found — skipping {branch}")
            continue

        b_npz  = os.path.join(neighbors_dir, f"train_embeddings_{name_tag}_{branch}.npz")
        b_mani = os.path.join(neighbors_dir, f"train_manifest_{name_tag}_{branch}.csv")
        b_embed = build_embedding_model(full_model, blayer)
        export_train_embeddings(
            b_embed, train_dataset_eval, train_rel_paths,
            npz_path=b_npz, manifest_csv_path=b_mani, class_names=class_names,
        )
        b_data = np.load(b_npz)
        b_table, _ = run_val_neighbor_audit(
            full_model=full_model,
            val_dataset=val_dataset,
            val_rel_paths=val_rel_paths,
            train_embeddings=np.asarray(b_data["embeddings"], dtype=np.float32),
            train_manifest_df=pd.read_csv(b_mani),
            class_names=class_names,
            neighbors_dir=neighbors_dir,
            name_tag=f"{name_tag}_{branch}",
            run_id=run_id,
            split_label=split_label_val,
            top_k=top_k,
            max_queries=val_query_samples,
            max_grid_figures=max_grid_figures,
            embedding_layer=blayer,
            project_root=project_root,
            query_path_prefix=None,
        )
        results[f"val_neighbor_table_{branch}_csv"] = b_table or ""
        print(f"[neighbors] {branch} branch bank done → {b_table}")

    return results


def run_field_neighbor_pipeline(
    *,
    full_model: tf.keras.Model,
    field_dataset: tf.data.Dataset,
    field_rel_paths: list[str],
    field_images_dir: str,
    train_embeddings_npz: str,
    train_manifest_csv: str,
    class_names: list[str],
    neighbors_dir: str,
    name_tag: str,
    run_id: str,
    top_k: int,
    max_queries: int,
    max_grid_figures: int,
    project_root: str,
    embedding_layer: str | None = None,
) -> dict[str, str]:
    """Field queries against a precomputed training embedding bank."""
    if not os.path.isfile(train_embeddings_npz):
        raise FileNotFoundError(f"train embeddings not found: {train_embeddings_npz}")
    if not os.path.isfile(train_manifest_csv):
        raise FileNotFoundError(f"train manifest not found: {train_manifest_csv}")
    layer = embedding_layer or config.NEIGHBOR_EMBEDDING_LAYER
    data = np.load(train_embeddings_npz)
    train_emb = np.asarray(data["embeddings"], dtype=np.float32)
    manifest_df = pd.read_csv(train_manifest_csv)
    if len(manifest_df) != len(train_emb):
        raise ValueError(
            f"manifest rows ({len(manifest_df)}) != embeddings rows ({len(train_emb)})"
        )

    em = build_embedding_model(full_model, layer)
    for images, _ in field_dataset.take(1):
        t = _predict_embed_batch(em, images)
        if t.shape[-1] != train_emb.shape[-1]:
            raise ValueError(
                f"Embedding dim mismatch: model {t.shape[-1]} vs bank {train_emb.shape[-1]}"
            )
        break

    table_csv, _ = run_val_neighbor_audit(
        full_model=full_model,
        val_dataset=field_dataset,
        val_rel_paths=field_rel_paths,
        train_embeddings=train_emb,
        train_manifest_df=manifest_df,
        class_names=class_names,
        neighbors_dir=neighbors_dir,
        name_tag=name_tag,
        run_id=run_id,
        split_label="field",
        top_k=top_k,
        max_queries=max_queries,
        max_grid_figures=max_grid_figures,
        embedding_layer=layer,
        project_root=project_root,
        query_path_prefix=field_images_dir,
    )
    return {
        "field_neighbor_table_csv": table_csv or "",
        "embedding_layer": layer,
    }
