"""Grad-CAM overlays: RGB (config layer names) and Sobel; used by train_model and field eval."""

import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from prediction_utils import vector_to_probs as _prediction_probabilities


def _class_score_for_gradcam(predictions: tf.Tensor, class_index: int | None) -> tf.Tensor:
    """Scalar loss signal for Grad-CAM: mean score of chosen class over batch."""
    if class_index is None:
        ci = tf.argmax(predictions, axis=-1, output_type=tf.int32)
    else:
        batch = tf.shape(predictions)[0]
        ci = tf.fill([batch], tf.constant(int(class_index), dtype=tf.int32))
    oh = tf.one_hot(ci, depth=tf.shape(predictions)[-1])
    per_sample = tf.reduce_sum(predictions * oh, axis=-1)
    return tf.reduce_mean(per_sample)


def _resolve_layer_output_tensor(model: tf.keras.Model, layer_name: str) -> tf.Tensor:
    try:
        layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(f"No layer named {layer_name!r} on fused model") from exc
    return layer.output


def ensure_grad_targets(model: tf.keras.Model) -> None:
    """Attach branch conv tensors for Grad-CAM; not persisted by model.save / load_model."""
    if getattr(model, "grad_targets", None):
        return
    # For EfficientNetB0 RGB branch: the backbone is a sub-model layer; use its output tensor.
    # For Gray/Sobel: use the final conv layer output (pre-BN, consistent with layer name).
    try:
        rgb_tensor = _resolve_layer_output_tensor(model, config.GRADCAM_RGB_DEEP_LAYER)
    except ValueError:
        raise ValueError(
            f"Could not find RGB Grad-CAM layer {config.GRADCAM_RGB_DEEP_LAYER!r}. "
            "If you changed the RGB backbone, update config.GRADCAM_RGB_DEEP_LAYER."
        )
    model.grad_targets = {
        "RGB": rgb_tensor,
        "Gray": _resolve_layer_output_tensor(model, "gray_conv3_final"),
        "Sobel": _resolve_layer_output_tensor(model, "sobel_conv3_final"),
    }


def _resolve_conv_tensor(model: tf.keras.Model, branch_key: str):
    return model.grad_targets[branch_key]


def compute_gradcam(
    model: tf.keras.Model,
    input_tensors,
    branch_key: str,
    *,
    class_index: int | None = None,
    target_conv_tensor=None,
):
    """class_index None → predicted class; else explain that class. target_conv_tensor overrides branch layer."""
    if target_conv_tensor is None:
        ensure_grad_targets(model)
    target_conv_output = (
        target_conv_tensor
        if target_conv_tensor is not None
        else _resolve_conv_tensor(model, branch_key)
    )
    viz_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_conv_output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = viz_model(input_tensors)
        class_score = _class_score_for_gradcam(predictions, class_index)

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    denom = tf.math.reduce_max(heatmap)
    heatmap = tf.where(denom > 0, tf.maximum(heatmap, 0.0) / denom, tf.zeros_like(heatmap))
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.6):
    """Jet colormap heatmap over RGB."""
    heatmap_resized = cv2.resize(
        heatmap, (original_img.shape[1], original_img.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if original_img.max() <= 1.0:
        img_uint8 = np.uint8(original_img * 255)
    else:
        img_uint8 = np.uint8(original_img)

    return cv2.addWeighted(img_uint8, alpha, heatmap_color, 1 - alpha, 0)


def visualize_gradcam_batch(
    model,
    validation_data,
    save_dir,
    name_tag,
    class_names,
    num_samples: int = config.GRADCAM_NUM_SAMPLES_DEFAULT,
    *,
    gradcam_target: str = "predicted",
    show_shallow_rgb: bool = True,
    shallow_rgb_layer: str = config.GRADCAM_RGB_SHALLOW_LAYER,
):
    """gradcam_target: predicted | true; optional shallow RGB panel."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        model.get_layer(shallow_rgb_layer)
        shallow_ok = True
    except ValueError:
        shallow_ok = False

    n_top_cols = 5 if (show_shallow_rgb and shallow_ok) else 4

    for (rgb_batch, gray_batch, sobel_batch), labels in validation_data.take(1):
        batch_size = rgb_batch.shape[0]
        num_to_visualize = min(num_samples, batch_size)

        for i in range(num_to_visualize):
            input_tensors = (
                rgb_batch[i : i + 1],
                gray_batch[i : i + 1],
                sobel_batch[i : i + 1],
            )
            original_rgb = rgb_batch[i].numpy()
            original_gray = gray_batch[i].numpy()  # (H, W, 1)
            gray_rgb = np.repeat(original_gray, 3, axis=-1)  # convert to 3-ch for overlay

            true_idx = int(labels[i].numpy())
            preds = model.predict(input_tensors, verbose=0)
            probs = _prediction_probabilities(preds[0])
            pred_idx = int(np.argmax(probs))

            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]
            top_pct = 100.0 * probs[pred_idx]

            if gradcam_target == "true":
                cam_class = true_idx
            else:
                cam_class = None

            rgb_heatmap_deep = compute_gradcam(
                model,
                input_tensors,
                "RGB",
                class_index=cam_class,
            )

            rgb_heatmap_shallow = None
            if show_shallow_rgb and shallow_ok:
                try:
                    shallow_t = _resolve_layer_output_tensor(model, shallow_rgb_layer)
                    rgb_heatmap_shallow = compute_gradcam(
                        model,
                        input_tensors,
                        "RGB",
                        class_index=cam_class,
                        target_conv_tensor=shallow_t,
                    )
                except (ValueError, KeyError):
                    rgb_heatmap_shallow = None

            gray_heatmap = compute_gradcam(
                model, input_tensors, "Gray", class_index=cam_class
            )
            sobel_heatmap = compute_gradcam(
                model, input_tensors, "Sobel", class_index=cam_class
            )

            rgb_viz_deep = overlay_heatmap(rgb_heatmap_deep, original_rgb)
            gray_viz = overlay_heatmap(gray_heatmap, gray_rgb)
            sobel_viz = overlay_heatmap(sobel_heatmap, original_rgb)

            fig = plt.figure(figsize=(20 if n_top_cols == 5 else 18, 10))
            gs = gridspec.GridSpec(
                2, n_top_cols, figure=fig, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.22
            )
            tgt_note = (
                f"Grad-CAM target: {gradcam_target} class"
                if gradcam_target == "true"
                else "Grad-CAM target: predicted class"
            )
            fig.suptitle(
                f"Grad-CAM | Actual: {true_label} | Predicted: {pred_label} ({top_pct:.1f}%)\n{tgt_note}",
                fontsize=13,
                fontweight="bold",
                y=0.99,
            )

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(original_rgb)
            ax0.axis("off")
            ax0.set_title("Original RGB")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(cv2.cvtColor(rgb_viz_deep, cv2.COLOR_BGR2RGB))
            ax1.axis("off")
            ax1.set_title(f"RGB focus (deep: {config.GRADCAM_RGB_DEEP_LAYER})")

            col_gray = 2
            if rgb_heatmap_shallow is not None:
                ax_mid = fig.add_subplot(gs[0, 2])
                ax_mid.imshow(
                    cv2.cvtColor(
                        overlay_heatmap(rgb_heatmap_shallow, original_rgb), cv2.COLOR_BGR2RGB
                    )
                )
                ax_mid.axis("off")
                ax_mid.set_title(f"RGB focus (shallow: {shallow_rgb_layer})")
                col_gray = 3

            ax_g = fig.add_subplot(gs[0, col_gray])
            ax_g.imshow(cv2.cvtColor(gray_viz, cv2.COLOR_BGR2RGB))
            ax_g.axis("off")
            ax_g.set_title("Grayscale branch focus")

            ax_s = fig.add_subplot(gs[0, col_gray + 1])
            ax_s.imshow(cv2.cvtColor(sobel_viz, cv2.COLOR_BGR2RGB))
            ax_s.axis("off")
            ax_s.set_title("Sobel branch focus")

            ax_bar = fig.add_subplot(gs[1, :])
            order = np.argsort(probs)[::-1]
            pct = 100.0 * probs[order]
            names_ord = [class_names[j] for j in order]
            y_pos = np.arange(len(names_ord))
            colors = ["#c44e52" if j == pred_idx else "#4c72b0" for j in order]
            ax_bar.barh(y_pos, pct, color=colors, edgecolor="none")
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(names_ord, fontsize=9)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Model probability (%)")
            ax_bar.set_xlim(0, max(105.0, float(np.max(pct)) + 5.0))
            ax_bar.set_title("All class probabilities (sorted, highest at top)")
            ax_bar.grid(axis="x", alpha=0.3)

            save_path = os.path.join(save_dir, f"fusion_gradcam_{name_tag}_{i}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        break


def explain_my_model(
    model,
    validation_data,
    save_dir,
    name_tag,
    class_names,
    num_samples: int = config.GRADCAM_NUM_SAMPLES_DEFAULT,
    gradcam_target: str = "predicted",
    show_shallow_rgb: bool = True,
):
    """Wrapper for train_model: Grad-CAM panels + prob bar chart."""
    return visualize_gradcam_batch(
        model=model,
        validation_data=validation_data,
        save_dir=save_dir,
        name_tag=name_tag,
        class_names=class_names,
        num_samples=num_samples,
        gradcam_target=gradcam_target,
        show_shallow_rgb=show_shallow_rgb,
    )
