# GRAD CAM 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from prediction_utils import vector_to_probs as _prediction_probabilities

def compute_gradcam(model,
                    input_tensors,
                    branch_key):
    """Compute a Grad-CAM heatmap for a single branch of the model."""

    # Submodel that outputs both the chosen conv feature map and final predictions
    target_conv_output = model.grad_targets[branch_key]
    viz_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_conv_output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = viz_model(input_tensors)

        # Use the top predicted class for Grad-CAM
        top_class_index = tf.argmax(predictions[0])
        top_class_score = predictions[:, top_class_index]

    grads = tape.gradient(top_class_score, conv_outputs)

    # Channel-wise importance weights (global average pooling over H, W, batch)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv activations by importance and collapse channels
    conv_outputs = conv_outputs[0]  # remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap,
                    original_img,
                    alpha = 0.6):
    """Overlay a Grad-CAM heatmap onto the original RGB image."""
    # Match heatmap size to original image
    heatmap_resized = cv2.resize(
        heatmap, (original_img.shape[1], original_img.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Ensure the original image is uint8 in [0, 255]
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
    num_samples = 3,
):
    """Visualize Grad-CAM results for a few samples from the validation set."""
    os.makedirs(save_dir, exist_ok = True)

    # We only need the first batch for visualization
    for (rgb_batch, gray_batch, sobel_batch), labels in validation_data.take(1):
        batch_size = rgb_batch.shape[0]
        num_to_visualize = min(num_samples, batch_size)

        for i in range(num_to_visualize):
            # Build a "batch of 1" input for the model
            input_tensors = (
                rgb_batch[i : i + 1],
                gray_batch[i : i + 1],
                sobel_batch[i : i + 1],
            )
            original_rgb = rgb_batch[i].numpy()

            true_idx = int(labels[i].numpy())
            preds = model.predict(input_tensors, verbose=0)
            probs = _prediction_probabilities(preds[0])
            pred_idx = int(np.argmax(probs))

            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]
            top_pct = 100.0 * probs[pred_idx]

            # Compute heatmaps for the RGB and Sobel branches
            rgb_heatmap = compute_gradcam(model, input_tensors, "RGB")
            sobel_heatmap = compute_gradcam(model, input_tensors, "Sobel")

            rgb_viz = overlay_heatmap(rgb_heatmap, original_rgb)
            sobel_viz = overlay_heatmap(sobel_heatmap, original_rgb)

            # Top row: images; bottom row: full class probability breakdown
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(
                2, 3, figure=fig, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25
            )
            fig.suptitle(
                f"Grad-CAM | Actual: {true_label} | Predicted: {pred_label} ({top_pct:.1f}%)",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(original_rgb)
            ax0.axis("off")
            ax0.set_title("Original RGB")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB))
            ax1.axis("off")
            ax1.set_title("RGB Branch Focus")

            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(cv2.cvtColor(sobel_viz, cv2.COLOR_BGR2RGB))
            ax2.axis("off")
            ax2.set_title("Sobel (Edge) Branch Focus")

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

        # Only process the first batch
        break


def explain_my_model(
    model,
    validation_data,
    save_dir,
    name_tag,
    class_names,
    num_samples: int = 3,
):
    """
    Backwards-compatible entrypoint used by `train_model.py`.
    Produces Grad-CAM overlays for a small batch of samples.
    """
    return visualize_gradcam_batch(
        model=model,
        validation_data=validation_data,
        save_dir=save_dir,
        name_tag=name_tag,
        class_names=class_names,
        num_samples=num_samples,
    )