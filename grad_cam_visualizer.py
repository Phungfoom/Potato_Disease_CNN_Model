# GRAD CAM 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
            pred_idx = int(np.argmax(preds[0]))

            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]

            # Compute heatmaps for the RGB and Sobel branches
            rgb_heatmap = compute_gradcam(model, input_tensors, "RGB")
            sobel_heatmap = compute_gradcam(model, input_tensors, "Sobel")

            rgb_viz = overlay_heatmap(rgb_heatmap, original_rgb)
            sobel_viz = overlay_heatmap(sobel_heatmap, original_rgb)

            plt.figure(figsize=(15, 6))
            plt.suptitle(
                f"Grad-CAM Analysis | Actual: {true_label} | Predicted: {pred_label}",
                fontsize=14,
                fontweight="bold",
                y=1.05,
            )

            plt.subplot(1, 3, 1)
            plt.imshow(original_rgb)
            plt.axis("off")
            plt.title("Original RGB")

            plt.subplot(1, 3, 2)
            plt.imshow(rgb_viz)
            plt.axis("off")
            plt.title("RGB Branch Focus")

            plt.subplot(1, 3, 3)
            plt.imshow(sobel_viz)
            plt.axis("off")
            plt.title("Sobel (Edge) Branch Focus")

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"fusion_gradcam_{name_tag}_{i}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        # Only process the first batch
        break