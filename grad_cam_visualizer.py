# GRAD CAM 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Sub model for activatioin for conv. layer and pred.

def compute_gradcam(model, img_tuple, branch_key):
    target_output = model.grad_targets[branch_key]

    viz_model = tf.keras.models.Model(
        inputs=  model.inputs,
        outputs = [target_output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = viz_model(img_tuple)

        # best predic
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # weight of each
    pooled_grads = tf.reduce_mean(grads, axis =(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Resize
def overlay_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_uint8 = np.uint8(original_img * 255) if original_img.max() <= 1.0 else np.uint8(original_img)
    return cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)

# Comparisons
def explain_my_model(train_model, validation_data, save_dir, name_tag, class_names, num_samp = 3):

    for (rgb_b, gray_b, sobel_b), labels in validation_data.take(1):   
        for i in range(num_samp):     
            img_tuple = (rgb_b[i:i+1], gray_b[i:i+1], sobel_b[i:i+1])
            original_rgb = rgb_b[i].numpy()

        true_idx = labels[i].numpy()
        preds = train_model.predict(img_tuple, verbose = 0)
        pred_idx = np.argmax(preds[0])

        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]

        # heatmap output
        rgb_h = compute_gradcam(train_model, img_tuple, "RGB")
        sobel_h = compute_gradcam(train_model, img_tuple, "Sobel")

        rgb_viz = overlay_heatmap(rgb_h, original_rgb)
        sobel_viz = overlay_heatmap(sobel_h, original_rgb)

        plt.figure(figsize = (15, 6))
        plt.suptitle(f"Grad-CAM Analysis Actual: {true_label} | Predicted: {pred_label}", 
                     fontsize = 14, fontweight = 'bold', y = 1.05)
        plt.subplot(1, 3, 1)
        plt.imshow(original_rgb)
        plt.title("Original RGB")
    
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_viz)
        plt.title("RGB Branch Focus")
    
        plt.subplot(1, 3, 3)
        plt.imshow(sobel_viz)
        plt.title("Sobel (Edge) Branch Focus")
    
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"fusion_gradcam_{name_tag}.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()
        break