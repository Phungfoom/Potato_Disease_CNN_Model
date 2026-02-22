# GRAD CAM 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_path = 'potato_grayscale_model.h5'
test_image_path = r'C:\Users\phung\Documents\potato_id\hot_potato\blackleg\2.jpg'

last_conv_layer = 'gray_conv3'

# Maps image to last conv layer as well as output predictions 
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = None):
    # Ouput feature maps and final predictions score
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    # Compute gradient of target class score
    with tf.GradientTape() as tape: 
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index] # score of winning classification

        grads = tape.gradient(class_channel, last_conv_layer_output) # important pixels
        pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2)) # gives a score of importance 

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # feature map x importance weight = grid of important visuals
        heatmap = tf.squeeze(heatmap)
       
       # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # RelU, deletes negatives.
        return heatmap.numpy()
        