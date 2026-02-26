import config
import tensorflow as tf
import matplotlib.pyplot as plt

from build_rgb_model import build_rgb_model
from build_grayscale_model import build_grayscale_model
from build_sobel_model import build_sobel_model


def build_combined_model(num_classes = config.NUM_CLASSES):
    
    img_size = config.DATA_PARAMS['image_size']

    rgb_branch = build_rgb_model(input_shape=(*img_size, 3))
    gray_branch = build_grayscale_model(input_shape=(*img_size, 1))
    sobel_branch = build_sobel_model(input_shape=(*img_size, 1))

    # 1 combined vector for features 
    merged = tf.keras.layers.Concatenate(name = "feature_combined")([
        rgb_branch.get_layer("rgb_global_pool").output,
        gray_branch.get_layer("gray_global_pool").output,
        sobel_branch.get_layer("sobel_global_pool").output])

    
    # Decision processor
    x = tf.keras.layers.Dense(units = 128, 
                              activation = 'relu', 
                              name = "fusion_dense_1")(merged)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units = 64, 
                              activation = 'relu', 
                              name = "fusion_dense_2")(x)
    
    output = tf.keras.layers.Dense(num_classes, 
                                   activation = 'softmax', 
                                   name = "final_prediction")(x)
    
    full_model = tf.keras.models.Model(
        inputs = [rgb_branch.input, gray_branch.input, sobel_branch.input],
        outputs = output,
        name = "Potato_Disease_Bunch")

    full_model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])
    
    return full_model