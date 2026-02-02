# 1. Dual-Branch structure (RGB/Grayscale)
import os 
import tensorflow as tf
import keras 

def build_rgb_model(input_shape = (224, 224, 3), num_classes = 3):
    
    inputs = tf.keras.Input(shape = input_shape, name = "rgb_input")
    
    # Features: What is the model looking for?
    # (a) Block 1: Detects Basics
    #  Groups of pixels (3x3), colors, edges, contrasts

    x = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'rgb_conv1')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), name = 'rgb_pool1')(x)

    # (b) Block 2: Dectect Patterns
    # Shape and Texture Detection

    # (c) Block 3: Detects Concepts
    # Combines shapes/textures
