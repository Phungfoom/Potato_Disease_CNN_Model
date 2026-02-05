# 1. Dual-Branch structure (RGB)
import tensorflow as tf

def build_rgb_model(input_shape = (224, 224, 3), num_classes = 3): # 3 sheets
    
    inputs = tf.keras.Input(shape = input_shape, 
                            name = "rgb_input")
    
    # Features: What is the model looking for?
    # (a) Block 1: Detects Basics
    #  Groups of pixels (3x3), colors, edges, contrasts

    x = tf.keras.layers.Conv2D(32, (3, 3), # 32 filters, 3x3 for details
                               activation = 'relu', # adding bends, positive signals to next layer
                               padding = 'same', # output size = input size
                               name = 'rgb_conv1')(inputs) # for grad-CAM to find later
    
    x = tf.keras.layers.MaxPooling2D((2, 2), name = 'rgb_pool1')(x)

    # (b) Block 2: Dectect Patterns
    # Shape and Texture Detection

    x = tf.keras.layers.Conv2D(64, (3, 3), # more feature layers for complexity
                               activation = 'relu', 
                               padding = 'same')(x)
    
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # (c) Block 3: Detects Concepts
    # Combines shapes/textures

    x = tf.keras.layers.Conv2D(128, (3, 3), 
                               activation = 'relu', 
                               padding = 'same')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # reduces paramter count

    # Classifier

    x = tf.keras.layers.Dense(64, 
                              activation = 'relu', 
                              name = 'rgb_dense')(x)
    
    x = tf.keras.layers.Dropout(0.3)(x) # tunes off 30% of neurons every training step

    # Output
    outputs = tf.keras.layers.Dense(num_classes, 
                                    activation = 'softmax', # probabilities 
                                    name = 'prediction')(x)

    # Create Model 

    model = tf.keras.models.Model(inputs = inputs, 
                                  outputs = outputs, 
                                  name = 'RGB_Specialists')

    # lower error 
    model.compile(
        optimizer = 'adam', # keep momentum/step size (convergence)
        loss = 'sparse_categorical_crossentropy', # grade model 
        metrics = ['accuracy']
    )
    return model 

