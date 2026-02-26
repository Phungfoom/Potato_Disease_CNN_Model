# 1. Dual-Branch structure (Grayscale)

import tensorflow as tf

def build_grayscale_model(input_shape = (224, 224, 1), num_classes = 3): # 1 sheet
    
    inputs = tf.keras.Input(shape = input_shape, name = "gray_input")
    
    # low 
    # (a) Texture & Edges
    x = tf.keras.layers.Conv2D(filters = 32, 
                               kernel_size = (3, 3), 
                               activation = 'relu', 
                               padding = 'same', 
                               name = 'gray_conv1')(inputs)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), name = 'gray_pool1')(x)

    # middle
    # (b) Shapes & Patterns
    x = tf.keras.layers.Conv2D(64, (3, 3), 
                               activation = 'relu', 
                               padding = 'same', 
                               name = 'gray_conv2')(x)
    
    x = tf.keras.layers.MaxPooling2D((2,2), name = 'gray_pool2')(x)

    # high
    # (c) Complex Geometry (The "Leaf Roll" detector)
    x = tf.keras.layers.Conv2D(128, (3, 3), 
                               activation = 'relu', 
                               padding = 'same', 
                               name = 'gray_conv3_final')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D(name = 'gray_global_pool')(x) 

    # Classifier
    x = tf.keras.layers.Dense(64, activation='relu', name = 'gray_dense')(x)
    x = tf.keras.layers.Dropout(0.3)(x) 

    # Output
    outputs = tf.keras.layers.Dense(num_classes, 
                                    activation = 'softmax', 
                                    name = 'prediction')(x)

    # Create Model
    model = tf.keras.models.Model(inputs = inputs, 
                                  outputs=outputs, 
                                  name = 'Grayscale_Brain')

    model.compile(
        optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model