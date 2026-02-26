import tensorflow as tf

def sobel_edge_layer(x):

    edges = tf.image.sobel_edges(x) # tf build in sobel edge

    # Vertical and horizontal edges
    dy = edges[...,0] # height
    dx = edges[...,1] # lenght 

    # Magnitude = sqrt(G_x^2 + G_y^2)
    magnitude = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy)) # strength of edge at pixel
    
    return magnitude # rate of change of pixel intensity

# model input layers (3x3)
def build_sobel_model(input_shape = (224, 224, 1), num_classes = 3):
        
    inputs = tf.keras.Input(shape = input_shape, name = "sobel_input")
    x = tf.keras.layers.Lambda(function = sobel_edge_layer, 
                               output_shape = None, # infer, not cropping images
                               mask = None, # ignore padding 
                               arguemnt = None, # threshold value
                               name = "sobel_input") 

    # Low: Edges  
    # convolution layers (activation)
    x = tf.keras.layers.Conv2D(filters = 32, 
                               kernel_size = (3,3),
                               activation = 'relu',
                               padding = 'same',
                               name = 'sobel_conv1')(x)
        
    # pooling layers 
    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2), 
                                     name = 'sobel_pool1')(x)
    # Middle: Patterns 
    # convolution layers (activation)
    x = tf.keras.layers.Conv2D(filters = 64,
                               kernel_size = (3, 3),
                               activation = 'relu',
                               padding = 'same',
                               name = 'sobel_conv2')(x)
        
    # pooling layers 
    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2),
                                     name = 'sobel_pool2')(x) 

    # High: Shape
    # convolution layers (activation)
    x = tf.keras.layers.Conv2D(filters = 128,
                               kernel_size = (3, 3),
                               activation = 'relu',
                               padding = 'same',
                               name = 'sobel_conv3_final')(x)
    
    # pooling layers 
    x = tf.keras.layers.GlobalAveragePooling2D(name = 'sobel_global_pool')(x)

    # connection layers (classifier)
    x = tf.keras.layers.Dense(filters = 64, 
                              activation = 'relu', 
                              name = 'sobel_dense')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # outputs
    outputs = tf.keras.layers.Dense(num_classes, 
                                    activation = 'softmax', 
                                    name = 'prediction')(x)
    
    model = tf.keras.models.Model(inputs = inputs, 
                                  outputs = outputs, 
                                  name = 'Sobel_Edge')
    model.compile(
        optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model