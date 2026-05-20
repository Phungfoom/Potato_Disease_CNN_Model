"""Grayscale branch (single-channel input).

Conv blocks follow the Conv → BatchNorm → ReLU pattern.
use_bias=False on Conv2D layers because BatchNormalization has its own bias term.
"""

import tensorflow as tf

import config


def build_grayscale_model(
    input_shape=(*config.DATA_PARAMS["image_size"], 1),
    num_classes=None,
):
    inputs = tf.keras.Input(shape=input_shape, name="gray_input")

    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", use_bias=False, name="gray_conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="gray_bn1")(x)
    x = tf.keras.layers.Activation("relu", name="gray_relu1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="gray_pool1")(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", use_bias=False, name="gray_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="gray_bn2")(x)
    x = tf.keras.layers.Activation("relu", name="gray_relu2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="gray_pool2")(x)

    # Return the Conv2D output tensor for Grad-CAM (consistent with ensure_grad_targets
    # which looks up "gray_conv3_final" by name and calls .output).
    final_conv_layer_gray = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", use_bias=False, name="gray_conv3_final"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="gray_bn3")(final_conv_layer_gray)
    x = tf.keras.layers.Activation("relu", name="gray_relu3")(x)

    # ------------------------------------------------------------------
    # Spatial attention: lightweight 1×1 conv mask suppresses irrelevant
    # background pixels before GAP collapses the spatial dimensions.
    # Grad-CAM target stays on gray_conv3_final (pre-attention) so it
    # reflects conv activations; the mask shapes what actually gets pooled.
    # ------------------------------------------------------------------
    attn_mask = tf.keras.layers.Conv2D(
        1, (1, 1), padding="same", activation="sigmoid",
        use_bias=True, name="gray_attn_mask"
    )(x)
    x = tf.keras.layers.Multiply(name="gray_attn_out")([x, attn_mask])

    x = tf.keras.layers.GlobalAveragePooling2D(name="gray_global_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="gray_dense")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="prediction"
    )(x)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name="Grayscale_Brain"
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, final_conv_layer_gray
