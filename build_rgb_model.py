"""RGB branch — EfficientNetB3 backbone (ImageNet pretrained, frozen) + classification head.

EfficientNetB3 is used over B0 because:
  - Stronger pretrained color/texture features (~2× more params, deeper architecture).
  - Richer 1536-channel feature map vs B0's 1280 — gives the spatial attention gate
    more signal to compute its per-location mask from.
  - NOTE: at 224×224 input both B0 and B3 output 7×7 (stride-32 architecture).
    To get 10×10 output, change config image_size to (300, 300) and re-preprocess.

Normalization: the pipeline normalises images to [0, 1]. EfficientNetB3 includes an
internal Rescaling(1/255) layer and expects [0, 255] input. A Rescaling(255.0) layer
is applied inside this model before the backbone so the backbone always receives the
correct range regardless of how the pipeline is configured.

The backbone is frozen by default. To fine-tune later, call:
    model.get_layer("efficientnetb3").trainable = True
and recompile with a lower learning rate (e.g. 1e-5).

Conv `name=` values that Grad-CAM depends on:
  Deep layer  → "rgb_backbone_output" (config.GRADCAM_RGB_DEEP_LAYER)
  Shallow layer → not applicable; visualize_gradcam_batch gracefully skips if not found.
"""

import tensorflow as tf

import config


def build_rgb_model(
    input_shape=(*config.DATA_PARAMS["image_size"], 3),
    num_classes=None,
):
    inputs = tf.keras.Input(shape=input_shape, name="rgb_input")

    # EfficientNetB3 expects [0, 255] input — rescale from the pipeline's [0, 1] norm.
    x = tf.keras.layers.Rescaling(255.0, name="rgb_rescale")(inputs)

    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False  # Freeze pretrained weights; unfreeze to fine-tune

    # training=False keeps BN layers inside EfficientNetB3 in inference mode
    # even when the outer model is in training mode — required when backbone is frozen.
    x = backbone(x, training=False)
    # Identity layer gives this spatial feature map a fixed name inside the fusion model's
    # computation graph — required for Grad-CAM to resolve the tensor reliably after load.
    x = tf.keras.layers.Activation("linear", name="rgb_backbone_output")(x)
    final_conv_output = x  # (None, 7, 7, 1536) at 224×224 — spatial feature map for Grad-CAM

    # ------------------------------------------------------------------
    # Spatial attention: lightweight 1×1 conv mask suppresses irrelevant
    # background pixels before GAP collapses the spatial dimensions.
    # Grad-CAM target stays on rgb_backbone_output (pre-attention) so it
    # reflects backbone activations; the mask shapes what actually gets pooled.
    # ------------------------------------------------------------------
    attn_mask = tf.keras.layers.Conv2D(
        1, (1, 1), padding="same", activation="sigmoid",
        use_bias=True, name="rgb_attn_mask"
    )(x)
    x = tf.keras.layers.Multiply(name="rgb_attn_out")([x, attn_mask])

    x = tf.keras.layers.GlobalAveragePooling2D(name="rgb_global_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="rgb_dense")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="prediction"
    )(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="RGB_Brain")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, final_conv_output
