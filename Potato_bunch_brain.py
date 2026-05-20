import tensorflow as tf

import config
from build_grayscale_model import build_grayscale_model
from build_rgb_model import build_rgb_model
from build_sobel_model import build_sobel_model
from focal_loss import focal_loss


def build_combined_model(num_classes):
    img_size = config.DATA_PARAMS["image_size"]

    rgb_branch, rgb_target = build_rgb_model(
        input_shape=(*img_size, 3), num_classes=num_classes
    )
    gray_branch, gray_target = build_grayscale_model(
        input_shape=(*img_size, 1), num_classes=num_classes
    )
    sobel_branch, sobel_target = build_sobel_model(
        input_shape=(*img_size, 1), num_classes=num_classes
    )

    rgb_out   = rgb_branch.get_layer("rgb_dense").output    # (B, 64)
    gray_out  = gray_branch.get_layer("gray_dense").output  # (B, 64)
    sobel_out = sobel_branch.get_layer("sobel_dense").output # (B, 64)

    # ------------------------------------------------------------------
    # Spatial side: RGB + Grayscale answer "what does the leaf look like?"
    # Compress their concatenation to the same 64-dim as the edge branch.
    # ------------------------------------------------------------------
    spatial_cat = tf.keras.layers.Concatenate(name="spatial_concat")(
        [rgb_out, gray_out]
    )  # (B, 128)
    spatial = tf.keras.layers.Dense(
        64, activation="relu", name="spatial_compress"
    )(spatial_cat)  # (B, 64)

    # ------------------------------------------------------------------
    # Edge side: Sobel answers "where are the boundaries and lesion edges?"
    # Already 64-dim from sobel_dense (includes multi-scale scale_fused).
    # ------------------------------------------------------------------
    edge = sobel_out  # (B, 64)

    # ------------------------------------------------------------------
    # β gate: per-image scalar in (0, 1) that balances spatial vs edge.
    #
    #   β → 1  model trusts spatial (colour/texture) more
    #   β → 0  model trusts edge detail more
    #
    # Computed from both signals so the gate can read the content of the
    # image before deciding which branch to weight up.
    #
    # Implementation avoids an explicit (1-β) multiply — mathematically
    # equivalent via: fused = edge + β × (spatial - edge)
    #   = β × spatial + (1-β) × edge        ✓
    # Uses only standard Keras layers (Subtract / Multiply / Add) so the
    # model serialises without custom objects.
    # ------------------------------------------------------------------
    gate_input = tf.keras.layers.Concatenate(name="gate_input")(
        [spatial, edge]
    )  # (B, 128)
    beta = tf.keras.layers.Dense(
        1, activation="sigmoid", name="beta_gate"
    )(gate_input)  # (B, 1)  — one scalar per image

    diff      = tf.keras.layers.Subtract(name="spatial_minus_edge")([spatial, edge])
    beta_diff = tf.keras.layers.Multiply(name="beta_diff")([beta, diff])
    fused     = tf.keras.layers.Add(name="feature_combined")([edge, beta_diff])
    # fused: (B, 64)

    # ------------------------------------------------------------------
    # Fusion head
    # fusion_dense_2 is the k-NN embedding extraction point
    # (config.NEIGHBOR_EMBEDDING_LAYER = "fusion_dense_2").
    # ------------------------------------------------------------------
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(fused)
    x = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense_2")(x)

    output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="final_prediction", dtype="float32"
    )(x)

    full_model = tf.keras.models.Model(
        inputs=[rgb_branch.input, gray_branch.input, sobel_branch.input],
        outputs=output,
        name="Potato_Disease_Bunch",
    )

    full_model.grad_targets = {
        "RGB": rgb_target,
        "Gray": gray_target,
        "Sobel": sobel_target,
    }

    full_model.compile(
        optimizer="adam",
        loss=focal_loss(),
        metrics=["accuracy"],
    )

    return full_model
