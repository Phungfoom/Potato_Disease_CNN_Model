"""Sobel-magnitude branch — trainable Sobel pair + attention gate + multi-scale edge fusion.

Architecture follows EEDB-CNN (PMC12777044):
  1. Trainable 3×3 Sobel pair (``sobel_learn``, 18 params) initialized to Gx/Gy,
     fine-tuned via backprop.  Magnitude map computed from the two-channel output.
  2. Spatial attention gate on magnitude — learned sigmoid mask suppresses background edges.
  3. Three-scale edge fusion:
       Scale 1 — Conv2D(32, 3×3)                           pixel-level gradients
       Scale 2 — dilated Conv2D(32) at rate 1 + rate 2,    mid-level (concat → 64 ch)
       Scale 3 — Conv2D(32, 1×1)                           channel compression
       Learnable attention vector α weights scale contributions.
  4. Classification head: sobel_conv3_final (Grad-CAM target) → spatial attention gate
     (128-ch, symmetric with RGB/Gray branches) → GAP combined with scale-fused vector
     → Dense(64) → Dropout → Softmax.

``sobel_edge_layer`` is kept as a standalone utility for:
  - Backward-compatible loading of older saved models that still contain the
    Lambda layer (pass ``custom_objects={"sobel_edge_layer": sobel_edge_layer}``).
  - Image-preview scripts (image_view.py) that visualize a fixed Sobel reference.

use_bias=False on Conv2D layers followed by BatchNormalization.
"""

import numpy as np
import tensorflow as tf

import config


def sobel_edge_layer(x):
    """Fixed Sobel magnitude — kept for backward compat and visualization only."""
    edges = tf.image.sobel_edges(x)
    dy, dx = edges[..., 0], edges[..., 1]
    return tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))


@tf.keras.utils.register_keras_serializable(package="potato_id")
class SobelKernelInitializer(tf.keras.initializers.Initializer):
    """Initialize a (3, 3, 1, 2) kernel with the classic Sobel-x / Sobel-y filters.

    The two output filters start as directional edge detectors; backprop then
    refines them toward the structural cues most useful for disease classification.
    """

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32
        k = np.zeros(shape, dtype=np.float32)
        # filter 0 — Sobel-x (responds to vertical edges)
        k[:, :, 0, 0] = [[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]]
        # filter 1 — Sobel-y (responds to horizontal edges)
        k[:, :, 0, 1] = [[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]]
        return tf.constant(k, dtype=dtype)

    def get_config(self):
        return {}


@tf.keras.utils.register_keras_serializable(package="potato_id")
class SobelMagnitude(tf.keras.layers.Layer):
    """Compute |gradient| = sqrt(Gx² + Gy²) from a 2-channel Sobel output.

    Replaces a Lambda layer so the model serialises cleanly without custom_objects.
    """

    def call(self, x):
        return tf.math.sqrt(
            tf.math.square(x[..., 0:1]) + tf.math.square(x[..., 1:2]) + 1e-7
        )

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="potato_id")
class AlphaWeightedSum(tf.keras.layers.Layer):
    """Per-sample α-weighted sum of three equal-width vectors.

    Inputs: [s1, s2, s3, alpha]
      s1, s2, s3 — (B, D) feature vectors (same D)
      alpha      — (B, 3) softmax weights

    Returns: (B, D)  =  α[:,0:1]*s1 + α[:,1:2]*s2 + α[:,2:3]*s3

    Replaces three Lambda layers so the model serialises cleanly.
    """

    def call(self, inputs):
        s1, s2, s3, alpha = inputs
        return (
            s1 * alpha[:, 0:1]
            + s2 * alpha[:, 1:2]
            + s3 * alpha[:, 2:3]
        )

    def get_config(self):
        return super().get_config()


def _attention_gate(x, name_prefix: str):
    """Learned spatial sigmoid mask — suppresses low-activation (background) regions."""
    gate = tf.keras.layers.Conv2D(
        1, (1, 1),
        padding="same",
        activation="sigmoid",
        use_bias=True,
        name=f"{name_prefix}_attn_gate",
    )(x)
    return tf.keras.layers.Multiply(name=f"{name_prefix}_attn_apply")([x, gate])


def build_sobel_model(
    input_shape=(*config.DATA_PARAMS["image_size"], 1),
    num_classes=None,
):
    inputs = tf.keras.Input(shape=input_shape, name="sobel_input")

    # ------------------------------------------------------------------
    # 1. Trainable Sobel pair → magnitude map
    # ------------------------------------------------------------------
    sobel_learn = tf.keras.layers.Conv2D(
        2, (3, 3), padding="same", use_bias=False,
        kernel_initializer=SobelKernelInitializer(),
        name="sobel_learn",
    )(inputs)  # (B, H, W, 2) — ch0: Gx, ch1: Gy

    mag = SobelMagnitude(name="sobel_magnitude")(sobel_learn)  # (B, H, W, 1)

    # ------------------------------------------------------------------
    # 2. Spatial attention gate on magnitude (background edge suppression)
    # ------------------------------------------------------------------
    mag_gated = _attention_gate(mag, name_prefix="sobel")  # (B, H, W, 1)

    # ------------------------------------------------------------------
    # 3. Multi-scale edge fusion
    # ------------------------------------------------------------------

    # Scale 1 — pixel-level
    s1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="sobel_s1_conv")(mag_gated)
    s1 = tf.keras.layers.BatchNormalization(name="sobel_s1_bn")(s1)
    s1 = tf.keras.layers.Activation("relu", name="sobel_s1_relu")(s1)          # (B, H, W, 32)

    # Scale 2 — mid-level dilated (rate 1 + rate 2 in parallel, then concat)
    s2a = tf.keras.layers.Conv2D(32, (3, 3), padding="same", dilation_rate=1, use_bias=False, name="sobel_s2a_conv")(mag_gated)
    s2a = tf.keras.layers.BatchNormalization(name="sobel_s2a_bn")(s2a)
    s2a = tf.keras.layers.Activation("relu", name="sobel_s2a_relu")(s2a)

    s2b = tf.keras.layers.Conv2D(32, (3, 3), padding="same", dilation_rate=2, use_bias=False, name="sobel_s2b_conv")(mag_gated)
    s2b = tf.keras.layers.BatchNormalization(name="sobel_s2b_bn")(s2b)
    s2b = tf.keras.layers.Activation("relu", name="sobel_s2b_relu")(s2b)

    s2 = tf.keras.layers.Concatenate(name="sobel_s2_concat")([s2a, s2b])       # (B, H, W, 64)

    # Scale 3 — 1×1 channel compression
    s3 = tf.keras.layers.Conv2D(32, (1, 1), padding="same", use_bias=False, name="sobel_s3_conv")(s2)
    s3 = tf.keras.layers.BatchNormalization(name="sobel_s3_bn")(s3)
    s3 = tf.keras.layers.Activation("relu", name="sobel_s3_relu")(s3)          # (B, H, W, 32)

    # Learnable α: weight each scale's contribution (per-sample, via AlphaWeightedSum)
    g1 = tf.keras.layers.GlobalAveragePooling2D(name="sobel_alpha_g1")(s1)     # (B, 32)
    g2 = tf.keras.layers.GlobalAveragePooling2D(name="sobel_alpha_g2")(s2)     # (B, 64)
    g2 = tf.keras.layers.Dense(32, use_bias=False, name="sobel_alpha_g2_proj")(g2)  # (B, 32)
    g3 = tf.keras.layers.GlobalAveragePooling2D(name="sobel_alpha_g3")(s3)     # (B, 32)

    alpha_in = tf.keras.layers.Concatenate(name="sobel_alpha_cat")([g1, g2, g3])    # (B, 96)
    alpha_logits = tf.keras.layers.Dense(3, name="sobel_alpha_logits")(alpha_in)
    alpha = tf.keras.layers.Activation("softmax", name="sobel_alpha")(alpha_logits)  # (B, 3)

    # Per-scale GAP vectors
    s1_gap = tf.keras.layers.GlobalAveragePooling2D(name="sobel_s1_gap")(s1)   # (B, 32)
    s2_gap = tf.keras.layers.Dense(32, use_bias=False, name="sobel_s2_gap_proj")(
        tf.keras.layers.GlobalAveragePooling2D(name="sobel_s2_gap")(s2)
    )                                                                            # (B, 32)
    s3_gap = tf.keras.layers.GlobalAveragePooling2D(name="sobel_s3_gap")(s3)   # (B, 32)

    # α-weighted sum — no Lambda; AlphaWeightedSum is a registered serialisable layer
    scale_fused = AlphaWeightedSum(name="sobel_scale_fused")(
        [s1_gap, s2_gap, s3_gap, alpha]
    )  # (B, 32)

    # ------------------------------------------------------------------
    # 4. Classification head
    # Named conv preserved for Grad-CAM (ensure_grad_targets looks up "sobel_conv3_final").
    # Spatial attention gate added after relu3 — symmetric with RGB/Gray branches —
    # so the 128-ch feature map is background-suppressed before GAP.
    # ------------------------------------------------------------------
    final_conv_layer_sobel = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", use_bias=False, name="sobel_conv3_final"
    )(s3)
    x = tf.keras.layers.BatchNormalization(name="sobel_bn3")(final_conv_layer_sobel)
    x = tf.keras.layers.Activation("relu", name="sobel_relu3")(x)

    # Spatial attention on the deep 128-ch feature map (mirrors rgb_attn / gray_attn)
    x = _attention_gate(x, name_prefix="sobel_final")                          # (B, H, W, 128)

    x = tf.keras.layers.GlobalAveragePooling2D(name="sobel_global_pool")(x)    # (B, 128)

    # Combine spatial GAP with scale-fused vector
    x = tf.keras.layers.Concatenate(name="sobel_head_concat")([x, scale_fused])  # (B, 160)

    x = tf.keras.layers.Dense(64, activation="relu", name="sobel_dense")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="prediction"
    )(x)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name="Sobel_Edge_Brain"
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, final_conv_layer_sobel
