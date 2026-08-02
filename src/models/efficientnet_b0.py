"""
EfficientNetB0 architecture for endometriosis classification.

Implements transfer learning from ImageNet with a custom
classification head optimised for efficient inference.
"""

from typing import Tuple

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.models.base_model import build_classification_head, freeze_base_layers


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
    unfreeze_top: int = 20,
) -> "tf.keras.Model":
    """
    Build EfficientNetB0 model with ImageNet pre-trained weights.

    Architecture:
        - EfficientNetB0 backbone (ImageNet weights, no top)
        - Global Average Pooling
        - Dense(256, ReLU)
        - Dropout(0.3)
        - Dense(2, Softmax)

    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        dense_units: Units in the dense layer.
        dropout_rate: Dropout rate.
        freeze_base: Whether to freeze base layers.
        unfreeze_top: Number of top layers to unfreeze.

    Returns:
        Compiled EfficientNetB0 model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for model building.")

    # Load pre-trained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Build classification head
    model = build_classification_head(
        base_model=base_model,
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )

    # Freeze/unfreeze layers
    if freeze_base:
        model = freeze_base_layers(model, unfreeze_top=unfreeze_top)

    return model


def get_gradcam_target_layer() -> str:
    """
    Get the target layer name for Grad-CAM visualisation.

    For EfficientNetB0, the 'top_conv' layer provides the final
    convolutional feature maps before the classification head.

    Returns:
        Name of the target layer for Grad-CAM.
    """
    return "top_conv"
