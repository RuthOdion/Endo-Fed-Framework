"""
ResNet50V2 architecture for endometriosis classification.

Implements transfer learning from ImageNet with a custom
classification head for binary endometriosis detection.
"""

from typing import Tuple

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.models.base_model import build_classification_head, freeze_base_layers


def build_resnet50v2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
    unfreeze_top: int = 20,
) -> "tf.keras.Model":
    """
    Build ResNet50V2 model with ImageNet pre-trained weights.

    Architecture:
        - ResNet50V2 backbone (ImageNet weights, no top)
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
        Compiled ResNet50V2 model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for model building.")

    # Load pre-trained ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
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

    For ResNet50V2, the 'post_relu' layer is the final activation
    before global average pooling, providing the best spatial resolution
    for class activation mapping.

    Returns:
        Name of the target layer for Grad-CAM.
    """
    return "post_relu"
