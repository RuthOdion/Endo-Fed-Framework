"""
Model factory for creating architecture variants.

Provides a registry-based approach to instantiate different model
architectures with consistent interfaces.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.models.resnet50v2 import build_resnet50v2
from src.models.resnet50v2 import get_gradcam_target_layer as resnet_gradcam_layer
from src.models.efficientnet_b0 import build_efficientnet_b0
from src.models.efficientnet_b0 import get_gradcam_target_layer as efficientnet_gradcam_layer
from src.models.mobilenetv2 import build_mobilenetv2
from src.models.mobilenetv2 import get_gradcam_target_layer as mobilenet_gradcam_layer

# Model registry mapping architecture names to builder functions
MODEL_REGISTRY: Dict[str, callable] = {
    "ResNet50V2": build_resnet50v2,
    "EfficientNetB0": build_efficientnet_b0,
    "MobileNetV2": build_mobilenetv2,
}

# Grad-CAM target layer registry
GRADCAM_LAYER_REGISTRY: Dict[str, callable] = {
    "ResNet50V2": resnet_gradcam_layer,
    "EfficientNetB0": efficientnet_gradcam_layer,
    "MobileNetV2": mobilenet_gradcam_layer,
}


def create_model(
    architecture: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
    unfreeze_top: int = 20,
) -> "tf.keras.Model":
    """
    Create a model by architecture name.

    Args:
        architecture: Name of the architecture (e.g., 'ResNet50V2').
        input_shape: Input shape (H, W, C).
        num_classes: Number of output classes.
        dense_units: Dense layer units.
        dropout_rate: Dropout rate.
        freeze_base: Whether to freeze base layers.
        unfreeze_top: Number of top layers to unfreeze.

    Returns:
        Instantiated Keras model.

    Raises:
        ValueError: If architecture not found in registry.
    """
    if architecture not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: '{architecture}'. Available: {available}"
        )

    builder_fn = MODEL_REGISTRY[architecture]
    model = builder_fn(
        input_shape=input_shape,
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        freeze_base=freeze_base,
        unfreeze_top=unfreeze_top,
    )

    return model


def create_model_from_config(config: Dict) -> "tf.keras.Model":
    """
    Create a model from configuration dictionary.

    Args:
        config: Full configuration dictionary with 'model' and 'data' sections.

    Returns:
        Instantiated and compiled Keras model.
    """
    from src.models.base_model import compile_model

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    image_size = data_config.get("image_size", 224)
    channels = data_config.get("channels", 3)

    architecture = model_config.get("architectures", ["ResNet50V2"])[0]

    model = create_model(
        architecture=architecture,
        input_shape=(image_size, image_size, channels),
        num_classes=model_config.get("num_classes", 2),
        dense_units=model_config.get("dense_units", 256),
        dropout_rate=model_config.get("dropout", 0.3),
        freeze_base=model_config.get("freeze_base", True),
        unfreeze_top=model_config.get("unfreeze_top", 20),
    )

    training_config = config.get("training", {})
    lr = training_config.get("optimiser", {}).get("learning_rate", 0.001)

    model = compile_model(
        model,
        learning_rate=lr,
        num_classes=model_config.get("num_classes", 2),
    )

    return model


def get_gradcam_layer(architecture: str) -> str:
    """
    Get the Grad-CAM target layer for an architecture.

    Args:
        architecture: Architecture name.

    Returns:
        Target layer name for Grad-CAM.

    Raises:
        ValueError: If architecture not found.
    """
    if architecture not in GRADCAM_LAYER_REGISTRY:
        available = ", ".join(GRADCAM_LAYER_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: '{architecture}'. Available: {available}"
        )

    return GRADCAM_LAYER_REGISTRY[architecture]()


def get_all_architectures() -> List[str]:
    """
    Get list of all available architecture names.

    Returns:
        List of architecture name strings.
    """
    return list(MODEL_REGISTRY.keys())


def get_model_size_mb(model: "tf.keras.Model") -> float:
    """
    Calculate model size in megabytes.

    Args:
        model: Keras model.

    Returns:
        Model size in MB.
    """
    if tf is None:
        return 0.0

    total_params = model.count_params()
    # Each parameter is float32 = 4 bytes
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)
