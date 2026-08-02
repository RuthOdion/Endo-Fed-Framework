"""
Image preprocessing pipeline for endometriosis classification.

Implements ImageNet normalisation, data augmentation, and MRI slice
extraction for the federated learning framework.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from PIL import Image
except ImportError:
    Image = None

# ImageNet normalisation constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W).
        target_size: Target (height, width) tuple.
        interpolation: Interpolation method ('bilinear', 'nearest', 'cubic').

    Returns:
        Resized image as numpy array.
    """
    if cv2 is not None:
        interp_map = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
        }
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interp)
    elif Image is not None:
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        resized = np.array(pil_image)
    else:
        raise ImportError("Either cv2 or PIL is required for image resizing.")

    return resized


def normalise_image(image: np.ndarray) -> np.ndarray:
    """
    Normalise image pixel values to [0, 1] range.

    Args:
        image: Input image with pixel values in [0, 255].

    Returns:
        Normalised image with values in [0, 1].
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def standardise_imagenet(image: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet standardisation (zero mean, unit variance).

    Args:
        image: Normalised image with values in [0, 1], shape (H, W, 3).

    Returns:
        Standardised image with ImageNet mean subtracted and divided by std.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    standardise: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Pipeline: Resize -> Normalise [0,1] -> ImageNet Standardisation.

    Args:
        image: Input image as numpy array.
        target_size: Target (height, width) for resizing.
        standardise: Whether to apply ImageNet standardisation.

    Returns:
        Preprocessed image ready for model input.
    """
    # Resize
    image = resize_image(image, target_size)

    # Ensure 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    # Normalise to [0, 1]
    image = normalise_image(image)

    # Apply ImageNet standardisation
    if standardise:
        image = standardise_imagenet(image)

    return image


def preprocess_batch(
    images: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    standardise: bool = True,
) -> np.ndarray:
    """
    Preprocess a batch of images.

    Args:
        images: Batch of images as numpy array (N, H, W, C) or list of arrays.
        target_size: Target (height, width).
        standardise: Whether to apply ImageNet standardisation.

    Returns:
        Preprocessed batch as numpy array (N, H, W, 3).
    """
    processed = []
    for img in images:
        processed.append(preprocess_image(img, target_size, standardise))
    return np.array(processed, dtype=np.float32)


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights using the formula: w_c = N / (num_classes * n_c).

    This addresses class imbalance by giving higher weight to minority classes.

    Args:
        labels: Array of class labels (integers or one-hot encoded).

    Returns:
        Dictionary mapping class index to weight value.
    """
    if len(labels.shape) > 1:
        # One-hot encoded -> convert to class indices
        labels = np.argmax(labels, axis=1)

    classes = np.unique(labels)
    num_classes = len(classes)
    total_samples = len(labels)

    class_weights = {}
    for cls in classes:
        n_c = np.sum(labels == cls)
        weight = total_samples / (num_classes * n_c)
        class_weights[int(cls)] = float(weight)

    return class_weights


def create_augmentation_pipeline(config: Optional[Dict] = None) -> Dict:
    """
    Create data augmentation configuration.

    Args:
        config: Optional augmentation config dict. Uses defaults if None.

    Returns:
        Dictionary with augmentation parameters.
    """
    default_config = {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_range": 30,
        "zoom_range": 0.2,
        "brightness_range": 0.2,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "fill_mode": "nearest",
    }

    if config is not None:
        default_config.update(config)

    return default_config


def create_tf_augmentation_layer(config: Optional[Dict] = None):
    """
    Create TensorFlow data augmentation layer using Keras preprocessing layers.

    Args:
        config: Augmentation configuration dictionary.

    Returns:
        tf.keras.Sequential model with augmentation layers.

    Raises:
        ImportError: If TensorFlow is not available.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for augmentation layers.")

    if config is None:
        config = create_augmentation_pipeline()

    layers = []

    if config.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))

    if config.get("vertical_flip", False):
        layers.append(tf.keras.layers.RandomFlip("vertical"))

    rotation = config.get("rotation_range", 30)
    if rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(rotation / 360.0))

    zoom = config.get("zoom_range", 0.2)
    if zoom > 0:
        layers.append(tf.keras.layers.RandomZoom((-zoom, zoom)))

    brightness = config.get("brightness_range", 0.2)
    if brightness > 0:
        layers.append(tf.keras.layers.RandomBrightness((-brightness, brightness)))

    augmentation_model = tf.keras.Sequential(layers, name="augmentation_pipeline")
    return augmentation_model


def extract_mri_slices(
    volume: np.ndarray,
    axis: int = 2,
    slice_range: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Extract 2D slices from a 3D MRI volume.

    Args:
        volume: 3D MRI volume array (D, H, W) or (H, W, D).
        axis: Axis along which to extract slices (0, 1, or 2).
        slice_range: Optional (start, end) tuple to extract a subset.

    Returns:
        List of 2D slice arrays.
    """
    num_slices = volume.shape[axis]

    if slice_range is not None:
        start, end = slice_range
        start = max(0, start)
        end = min(num_slices, end)
    else:
        # Skip first and last 10% (typically contain less useful information)
        margin = int(num_slices * 0.1)
        start = margin
        end = num_slices - margin

    slices = []
    for i in range(start, end):
        if axis == 0:
            s = volume[i, :, :]
        elif axis == 1:
            s = volume[:, i, :]
        else:
            s = volume[:, :, i]
        slices.append(s)

    return slices
