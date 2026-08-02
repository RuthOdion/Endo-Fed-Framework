"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Provides visual explanations for model predictions by highlighting
regions in the image that contribute most to the classification decision.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Computes class-discriminative localisation maps by using gradients
    flowing into the final convolutional layer to highlight important
    regions for prediction.
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        target_layer_name: str,
    ):
        """
        Initialise Grad-CAM.

        Args:
            model: Trained Keras model.
            target_layer_name: Name of the target convolutional layer.
        """
        if tf is None:
            raise ImportError("TensorFlow is required for Grad-CAM.")

        self.model = model
        self.target_layer_name = target_layer_name

        # Build gradient model
        self.grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(target_layer_name).output,
                model.output,
            ],
        )

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Process:
        1. Forward pass to get conv features and predictions
        2. Compute gradients of class score w.r.t. conv features
        3. Global average pool the gradients (importance weights)
        4. Weight combination of feature maps
        5. Apply ReLU (only positive contributions)
        6. Normalise to [0, 1]

        Args:
            image: Preprocessed input image (H, W, 3) or (1, H, W, 3).
            class_index: Target class. If None, uses predicted class.

        Returns:
            Heatmap array normalised to [0, 1], same spatial size as conv layer output.
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        image_tensor = tf.cast(image, tf.float32)

        # Forward pass with gradient recording
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]

        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by the importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU - only positive contributions
        heatmap = tf.maximum(heatmap, 0)

        # Normalise to [0, 1]
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    def generate_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colourmap: str = "jet",
    ) -> np.ndarray:
        """
        Generate heatmap overlay on the original image.

        Args:
            image: Original image (H, W, 3) in [0, 1] or [0, 255].
            heatmap: Grad-CAM heatmap.
            alpha: Overlay transparency (0=transparent, 1=opaque).
            colourmap: Matplotlib colourmap name.

        Returns:
            Overlay image (H, W, 3) in [0, 255] uint8.
        """
        # Ensure image is in [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Resize heatmap to image size
        h, w = image.shape[:2]
        if cv2 is not None:
            heatmap_resized = cv2.resize(heatmap, (w, h))
        else:
            # Simple nearest neighbour resize
            from PIL import Image as PILImage
            heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize((w, h))
            heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0

        # Apply colourmap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        if cv2 is not None:
            heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)
        else:
            if plt is not None:
                cmap = plt.get_cmap(colourmap)
                heatmap_coloured = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
            else:
                heatmap_coloured = np.stack([heatmap_uint8] * 3, axis=-1)

        # Blend
        overlay = (image * (1 - alpha) + heatmap_coloured * alpha).astype(np.uint8)

        return overlay

    def compute_batch_heatmaps(
        self,
        images: np.ndarray,
        class_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Compute Grad-CAM heatmaps for a batch of images.

        Args:
            images: Batch of images (N, H, W, 3).
            class_indices: Optional target class per image.

        Returns:
            List of heatmap arrays.
        """
        heatmaps = []
        for i in range(len(images)):
            class_idx = class_indices[i] if class_indices else None
            heatmap = self.compute_heatmap(images[i], class_idx)
            heatmaps.append(heatmap)
        return heatmaps

    def evaluate_against_masks(
        self,
        images: np.ndarray,
        masks: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate Grad-CAM heatmaps against ground truth segmentation masks.

        Computes IoU and Dice metrics to assess localisation quality.

        Args:
            images: Input images (N, H, W, 3).
            masks: Dictionary of ground truth masks.
            threshold: Threshold for binarising heatmaps.

        Returns:
            Dictionary with mean IoU and Dice scores.
        """
        iou_scores = []
        dice_scores = []

        mask_list = list(masks.values())
        num_eval = min(len(images), len(mask_list))

        for i in range(num_eval):
            heatmap = self.compute_heatmap(images[i], class_index=1)

            # Resize heatmap to mask size
            mask = mask_list[i]
            h, w = mask.shape[:2]
            if cv2 is not None:
                heatmap_resized = cv2.resize(heatmap, (w, h))
            else:
                heatmap_resized = heatmap  # fallback

            # Binarise heatmap
            heatmap_binary = (heatmap_resized >= threshold).astype(np.float32)
            mask_binary = (mask >= 0.5).astype(np.float32)

            iou = compute_iou(heatmap_binary, mask_binary)
            dice = compute_dice(heatmap_binary, mask_binary)

            iou_scores.append(iou)
            dice_scores.append(dice)

        return {
            "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            "std_iou": float(np.std(iou_scores)) if iou_scores else 0.0,
            "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
            "std_dice": float(np.std(dice_scores)) if dice_scores else 0.0,
            "num_evaluated": num_eval,
        }


def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        prediction: Predicted binary mask.
        ground_truth: Ground truth binary mask.

    Returns:
        IoU score in [0, 1].
    """
    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        prediction: Predicted binary mask.
        ground_truth: Ground truth binary mask.

    Returns:
        Dice score in [0, 1].
    """
    intersection = np.logical_and(prediction, ground_truth).sum()
    total = prediction.sum() + ground_truth.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(2.0 * intersection / total)


def save_gradcam_visualisation(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    save_path: str,
    title: str = "Grad-CAM Visualisation",
) -> None:
    """
    Save a 3-panel Grad-CAM visualisation figure.

    Panels: [Original Image | Heatmap | Overlay]

    Args:
        image: Original input image.
        heatmap: Grad-CAM heatmap.
        overlay: Heatmap overlay on original.
        save_path: Path to save the figure.
        title: Figure title.
    """
    if plt is None:
        print("matplotlib required for saving visualisations.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    if image.max() <= 1.0:
        display_image = image
    else:
        display_image = image / 255.0

    axes[0].imshow(display_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    if overlay.max() > 1.0:
        overlay_display = overlay / 255.0
    else:
        overlay_display = overlay
    axes[2].imshow(overlay_display)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_gradcam_report(
    model: "tf.keras.Model",
    images: np.ndarray,
    labels: np.ndarray,
    target_layer: str,
    save_dir: str,
    num_samples: int = 10,
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Generate a complete Grad-CAM analysis report.

    Args:
        model: Trained model.
        images: Input images for analysis.
        labels: True labels.
        target_layer: Target layer name for Grad-CAM.
        save_dir: Directory to save visualisations.
        num_samples: Number of samples to visualise.
        masks: Optional ground truth masks for evaluation.

    Returns:
        Report dictionary with metrics and file paths.
    """
    gradcam = GradCAM(model=model, target_layer_name=target_layer)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    report = {"visualisations": [], "metrics": {}}

    # Generate visualisations for sample images
    num_to_show = min(num_samples, len(images))

    for i in range(num_to_show):
        heatmap = gradcam.compute_heatmap(images[i], class_index=int(labels[i]))
        overlay = gradcam.generate_overlay(images[i], heatmap)

        fig_path = str(save_path / f"gradcam_sample_{i:03d}.png")
        save_gradcam_visualisation(
            image=images[i],
            heatmap=heatmap,
            overlay=overlay,
            save_path=fig_path,
            title=f"Sample {i} (Label: {labels[i]})",
        )
        report["visualisations"].append(fig_path)

    # Evaluate against masks if provided
    if masks is not None and len(masks) > 0:
        mask_metrics = gradcam.evaluate_against_masks(images, masks)
        report["metrics"] = mask_metrics

    return report
