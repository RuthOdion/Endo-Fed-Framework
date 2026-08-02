"""Explainability modules for model interpretation using Grad-CAM."""

from src.explainability.gradcam import (
    GradCAM,
    compute_iou,
    compute_dice,
    save_gradcam_visualisation,
    generate_gradcam_report,
)
