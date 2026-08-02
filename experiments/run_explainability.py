"""
Dedicated Grad-CAM explainability analysis experiment.

Generates comprehensive Grad-CAM visualisations and evaluates
localisation quality against ground truth segmentation masks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed, set_gpu_config
from src.utils.config import load_config, get_results_path
from src.utils.logger import get_experiment_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Grad-CAM explainability analysis")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained model (.h5)"
    )
    parser.add_argument(
        "--architecture", type=str, default="ResNet50V2",
        choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"],
        help="Model architecture"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of samples to visualise"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    return parser.parse_args()


def main():
    """
    Run dedicated Grad-CAM analysis.

    Generates:
    - Individual heatmap visualisations (3-panel: original/heatmap/overlay)
    - Batch heatmap analysis across test set
    - IoU and Dice metrics against ground truth masks (if available)
    - Summary report with localisation statistics
    """
    args = parse_args()

    # Setup
    set_seed(42)
    set_gpu_config()

    config = load_config(args.config)
    logger = get_experiment_logger("explainability")

    output_dir = args.output_dir or str(get_results_path("explainability"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not available.")
        return

    from src.data.dataset_loader import CombinedDatasetLoader, GLENDADataset
    from src.data.partition import split_dataset
    from src.models.model_factory import create_model, get_gradcam_layer
    from src.models.base_model import compile_model
    from src.explainability.gradcam import (
        GradCAM,
        save_gradcam_visualisation,
        generate_gradcam_report,
    )

    # Load data
    data_config = config.get("data", {})
    image_size = data_config.get("image_size", 224)

    if args.synthetic:
        images, labels = CombinedDatasetLoader.create_synthetic_dataset(
            num_samples=100,
            image_size=(image_size, image_size),
        )
    else:
        loader = CombinedDatasetLoader(config)
        images, labels = loader.load_all_data()
        if len(images) == 0:
            logger.warning("No data found, using synthetic data.")
            images, labels = CombinedDatasetLoader.create_synthetic_dataset(
                num_samples=100,
                image_size=(image_size, image_size),
            )

    # Split data
    splits = split_dataset(images, labels, seed=42)
    test_images, test_labels = splits["test"]

    # Load or create model
    model_config = config.get("model", {})
    architecture = args.architecture

    if args.model_path and Path(args.model_path).exists():
        logger.info(f"Loading model from: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
    else:
        logger.info(f"Building {architecture} model")
        model = create_model(
            architecture=architecture,
            input_shape=(image_size, image_size, 3),
            num_classes=model_config.get("num_classes", 2),
        )
        model = compile_model(model)

    # Get target layer for Grad-CAM
    target_layer = get_gradcam_layer(architecture)
    logger.info(f"Target layer for Grad-CAM: {target_layer}")

    # Load masks if available
    masks = {}
    glenda_config = data_config.get("datasets", {}).get("glenda", {})
    if glenda_config.get("path"):
        glenda = GLENDADataset(data_dir=glenda_config["path"])
        masks = glenda.load_masks()
        logger.info(f"Loaded {len(masks)} ground truth masks")

    # Generate Grad-CAM report
    num_samples = min(args.num_samples, len(test_images))
    logger.info(f"Generating Grad-CAM for {num_samples} samples...")

    report = generate_gradcam_report(
        model=model,
        images=test_images[:num_samples],
        labels=test_labels[:num_samples],
        target_layer=target_layer,
        save_dir=str(output_path / "visualisations"),
        num_samples=num_samples,
        masks=masks if len(masks) > 0 else None,
    )

    # Compute additional statistics
    gradcam = GradCAM(model=model, target_layer_name=target_layer)

    # Analyse heatmap statistics
    heatmap_stats = {
        "mean_activation": [],
        "max_activation": [],
        "coverage": [],
    }

    for i in range(min(num_samples, len(test_images))):
        heatmap = gradcam.compute_heatmap(test_images[i])
        heatmap_stats["mean_activation"].append(float(np.mean(heatmap)))
        heatmap_stats["max_activation"].append(float(np.max(heatmap)))
        heatmap_stats["coverage"].append(float(np.mean(heatmap > 0.5)))

    report["heatmap_statistics"] = {
        "mean_activation": float(np.mean(heatmap_stats["mean_activation"])),
        "std_activation": float(np.std(heatmap_stats["mean_activation"])),
        "mean_max_activation": float(np.mean(heatmap_stats["max_activation"])),
        "mean_coverage": float(np.mean(heatmap_stats["coverage"])),
    }

    # Save report
    report_file = output_path / "explainability_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nGrad-CAM Report:")
    logger.info(f"  Samples analysed: {num_samples}")
    logger.info(f"  Mean activation: {report['heatmap_statistics']['mean_activation']:.4f}")
    logger.info(f"  Mean coverage (>0.5): {report['heatmap_statistics']['mean_coverage']:.4f}")

    if report.get("metrics"):
        logger.info(f"  Mean IoU: {report['metrics'].get('mean_iou', 0):.4f}")
        logger.info(f"  Mean Dice: {report['metrics'].get('mean_dice', 0):.4f}")

    logger.info(f"\nResults saved to: {output_path}")

    return report


if __name__ == "__main__":
    main()
