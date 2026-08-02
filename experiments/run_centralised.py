"""
Centralised training experiment runner.

Serves as the baseline comparison for federated learning by
training on all data centrally without partitioning or privacy.
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
    parser = argparse.ArgumentParser(description="Run centralised baseline experiment")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--architecture", type=str, default="ResNet50V2",
        choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"],
        help="Model architecture"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs"
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


def run_centralised_experiment(
    config: dict,
    architecture: str = "ResNet50V2",
    epochs: int = None,
    use_synthetic: bool = False,
    output_dir: str = None,
    logger=None,
) -> dict:
    """
    Run centralised training experiment as baseline.

    Pipeline:
    1. Load data
    2. Split into train/val/test
    3. Build model
    4. Train with callbacks (early stopping, LR scheduling)
    5. Evaluate on test set
    6. Generate Grad-CAM visualisations
    7. Save results

    Args:
        config: Configuration dictionary.
        architecture: Model architecture name.
        epochs: Number of training epochs.
        use_synthetic: Whether to use synthetic data.
        output_dir: Directory for saving results.
        logger: Logger instance.

    Returns:
        Dictionary with experiment results.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available. Cannot run experiment.")
        return {}

    from src.data.dataset_loader import CombinedDatasetLoader
    from src.data.partition import split_dataset
    from src.data.preprocessing import compute_class_weights
    from src.models.model_factory import create_model, get_gradcam_layer
    from src.models.base_model import compile_model, get_cosine_annealing_schedule
    from src.explainability.gradcam import generate_gradcam_report
    from src.evaluation.metrics import evaluate_model

    if logger:
        logger.info(f"Starting centralised experiment: arch={architecture}")

    start_time = time.time()

    # Config parameters
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    fed_config = config.get("federated", {})

    image_size = data_config.get("image_size", 224)
    batch_size = fed_config.get("batch_size", 32)

    if epochs is None:
        # Centralised equivalent: num_rounds * local_epochs
        epochs = fed_config.get("num_rounds", 20) * fed_config.get("local_epochs", 10)
        epochs = min(epochs, 100)  # Cap at 100

    # Step 1: Load data
    if use_synthetic:
        images, labels = CombinedDatasetLoader.create_synthetic_dataset(
            num_samples=200,
            image_size=(image_size, image_size),
            seed=42,
        )
    else:
        loader = CombinedDatasetLoader(config)
        images, labels = loader.load_all_data()
        if len(images) == 0:
            if logger:
                logger.warning("No data found. Using synthetic data.")
            images, labels = CombinedDatasetLoader.create_synthetic_dataset(
                num_samples=200,
                image_size=(image_size, image_size),
            )

    if logger:
        logger.info(f"Data loaded: {len(images)} samples")

    # Step 2: Split data
    splits = split_dataset(images, labels, seed=42)
    train_images, train_labels = splits["train"]
    val_images, val_labels = splits["val"]
    test_images, test_labels = splits["test"]

    # Compute class weights
    class_weights = compute_class_weights(train_labels)

    # Step 3: Build model
    model = create_model(
        architecture=architecture,
        input_shape=(image_size, image_size, 3),
        num_classes=model_config.get("num_classes", 2),
        dense_units=model_config.get("dense_units", 256),
        dropout_rate=model_config.get("dropout", 0.3),
        freeze_base=model_config.get("freeze_base", True),
        unfreeze_top=model_config.get("unfreeze_top", 20),
    )

    lr = training_config.get("optimiser", {}).get("learning_rate", 0.001)
    model = compile_model(model, learning_rate=lr)

    # Step 4: Train with callbacks
    num_classes = model_config.get("num_classes", 2)
    train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=10,
            restore_best_weights=True,
            mode="max",
        ),
        get_cosine_annealing_schedule(
            initial_lr=lr,
            min_lr=1e-6,
            total_epochs=epochs,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_images,
        train_labels_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_images, val_labels_cat),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 5: Evaluate
    test_results = evaluate_model(model, test_images, test_labels, num_classes)

    if logger:
        logger.info(f"Test accuracy: {test_results.get('accuracy', 0):.4f}")
        logger.info(f"Test AUC: {test_results.get('roc_auc', 0):.4f}")

    # Step 6: Grad-CAM analysis
    gradcam_results = {}
    if output_dir:
        try:
            target_layer = get_gradcam_layer(architecture)
            gradcam_dir = str(Path(output_dir) / "gradcam")
            gradcam_results = generate_gradcam_report(
                model=model,
                images=test_images[:10],
                labels=test_labels[:10],
                target_layer=target_layer,
                save_dir=gradcam_dir,
                num_samples=10,
            )
        except Exception as e:
            if logger:
                logger.warning(f"Grad-CAM analysis failed: {e}")

    # Step 7: Compile and save results
    total_time = time.time() - start_time

    results = {
        "experiment_type": "centralised",
        "architecture": architecture,
        "epochs": epochs,
        "test_results": test_results,
        "training_history": {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
        "gradcam_results": gradcam_results,
        "total_time": total_time,
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "best_val_auc": float(max(history.history.get("val_auc", [0]))),
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "results.json"

        serialisable = json.loads(
            json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
        )
        with open(results_file, "w") as f:
            json.dump(serialisable, f, indent=2)

        # Save model
        model_path = str(output_path / "model.h5")
        model.save(model_path)

        if logger:
            logger.info(f"Results saved to: {results_file}")
            logger.info(f"Model saved to: {model_path}")

    return results


def main():
    """Main entry point for centralised experiment."""
    args = parse_args()

    # Setup
    set_seed(42)
    set_gpu_config()

    config = load_config(args.config)
    logger = get_experiment_logger("centralised")

    output_dir = args.output_dir or str(get_results_path("centralised"))

    # Run experiment
    results = run_centralised_experiment(
        config=config,
        architecture=args.architecture,
        epochs=args.epochs,
        use_synthetic=args.synthetic,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Centralised experiment completed successfully.")
    return results


if __name__ == "__main__":
    main()
