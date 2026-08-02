"""
Federated learning experiment runner.

Executes the complete federated learning pipeline including data
loading, partitioning, model creation, privacy mechanisms,
training, evaluation, and Grad-CAM analysis.
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
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
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
        "--epsilon", type=float, default=1.0,
        help="Differential privacy epsilon"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=None,
        help="Number of federated rounds (overrides config)"
    )
    parser.add_argument(
        "--non_iid", action="store_true",
        help="Use non-IID data partitioning"
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


def run_single_experiment(
    config: dict,
    architecture: str = "ResNet50V2",
    epsilon: float = 1.0,
    non_iid: bool = True,
    use_synthetic: bool = False,
    output_dir: str = None,
    logger=None,
) -> dict:
    """
    Run a single federated learning experiment.

    Pipeline:
    1. Load/create data
    2. Partition data across clients
    3. Create model
    4. Initialise privacy mechanisms (DP + SA)
    5. Create federated clients and server
    6. Train federated model
    7. Evaluate on test set
    8. Generate Grad-CAM visualisations
    9. Save results

    Args:
        config: Configuration dictionary.
        architecture: Model architecture name.
        epsilon: DP epsilon value.
        non_iid: Whether to use non-IID partitioning.
        use_synthetic: Whether to use synthetic data.
        output_dir: Directory for saving results.
        logger: Logger instance.

    Returns:
        Dictionary with complete experiment results.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available. Cannot run experiment.")
        return {}

    from src.data.dataset_loader import CombinedDatasetLoader
    from src.data.partition import DataPartitioner, split_dataset
    from src.data.preprocessing import compute_class_weights
    from src.models.model_factory import create_model, get_gradcam_layer
    from src.models.base_model import compile_model
    from src.federated.server import FederatedServer, create_federated_clients
    from src.privacy.differential_privacy import create_dp_handler_for_epsilon
    from src.privacy.secure_aggregation import SecureAggregationHandler
    from src.explainability.gradcam import GradCAM, generate_gradcam_report
    from src.evaluation.metrics import evaluate_model

    if logger:
        logger.info(f"Starting federated experiment: arch={architecture}, "
                    f"epsilon={epsilon}, non_iid={non_iid}")

    start_time = time.time()

    # Extract config parameters
    fed_config = config.get("federated", {})
    data_config = config.get("data", {})
    privacy_config = config.get("privacy", {})
    model_config = config.get("model", {})

    num_clients = fed_config.get("num_clients", 5)
    num_rounds = fed_config.get("num_rounds", 20)
    local_epochs = fed_config.get("local_epochs", 10)
    batch_size = fed_config.get("batch_size", 32)
    image_size = data_config.get("image_size", 224)

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

    # Step 2: Split into train/val/test
    splits = split_dataset(images, labels, seed=42)
    train_images, train_labels = splits["train"]
    val_images, val_labels = splits["val"]
    test_images, test_labels = splits["test"]

    # Step 3: Partition training data
    partitioner = DataPartitioner(num_clients=num_clients, seed=42)
    if non_iid:
        alpha = fed_config.get("non_iid", {}).get("dirichlet_alpha", 0.5)
        partitions = partitioner.partition_non_iid(train_labels, alpha=alpha)
    else:
        partitions = partitioner.partition_iid(train_labels)

    partition_stats = partitioner.get_partition_statistics(partitions, train_labels)

    # Step 4: Create model
    model = create_model(
        architecture=architecture,
        input_shape=(image_size, image_size, 3),
        num_classes=model_config.get("num_classes", 2),
        dense_units=model_config.get("dense_units", 256),
        dropout_rate=model_config.get("dropout", 0.3),
        freeze_base=model_config.get("freeze_base", True),
        unfreeze_top=model_config.get("unfreeze_top", 20),
    )
    model = compile_model(model, learning_rate=0.001)

    # Step 5: Initialise privacy mechanisms
    dp_config = privacy_config.get("differential_privacy", {})
    dp_handler = None
    if dp_config.get("enabled", True):
        dp_handler = create_dp_handler_for_epsilon(
            epsilon=epsilon,
            delta=dp_config.get("delta", 1e-5),
            l2_norm_clip=dp_config.get("l2_norm_clip", 1.0),
            num_clients=num_clients,
            num_rounds=num_rounds,
        )

    sa_config = privacy_config.get("secure_aggregation", {})
    sa_handler = None
    if sa_config.get("enabled", True):
        sa_handler = SecureAggregationHandler(
            num_clients=num_clients,
            seed=42,
        )

    # Step 6: Create federated clients and server
    clients = create_federated_clients(
        images=train_images,
        labels=train_labels,
        partitions=partitions,
        local_epochs=local_epochs,
        batch_size=batch_size,
    )

    server = FederatedServer(
        model=model,
        clients=clients,
        val_data=(val_images, val_labels),
        num_rounds=num_rounds,
        aggregation=fed_config.get("aggregation", "FedProx"),
        mu=fed_config.get("fedprox_mu", 0.01),
        dp_handler=dp_handler,
        sa_handler=sa_handler,
        early_stopping_patience=fed_config.get("early_stopping", {}).get("patience", 5),
        logger=logger,
    )

    # Step 7: Train
    training_results = server.train()

    # Step 8: Evaluate
    final_model = server.get_global_model()
    test_results = evaluate_model(final_model, test_images, test_labels)

    if logger:
        logger.info(f"Test accuracy: {test_results.get('accuracy', 0):.4f}")
        logger.info(f"Test AUC: {test_results.get('roc_auc', 0):.4f}")

    # Step 9: Grad-CAM analysis
    gradcam_results = {}
    if output_dir:
        try:
            target_layer = get_gradcam_layer(architecture)
            gradcam_dir = str(Path(output_dir) / "gradcam")
            gradcam_results = generate_gradcam_report(
                model=final_model,
                images=test_images[:10],
                labels=test_labels[:10],
                target_layer=target_layer,
                save_dir=gradcam_dir,
                num_samples=10,
            )
        except Exception as e:
            if logger:
                logger.warning(f"Grad-CAM analysis failed: {e}")

    # Step 10: Compile results
    total_time = time.time() - start_time

    results = {
        "experiment_type": "federated",
        "architecture": architecture,
        "epsilon": epsilon,
        "non_iid": non_iid,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "training_results": training_results,
        "test_results": test_results,
        "partition_stats": partition_stats,
        "gradcam_results": gradcam_results,
        "total_time": total_time,
        "convergence_round": training_results.get("convergence_round"),
    }

    if dp_handler:
        results["privacy_spent"] = dp_handler.get_privacy_spent()

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "results.json"

        # Convert numpy types for JSON serialisation
        serialisable = json.loads(
            json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
        )
        with open(results_file, "w") as f:
            json.dump(serialisable, f, indent=2)

        if logger:
            logger.info(f"Results saved to: {results_file}")

    return results


def main():
    """Main entry point for federated experiment."""
    args = parse_args()

    # Setup
    set_seed(42)
    set_gpu_config()

    config = load_config(args.config)
    logger = get_experiment_logger("federated")

    # Override config with CLI arguments
    if args.num_rounds:
        config.setdefault("federated", {})["num_rounds"] = args.num_rounds

    output_dir = args.output_dir or str(get_results_path("federated"))

    # Run experiment
    results = run_single_experiment(
        config=config,
        architecture=args.architecture,
        epsilon=args.epsilon,
        non_iid=args.non_iid,
        use_synthetic=args.synthetic,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Federated experiment completed successfully.")
    return results


if __name__ == "__main__":
    main()
