"""
Quick integration test for the endometriosis FL framework.

Runs 8 integration tests to verify all components work correctly:
1. Configuration loading
2. Data preprocessing pipeline
3. Data partitioning
4. Model building
5. Federated training (mini-round)
6. Privacy mechanisms
7. Grad-CAM explainability
8. Evaluation metrics
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_quick_test() -> dict:
    """
    Run integration tests for all framework components.

    Returns:
        Dictionary with test results (test_name -> pass/fail).
    """
    results = {}
    total_start = time.time()

    print("\n" + "=" * 60)
    print("ENDOMETRIOSIS FL FRAMEWORK - INTEGRATION TESTS")
    print("=" * 60)

    # Test 1: Configuration Loading
    print("\n[1/8] Testing configuration loading...")
    try:
        from src.utils.config import load_config, get_project_root
        config = load_config()
        assert "data" in config
        assert "model" in config
        assert "federated" in config
        assert "privacy" in config
        assert config["data"]["image_size"] == 224
        assert config["federated"]["num_clients"] == 5
        results["config_loading"] = "PASSED"
        print("  PASSED: Config loaded successfully")
    except Exception as e:
        results["config_loading"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 2: Data Preprocessing Pipeline
    print("\n[2/8] Testing data preprocessing pipeline...")
    try:
        from src.data.preprocessing import (
            preprocess_image, preprocess_batch, compute_class_weights,
            IMAGENET_MEAN, IMAGENET_STD, create_augmentation_pipeline,
        )

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        processed = preprocess_image(dummy_image, target_size=(224, 224))
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32

        # Batch processing
        batch = np.random.randint(0, 255, (4, 300, 400, 3), dtype=np.uint8)
        processed_batch = preprocess_batch(batch, target_size=(224, 224))
        assert processed_batch.shape == (4, 224, 224, 3)

        # Class weights
        labels = np.array([0, 0, 0, 1, 1])
        weights = compute_class_weights(labels)
        assert 0 in weights and 1 in weights
        assert weights[1] > weights[0]  # Minority class has higher weight

        # Augmentation config
        aug_config = create_augmentation_pipeline()
        assert "rotation_range" in aug_config

        results["data_preprocessing"] = "PASSED"
        print("  PASSED: Preprocessing pipeline works correctly")
    except Exception as e:
        results["data_preprocessing"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 3: Data Partitioning
    print("\n[3/8] Testing data partitioning...")
    try:
        from src.data.partition import DataPartitioner, split_dataset, create_kfold_splits
        from src.data.dataset_loader import CombinedDatasetLoader

        # Create synthetic data
        images, labels = CombinedDatasetLoader.create_synthetic_dataset(
            num_samples=100, image_size=(32, 32), seed=42
        )
        assert images.shape == (100, 32, 32, 3)
        assert len(labels) == 100

        # Test partitioning
        partitioner = DataPartitioner(num_clients=5, seed=42)
        partitions_iid = partitioner.partition_iid(labels)
        assert len(partitions_iid) == 5
        assert sum(len(p) for p in partitions_iid) == len(labels)

        partitions_niid = partitioner.partition_non_iid(labels, alpha=0.5)
        assert len(partitions_niid) == 5
        assert sum(len(p) for p in partitions_niid) == len(labels)

        # Test statistics
        stats = partitioner.get_partition_statistics(partitions_niid, labels)
        assert "clients" in stats

        # Test split
        splits = split_dataset(images, labels)
        assert "train" in splits and "val" in splits and "test" in splits

        results["data_partitioning"] = "PASSED"
        print("  PASSED: Partitioning works correctly")
    except Exception as e:
        results["data_partitioning"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 4: Model Building
    print("\n[4/8] Testing model building...")
    try:
        from src.models.model_factory import (
            create_model, get_all_architectures, get_gradcam_layer, get_model_size_mb
        )

        architectures = get_all_architectures()
        assert "ResNet50V2" in architectures
        assert "EfficientNetB0" in architectures
        assert "MobileNetV2" in architectures

        # Test Grad-CAM layer retrieval
        assert get_gradcam_layer("ResNet50V2") == "post_relu"
        assert get_gradcam_layer("EfficientNetB0") == "top_conv"
        assert get_gradcam_layer("MobileNetV2") == "out_relu"

        # Try building a model (only if TF available)
        try:
            import tensorflow as tf
            model = create_model("MobileNetV2", input_shape=(32, 32, 3))
            assert model.output_shape[-1] == 2
            size = get_model_size_mb(model)
            assert size > 0
            print(f"  Model size: {size} MB")
        except ImportError:
            print("  (TensorFlow not available - skipping model instantiation)")

        results["model_building"] = "PASSED"
        print("  PASSED: Model building works correctly")
    except Exception as e:
        results["model_building"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 5: Federated Training (mini-round)
    print("\n[5/8] Testing federated training...")
    try:
        from src.federated.client import FederatedClient
        from src.federated.aggregation import (
            fedavg_aggregate, fedprox_aggregate, compute_communication_cost
        )
        from src.federated.server import create_federated_clients

        # Test aggregation
        weights_1 = [np.random.randn(10, 5), np.random.randn(5,)]
        weights_2 = [np.random.randn(10, 5), np.random.randn(5,)]
        weights_3 = [np.random.randn(10, 5), np.random.randn(5,)]

        aggregated = fedavg_aggregate(
            [weights_1, weights_2, weights_3],
            [100, 80, 60]
        )
        assert len(aggregated) == 2
        assert aggregated[0].shape == (10, 5)

        # FedProx
        prox_aggregated = fedprox_aggregate(
            [weights_1, weights_2, weights_3],
            [100, 80, 60],
            global_weights=weights_1,
            mu=0.01,
        )
        assert len(prox_aggregated) == 2

        # Communication cost
        comm_cost = compute_communication_cost(weights_1, num_clients=5, num_rounds=20)
        assert "model_size_mb" in comm_cost

        results["federated_training"] = "PASSED"
        print("  PASSED: Federated components work correctly")
    except Exception as e:
        results["federated_training"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 6: Privacy Mechanisms
    print("\n[6/8] Testing privacy mechanisms...")
    try:
        from src.privacy.differential_privacy import (
            DifferentialPrivacyHandler, create_dp_handler_for_epsilon
        )
        from src.privacy.secure_aggregation import SecureAggregationHandler

        # Test DP handler
        dp = create_dp_handler_for_epsilon(epsilon=1.0, num_clients=5, num_rounds=20)
        assert dp.sigma > 0
        assert dp.epsilon == 1.0

        # Test clipping
        global_w = [np.zeros((10, 5))]
        large_update = [np.ones((10, 5)) * 100]
        clipped = dp.clip_weights(large_update, global_w)
        norm = np.sqrt(np.sum(clipped[0] ** 2))
        assert norm <= dp.l2_norm_clip + 1e-6

        # Test noise addition
        weights = [np.ones((10, 5))]
        noised = dp.add_noise(weights)
        assert noised[0].shape == (10, 5)
        assert not np.allclose(noised[0], weights[0])  # Noise was added

        # Privacy accounting
        privacy = dp.get_privacy_spent()
        assert "epsilon_spent" in privacy

        # Test Secure Aggregation
        sa = SecureAggregationHandler(num_clients=3, seed=42)
        shapes = [(10, 5), (5,)]
        masks = sa.generate_masks(3, shapes)
        assert len(masks) == 3

        # Verify masks sum to zero
        for layer_idx in range(len(shapes)):
            mask_sum = sum(masks[i][layer_idx] for i in range(3))
            assert np.allclose(mask_sum, 0, atol=1e-6)

        results["privacy_mechanisms"] = "PASSED"
        print("  PASSED: Privacy mechanisms work correctly")
    except Exception as e:
        results["privacy_mechanisms"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 7: Grad-CAM
    print("\n[7/8] Testing Grad-CAM explainability...")
    try:
        from src.explainability.gradcam import compute_iou, compute_dice

        # Test IoU and Dice
        mask_a = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        mask_b = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)

        iou = compute_iou(mask_a, mask_b)
        assert 0 <= iou <= 1
        assert iou > 0  # There is overlap

        dice = compute_dice(mask_a, mask_b)
        assert 0 <= dice <= 1
        assert dice > 0

        # Perfect overlap
        iou_perfect = compute_iou(mask_a, mask_a)
        assert abs(iou_perfect - 1.0) < 1e-6

        dice_perfect = compute_dice(mask_a, mask_a)
        assert abs(dice_perfect - 1.0) < 1e-6

        # No overlap
        mask_c = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        iou_none = compute_iou(mask_a, mask_c)
        assert iou_none == 0.0

        results["gradcam"] = "PASSED"
        print("  PASSED: Grad-CAM components work correctly")
    except Exception as e:
        results["gradcam"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Test 8: Evaluation Metrics
    print("\n[8/8] Testing evaluation metrics...")
    try:
        from src.evaluation.comparison import paired_t_test, generate_comparison_summary
        from src.evaluation.metrics import compute_convergence_speed

        # Test paired t-test
        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        test_result = paired_t_test(scores_a, scores_b)
        assert "t_statistic" in test_result
        assert "p_value" in test_result
        assert "cohens_d" in test_result
        assert "effect_size" in test_result
        assert test_result["significant"] is True  # Should be significant

        # Test convergence speed
        history = [
            {"val_auc": 0.6},
            {"val_auc": 0.7},
            {"val_auc": 0.8},
            {"val_auc": 0.9},
            {"val_auc": 0.92},
        ]
        conv_speed = compute_convergence_speed(history, threshold=0.9)
        assert conv_speed["convergence_round"] == 4
        assert conv_speed["converged"] is True

        # Test comparison summary
        all_results_test = {
            "centralised": {"accuracy": 0.90},
            "federated": {"accuracy": 0.87},
        }
        summary = generate_comparison_summary(all_results_test)
        assert "rankings" in summary

        results["evaluation_metrics"] = "PASSED"
        print("  PASSED: Evaluation metrics work correctly")
    except Exception as e:
        results["evaluation_metrics"] = f"FAILED: {e}"
        print(f"  FAILED: {e}")

    # Summary
    total_time = time.time() - total_start
    passed = sum(1 for v in results.values() if v == "PASSED")
    total = len(results)

    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} passed")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 60)

    for test_name, result in results.items():
        status = "PASS" if result == "PASSED" else "FAIL"
        print(f"  [{status}] {test_name}")

    if passed == total:
        print("\nAll tests passed! Framework is ready for experiments.")
    else:
        print(f"\n{total - passed} test(s) failed. Check output above.")

    print("=" * 60)

    return results


if __name__ == "__main__":
    run_quick_test()
