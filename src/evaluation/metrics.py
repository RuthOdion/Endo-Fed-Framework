"""
Evaluation metrics for endometriosis classification.

Computes accuracy, precision, recall, F1 score, AUC-ROC, and
confusion matrix with support for K-fold cross-validation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    accuracy_score = None


def evaluate_model(
    model: "tf.keras.Model",
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int = 2,
) -> Dict:
    """
    Comprehensive model evaluation with all metrics.

    Computes: accuracy, precision, recall, F1, AUC-ROC, confusion matrix.

    Args:
        model: Trained Keras model.
        test_images: Test images (N, H, W, 3).
        test_labels: True labels (N,) integer encoded.
        num_classes: Number of classes.

    Returns:
        Dictionary with all evaluation metrics.
    """
    if tf is None:
        raise ImportError("TensorFlow required for evaluation.")

    # Get predictions
    predictions = model.predict(test_images, verbose=0)

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions[:, 1]  # Probability of positive class
    else:
        pred_classes = (predictions.flatten() > 0.5).astype(int)
        pred_probs = predictions.flatten()

    # Ensure labels are integer
    true_labels = test_labels.astype(int)

    results = {}

    if accuracy_score is not None:
        results["accuracy"] = float(accuracy_score(true_labels, pred_classes))
        results["precision"] = float(precision_score(
            true_labels, pred_classes, average="binary", zero_division=0
        ))
        results["recall"] = float(recall_score(
            true_labels, pred_classes, average="binary", zero_division=0
        ))
        results["f1_score"] = float(f1_score(
            true_labels, pred_classes, average="binary", zero_division=0
        ))

        try:
            results["roc_auc"] = float(roc_auc_score(true_labels, pred_probs))
        except ValueError:
            results["roc_auc"] = 0.0

        cm = confusion_matrix(true_labels, pred_classes)
        results["confusion_matrix"] = cm.tolist()

        # Specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            results["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        # Fallback without sklearn
        results["accuracy"] = float(np.mean(pred_classes == true_labels))
        results["precision"] = 0.0
        results["recall"] = 0.0
        results["f1_score"] = 0.0
        results["roc_auc"] = 0.0
        results["confusion_matrix"] = []

    results["num_samples"] = len(test_labels)
    results["predictions"] = pred_probs.tolist()

    return results


def evaluate_kfold(
    model_builder,
    images: np.ndarray,
    labels: np.ndarray,
    k_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict:
    """
    Evaluate model using K-fold cross-validation.

    Args:
        model_builder: Callable that returns a compiled Keras model.
        images: Full image dataset.
        labels: Full label array.
        k_folds: Number of folds.
        epochs: Training epochs per fold.
        batch_size: Training batch size.
        seed: Random seed.

    Returns:
        Dictionary with per-fold and aggregated metrics.
    """
    if tf is None:
        raise ImportError("TensorFlow required for K-fold evaluation.")

    from src.data.partition import create_kfold_splits

    folds = create_kfold_splits(images, labels, n_splits=k_folds, seed=seed)
    fold_results = []

    for fold_idx, fold_data in enumerate(folds):
        print(f"Evaluating fold {fold_idx + 1}/{k_folds}...")

        # Build fresh model
        model = model_builder()

        train_images, train_labels = fold_data["train"]
        val_images, val_labels = fold_data["val"]

        num_classes = model.output_shape[-1]
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes)
        val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes)

        # Train
        model.fit(
            train_images,
            train_labels_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_images, val_labels_cat),
            verbose=0,
        )

        # Evaluate
        fold_metrics = evaluate_model(model, val_images, val_labels, num_classes)
        fold_metrics["fold"] = fold_idx
        fold_results.append(fold_metrics)

    # Aggregate results
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    aggregated = {}

    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            aggregated[f"mean_{key}"] = float(np.mean(values))
            aggregated[f"std_{key}"] = float(np.std(values))

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
        "k_folds": k_folds,
    }


def compute_convergence_speed(
    training_history: List[Dict],
    metric: str = "val_auc",
    threshold: float = 0.9,
) -> Dict:
    """
    Compute convergence speed from training history.

    Args:
        training_history: List of round metrics dictionaries.
        metric: Metric to monitor for convergence.
        threshold: Target metric value for convergence.

    Returns:
        Dictionary with convergence analysis.
    """
    values = []
    for entry in training_history:
        val = entry.get(metric, 0.0)
        values.append(val)

    convergence_round = None
    for i, val in enumerate(values):
        if val >= threshold:
            convergence_round = i + 1
            break

    # Compute improvement rates
    improvements = []
    for i in range(1, len(values)):
        improvements.append(values[i] - values[i - 1])

    return {
        "convergence_round": convergence_round,
        "converged": convergence_round is not None,
        "final_value": float(values[-1]) if values else 0.0,
        "max_value": float(max(values)) if values else 0.0,
        "mean_improvement_per_round": float(np.mean(improvements)) if improvements else 0.0,
        "total_rounds": len(values),
        "threshold": threshold,
        "metric": metric,
    }
