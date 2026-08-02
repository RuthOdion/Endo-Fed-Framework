"""
Visualisation utilities for experiment results.

Generates publication-quality plots for convergence curves,
ROC curves, confusion matrices, and comparative analyses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


def plot_convergence_curves(
    histories: Dict[str, List[float]],
    metric: str = "val_accuracy",
    save_path: Optional[str] = None,
    title: str = "Training Convergence",
) -> None:
    """
    Plot convergence curves for multiple experimental conditions.

    Args:
        histories: Dict mapping condition name to list of metric values per round.
        metric: Name of the metric being plotted.
        save_path: Path to save figure. If None, displays plot.
        title: Plot title.
    """
    if plt is None:
        print("matplotlib required for plotting.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, values in histories.items():
        rounds = list(range(1, len(values) + 1))
        ax.plot(rounds, values, marker="o", markersize=3, label=name)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.close()


def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "ROC Curves Comparison",
) -> None:
    """
    Plot ROC curves for multiple models/conditions.

    Args:
        results: Dict mapping condition name to results with predictions and labels.
        save_path: Path to save figure.
        title: Plot title.
    """
    if plt is None:
        return

    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("scikit-learn required for ROC curves.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for name, result in results.items():
        if "predictions" in result and "true_labels" in result:
            fpr, tpr, _ = roc_curve(result["true_labels"], result["predictions"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (num_classes x num_classes).
        class_names: Class label names.
        save_path: Path to save figure.
        title: Plot title.
    """
    if plt is None:
        return

    if class_names is None:
        class_names = ["Non-pathological", "Pathological"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_privacy_tradeoff(
    epsilon_values: List[float],
    accuracies: List[float],
    baseline_accuracy: float,
    save_path: Optional[str] = None,
    title: str = "Privacy-Performance Tradeoff",
) -> None:
    """
    Plot privacy (epsilon) vs performance tradeoff.

    Args:
        epsilon_values: List of epsilon values.
        accuracies: Corresponding accuracy values.
        baseline_accuracy: Non-private baseline accuracy.
        save_path: Path to save figure.
        title: Plot title.
    """
    if plt is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(epsilon_values, accuracies, "b-o", linewidth=2, markersize=8, label="DP-FL")
    ax.axhline(
        y=baseline_accuracy, color="r", linestyle="--", linewidth=2,
        label=f"Non-private baseline ({baseline_accuracy:.3f})"
    )

    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate regions
    ax.axvspan(0, 1, alpha=0.1, color="green", label="Strong privacy")
    ax.axvspan(1, 5, alpha=0.1, color="yellow")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_architecture_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Architecture Comparison",
) -> None:
    """
    Plot bar chart comparing architectures across metrics.

    Args:
        results: Dict mapping architecture to metrics dict.
        metrics: Metrics to compare.
        save_path: Path to save figure.
        title: Plot title.
    """
    if plt is None:
        return

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    architectures = list(results.keys())
    num_metrics = len(metrics)
    x = np.arange(num_metrics)
    width = 0.8 / len(architectures)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, arch in enumerate(architectures):
        values = [results[arch].get(m, 0) for m in metrics]
        offset = (i - len(architectures) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=arch)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_client_data_distribution(
    partition_stats: Dict,
    save_path: Optional[str] = None,
    title: str = "Client Data Distribution",
) -> None:
    """
    Plot data distribution across federated clients.

    Args:
        partition_stats: Partition statistics dictionary.
        save_path: Path to save figure.
        title: Plot title.
    """
    if plt is None:
        return

    clients = partition_stats.get("clients", [])
    if not clients:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Sample counts per client
    client_ids = [f"Client {c['client_id']}" for c in clients]
    num_samples = [c["num_samples"] for c in clients]

    ax1.bar(client_ids, num_samples, color="steelblue")
    ax1.set_xlabel("Client", fontsize=12)
    ax1.set_ylabel("Number of Samples", fontsize=12)
    ax1.set_title("Samples per Client", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Class distribution per client (stacked bar)
    class_0_counts = []
    class_1_counts = []
    for c in clients:
        dist = c.get("class_distribution", {})
        class_0_counts.append(dist.get(0, 0))
        class_1_counts.append(dist.get(1, 0))

    ax2.bar(client_ids, class_0_counts, label="Non-pathological", color="lightblue")
    ax2.bar(client_ids, class_1_counts, bottom=class_0_counts,
            label="Pathological", color="salmon")
    ax2.set_xlabel("Client", fontsize=12)
    ax2.set_ylabel("Number of Samples", fontsize=12)
    ax2.set_title("Class Distribution per Client", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    results: Dict,
    save_dir: str,
) -> List[str]:
    """
    Generate all visualisation plots from experiment results.

    Args:
        results: Complete experiment results dictionary.
        save_dir: Directory to save all figures.

    Returns:
        List of saved figure paths.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Convergence curves
    if "convergence_histories" in results:
        path = str(save_path / "convergence_curves.png")
        plot_convergence_curves(results["convergence_histories"], save_path=path)
        saved_files.append(path)

    # Confusion matrix
    if "confusion_matrix" in results:
        path = str(save_path / "confusion_matrix.png")
        cm = np.array(results["confusion_matrix"])
        plot_confusion_matrix(cm, save_path=path)
        saved_files.append(path)

    # Privacy tradeoff
    if "privacy_tradeoff" in results:
        pt = results["privacy_tradeoff"]
        path = str(save_path / "privacy_tradeoff.png")
        plot_privacy_tradeoff(
            epsilon_values=pt.get("epsilons", []),
            accuracies=pt.get("accuracies", []),
            baseline_accuracy=pt.get("baseline", 0.0),
            save_path=path,
        )
        saved_files.append(path)

    # Architecture comparison
    if "architecture_results" in results:
        path = str(save_path / "architecture_comparison.png")
        plot_architecture_comparison(results["architecture_results"], save_path=path)
        saved_files.append(path)

    # Client distribution
    if "partition_stats" in results:
        path = str(save_path / "client_distribution.png")
        plot_client_data_distribution(results["partition_stats"], save_path=path)
        saved_files.append(path)

    return saved_files
