"""
Data partitioning for federated learning experiments.

Implements IID and non-IID (Dirichlet-based) data partitioning
strategies to simulate heterogeneous client data distributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except ImportError:
    StratifiedKFold = None
    train_test_split = None


class DataPartitioner:
    """
    Federated learning data partitioner.

    Supports IID and non-IID partitioning using Dirichlet distribution
    to simulate heterogeneous client data in federated settings.
    """

    def __init__(
        self,
        num_clients: int = 5,
        seed: int = 42,
    ):
        """
        Initialise data partitioner.

        Args:
            num_clients: Number of federated clients.
            seed: Random seed for reproducibility.
        """
        self.num_clients = num_clients
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def partition_non_iid(
        self,
        labels: np.ndarray,
        alpha: float = 0.5,
        min_samples: int = 10,
    ) -> List[np.ndarray]:
        """
        Partition data in a non-IID manner using Dirichlet distribution.

        The Dirichlet distribution with parameter alpha controls the degree
        of heterogeneity. Lower alpha = more heterogeneous.

        Args:
            labels: Array of class labels.
            alpha: Dirichlet concentration parameter.
            min_samples: Minimum samples per client.

        Returns:
            List of index arrays, one per client.
        """
        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Group indices by class
        class_indices = {}
        for cls in range(num_classes):
            class_indices[cls] = np.where(labels == cls)[0]

        # Sample Dirichlet proportions for each class
        client_indices = [[] for _ in range(self.num_clients)]

        for cls in range(num_classes):
            cls_idx = class_indices[cls]
            self.rng.shuffle(cls_idx)

            # Sample proportions from Dirichlet distribution
            proportions = self.rng.dirichlet(
                np.repeat(alpha, self.num_clients)
            )

            # Ensure minimum samples per client
            proportions = np.maximum(proportions, min_samples / len(cls_idx))
            proportions = proportions / proportions.sum()

            # Split class indices according to proportions
            splits = (proportions * len(cls_idx)).astype(int)
            # Distribute remainder
            remainder = len(cls_idx) - splits.sum()
            for i in range(remainder):
                splits[i % self.num_clients] += 1

            current = 0
            for client_id in range(self.num_clients):
                end = current + splits[client_id]
                client_indices[client_id].extend(cls_idx[current:end].tolist())
                current = end

        # Convert to numpy arrays
        result = []
        for indices in client_indices:
            arr = np.array(indices, dtype=np.int64)
            self.rng.shuffle(arr)
            result.append(arr)

        return result

    def partition_iid(self, labels: np.ndarray) -> List[np.ndarray]:
        """
        Partition data in an IID manner (uniform random).

        Args:
            labels: Array of class labels.

        Returns:
            List of index arrays, one per client.
        """
        num_samples = len(labels)
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)

        # Split evenly
        splits = np.array_split(indices, self.num_clients)
        return [split for split in splits]

    def get_partition_statistics(
        self,
        partitions: List[np.ndarray],
        labels: np.ndarray,
    ) -> Dict:
        """
        Compute statistics for data partitions.

        Args:
            partitions: List of index arrays per client.
            labels: Full array of labels.

        Returns:
            Dictionary with partition statistics.
        """
        stats = {
            "num_clients": self.num_clients,
            "total_samples": len(labels),
            "clients": [],
        }

        for i, indices in enumerate(partitions):
            client_labels = labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)

            client_stats = {
                "client_id": i,
                "num_samples": len(indices),
                "proportion": len(indices) / len(labels),
                "class_distribution": {
                    int(cls): int(count) for cls, count in zip(unique, counts)
                },
            }
            stats["clients"].append(client_stats)

        return stats


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train/validation/test sets with stratification.

    Args:
        images: Image array (N, H, W, C).
        labels: Label array (N,).
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed.

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        (images, labels) tuple.
    """
    if train_test_split is None:
        raise ImportError("scikit-learn required for stratified splitting.")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )

    # Second split: val vs test
    test_proportion = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_proportion, random_state=seed, stratify=y_temp
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def create_kfold_splits(
    images: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Create stratified K-fold cross-validation splits.

    Args:
        images: Image array (N, H, W, C).
        labels: Label array (N,).
        n_splits: Number of folds.
        seed: Random seed.

    Returns:
        List of dictionaries, each with 'train' and 'val' keys.
    """
    if StratifiedKFold is None:
        raise ImportError("scikit-learn required for K-fold splitting.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_idx, val_idx in skf.split(images, labels):
        fold = {
            "train": (images[train_idx], labels[train_idx]),
            "val": (images[val_idx], labels[val_idx]),
        }
        folds.append(fold)

    return folds


def save_partition_info(
    partitions: List[np.ndarray],
    labels: np.ndarray,
    save_path: str,
) -> None:
    """
    Save partition information to a JSON file for reproducibility.

    Args:
        partitions: List of index arrays per client.
        labels: Full array of labels.
        save_path: Path to save the JSON file.
    """
    partitioner = DataPartitioner(num_clients=len(partitions))
    stats = partitioner.get_partition_statistics(partitions, labels)

    # Add index information
    stats["partition_indices"] = [
        indices.tolist() for indices in partitions
    ]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
