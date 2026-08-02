"""
Federated learning client implementation.

Each client represents a hospital/institution with local data
that trains a model locally and communicates weight updates.
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


class FederatedClient:
    """
    Federated learning client for local model training.

    Represents a single institution in the federated learning setup,
    performing local training on private data and returning weight updates.
    """

    def __init__(
        self,
        client_id: int,
        train_data: Tuple[np.ndarray, np.ndarray],
        local_epochs: int = 10,
        batch_size: int = 32,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Initialise federated client.

        Args:
            client_id: Unique identifier for this client.
            train_data: Tuple of (images, labels) for local training.
            local_epochs: Number of local training epochs per round.
            batch_size: Training batch size.
            val_data: Optional local validation data.
        """
        self.client_id = client_id
        self.train_images, self.train_labels = train_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.val_data = val_data
        self._training_history = []

    def train_local(
        self,
        model: "tf.keras.Model",
        global_weights: List[np.ndarray],
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Perform local training on client's data.

        Args:
            model: Keras model to train.
            global_weights: Current global model weights to initialise from.
            class_weights: Optional class weights for imbalanced data.

        Returns:
            Tuple of (updated_weights, training_metrics).
        """
        if tf is None:
            raise ImportError("TensorFlow is required for training.")

        # Set model to global weights
        model.set_weights(global_weights)

        # Convert labels to categorical
        num_classes = model.output_shape[-1]
        train_labels_cat = tf.keras.utils.to_categorical(
            self.train_labels, num_classes=num_classes
        )

        # Prepare validation data
        validation_data = None
        if self.val_data is not None:
            val_images, val_labels = self.val_data
            val_labels_cat = tf.keras.utils.to_categorical(
                val_labels, num_classes=num_classes
            )
            validation_data = (val_images, val_labels_cat)

        # Train locally
        history = model.fit(
            self.train_images,
            train_labels_cat,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=0,
        )

        # Collect metrics
        metrics = {
            "client_id": self.client_id,
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "num_samples": len(self.train_labels),
        }

        if "val_loss" in history.history:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            metrics["val_accuracy"] = float(history.history["val_accuracy"][-1])

        self._training_history.append(metrics)

        return model.get_weights(), metrics

    def compute_weight_update(
        self,
        global_weights: List[np.ndarray],
        local_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Compute the weight update (delta) from local training.

        Args:
            global_weights: Original global model weights.
            local_weights: Weights after local training.

        Returns:
            List of weight update arrays (local - global).
        """
        updates = []
        for gw, lw in zip(global_weights, local_weights):
            updates.append(lw - gw)
        return updates

    def get_num_samples(self) -> int:
        """
        Get the number of training samples for this client.

        Returns:
            Number of local training samples.
        """
        return len(self.train_labels)

    def get_data_distribution(self) -> Dict[int, int]:
        """
        Get the class distribution of local data.

        Returns:
            Dictionary mapping class index to count.
        """
        unique, counts = np.unique(self.train_labels, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}
