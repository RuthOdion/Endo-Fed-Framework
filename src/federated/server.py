"""
Federated learning server implementation.

Orchestrates the federated training process including client
coordination, aggregation, privacy mechanisms, and convergence
monitoring.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.federated.client import FederatedClient
from src.federated.aggregation import fedavg_aggregate, fedprox_aggregate


class FederatedServer:
    """
    Federated learning server for orchestrating distributed training.

    Manages the federated learning loop including:
    - Broadcasting global model to clients
    - Collecting and aggregating client updates
    - Applying differential privacy
    - Secure aggregation
    - Early stopping and convergence monitoring
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        clients: List[FederatedClient],
        val_data: Tuple[np.ndarray, np.ndarray],
        num_rounds: int = 20,
        aggregation: str = "FedProx",
        mu: float = 0.01,
        dp_handler=None,
        sa_handler=None,
        early_stopping_patience: int = 5,
        logger=None,
    ):
        """
        Initialise federated server.

        Args:
            model: Global Keras model.
            clients: List of FederatedClient instances.
            val_data: Validation data tuple (images, labels).
            num_rounds: Number of federated communication rounds.
            aggregation: Aggregation strategy ('FedAvg' or 'FedProx').
            mu: FedProx proximal term coefficient.
            dp_handler: Optional DifferentialPrivacyHandler.
            sa_handler: Optional SecureAggregationHandler.
            early_stopping_patience: Rounds without improvement before stopping.
            logger: Optional logger instance.
        """
        self.model = model
        self.clients = clients
        self.val_data = val_data
        self.num_rounds = num_rounds
        self.aggregation = aggregation
        self.mu = mu
        self.dp_handler = dp_handler
        self.sa_handler = sa_handler
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger

        # Training state
        self.global_weights = model.get_weights()
        self.training_history = []
        self.best_val_metric = 0.0
        self.best_weights = None
        self.rounds_without_improvement = 0
        self.convergence_round = None

    def train(self) -> Dict:
        """
        Execute the full federated training loop.

        Process per round:
        1. Broadcast global model to all clients
        2. Clients perform local training
        3. Apply differential privacy (clip + noise)
        4. Apply secure aggregation masking
        5. Aggregate client updates
        6. Evaluate global model
        7. Check early stopping

        Returns:
            Dictionary with complete training history and metrics.
        """
        if tf is None:
            raise ImportError("TensorFlow is required for training.")

        self._log(f"Starting federated training: {self.num_rounds} rounds, "
                  f"{len(self.clients)} clients, aggregation={self.aggregation}")

        start_time = time.time()

        for round_num in range(self.num_rounds):
            round_start = time.time()
            self._log(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

            # Step 1-2: Client local training
            client_weights = []
            client_num_samples = []
            round_metrics = {"round": round_num + 1, "client_metrics": []}

            for client in self.clients:
                updated_weights, metrics = client.train_local(
                    model=self.model,
                    global_weights=self.global_weights,
                )
                client_weights.append(updated_weights)
                client_num_samples.append(client.get_num_samples())
                round_metrics["client_metrics"].append(metrics)

            # Step 3: Apply differential privacy
            if self.dp_handler is not None:
                client_weights = [
                    self.dp_handler.clip_weights(w, self.global_weights)
                    for w in client_weights
                ]
                client_weights = [
                    self.dp_handler.add_noise(w) for w in client_weights
                ]
                round_metrics["privacy_spent"] = self.dp_handler.get_privacy_spent()

            # Step 4: Apply secure aggregation
            if self.sa_handler is not None:
                masks = self.sa_handler.generate_masks(
                    num_clients=len(self.clients),
                    weight_shapes=[w.shape for w in client_weights[0]],
                )
                client_weights = [
                    self.sa_handler.mask_weights(w, masks[i])
                    for i, w in enumerate(client_weights)
                ]

            # Step 5: Aggregate
            if self.aggregation == "FedProx":
                self.global_weights = fedprox_aggregate(
                    client_weights=client_weights,
                    client_num_samples=client_num_samples,
                    global_weights=self.global_weights,
                    mu=self.mu,
                )
            else:
                self.global_weights = fedavg_aggregate(
                    client_weights=client_weights,
                    client_num_samples=client_num_samples,
                )

            # Unmask after aggregation if using SA
            if self.sa_handler is not None:
                self.global_weights = self.sa_handler.unmask_aggregate(
                    self.global_weights, masks
                )

            # Step 6: Evaluate global model
            self.model.set_weights(self.global_weights)
            val_metrics = self._evaluate_global_model()
            round_metrics.update(val_metrics)

            round_metrics["round_time"] = time.time() - round_start
            self.training_history.append(round_metrics)

            self._log(
                f"Round {round_num + 1}: "
                f"val_loss={val_metrics.get('val_loss', 0):.4f}, "
                f"val_accuracy={val_metrics.get('val_accuracy', 0):.4f}, "
                f"val_auc={val_metrics.get('val_auc', 0):.4f}"
            )

            # Step 7: Check early stopping
            if self._check_early_stopping(val_metrics):
                self._log(f"Early stopping at round {round_num + 1}")
                self.convergence_round = round_num + 1
                break

        # Restore best weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.global_weights = self.best_weights

        total_time = time.time() - start_time
        self._log(f"\nTraining completed in {total_time:.1f}s")

        return {
            "history": self.training_history,
            "total_time": total_time,
            "convergence_round": self.convergence_round,
            "best_val_metric": self.best_val_metric,
            "num_rounds_completed": len(self.training_history),
        }

    def _evaluate_global_model(self) -> Dict:
        """
        Evaluate the global model on validation data.

        Returns:
            Dictionary with validation metrics.
        """
        val_images, val_labels = self.val_data
        num_classes = self.model.output_shape[-1]
        val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes)

        results = self.model.evaluate(
            val_images, val_labels_cat, verbose=0, return_dict=True
        )

        return {
            "val_loss": float(results.get("loss", 0)),
            "val_accuracy": float(results.get("accuracy", 0)),
            "val_auc": float(results.get("auc", 0)),
            "val_precision": float(results.get("precision", 0)),
            "val_recall": float(results.get("recall", 0)),
        }

    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """
        Check if training should stop early.

        Args:
            val_metrics: Current round validation metrics.

        Returns:
            True if training should stop.
        """
        current_metric = val_metrics.get("val_auc", val_metrics.get("val_accuracy", 0))

        if current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            self.best_weights = copy.deepcopy(self.global_weights)
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        return self.rounds_without_improvement >= self.early_stopping_patience

    def save_global_model(self, path: str) -> None:
        """
        Save the global model to disk.

        Args:
            path: File path for saving the model.
        """
        if tf is None:
            return

        self.model.set_weights(self.global_weights)
        self.model.save(path)
        self._log(f"Global model saved to: {path}")

    def get_global_model(self) -> "tf.keras.Model":
        """
        Get the current global model with latest weights.

        Returns:
            Keras model with global weights applied.
        """
        self.model.set_weights(self.global_weights)
        return self.model

    def get_convergence_round(self) -> Optional[int]:
        """
        Get the round at which the model converged.

        Returns:
            Convergence round number or None if not converged.
        """
        return self.convergence_round

    def _log(self, message: str) -> None:
        """Log a message using logger or print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def create_federated_clients(
    images: np.ndarray,
    labels: np.ndarray,
    partitions: List[np.ndarray],
    local_epochs: int = 10,
    batch_size: int = 32,
) -> List[FederatedClient]:
    """
    Factory function to create federated clients from partitioned data.

    Args:
        images: Full image dataset.
        labels: Full label array.
        partitions: List of index arrays per client.
        local_epochs: Local training epochs per round.
        batch_size: Training batch size.

    Returns:
        List of FederatedClient instances.
    """
    clients = []

    for client_id, indices in enumerate(partitions):
        client_images = images[indices]
        client_labels = labels[indices]

        client = FederatedClient(
            client_id=client_id,
            train_data=(client_images, client_labels),
            local_epochs=local_epochs,
            batch_size=batch_size,
        )
        clients.append(client)

    return clients
