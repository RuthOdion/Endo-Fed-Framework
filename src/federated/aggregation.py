"""
Federated aggregation strategies.

Implements FedAvg and FedProx aggregation methods for combining
client model updates in federated learning.
"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_num_samples: List[int],
) -> List[np.ndarray]:
    """
    Federated Averaging (FedAvg) aggregation.

    Computes weighted average of client models proportional to
    their dataset sizes.

    Args:
        client_weights: List of weight arrays for each client.
        client_num_samples: Number of samples per client.

    Returns:
        Aggregated global model weights.
    """
    total_samples = sum(client_num_samples)

    # Weighted average
    aggregated_weights = []
    num_layers = len(client_weights[0])

    for layer_idx in range(num_layers):
        layer_weights = np.zeros_like(client_weights[0][layer_idx])

        for client_idx, weights in enumerate(client_weights):
            weight_factor = client_num_samples[client_idx] / total_samples
            layer_weights += weights[layer_idx] * weight_factor

        aggregated_weights.append(layer_weights)

    return aggregated_weights


def fedprox_aggregate(
    client_weights: List[List[np.ndarray]],
    client_num_samples: List[int],
    global_weights: Optional[List[np.ndarray]] = None,
    mu: float = 0.01,
) -> List[np.ndarray]:
    """
    FedProx aggregation with proximal term regularisation.

    Combines FedAvg with a proximal term that penalises deviation
    from the global model, improving convergence with heterogeneous data.

    Args:
        client_weights: List of weight arrays for each client.
        client_num_samples: Number of samples per client.
        global_weights: Current global model weights (for proximal term).
        mu: Proximal term coefficient.

    Returns:
        Aggregated global model weights.
    """
    # First compute FedAvg
    aggregated_weights = fedavg_aggregate(client_weights, client_num_samples)

    # Apply proximal term if global weights provided
    if global_weights is not None and mu > 0:
        for layer_idx in range(len(aggregated_weights)):
            # Proximal regularisation: move towards global model
            proximal_term = mu * (
                aggregated_weights[layer_idx] - global_weights[layer_idx]
            )
            aggregated_weights[layer_idx] -= proximal_term

    return aggregated_weights


def fedavg_aggregate_updates(
    client_updates: List[List[np.ndarray]],
    client_num_samples: List[int],
    global_weights: List[np.ndarray],
) -> List[np.ndarray]:
    """
    FedAvg aggregation using weight updates (deltas) instead of full weights.

    Args:
        client_updates: List of weight update arrays for each client.
        client_num_samples: Number of samples per client.
        global_weights: Current global model weights.

    Returns:
        New global model weights after applying averaged updates.
    """
    total_samples = sum(client_num_samples)

    # Aggregate updates
    aggregated_updates = []
    num_layers = len(client_updates[0])

    for layer_idx in range(num_layers):
        layer_update = np.zeros_like(client_updates[0][layer_idx])

        for client_idx, updates in enumerate(client_updates):
            weight_factor = client_num_samples[client_idx] / total_samples
            layer_update += updates[layer_idx] * weight_factor

        aggregated_updates.append(layer_update)

    # Apply aggregated updates to global weights
    new_weights = []
    for gw, update in zip(global_weights, aggregated_updates):
        new_weights.append(gw + update)

    return new_weights


def compute_aggregation_statistics(
    client_weights: List[List[np.ndarray]],
    aggregated_weights: List[np.ndarray],
) -> Dict:
    """
    Compute statistics about the aggregation process.

    Args:
        client_weights: Client weight arrays.
        aggregated_weights: Resulting aggregated weights.

    Returns:
        Dictionary with aggregation statistics.
    """
    num_clients = len(client_weights)
    num_layers = len(aggregated_weights)

    # Compute weight divergence per client
    divergences = []
    for client_idx in range(num_clients):
        client_divergence = 0.0
        for layer_idx in range(num_layers):
            diff = client_weights[client_idx][layer_idx] - aggregated_weights[layer_idx]
            client_divergence += np.sum(diff ** 2)
        divergences.append(float(np.sqrt(client_divergence)))

    stats = {
        "num_clients": num_clients,
        "num_layers": num_layers,
        "mean_divergence": float(np.mean(divergences)),
        "max_divergence": float(np.max(divergences)),
        "min_divergence": float(np.min(divergences)),
        "std_divergence": float(np.std(divergences)),
        "client_divergences": divergences,
    }

    return stats


def compute_communication_cost(
    model_weights: List[np.ndarray],
    num_clients: int,
    num_rounds: int,
    precision_bytes: int = 4,
) -> Dict[str, float]:
    """
    Compute communication cost of federated learning.

    Args:
        model_weights: Model weight arrays.
        num_clients: Number of clients.
        num_rounds: Number of communication rounds.
        precision_bytes: Bytes per parameter (4 for float32).

    Returns:
        Dictionary with communication cost metrics in MB.
    """
    total_params = sum(w.size for w in model_weights)
    model_size_bytes = total_params * precision_bytes
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Per round: each client sends update + receives global model
    per_round_upload = model_size_mb * num_clients
    per_round_download = model_size_mb * num_clients
    per_round_total = per_round_upload + per_round_download

    total_communication = per_round_total * num_rounds

    return {
        "model_size_mb": round(model_size_mb, 2),
        "total_params": total_params,
        "per_round_upload_mb": round(per_round_upload, 2),
        "per_round_download_mb": round(per_round_download, 2),
        "per_round_total_mb": round(per_round_total, 2),
        "total_communication_mb": round(total_communication, 2),
    }
