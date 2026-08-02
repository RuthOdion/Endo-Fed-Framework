"""
Secure Aggregation handler for federated learning.

Implements additive masking protocol to ensure individual client
updates cannot be observed by the server during aggregation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class SecureAggregationHandler:
    """
    Secure aggregation using additive masking protocol.

    Generates random masks that sum to zero across all clients.
    Each client adds their mask to their weights before sending,
    and the server can only see the aggregate (masks cancel out).

    This prevents the server from seeing individual client updates
    while still enabling correct aggregation.
    """

    def __init__(
        self,
        num_clients: int = 5,
        seed: int = 42,
        protocol: str = "additive_masking",
        num_shares: int = 3,
        threshold: int = 2,
    ):
        """
        Initialise secure aggregation handler.

        Args:
            num_clients: Number of participating clients.
            seed: Random seed for reproducibility.
            protocol: Protocol type ('additive_masking').
            num_shares: Number of secret shares.
            threshold: Minimum shares needed for reconstruction.
        """
        self.num_clients = num_clients
        self.seed = seed
        self.protocol = protocol
        self.num_shares = num_shares
        self.threshold = threshold
        self.rng = np.random.RandomState(seed)
        self._masks_generated = 0

    def generate_masks(
        self,
        num_clients: int,
        weight_shapes: List[Tuple],
        mask_scale: float = 1.0,
    ) -> List[List[np.ndarray]]:
        """
        Generate additive masks that sum to zero across clients.

        For N clients and each weight layer, generates N random masks
        such that sum(masks) = 0 for each layer.

        Args:
            num_clients: Number of clients to generate masks for.
            weight_shapes: List of shapes for each weight layer.
            mask_scale: Scale factor for mask magnitude.

        Returns:
            List of mask lists, one per client. Each contains arrays
            matching the weight shapes.
        """
        masks = []

        for client_idx in range(num_clients):
            client_masks = []
            for shape in weight_shapes:
                if client_idx < num_clients - 1:
                    # Generate random mask for all but last client
                    mask = self.rng.normal(0, mask_scale, size=shape).astype(np.float32)
                else:
                    # Last client's mask is negative sum of all others
                    mask = np.zeros(shape, dtype=np.float32)
                client_masks.append(mask)
            masks.append(client_masks)

        # Set last client's masks to negative sum of others
        for layer_idx in range(len(weight_shapes)):
            sum_masks = np.zeros(weight_shapes[layer_idx], dtype=np.float32)
            for client_idx in range(num_clients - 1):
                sum_masks += masks[client_idx][layer_idx]
            masks[num_clients - 1][layer_idx] = -sum_masks

        self._masks_generated += 1
        return masks

    def mask_weights(
        self,
        weights: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply masks to client weights before transmission.

        Args:
            weights: Client's model weights.
            masks: Client's assigned masks.

        Returns:
            Masked weights (weights + masks).
        """
        masked = []
        for w, m in zip(weights, masks):
            masked.append(w + m)
        return masked

    def unmask_aggregate(
        self,
        aggregated_weights: List[np.ndarray],
        all_masks: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Remove mask effects from aggregated weights.

        Since masks sum to zero, the aggregate of masked weights
        equals the aggregate of unmasked weights. This method
        verifies the protocol correctness.

        Args:
            aggregated_weights: Aggregated (masked) weights from server.
            all_masks: All client masks (used for verification only).

        Returns:
            Unmasked aggregated weights (same as input if protocol is correct).
        """
        # For additive masking with masks summing to zero,
        # the aggregation naturally removes the masks.
        # This function serves as a verification step.

        # Verify mask sum is approximately zero
        for layer_idx in range(len(aggregated_weights)):
            mask_sum = np.zeros_like(aggregated_weights[layer_idx])
            for client_masks in all_masks:
                mask_sum += client_masks[layer_idx]

            max_error = np.max(np.abs(mask_sum))
            if max_error > 1e-6:
                print(f"WARNING: Mask sum not zero (max error: {max_error:.2e})")

        return aggregated_weights

    def verify_protocol(
        self,
        original_weights: List[List[np.ndarray]],
        masked_weights: List[List[np.ndarray]],
        masks: List[List[np.ndarray]],
    ) -> bool:
        """
        Verify that the secure aggregation protocol is correct.

        Checks that:
        1. Masks sum to zero for each layer
        2. Aggregated masked weights == aggregated original weights

        Args:
            original_weights: Unmasked client weights.
            masked_weights: Masked client weights.
            masks: Client masks.

        Returns:
            True if protocol verification passes.
        """
        num_clients = len(original_weights)
        num_layers = len(original_weights[0])

        # Check 1: Masks sum to zero
        for layer_idx in range(num_layers):
            mask_sum = np.zeros_like(masks[0][layer_idx])
            for client_idx in range(num_clients):
                mask_sum += masks[client_idx][layer_idx]

            if np.max(np.abs(mask_sum)) > 1e-5:
                return False

        # Check 2: Aggregated masked == aggregated original
        for layer_idx in range(num_layers):
            orig_sum = np.zeros_like(original_weights[0][layer_idx])
            masked_sum = np.zeros_like(masked_weights[0][layer_idx])

            for client_idx in range(num_clients):
                orig_sum += original_weights[client_idx][layer_idx]
                masked_sum += masked_weights[client_idx][layer_idx]

            if np.max(np.abs(orig_sum - masked_sum)) > 1e-5:
                return False

        return True

    def get_statistics(self) -> Dict:
        """
        Get secure aggregation statistics.

        Returns:
            Dictionary with protocol statistics.
        """
        return {
            "protocol": self.protocol,
            "num_clients": self.num_clients,
            "num_shares": self.num_shares,
            "threshold": self.threshold,
            "masks_generated": self._masks_generated,
        }
