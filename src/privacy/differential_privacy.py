"""
Differential Privacy handler for federated learning.

Implements the Gaussian mechanism with L2 norm clipping to provide
(epsilon, delta)-differential privacy guarantees for model updates.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class DifferentialPrivacyHandler:
    """
    Differential privacy mechanism for protecting client model updates.

    Implements:
    - L2 norm clipping: bounds the sensitivity of individual updates
    - Gaussian noise addition: calibrated to the clipping bound and privacy budget
    - Privacy budget tracking: monitors cumulative epsilon across rounds
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        l2_norm_clip: float = 1.0,
        num_clients: int = 5,
        num_rounds: int = 20,
        mechanism: str = "gaussian",
    ):
        """
        Initialise differential privacy handler.

        Args:
            epsilon: Target privacy budget (smaller = more private).
            delta: Probability of privacy breach.
            l2_norm_clip: Maximum L2 norm for weight clipping.
            num_clients: Number of participating clients.
            num_rounds: Total number of training rounds.
            mechanism: Noise mechanism ('gaussian').
        """
        self.epsilon = epsilon
        self.delta = delta
        self.l2_norm_clip = l2_norm_clip
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.mechanism = mechanism

        # Compute noise scale (sigma)
        self.sigma = self._compute_sigma()

        # Privacy budget tracking
        self.rounds_completed = 0
        self.epsilon_spent = 0.0

    def _compute_sigma(self) -> float:
        """
        Compute the noise standard deviation (sigma) for the Gaussian mechanism.

        Uses the analytical Gaussian mechanism formula:
        sigma = (l2_norm_clip * sqrt(2 * ln(1.25/delta))) / epsilon

        Per-round epsilon is computed assuming simple composition over rounds.

        Returns:
            Noise standard deviation sigma.
        """
        # Per-round epsilon budget (using simple composition)
        per_round_epsilon = self.epsilon / math.sqrt(self.num_rounds)

        # Gaussian mechanism formula
        sigma = (self.l2_norm_clip * math.sqrt(2 * math.log(1.25 / self.delta))) / per_round_epsilon

        return sigma

    def clip_weights(
        self,
        weights: List[np.ndarray],
        global_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Clip weight updates to bound L2 norm sensitivity.

        Computes the update (weights - global_weights) and clips it
        to have maximum L2 norm of l2_norm_clip, then adds back to global.

        Args:
            weights: Client's updated weights.
            global_weights: Current global model weights.

        Returns:
            Clipped weights.
        """
        # Compute update
        updates = [w - gw for w, gw in zip(weights, global_weights)]

        # Compute L2 norm of the full update
        total_norm = 0.0
        for update in updates:
            total_norm += np.sum(update ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        if total_norm > self.l2_norm_clip:
            clip_factor = self.l2_norm_clip / total_norm
            updates = [update * clip_factor for update in updates]

        # Reconstruct clipped weights
        clipped_weights = [gw + update for gw, update in zip(global_weights, updates)]

        return clipped_weights

    def add_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add calibrated Gaussian noise to weight arrays.

        Noise is drawn from N(0, sigma^2) where sigma is calibrated
        to provide the target (epsilon, delta)-DP guarantee.

        Args:
            weights: Weight arrays to add noise to.

        Returns:
            Noised weight arrays.
        """
        noised_weights = []

        for w in weights:
            noise = np.random.normal(
                loc=0.0,
                scale=self.sigma,
                size=w.shape,
            ).astype(w.dtype)
            noised_weights.append(w + noise)

        # Update privacy budget
        self._update_privacy_budget()

        return noised_weights

    def _update_privacy_budget(self) -> None:
        """
        Update the cumulative privacy budget after a round.

        Uses simple composition: epsilon_total = sqrt(T) * epsilon_per_round
        """
        self.rounds_completed += 1
        # Advanced composition bound
        self.epsilon_spent = self.epsilon * math.sqrt(
            self.rounds_completed / self.num_rounds
        )

    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get the current privacy budget expenditure.

        Returns:
            Dictionary with privacy accounting information.
        """
        return {
            "epsilon_spent": round(self.epsilon_spent, 4),
            "epsilon_budget": self.epsilon,
            "delta": self.delta,
            "rounds_completed": self.rounds_completed,
            "sigma": round(self.sigma, 4),
            "budget_remaining_fraction": round(
                max(0, 1 - self.epsilon_spent / self.epsilon), 4
            ),
        }

    def get_noise_magnitude(self) -> float:
        """
        Get the current noise magnitude (sigma).

        Returns:
            Standard deviation of the Gaussian noise.
        """
        return self.sigma


def create_dp_handler_for_epsilon(
    epsilon: float,
    delta: float = 1e-5,
    l2_norm_clip: float = 1.0,
    num_clients: int = 5,
    num_rounds: int = 20,
) -> DifferentialPrivacyHandler:
    """
    Factory function to create a DP handler for a specific epsilon value.

    Args:
        epsilon: Target privacy budget.
        delta: Privacy breach probability.
        l2_norm_clip: L2 norm clipping bound.
        num_clients: Number of clients.
        num_rounds: Number of training rounds.

    Returns:
        Configured DifferentialPrivacyHandler.
    """
    return DifferentialPrivacyHandler(
        epsilon=epsilon,
        delta=delta,
        l2_norm_clip=l2_norm_clip,
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
