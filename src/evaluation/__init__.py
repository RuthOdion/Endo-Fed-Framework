"""Evaluation metrics, statistical comparison, and visualisation modules."""

from src.evaluation.metrics import evaluate_model, evaluate_kfold, compute_convergence_speed
from src.evaluation.comparison import (
    paired_t_test,
    compare_federated_vs_centralised,
    evaluate_privacy_performance_tradeoff,
    generate_comparison_summary,
)
from src.evaluation.visualisation import generate_all_plots
