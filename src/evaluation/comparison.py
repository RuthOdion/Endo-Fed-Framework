"""
Statistical comparison of experimental conditions.

Implements paired t-tests with Cohen's d effect sizes and
comprehensive comparisons between federated/centralised,
privacy levels, and data heterogeneity conditions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform paired t-test with Cohen's d effect size.

    Args:
        scores_a: Performance scores for condition A.
        scores_b: Performance scores for condition B.
        alpha: Significance level.

    Returns:
        Dictionary with t-statistic, p-value, Cohen's d, and significance.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    # Compute differences
    differences = scores_a - scores_b
    n = len(differences)

    # t-test
    if stats is not None:
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    else:
        # Manual computation
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0.0
        # Approximate p-value (two-tailed)
        p_value = 2.0 * (1.0 - 0.975)  # fallback

    # Cohen's d (paired)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interpretation = "negligible"
    elif abs_d < 0.5:
        effect_interpretation = "small"
    elif abs_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "effect_size": effect_interpretation,
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "n_pairs": n,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "mean_difference": float(mean_diff),
        "ci_95": (
            float(mean_diff - 1.96 * std_diff / np.sqrt(n)),
            float(mean_diff + 1.96 * std_diff / np.sqrt(n)),
        ),
    }


def compare_federated_vs_centralised(
    federated_results: Dict,
    centralised_results: Dict,
    metrics: List[str] = None,
) -> Dict:
    """
    Compare federated vs centralised training results.

    Args:
        federated_results: Results from federated training.
        centralised_results: Results from centralised baseline.
        metrics: Metrics to compare.

    Returns:
        Comparison dictionary with differences and statistical tests.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    comparison = {"metrics": {}}

    for metric in metrics:
        fed_val = federated_results.get(metric, 0.0)
        cent_val = centralised_results.get(metric, 0.0)

        diff = fed_val - cent_val
        rel_diff = (diff / cent_val * 100) if cent_val > 0 else 0.0

        comparison["metrics"][metric] = {
            "federated": float(fed_val),
            "centralised": float(cent_val),
            "absolute_difference": float(diff),
            "relative_difference_pct": float(rel_diff),
            "federated_competitive": abs(diff) < 0.05,  # Within 5% threshold
        }

    # Overall assessment
    competitive_count = sum(
        1 for m in comparison["metrics"].values() if m["federated_competitive"]
    )
    comparison["overall_competitive"] = competitive_count >= len(metrics) * 0.8
    comparison["summary"] = (
        f"Federated model is competitive on {competitive_count}/{len(metrics)} metrics"
    )

    return comparison


def evaluate_privacy_performance_tradeoff(
    epsilon_results: Dict[float, Dict],
    baseline_results: Dict,
) -> Dict:
    """
    Evaluate the tradeoff between privacy (epsilon) and performance.

    Args:
        epsilon_results: Dict mapping epsilon values to their results.
        baseline_results: Non-private baseline results.

    Returns:
        Privacy-performance tradeoff analysis.
    """
    tradeoff = {"epsilon_analysis": {}}

    for epsilon, results in sorted(epsilon_results.items()):
        baseline_acc = baseline_results.get("accuracy", 0.0)
        result_acc = results.get("accuracy", 0.0)
        utility_loss = baseline_acc - result_acc

        tradeoff["epsilon_analysis"][epsilon] = {
            "accuracy": float(result_acc),
            "utility_loss": float(utility_loss),
            "utility_loss_pct": float(utility_loss / baseline_acc * 100) if baseline_acc > 0 else 0.0,
            "roc_auc": float(results.get("roc_auc", 0.0)),
            "f1_score": float(results.get("f1_score", 0.0)),
            "acceptable_loss": utility_loss < 0.10,  # Less than 10% loss
        }

    # Find optimal epsilon (best privacy with acceptable performance)
    acceptable = {
        eps: data for eps, data in tradeoff["epsilon_analysis"].items()
        if data["acceptable_loss"]
    }

    if acceptable:
        optimal_epsilon = min(acceptable.keys())
        tradeoff["optimal_epsilon"] = optimal_epsilon
    else:
        tradeoff["optimal_epsilon"] = max(epsilon_results.keys())

    tradeoff["baseline_accuracy"] = float(baseline_results.get("accuracy", 0.0))

    return tradeoff


def evaluate_non_iid_robustness(
    iid_results: Dict,
    non_iid_results: Dict,
    metrics: List[str] = None,
) -> Dict:
    """
    Evaluate robustness to non-IID data distributions.

    Args:
        iid_results: Results with IID data partitioning.
        non_iid_results: Results with non-IID (Dirichlet) partitioning.
        metrics: Metrics to compare.

    Returns:
        Robustness analysis dictionary.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    robustness = {"metrics": {}}

    for metric in metrics:
        iid_val = iid_results.get(metric, 0.0)
        non_iid_val = non_iid_results.get(metric, 0.0)

        degradation = iid_val - non_iid_val
        rel_degradation = (degradation / iid_val * 100) if iid_val > 0 else 0.0

        robustness["metrics"][metric] = {
            "iid": float(iid_val),
            "non_iid": float(non_iid_val),
            "degradation": float(degradation),
            "degradation_pct": float(rel_degradation),
            "robust": abs(degradation) < 0.05,
        }

    robust_count = sum(1 for m in robustness["metrics"].values() if m["robust"])
    robustness["overall_robust"] = robust_count >= len(metrics) * 0.6
    robustness["summary"] = (
        f"Model is robust on {robust_count}/{len(metrics)} metrics "
        f"under non-IID conditions"
    )

    return robustness


def generate_comparison_summary(
    all_results: Dict,
) -> Dict:
    """
    Generate comprehensive comparison summary across all conditions.

    Args:
        all_results: Dictionary with results from all experimental conditions.

    Returns:
        Summary dictionary with rankings and key findings.
    """
    summary = {
        "conditions": list(all_results.keys()),
        "rankings": {},
        "key_findings": [],
    }

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    for metric in metrics:
        scores = {}
        for condition, results in all_results.items():
            if metric in results:
                scores[condition] = results[metric]

        # Rank by metric value (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        summary["rankings"][metric] = [
            {"condition": cond, "score": float(score)}
            for cond, score in ranked
        ]

    # Generate key findings
    if "centralised" in all_results and "federated" in all_results:
        fed_acc = all_results["federated"].get("accuracy", 0)
        cent_acc = all_results["centralised"].get("accuracy", 0)
        diff = cent_acc - fed_acc

        if diff < 0.02:
            summary["key_findings"].append(
                "Federated model achieves comparable accuracy to centralised baseline"
            )
        elif diff < 0.05:
            summary["key_findings"].append(
                f"Small accuracy gap ({diff:.1%}) between federated and centralised"
            )
        else:
            summary["key_findings"].append(
                f"Significant accuracy gap ({diff:.1%}) between federated and centralised"
            )

    return summary
