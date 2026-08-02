"""
Comparative analysis experiment runner.

Runs all experimental conditions and generates comprehensive
statistical comparisons between centralised, federated, and
privacy-enhanced models.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed, set_gpu_config
from src.utils.config import load_config, get_results_path
from src.utils.logger import get_experiment_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comparative analysis")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--architecture", type=str, default="ResNet50V2",
        choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"],
        help="Model architecture"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick comparison with reduced rounds"
    )
    return parser.parse_args()


def main():
    """
    Run all experiments and generate comparative analysis.

    Experimental conditions:
    1. Centralised baseline (no privacy, all data)
    2. Federated (IID) with different epsilon values
    3. Federated (non-IID) with different epsilon values
    4. Privacy-performance tradeoff analysis
    5. Statistical significance tests
    """
    args = parse_args()

    # Setup
    set_seed(42)
    set_gpu_config()

    config = load_config(args.config)
    logger = get_experiment_logger("comparison")

    output_dir = args.output_dir or str(get_results_path("comparison"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.quick:
        config.setdefault("federated", {})["num_rounds"] = 5
        config["federated"]["local_epochs"] = 2

    # Import experiment runners
    from experiments.run_centralised import run_centralised_experiment
    from experiments.run_federated import run_single_experiment

    all_results = {}
    start_time = time.time()

    # Experiment 1: Centralised baseline
    logger.info("=" * 60)
    logger.info("Running Experiment 1: Centralised Baseline")
    logger.info("=" * 60)

    centralised_results = run_centralised_experiment(
        config=config,
        architecture=args.architecture,
        use_synthetic=args.synthetic,
        output_dir=str(output_path / "centralised"),
        logger=logger,
    )
    all_results["centralised"] = centralised_results.get("test_results", {})

    # Experiment 2: Federated IID (no DP)
    logger.info("=" * 60)
    logger.info("Running Experiment 2: Federated IID (no DP)")
    logger.info("=" * 60)

    fed_iid_results = run_single_experiment(
        config=config,
        architecture=args.architecture,
        epsilon=float("inf"),  # No DP
        non_iid=False,
        use_synthetic=args.synthetic,
        output_dir=str(output_path / "federated_iid"),
        logger=logger,
    )
    all_results["federated_iid"] = fed_iid_results.get("test_results", {})

    # Experiment 3: Federated non-IID (no DP)
    logger.info("=" * 60)
    logger.info("Running Experiment 3: Federated non-IID (no DP)")
    logger.info("=" * 60)

    fed_non_iid_results = run_single_experiment(
        config=config,
        architecture=args.architecture,
        epsilon=float("inf"),
        non_iid=True,
        use_synthetic=args.synthetic,
        output_dir=str(output_path / "federated_non_iid"),
        logger=logger,
    )
    all_results["federated_non_iid"] = fed_non_iid_results.get("test_results", {})

    # Experiment 4: Federated with different epsilon values
    privacy_config = config.get("privacy", {}).get("differential_privacy", {})
    epsilon_values = privacy_config.get("epsilon_values", [0.5, 1.0, 5.0])
    epsilon_results = {}

    for eps in epsilon_values:
        logger.info("=" * 60)
        logger.info(f"Running Experiment 4: Federated with epsilon={eps}")
        logger.info("=" * 60)

        eps_results = run_single_experiment(
            config=config,
            architecture=args.architecture,
            epsilon=eps,
            non_iid=True,
            use_synthetic=args.synthetic,
            output_dir=str(output_path / f"federated_eps_{eps}"),
            logger=logger,
        )
        epsilon_results[eps] = eps_results.get("test_results", {})
        all_results[f"federated_eps_{eps}"] = epsilon_results[eps]

    # Generate comparative analysis
    logger.info("=" * 60)
    logger.info("Generating Comparative Analysis")
    logger.info("=" * 60)

    from src.evaluation.comparison import (
        compare_federated_vs_centralised,
        evaluate_privacy_performance_tradeoff,
        evaluate_non_iid_robustness,
        generate_comparison_summary,
    )
    from src.evaluation.visualisation import (
        plot_convergence_curves,
        plot_privacy_tradeoff,
        plot_architecture_comparison,
    )

    # Comparison: Federated vs Centralised
    fed_vs_cent = compare_federated_vs_centralised(
        federated_results=all_results.get("federated_iid", {}),
        centralised_results=all_results.get("centralised", {}),
    )

    # Privacy-performance tradeoff
    privacy_tradeoff = evaluate_privacy_performance_tradeoff(
        epsilon_results=epsilon_results,
        baseline_results=all_results.get("federated_iid", {}),
    )

    # Non-IID robustness
    non_iid_robustness = evaluate_non_iid_robustness(
        iid_results=all_results.get("federated_iid", {}),
        non_iid_results=all_results.get("federated_non_iid", {}),
    )

    # Overall summary
    summary = generate_comparison_summary(all_results)

    # Generate plots
    if epsilon_results:
        epsilons = sorted(epsilon_results.keys())
        accuracies = [epsilon_results[e].get("accuracy", 0) for e in epsilons]
        baseline_acc = all_results.get("centralised", {}).get("accuracy", 0)

        plot_privacy_tradeoff(
            epsilon_values=epsilons,
            accuracies=accuracies,
            baseline_accuracy=baseline_acc,
            save_path=str(output_path / "privacy_tradeoff.png"),
        )

    # Save comparison results
    comparison_results = {
        "all_results": all_results,
        "federated_vs_centralised": fed_vs_cent,
        "privacy_tradeoff": privacy_tradeoff,
        "non_iid_robustness": non_iid_robustness,
        "summary": summary,
        "total_time": time.time() - start_time,
    }

    results_file = output_path / "comparison_results.json"
    serialisable = json.loads(
        json.dumps(comparison_results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
    )
    with open(results_file, "w") as f:
        json.dump(serialisable, f, indent=2)

    logger.info(f"\nComparison results saved to: {results_file}")
    logger.info(f"Total experiment time: {comparison_results['total_time']:.1f}s")
    logger.info(f"\nSummary: {summary.get('summary', '')}")

    return comparison_results


if __name__ == "__main__":
    main()
