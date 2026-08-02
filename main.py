"""
Main entry point for the Endometriosis Federated Learning Framework.

Provides CLI interface with subcommands for running different
experimental modes: test, federated, centralised, compare, explain.

Usage:
    python main.py test                 # Run integration tests
    python main.py federated            # Run federated experiment
    python main.py centralised          # Run centralised baseline
    python main.py compare              # Run comparative analysis
    python main.py explain              # Run Grad-CAM analysis
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    """Parse arguments and dispatch to appropriate experiment."""
    parser = argparse.ArgumentParser(
        description="Endometriosis Federated Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py test
    python main.py federated --architecture ResNet50V2 --epsilon 1.0 --non_iid
    python main.py centralised --architecture EfficientNetB0 --epochs 50
    python main.py compare --synthetic --quick
    python main.py explain --architecture MobileNetV2 --num_samples 20
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Experiment type")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Run integration tests")

    # Federated subcommand
    fed_parser = subparsers.add_parser("federated", help="Run federated experiment")
    fed_parser.add_argument("--config", type=str, default=None)
    fed_parser.add_argument("--architecture", type=str, default="ResNet50V2",
                           choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"])
    fed_parser.add_argument("--epsilon", type=float, default=1.0)
    fed_parser.add_argument("--num_rounds", type=int, default=None)
    fed_parser.add_argument("--non_iid", action="store_true")
    fed_parser.add_argument("--synthetic", action="store_true")
    fed_parser.add_argument("--output_dir", type=str, default=None)

    # Centralised subcommand
    cent_parser = subparsers.add_parser("centralised", help="Run centralised baseline")
    cent_parser.add_argument("--config", type=str, default=None)
    cent_parser.add_argument("--architecture", type=str, default="ResNet50V2",
                            choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"])
    cent_parser.add_argument("--epochs", type=int, default=None)
    cent_parser.add_argument("--synthetic", action="store_true")
    cent_parser.add_argument("--output_dir", type=str, default=None)

    # Compare subcommand
    comp_parser = subparsers.add_parser("compare", help="Run comparative analysis")
    comp_parser.add_argument("--config", type=str, default=None)
    comp_parser.add_argument("--architecture", type=str, default="ResNet50V2",
                            choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"])
    comp_parser.add_argument("--synthetic", action="store_true")
    comp_parser.add_argument("--output_dir", type=str, default=None)
    comp_parser.add_argument("--quick", action="store_true")

    # Explain subcommand
    exp_parser = subparsers.add_parser("explain", help="Run Grad-CAM analysis")
    exp_parser.add_argument("--config", type=str, default=None)
    exp_parser.add_argument("--model_path", type=str, default=None)
    exp_parser.add_argument("--architecture", type=str, default="ResNet50V2",
                           choices=["ResNet50V2", "EfficientNetB0", "MobileNetV2"])
    exp_parser.add_argument("--num_samples", type=int, default=50)
    exp_parser.add_argument("--synthetic", action="store_true")
    exp_parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to appropriate experiment
    if args.command == "test":
        from experiments.run_quick_test import run_quick_test
        results = run_quick_test()
        # Exit with error code if any tests failed
        passed = sum(1 for v in results.values() if v == "PASSED")
        sys.exit(0 if passed == len(results) else 1)

    elif args.command == "federated":
        from experiments.run_federated import main as run_federated
        sys.argv = ["run_federated.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        sys.argv.extend(["--architecture", args.architecture])
        sys.argv.extend(["--epsilon", str(args.epsilon)])
        if args.num_rounds:
            sys.argv.extend(["--num_rounds", str(args.num_rounds)])
        if args.non_iid:
            sys.argv.append("--non_iid")
        if args.synthetic:
            sys.argv.append("--synthetic")
        if args.output_dir:
            sys.argv.extend(["--output_dir", args.output_dir])
        run_federated()

    elif args.command == "centralised":
        from experiments.run_centralised import main as run_centralised
        sys.argv = ["run_centralised.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        sys.argv.extend(["--architecture", args.architecture])
        if args.epochs:
            sys.argv.extend(["--epochs", str(args.epochs)])
        if args.synthetic:
            sys.argv.append("--synthetic")
        if args.output_dir:
            sys.argv.extend(["--output_dir", args.output_dir])
        run_centralised()

    elif args.command == "compare":
        from experiments.run_comparison import main as run_comparison
        sys.argv = ["run_comparison.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        sys.argv.extend(["--architecture", args.architecture])
        if args.synthetic:
            sys.argv.append("--synthetic")
        if args.output_dir:
            sys.argv.extend(["--output_dir", args.output_dir])
        if args.quick:
            sys.argv.append("--quick")
        run_comparison()

    elif args.command == "explain":
        from experiments.run_explainability import main as run_explain
        sys.argv = ["run_explainability.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.model_path:
            sys.argv.extend(["--model_path", args.model_path])
        sys.argv.extend(["--architecture", args.architecture])
        sys.argv.extend(["--num_samples", str(args.num_samples)])
        if args.synthetic:
            sys.argv.append("--synthetic")
        if args.output_dir:
            sys.argv.extend(["--output_dir", args.output_dir])
        run_explain()


if __name__ == "__main__":
    main()
