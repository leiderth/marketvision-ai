"""Recommend (never auto-activate) a confidence threshold from walk-forward data."""
import argparse
import json

from backtesting.walk_forward import optimize_confidence_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV histórico etiquetado con retornos observados")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--transaction-cost-pct", type=float, default=0.0)
    args = parser.parse_args()
    result = optimize_confidence_thresholds(
        args.dataset,
        folds=args.folds,
        transaction_cost_pct=args.transaction_cost_pct,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
