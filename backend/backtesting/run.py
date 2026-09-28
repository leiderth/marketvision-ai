"""Run an offline walk-forward evaluation for a labeled CSV dataset."""
import argparse
import json

from backtesting.walk_forward import run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV con timestamp, labels y retornos futuros observados")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--transaction-cost-pct", type=float, default=0.0)
    args = parser.parse_args()
    result = run_walk_forward(
        args.dataset,
        folds=args.folds,
        min_confidence=args.min_confidence,
        transaction_cost_pct=args.transaction_cost_pct,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
