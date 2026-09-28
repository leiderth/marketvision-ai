"""Explicit offline model training; registered models are never activated here."""
import argparse
import json

from ml.model_registry import train_and_register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV con features de mercado y etiquetas CALL/PUT")
    parser.add_argument(
        "--registry",
        default="./models/registry",
        help="Directorio local para modelos versionados",
    )
    parser.add_argument(
        "--min-balanced-accuracy",
        type=float,
        default=0.55,
        help="Balanced accuracy mínima requerida en validation y test",
    )
    args = parser.parse_args()
    result = train_and_register(
        args.dataset,
        args.registry,
        min_balanced_accuracy=args.min_balanced_accuracy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "Modelo registrado inactivo. Revisa su evaluación y configura "
        "ML_ACTIVE_MODEL_VERSION explícitamente para habilitar inferencia."
    )


if __name__ == "__main__":
    main()
