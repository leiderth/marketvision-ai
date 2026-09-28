"""Packaged Windows desktop entry point for the local analysis API."""

import argparse
import os
from pathlib import Path

import uvicorn


def configure_runtime() -> None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is required to persist desktop application state.")

    app_data = Path(local_app_data) / "MarketVision AI"
    data_dir = app_data / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    env_file = app_data / "settings.env"
    if not env_file.exists():
        env_file.write_text(
            "DEFAULT_MODE=ANALYSIS_MODE\n"
            "MARKET_DATA_PROVIDER=twelve_data\n"
            "MARKET_DATA_API_KEY=\n"
            "MARKET_DATA_SYMBOL=\n"
            "MARKET_DATA_INTERVAL=1min\n"
            "ECONOMIC_CALENDAR_API_KEY=\n",
            encoding="utf-8",
        )

    os.environ["MARKETVISION_ENV_FILE"] = str(env_file)
    os.environ.setdefault("ML_MODEL_REGISTRY_DIR", str(data_dir / "models"))
    os.environ.setdefault("ML_DRIFT_STATE_DIR", str(data_dir / "model_drift"))
    os.environ.setdefault("RISK_STATE_PATH", str(data_dir / "risk_state.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketVision AI local API")
    parser.add_argument("--port", type=int, default=18731)
    args = parser.parse_args()

    configure_runtime()
    from api.main import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
