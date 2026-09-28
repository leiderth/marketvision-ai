import os

from desktop_server import configure_runtime


def test_configure_runtime_creates_user_scoped_config_and_state(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("MARKETVISION_ENV_FILE", raising=False)
    monkeypatch.delenv("ML_MODEL_REGISTRY_DIR", raising=False)
    monkeypatch.delenv("ML_DRIFT_STATE_DIR", raising=False)
    monkeypatch.delenv("RISK_STATE_PATH", raising=False)

    configure_runtime()

    app_data = tmp_path / "MarketVision AI"
    env_file = app_data / "settings.env"
    data_dir = app_data / "data"
    assert env_file.is_file()
    assert data_dir.is_dir()
    assert "MARKET_DATA_API_KEY=\n" in env_file.read_text(encoding="utf-8")
    assert "ECONOMIC_CALENDAR_API_KEY=\n" in env_file.read_text(encoding="utf-8")
    assert os.environ["MARKETVISION_ENV_FILE"] == str(env_file)
    assert os.environ["ML_MODEL_REGISTRY_DIR"] == str(data_dir / "models")
    assert os.environ["ML_DRIFT_STATE_DIR"] == str(data_dir / "model_drift")
    assert os.environ["RISK_STATE_PATH"] == str(data_dir / "risk_state.json")
