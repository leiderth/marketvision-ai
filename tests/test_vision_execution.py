import asyncio
from types import SimpleNamespace

import pytest

from api.routes import vision as vision_routes
from vision import desktop_input


def test_cursor_probe_uses_absolute_windows_coordinates_without_click(monkeypatch):
    calls = []

    class FakeFunction:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, x, y):
            calls.append((x, y))
            return True

    fake_user32 = SimpleNamespace(SetCursorPos=FakeFunction())
    monkeypatch.setattr(desktop_input, "_platform_name", lambda: "nt")
    monkeypatch.setattr(desktop_input, "_load_user32", lambda: fake_user32)

    desktop_input.move_cursor_to(-1920, 400)

    assert calls == [(-1920, 400)]


@pytest.mark.parametrize("point", [(True, 1), (1, False), (1.5, 2), ("1", 2)])
def test_cursor_probe_rejects_invalid_coordinates(point):
    with pytest.raises(ValueError, match="enteros"):
        desktop_input.move_cursor_to(*point)


def test_cursor_probe_rejects_non_windows_platform(monkeypatch):
    monkeypatch.setattr(desktop_input, "_platform_name", lambda: "posix")

    with pytest.raises(RuntimeError, match="Windows"):
        desktop_input.move_cursor_to(10, 20)


def test_probe_endpoint_moves_cursor_and_reports_no_click(monkeypatch, caplog):
    positions = []
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                screen_vision=SimpleNamespace(
                    get_button_position=lambda side: (
                        (-500, 240) if side == "CALL" else (100, 800)
                    )
                )
            )
        )
    )
    monkeypatch.setattr(
        vision_routes,
        "move_cursor_to",
        lambda x, y: positions.append((x, y)),
    )

    with caplog.at_level("INFO", logger="api.routes.vision"):
        result = asyncio.run(
            vision_routes.probe_vision_button(
                vision_routes.ButtonProbeRequest(side="CALL"),
                request,
            )
        )

    assert positions == [(-500, 240)]
    assert result == {
        "side": "CALL",
        "positioned": True,
        "clicked": False,
        "x": -500,
        "y": 240,
        "message": "Cursor movido; no se hizo clic.",
    }
    assert any(
        record.message == "vision_button_cursor_probe"
        and record.clicked is False
        for record in caplog.records
    )


def test_probe_endpoint_fails_closed_when_cursor_cannot_move(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                screen_vision=SimpleNamespace(
                    get_button_position=lambda _side: (10, 20)
                )
            )
        )
    )

    def fail_to_move(_x, _y):
        raise RuntimeError("input unavailable")

    monkeypatch.setattr(vision_routes, "move_cursor_to", fail_to_move)

    with pytest.raises(vision_routes.HTTPException) as error:
        asyncio.run(
            vision_routes.probe_vision_button(
                vision_routes.ButtonProbeRequest(side="PUT"),
                request,
            )
        )

    assert error.value.status_code == 503


def test_vision_state_keeps_real_execution_unavailable():
    class Learning:
        def get_context_edge_margin(self, **kwargs):
            return kwargs["configured_margin_pct"]

        def get_confidence_threshold_pct(self):
            return 70

    class Decisions:
        def get_calibration(self, *_args, **_kwargs):
            return {"decision": "NO_TRADE", "reason": "insufficient history"}

        def get_auto_outcome_watch(self):
            return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                screen_vision=SimpleNamespace(
                    get_state=lambda: {
                        "readable": False,
                        "reason": "chart unreadable",
                        "directional_scores": {"CALL": None, "PUT": None},
                        "market_regime": "unknown",
                    }
                ),
                vision_capture_fps=5,
                vision_edge_margin_pct=5,
                vision_minimum_margin_pct=15,
                vision_learning_manager=Learning(),
                vision_decisions=Decisions(),
            )
        )
    )

    result = asyncio.run(vision_routes.get_vision_state(request))

    assert result["live_execution"]["enabled"] is False
    assert result["live_execution"]["available"] is False
    assert "ventana objetivo" in result["live_execution"]["reason"]
