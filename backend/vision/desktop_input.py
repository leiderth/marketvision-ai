"""Windows cursor positioning for non-clicking calibration probes."""
from __future__ import annotations

import ctypes
import os


def _platform_name() -> str:
    return os.name


def _load_user32():
    return ctypes.WinDLL("user32", use_last_error=True)


def _last_error() -> int:
    return ctypes.get_last_error()


def move_cursor_to(x: int, y: int) -> None:
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, int)
        or not isinstance(y, int)
    ):
        raise ValueError("Las coordenadas del cursor deben ser enteros.")
    if _platform_name() != "nt":
        raise RuntimeError("La prueba de cursor solo está disponible en Windows.")

    user32 = _load_user32()
    set_cursor_position = user32.SetCursorPos
    set_cursor_position.argtypes = (ctypes.c_int, ctypes.c_int)
    set_cursor_position.restype = ctypes.c_bool
    if not set_cursor_position(x, y):
        raise OSError(_last_error(), "Windows no pudo mover el cursor a la coordenada calibrada.")
