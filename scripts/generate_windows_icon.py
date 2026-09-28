"""Generate a small branded Windows icon without third-party image tools."""

import struct
import zlib
from pathlib import Path

SIZE = 256
BACKGROUND = (13, 22, 34, 255)
ACCENT = (45, 212, 191, 255)


def chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def make_png() -> bytes:
    pixels = bytearray()
    points = [(57, 166), (101, 124), (145, 146), (199, 83)]

    def inside_round_rect(x: int, y: int) -> bool:
        radius = 48
        nearest_x = min(max(x, radius), SIZE - radius - 1)
        nearest_y = min(max(y, radius), SIZE - radius - 1)
        return (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius**2

    def on_line(x: int, y: int, start: tuple[int, int], end: tuple[int, int]) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return (x - start[0]) ** 2 + (y - start[1]) ** 2 <= 7**2
        projection = min(
            1.0,
            max(0.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_squared),
        )
        closest_x = start[0] + projection * dx
        closest_y = start[1] + projection * dy
        return (x - closest_x) ** 2 + (y - closest_y) ** 2 <= 7**2

    for y in range(SIZE):
        pixels.append(0)
        for x in range(SIZE):
            color = BACKGROUND if inside_round_rect(x, y) else (0, 0, 0, 0)
            if any(on_line(x, y, points[index], points[index + 1]) for index in range(3)):
                color = ACCENT
            pixels.extend(color)

    header = struct.pack(">2I5B", SIZE, SIZE, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(pixels), level=9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    icon_directory = Path(__file__).resolve().parents[1] / "desktop" / "tauri" / "src-tauri" / "icons"
    icon_directory.mkdir(parents=True, exist_ok=True)
    png = make_png()
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
    (icon_directory / "icon.ico").write_bytes(struct.pack("<HHH", 0, 1, 1) + entry + png)


if __name__ == "__main__":
    main()
