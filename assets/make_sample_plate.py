"""Generate a small placeholder plate image so the image path is runnable."""
import struct
import zlib
from pathlib import Path

W = H = 96
# Concentric circles in food-ish colours: a crude "plate".
rows = []
for y in range(H):
    row = bytearray([0])  # filter byte
    for x in range(W):
        dx, dy = x - W / 2, y - H / 2
        r = (dx * dx + dy * dy) ** 0.5
        if r < 18:
            px = (232, 226, 190)   # rice
        elif r < 30:
            px = (196, 142, 58)    # dal
        elif r < 42:
            px = (214, 186, 130)   # roti
        else:
            px = (245, 245, 245)   # plate
        row.extend(px)
    rows.append(bytes(row))

raw = zlib.compress(b"".join(rows), 9)


def chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", raw)
       + chunk(b"IEND", b""))

out = Path("assets/sample_plate.png")
out.write_bytes(png)
print("wrote", out, len(png), "bytes")
