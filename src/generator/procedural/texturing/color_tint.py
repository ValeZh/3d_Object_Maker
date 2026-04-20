"""
Тинт diffuse-текстур: умножение каналов на заданный RGB (0–255 или 0–1).
Используется при сборке атласов из файлов, чтобы задать оттенок без правки PNG вручную.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from PIL import Image


def parse_texture_color_tint(value: Any) -> Tuple[int, int, int] | None:
    """
    Из JSON/батча: ``[220, 200, 180]`` (0–255) или ``[0.86, 0.78, 0.71]`` (0–1).
    Некорректные значения — ``None`` (без тинта).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        a, b, c = (float(x) for x in value)
        if max(a, b, c) <= 1.0 + 1e-6:
            return (
                int(max(0, min(255, round(a * 255.0)))),
                int(max(0, min(255, round(b * 255.0)))),
                int(max(0, min(255, round(c * 255.0)))),
            )
        return (
            int(max(0, min(255, round(a)))),
            int(max(0, min(255, round(b)))),
            int(max(0, min(255, round(c)))),
        )
    return None


def apply_texture_color_tint(im: Image.Image, rgb: Tuple[int, int, int] | None) -> Image.Image:
    """Умножение RGB картинки на ``rgb/255`` (типичный тинт для белых/нейтральных текстур)."""
    if rgb is None:
        return im
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    t = np.array(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    out = np.clip(a * t, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
