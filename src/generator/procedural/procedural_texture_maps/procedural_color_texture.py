"""
Процедурные RGB-текстуры (стены, штукатурка, простые поверхности) как PIL.Image.

Карты нормалей — в модуле ``normal_map`` (процедурная генерация, без карты высот).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _ensure_size(size: int) -> int:
    return max(64, int(size))


def make_uniform_noise_texture(
    size: int = 512,
    *,
    base_rgb: Tuple[int, int, int] = (168, 158, 148),
    noise_sigma: float = 6.0,
    seed: int = 42,
) -> Image.Image:
    """Ровная штукатурка: базовый цвет + гауссов шум (подходит как diffuse)."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    base = np.array(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.ones((s, s, 3), dtype=np.float32) * base
    rgb += rng.normal(0.0, float(noise_sigma), (s, s, 3)).astype(np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def make_plaster_facade_texture(
    size: int = 512,
    *,
    base_rgb: Tuple[int, int, int] = (152, 148, 138),
    fine_noise: float = 4.0,
    coarse_strength: float = 12.0,
    seed: int = 7,
) -> Image.Image:
    """Фасадная штукатурка: мелкий шум + низкочастотные пятна (value noise по крупной сетке)."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    base = np.array(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.ones((s, s, 3), dtype=np.float32) * base
    rgb += rng.normal(0.0, fine_noise, (s, s, 3)).astype(np.float32)

    grid = max(8, s // 32)
    gx = np.linspace(0, 1, grid, dtype=np.float32)
    gy = np.linspace(0, 1, grid, dtype=np.float32)
    corners = rng.normal(0.0, coarse_strength, (grid, grid, 3)).astype(np.float32)
    yy = np.linspace(0, 1, s, dtype=np.float32)
    xx = np.linspace(0, 1, s, dtype=np.float32)
    ix = (xx * (grid - 1)).astype(np.float32)
    iy = (yy * (grid - 1)).astype(np.float32)
    ix0 = np.clip(ix.astype(np.int32), 0, grid - 2)
    iy0 = np.clip(iy.astype(np.int32), 0, grid - 2)
    tx = (ix - ix0.astype(np.float32))[None, :, None]
    ty = (iy - iy0.astype(np.float32))[:, None, None]
    c00 = corners[iy0[:, None], ix0[None, :]]
    c10 = corners[iy0[:, None], ix0[None, :] + 1]
    c01 = corners[iy0[:, None] + 1, ix0[None, :]]
    c11 = corners[iy0[:, None] + 1, ix0[None, :] + 1]
    sx = 3 * tx**2 - 2 * tx**3
    sy = 3 * ty**2 - 2 * ty**3
    a = c00 * (1 - sx) + c10 * sx
    b = c01 * (1 - sx) + c11 * sx
    low = a * (1 - sy) + b * sy
    rgb += low.astype(np.float32)

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def make_vertical_stripes_texture(
    size: int = 512,
    *,
    stripe_period_px: int = 24,
    rgb_a: Tuple[int, int, int] = (130, 125, 118),
    rgb_b: Tuple[int, int, int] = (150, 145, 138),
    noise_sigma: float = 2.5,
    seed: int = 11,
) -> Image.Image:
    """Полосы по X (имитация досок / реек) + лёгкий шум."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    period = max(4, int(stripe_period_px))
    x = np.arange(s, dtype=np.int32)
    band = (x // period) % 2
    a = np.array(rgb_a, dtype=np.float32)
    b = np.array(rgb_b, dtype=np.float32)
    stripe = np.where(band[None, :, None] == 0, a.reshape(1, 1, 3), b.reshape(1, 1, 3))
    rgb = np.broadcast_to(stripe, (s, s, 3)).copy()
    rgb += rng.normal(0.0, noise_sigma, (s, s, 3)).astype(np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def make_wood_plank_color_texture(
    size: int = 512,
    *,
    plank_width_px: int = 28,
    seed: int = 101,
) -> Image.Image:
    """Процедурное дерево: доски по X, волокна и лёгкий перелив по ширине доски."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, s, dtype=np.float64)[:, None]
    xx = np.linspace(0.0, 1.0, s, dtype=np.float64)[None, :]
    pw = max(8, int(plank_width_px))
    x_idx = np.arange(s, dtype=np.int64)[None, :]
    plank = np.broadcast_to((x_idx // pw) % 131, (s, s))
    tint = (plank.astype(np.float64) * 0.618034) % 1.0

    base_r = 78.0 + 26.0 * tint + 10.0 * np.sin(xx * np.pi * 16.0 + yy * np.pi * 3.5)
    base_g = 54.0 + 20.0 * tint + 7.0 * np.sin(xx * np.pi * 13.0 + yy * np.pi * 2.8)
    base_b = 32.0 + 14.0 * tint + 5.0 * np.sin(xx * np.pi * 11.0)
    rgb = np.stack([base_r, base_g, base_b], axis=-1)

    grain = 0.045 * np.sin(xx * np.pi * 110.0 + yy * np.pi * 5.0 + rng.standard_normal((s, s)) * 0.25)
    grain2 = 0.03 * np.sin(xx * np.pi * 38.0 + yy * np.pi * 24.0)
    rgb[..., 0] += grain * 42.0 + grain2 * 32.0
    rgb[..., 1] += grain * 30.0 + grain2 * 24.0
    rgb[..., 2] += grain * 20.0 + grain2 * 16.0
    rgb += rng.normal(0.0, 2.2, (s, s, 3)).astype(np.float64)
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")
