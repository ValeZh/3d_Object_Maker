"""
Карты нормалей (RGB, tangent-friendly): строятся процедурно, без отдельной карты высот как входа/выхода.

Внутри может использоваться скалярное поле только для градиентов; на диск карта высот не пишется.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _ensure_size(size: int) -> int:
    return max(64, int(size))


def _normalize_normals(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    return nx / ln, ny / ln, nz / ln


def _pack_normal_rgb(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, *, invert_green: bool = True) -> np.ndarray:
    """В [0,255]^3: (0.5+0.5*n) в стиле OpenGL; invert_green — под Blender/DirectX-ориентиры."""
    if invert_green:
        ny = -ny
    r = np.clip(0.5 + 0.5 * nx, 0.0, 1.0)
    g = np.clip(0.5 + 0.5 * ny, 0.0, 1.0)
    b = np.clip(0.5 + 0.5 * nz, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def normals_from_scalar_slopes(
    height_like: np.ndarray,
    *,
    strength: float = 2.5,
    invert_green: bool = True,
) -> Image.Image:
    """
    Переводит 2D массив (любой «рельеф» в памяти) в карту нормалей по градиентам.
    Не создаёт файла карты высот; удобно как низкоуровневый шаг для процедурных полей ниже.
    """
    h = np.asarray(height_like, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height_like must be 2D")
    gy, gx = np.gradient(h)
    nx = -gx * float(strength)
    ny = -gy * float(strength)
    nz = np.ones_like(h, dtype=np.float64)
    nx, ny, nz = _normalize_normals(nx, ny, nz)
    rgb = _pack_normal_rgb(nx, ny, nz, invert_green=invert_green)
    return Image.fromarray(rgb, mode="RGB")


def _value_noise_2d(s: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    g = max(4, int(grid))
    corners = rng.standard_normal((g, g)).astype(np.float64)
    yy = np.linspace(0, 1, s, dtype=np.float64)
    xx = np.linspace(0, 1, s, dtype=np.float64)
    ix = (xx * (g - 1)).astype(np.float64)
    iy = (yy * (g - 1)).astype(np.float64)
    ix0 = np.clip(ix.astype(np.int32), 0, g - 2)
    iy0 = np.clip(iy.astype(np.int32), 0, g - 2)
    tx = (ix - ix0.astype(np.float64))[None, :]
    ty = (iy - iy0.astype(np.float64))[:, None]
    c00 = corners[iy0[:, None], ix0[None, :]]
    c10 = corners[iy0[:, None], ix0[None, :] + 1]
    c01 = corners[iy0[:, None] + 1, ix0[None, :]]
    c11 = corners[iy0[:, None] + 1, ix0[None, :] + 1]
    sx = 3 * tx**2 - 2 * tx**3
    sy = 3 * ty**2 - 2 * ty**3
    a = c00 * (1 - sx) + c10 * sx
    b = c01 * (1 - sx) + c11 * sx
    return (a * (1 - sy) + b * sy).astype(np.float64)


def make_stucco_like_normal_map(
    size: int = 512,
    *,
    strength: float = 3.0,
    coarse_grid: int = 16,
    fine_octaves: int = 2,
    seed: int = 19,
    invert_green: bool = True,
) -> Image.Image:
    """Крупные неровности штукатурки + пара слоёв мелкого value-noise (только нормали на выходе)."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    h = _value_noise_2d(s, coarse_grid, rng)
    for i in range(max(0, int(fine_octaves))):
        g = max(8, coarse_grid * (2 + i * 2))
        h += 0.35 / (1 + i) * _value_noise_2d(s, g, rng)
    h -= float(np.mean(h))
    return normals_from_scalar_slopes(h, strength=strength, invert_green=invert_green)


def make_fine_noise_normal_map(
    size: int = 512,
    *,
    strength: float = 8.0,
    grid: int = 48,
    seed: int = 3,
    invert_green: bool = True,
) -> Image.Image:
    """Мелкий «песок» по градиентам плотного value-noise (без карты высот)."""
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    h = _value_noise_2d(s, max(12, int(grid)), rng)
    h -= float(np.mean(h))
    return normals_from_scalar_slopes(h, strength=strength, invert_green=invert_green)


def make_wood_grain_normal_map(
    size: int = 512,
    *,
    plank_width_px: int = 28,
    seam_strength: float = 1.5,
    grain_strength: float = 0.62,
    bump_octaves: int = 5,
    slope_strength: float = 9.0,
    fine_grid: int = 28,
    seed: int = 103,
    invert_green: bool = True,
) -> Image.Image:
    """
    Нормали под деревянную раму: швы досок, волокна и многослойный шум (зёрна / «бугорки»).

    Несколько октав ``_value_noise_2d`` дают локальные выпуклости; ``slope_strength`` усиливает
    видимый рельеф при переводе градиентов в нормали. Карта высот на диск не пишется.
    """
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:s, 0:s].astype(np.float64)
    xx = xs / max(float(s - 1), 1.0)
    yy = ys / max(float(s - 1), 1.0)
    pw = max(8, int(plank_width_px))

    h = np.sin(xs * (2.0 * np.pi / float(pw))).astype(np.float64) * float(seam_strength)

    g_long = np.sin(xx * np.pi * 92.0 + yy * np.pi * 30.0)
    h += g_long * float(grain_strength)
    h += (np.abs(g_long) ** 1.4) * float(grain_strength) * 0.42

    g_med = np.sin(xx * np.pi * 24.0 + yy * np.pi * 12.0 + 1.15)
    h += g_med * (float(grain_strength) * 0.55)
    h += np.sin(xx * np.pi * 160.0 + yy * np.pi * 6.0) * (float(grain_strength) * 0.22)

    base_grid = max(10, int(fine_grid))
    for i in range(max(1, int(bump_octaves))):
        gsz = max(8, base_grid + i * 12)
        amp = (0.52**i) * float(grain_strength) * 2.05
        h += _value_noise_2d(s, gsz, rng) * amp

    pore = _value_noise_2d(s, max(s // 2, 40), rng)
    h += pore * pore * (float(grain_strength) * 0.55)
    h += _value_noise_2d(s, max(s - 4, 96), rng) * (float(grain_strength) * 0.18)

    h -= float(np.mean(h))
    return normals_from_scalar_slopes(h, strength=float(slope_strength), invert_green=invert_green)


def make_ceramic_tile_normal_map(
    size: int = 512,
    *,
    tiles_per_side: int = 8,
    grout_width_frac: float = 0.06,
    grout_depth: float = 0.62,
    edge_bevel: float = 0.12,
    micro_noise: float = 0.018,
    slope_strength: float = 7.5,
    seed: int = 97,
    invert_green: bool = True,
) -> Image.Image:
    """
    Нормали под керамическую плитку: углублённые швы + лёгкая фаска по краям + мелкий микрорельеф.

    Высотная карта используется только как промежуточное поле в памяти; на диск не сохраняется.
    """
    s = _ensure_size(size)
    rng = np.random.default_rng(seed)
    n = max(2, int(tiles_per_side))
    gw = float(np.clip(grout_width_frac, 0.01, 0.24))
    depth = float(max(0.0, grout_depth))
    bevel = float(max(0.0, edge_bevel))

    yy, xx = np.mgrid[0:s, 0:s]
    fx = (xx / max(s - 1, 1)) * n
    fy = (yy / max(s - 1, 1)) * n
    frac_x = fx - np.floor(fx)
    frac_y = fy - np.floor(fy)
    edge = np.minimum(np.minimum(frac_x, 1.0 - frac_x), np.minimum(frac_y, 1.0 - frac_y))

    # Base tile level with shallow center crown.
    h = np.full((s, s), 1.0, dtype=np.float64)
    center = np.clip((edge - gw * 0.5) / max(0.5 - gw * 0.5, 1e-6), 0.0, 1.0)
    h += (center * center) * 0.055

    # Grout valley (smooth transition to avoid jagged normals at seam borders).
    seam_band = np.clip((gw * 0.5 - edge) / max(gw * 0.5, 1e-6), 0.0, 1.0)
    seam_smooth = seam_band * seam_band * (3.0 - 2.0 * seam_band)
    h -= seam_smooth * depth

    # Slight bevel near tile border.
    bevel_zone = np.clip((gw * 0.5 + bevel - edge) / max(bevel, 1e-6), 0.0, 1.0)
    h -= bevel_zone * 0.08

    # Subtle micro-roughness over tile faces (less inside grout).
    noise = _value_noise_2d(s, max(20, n * 6), rng)
    noise -= float(np.mean(noise))
    h += noise * float(micro_noise) * (1.0 - seam_smooth * 0.75)
    return normals_from_scalar_slopes(h, strength=float(slope_strength), invert_green=invert_green)


def make_soft_frosted_glass_normal_map(
    size: int = 512,
    *,
    strength: float = 1.8,
    grid: int = 56,
    seed: int = 201,
    invert_green: bool = True,
) -> Image.Image:
    """Почти гладкая нормаль для стекла (лёгкий шум)."""
    return make_fine_noise_normal_map(size, strength=strength, grid=grid, seed=seed, invert_green=invert_green)


def make_neutral_flat_normal_map(
    size: int = 512,
    *,
    invert_green: bool = True,
) -> Image.Image:
    """Идеально гладкая поверхность в tangent space: нормаль (0,0,1) → RGB без микрорельефа."""
    s = _ensure_size(size)
    nx = np.zeros((s, s), dtype=np.float64)
    ny = np.zeros((s, s), dtype=np.float64)
    nz = np.ones((s, s), dtype=np.float64)
    rgb = _pack_normal_rgb(nx, ny, nz, invert_green=invert_green)
    return Image.fromarray(rgb, mode="RGB")
