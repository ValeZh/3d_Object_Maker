"""
Procedural surface texture packs with normal maps.

Generates several material families:
- rough wall
- cracked wall
- plaster wall
- roof shingles
- ceramic tiles
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_textures_dir() -> Path:
    return _repo_root() / "data" / "textures"


def _smooth3(a: np.ndarray, rounds: int = 1) -> np.ndarray:
    out = a.astype(np.float32, copy=True)
    for _ in range(max(1, rounds)):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), -1, axis=1)
        ) / 9.0
    return out


def _fractal_noise(size: int, *, seed: int, octaves: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for i in range(max(1, octaves)):
        layer = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        layer = _smooth3(layer, rounds=2 + i * 2)
        out += layer * amp
        total += amp
        amp *= 0.5
    out = out / max(total, 1e-6)
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def _height_to_normal(height: np.ndarray, *, strength: float = 4.0) -> Image.Image:
    h = height.astype(np.float32)
    dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(h, dtype=np.float32)
    n = np.stack([nx, ny, nz], axis=-1)
    l2 = np.sqrt((n * n).sum(axis=-1, keepdims=True))
    n = n / np.clip(l2, 1e-6, None)
    rgb = ((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _gray_to_rgb(gray: np.ndarray) -> Image.Image:
    g8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.repeat(g8[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _pack_from_height(height: np.ndarray, base_color: tuple[int, int, int], *, normal_strength: float) -> Dict[str, Image.Image]:
    h = np.clip(height.astype(np.float32), 0.0, 1.0)
    color = np.array(base_color, dtype=np.float32).reshape(1, 1, 3)
    tint = 0.65 + 0.5 * h[:, :, None]
    albedo = np.clip(color * tint, 0.0, 255.0).astype(np.uint8)
    rough = np.clip(0.45 + 0.5 * h, 0.0, 1.0)
    return {
        "albedo": Image.fromarray(albedo, mode="RGB"),
        "normal": _height_to_normal(h, strength=normal_strength),
        "roughness": _gray_to_rgb(rough),
    }


def make_rough_wall_pack(
    size: int = 512,
    *,
    seed: int = 101,
    rough_scale: float = 1.0,
    rough_contrast: float = 1.0,
) -> Dict[str, Image.Image]:
    s = max(int(size), 64)
    n = _fractal_noise(s, seed=seed, octaves=5)
    smooth_rounds = int(max(1.0, 1.0 + float(rough_scale) * 2.0))
    n = _smooth3(n, rounds=smooth_rounds)
    n = np.clip(0.5 + (n - 0.5) * float(max(0.1, rough_contrast)), 0.0, 1.0)
    return _pack_from_height(n, (148, 142, 134), normal_strength=5.0)


def make_cracked_wall_pack(
    size: int = 512,
    *,
    seed: int = 202,
    crack_density: float = 1.0,
    crack_width: int = 2,
    crack_length_scale: float = 1.0,
    crack_depth: float = 1.35,
) -> Dict[str, Image.Image]:
    s = max(int(size), 64)
    rng = np.random.default_rng(seed)
    base = _fractal_noise(s, seed=seed, octaves=4)
    cracks = np.zeros((s, s), dtype=np.float32)
    n_cracks = int(max(4, (s // 28) * max(0.1, crack_density)))
    line_w = int(max(1, crack_width))
    for _ in range(n_cracks):
        x0 = rng.integers(0, s)
        y0 = rng.integers(0, s)
        min_len = int(max(8, (s // 18) * max(0.2, crack_length_scale)))
        max_len = int(max(min_len + 1, (s // 3) * max(0.3, crack_length_scale)))
        length = int(rng.integers(min_len, max_len))
        angle = float(rng.uniform(0.0, np.pi))
        for t in range(length):
            x = int(x0 + np.cos(angle) * t)
            y = int(y0 + np.sin(angle) * t)
            if 0 <= x < s and 0 <= y < s:
                cracks[y, x] = 1.0
                for dy in range(-line_w + 1, line_w):
                    for dx in range(-line_w + 1, line_w):
                        xx = x + dx
                        yy = y + dy
                        if 0 <= xx < s and 0 <= yy < s:
                            d = abs(dx) + abs(dy)
                            v = max(0.2, 1.0 - 0.25 * d)
                            cracks[yy, xx] = max(cracks[yy, xx], v)
    # Keep crack edges sharper so normal map gets stronger local gradients.
    cracks = _smooth3(cracks, rounds=max(1, line_w // 3))
    depth = float(max(0.2, crack_depth))
    height = np.clip(base * 0.8 - cracks * (0.85 * depth) + 0.25, 0.0, 1.0)
    pack = _pack_from_height(height, (152, 145, 138), normal_strength=12.0 * min(depth, 3.0))
    # Cracked wall should be mostly matte; only crack valleys slightly smoother.
    rough = np.clip(0.82 - cracks * 0.18, 0.55, 0.95)
    pack["roughness"] = _gray_to_rgb(rough)
    return pack


def make_plaster_wall_pack(size: int = 512, *, seed: int = 303) -> Dict[str, Image.Image]:
    s = max(int(size), 64)
    low = _fractal_noise(s, seed=seed, octaves=3)
    high = _fractal_noise(s, seed=seed + 1, octaves=6)
    height = np.clip(0.7 * low + 0.3 * high, 0.0, 1.0)
    return _pack_from_height(height, (210, 206, 198), normal_strength=3.2)


def make_roof_shingles_pack(
    size: int = 512,
    *,
    seed: int = 404,
    shingle_rows: int = 12,
    shingle_wave_freq: float = 8.0,
) -> Dict[str, Image.Image]:
    s = max(int(size), 64)
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, s, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, s, dtype=np.float32)[None, :]
    rows = int(max(3, shingle_rows))
    row_band = np.mod(y * rows, 1.0)
    scallop = np.sin(2.0 * np.pi * (x * max(1.0, shingle_wave_freq) + y * 0.7)) * 0.5 + 0.5
    grain = _fractal_noise(s, seed=seed + 2, octaves=4)
    height = 0.35 + 0.35 * (1.0 - row_band) + 0.18 * scallop + 0.12 * grain
    height = np.clip(height + rng.normal(0.0, 0.02, (s, s)).astype(np.float32), 0.0, 1.0)
    return _pack_from_height(height, (122, 82, 62), normal_strength=6.0)


def make_ceramic_tiles_pack(
    size: int = 512,
    *,
    seed: int = 505,
    tiles_per_side: int = 8,
    grout_width: float = 0.06,
) -> Dict[str, Image.Image]:
    s = max(int(size), 64)
    n_tiles = int(max(2, tiles_per_side))
    grout_w = float(max(0.01, min(0.2, grout_width)))
    yy, xx = np.mgrid[0:s, 0:s]
    fx = (xx / s) * n_tiles
    fy = (yy / s) * n_tiles
    gx = np.abs((fx - np.floor(fx)) - 0.5)
    gy = np.abs((fy - np.floor(fy)) - 0.5)
    edge = np.minimum(gx, gy)
    grout = np.clip((grout_w - edge) / grout_w, 0.0, 1.0)
    noise = _fractal_noise(s, seed=seed, octaves=5)
    tile = np.clip(0.72 + 0.2 * noise, 0.0, 1.0)
    height = np.clip(tile - grout * 0.55, 0.0, 1.0)
    return _pack_from_height(height, (190, 194, 201), normal_strength=5.2)


def ensure_surface_textures(
    out_dir: Path | None = None,
    *,
    size: int = 512,
    force: bool = False,
    rough_scale: float = 1.0,
    rough_contrast: float = 1.0,
    crack_density: float = 1.0,
    crack_width: int = 2,
    crack_length_scale: float = 1.0,
    crack_depth: float = 1.35,
    tiles_per_side: int = 8,
    grout_width: float = 0.06,
    shingle_rows: int = 12,
    shingle_wave_freq: float = 8.0,
) -> Dict[str, Path]:
    out = out_dir or default_textures_dir()
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    packs = {
        "wall_rough": make_rough_wall_pack(
            size=size,
            rough_scale=rough_scale,
            rough_contrast=rough_contrast,
        ),
        "wall_cracked": make_cracked_wall_pack(
            size=size,
            crack_density=crack_density,
            crack_width=crack_width,
            crack_length_scale=crack_length_scale,
            crack_depth=crack_depth,
        ),
        "wall_plaster": make_plaster_wall_pack(size=size),
        "roof_shingles": make_roof_shingles_pack(
            size=size,
            shingle_rows=shingle_rows,
            shingle_wave_freq=shingle_wave_freq,
        ),
        "tile_ceramic": make_ceramic_tiles_pack(
            size=size,
            tiles_per_side=tiles_per_side,
            grout_width=grout_width,
        ),
    }
    for name, pack in packs.items():
        for map_name, img in pack.items():
            p = out / f"{name}_{map_name}.png"
            if force or not p.is_file():
                img.save(p)
            written[f"{name}_{map_name}"] = p
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate procedural wall/roof/tile textures with normal maps.")
    ap.add_argument("--size", type=int, default=512, help="Texture side size in pixels.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    ap.add_argument("--tiles-per-side", type=int, default=8, help="How many tiles fit one side of tile texture.")
    ap.add_argument("--grout-width", type=float, default=0.06, help="Tile grout width in UV-space (0.01..0.2).")
    ap.add_argument("--rough-scale", type=float, default=1.0, help="Rough wall feature scale (larger = broader bumps).")
    ap.add_argument("--rough-contrast", type=float, default=1.0, help="Rough wall contrast (larger = stronger relief).")
    ap.add_argument("--crack-density", type=float, default=1.0, help="Crack count multiplier.")
    ap.add_argument("--crack-width", type=int, default=2, help="Crack line thickness in pixels.")
    ap.add_argument("--crack-length-scale", type=float, default=1.0, help="Crack length multiplier.")
    ap.add_argument("--crack-depth", type=float, default=1.35, help="Crack depth multiplier (larger = deeper cracks).")
    ap.add_argument("--shingle-rows", type=int, default=12, help="Roof shingle rows per texture.")
    ap.add_argument("--shingle-wave-freq", type=float, default=8.0, help="Roof shingle wave frequency.")
    ap.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: data/textures).",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    files = ensure_surface_textures(
        out_dir=out_dir,
        size=args.size,
        force=args.force,
        rough_scale=args.rough_scale,
        rough_contrast=args.rough_contrast,
        crack_density=args.crack_density,
        crack_width=args.crack_width,
        crack_length_scale=args.crack_length_scale,
        crack_depth=args.crack_depth,
        tiles_per_side=args.tiles_per_side,
        grout_width=args.grout_width,
        shingle_rows=args.shingle_rows,
        shingle_wave_freq=args.shingle_wave_freq,
    )
    print("[OK] Generated texture pack:")
    for key, path in files.items():
        print(f"  - {key}: {path}")

