"""
Атлас подъезда 3×1 (стена | крыша | дверь) и склейка мешей с per-vertex UV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import trimesh
from PIL import Image

from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint
from src.generator.procedural.procedural_texture_maps.procedural_color_texture import (
    make_ceramic_tile_color_texture,
    make_plaster_facade_texture,
    make_uniform_noise_texture,
    make_vertical_stripes_texture,
    make_wood_plank_color_texture,
)

ENTRANCE_ATLAS_NUM_TILES = 3
TILE_WALL = 0
TILE_ROOF = 1
TILE_DOOR = 2


def entrance_part_tile_index(part_name: str) -> int:
    n = part_name.lower()
    if "door" in n:
        return TILE_DOOR
    if n.startswith("ceiling") or n == "canopy":
        return TILE_ROOF
    return TILE_WALL


def scale_uv_to_atlas_tile(uv: np.ndarray, tile_index: int, n_tiles: int = ENTRANCE_ATLAS_NUM_TILES) -> np.ndarray:
    w = 1.0 / float(n_tiles)
    out = np.asarray(uv, dtype=np.float64).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0) * w + w * float(tile_index)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


def _proc_tile_rgb(tile_i: int, rgb: Tuple[int, int, int], size: int) -> Image.Image:
    rng = np.random.default_rng(42 + tile_i * 17)
    s = max(int(size), 64)
    b = np.ones((s, s, 3), dtype=np.float32) * np.array(rgb, dtype=np.float32)
    b += rng.normal(0.0, 5.0, (s, s, 3)).astype(np.float32)
    return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), mode="RGB")


def _proc_tile_preset(tile_i: int, preset: str | None, size: int, default_rgb: Tuple[int, int, int]) -> Image.Image:
    s = max(int(size), 64)
    kind = str(preset or "").strip().lower()
    if kind in ("plaster", "stucco"):
        return make_plaster_facade_texture(s, base_rgb=default_rgb, seed=41 + tile_i * 17)
    if kind in ("uniform_noise", "noise"):
        return make_uniform_noise_texture(s, base_rgb=default_rgb, noise_sigma=6.0, seed=53 + tile_i * 19)
    if kind in ("vertical_stripes", "stripes"):
        return make_vertical_stripes_texture(s, stripe_period_px=max(8, s // 16), seed=67 + tile_i * 13)
    if kind in ("wood", "wood_plank"):
        return make_wood_plank_color_texture(s, plank_width_px=max(10, s // 18), seed=71 + tile_i * 11)
    if kind in ("ceramic", "tile", "ceramic_tile"):
        return make_ceramic_tile_color_texture(s, tiles_per_side=max(6, s // 64), grout_width_frac=0.06, seed=89 + tile_i * 7)
    return _proc_tile_rgb(tile_i, default_rgb, s)


def _open_rgb_resize(path: Path | str, size: int) -> Image.Image:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return im.resize((size, size), resample)


def make_entrance_atlas(
    *,
    tile: int = 256,
    wall_tex: str | Path | None = None,
    roof_tex: str | Path | None = None,
    door_tex: str | Path | None = None,
    wall_tex_color: Any = None,
    roof_tex_color: Any = None,
    door_tex_color: Any = None,
    wall_proc_preset: str | None = None,
    roof_proc_preset: str | None = None,
    door_proc_preset: str | None = None,
) -> Image.Image:
    """Горизонтальный атлас: [стена | крыша | дверь].

    ``*_tex_color`` — опциональный тинт RGB для соответствующего тайла (файл или процедурный),
    см. ``parse_texture_color_tint``.
    """
    t = max(int(tile), 64)
    paths = (wall_tex, roof_tex, door_tex)
    color_specs = (wall_tex_color, roof_tex_color, door_tex_color)
    defaults = ((150, 142, 132), (120, 128, 140), (92, 72, 58))
    presets = (wall_proc_preset, roof_proc_preset, door_proc_preset)
    tints = tuple(parse_texture_color_tint(c) for c in color_specs)
    tiles: List[Image.Image] = []
    for i, (p, rgb) in enumerate(zip(paths, defaults)):
        if p is not None:
            pp = Path(p).expanduser().resolve()
            if pp.is_file():
                tiles.append(apply_texture_color_tint(_open_rgb_resize(pp, t), tints[i]))
                continue
        tiles.append(apply_texture_color_tint(_proc_tile_preset(i, presets[i], t, rgb), tints[i]))
    w = t * ENTRANCE_ATLAS_NUM_TILES
    out = Image.new("RGB", (w, t))
    for i, im in enumerate(tiles):
        out.paste(im, (i * t, 0))
    return out


def concatenate_entrance_uv_meshes(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    vs: List[np.ndarray] = []
    fs: List[np.ndarray] = []
    uvs: List[np.ndarray] = []
    off = 0
    for m in parts:
        if m is None or len(m.faces) == 0:
            continue
        vv = np.asarray(m.vertices, dtype=np.float64)
        ff = np.asarray(m.faces, dtype=np.int64) + off
        u = getattr(m.visual, "uv", None)
        if u is None:
            raise RuntimeError("entrance atlas: submesh without per-vertex uv")
        u = np.asarray(u, dtype=np.float64)
        if len(u) != len(vv):
            raise RuntimeError("entrance atlas: uv/vertex count mismatch")
        vs.append(vv)
        fs.append(ff)
        uvs.append(u)
        off += len(vv)
    if not vs:
        return trimesh.Trimesh()
    out = trimesh.Trimesh(vertices=np.vstack(vs), faces=np.vstack(fs), process=False, validate=False)
    out.visual = trimesh.visual.texture.TextureVisuals(uv=np.vstack(uvs))
    out.update_faces(out.nondegenerate_faces())
    return out
