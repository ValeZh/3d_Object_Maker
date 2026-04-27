"""
Процедурный балкон (трапеция в плане), как типичный лоджия/балкон в МКД:

  • Низ — полноширинный парапет (глухая зона) по переду и бокам.
  • По умолчанию без окон (window_mode=none, глухой фронт и бока); остекление — --window-mode with_glass и т.д.
  • Над парапетом на боках — стена или «стекло» (side_upper_mode).
  • Основание: либо width_back / width_front / depth, либо четыре угла (левый/правый у стены, перед лев/прав).
  • По умолчанию вертикальная призма (бока — прямоугольники); --legacy-tilt-top — старый наклон верха.
  • Перед под окном — один прямоугольник парапета; опционально окна на левой/правой боковой стене.
  • Внутренняя (задняя) стена BL—BR: окна inner_wall_windows / --inner-wall-window; двери inner_wall_doors / --inner-wall-door (procedural_door, вырез в стене).
  • Отдельно фронт над парапетом: --front-window-mode (open = дыра, none = стена, frame_only | with_glass).
  • Проём без геометрии на боку: --open-side-left / --open-side-right (над парапетом; по сторонам BL—FL и BR—FR соответственно).
  • Толщина вертикальных стен (зад/бок/парапет): wall_thickness / --wall-thickness (м); вынос наружу в −n_in (в плане XY к центру пола). 0 — одна грань.
  • Атлас 7 PNG (колонки): низ | верх | рама | стекло | бок «корзина» | бок у окна | грань между ними.
    Низ/верх/рама/стекло: --wall-lower-tex, --wall-upper-tex, --frame-tex, --glass-tex.
    Боковые зоны: --side-basket-tex, --side-jamb-tex, --side-separator-tex; доля полосы у окна — --side-parapet-split-frac (0 = без разреза).
    Грань-перегородка и разрез не добавляются на стороне, где есть боковое окно (если window_mode не none).

Оси: X — вдоль фасада, Y — наружу от здания, Z — вверх. Для углов основания задаёте (x,y) при z=0.

Запуск:
  python -m src.generator.procedural.procedural_balcony export -o data/balcony_export
  python -m src.generator.procedural.procedural_balcony export --parapet-frac 0.45 --mullions-vertical 3
  python -m src.generator.procedural.procedural_balcony export --simple-box --mullions-vertical 5
  python -m src.generator.procedural.procedural_balcony export --window-left-wall --window-right-wall
  # Лоджия: дыры на фронте и на боку BL—FL, окно в задней стене:
  #   --window-mode none --front-window-mode open --open-side-right --inner-wall-window "0.06,0.94,Zb,Zt,mv=2"
  # PowerShell: углы основания так — --floor-left-wall=-0.8,0 --floor-right-wall=0.8,0 ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generator.procedural.procedural_door import build_french_double_door_parts, build_simple_door_slab
from src.generator.procedural.open3d_preview import preview_balcony_obj_open3d
from src.generator.procedural.procedural_window import (
    build_window_frame_glass_meshes,
    _frame_thickness,
    _normalize_partial_horizontal_bars,
    _parse_partial_h_tokens,
    _pick_kind,
    _pick_nonneg_int,
)
from src.generator.procedural.texturing import make_window_frame_texture, make_window_glass_texture
from src.generator.procedural.texturing.pbr_map_utils import make_normal_map_from_albedo, make_roughness_map_from_albedo
from src.generator.procedural.unfolding import faceted_triplanar_uv


# Плитки атласа (слева направо): низ | верх | рама | стекло | бок корзина | бок у окна | перегородка
BALCONY_ATLAS_NUM_TILES = 7
BALCONY_TILE_WALL_LOWER = 0
BALCONY_TILE_WALL_UPPER = 1
BALCONY_TILE_FRAME = 2
BALCONY_TILE_GLASS = 3
BALCONY_TILE_SIDE_BASKET = 4
BALCONY_TILE_SIDE_JAMB = 5
BALCONY_TILE_SIDE_SEPARATOR = 6


USER_BALCONY: dict[str, Any] = {
    "width_back": 1.6,
    "width_front": 2.0,
    "depth": 1.15,
    "height": 2.15,
    "simple_box": False,
    # Углы основания z=0 (x, y), если все None — из width_back/width_front/depth:
    # левый у стены, правый у стены, передний левый, передний правый (к улице).
    "floor_corner_left_wall": None,
    "floor_corner_right_wall": None,
    "floor_corner_front_left": None,
    "floor_corner_front_right": None,
    "vertical_prism": True,
    "window_left_wall": False,
    "window_right_wall": False,
    "floor_thickness": 0.14,
    "wall_thickness": 0.0,
    "parapet_z_frac": 0.42,
    "parapet_height": None,
    "window_mode": "none",
    # Если None — как window_mode. Иначе только фронт: open (дыра) | none (стена) | frame_only | with_glass.
    "front_window_mode": None,
    # Имена как у CLI: left → убрать грань BR—FR; right → убрать BL—FL (см. дефолтную ориентацию -X/+X).
    "open_left_above_parapet": False,
    "open_right_above_parapet": False,
    "window_depth": 0.14,
    "tilt_left_deg": 0.0,
    "tilt_right_deg": 0.0,
    "wall_upper_z_frac": 0.35,
    "side_upper_mode": "wall",
    "mullions_vertical": 0,
    "mullions_horizontal": 0,
    "mullion_offset_x": 0.0,
    "mullion_offset_z": 0.0,
    "partial_horizontal_bars": [],
    "window_kind": "fixed",
    "parapet_sill": True,
    "sill_thickness": 0.06,
    "sill_depth": 0.1,
    # Доля глубины бокового парапета от угла у окна (FL/FR) к заднему углу — полоса «у окна»; остальное — «корзина».
    # По умолчанию 0 — один квад на бок (без лишних рёбер jamb/basket/separator); >0 — три зоны и грань-перегородка.
    "side_parapet_split_frac": 0.0,
    "side_parapet_separator_depth": 0.022,
    # Окна во внутренней (задней) стене BL—BR, в сторону балкона. Список словарей:
    # u0,u1 (0..1 вдоль BL→BR), z_bottom,z_top (м мира); опционально window_mode, mullions_*, window_depth, window_kind, …
    "inner_wall_windows": [],
    "inner_wall_doors": [],
}


def _repo_root() -> Path:
    return _REPO_ROOT


def _parse_floor_xy_arg(s: str) -> Tuple[float, float]:
    """Строка 'x,y' или 'x y' — угол основания в метрах (план XY, z=0)."""
    t = s.replace(",", " ").split()
    if len(t) != 2:
        raise argparse.ArgumentTypeError(f"Нужно два числа x,y, получено: {s!r}")
    return float(t[0]), float(t[1])


def _quad(v: List[List[float]], flip: bool = False) -> trimesh.Trimesh:
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in v]
    f = [[0, 1, 2], [0, 2, 3]]
    if flip:
        f = [[0, 2, 1], [0, 3, 2]]
    return trimesh.Trimesh(vertices=[a, b, c, d], faces=f, process=False)


def _point_close_to_segment_2d(p_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray, eps: float) -> bool:
    """Точка p близка к отрезку ab в плоскости XY."""
    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    a = np.asarray(a_xy, dtype=np.float64).reshape(2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(2)
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-18:
        return float(np.linalg.norm(p - a)) < eps
    t = float(np.dot(p - a, ab) / l2)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)) < eps


def _segment_inward_normal_xy(a_xy: np.ndarray, b_xy: np.ndarray, floor_cxy: np.ndarray) -> np.ndarray:
    """Единичная нормаль к ребру ab в XY, направленная к центру пола (внутрь балкона)."""
    mid = 0.5 * (np.asarray(a_xy, dtype=np.float64) + np.asarray(b_xy, dtype=np.float64))
    e = np.asarray(b_xy, dtype=np.float64) - np.asarray(a_xy, dtype=np.float64)
    le = float(np.linalg.norm(e))
    if le < 1e-12:
        return np.array([0.0, 1.0], dtype=np.float64)
    e = e / le
    c1 = np.array([-e[1], e[0]], dtype=np.float64)
    c2 = np.array([e[1], -e[0]], dtype=np.float64)
    fc = np.asarray(floor_cxy, dtype=np.float64)[:2]
    toward = fc - mid
    n = c1 if float(np.dot(c1, toward)) >= float(np.dot(c2, toward)) else c2
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.array([0.0, 1.0], dtype=np.float64)


def _miter_outward_offset_xy(
    p_xy: np.ndarray,
    floor_xy_ring: List[np.ndarray],
    floor_cxy: np.ndarray,
    thickness: float,
    eps: float = 0.008,
) -> Optional[np.ndarray]:
    """Сдвиг наружу в XY: −t·Σ n_i по рёрам контура, к которым близка p (внешний угол без щели)."""
    acc = np.zeros(2, dtype=np.float64)
    fc = np.asarray(floor_cxy, dtype=np.float64)[:2]
    px = np.asarray(p_xy, dtype=np.float64).reshape(2)
    nseg = len(floor_xy_ring)
    for i in range(nseg):
        ai = np.asarray(floor_xy_ring[i][:2], dtype=np.float64)
        bi = np.asarray(floor_xy_ring[(i + 1) % nseg][:2], dtype=np.float64)
        if _point_close_to_segment_2d(px, ai, bi, eps):
            acc += _segment_inward_normal_xy(ai, bi, fc)
    if float(np.linalg.norm(acc)) < 1e-10:
        return None
    return -float(thickness) * acc


def _raw_wall_slab_from_quad(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    n_in_xy: np.ndarray,
    thickness: float,
    *,
    flip: bool,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
    floor_cxy: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """Призма: внутренняя грань bl—br—tr—tl, наружная — митер по контуру основания или −n·t."""
    wt = max(float(thickness), 0.0)
    if wt <= 1e-9:
        return _quad([bl, br, tr, tl], flip=flip)
    n2 = np.asarray(n_in_xy[:2], dtype=np.float64)
    ln = float(np.linalg.norm(n2))
    if ln < 1e-9:
        n2 = np.array([1.0, 0.0], dtype=np.float64)
    else:
        n2 = n2 / ln
    n3 = np.array([float(n2[0]), float(n2[1]), 0.0], dtype=np.float64)
    uni_off = -n3 * wt
    bi = [np.asarray(bl, dtype=np.float64), np.asarray(br, dtype=np.float64), np.asarray(tr, dtype=np.float64), np.asarray(tl, dtype=np.float64)]
    bo: List[np.ndarray] = []
    for p in bi:
        off_xy: Optional[np.ndarray] = None
        if floor_xy_ring is not None and floor_cxy is not None:
            off_xy = _miter_outward_offset_xy(p[:2], floor_xy_ring, floor_cxy, wt)
        if off_xy is None:
            off_xy = uni_off[:2]
        bo.append(p + np.array([float(off_xy[0]), float(off_xy[1]), 0.0], dtype=np.float64))
    parts: List[trimesh.Trimesh] = []
    parts.append(_quad(bi, flip=flip))
    parts.append(_quad(bo, flip=not flip))
    for i in range(4):
        j = (i + 1) % 4
        parts.append(_quad([bi[i], bi[j], bo[j], bo[i]], flip=False))
    m = trimesh.util.concatenate(parts)
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def _lerp_z(p0: np.ndarray, p1: np.ndarray, z_cut: float) -> np.ndarray:
    z0, z1 = float(p0[2]), float(p1[2])
    if abs(z1 - z0) < 1e-9:
        return p0.copy()
    t = (z_cut - z0) / (z1 - z0)
    t = float(np.clip(t, 0.0, 1.0))
    return p0 + t * (p1 - p0)


def _edge_at_height(p_bot: np.ndarray, p_top: np.ndarray, z: float) -> np.ndarray:
    """Точка на отрезке нижняя (z=0) — верхняя (z=H) при заданной высоте z."""
    return _lerp_z(p_bot, p_top, z)


def _make_wall_stack(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    z_cut: float,
    *,
    flip: bool = False,
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Нижняя кромка bl—br, верхняя tl—tr (tl над bl, tr над br). flip=True — инвертировать нормаль."""
    blc = _lerp_z(bl, tl, z_cut)
    brc = _lerp_z(br, tr, z_cut)
    wt = max(float(wall_thickness), 0.0)
    if wt <= 1e-9 or floor_cxy is None:
        lower = _quad([bl, br, brc, blc], flip=flip)
        upper = _quad([blc, brc, tr, tl], flip=flip)
        return lower, upper
    mid_lo = 0.25 * (bl[:2] + br[:2] + brc[:2] + blc[:2])
    n_lo = _inward_horizontal(mid_lo, floor_cxy)
    lower = _raw_wall_slab_from_quad(
        bl, br, brc, blc, n_lo, wt, flip=flip, floor_xy_ring=floor_xy_ring, floor_cxy=floor_cxy
    )
    mid_hi = 0.25 * (blc[:2] + brc[:2] + tr[:2] + tl[:2])
    n_hi = _inward_horizontal(mid_hi, floor_cxy)
    upper = _raw_wall_slab_from_quad(
        blc, brc, tr, tl, n_hi, wt, flip=flip, floor_xy_ring=floor_xy_ring, floor_cxy=floor_cxy
    )
    return lower, upper


def _vertical_wall_slab_textured(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    tile_i: int,
    uv4: np.ndarray,
    *,
    flip: bool,
    n_in_xy: np.ndarray,
    thickness: float,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
    floor_cxy: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """Вертикальная стена с UV: при thickness>0 — призма наружу; углы — митер по контуру основания."""
    wt = max(float(thickness), 0.0)
    if wt <= 1e-9:
        return _textured_quad([bl, br, tr, tl], tile_i, uv4, flip=flip)
    n2 = np.asarray(n_in_xy[:2], dtype=np.float64)
    ln = float(np.linalg.norm(n2))
    if ln < 1e-9:
        n2 = np.array([1.0, 0.0], dtype=np.float64)
    else:
        n2 = n2 / ln
    n3 = np.array([float(n2[0]), float(n2[1]), 0.0], dtype=np.float64)
    uni_off = -n3 * wt
    bi = [np.asarray(bl, dtype=np.float64), np.asarray(br, dtype=np.float64), np.asarray(tr, dtype=np.float64), np.asarray(tl, dtype=np.float64)]
    bo: List[np.ndarray] = []
    for p in bi:
        off_xy: Optional[np.ndarray] = None
        if floor_xy_ring is not None and floor_cxy is not None:
            off_xy = _miter_outward_offset_xy(p[:2], floor_xy_ring, floor_cxy, wt)
        if off_xy is None:
            off_xy = uni_off[:2]
        bo.append(p + np.array([float(off_xy[0]), float(off_xy[1]), 0.0], dtype=np.float64))
    parts: List[trimesh.Trimesh] = []
    parts.append(_textured_quad(bi, tile_i, uv4, flip=flip))
    parts.append(_textured_quad(bo, tile_i, uv4, flip=not flip))
    for i in range(4):
        j = (i + 1) % 4
        uvi = np.stack([uv4[i], uv4[j], uv4[j], uv4[i]])
        parts.append(_textured_quad([bi[i], bi[j], bo[j], bo[i]], tile_i, uvi, flip=False))
    m = trimesh.util.concatenate(parts)
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def _slab_or_quad_vertical_wall(
    vertices_4: List[np.ndarray],
    tile_i: int,
    uvs01: np.ndarray,
    *,
    flip: bool,
    floor_cxy: Optional[np.ndarray],
    wall_thickness: float,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> trimesh.Trimesh:
    """Вертикальная грань: при wall_thickness>0 — призма наружу (−n_in), иначе один квад."""
    wt = max(float(wall_thickness), 0.0)
    if wt <= 1e-9 or floor_cxy is None:
        return _textured_quad(vertices_4, tile_i, uvs01, flip=flip)
    bl, br, tr, tl = (np.asarray(p, dtype=np.float64) for p in vertices_4)
    mid = 0.25 * (bl[:2] + br[:2] + tr[:2] + tl[:2])
    n_in = _inward_horizontal(mid, floor_cxy)
    return _vertical_wall_slab_textured(
        bl,
        br,
        tr,
        tl,
        tile_i,
        uvs01,
        flip=flip,
        n_in_xy=n_in,
        thickness=wt,
        floor_xy_ring=floor_xy_ring,
        floor_cxy=floor_cxy,
    )


def _outer_vertices_match_vertical_slab(
    vertices_4: List[np.ndarray],
    *,
    floor_cxy: np.ndarray,
    wall_thickness: float,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Наружные углы той же вертикальной призмы, что и у _slab_or_quad_vertical_wall
    (порядок bl, br, tr, tl совпадает с внутренним квадом).
    """
    wt = max(float(wall_thickness), 0.0)
    bl, br, tr, tl = (np.asarray(p, dtype=np.float64) for p in vertices_4)
    if wt <= 1e-9:
        return (bl.copy(), br.copy(), tr.copy(), tl.copy())
    fc = np.asarray(floor_cxy, dtype=np.float64)
    mid = 0.25 * (bl[:2] + br[:2] + tr[:2] + tl[:2])
    n_in = _inward_horizontal(mid, fc)
    n2 = np.asarray(n_in[:2], dtype=np.float64)
    ln = float(np.linalg.norm(n2))
    if ln < 1e-9:
        n2 = np.array([1.0, 0.0], dtype=np.float64)
    else:
        n2 = n2 / ln
    n3 = np.array([float(n2[0]), float(n2[1]), 0.0], dtype=np.float64)
    uni_off = -n3 * wt
    bi = [bl, br, tr, tl]
    bo: List[np.ndarray] = []
    for p in bi:
        off_xy: Optional[np.ndarray] = None
        if floor_xy_ring is not None:
            off_xy = _miter_outward_offset_xy(p[:2], floor_xy_ring, fc, wt)
        if off_xy is None:
            off_xy = uni_off[:2]
        bo.append(
            p
            + np.array(
                [float(off_xy[0]), float(off_xy[1]), 0.0],
                dtype=np.float64,
            )
        )
    return (bo[0], bo[1], bo[2], bo[3])


def _proc_wall_texture(size: int, base_rgb: Tuple[int, int, int]) -> Image.Image:
    rng = np.random.default_rng(91)
    s = max(size, 64)
    b = np.ones((s, s, 3), dtype=np.float32) * np.array(base_rgb, dtype=np.float32)
    b += rng.normal(0, 4.0, (s, s, 3)).astype(np.float32)
    return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), mode="RGB")


def make_balcony_atlas(
    *,
    tile: int = 256,
    wall_lower_path: Path | str | None = None,
    wall_upper_path: Path | str | None = None,
    frame_path: Path | str | None = None,
    glass_path: Path | str | None = None,
    side_basket_path: Path | str | None = None,
    side_jamb_path: Path | str | None = None,
    side_separator_path: Path | str | None = None,
    wall_lower_color: Tuple[int, int, int] | None = None,
    wall_upper_color: Tuple[int, int, int] | None = None,
    frame_color: Tuple[int, int, int] | None = None,
    glass_color: Tuple[int, int, int] | None = None,
    side_basket_color: Tuple[int, int, int] | None = None,
    side_jamb_color: Tuple[int, int, int] | None = None,
    side_separator_color: Tuple[int, int, int] | None = None,
) -> Image.Image:
    """Атлас BALCONY_ATLAS_NUM_TILES×1: низ | верх | рама | стекло | бок корзина | бок у окна | перегородка."""
    t = max(tile, 64)
    wl = (
        _open_rgb(Path(wall_lower_path)).resize((t, t), _resample())
        if wall_lower_path and Path(wall_lower_path).expanduser().resolve().is_file()
        else _proc_wall_texture(t, (168, 158, 148))
    )
    wl = apply_texture_color_tint(wl, wall_lower_color)
    wu = (
        _open_rgb(Path(wall_upper_path)).resize((t, t), _resample())
        if wall_upper_path and Path(wall_upper_path).expanduser().resolve().is_file()
        else _proc_wall_texture(t, (210, 205, 198))
    )
    wu = apply_texture_color_tint(wu, wall_upper_color)
    fr = (
        _open_rgb(Path(frame_path)).resize((t, t), _resample())
        if frame_path and Path(frame_path).expanduser().resolve().is_file()
        else make_window_frame_texture(t)
    )
    fr = apply_texture_color_tint(fr, frame_color)
    gl = (
        _open_rgb(Path(glass_path)).resize((t, t), _resample())
        if glass_path and Path(glass_path).expanduser().resolve().is_file()
        else make_window_glass_texture(t)
    )
    gl = apply_texture_color_tint(gl, glass_color)
    sb = (
        _open_rgb(Path(side_basket_path)).resize((t, t), _resample())
        if side_basket_path and Path(side_basket_path).expanduser().resolve().is_file()
        else wl.copy()
    )
    sb = apply_texture_color_tint(sb, side_basket_color)
    sj = (
        _open_rgb(Path(side_jamb_path)).resize((t, t), _resample())
        if side_jamb_path and Path(side_jamb_path).expanduser().resolve().is_file()
        else wl.copy()
    )
    sj = apply_texture_color_tint(sj, side_jamb_color)
    sep = (
        _open_rgb(Path(side_separator_path)).resize((t, t), _resample())
        if side_separator_path and Path(side_separator_path).expanduser().resolve().is_file()
        else _proc_wall_texture(t, (140, 136, 130))
    )
    sep = apply_texture_color_tint(sep, side_separator_color)
    n = BALCONY_ATLAS_NUM_TILES
    out = Image.new("RGB", (t * n, t))
    out.paste(wl, (0, 0))
    out.paste(wu, (t, 0))
    out.paste(fr, (2 * t, 0))
    out.paste(gl, (3 * t, 0))
    out.paste(sb, (4 * t, 0))
    out.paste(sj, (5 * t, 0))
    out.paste(sep, (6 * t, 0))
    return out


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im


def _scale_uv_to_tile(uv: np.ndarray, tile_index: int) -> np.ndarray:
    """tile_index в пределах атласа -> u в [i/N, (i+1)/N)."""
    n = float(BALCONY_ATLAS_NUM_TILES)
    w = 1.0 / n
    out = np.asarray(uv, dtype=np.float64).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0) * w + w * float(tile_index)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


def _textured_quad_right_parapet_no_diag_artifact(
    BR_b: np.ndarray,
    FR_b: np.ndarray,
    FR_zp: np.ndarray,
    BR_zp: np.ndarray,
    tile_i: int,
    *,
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> trimesh.Trimesh:
    """
    Один квад от низа плиты до Zp (без горизонтального разреза у z=0).
    Обход FR_b→FR_zp→BR_zp→BR_b, чтобы внутренняя диагональ была FR_b–BR_zp, а не BR_b–FR_zp.
    """
    u01 = _uv_planar_quad_bl_br_tr_tl(BR_b, FR_b, FR_zp, BR_zp)
    uv = np.stack([u01[1], u01[2], u01[3], u01[0]])
    return _slab_or_quad_vertical_wall(
        [FR_b, FR_zp, BR_zp, BR_b],
        tile_i,
        uv,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wall_thickness,
        floor_xy_ring=floor_xy_ring,
    )


def _textured_quad_left_parapet_no_diag_artifact(
    BL_b: np.ndarray,
    FL_b: np.ndarray,
    FL_zp: np.ndarray,
    BL_zp: np.ndarray,
    tile_i: int,
    *,
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> trimesh.Trimesh:
    """Левая боковина: зеркально правой, один квад от низа плиты до Zp."""
    u01 = _uv_planar_quad_bl_br_tr_tl(BL_b, FL_b, FL_zp, BL_zp)
    uv = np.stack([u01[1], u01[2], u01[3], u01[0]])
    return _slab_or_quad_vertical_wall(
        [FL_b, FL_zp, BL_zp, BL_b],
        tile_i,
        uv,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wall_thickness,
        floor_xy_ring=floor_xy_ring,
    )


def _side_parapet_left_meshes(
    BL_b: np.ndarray,
    FL_b: np.ndarray,
    BL_zp: np.ndarray,
    FL_zp: np.ndarray,
    floor_cxy: np.ndarray,
    *,
    split_frac: float,
    separator_depth: float,
    window_on_side: bool,
    window_mode: str,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Левый нижний парапет: полоса у фронтового угла (у окна) | корзина | тонкая грань-перегородка внутрь.
    Если на этой стороне окно (не mode=none) — один квад wall_lower, без разбиения.
    """
    wt = max(float(wall_thickness), 0.0)
    if window_on_side and window_mode != "none":
        return [
            (
                "wall_lower",
                _textured_quad_left_parapet_no_diag_artifact(
                    BL_b,
                    FL_b,
                    FL_zp,
                    BL_zp,
                    BALCONY_TILE_WALL_LOWER,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        ]
    if float(split_frac) <= 1e-6:
        return [
            (
                "wall_lower",
                _textured_quad_left_parapet_no_diag_artifact(
                    BL_b,
                    FL_b,
                    FL_zp,
                    BL_zp,
                    BALCONY_TILE_WALL_LOWER,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        ]
    a = float(np.clip(split_frac, 0.04, 0.5))
    S_b = FL_b + a * (BL_b - FL_b)
    S_zp = FL_zp + a * (BL_zp - FL_zp)
    uv_j = _uv_planar_quad_bl_br_tr_tl(FL_b, S_b, S_zp, FL_zp)
    jamb = _slab_or_quad_vertical_wall(
        [FL_b, S_b, S_zp, FL_zp],
        BALCONY_TILE_SIDE_JAMB,
        uv_j,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wt,
        floor_xy_ring=floor_xy_ring,
    )
    uv_b = _uv_planar_quad_bl_br_tr_tl(S_b, BL_b, BL_zp, S_zp)
    basket = _slab_or_quad_vertical_wall(
        [S_b, BL_b, BL_zp, S_zp],
        BALCONY_TILE_SIDE_BASKET,
        uv_b,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wt,
        floor_xy_ring=floor_xy_ring,
    )
    mid = 0.5 * (S_b + S_zp)
    inward = _inward_horizontal(mid[:2], floor_cxy)
    in3 = np.array([float(inward[0]), float(inward[1]), 0.0], dtype=np.float64)
    ln = float(np.linalg.norm(in3))
    if ln < 1e-9:
        in3 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        in3 = in3 / ln
    sd = max(float(separator_depth), 0.006)
    S_bi = S_b + in3 * sd
    S_zpi = S_zp + in3 * sd
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    sep = _textured_quad([S_b, S_zp, S_zpi, S_bi], BALCONY_TILE_SIDE_SEPARATOR, unit, flip=False)
    return [
        ("side_lower_jamb", jamb),
        ("side_lower_basket", basket),
        ("side_separator", sep),
    ]


def _side_parapet_right_meshes(
    BR_b: np.ndarray,
    FR_b: np.ndarray,
    BR_zp: np.ndarray,
    FR_zp: np.ndarray,
    floor_cxy: np.ndarray,
    *,
    split_frac: float,
    separator_depth: float,
    window_on_side: bool,
    window_mode: str,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Правый нижний парапет — зеркально левому."""
    wt = max(float(wall_thickness), 0.0)
    if window_on_side and window_mode != "none":
        return [
            (
                "wall_lower",
                _textured_quad_right_parapet_no_diag_artifact(
                    BR_b,
                    FR_b,
                    FR_zp,
                    BR_zp,
                    BALCONY_TILE_WALL_LOWER,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        ]
    if float(split_frac) <= 1e-6:
        return [
            (
                "wall_lower",
                _textured_quad_right_parapet_no_diag_artifact(
                    BR_b,
                    FR_b,
                    FR_zp,
                    BR_zp,
                    BALCONY_TILE_WALL_LOWER,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        ]
    a = float(np.clip(split_frac, 0.04, 0.5))
    S_b = FR_b + a * (BR_b - FR_b)
    S_zp = FR_zp + a * (BR_zp - FR_zp)
    uv_j = _uv_planar_quad_bl_br_tr_tl(FR_b, S_b, S_zp, FR_zp)
    jamb = _slab_or_quad_vertical_wall(
        [FR_b, S_b, S_zp, FR_zp],
        BALCONY_TILE_SIDE_JAMB,
        uv_j,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wt,
        floor_xy_ring=floor_xy_ring,
    )
    uv_b = _uv_planar_quad_bl_br_tr_tl(S_b, BR_b, BR_zp, S_zp)
    basket = _slab_or_quad_vertical_wall(
        [S_b, BR_b, BR_zp, S_zp],
        BALCONY_TILE_SIDE_BASKET,
        uv_b,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wt,
        floor_xy_ring=floor_xy_ring,
    )
    mid = 0.5 * (S_b + S_zp)
    inward = _inward_horizontal(mid[:2], floor_cxy)
    in3 = np.array([float(inward[0]), float(inward[1]), 0.0], dtype=np.float64)
    ln = float(np.linalg.norm(in3))
    if ln < 1e-9:
        in3 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        in3 = in3 / ln
    sd = max(float(separator_depth), 0.006)
    S_bi = S_b + in3 * sd
    S_zpi = S_zp + in3 * sd
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    sep = _textured_quad([S_b, S_zp, S_zpi, S_bi], BALCONY_TILE_SIDE_SEPARATOR, unit, flip=False)
    return [
        ("side_lower_jamb", jamb),
        ("side_lower_basket", basket),
        ("side_separator", sep),
    ]


def _textured_quad(
    vertices_4: List[np.ndarray],
    tile_i: int,
    uvs01: np.ndarray,
    *,
    flip: bool = False,
) -> trimesh.Trimesh:
    """Четыре вершины + UV в [0,1]², раскладка в колонку атласа tile_i. flip — инверсия обхода."""
    v = np.stack([np.asarray(p, dtype=np.float64) for p in vertices_4])
    uv = _scale_uv_to_tile(np.asarray(uvs01, dtype=np.float64), tile_i)
    if flip:
        f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    else:
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False, validate=False)
    m.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    return m


def _planar_wall_uv_basis(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    *,
    repeat_m: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Опорная точка bl и орты (u_hat, v_hat), масштаб rm — как у парапетных граней."""
    bl, br, tr, tl = (np.asarray(x, dtype=np.float64).copy() for x in (bl, br, tr, tl))
    e_bot = br - bl
    lb = float(np.linalg.norm(e_bot))
    if lb < 1e-9:
        lb = 1.0
        u_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u_hat = e_bot / lb
    n = np.cross(br - bl, tl - bl)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / ln
    v_raw = tl - bl
    v_hat = v_raw - float(np.dot(v_raw, u_hat)) * u_hat
    lv = float(np.linalg.norm(v_hat))
    if lv < 1e-9:
        v_hat = np.cross(n, u_hat)
        lv = float(np.linalg.norm(v_hat))
    v_hat = v_hat / max(lv, 1e-9)
    rm = max(float(repeat_m), 0.08)
    return bl, u_hat, v_hat, rm


def _planar_wall_uv01_at(
    bl: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    rm: float,
    p: np.ndarray,
) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64) - bl
    uu = float(np.dot(q, u_hat)) / rm
    vv = float(np.dot(q, v_hat)) / rm
    uu -= np.floor(uu)
    vv -= np.floor(vv)
    return np.array([uu, vv], dtype=np.float64)


def _uv_planar_quad_bl_br_tr_tl(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    *,
    repeat_m: float = 0.55,
) -> np.ndarray:
    """
    UV в плоскости грани: нижняя кромка bl→br, верх tr/tl. Повтор текстуры каждые repeat_m м
    (иначе на длинных наклонных гранях тайл растягивается в один пиксель атласа).
    """
    bl0, u_hat, v_hat, rm = _planar_wall_uv_basis(bl, br, tr, tl, repeat_m=repeat_m)
    b, r, t1, t0 = (np.asarray(x, dtype=np.float64) for x in (bl, br, tr, tl))

    def uv1(p: np.ndarray) -> np.ndarray:
        return _planar_wall_uv01_at(bl0, u_hat, v_hat, rm, p)

    return np.stack([uv1(b), uv1(r), uv1(t1), uv1(t0)])


def _double_sided_copy_uv(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Дублирует грани с обратным обходом (двусторонность), UV копируется — нужно для стекла и OpenGL culling."""
    if m is None or len(m.faces) == 0:
        return m
    uv = np.asarray(m.visual.uv, dtype=np.float64)
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    v2 = np.vstack([v, v])
    uv2 = np.vstack([uv, uv])
    f_rev = f[:, ::-1].copy() + len(v)
    f2 = np.vstack([f, f_rev])
    out = trimesh.Trimesh(vertices=v2, faces=f2, process=False, validate=False)
    out.visual = trimesh.visual.texture.TextureVisuals(uv=uv2)
    return out


def _mesh_has_per_vertex_uv(m: trimesh.Trimesh) -> bool:
    if m.visual is None:
        return False
    uv = getattr(m.visual, "uv", None)
    if uv is None:
        return False
    return int(len(np.asarray(uv))) == int(len(m.vertices))


def _floor_y_span(BL: np.ndarray, FL: np.ndarray, FR: np.ndarray, BR: np.ndarray) -> float:
    ys = [float(BL[1]), float(FL[1]), float(FR[1]), float(BR[1])]
    return max(max(ys) - min(ys), 1e-6)


def _resolve_floor_quad(
    width_back: float,
    width_front: float,
    depth: float,
    corner_left_wall: Optional[Tuple[float, float]],
    corner_right_wall: Optional[Tuple[float, float]],
    corner_front_left: Optional[Tuple[float, float]],
    corner_front_right: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Четыре угла основания z=0 (против часовой в порядке: зад слева, зад справа, перед справа, перед слева):
    левый_у_стены, правый_у_стены, передний_левый, передний_правый.
    Если хоть один угол None — старая схема width_back × width_front × depth.
    """
    if all(
        c is not None
        for c in (corner_left_wall, corner_right_wall, corner_front_left, corner_front_right)
    ):
        BL = np.array([corner_left_wall[0], corner_left_wall[1], 0.0], dtype=np.float64)
        BR = np.array([corner_right_wall[0], corner_right_wall[1], 0.0], dtype=np.float64)
        FL = np.array([corner_front_left[0], corner_front_left[1], 0.0], dtype=np.float64)
        FR = np.array([corner_front_right[0], corner_front_right[1], 0.0], dtype=np.float64)
        return BL, BR, FL, FR
    Wb = max(width_back, 0.05)
    Wf = max(width_front, 0.05)
    Dm = max(depth, 0.05)
    BL = np.array([-Wb / 2, 0.0, 0.0], dtype=np.float64)
    BR = np.array([Wb / 2, 0.0, 0.0], dtype=np.float64)
    FL = np.array([-Wf / 2, Dm, 0.0], dtype=np.float64)
    FR = np.array([Wf / 2, Dm, 0.0], dtype=np.float64)
    return BL, BR, FL, FR


def _compute_top_ring(
    BL: np.ndarray,
    BR: np.ndarray,
    FL: np.ndarray,
    FR: np.ndarray,
    H: float,
    vertical_prism: bool,
    tilt_left_deg: float,
    tilt_right_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Верхнее кольцо: вертикальная призма (+Z) или старый наклон tilt_* по Y."""
    if vertical_prism:
        ez = np.array([0.0, 0.0, H], dtype=np.float64)
        return BL + ez, BR + ez, FR + ez, FL + ez
    tl = math.tan(math.radians(tilt_left_deg))
    tr_ = math.tan(math.radians(tilt_right_deg))
    TBL = BL + np.array([0.0, tl * H, H], dtype=np.float64)
    TBR = BR + np.array([0.0, tr_ * H, H], dtype=np.float64)
    TFR = FR + np.array([0.0, tr_ * H, H], dtype=np.float64)
    TFL = FL + np.array([0.0, tl * H, H], dtype=np.float64)
    return TBL, TBR, TFR, TFL


def _floor_center_xy(BL: np.ndarray, BR: np.ndarray, FL: np.ndarray, FR: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(BL[0] + BR[0] + FL[0] + FR[0]) * 0.25,
            float(BL[1] + BR[1] + FL[1] + FR[1]) * 0.25,
        ],
        dtype=np.float64,
    )


def _inward_horizontal(wall_mid_xy: np.ndarray, floor_center_xy: np.ndarray) -> np.ndarray:
    d = floor_center_xy - wall_mid_xy
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d = d / L
    return np.array([d[0], d[1], 0.0], dtype=np.float64)


def _back_face_us_to_point(
    p_bl: np.ndarray,
    p_br: np.ndarray,
    p_tr: np.ndarray,
    p_tl: np.ndarray,
    u: float,
    s: float,
) -> np.ndarray:
    """u,s в [0,1]: низ bl→br, верх tl→tr (как четырёхугольник задней грани)."""
    pb = (1.0 - u) * p_bl + u * p_br
    pt = (1.0 - u) * p_tl + u * p_tr
    return (1.0 - s) * pb + s * pt


def _split_axis_aligned_rect(
    r: Tuple[float, float, float, float],
    h: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """r,h как (u0,u1,s0,s1). Возвращает фрагменты r без пересечения с h."""
    u0, u1, s0, s1 = r
    hu0, hu1, hs0, hs1 = h
    du0, du1 = max(u0, hu0), min(u1, hu1)
    ds0, ds1 = max(s0, hs0), min(s1, hs1)
    if du1 <= du0 + 1e-9 or ds1 <= ds0 + 1e-9:
        return [r]
    pieces: List[Tuple[float, float, float, float]] = []
    if u0 < du0 - 1e-9:
        pieces.append((u0, du0, s0, s1))
    if du1 < u1 - 1e-9:
        pieces.append((du1, u1, s0, s1))
    if s0 < ds0 - 1e-9:
        pieces.append((du0, du1, s0, ds0))
    if ds1 < s1 - 1e-9:
        pieces.append((du0, du1, ds1, s1))
    return pieces


def _rects_subtract_axis_hole(
    rects: List[Tuple[float, float, float, float]],
    h: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for r in rects:
        out.extend(_split_axis_aligned_rect(r, h))
    return out


def _inner_holes_us_in_z_span(
    opening_specs: List[dict],
    z_lo: float,
    z_hi: float,
) -> List[Tuple[float, float, float, float]]:
    """Проёмы (окна + двери) на задней стене: u0,u1, z_bottom, z_top."""
    span = max(float(z_hi) - float(z_lo), 1e-9)
    holes: List[Tuple[float, float, float, float]] = []
    for spec in opening_specs:
        u0 = float(np.clip(spec["u0"], 0.0, 1.0))
        u1 = float(np.clip(spec["u1"], 0.0, 1.0))
        if u1 <= u0 + 1e-4:
            continue
        za = max(float(spec["z_bottom"]), float(z_lo))
        zb = min(float(spec["z_top"]), float(z_hi))
        if zb <= za + 1e-4:
            continue
        s0 = (za - float(z_lo)) / span
        s1 = (zb - float(z_lo)) / span
        holes.append((u0, u1, s0, s1))
    return holes


def _perforated_back_wall_patches(
    p_bl: np.ndarray,
    p_br: np.ndarray,
    p_tr: np.ndarray,
    p_tl: np.ndarray,
    holes_us: List[Tuple[float, float, float, float]],
    uv_for_point: Any,
    *,
    flip: bool,
    tile_i: int,
    part_name: str,
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    rects: List[Tuple[float, float, float, float]] = [(0.0, 1.0, 0.0, 1.0)]
    for h in holes_us:
        rects = _rects_subtract_axis_hole(rects, h)
    out: List[Tuple[str, trimesh.Trimesh]] = []
    for ua, ub, sa, sb in rects:
        if ub <= ua + 1e-6 or sb <= sa + 1e-6:
            continue
        bl = _back_face_us_to_point(p_bl, p_br, p_tr, p_tl, ua, sa)
        br = _back_face_us_to_point(p_bl, p_br, p_tr, p_tl, ub, sa)
        tr = _back_face_us_to_point(p_bl, p_br, p_tr, p_tl, ub, sb)
        tl = _back_face_us_to_point(p_bl, p_br, p_tr, p_tl, ua, sb)
        if uv_for_point is not None:
            uv = np.stack([uv_for_point(bl), uv_for_point(br), uv_for_point(tr), uv_for_point(tl)])
        else:
            uv = _uv_planar_quad_bl_br_tr_tl(bl, br, tr, tl)
        wt = max(float(wall_thickness), 0.0)
        if wt > 1e-9 and floor_cxy is not None:
            mid = 0.25 * (bl[:2] + br[:2] + tr[:2] + tl[:2])
            n_in = _inward_horizontal(mid, floor_cxy)
            mesh = _vertical_wall_slab_textured(
                bl,
                br,
                tr,
                tl,
                tile_i,
                uv,
                flip=flip,
                n_in_xy=n_in,
                thickness=wt,
                floor_xy_ring=floor_xy_ring,
                floor_cxy=floor_cxy,
            )
        else:
            mesh = _textured_quad([bl, br, tr, tl], tile_i, uv, flip=flip)
        out.append((part_name, mesh))
    return out


def _normalize_inner_wall_windows(
    raw: Any,
    *,
    default_mode: str,
    default_depth: float,
    default_kind: str,
    default_mv: int,
    default_mh: int,
    default_ox: float,
    default_oz: float,
    default_partial: List[Tuple[int, float]],
) -> List[dict]:
    if not raw or not isinstance(raw, list):
        return []
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            u0, u1 = float(item["u0"]), float(item["u1"])
            zb, zt = float(item["z_bottom"]), float(item["z_top"])
        except (KeyError, TypeError, ValueError):
            continue
        if u1 <= u0 + 1e-6 or zt <= zb + 1e-6:
            continue
        wm = str(item.get("window_mode", default_mode)).lower().strip()
        if wm not in ("with_glass", "frame_only", "none"):
            wm = str(default_mode).lower().strip()
        out.append(
            {
                "u0": u0,
                "u1": u1,
                "z_bottom": zb,
                "z_top": zt,
                "window_mode": wm,
                "window_depth": float(item.get("window_depth", default_depth)),
                "window_kind": str(item.get("window_kind", default_kind)),
                "mullions_vertical": _pick_nonneg_int(item.get("mullions_vertical", default_mv), 0),
                "mullions_horizontal": _pick_nonneg_int(item.get("mullions_horizontal", default_mh), 0),
                "mullion_offset_x": float(item.get("mullion_offset_x", default_ox)),
                "mullion_offset_z": float(item.get("mullion_offset_z", default_oz)),
                "partial_horizontal_bars": _normalize_partial_horizontal_bars(
                    item.get("partial_horizontal_bars", default_partial)
                ),
            }
        )
    return out


def _normalize_inner_wall_doors(raw: Any) -> List[dict]:
    """Двери во внутренней стене: u0,u1, z_bottom, z_top; style french | slab."""
    if not raw or not isinstance(raw, list):
        return []
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            u0, u1 = float(item["u0"]), float(item["u1"])
            zb, zt = float(item["z_bottom"]), float(item["z_top"])
        except (KeyError, TypeError, ValueError):
            continue
        if u1 <= u0 + 1e-6 or zt <= zb + 1e-6:
            continue
        style = str(item.get("style", "french")).lower().strip()
        if style not in ("french", "slab"):
            style = "french"
        out.append(
            {
                "u0": u0,
                "u1": u1,
                "z_bottom": zb,
                "z_top": zt,
                "style": style,
                "frame_width": max(float(item.get("frame_width", 0.09)), 0.025),
                "frame_depth": max(float(item.get("frame_depth", 0.06)), 0.02),
                "leaf_gap": max(float(item.get("leaf_gap", 0.025)), 0.012),
                "midrail_z_frac": float(np.clip(float(item.get("midrail_z_frac", 0.58)), 0.2, 0.82)),
                "y_outer": max(float(item.get("y_outer", 0.05)), 0.02),
            }
        )
    return out


def _inner_back_edge_points_at_z(
    BL_b: np.ndarray,
    BR_b: np.ndarray,
    TBL: np.ndarray,
    TBR: np.ndarray,
    u0: float,
    u1: float,
    zq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    zb = float(BL_b[2])
    zt = float(TBL[2])
    span = max(zt - zb, 1e-9)
    s = (float(zq) - zb) / span
    s = float(np.clip(s, 0.0, 1.0))

    def p_at(u: float) -> np.ndarray:
        pb = (1.0 - u) * BL_b + u * BR_b
        pt = (1.0 - u) * TBL + u * TBR
        return (1.0 - s) * pb + s * pt

    return p_at(u0), p_at(u1)


def _append_inner_back_wall_window_meshes(
    window_parts: List[Tuple[str, trimesh.Trimesh]],
    inner_wall_windows: List[dict],
    BL_b: np.ndarray,
    BR_b: np.ndarray,
    TBL: np.ndarray,
    TBR: np.ndarray,
    BL: np.ndarray,
    BR: np.ndarray,
    floor_cxy: np.ndarray,
) -> None:
    mid_back_xy = 0.5 * (np.asarray(BL, dtype=np.float64)[:2] + np.asarray(BR, dtype=np.float64)[:2])
    inward_xy = _inward_horizontal(mid_back_xy, floor_cxy)
    for spec in inner_wall_windows:
        if spec["window_mode"] == "none":
            continue
        z0, z1 = float(spec["z_bottom"]), float(spec["z_top"])
        u0, u1 = float(spec["u0"]), float(spec["u1"])
        bb, bf = _inner_back_edge_points_at_z(BL_b, BR_b, TBL, TBR, u0, u1, z0)
        window_parts.extend(
            _window_parts_on_wall_edge(
                mode=spec["window_mode"],
                p_back_zp=bb,
                p_front_zp=bf,
                z_bottom=z0,
                z_top=z1,
                window_depth=float(spec["window_depth"]),
                window_kind=str(spec["window_kind"]),
                mullions_vertical=int(spec["mullions_vertical"]),
                mullions_horizontal=int(spec["mullions_horizontal"]),
                mullion_offset_x=float(spec["mullion_offset_x"]),
                mullion_offset_z=float(spec["mullion_offset_z"]),
                partial_horizontal_bars=list(spec["partial_horizontal_bars"]),
                inward_xy=inward_xy,
            )
        )


def _append_inner_back_wall_door_meshes(
    window_parts: List[Tuple[str, trimesh.Trimesh]],
    inner_wall_doors: List[dict],
    BL_b: np.ndarray,
    BR_b: np.ndarray,
    TBL: np.ndarray,
    TBR: np.ndarray,
    BL: np.ndarray,
    BR: np.ndarray,
    floor_cxy: np.ndarray,
) -> None:
    """Процедурная дверь (procedural_door) в плоскости внутренней стены; ось глубины двери — −n (к фасаду)."""
    mid_back_xy = 0.5 * (np.asarray(BL, dtype=np.float64)[:2] + np.asarray(BR, dtype=np.float64)[:2])
    inward_xy = _inward_horizontal(mid_back_xy, floor_cxy)
    for di, spec in enumerate(inner_wall_doors):
        z0, z1 = float(spec["z_bottom"]), float(spec["z_top"])
        u0, u1 = float(spec["u0"]), float(spec["u1"])
        bb, bf = _inner_back_edge_points_at_z(BL_b, BR_b, TBL, TBR, u0, u1, z0)
        u_raw = bf - bb
        win_w = float(np.linalg.norm(u_raw))
        win_h = float(z1 - z0)
        if win_w < 0.14 or win_h < 0.45:
            continue
        u_hat = u_raw / win_w
        v_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n_hat = np.cross(u_hat, v_hat)
        ln = float(np.linalg.norm(n_hat))
        if ln < 1e-12:
            continue
        n_hat = n_hat / ln
        inward3 = np.asarray(inward_xy, dtype=np.float64)
        if float(np.dot(n_hat[:2], inward3[:2])) < 0.0:
            n_hat = -n_hat
        mid = 0.5 * (bb + bf)
        center = np.array([float(mid[0]), float(mid[1]), 0.5 * (z0 + z1)], dtype=np.float64)
        R = np.column_stack([u_hat, -n_hat, v_hat])
        hw = 0.5 * win_w
        hh = 0.5 * win_h
        y_outer = float(spec["y_outer"])
        fd = float(spec["frame_depth"])
        if spec["style"] == "slab":
            parts = build_simple_door_slab(
                x0=-hw,
                x1=hw,
                z0=-hh,
                z1=hh,
                y_outer=y_outer,
                depth=max(fd, 0.05),
                niche_depth=None,
            )
        else:
            parts = build_french_double_door_parts(
                x0=-hw,
                x1=hw,
                z0=-hh,
                z1=hh,
                y_outer=y_outer,
                frame_width=float(spec["frame_width"]),
                frame_depth=fd,
                leaf_gap=float(spec["leaf_gap"]),
                midrail_z_frac=float(spec["midrail_z_frac"]),
                niche_depth=None,
            )
        for tag, m in parts:
            if m is None or len(m.vertices) == 0:
                continue
            vv = np.asarray(m.vertices, dtype=np.float64)
            vv = (R @ vv.T).T + center
            m2 = trimesh.Trimesh(vertices=vv, faces=m.faces, process=False, validate=False)
            window_parts.append((f"{tag}_{di}", m2))


def _parse_inner_wall_window_cli(s: str) -> dict:
    """u0,u1,z_bottom,z_top[,mv=2,mh=0,mode=frame_only,depth=0.12,kind=fixed,ox=0,oz=0]"""
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if len(toks) < 4:
        raise argparse.ArgumentTypeError(
            "inner-wall-window: нужно u0,u1,z_bottom,z_top и опционально key=value через запятую"
        )
    try:
        d: dict[str, Any] = {
            "u0": float(toks[0]),
            "u1": float(toks[1]),
            "z_bottom": float(toks[2]),
            "z_top": float(toks[3]),
        }
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    for t in toks[4:]:
        if "=" not in t:
            raise argparse.ArgumentTypeError(f"ожидалось key=value, получено {t!r}")
        k, v = t.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k in ("mv", "mullions_vertical"):
            d["mullions_vertical"] = int(v)
        elif k in ("mh", "mullions_horizontal"):
            d["mullions_horizontal"] = int(v)
        elif k == "mode":
            d["window_mode"] = v
        elif k == "depth":
            d["window_depth"] = float(v)
        elif k == "kind":
            d["window_kind"] = v
        elif k == "ox":
            d["mullion_offset_x"] = float(v)
        elif k == "oz":
            d["mullion_offset_z"] = float(v)
        else:
            raise argparse.ArgumentTypeError(f"неизвестный ключ: {k}")
    return d


def _parse_inner_wall_door_cli(s: str) -> dict:
    """u0,u1,z_bottom,z_top[,style=french|slab,fw=,fd=,gap=,mid=,y0=]"""
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if len(toks) < 4:
        raise argparse.ArgumentTypeError(
            "inner-wall-door: нужно u0,u1,z_bottom,z_top и опционально key=value через запятую"
        )
    try:
        d: dict[str, Any] = {
            "u0": float(toks[0]),
            "u1": float(toks[1]),
            "z_bottom": float(toks[2]),
            "z_top": float(toks[3]),
        }
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    for t in toks[4:]:
        if "=" not in t:
            raise argparse.ArgumentTypeError(f"ожидалось key=value, получено {t!r}")
        k, v = t.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "style":
            d["style"] = v
        elif k in ("fw", "frame_width"):
            d["frame_width"] = float(v)
        elif k in ("fd", "frame_depth"):
            d["frame_depth"] = float(v)
        elif k in ("gap", "leaf_gap"):
            d["leaf_gap"] = float(v)
        elif k in ("mid", "midrail_z_frac"):
            d["midrail_z_frac"] = float(v)
        elif k in ("y0", "y_outer"):
            d["y_outer"] = float(v)
        else:
            raise argparse.ArgumentTypeError(f"неизвестный ключ: {k}")
    return d


def _uv_front_vertical_wall_quad(
    p_bl: np.ndarray,
    p_br: np.ndarray,
    p_tr: np.ndarray,
    p_tl: np.ndarray,
) -> np.ndarray:
    """Вертикальная передняя грань: u вдоль p_bl→p_br, v от нижней z к верхней (0..1)."""
    p_bl = np.asarray(p_bl, dtype=np.float64)
    p_br = np.asarray(p_br, dtype=np.float64)
    e = p_br - p_bl
    L = float(np.linalg.norm(e))
    if L < 1e-9:
        e_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        L = 1.0
    else:
        e_hat = e / L
    zb = float(p_bl[2])
    zt = float(p_tl[2])
    zspan = max(zt - zb, 1e-6)

    def one(p: np.ndarray) -> np.ndarray:
        pa = np.asarray(p, dtype=np.float64)
        uu = float(np.dot(pa - p_bl, e_hat)) / L
        uu = float(np.clip(uu, 0.0, 1.0))
        vv = (float(pa[2]) - zb) / zspan
        vv = float(np.clip(vv, 0.0, 1.0))
        return np.array([uu, vv], dtype=np.float64)

    return np.stack([one(p_bl), one(p_br), one(p_tr), one(p_tl)])


def _front_parapet_one_quad(
    FL: np.ndarray,
    FR: np.ndarray,
    FL_zp: np.ndarray,
    FR_zp: np.ndarray,
    Zp: float,
    *,
    floor_thickness: float,
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Передняя стена под окном + наружный торец пола одним квадом (без отдельной «губы» у z=0)."""
    t = max(float(floor_thickness), 0.02)
    FL_b = np.asarray(FL, dtype=np.float64).copy()
    FL_b[2] -= t
    FR_b = np.asarray(FR, dtype=np.float64).copy()
    FR_b[2] -= t
    uv = _uv_front_vertical_wall_quad(FL_b, FR_b, FR_zp, FL_zp)
    mesh = _slab_or_quad_vertical_wall(
        [FL_b, FR_b, FR_zp, FL_zp],
        BALCONY_TILE_WALL_LOWER,
        uv,
        flip=False,
        floor_cxy=floor_cxy,
        wall_thickness=wall_thickness,
        floor_xy_ring=floor_xy_ring,
    )
    return [("wall_lower", mesh)]


def _window_parts_on_wall_edge(
    *,
    mode: str,
    p_back_zp: np.ndarray,
    p_front_zp: np.ndarray,
    z_bottom: float,
    z_top: float,
    window_depth: float,
    window_kind: str,
    mullions_vertical: int,
    mullions_horizontal: int,
    mullion_offset_x: float,
    mullion_offset_z: float,
    partial_horizontal_bars: List[Tuple[int, float]],
    inward_xy: np.ndarray,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Окно в плоскости стены: нижняя кромка проёма p_back_zp → p_front_zp (на одной высоте z_bottom)."""
    if mode == "none":
        return []
    bb = np.asarray(p_back_zp, dtype=np.float64)
    bf = np.asarray(p_front_zp, dtype=np.float64)
    u_raw = bf - bb
    win_w = float(np.linalg.norm(u_raw))
    win_h = float(z_top - z_bottom)
    if win_w < 0.11 or win_h < 0.05:
        return []
    u_hat = u_raw / win_w
    v_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_hat = np.cross(u_hat, v_hat)
    ln = float(np.linalg.norm(n_hat))
    if ln < 1e-12:
        return []
    n_hat = n_hat / ln
    inward3 = np.asarray(inward_xy, dtype=np.float64)
    if float(np.dot(n_hat[:2], inward3[:2])) < 0.0:
        n_hat = -n_hat
    # n_hat — внутрь балкона (к центру пола в XY). Рама/стекло — снаружи стены: вторая ось локали = −n_hat.
    mid = 0.5 * (bb + bf)
    # z_bottom / z_top — абсолютные Z; bb/bf уже на высоте z_bottom, нельзя снова прибавлять mid[2].
    center = np.array(
        [float(mid[0]), float(mid[1]), float(z_bottom) + 0.5 * win_h],
        dtype=np.float64,
    )
    R = np.column_stack([u_hat, -n_hat, v_hat])
    kind = _pick_kind(window_kind, "fixed")
    ft = _frame_thickness(win_w, win_h)
    glass_t = max(window_depth * 0.12, 0.004)
    partial_bars = _normalize_partial_horizontal_bars(partial_horizontal_bars)
    mf, mg = build_window_frame_glass_meshes(
        width=win_w,
        height=win_h,
        depth=window_depth,
        profile="rect",
        kind=kind,
        mullions_vertical=mullions_vertical,
        mullions_horizontal=mullions_horizontal,
        mullion_offset_x=mullion_offset_x,
        mullion_offset_z=mullion_offset_z,
        partial_horizontal_bars=partial_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=0.0,
        with_glass=(mode == "with_glass"),
        merge_vertices=False,
    )
    wp: List[Tuple[str, trimesh.Trimesh]] = []
    for name, m in (("frame", mf), ("glass", mg)):
        if name == "glass" and mode != "with_glass":
            continue
        if len(m.vertices) == 0:
            continue
        vv = np.asarray(m.vertices, dtype=np.float64)
        vv = (R @ vv.T).T + center
        m2 = trimesh.Trimesh(vertices=vv, faces=m.faces, process=False, validate=False)
        wp.append((name, m2))
    return wp


def _concatenate_uv_meshes(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Склейка без trimesh.concatenate: UV остаётся 1:1 с вершинами."""
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
            raise RuntimeError("concatenate_uv_meshes: submesh without per-vertex uv")
        u = np.asarray(u, dtype=np.float64)
        if len(u) != len(vv):
            raise RuntimeError("concatenate_uv_meshes: uv/vertex count mismatch")
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


def _balcony_floor_textured_parts(
    BL: np.ndarray,
    FL: np.ndarray,
    FR: np.ndarray,
    BR: np.ndarray,
    T: float,
    *,
    FR_zp: Optional[np.ndarray] = None,
    BR_zp: Optional[np.ndarray] = None,
    FL_zp: Optional[np.ndarray] = None,
    BL_zp: Optional[np.ndarray] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Пол: верх и низ; боковые торцы пола не добавляем — их закрывают боковые стены."""
    d_safe = _floor_y_span(BL, FL, FR, BR)
    zt, zb = 0.0, -T
    BL_t = BL + np.array([0.0, 0.0, zt], dtype=np.float64)
    FL_t = FL + np.array([0.0, 0.0, zt], dtype=np.float64)
    FR_t = FR + np.array([0.0, 0.0, zt], dtype=np.float64)
    BR_t = BR + np.array([0.0, 0.0, zt], dtype=np.float64)
    BL_b = BL + np.array([0.0, 0.0, zb], dtype=np.float64)
    FL_b = FL + np.array([0.0, 0.0, zb], dtype=np.float64)
    FR_b = FR + np.array([0.0, 0.0, zb], dtype=np.float64)
    BR_b = BR + np.array([0.0, 0.0, zb], dtype=np.float64)
    xs = np.array([BL[0], FL[0], FR[0], BR[0]], dtype=np.float64)
    xmin_f, xmax_f = float(xs.min()), float(xs.max())
    xspan = max(xmax_f - xmin_f, 1e-6)
    def _uv_floor_top(p: np.ndarray) -> np.ndarray:
        return np.array([(float(p[0]) - xmin_f) / xspan, float(p[1]) / d_safe], dtype=np.float64)

    uv_bl = _uv_floor_top(BL_t)
    uv_fl = _uv_floor_top(FL_t)
    uv_fr = _uv_floor_top(FR_t)
    uv_br = _uv_floor_top(BR_t)
    if BL_zp is not None and FL_zp is not None:
        b0, uh, vh, rm = _planar_wall_uv_basis(BL_b, FL_b, FL_zp, BL_zp)
        uv_bl = _planar_wall_uv01_at(b0, uh, vh, rm, BL_t)
        uv_fl = _planar_wall_uv01_at(b0, uh, vh, rm, FL_t)
    if FR_zp is not None and BR_zp is not None:
        b0, uh, vh, rm = _planar_wall_uv_basis(BR_b, FR_b, FR_zp, BR_zp)
        uv_fr = _planar_wall_uv01_at(b0, uh, vh, rm, FR_t)
        uv_br = _planar_wall_uv01_at(b0, uh, vh, rm, BR_t)
    uv_ft = np.stack([uv_bl, uv_fl, uv_fr, uv_br])
    f_top = _textured_quad([BL_t, FL_t, FR_t, BR_t], BALCONY_TILE_WALL_LOWER, uv_ft, flip=False)
    uv_fb = np.stack([_uv_floor_top(BL_b), _uv_floor_top(BR_b), _uv_floor_top(FR_b), _uv_floor_top(FL_b)])
    f_bot = _textured_quad([BL_b, BR_b, FR_b, FL_b], BALCONY_TILE_WALL_LOWER, uv_fb, flip=True)
    return [("wall_lower", fp) for fp in (f_top, f_bot)]


def _simple_box_back_and_sides(
    BL: np.ndarray,
    BR: np.ndarray,
    FL: np.ndarray,
    FR: np.ndarray,
    TBL: np.ndarray,
    TBR: np.ndarray,
    TFR: np.ndarray,
    TFL: np.ndarray,
    BL_zp: np.ndarray,
    FL_zp: np.ndarray,
    BR_zp: np.ndarray,
    FR_zp: np.ndarray,
    *,
    floor_thickness: float,
    window_left_wall: bool,
    window_right_wall: bool,
    mode: str,
    sumode: str,
    floor_cxy: np.ndarray,
    side_parapet_split_frac: float,
    side_parapet_separator_depth: float,
    inner_wall_windows: List[dict],
    inner_wall_doors: List[dict],
    open_left_above_parapet: bool,
    open_right_above_parapet: bool,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Зад + бока: парапет (прямоугольник) и верх; без передней стены."""
    out: List[Tuple[str, trimesh.Trimesh]] = []
    wt = max(float(wall_thickness), 0.0)
    t = max(float(floor_thickness), 0.02)
    BL_b = np.asarray(BL, dtype=np.float64).copy()
    BL_b[2] -= t
    BR_b = np.asarray(BR, dtype=np.float64).copy()
    BR_b[2] -= t
    FR_b = np.asarray(FR, dtype=np.float64).copy()
    FR_b[2] -= t
    FL_b = np.asarray(FL, dtype=np.float64).copy()
    FL_b[2] -= t
    holes_b = _inner_holes_us_in_z_span(
        list(inner_wall_windows) + list(inner_wall_doors),
        float(BL_b[2]),
        float(TBL[2]),
    )
    out.extend(
        _perforated_back_wall_patches(
            BL_b,
            BR_b,
            TBR,
            TBL,
            holes_b,
            None,
            flip=True,
            tile_i=BALCONY_TILE_WALL_LOWER,
            part_name="wall_lower",
            floor_cxy=floor_cxy,
            wall_thickness=wt,
            floor_xy_ring=floor_xy_ring,
        )
    )
    side_tile = BALCONY_TILE_GLASS if sumode == "glass" else BALCONY_TILE_WALL_UPPER
    side_name = "side_glass" if sumode == "glass" else "wall_upper"
    out.extend(
        _side_parapet_left_meshes(
            BL_b,
            FL_b,
            BL_zp,
            FL_zp,
            floor_cxy,
            split_frac=side_parapet_split_frac,
            separator_depth=side_parapet_separator_depth,
            window_on_side=window_left_wall,
            window_mode=mode,
            wall_thickness=wt,
            floor_xy_ring=floor_xy_ring,
        )
    )
    # open_side_right → грань BL—FL (левый борт в -X); open_side_left → BR—FR (+X) — как в CLI-именах у пользователя.
    if ((not window_left_wall) or mode == "none") and not open_right_above_parapet:
        uv_sl = _uv_planar_quad_bl_br_tr_tl(BL_zp, FL_zp, TFL, TBL)
        out.append(
            (
                side_name,
                _slab_or_quad_vertical_wall(
                    [BL_zp, FL_zp, TFL, TBL],
                    side_tile,
                    uv_sl,
                    flip=False,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        )
    out.extend(
        _side_parapet_right_meshes(
            BR_b,
            FR_b,
            BR_zp,
            FR_zp,
            floor_cxy,
            split_frac=side_parapet_split_frac,
            separator_depth=side_parapet_separator_depth,
            window_on_side=window_right_wall,
            window_mode=mode,
            wall_thickness=wt,
            floor_xy_ring=floor_xy_ring,
        )
    )
    if ((not window_right_wall) or mode == "none") and not open_left_above_parapet:
        uv_sr = _uv_planar_quad_bl_br_tr_tl(BR_zp, FR_zp, TFR, TBR)
        out.append(
            (
                side_name,
                _slab_or_quad_vertical_wall(
                    [BR_zp, FR_zp, TFR, TBR],
                    side_tile,
                    uv_sr,
                    flip=False,
                    floor_cxy=floor_cxy,
                    wall_thickness=wt,
                    floor_xy_ring=floor_xy_ring,
                ),
            )
        )
    return out


def _balcony_procedural_window_parts(
    *,
    mode: str,
    FL_zp: np.ndarray,
    FR_zp: np.ndarray,
    H: float,
    Zp: float,
    window_depth: float,
    window_kind: str,
    mullions_vertical: int,
    mullions_horizontal: int,
    mullion_offset_x: float,
    mullion_offset_z: float,
    partial_horizontal_bars: List[Tuple[int, float]],
    floor_cxy: Optional[np.ndarray] = None,
    wall_thickness: float = 0.0,
    floor_xy_ring: Optional[List[np.ndarray]] = None,
    FL_b_parapet: Optional[np.ndarray] = None,
    FR_b_parapet: Optional[np.ndarray] = None,
    extend_along_front_for_side_corners: bool = False,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Фронт над парапетом: при толстой стене — наружная кромка как у _front_parapet_one_quad,
    центр по глубине на наружной плоскости (стык с боковыми окнами без щели).
    extend_along_front_for_side_corners — слегка удлинить вдоль фасада, чтобы рамы перекрывались в углу.
    """
    win_h = H - Zp
    if mode in ("none", "open") or win_h <= 0.1:
        return []
    fl_i = np.asarray(FL_zp, dtype=np.float64)
    fr_i = np.asarray(FR_zp, dtype=np.float64)
    wt = max(float(wall_thickness), 0.0)
    mi = 0.5 * (fl_i + fr_i)
    o_fl = fl_i.copy()
    o_fr = fr_i.copy()
    if (
        wt > 1e-9
        and floor_xy_ring is not None
        and floor_cxy is not None
        and FL_b_parapet is not None
        and FR_b_parapet is not None
    ):
        ov = _outer_vertices_match_vertical_slab(
            [
                np.asarray(FL_b_parapet, dtype=np.float64),
                np.asarray(FR_b_parapet, dtype=np.float64),
                fr_i,
                fl_i,
            ],
            floor_cxy=np.asarray(floor_cxy, dtype=np.float64),
            wall_thickness=wt,
            floor_xy_ring=floor_xy_ring,
        )
        o_fl = np.asarray(ov[3], dtype=np.float64)
        o_fr = np.asarray(ov[2], dtype=np.float64)
    win_w_base = float(np.linalg.norm(o_fr - o_fl))
    if win_w_base <= 0.1:
        return []
    u_edge = (o_fr - o_fl) / win_w_base
    ext = 0.0025 if extend_along_front_for_side_corners else 0.0
    o_fl_e = o_fl - u_edge * ext
    o_fr_e = o_fr + u_edge * ext
    win_w = float(np.linalg.norm(o_fr_e - o_fl_e))
    mid_open = 0.5 * (o_fl_e + o_fr_e)
    n2 = mid_open[:2] - mi[:2]
    ln2 = float(np.linalg.norm(n2))
    wd = max(window_depth, 0.05)
    if ln2 < 1e-9:
        # Тонкая стена: внутренняя и наружная кромки совпадают — наружу по +Y (как в старой формуле cy).
        n2 = np.array([0.0, 1.0], dtype=np.float64)
        center_xy = mi[:2] + n2 * (wd * 0.5)
    else:
        n2 = n2 / ln2
        # mid_open на наружной плоскости; центр рамы — на половину глубины внутрь от фасада.
        center_xy = mid_open[:2] - n2 * (wd * 0.5)
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    cz = float(Zp) + win_h * 0.5
    kind = _pick_kind(window_kind, "fixed")
    ft = _frame_thickness(win_w, win_h)
    glass_t = max(window_depth * 0.12, 0.004)
    partial_bars = _normalize_partial_horizontal_bars(partial_horizontal_bars)
    mf, mg = build_window_frame_glass_meshes(
        width=win_w,
        height=win_h,
        depth=window_depth,
        profile="rect",
        kind=kind,
        mullions_vertical=mullions_vertical,
        mullions_horizontal=mullions_horizontal,
        mullion_offset_x=mullion_offset_x,
        mullion_offset_z=mullion_offset_z,
        partial_horizontal_bars=partial_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=0.0,
        with_glass=(mode == "with_glass"),
        merge_vertices=False,
    )
    wp: List[Tuple[str, trimesh.Trimesh]] = []
    if len(mf.vertices):
        mf.apply_translation([cx, cy, cz])
        wp.append(("frame", mf))
    if mode == "with_glass" and len(mg.vertices):
        mg.apply_translation([cx, cy, cz])
        wp.append(("glass", mg))
    return wp


def build_balcony_meshes(
    *,
    width_back: float,
    width_front: float,
    depth: float,
    height: float,
    floor_thickness: float,
    parapet_z_frac: float,
    parapet_height: float | None,
    window_mode: str,
    front_window_mode: str | None = None,
    window_depth: float,
    tilt_left_deg: float,
    tilt_right_deg: float,
    wall_upper_z_frac: float,
    side_upper_mode: str = "glass",
    mullions_vertical: int = 0,
    mullions_horizontal: int = 0,
    mullion_offset_x: float = 0.0,
    mullion_offset_z: float = 0.0,
    partial_horizontal_bars: List[Tuple[int, float]] | None = None,
    window_kind: str = "fixed",
    parapet_sill: bool = True,
    sill_thickness: float = 0.06,
    sill_depth: float = 0.1,
    simple_box: bool = False,
    floor_corner_left_wall: Optional[Tuple[float, float]] = None,
    floor_corner_right_wall: Optional[Tuple[float, float]] = None,
    floor_corner_front_left: Optional[Tuple[float, float]] = None,
    floor_corner_front_right: Optional[Tuple[float, float]] = None,
    vertical_prism: bool = True,
    window_left_wall: bool = False,
    window_right_wall: bool = False,
    side_parapet_split_frac: float = 0.0,
    side_parapet_separator_depth: float = 0.022,
    inner_wall_windows: Any = None,
    inner_wall_doors: Any = None,
    open_left_above_parapet: bool = False,
    open_right_above_parapet: bool = False,
    wall_thickness: float = 0.0,
) -> Tuple[List[Tuple[str, trimesh.Trimesh]], List[Tuple[str, trimesh.Trimesh]]]:
    """
    Парапет снизу по периметру фронта/боков, сверху — окно на всю ширину фронта.
    Углы основания (x,y) при z=0: floor_corner_left_wall / right_wall / front_left / front_right
    (левый у стены, правый у стены, передний левый, передний правый); если все None — width_back/width_front/depth.
    vertical_prism: верхнее кольцо = нижнее + (0,0,H) — боковые грани прямоугольники в вертикали.
    window_left_wall / window_right_wall — вторичные окна на боковых стенах (над парапетом).
    inner_wall_windows — окна во внутренней стене; inner_wall_doors — двери (вырез + procedural_door).
    front_window_mode — режим только для фронта над парапетом; None = как window_mode; open = без грани (дыра).
    open_left_above_parapet / open_right_above_parapet — убрать верх грани BR—FR / BL—FL (как --open-side-left / --open-side-right).
    wall_thickness — толщина вертикальных стен (м); при vertical_prism углы с митером по контуру низа пола.
    """
    H = max(height, 0.05)
    T = max(floor_thickness, 0.02)
    mode = window_mode.lower().strip()
    if mode not in ("with_glass", "frame_only", "none"):
        mode = "with_glass"
    fw = front_window_mode
    if fw is None or (isinstance(fw, str) and not str(fw).strip()):
        front_mode = mode
    else:
        front_mode = str(fw).lower().strip()
        if front_mode not in ("with_glass", "frame_only", "none", "open"):
            front_mode = mode
    sumode = side_upper_mode.lower().strip()
    if sumode not in ("glass", "wall"):
        sumode = "glass"
    wthick = max(0.0, float(wall_thickness))

    if parapet_height is not None:
        Zp = float(parapet_height)
    else:
        Zp = float(H) * max(0.0, min(1.0, float(parapet_z_frac)))
    Zp = float(np.clip(Zp, 0.07, H - 0.1))

    band = max(0.0, min(1.0, float(wall_upper_z_frac)))
    z_cut_back = H * (1.0 - band)

    BL, BR, FL, FR = _resolve_floor_quad(
        width_back,
        width_front,
        depth,
        floor_corner_left_wall,
        floor_corner_right_wall,
        floor_corner_front_left,
        floor_corner_front_right,
    )
    TBL, TBR, TFR, TFL = _compute_top_ring(
        BL, BR, FL, FR, H, vertical_prism, tilt_left_deg, tilt_right_deg
    )
    y_front = 0.5 * (float(FL[1]) + float(FR[1]))
    floor_cxy = _floor_center_xy(BL, BR, FL, FR)
    ph = partial_horizontal_bars if partial_horizontal_bars is not None else []
    ph_norm = _normalize_partial_horizontal_bars(ph)
    raw_inner = inner_wall_windows if inner_wall_windows is not None else []
    inner_default_mode = mode
    if mode == "none" and raw_inner:
        inner_default_mode = "with_glass"
    inner_w = _normalize_inner_wall_windows(
        raw_inner,
        default_mode=inner_default_mode,
        default_depth=float(window_depth),
        default_kind=str(window_kind),
        default_mv=mullions_vertical,
        default_mh=mullions_horizontal,
        default_ox=mullion_offset_x,
        default_oz=mullion_offset_z,
        default_partial=ph_norm,
    )
    raw_doors = inner_wall_doors if inner_wall_doors is not None else []
    inner_d = _normalize_inner_wall_doors(raw_doors)
    inner_openings_holes: List[dict] = list(inner_w) + list(inner_d)

    FL_zp = _edge_at_height(FL, TFL, Zp)
    FR_zp = _edge_at_height(FR, TFR, Zp)
    BL_zp = _edge_at_height(BL, TBL, Zp)
    BR_zp = _edge_at_height(BR, TBR, Zp)

    t_mit = max(float(T), 0.02)
    floor_ring_miter: Optional[List[np.ndarray]] = None
    if vertical_prism:
        _b_bl = np.asarray(BL, dtype=np.float64).copy()
        _b_bl[2] -= t_mit
        _b_br = np.asarray(BR, dtype=np.float64).copy()
        _b_br[2] -= t_mit
        _b_fr = np.asarray(FR, dtype=np.float64).copy()
        _b_fr[2] -= t_mit
        _b_fl = np.asarray(FL, dtype=np.float64).copy()
        _b_fl[2] -= t_mit
        floor_ring_miter = [_b_bl, _b_br, _b_fr, _b_fl]

    if simple_box:
        parts: List[Tuple[str, trimesh.Trimesh]] = []
        parts.extend(
            _balcony_floor_textured_parts(
                BL,
                FL,
                FR,
                BR,
                T,
                FR_zp=FR_zp,
                BR_zp=BR_zp,
                FL_zp=FL_zp,
                BL_zp=BL_zp,
            )
        )
        parts.extend(
            _simple_box_back_and_sides(
                BL,
                BR,
                FL,
                FR,
                TBL,
                TBR,
                TFR,
                TFL,
                BL_zp,
                FL_zp,
                BR_zp,
                FR_zp,
                floor_thickness=T,
                window_left_wall=window_left_wall,
                window_right_wall=window_right_wall,
                mode=mode,
                sumode=sumode,
                floor_cxy=floor_cxy,
                side_parapet_split_frac=float(side_parapet_split_frac),
                side_parapet_separator_depth=float(side_parapet_separator_depth),
                inner_wall_windows=inner_w,
                inner_wall_doors=inner_d,
                open_left_above_parapet=bool(open_left_above_parapet),
                open_right_above_parapet=bool(open_right_above_parapet),
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
            )
        )
        parts.extend(
            _front_parapet_one_quad(
                FL,
                FR,
                FL_zp,
                FR_zp,
                Zp,
                floor_thickness=T,
                floor_cxy=floor_cxy,
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
            )
        )
        if parapet_sill:
            st = max(sill_thickness, 0.03)
            sd = max(sill_depth, 0.05)
            cx_f = 0.5 * (FL_zp[0] + FR_zp[0])
            w_f = float(np.linalg.norm(FR_zp - FL_zp)) + 0.04
            sill = trimesh.creation.box(extents=[w_f, sd, st])
            sill.apply_translation([cx_f, y_front + sd * 0.5 + 0.01, Zp + st * 0.48])
            parts.append(("divider_frame", sill))
        win_h = H - Zp
        window_parts: List[Tuple[str, trimesh.Trimesh]] = []
        _flb_p = np.asarray(FL, dtype=np.float64).copy()
        _flb_p[2] -= T
        _frb_p = np.asarray(FR, dtype=np.float64).copy()
        _frb_p[2] -= T
        _corner_ext = (window_left_wall or window_right_wall) and mode != "none" and front_mode not in (
            "none",
            "open",
        )
        window_parts.extend(
            _balcony_procedural_window_parts(
                mode=front_mode,
                FL_zp=FL_zp,
                FR_zp=FR_zp,
                H=H,
                Zp=Zp,
                window_depth=window_depth,
                window_kind=window_kind,
                mullions_vertical=mullions_vertical,
                mullions_horizontal=mullions_horizontal,
                mullion_offset_x=mullion_offset_x,
                mullion_offset_z=mullion_offset_z,
                partial_horizontal_bars=ph,
                floor_cxy=floor_cxy,
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
                FL_b_parapet=_flb_p,
                FR_b_parapet=_frb_p,
                extend_along_front_for_side_corners=_corner_ext,
            )
        )
        _side_corner_ext = 0.0025 if front_mode not in ("none", "open") else 0.0
        if window_left_wall and mode != "none":
            pb_l, pf_l = np.asarray(BL_zp, dtype=np.float64), np.asarray(FL_zp, dtype=np.float64)
            if wthick > 1e-9 and floor_ring_miter is not None:
                fl_b = np.asarray(FL, dtype=np.float64).copy()
                fl_b[2] -= T
                bl_b = np.asarray(BL, dtype=np.float64).copy()
                bl_b[2] -= T
                ov = _outer_vertices_match_vertical_slab(
                    [fl_b, FL_zp, BL_zp, bl_b],
                    floor_cxy=floor_cxy,
                    wall_thickness=wthick,
                    floor_xy_ring=floor_ring_miter,
                )
                pb_l, pf_l = np.asarray(ov[2], dtype=np.float64), np.asarray(ov[1], dtype=np.float64)
            if _side_corner_ext > 0.0:
                uu = pf_l - pb_l
                lu = float(np.linalg.norm(uu))
                if lu > 1e-9:
                    pf_l = pf_l + (uu / lu) * _side_corner_ext
            mid_l = 0.5 * (pb_l[:2] + pf_l[:2])
            window_parts.extend(
                _window_parts_on_wall_edge(
                    mode=mode,
                    p_back_zp=pb_l,
                    p_front_zp=pf_l,
                    z_bottom=Zp,
                    z_top=H,
                    window_depth=window_depth,
                    window_kind=window_kind,
                    mullions_vertical=mullions_vertical,
                    mullions_horizontal=mullions_horizontal,
                    mullion_offset_x=mullion_offset_x,
                    mullion_offset_z=mullion_offset_z,
                    partial_horizontal_bars=ph,
                    inward_xy=_inward_horizontal(mid_l, floor_cxy),
                )
            )
        if window_right_wall and mode != "none":
            pb_r, pf_r = np.asarray(BR_zp, dtype=np.float64), np.asarray(FR_zp, dtype=np.float64)
            if wthick > 1e-9 and floor_ring_miter is not None:
                fr_b = np.asarray(FR, dtype=np.float64).copy()
                fr_b[2] -= T
                br_b = np.asarray(BR, dtype=np.float64).copy()
                br_b[2] -= T
                ov = _outer_vertices_match_vertical_slab(
                    [fr_b, FR_zp, BR_zp, br_b],
                    floor_cxy=floor_cxy,
                    wall_thickness=wthick,
                    floor_xy_ring=floor_ring_miter,
                )
                pb_r, pf_r = np.asarray(ov[2], dtype=np.float64), np.asarray(ov[1], dtype=np.float64)
            if _side_corner_ext > 0.0:
                uu = pf_r - pb_r
                lu = float(np.linalg.norm(uu))
                if lu > 1e-9:
                    pf_r = pf_r + (uu / lu) * _side_corner_ext
            mid_r = 0.5 * (pb_r[:2] + pf_r[:2])
            window_parts.extend(
                _window_parts_on_wall_edge(
                    mode=mode,
                    p_back_zp=pb_r,
                    p_front_zp=pf_r,
                    z_bottom=Zp,
                    z_top=H,
                    window_depth=window_depth,
                    window_kind=window_kind,
                    mullions_vertical=mullions_vertical,
                    mullions_horizontal=mullions_horizontal,
                    mullion_offset_x=mullion_offset_x,
                    mullion_offset_z=mullion_offset_z,
                    partial_horizontal_bars=ph,
                    inward_xy=_inward_horizontal(mid_r, floor_cxy),
                )
            )
        BL_b_in = np.asarray(BL, dtype=np.float64).copy()
        BL_b_in[2] -= T
        BR_b_in = np.asarray(BR, dtype=np.float64).copy()
        BR_b_in[2] -= T
        _append_inner_back_wall_window_meshes(
            window_parts, inner_w, BL_b_in, BR_b_in, TBL, TBR, BL, BR, floor_cxy
        )
        _append_inner_back_wall_door_meshes(
            window_parts, inner_d, BL_b_in, BR_b_in, TBL, TBR, BL, BR, floor_cxy
        )
        if front_mode == "none" and win_h > 0.05:
            lo_u, hi_u = _make_wall_stack(
                FL_zp,
                FR_zp,
                TFR,
                TFL,
                z_cut_back,
                floor_cxy=floor_cxy,
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
            )
            parts.append(("wall_upper", lo_u))
            parts.append(("wall_upper", hi_u))
        return parts, window_parts

    ol = bool(open_left_above_parapet)
    orr = bool(open_right_above_parapet)

    wall_parts: List[Tuple[str, trimesh.Trimesh]] = []
    wall_parts.extend(
        _balcony_floor_textured_parts(
            BL,
            FL,
            FR,
            BR,
            T,
            FR_zp=FR_zp,
            BR_zp=BR_zp,
            FL_zp=FL_zp,
            BL_zp=BL_zp,
        )
    )
    BL_b = np.asarray(BL, dtype=np.float64).copy()
    BL_b[2] -= T
    BR_b = np.asarray(BR, dtype=np.float64).copy()
    BR_b[2] -= T
    FR_b = np.asarray(FR, dtype=np.float64).copy()
    FR_b[2] -= T
    FL_b = np.asarray(FL, dtype=np.float64).copy()
    FL_b[2] -= T

    # --- задняя стена по углам BL—BR—TBR—TBL ---
    b_blc = _lerp_z(BL, TBL, z_cut_back)
    b_brc = _lerp_z(BR, TBR, z_cut_back)
    e_b = BR - BL
    lb = float(np.linalg.norm(e_b))
    if lb < 1e-9:
        e_bh = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lb = 1.0
    else:
        e_bh = e_b / lb
    zb_back = float(BL_b[2])
    z_lo_span_ext = max(float(z_cut_back) - zb_back, 1e-6)
    z_hi_span = max(H - z_cut_back, 1e-6)

    def _uv_back_lo(p: np.ndarray) -> np.ndarray:
        uu = float(np.dot(np.asarray(p, dtype=np.float64) - BL, e_bh)) / lb
        uu = float(np.clip(uu, 0.0, 1.0))
        vv = (float(p[2]) - zb_back) / z_lo_span_ext
        return np.array([uu, vv], dtype=np.float64)

    def _uv_back_hi(p: np.ndarray) -> np.ndarray:
        uu = float(np.dot(np.asarray(p, dtype=np.float64) - BL, e_bh)) / lb
        uu = float(np.clip(uu, 0.0, 1.0))
        vv = (float(p[2]) - z_cut_back) / z_hi_span
        return np.array([uu, vv], dtype=np.float64)

    holes_lo = _inner_holes_us_in_z_span(inner_openings_holes, float(BL_b[2]), float(b_blc[2]))
    wall_parts.extend(
        _perforated_back_wall_patches(
            BL_b,
            BR_b,
            b_brc,
            b_blc,
            holes_lo,
            _uv_back_lo,
            flip=True,
            tile_i=BALCONY_TILE_WALL_LOWER,
            part_name="wall_lower",
            floor_cxy=floor_cxy,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
    )
    holes_hi = _inner_holes_us_in_z_span(inner_openings_holes, float(b_blc[2]), float(TBL[2]))
    wall_parts.extend(
        _perforated_back_wall_patches(
            b_blc,
            b_brc,
            TBR,
            TBL,
            holes_hi,
            _uv_back_hi,
            flip=True,
            tile_i=BALCONY_TILE_WALL_UPPER,
            part_name="wall_upper",
            floor_cxy=floor_cxy,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
    )

    # --- перед: один прямоугольник парапета под окном ---
    wall_parts.extend(
        _front_parapet_one_quad(
            FL,
            FR,
            FL_zp,
            FR_zp,
            Zp,
            floor_thickness=T,
            floor_cxy=floor_cxy,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
    )

    # --- боковины: прямоугольник парапета + верх (стекло/стена), если нет окна на этой стене ---
    side_tile = BALCONY_TILE_GLASS if sumode == "glass" else BALCONY_TILE_WALL_UPPER
    side_name = "side_glass" if sumode == "glass" else "wall_upper"
    wall_parts.extend(
        _side_parapet_left_meshes(
            BL_b,
            FL_b,
            BL_zp,
            FL_zp,
            floor_cxy,
            split_frac=float(side_parapet_split_frac),
            separator_depth=float(side_parapet_separator_depth),
            window_on_side=window_left_wall,
            window_mode=mode,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
    )
    if ((not window_left_wall) or mode == "none") and not orr:
        uv_nl = _uv_planar_quad_bl_br_tr_tl(BL_zp, FL_zp, TFL, TBL)
        wall_parts.append(
            (
                side_name,
                _slab_or_quad_vertical_wall(
                    [BL_zp, FL_zp, TFL, TBL],
                    side_tile,
                    uv_nl,
                    flip=False,
                    floor_cxy=floor_cxy,
                    wall_thickness=wthick,
                    floor_xy_ring=floor_ring_miter,
                ),
            )
        )
    wall_parts.extend(
        _side_parapet_right_meshes(
            BR_b,
            FR_b,
            BR_zp,
            FR_zp,
            floor_cxy,
            split_frac=float(side_parapet_split_frac),
            separator_depth=float(side_parapet_separator_depth),
            window_on_side=window_right_wall,
            window_mode=mode,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
    )
    if ((not window_right_wall) or mode == "none") and not ol:
        uv_nr = _uv_planar_quad_bl_br_tr_tl(BR_zp, FR_zp, TFR, TBR)
        wall_parts.append(
            (
                side_name,
                _slab_or_quad_vertical_wall(
                    [BR_zp, FR_zp, TFR, TBR],
                    side_tile,
                    uv_nr,
                    flip=False,
                    floor_cxy=floor_cxy,
                    wall_thickness=wthick,
                    floor_xy_ring=floor_ring_miter,
                ),
            )
        )

    if parapet_sill:
        st = max(sill_thickness, 0.03)
        sd = max(sill_depth, 0.05)
        cx_f = 0.5 * (FL_zp[0] + FR_zp[0])
        w_f = float(np.linalg.norm(FR_zp - FL_zp)) + 0.04
        sill = trimesh.creation.box(extents=[w_f, sd, st])
        sill.apply_translation([cx_f, y_front + sd * 0.5 + 0.01, Zp + st * 0.48])
        wall_parts.append(("divider_frame", sill))

    window_parts: List[Tuple[str, trimesh.Trimesh]] = []
    _corner_ext2 = (window_left_wall or window_right_wall) and mode != "none" and front_mode not in (
        "none",
        "open",
    )
    _side_corner_ext2 = 0.0025 if front_mode not in ("none", "open") else 0.0
    window_parts.extend(
        _balcony_procedural_window_parts(
            mode=front_mode,
            FL_zp=FL_zp,
            FR_zp=FR_zp,
            H=H,
            Zp=Zp,
            window_depth=window_depth,
            window_kind=window_kind,
            mullions_vertical=mullions_vertical,
            mullions_horizontal=mullions_horizontal,
            mullion_offset_x=mullion_offset_x,
            mullion_offset_z=mullion_offset_z,
            partial_horizontal_bars=ph,
            floor_cxy=floor_cxy,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
            FL_b_parapet=FL_b,
            FR_b_parapet=FR_b,
            extend_along_front_for_side_corners=_corner_ext2,
        )
    )
    if window_left_wall and mode != "none":
        pb_l, pf_l = np.asarray(BL_zp, dtype=np.float64), np.asarray(FL_zp, dtype=np.float64)
        if wthick > 1e-9 and floor_ring_miter is not None:
            ov = _outer_vertices_match_vertical_slab(
                [FL_b, FL_zp, BL_zp, BL_b],
                floor_cxy=floor_cxy,
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
            )
            pb_l, pf_l = np.asarray(ov[2], dtype=np.float64), np.asarray(ov[1], dtype=np.float64)
        if _side_corner_ext2 > 0.0:
            uu = pf_l - pb_l
            lu = float(np.linalg.norm(uu))
            if lu > 1e-9:
                pf_l = pf_l + (uu / lu) * _side_corner_ext2
        mid_l = 0.5 * (pb_l[:2] + pf_l[:2])
        window_parts.extend(
            _window_parts_on_wall_edge(
                mode=mode,
                p_back_zp=pb_l,
                p_front_zp=pf_l,
                z_bottom=Zp,
                z_top=H,
                window_depth=window_depth,
                window_kind=window_kind,
                mullions_vertical=mullions_vertical,
                mullions_horizontal=mullions_horizontal,
                mullion_offset_x=mullion_offset_x,
                mullion_offset_z=mullion_offset_z,
                partial_horizontal_bars=ph,
                inward_xy=_inward_horizontal(mid_l, floor_cxy),
            )
        )
    if window_right_wall and mode != "none":
        pb_r, pf_r = np.asarray(BR_zp, dtype=np.float64), np.asarray(FR_zp, dtype=np.float64)
        if wthick > 1e-9 and floor_ring_miter is not None:
            ov = _outer_vertices_match_vertical_slab(
                [FR_b, FR_zp, BR_zp, BR_b],
                floor_cxy=floor_cxy,
                wall_thickness=wthick,
                floor_xy_ring=floor_ring_miter,
            )
            pb_r, pf_r = np.asarray(ov[2], dtype=np.float64), np.asarray(ov[1], dtype=np.float64)
        if _side_corner_ext2 > 0.0:
            uu = pf_r - pb_r
            lu = float(np.linalg.norm(uu))
            if lu > 1e-9:
                pf_r = pf_r + (uu / lu) * _side_corner_ext2
        mid_r = 0.5 * (pb_r[:2] + pf_r[:2])
        window_parts.extend(
            _window_parts_on_wall_edge(
                mode=mode,
                p_back_zp=pb_r,
                p_front_zp=pf_r,
                z_bottom=Zp,
                z_top=H,
                window_depth=window_depth,
                window_kind=window_kind,
                mullions_vertical=mullions_vertical,
                mullions_horizontal=mullions_horizontal,
                mullion_offset_x=mullion_offset_x,
                mullion_offset_z=mullion_offset_z,
                partial_horizontal_bars=ph,
                inward_xy=_inward_horizontal(mid_r, floor_cxy),
            )
        )
    _append_inner_back_wall_window_meshes(
        window_parts, inner_w, BL_b, BR_b, TBL, TBR, BL, BR, floor_cxy
    )
    _append_inner_back_wall_door_meshes(
        window_parts, inner_d, BL_b, BR_b, TBL, TBR, BL, BR, floor_cxy
    )

    win_h = H - Zp
    if front_mode == "none" and win_h > 0.05:
        lo_u, hi_u = _make_wall_stack(
            FL_zp,
            FR_zp,
            TFR,
            TFL,
            z_cut_back,
            floor_cxy=floor_cxy,
            wall_thickness=wthick,
            floor_xy_ring=floor_ring_miter,
        )
        wall_parts.append(("wall_upper", lo_u))
        wall_parts.append(("wall_upper", hi_u))

    return wall_parts, window_parts


def export_balcony(
    out_dir: Path | None = None,
    *,
    no_view: bool = False,
    atlas_tile: int = 256,
    wall_lower_tex: str | Path | None = None,
    wall_upper_tex: str | Path | None = None,
    frame_tex: str | Path | None = None,
    glass_tex: str | Path | None = None,
    side_basket_tex: str | Path | None = None,
    side_jamb_tex: str | Path | None = None,
    side_separator_tex: str | Path | None = None,
    generate_normal_map: bool = True,
    generate_roughness_map: bool = True,
    **kwargs: Any,
) -> Path:
    out_dir = out_dir or (_repo_root() / "data" / "balcony_export")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _BALCONY_TEX_COLOR_KEYS = (
        "wall_lower_tex_color",
        "wall_upper_tex_color",
        "frame_tex_color",
        "glass_tex_color",
        "side_basket_tex_color",
        "side_jamb_tex_color",
        "side_separator_tex_color",
    )
    kw_rest = dict(kwargs)
    tex_color_raw = {k: kw_rest.pop(k, None) for k in _BALCONY_TEX_COLOR_KEYS}
    wl_c = parse_texture_color_tint(tex_color_raw["wall_lower_tex_color"])
    wu_c = parse_texture_color_tint(tex_color_raw["wall_upper_tex_color"])
    fr_c = parse_texture_color_tint(tex_color_raw["frame_tex_color"])
    gl_c = parse_texture_color_tint(tex_color_raw["glass_tex_color"])
    sb_c = parse_texture_color_tint(tex_color_raw["side_basket_tex_color"])
    sj_c = parse_texture_color_tint(tex_color_raw["side_jamb_tex_color"])
    sep_c = parse_texture_color_tint(tex_color_raw["side_separator_tex_color"])

    u = USER_BALCONY
    p = {**u, **kw_rest}
    ph = p.get("parapet_height")
    wall_parts, win_parts = build_balcony_meshes(
        width_back=float(p["width_back"]),
        width_front=float(p["width_front"]),
        depth=float(p["depth"]),
        height=float(p["height"]),
        floor_thickness=float(p["floor_thickness"]),
        parapet_z_frac=float(p.get("parapet_z_frac", 0.42)),
        parapet_height=float(ph) if ph is not None else None,
        window_mode=str(p["window_mode"]),
        front_window_mode=p.get("front_window_mode"),
        window_depth=float(p["window_depth"]),
        tilt_left_deg=float(p["tilt_left_deg"]),
        tilt_right_deg=float(p["tilt_right_deg"]),
        wall_upper_z_frac=float(p["wall_upper_z_frac"]),
        side_upper_mode=str(p.get("side_upper_mode", "glass")),
        mullions_vertical=_pick_nonneg_int(p.get("mullions_vertical"), 0),
        mullions_horizontal=_pick_nonneg_int(p.get("mullions_horizontal"), 0),
        mullion_offset_x=float(p.get("mullion_offset_x", 0.0)),
        mullion_offset_z=float(p.get("mullion_offset_z", 0.0)),
        partial_horizontal_bars=_normalize_partial_horizontal_bars(p.get("partial_horizontal_bars")),
        window_kind=str(p.get("window_kind", "fixed")),
        parapet_sill=bool(p.get("parapet_sill", True)),
        sill_thickness=float(p.get("sill_thickness", 0.06)),
        sill_depth=float(p.get("sill_depth", 0.1)),
        simple_box=bool(p.get("simple_box", False)),
        floor_corner_left_wall=p.get("floor_corner_left_wall"),
        floor_corner_right_wall=p.get("floor_corner_right_wall"),
        floor_corner_front_left=p.get("floor_corner_front_left"),
        floor_corner_front_right=p.get("floor_corner_front_right"),
        vertical_prism=bool(p.get("vertical_prism", True)),
        window_left_wall=bool(p.get("window_left_wall", False)),
        window_right_wall=bool(p.get("window_right_wall", False)),
        side_parapet_split_frac=float(p.get("side_parapet_split_frac", 0.0)),
        side_parapet_separator_depth=float(p.get("side_parapet_separator_depth", 0.022)),
        inner_wall_windows=p.get("inner_wall_windows"),
        inner_wall_doors=p.get("inner_wall_doors"),
        open_left_above_parapet=bool(p.get("open_left_above_parapet", False)),
        open_right_above_parapet=bool(p.get("open_right_above_parapet", False)),
        wall_thickness=float(p.get("wall_thickness", 0.0)),
    )

    atlas_img = make_balcony_atlas(
        tile=max(atlas_tile, 64),
        wall_lower_path=wall_lower_tex,
        wall_upper_path=wall_upper_tex,
        frame_path=frame_tex,
        glass_path=glass_tex,
        side_basket_path=side_basket_tex,
        side_jamb_path=side_jamb_tex,
        side_separator_path=side_separator_tex,
        wall_lower_color=wl_c,
        wall_upper_color=wu_c,
        frame_color=fr_c,
        glass_color=gl_c,
        side_basket_color=sb_c,
        side_jamb_color=sj_c,
        side_separator_color=sep_c,
    )
    tex_name = "balcony_atlas.png"
    tex_path = out_dir / tex_name
    atlas_img.save(tex_path)
    normal_name = "balcony_normal_atlas.png"
    rough_name = "balcony_roughness_atlas.png"
    if generate_normal_map:
        make_normal_map_from_albedo(atlas_img, strength=3.4).save(out_dir / normal_name)
    if generate_roughness_map:
        make_roughness_map_from_albedo(atlas_img, min_roughness=0.3, max_roughness=0.92).save(out_dir / rough_name)

    mesh_blocks: List[trimesh.Trimesh] = []

    for name, m in wall_parts:
        if len(m.faces) == 0:
            continue
        if _mesh_has_per_vertex_uv(m):
            mesh_blocks.append(m)
            continue
        m2, uv = faceted_triplanar_uv(m)
        if name == "wall_lower":
            tile_i = BALCONY_TILE_WALL_LOWER
        elif name == "wall_upper":
            tile_i = BALCONY_TILE_WALL_UPPER
        elif name == "divider_frame":
            tile_i = BALCONY_TILE_FRAME
        elif name == "side_glass":
            tile_i = BALCONY_TILE_GLASS
        elif name == "side_lower_basket":
            tile_i = BALCONY_TILE_SIDE_BASKET
        elif name == "side_lower_jamb":
            tile_i = BALCONY_TILE_SIDE_JAMB
        elif name == "side_separator":
            tile_i = BALCONY_TILE_SIDE_SEPARATOR
        else:
            tile_i = BALCONY_TILE_WALL_LOWER
        m2.visual = trimesh.visual.texture.TextureVisuals(uv=_scale_uv_to_tile(uv, tile_i))
        mesh_blocks.append(m2)

    for name, m in win_parts:
        if len(m.faces) == 0:
            continue
        m2, uv = faceted_triplanar_uv(m)
        is_door_glass = name.startswith("door_") and "glass" in name
        is_door_solid = name.startswith("door_") and not is_door_glass
        if name == "frame" or is_door_solid:
            uv_t = _scale_uv_to_tile(uv, BALCONY_TILE_FRAME)
        else:
            uv_t = _scale_uv_to_tile(uv, BALCONY_TILE_GLASS)
        m2.visual = trimesh.visual.texture.TextureVisuals(uv=uv_t)
        if name == "glass" or is_door_glass:
            m2 = _double_sided_copy_uv(m2)
        mesh_blocks.append(m2)

    if not mesh_blocks:
        raise RuntimeError("balcony: empty mesh")

    work = _concatenate_uv_meshes(mesh_blocks)
    _uv = np.asarray(work.visual.uv, dtype=np.float64)
    work.visual = trimesh.visual.texture.TextureVisuals(uv=_uv, image=atlas_img)

    obj_path = out_dir / "balcony.obj"
    work.export(str(obj_path), include_texture=True)

    mtl_path = out_dir / "material.mtl"
    if mtl_path.is_file():
        txt = mtl_path.read_text(encoding="utf-8")
        txt = txt.replace("map_Kd material_0.png", f"map_Kd {tex_name}")
        txt = txt.replace("map_Kd material_0.jpg", f"map_Kd {tex_name}")
        txt = re.sub(r"(?m)^Ka\s+.*$", "Ka 1 1 1", txt)
        txt = re.sub(r"(?m)^Kd\s+.*$", "Kd 1 1 1", txt)
        txt = re.sub(r"(?m)^Ks\s+.*$", "Ks 0 0 0", txt)
        low = txt.lower()
        if generate_normal_map and "map_bump" not in low and "map_kn" not in low:
            txt = txt.rstrip() + f"\nmap_Bump -bm 0.700 {normal_name}\n"
        if generate_roughness_map and "map_pr" not in low:
            txt = txt.rstrip() + f"\nmap_Pr {rough_name}\n"
        mtl_path.write_text(txt, encoding="utf-8")

    print(f"[OK] Balcony export: {obj_path}")
    print(f"     Atlas: {tex_path}")

    if not no_view:
        preview_balcony_obj_open3d(obj_path)
    return obj_path


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Экспорт процедурного балкона (OBJ + атлас).")
    ap.add_argument("command", nargs="?", default="export", choices=("export",), help="Команда")
    ap.add_argument("-o", "--output", type=str, default=None, help="Папка вывода")
    ap.add_argument("--no-view", action="store_true", help="Не открывать Open3D")
    ap.add_argument(
        "--simple-box",
        action="store_true",
        help="Грубая коробка: пол + зад/бока + передний парапет и окно; верх открыт",
    )
    ap.add_argument(
        "--floor-left-wall",
        type=str,
        default=None,
        metavar="X,Y",
        help="Угол «левый у стены» (x,y) в м; вместе с остальными тремя — четырёхугольник основания",
    )
    ap.add_argument("--floor-right-wall", type=str, default=None, metavar="X,Y", help="Правый у стены")
    ap.add_argument("--floor-front-left", type=str, default=None, metavar="X,Y", help="Передний левый")
    ap.add_argument("--floor-front-right", type=str, default=None, metavar="X,Y", help="Передний правый")
    ap.add_argument(
        "--legacy-tilt-top",
        action="store_true",
        help="Верхнее кольцо по tilt-left/right (наклон); иначе вертикальная призма (бока прямоугольники)",
    )
    ap.add_argument("--window-left-wall", action="store_true", help="Окно на левой боковой стене")
    ap.add_argument("--window-right-wall", action="store_true", help="Окно на правой боковой стене")
    ap.add_argument(
        "--inner-wall-window",
        action="append",
        default=None,
        metavar="SPEC",
        help="Окно на внутренней (задней) стене: u0,u1,z_bottom,z_top[,mv=,mh=,mode=,depth=,kind=,ox=,oz=]",
    )
    ap.add_argument(
        "--inner-wall-windows-json",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON-массив объектов окон (u0,u1,z_bottom,z_top и опции); суммируется с --inner-wall-window",
    )
    ap.add_argument(
        "--inner-wall-door",
        action="append",
        default=None,
        metavar="SPEC",
        help="Дверь на внутренней стене: u0,u1,z_bottom,z_top[,style=french|slab,fw=,fd=,gap=,mid=,y0=]",
    )
    ap.add_argument(
        "--inner-wall-doors-json",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON-массив дверей; суммируется с --inner-wall-door",
    )
    ap.add_argument("--width-back", type=float, default=None)
    ap.add_argument("--width-front", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--floor-thickness", type=float, default=None)
    ap.add_argument(
        "--wall-thickness",
        type=float,
        default=None,
        metavar="M",
        help="Толщина вертикальных стен (м): призма наружу от внутренней грани в −n_in; 0 — как раньше",
    )
    ap.add_argument(
        "--parapet-frac",
        type=float,
        default=None,
        help="Высота парапета как доля от height (если не задан --parapet-height)",
    )
    ap.add_argument("--parapet-height", type=float, default=None, help="Абсолютная высота парапета (м), перекрывает --parapet-frac")
    ap.add_argument(
        "--side-upper",
        type=str,
        default=None,
        choices=("glass", "wall"),
        help="Над парапетом на боках: стекло (как ограждение) или та же стена",
    )
    ap.add_argument(
        "--window-mode",
        type=str,
        default=None,
        choices=("with_glass", "frame_only", "none"),
        help="Стекло | только рама | без окна (глухой фронт)",
    )
    ap.add_argument(
        "--front-window-mode",
        type=str,
        default=None,
        choices=("with_glass", "frame_only", "none", "open"),
        help="Только фронт: open=дыра, none=стена, with_glass|frame_only=окно; по умолчанию как --window-mode",
    )
    ap.add_argument(
        "--open-side-left",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Убрать верх боковой грани BR—FR (+X от центра), над парапетом",
    )
    ap.add_argument(
        "--open-side-right",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Убрать верх боковой грани BL—FL (−X), над парапетом",
    )
    ap.add_argument("--window-depth", type=float, default=None)
    ap.add_argument("--tilt-left-deg", type=float, default=None, help="Наклон левой боковины вперёд (+Y) наверху")
    ap.add_argument("--tilt-right-deg", type=float, default=None)
    ap.add_argument("--wall-upper-frac", type=float, default=None, help="Доля высоты ограды под 'верхнюю' текстуру")
    ap.add_argument("--atlas-tile", type=int, default=256, help="Сторона одной плитки атласа")
    ap.add_argument("--wall-lower-tex", type=str, default=None, help="PNG нижней зоны (перед/зад/пол/сплошные бока)")
    ap.add_argument("--wall-upper-tex", type=str, default=None)
    ap.add_argument("--frame-tex", type=str, default=None)
    ap.add_argument("--glass-tex", type=str, default=None)
    ap.add_argument(
        "--side-basket-tex",
        type=str,
        default=None,
        help="PNG глубокой части бокового парапета (5-я колонка атласа из 7)",
    )
    ap.add_argument(
        "--side-jamb-tex",
        type=str,
        default=None,
        help="PNG узкой полосы бокового парапета у переднего угла (6-я колонка)",
    )
    ap.add_argument(
        "--side-separator-tex",
        type=str,
        default=None,
        help="PNG вертикальной грани между корзиной и полосой у окна (7-я колонка); нет при боковом окне",
    )
    ap.add_argument(
        "--side-parapet-split-frac",
        type=float,
        default=None,
        help="Доля глубины от переднего угла к заднему для полосы «у окна» (0 = без разбиения)",
    )
    ap.add_argument(
        "--side-separator-depth",
        type=float,
        default=None,
        help="Вынос грани-перегородки внутрь балкона (м)",
    )
    ap.add_argument("--mullions-vertical", type=int, default=None)
    ap.add_argument("--mullions-horizontal", type=int, default=None)
    ap.add_argument("--partial-h", action="append", default=None, metavar="BAY:ZFRAC")
    ap.add_argument("--no-sill", action="store_true", help="Убрать горизонтальный отлив на линии парапета")
    ap.add_argument("--sill-thickness", type=float, default=None)
    ap.add_argument("--sill-depth", type=float, default=None)
    return ap


def main(argv: List[str] | None = None) -> None:
    ap = _build_cli()
    args = ap.parse_args(argv)
    kw: dict[str, Any] = {}
    if args.width_back is not None:
        kw["width_back"] = args.width_back
    if args.width_front is not None:
        kw["width_front"] = args.width_front
    if args.depth is not None:
        kw["depth"] = args.depth
    if args.height is not None:
        kw["height"] = args.height
    if args.floor_thickness is not None:
        kw["floor_thickness"] = args.floor_thickness
    if args.wall_thickness is not None:
        kw["wall_thickness"] = args.wall_thickness
    if args.parapet_frac is not None:
        kw["parapet_z_frac"] = args.parapet_frac
    if args.parapet_height is not None:
        kw["parapet_height"] = args.parapet_height
    if args.side_upper is not None:
        kw["side_upper_mode"] = args.side_upper
    if args.window_mode is not None:
        kw["window_mode"] = args.window_mode
    if args.front_window_mode is not None:
        kw["front_window_mode"] = args.front_window_mode
    if args.open_side_left is not None:
        kw["open_left_above_parapet"] = args.open_side_left
    if args.open_side_right is not None:
        kw["open_right_above_parapet"] = args.open_side_right
    if args.window_depth is not None:
        kw["window_depth"] = args.window_depth
    if args.tilt_left_deg is not None:
        kw["tilt_left_deg"] = args.tilt_left_deg
    if args.tilt_right_deg is not None:
        kw["tilt_right_deg"] = args.tilt_right_deg
    if args.wall_upper_frac is not None:
        kw["wall_upper_z_frac"] = args.wall_upper_frac
    if args.mullions_vertical is not None:
        kw["mullions_vertical"] = args.mullions_vertical
    if args.mullions_horizontal is not None:
        kw["mullions_horizontal"] = args.mullions_horizontal
    partial_kw: List[Tuple[int, float]] | None = None
    if args.partial_h:
        kw["partial_horizontal_bars"] = _parse_partial_h_tokens(args.partial_h)
    if args.no_sill:
        kw["parapet_sill"] = False
    if args.sill_thickness is not None:
        kw["sill_thickness"] = args.sill_thickness
    if args.sill_depth is not None:
        kw["sill_depth"] = args.sill_depth
    if args.simple_box:
        kw["simple_box"] = True
    corners = [
        args.floor_left_wall,
        args.floor_right_wall,
        args.floor_front_left,
        args.floor_front_right,
    ]
    if any(corners):
        if not all(corners):
            ap.error("Задайте все четыре угла: --floor-left-wall, --floor-right-wall, --floor-front-left, --floor-front-right")
        kw["floor_corner_left_wall"] = _parse_floor_xy_arg(args.floor_left_wall)
        kw["floor_corner_right_wall"] = _parse_floor_xy_arg(args.floor_right_wall)
        kw["floor_corner_front_left"] = _parse_floor_xy_arg(args.floor_front_left)
        kw["floor_corner_front_right"] = _parse_floor_xy_arg(args.floor_front_right)
    if args.legacy_tilt_top:
        kw["vertical_prism"] = False
    if args.window_left_wall:
        kw["window_left_wall"] = True
    if args.window_right_wall:
        kw["window_right_wall"] = True
    if args.side_parapet_split_frac is not None:
        kw["side_parapet_split_frac"] = args.side_parapet_split_frac
    if args.side_separator_depth is not None:
        kw["side_parapet_separator_depth"] = args.side_separator_depth

    inner_specs: List[dict] = []
    if args.inner_wall_windows_json:
        jpath = Path(args.inner_wall_windows_json).expanduser()
        raw = json.loads(jpath.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            inner_specs.extend(x for x in raw if isinstance(x, dict))
        else:
            ap.error("--inner-wall-windows-json: ожидается JSON-массив объектов")
    if args.inner_wall_window:
        for spec_s in args.inner_wall_window:
            inner_specs.append(_parse_inner_wall_window_cli(spec_s))
    if inner_specs:
        kw["inner_wall_windows"] = inner_specs

    door_specs: List[dict] = []
    if args.inner_wall_doors_json:
        djpath = Path(args.inner_wall_doors_json).expanduser()
        draw = json.loads(djpath.read_text(encoding="utf-8"))
        if isinstance(draw, list):
            door_specs.extend(x for x in draw if isinstance(x, dict))
        else:
            ap.error("--inner-wall-doors-json: ожидается JSON-массив объектов")
    if args.inner_wall_door:
        for spec_s in args.inner_wall_door:
            door_specs.append(_parse_inner_wall_door_cli(spec_s))
    if door_specs:
        kw["inner_wall_doors"] = door_specs

    out = Path(args.output).resolve() if args.output else None
    export_balcony(
        out,
        no_view=args.no_view,
        atlas_tile=args.atlas_tile,
        wall_lower_tex=args.wall_lower_tex,
        wall_upper_tex=args.wall_upper_tex,
        frame_tex=args.frame_tex,
        glass_tex=args.glass_tex,
        side_basket_tex=args.side_basket_tex,
        side_jamb_tex=args.side_jamb_tex,
        side_separator_tex=args.side_separator_tex,
        **kw,
    )


if __name__ == "__main__":
    main()
