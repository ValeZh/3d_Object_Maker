"""
Процедурная геометрия окна как единый trimesh-меш (рамка + стекло).
Оси: X — ширина, Y — глубина в стену, Z — высота. Лицевая сторона смотрит в +Y.

Параметры:
  width, height, depth — размеры (условные метры).
  profile — форма проёма: rect | arch | round.
  kind — тип/раскладка: fixed | double_hung | casement | french (если число балок = 0).
  mullions_vertical / mullions_horizontal — число вертикальных / горизонтальных импостов
    внутри проёма (>0 включает явную сетку и отключает пресеты kind).
  mullion_offset_x / mullion_offset_z — сдвиг сетки по X (влево −) и по Z.
  partial_horizontal_bars — локальные поперечины только в одном проёме (форточка):
    список (bay_index, z_frac): bay 0..nv слева направо, z_frac 0=низ проёма, 1=верх.

Обзор в Open3D (нужен пакет open3d: pip install open3d):
  python -m src.generator.procedural.procedural_window preview
Экспорт OBJ без Open3D:
  python -m src.generator.procedural.procedural_window export -o ./out

Числа и списки ниже (USER_*) можно править вручную — они подхватываются при вызове
build_window_mesh() / preview_windows_open3d() без аргументов; экспорт — python -m … procedural_window export.
"""
from __future__ import annotations

import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np
import trimesh
from PIL import Image

from src.generator.procedural.open3d_preview import (
    preview_window_obj_open3d,
    require_open3d,
    trimesh_to_open3d_mesh,
)
from src.generator.procedural.window_texture_assets import (
    ensure_window_textures,
    make_atlas_from_sources,
    resolve_texture_path,
)

Profile = Literal["rect", "arch", "round"]
Kind = Literal["fixed", "double_hung", "casement", "french"]


# ==========================================================
# ✏️ Открытые настройки — вписывайте свои значения здесь
# ==========================================================

# Окно по умолчанию (если в build_window_mesh не передать аргумент)
USER_WINDOW_MESH: dict[str, Any] = {
    "width": 1.2,
    "height": 1.5,
    "depth": 0.12,
    "profile": "rect",  # rect | arch | round
    "kind": "fixed",  # fixed | double_hung | casement | french (только если балки ниже = 0)
    "mullions_vertical": 0,  # число вертикальных балок (импостов); 0 = не задавать сетку
    "mullions_horizontal": 0,
    "mullion_offset_x": 0.0,  # сдвиг сетки вдоль X (+ вправо)
    "mullion_offset_z": 0.0,  # сдвиг горизонтальных балок вдоль Z (+ вверх)
    # Форточка только в правой секции: 2 вертикали -> проёмы 0,1,2; в проёме 2 — планка сверху
    "partial_horizontal_bars": [],  # например [(2, 0.78)] — bay справа, z снизу 0..1
}

# Превью Open3D: несколько окон в ряд (python -m …procedural_window)
USER_PREVIEW_OPEN3D: dict[str, Any] = {
    "profiles": ["rect", "arch", "round"],
    "kind": "french",
    "width": 1.15,
    "height": 1.45,
    "depth": 0.11,
    "spacing": 2.8,
    "mesh_colors_rgb": [
        [0.6, 0.7, 1.0],
        [1.0, 0.6, 0.6],
        [0.6, 1.0, 0.6],
        [1.0, 0.8, 0.4],
        [0.8, 0.5, 1.0],
        [1.0, 0.6, 0.8],
    ],
}


def _pick_profile(value: Profile | str | None, fallback: str) -> Profile:
    s = str(fallback if value is None else value).lower().strip()
    if s in ("rect", "arch", "round"):
        return s  # type: ignore[return-value]
    return "rect"


def _pick_kind(value: Kind | str | None, fallback: str) -> Kind:
    s = str(fallback if value is None else value).lower().strip()
    if s in ("fixed", "double_hung", "casement", "french"):
        return s  # type: ignore[return-value]
    return "fixed"


def _pick_nonneg_int(value: int | None, fallback: Any, cap: int = 32) -> int:
    v = int(fallback if value is None else value)
    return int(np.clip(v, 0, cap))


def _pick_float_param(value: float | None, fallback: Any) -> float:
    return float(fallback if value is None else value)


def _normalize_partial_horizontal_bars(raw: Any) -> List[Tuple[int, float]]:
    """[(bay, z_frac), ...] из списка кортежей/списков или пусто."""
    if raw is None:
        return []
    out: List[Tuple[int, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((int(item[0]), float(item[1])))
    return out


def _vertical_mullion_centers(nv: int, inner_w: float, ihw: float, ox: float) -> List[float]:
    if nv <= 0:
        return []
    step = inner_w / float(nv + 1)
    return [-ihw + (i + 1) * step + ox for i in range(nv)]


def _bay_x_intervals(nv: int, inner_w: float, ihw: float, ox: float, m: float) -> List[Tuple[float, float]]:
    """Границы проёмов по X: проём j между вертикалями, j=0..nv (всего nv+1)."""
    xs = _vertical_mullion_centers(nv, inner_w, ihw, ox)
    bays: List[Tuple[float, float]] = []
    for j in range(nv + 1):
        xl = -ihw if j == 0 else xs[j - 1] + m * 0.5
        xr = xs[j] - m * 0.5 if j < nv else ihw
        bays.append((xl, xr))
    return bays


def _partial_horizontal_bars_geometry(
    entries: List[Tuple[int, float]],
    nv: int,
    inner_w: float,
    inner_h: float,
    depth: float,
    ft: float,
    z_center: float,
    ox: float,
    fp: List[trimesh.Trimesh],
) -> None:
    """
    Горизонтальные импосты только внутри одного вертикального проёма (не на всю ширину).
    z_frac: 0 — низ внутреннего проёма, 1 — верх.
    """
    if not entries:
        return
    m = _mullion_thickness(ft, inner_w, inner_h)
    ihw, ihh = inner_w * 0.5, inner_h * 0.5
    bays = _bay_x_intervals(nv, inner_w, ihw, ox, m)
    for bay_idx, z_frac in entries:
        j = int(bay_idx)
        if j < 0 or j > nv or j >= len(bays):
            continue
        zf = float(np.clip(z_frac, 0.0, 1.0))
        xl, xr = bays[j]
        w_bay = max(xr - xl, 1e-4)
        cx = 0.5 * (xl + xr)
        z = z_center - ihh + zf * inner_h
        fp.append(_box_at(cx, 0.0, z, w_bay, depth, m))


def _mullion_thickness(ft: float, inner_w: float, inner_h: float) -> float:
    return float(max(ft * 0.85, min(inner_w, inner_h) * 0.04))


def _mullions_grid_rect(
    nv: int,
    nh: int,
    offset_x: float,
    offset_z: float,
    inner_w: float,
    inner_h: float,
    depth: float,
    ft: float,
    z_center: float,
    fp: List[trimesh.Trimesh],
) -> None:
    """Равномерно распределённые импосты в прямоугольном проёме (центр по X, z_center по Z)."""
    if nv <= 0 and nh <= 0:
        return
    m = _mullion_thickness(ft, inner_w, inner_h)
    ihw, ihh = inner_w * 0.5, inner_h * 0.5
    # Полная высота/ширина внутреннего проёма — балки упираются во внутренние грани рамы
    span_z = max(inner_h, 1e-4)
    span_x = max(inner_w, 1e-4)

    for x in _vertical_mullion_centers(nv, inner_w, ihw, offset_x):
        fp.append(_box_at(x, 0.0, z_center, m, depth, span_z))

    for j in range(nh):
        step = inner_h / float(nh + 1)
        z = z_center - ihh + (j + 1) * step + offset_z
        fp.append(_box_at(0.0, 0.0, z, span_x, depth, m))


def _mullions_kind_presets(
    kind: Kind,
    inner_w: float,
    inner_h: float,
    depth: float,
    ft: float,
    z_center: float,
    fp: List[trimesh.Trimesh],
) -> None:
    if kind == "fixed":
        return
    mullion_scale = _mullion_thickness(ft, inner_w, inner_h)

    if kind == "double_hung":
        fp.append(
            _box_at(0.0, 0.0, z_center, max(inner_w, 1e-4), depth, mullion_scale)
        )
    elif kind == "casement":
        x_off = inner_w * 0.15
        fp.append(
            _box_at(x_off, 0.0, z_center, mullion_scale, depth, max(inner_h, 1e-4))
        )
    elif kind == "french":
        fp.append(
            _box_at(0.0, 0.0, z_center, mullion_scale, depth, max(inner_h, 1e-4))
        )
        fp.append(
            _box_at(0.0, 0.0, z_center, max(inner_w, 1e-4), depth, mullion_scale)
        )


def _mullions_rect_or_arch(
    kind: Kind,
    nv: int,
    nh: int,
    offset_x: float,
    offset_z: float,
    inner_w: float,
    inner_h: float,
    depth: float,
    ft: float,
    z_center: float,
    fp: List[trimesh.Trimesh],
) -> None:
    if nv > 0 or nh > 0:
        _mullions_grid_rect(nv, nh, offset_x, offset_z, inner_w, inner_h, depth, ft, z_center, fp)
    else:
        _mullions_kind_presets(kind, inner_w, inner_h, depth, ft, z_center, fp)


def _mullions_round_explicit(
    nv: int,
    nh: int,
    offset_x: float,
    offset_z: float,
    glass_r: float,
    glass_y: float,
    depth: float,
    ft: float,
    fp: List[trimesh.Trimesh],
) -> None:
    """Импосты как хорды круга (ось X — горизонталь фасада, Z — вертикаль)."""
    m = max(ft * 0.8, glass_r * 0.06)
    for i in range(nv):
        step = 2.0 * glass_r / float(nv + 1)
        x0 = -glass_r + (i + 1) * step + offset_x
        if abs(x0) >= glass_r - m * 0.5:
            continue
        hz = 2.0 * math.sqrt(max(0.0, glass_r * glass_r - x0 * x0))
        if hz <= 1e-4:
            continue
        fp.append(_box_at(x0, glass_y, 0.0, m, depth, hz))
    for j in range(nh):
        step = 2.0 * glass_r / float(nh + 1)
        z0 = -glass_r + (j + 1) * step + offset_z
        if abs(z0) >= glass_r - m * 0.5:
            continue
        wx = 2.0 * math.sqrt(max(0.0, glass_r * glass_r - z0 * z0))
        if wx <= 1e-4:
            continue
        fp.append(_box_at(0.0, glass_y, z0, wx, depth, m))


def _mullions_round_kind(
    kind: Kind,
    glass_r: float,
    glass_y: float,
    depth: float,
    ft: float,
    fp: List[trimesh.Trimesh],
) -> None:
    if kind == "fixed":
        return
    mullion_scale = max(ft * 0.8, glass_r * 0.08)
    diam = glass_r * 2.0
    if kind == "double_hung":
        fp.append(_box_at(0.0, glass_y, 0.0, max(diam, 1e-4), depth, mullion_scale))
    elif kind == "casement":
        fp.append(
            _box_at(glass_r * 0.2, glass_y, 0.0, mullion_scale, depth, max(diam, 1e-4))
        )
    elif kind == "french":
        fp.append(_box_at(0.0, glass_y, 0.0, mullion_scale, depth, max(diam, 1e-4)))
        fp.append(_box_at(0.0, glass_y, 0.0, max(diam, 1e-4), depth, mullion_scale))


def _box_at(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[sx, sy, sz])
    m.apply_translation([cx, cy, cz])
    return m


def _merge(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    valid = [p for p in parts if p is not None and len(p.faces) > 0]
    if not valid:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(valid)


def _frame_thickness(w: float, h: float) -> float:
    t = min(w, h) * 0.07
    return float(np.clip(t, 0.02, max(0.02, min(w, h) * 0.18)))


def _extruded_radial_wedge(
    cx: float,
    cz: float,
    r_outer: float,
    r_inner: float,
    a0: float,
    a1: float,
    y0: float,
    y1: float,
) -> trimesh.Trimesh:
    """Сегмент кольца в плоскости XZ, выдавленный по Y (от y0 до y1)."""
    ox0, oz0 = cx + r_outer * math.cos(a0), cz + r_outer * math.sin(a0)
    ox1, oz1 = cx + r_outer * math.cos(a1), cz + r_outer * math.sin(a1)
    ix0, iz0 = cx + r_inner * math.cos(a0), cz + r_inner * math.sin(a0)
    ix1, iz1 = cx + r_inner * math.cos(a1), cz + r_inner * math.sin(a1)
    verts = [
        [ox0, y0, oz0],
        [ox1, y0, oz1],
        [ix1, y0, iz1],
        [ix0, y0, iz0],
        [ox0, y1, oz0],
        [ox1, y1, oz1],
        [ix1, y1, iz1],
        [ix0, y1, iz0],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _rect_frame(
    width: float,
    height: float,
    depth: float,
    ft: float,
    fp: List[trimesh.Trimesh],
) -> None:
    hw, hh = width * 0.5, height * 0.5
    fp.append(_box_at(0.0, 0.0, -hh + ft * 0.5, width, depth, ft))
    fp.append(_box_at(0.0, 0.0, hh - ft * 0.5, width, depth, ft))
    bar_h = height - 2 * ft
    fp.append(_box_at(-hw + ft * 0.5, 0.0, 0.0, ft, depth, bar_h))
    fp.append(_box_at(hw - ft * 0.5, 0.0, 0.0, ft, depth, bar_h))


def _rect_glass(inner_w: float, inner_h: float, glass_y: float, glass_t: float, gp: List[trimesh.Trimesh]) -> None:
    if inner_w <= 1e-6 or inner_h <= 1e-6:
        return
    w = max(inner_w - 0.002, 1e-4)
    h = max(inner_h - 0.002, 1e-4)
    hw, hh = w * 0.5, h * 0.5
    y_lo = glass_y - glass_t * 0.5
    y_hi = glass_y + glass_t * 0.5
    # Два листа в плоскости XZ вместо полного box: грани «крышки» box (размер X×толщина и Z×толщина)
    # давали треугольники с aspect ratio ~100+ и «улётные» спайки в OpenGL.
    pts_hi = np.array(
        [[-hw, y_hi, -hh], [hw, y_hi, -hh], [hw, y_hi, hh], [-hw, y_hi, hh]], dtype=np.float64
    )
    pts_lo = np.array(
        [[-hw, y_lo, -hh], [hw, y_lo, -hh], [hw, y_lo, hh], [-hw, y_lo, hh]], dtype=np.float64
    )
    fac_out = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    fac_in = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    gp.append(trimesh.Trimesh(vertices=pts_hi, faces=fac_out, process=False, validate=False))
    gp.append(trimesh.Trimesh(vertices=pts_lo, faces=fac_in, process=False, validate=False))


@dataclass(frozen=True)
class WindowFrameGlassParams:
    """Размеры и сетка окна из USER_WINDOW_MESH (или переданного user) плюс переопределения."""

    width: float
    height: float
    depth: float
    profile: Profile
    kind: Kind
    mullions_vertical: int
    mullions_horizontal: int
    mullion_offset_x: float
    mullion_offset_z: float
    partial_horizontal_bars: List[Tuple[int, float]]


def resolve_window_frame_glass_params(
    *,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    profile: str | None = None,
    kind: str | None = None,
    mullions_vertical: int | None = None,
    mullions_horizontal: int | None = None,
    mullion_offset_x: float | None = None,
    mullion_offset_z: float | None = None,
    partial_horizontal_bars: Any | None = None,
    user: dict[str, Any] | None = None,
) -> WindowFrameGlassParams:
    """Общий разбор полей для build_window_frame_glass_meshes (экспорт окна, стена+окно, CLI)."""
    u = user if user is not None else USER_WINDOW_MESH
    w = float(width if width is not None else u["width"])
    h = float(height if height is not None else u["height"])
    d = float(depth if depth is not None else u["depth"])
    prof = profile if profile is not None else str(u["profile"])
    knd = kind if kind is not None else str(u["kind"])
    nv = _pick_nonneg_int(mullions_vertical, u.get("mullions_vertical", 0))
    nh = _pick_nonneg_int(mullions_horizontal, u.get("mullions_horizontal", 0))
    ox = _pick_float_param(mullion_offset_x, u.get("mullion_offset_x", 0.0))
    oz = _pick_float_param(mullion_offset_z, u.get("mullion_offset_z", 0.0))
    ph_raw = partial_horizontal_bars if partial_horizontal_bars is not None else u.get("partial_horizontal_bars")
    partial_bars = _normalize_partial_horizontal_bars(ph_raw)
    pf = _pick_profile(prof, str(u.get("profile", "rect")))
    kd = _pick_kind(knd, str(u.get("kind", "fixed")))
    return WindowFrameGlassParams(
        width=w,
        height=h,
        depth=d,
        profile=pf,
        kind=kd,
        mullions_vertical=nv,
        mullions_horizontal=nh,
        mullion_offset_x=ox,
        mullion_offset_z=oz,
        partial_horizontal_bars=partial_bars,
    )


def faceted_triplanar_uv(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    UV по нормалям граней (вершины дублируются по граням): для бокса нет «уголковых»
    усреднённых нормалей и артефактов освещения/текстуры на тонком стекле.
    """
    mesh = mesh.copy()
    if len(mesh.faces) == 0:
        return mesh, np.zeros((0, 2), dtype=np.float64)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v_exp = verts[faces].reshape(-1, 3)
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fn_exp = np.repeat(fn, 3, axis=0)

    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    eps = 1e-9
    dx = max(xmax - xmin, eps)
    dy = max(ymax - ymin, eps)
    dz = max(zmax - zmin, eps)

    dom = np.argmax(np.abs(fn_exp), axis=1)
    u = np.zeros(len(v_exp), dtype=np.float64)
    vv = np.zeros(len(v_exp), dtype=np.float64)
    mx = dom == 0
    my = dom == 1
    mz = dom == 2
    u[mx] = (v_exp[mx, 1] - ymin) / dy
    vv[mx] = (v_exp[mx, 2] - zmin) / dz
    u[my] = (v_exp[my, 0] - xmin) / dx
    vv[my] = (v_exp[my, 2] - zmin) / dz
    u[mz] = (v_exp[mz, 0] - xmin) / dx
    vv[mz] = (v_exp[mz, 1] - ymin) / dy

    u = np.clip(u, 0.0, 1.0)
    vv = np.clip(vv, 0.0, 1.0)
    uv = np.stack([u, vv], axis=1)
    new_faces = np.arange(len(v_exp), dtype=np.int64).reshape(-1, 3)
    out = trimesh.Trimesh(vertices=v_exp, faces=new_faces, process=False)
    out.fix_normals()
    return out, uv


def frame_glass_atlas_uv_mesh(
    frame_mesh: trimesh.Trimesh,
    glass_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Один меш рама+стекло: левая половина U атласа — рама, правая — стекло (как в OBJ-экспорте)."""
    mf_uv, uv_f = faceted_triplanar_uv(frame_mesh)
    mg_uv, uv_g = faceted_triplanar_uv(glass_mesh)
    uv_f = np.asarray(uv_f, dtype=np.float64).copy()
    uv_g = np.asarray(uv_g, dtype=np.float64).copy()
    uv_f[:, 0] = uv_f[:, 0] * 0.5
    uv_g[:, 0] = uv_g[:, 0] * 0.5 + 0.5
    uv = np.vstack([uv_f, uv_g]) if len(uv_f) + len(uv_g) else np.zeros((0, 2))

    if len(mf_uv.faces) == 0 and len(mg_uv.faces) == 0:
        wt = trimesh.Trimesh()
    elif len(mf_uv.faces) == 0:
        wt = mg_uv
        uv = uv_g
    elif len(mg_uv.faces) == 0:
        wt = mf_uv
        uv = uv_f
    else:
        wt = trimesh.util.concatenate([mf_uv, mg_uv])
    return wt, uv


_DEFAULT_WINDOW_OBJ_EXPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "window_export"


def export_window_demo(
    out_dir: Path | None = None,
    *,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    profile: str | None = None,
    kind: str | None = None,
    mullions_vertical: int | None = None,
    mullions_horizontal: int | None = None,
    mullion_offset_x: float | None = None,
    mullion_offset_z: float | None = None,
    partial_horizontal_bars: list | None = None,
    frame_texture: str | Path | None = None,
    glass_texture: str | Path | None = None,
    atlas_half_size: int = 512,
) -> Path:
    """Экспорт window.obj + MTL + атлас (для батча и CLI export)."""
    p = resolve_window_frame_glass_params(
        width=width,
        height=height,
        depth=depth,
        profile=profile,
        kind=kind,
        mullions_vertical=mullions_vertical,
        mullions_horizontal=mullions_horizontal,
        mullion_offset_x=mullion_offset_x,
        mullion_offset_z=mullion_offset_z,
        partial_horizontal_bars=partial_horizontal_bars,
    )

    out_dir = Path(out_dir or _DEFAULT_WINDOW_OBJ_EXPORT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_name = "window_atlas.png"
    tex_path = out_dir / tex_name
    fp = resolve_texture_path(frame_texture)
    gp = resolve_texture_path(glass_texture)
    if frame_texture is not None and fp is None:
        print(f"[warn] frame_texture missing, procedural frame: {frame_texture}")
    if glass_texture is not None and gp is None:
        print(f"[warn] glass_texture missing, procedural glass: {glass_texture}")

    if fp is not None or gp is not None:
        atlas_img = make_atlas_from_sources(
            frame_path=fp,
            glass_path=gp,
            half_size=max(atlas_half_size, 64),
        )
        atlas_img.save(tex_path)
        src_note = "custom image(s) + procedural fallback if side omitted"
    else:
        tex_dir = Path(__file__).resolve().parents[3] / "data" / "textures"
        paths = ensure_window_textures(tex_dir)
        shutil.copyfile(paths["atlas"], tex_path)
        src_note = f"{paths['frame'].name} + {paths['glass'].name} (data/textures)"

    ft = _frame_thickness(p.width, p.height)
    glass_t = max(p.depth * 0.12, 0.004)

    mf, mg = build_window_frame_glass_meshes(
        width=p.width,
        height=p.height,
        depth=p.depth,
        profile=p.profile,
        kind=p.kind,
        mullions_vertical=p.mullions_vertical,
        mullions_horizontal=p.mullions_horizontal,
        mullion_offset_x=p.mullion_offset_x,
        mullion_offset_z=p.mullion_offset_z,
        partial_horizontal_bars=p.partial_horizontal_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=0.0,
    )

    work, uv = frame_glass_atlas_uv_mesh(mf, mg)

    img = Image.open(tex_path)
    work.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)

    obj_path = out_dir / "window.obj"
    work.export(str(obj_path), include_texture=True)

    mtl_path = out_dir / "material.mtl"
    if mtl_path.is_file():
        txt = mtl_path.read_text(encoding="utf-8")
        txt = txt.replace("map_Kd material_0.png", f"map_Kd {tex_name}")
        txt = txt.replace("map_Kd material_0.jpg", f"map_Kd {tex_name}")
        txt = re.sub(r"(?m)^Ka\s+.*$", "Ka 1 1 1", txt)
        txt = re.sub(r"(?m)^Kd\s+.*$", "Kd 1 1 1", txt)
        txt = re.sub(r"(?m)^Ks\s+.*$", "Ks 0 0 0", txt)
        mtl_path.write_text(txt, encoding="utf-8")

    print(f"[OK] Window export: {obj_path}")
    print(f"     Atlas (frame|glass): {tex_path}")
    print(f"     Textures: {src_note}")
    return obj_path


def build_window_mesh(
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    profile: Profile | str | None = None,
    kind: Kind | str | None = None,
    mullions_vertical: int | None = None,
    mullions_horizontal: int | None = None,
    mullion_offset_x: float | None = None,
    mullion_offset_z: float | None = None,
    partial_horizontal_bars: Any | None = None,
) -> trimesh.Trimesh:
    p = resolve_window_frame_glass_params(
        width=width,
        height=height,
        depth=depth,
        profile=profile,
        kind=kind,
        mullions_vertical=mullions_vertical,
        mullions_horizontal=mullions_horizontal,
        mullion_offset_x=mullion_offset_x,
        mullion_offset_z=mullion_offset_z,
        partial_horizontal_bars=partial_horizontal_bars,
    )
    width = max(p.width, 0.05)
    height = max(p.height, 0.05)
    depth = max(p.depth, 0.02)
    ft = _frame_thickness(width, height)
    glass_t = max(depth * 0.12, 0.004)
    glass_y = 0.0

    mf, mg = build_window_frame_glass_meshes(
        width=width,
        height=height,
        depth=depth,
        profile=p.profile,
        kind=p.kind,
        mullions_vertical=p.mullions_vertical,
        mullions_horizontal=p.mullions_horizontal,
        mullion_offset_x=p.mullion_offset_x,
        mullion_offset_z=p.mullion_offset_z,
        partial_horizontal_bars=p.partial_horizontal_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=glass_y,
        with_glass=True,
    )
    if len(mf.faces) == 0 and len(mg.faces) == 0:
        return trimesh.Trimesh()
    if len(mf.faces) == 0:
        mesh = mg
    elif len(mg.faces) == 0:
        mesh = mf
    else:
        mesh = trimesh.util.concatenate([mf, mg])
    mesh.merge_vertices(merge_tex=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def build_window_frame_glass_meshes(
    *,
    width: float,
    height: float,
    depth: float,
    profile: Profile,
    kind: Kind,
    mullions_vertical: int,
    mullions_horizontal: int,
    mullion_offset_x: float,
    mullion_offset_z: float,
    partial_horizontal_bars: List[Tuple[int, float]],
    ft: float,
    glass_t: float,
    glass_y: float,
    with_glass: bool = True,
    merge_vertices: bool = True,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Рама + импосты и стекло отдельно (для разных текстур / UV-атласа). with_glass=False — только рама."""
    fp: List[trimesh.Trimesh] = []
    gp: List[trimesh.Trimesh] = []
    nv = mullions_vertical
    nh = mullions_horizontal
    ox = mullion_offset_x
    oz = mullion_offset_z
    if profile == "round":
        _round_window_parts(
            width, height, depth, ft, glass_y, glass_t, kind, nv, nh, ox, oz, partial_horizontal_bars, fp, gp
        )
    elif profile == "arch":
        _arch_window_parts(
            width, height, depth, ft, glass_y, glass_t, kind, nv, nh, ox, oz, partial_horizontal_bars, fp, gp
        )
    else:
        _rect_window_parts(
            width,
            height,
            depth,
            ft,
            glass_y,
            glass_t,
            kind,
            nv,
            nh,
            ox,
            oz,
            partial_horizontal_bars,
            fp,
            gp,
            with_glass=with_glass,
        )

    mf = _merge(fp)
    mg = _merge(gp)
    if len(mf.vertices):
        mf.remove_unreferenced_vertices()
        mf.fix_normals()
        if merge_vertices:
            mf.merge_vertices(merge_tex=False)
    if len(mg.vertices):
        mg.remove_unreferenced_vertices()
        mg.fix_normals()
        if merge_vertices:
            mg.merge_vertices(merge_tex=False)
    return mf, mg


def _rect_window_parts(
    width: float,
    height: float,
    depth: float,
    ft: float,
    glass_y: float,
    glass_t: float,
    kind: Kind,
    nv: int,
    nh: int,
    ox: float,
    oz: float,
    partial_bars: List[Tuple[int, float]],
    fp: List[trimesh.Trimesh],
    gp: List[trimesh.Trimesh],
    *,
    with_glass: bool = True,
) -> None:
    inner_w = max(width - 2 * ft, 1e-4)
    inner_h = max(height - 2 * ft, 1e-4)
    _rect_frame(width, height, depth, ft, fp)
    if with_glass:
        _rect_glass(inner_w, inner_h, glass_y, glass_t, gp)
    _mullions_rect_or_arch(kind, nv, nh, ox, oz, inner_w, inner_h, depth, ft, 0.0, fp)
    _partial_horizontal_bars_geometry(partial_bars, nv, inner_w, inner_h, depth, ft, 0.0, ox, fp)


def _arch_window_parts(
    width: float,
    height: float,
    depth: float,
    ft: float,
    glass_y: float,
    glass_t: float,
    kind: Kind,
    nv: int,
    nh: int,
    ox: float,
    oz: float,
    partial_bars: List[Tuple[int, float]],
    fp: List[trimesh.Trimesh],
    gp: List[trimesh.Trimesh],
) -> None:
    R = float(np.clip(min(width * 0.5, height * 0.45), width * 0.2, width * 0.5))
    rect_h = max(height - R, ft * 2.5)
    z0 = -height * 0.5
    z_spring = z0 + rect_h
    rect_center_z = z0 + rect_h * 0.5

    inner_w = max(width - 2 * ft, 1e-4)
    inner_h_rect = max(rect_h - 2 * ft, 1e-4)

    sub_f: List[trimesh.Trimesh] = []
    sub_g: List[trimesh.Trimesh] = []
    _rect_frame(width, rect_h, depth, ft, sub_f)
    _rect_glass(inner_w, inner_h_rect, glass_y, glass_t, sub_g)
    for m in sub_f:
        m.apply_translation([0.0, 0.0, rect_center_z])
        fp.append(m)
    for m in sub_g:
        m.apply_translation([0.0, 0.0, rect_center_z])
        gp.append(m)

    cx, cz_center = 0.0, z_spring
    outer_r, inner_r = R, max(R - ft, 1e-4)
    n = max(12, int(36 * (outer_r / max(width, 0.1))))
    y0, y1 = -depth * 0.5, depth * 0.5
    for i in range(n):
        a0 = math.pi * (1.0 - i / n)
        a1 = math.pi * (1.0 - (i + 1) / n)
        fp.append(_extruded_radial_wedge(cx, cz_center, outer_r, inner_r, a0, a1, y0, y1))

    glass_r = max(inner_r - 0.004, 1e-4)
    gp.append(_fan_glass_arch(cx, cz_center, glass_r, glass_y, glass_t, max(n * 2, 16)))

    _mullions_rect_or_arch(kind, nv, nh, ox, oz, inner_w, inner_h_rect, depth, ft, rect_center_z, fp)
    _partial_horizontal_bars_geometry(
        partial_bars, nv, inner_w, inner_h_rect, depth, ft, rect_center_z, ox, fp
    )


def _fan_glass_arch(cx: float, cz: float, radius: float, gy: float, gt: float, segments: int) -> trimesh.Trimesh:
    verts: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []
    segs = max(int(segments), 8)
    y0, y1 = gy - gt * 0.5, gy + gt * 0.5
    for i in range(segs + 1):
        t = math.pi * i / segs
        x, z = cx + radius * math.cos(t), cz + radius * math.sin(t)
        verts.append((x, y0, z))
        verts.append((x, y1, z))
    c0 = len(verts)
    verts.append((cx, y0, cz))
    verts.append((cx, y1, cz))
    c1 = c0 + 1
    for i in range(segs):
        a0, a1 = i * 2, (i + 1) * 2
        b0, b1 = a0 + 1, a1 + 1
        faces.extend([[a0, a1, b1], [a0, b1, b0], [c0, a1, a0], [c1, b0, b1]])
    return trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)


def _round_window_parts(
    width: float,
    height: float,
    depth: float,
    ft: float,
    glass_y: float,
    glass_t: float,
    kind: Kind,
    nv: int,
    nh: int,
    ox: float,
    oz: float,
    partial_bars: List[Tuple[int, float]],
    fp: List[trimesh.Trimesh],
    gp: List[trimesh.Trimesh],
) -> None:
    R = min(width, height) * 0.5
    outer_r, inner_r = R, max(R - ft, 1e-4)
    n = max(24, int(48 * (outer_r / max(width, 0.1))))
    y0, y1 = -depth * 0.5, depth * 0.5
    for i in range(n):
        a0 = 2 * math.pi * i / n
        a1 = 2 * math.pi * (i + 1) / n
        fp.append(_extruded_radial_wedge(0.0, 0.0, outer_r, inner_r, a0, a1, y0, y1))

    glass_r = max(inner_r - 0.004, 1e-4)
    segs = n * 2
    gv: List[Tuple[float, float, float]] = []
    gf: List[List[int]] = []
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        gv.append((glass_r * math.cos(a), glass_y - glass_t * 0.5, glass_r * math.sin(a)))
        gv.append((glass_r * math.cos(a), glass_y + glass_t * 0.5, glass_r * math.sin(a)))
    o0 = len(gv)
    gv.append((0.0, glass_y - glass_t * 0.5, 0.0))
    gv.append((0.0, glass_y + glass_t * 0.5, 0.0))
    o1 = o0 + 1
    for i in range(segs):
        a0, a1 = i * 2, (i + 1) * 2
        b0, b1 = a0 + 1, a1 + 1
        gf.extend([[a0, a1, b1], [a0, b1, b0], [o0, a1, a0], [o1, b0, b1]])
    gp.append(trimesh.Trimesh(vertices=np.asarray(gv, dtype=np.float64), faces=np.asarray(gf, dtype=np.int64), process=False))

    if nv > 0 or nh > 0:
        _mullions_round_explicit(nv, nh, ox, oz, glass_r, glass_y, depth, ft, fp)
    else:
        _mullions_round_kind(kind, glass_r, glass_y, depth, ft, fp)

    if partial_bars:
        iw = 2.0 * glass_r
        _partial_horizontal_bars_geometry(partial_bars, nv, iw, iw, depth, ft, 0.0, ox, fp)


# ==========================================================
# 📌 Обзор меша в Open3D (аналог reconstruct_meshes.py)
# ==========================================================
def preview_windows_open3d(
    profiles: List[Profile] | str | None = None,
    kind: Kind | str | None = None,
    spacing: float | None = None,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    mesh_colors_rgb: List[List[float]] | None = None,
    use_user_config: bool = True,
    mullions_vertical: int | None = None,
    mullions_horizontal: int | None = None,
    mullion_offset_x: float | None = None,
    mullion_offset_z: float | None = None,
    partial_horizontal_bars: List[Tuple[int, float]] | None = None,
) -> None:
    """
    Строит несколько окон, сдвигает по X и вызывает draw_geometries.
    Если все опции None и use_user_config=True — берётся USER_PREVIEW_OPEN3D.
    Явно переданные аргументы перекрывают словарь.
    """
    o3d = require_open3d()

    cfg = USER_PREVIEW_OPEN3D if use_user_config else {}

    def _cfg(key: str, override: Any, default: Any) -> Any:
        if override is not None:
            return override
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
        return default

    if profiles is None:
        profiles = list(_cfg("profiles", None, ["rect", "arch", "round"]))
    kind_eff = _pick_kind(kind, str(_cfg("kind", None, "french")))
    spacing_f = float(_cfg("spacing", spacing, 2.8))
    w = float(_cfg("width", width, USER_WINDOW_MESH["width"]))
    h = float(_cfg("height", height, USER_WINDOW_MESH["height"]))
    d = float(_cfg("depth", depth, USER_WINDOW_MESH["depth"]))

    colors = mesh_colors_rgb
    if colors is None:
        colors = cfg.get("mesh_colors_rgb") if isinstance(cfg, dict) else None
    if not colors:
        colors = [
            [0.6, 0.7, 1.0],
            [1.0, 0.6, 0.6],
            [0.6, 1.0, 0.6],
            [1.0, 0.8, 0.4],
            [0.8, 0.5, 1.0],
            [1.0, 0.6, 0.8],
        ]

    meshes_o3d = []

    for i, prof in enumerate(profiles):
        prof_eff = _pick_profile(prof, "rect")
        print(f"\n[{i}] Процедурное окно: profile={prof_eff!r} kind={kind_eff!r}")

        tm = build_window_mesh(
            width=w,
            height=h,
            depth=d,
            profile=prof_eff,
            kind=kind_eff,
            mullions_vertical=mullions_vertical,
            mullions_horizontal=mullions_horizontal,
            mullion_offset_x=mullion_offset_x,
            mullion_offset_z=mullion_offset_z,
            partial_horizontal_bars=partial_horizontal_bars,
        )
        print(f"    vertices={len(tm.vertices)}  faces={len(tm.faces)}")

        g = trimesh_to_open3d_mesh(tm, colors[i % len(colors)])
        g.translate([i * spacing_f, 0.0, 0.0])
        meshes_o3d.append(g)

    o3d.visualization.draw_geometries(meshes_o3d)


def _add_window_build_args(p: Any) -> None:
    """Общие аргументы для preview / export (значения по умолчанию — из USER_WINDOW_MESH)."""
    p.add_argument("--width", type=float, default=None, help="Ширина окна")
    p.add_argument("--height", type=float, default=None, help="Высота окна")
    p.add_argument("--depth", type=float, default=None, help="Глубина (толщина в стену)")
    p.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=("rect", "arch", "round"),
        help="Форма проёма",
    )
    p.add_argument(
        "--kind",
        type=str,
        default=None,
        choices=("fixed", "double_hung", "casement", "french"),
        help="Тип (если импосты 0 — пресеты kind)",
    )
    p.add_argument("--mullions-vertical", type=int, default=None, metavar="N", dest="mullions_vertical")
    p.add_argument("--mullions-horizontal", type=int, default=None, metavar="N", dest="mullions_horizontal")
    p.add_argument("--mullion-offset-x", type=float, default=None, dest="mullion_offset_x")
    p.add_argument("--mullion-offset-z", type=float, default=None, dest="mullion_offset_z")
    p.add_argument(
        "--partial-h",
        action="append",
        default=None,
        metavar="BAY:ZFRAC",
        dest="partial_h",
        help="Форточка: горизонталь только в проёме BAY (0 слева), ZFRAC снизу 0..1. Повторите флаг.",
    )


def _parse_partial_h_tokens(tokens: List[str]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for t in tokens:
        t = t.strip()
        if ":" in t:
            a, b = t.split(":", 1)
        elif "," in t:
            a, b = t.split(",", 1)
        else:
            raise ValueError(f"--partial-h ожидает BAY:ZFRAC или BAY,ZFRAC, получено: {t!r}")
        out.append((int(a.strip()), float(b.strip())))
    return out


def _build_arg_parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(
        description="Процедурное окно: превью Open3D или экспорт OBJ+MTL+текстура.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m src.generator.procedural.procedural_window preview
  python -m src.generator.procedural.procedural_window preview --profile arch --kind french
  python -m src.generator.procedural.procedural_window preview --profiles rect round --spacing 3.2
  python -m src.generator.procedural.procedural_window export -o ./out --width 1.4 --mullions-vertical 2
  python -m src.generator.procedural.procedural_window export --mullions-vertical 2 --partial-h 2:0.78 -o ./out
  python -m src.generator.procedural.procedural_window export --frame-tex wood.jpg --glass-tex frosted.png -o ./out
""".strip(),
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser(
        "preview",
        help="Интерактивный просмотр (Open3D; нужен: pip install open3d)",
    )
    _add_window_build_args(pr)
    pr.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        metavar="P",
        choices=("rect", "arch", "round"),
        help="Несколько форм в ряд (по умолчанию из USER_PREVIEW_OPEN3D)",
    )
    pr.add_argument("--spacing", type=float, default=None, help="Расстояние между окнами по X")
    pr.add_argument(
        "--no-user-config",
        action="store_true",
        help="Не подмешивать USER_PREVIEW_OPEN3D (только явные флаги и дефолты)",
    )
    pr.set_defaults(_handler=_cli_preview)

    ex = sub.add_parser("export", help="Сохранить window.obj + material.mtl + текстуру")
    _add_window_build_args(ex)
    ex.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help="Папка вывода (по умолчанию: data/window_export)",
    )
    ex.add_argument(
        "--frame-tex",
        type=str,
        default=None,
        metavar="PATH",
        dest="frame_texture",
        help="Изображение текстуры рамы (jpg/png); без флага — процедурная",
    )
    ex.add_argument(
        "--glass-tex",
        type=str,
        default=None,
        metavar="PATH",
        dest="glass_texture",
        help="Изображение текстуры стекла",
    )
    ex.add_argument(
        "--texture-size",
        type=int,
        default=512,
        metavar="N",
        dest="texture_half_size",
        help="Сторона квадрата половины атласа при сборке из файлов",
    )
    ex.add_argument(
        "--no-view",
        action="store_true",
        help="Не открывать Open3D после экспорта",
    )
    ex.set_defaults(_handler=_cli_export)

    return p


def _cli_preview(args: Any) -> None:
    profiles = args.profiles
    if profiles is None and args.profile is not None:
        profiles = [args.profile]

    partial_kw: List[Tuple[int, float]] | None = None
    if args.partial_h is not None:
        partial_kw = _parse_partial_h_tokens(args.partial_h)

    preview_windows_open3d(
        profiles=profiles,
        kind=args.kind,
        spacing=args.spacing,
        width=args.width,
        height=args.height,
        depth=args.depth,
        use_user_config=not args.no_user_config,
        mullions_vertical=args.mullions_vertical,
        mullions_horizontal=args.mullions_horizontal,
        mullion_offset_x=args.mullion_offset_x,
        mullion_offset_z=args.mullion_offset_z,
        partial_horizontal_bars=partial_kw,
    )


def _cli_export(args: Any) -> None:
    partial_kw: List[Tuple[int, float]] | None = None
    if args.partial_h is not None:
        partial_kw = _parse_partial_h_tokens(args.partial_h)

    out = Path(args.output).resolve() if args.output else None
    obj_path = export_window_demo(
        out,
        width=args.width,
        height=args.height,
        depth=args.depth,
        profile=args.profile,
        kind=args.kind,
        mullions_vertical=args.mullions_vertical,
        mullions_horizontal=args.mullions_horizontal,
        mullion_offset_x=args.mullion_offset_x,
        mullion_offset_z=args.mullion_offset_z,
        partial_horizontal_bars=partial_kw,
        frame_texture=args.frame_texture,
        glass_texture=args.glass_texture,
        atlas_half_size=max(getattr(args, "texture_half_size", 512), 64),
    )
    if not args.no_view:
        preview_window_obj_open3d(obj_path)


def main(argv: List[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args._handler(args)


if __name__ == "__main__":
    main()
