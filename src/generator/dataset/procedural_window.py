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
  python -m src.generator.dataset.procedural_window preview
Экспорт OBJ без Open3D:
  python -m src.generator.dataset.procedural_window export -o ./out

Числа и списки ниже (USER_*) можно править вручную — они подхватываются при вызове
build_window_mesh() / preview_windows_open3d() без аргументов и в run_window_demo.
"""
from __future__ import annotations

import math
import sys
from typing import Any, List, Literal, Tuple

import numpy as np
import trimesh

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
    gp.append(
        _box_at(
            0.0,
            glass_y,
            0.0,
            max(inner_w - 0.002, 1e-4),
            glass_t,
            max(inner_h - 0.002, 1e-4),
        )
    )


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
    u = USER_WINDOW_MESH
    width = max(float(width if width is not None else u["width"]), 0.05)
    height = max(float(height if height is not None else u["height"]), 0.05)
    depth = max(float(depth if depth is not None else u["depth"]), 0.02)
    profile = _pick_profile(profile, str(u.get("profile", "rect")))
    kind = _pick_kind(kind, str(u.get("kind", "fixed")))
    nv = _pick_nonneg_int(mullions_vertical, u.get("mullions_vertical", 0))
    nh = _pick_nonneg_int(mullions_horizontal, u.get("mullions_horizontal", 0))
    ox = _pick_float_param(mullion_offset_x, u.get("mullion_offset_x", 0.0))
    oz = _pick_float_param(mullion_offset_z, u.get("mullion_offset_z", 0.0))
    ph_raw = partial_horizontal_bars if partial_horizontal_bars is not None else u.get("partial_horizontal_bars")
    partial_bars = _normalize_partial_horizontal_bars(ph_raw)

    ft = _frame_thickness(width, height)
    glass_t = max(depth * 0.12, 0.004)
    glass_y = 0.0

    mf, mg = build_window_frame_glass_meshes(
        width=width,
        height=height,
        depth=depth,
        profile=profile,
        kind=kind,
        mullions_vertical=nv,
        mullions_horizontal=nh,
        mullion_offset_x=ox,
        mullion_offset_z=oz,
        partial_horizontal_bars=partial_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=glass_y,
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
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Рама + импосты и стекло отдельно (для разных текстур / UV-атласа)."""
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
            width, height, depth, ft, glass_y, glass_t, kind, nv, nh, ox, oz, partial_horizontal_bars, fp, gp
        )

    mf = _merge(fp)
    mg = _merge(gp)
    if len(mf.vertices):
        mf.remove_unreferenced_vertices()
        mf.fix_normals()
        mf.merge_vertices(merge_tex=False)
    if len(mg.vertices):
        mg.remove_unreferenced_vertices()
        mg.fix_normals()
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
) -> None:
    inner_w = max(width - 2 * ft, 1e-4)
    inner_h = max(height - 2 * ft, 1e-4)
    _rect_frame(width, height, depth, ft, fp)
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
def _require_open3d():
    """Импорт open3d с понятным сообщением, если пакет не установлен."""
    try:
        import open3d as o3d

        return o3d
    except ModuleNotFoundError:
        print(
            "Команда preview требует пакет open3d.\n"
            "  pip install open3d\n"
            "Без Open3D можно только экспортировать меш:\n"
            "  python -m src.generator.dataset.procedural_window export -o ./out",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def trimesh_to_open3d_mesh(
    mesh: trimesh.Trimesh,
    color_rgb: Tuple[float, float, float] | List[float] | None = None,
):
    """Конвертация trimesh → o3d.TriangleMesh с нормалями и цветом."""
    o3d = _require_open3d()

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(v)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d_mesh.compute_vertex_normals()
    if color_rgb is None:
        color_rgb = (0.65, 0.72, 1.0)
    o3d_mesh.paint_uniform_color(list(color_rgb))
    return o3d_mesh


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
    o3d = _require_open3d()

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
  python -m src.generator.dataset.procedural_window preview
  python -m src.generator.dataset.procedural_window preview --profile arch --kind french
  python -m src.generator.dataset.procedural_window preview --profiles rect round --spacing 3.2
  python -m src.generator.dataset.procedural_window export -o ./out --width 1.4 --mullions-vertical 2
  python -m src.generator.dataset.procedural_window export --mullions-vertical 2 --partial-h 2:0.78 -o ./out
  python -m src.generator.dataset.procedural_window export --frame-tex wood.jpg --glass-tex frosted.png -o ./out
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
    from pathlib import Path

    from src.generator.dataset.run_window_demo import export_window_demo

    partial_kw: List[Tuple[int, float]] | None = None
    if args.partial_h is not None:
        partial_kw = _parse_partial_h_tokens(args.partial_h)

    out = Path(args.output).resolve() if args.output else None
    export_window_demo(
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


def main(argv: List[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args._handler(args)


if __name__ == "__main__":
    main()
