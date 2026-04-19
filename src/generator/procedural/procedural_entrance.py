"""
Процедурный подъезд: два режима entrance_style.

  canopy — козырёк: стена y=0, глубина в +Y наружу, плита, опционально бока/перегородки/столб.

  niche — углублённая ниша (тамбур): +Y внутрь от проёма, потолок, два борта, задняя стена с двустворчатой
  дверью (рама, средняя перекладина, стёкла), пол, ступенька, цоколь.

Оси: X — вдоль фасада, Y — глубина, Z — вверх.

Запуск:
  python -m src.generator.procedural.procedural_entrance export --style niche -o data/podezd_nisha
  python -m src.generator.procedural.procedural_entrance export --style canopy ...
  Экспорт с атласом (текстуры стен / крыши / двери): procedural_entrance_textured — см. модуль
  src.generator.procedural.procedural_entrance_textured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import trimesh

from src.generator.procedural.open3d_preview import preview_entrance_obj_open3d
from src.generator.procedural.procedural_door import (
    build_french_double_door_parts,
    build_simple_door_slab,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


USER_ENTRANCE: dict[str, Any] = {
    "entrance_style": "canopy",
    # --- niche (углублённая ниша) ---
    "niche_clear_height": 2.52,
    "niche_ceiling_thickness": 0.16,
    "niche_floor_z": 0.13,
    "step_front_depth": 0.36,
    "plinth_height": 0.13,
    "double_door": True,
    "niche_door_u0": 0.24,
    "niche_door_u1": 0.76,
    "niche_door_z_bottom": None,
    "niche_door_z_top": None,
    "door_frame_depth": 0.06,
    "door_frame_width": 0.09,
    "door_leaf_gap": 0.025,
    "door_midrail_z_frac": 0.58,
    "door_recess_y": 0.045,
    # --- canopy ---
    "width": 3.6,
    "depth": 1.75,
    "canopy_thickness": 0.2,
    "canopy_z_bottom": 2.45,
    "platform_height": 0.16,
    "platform_depth": 0.4,
    "platform_width": None,
    "has_left_wall": False,
    "has_right_wall": False,
    "wall_thickness": 0.2,
    "partition_thickness": 0.22,
    "partitions_x": [0.0],
    "right_support_pole": True,
    "pole_radius": 0.055,
    "pole_front_inset": 0.12,
    "pole_side_inset": 0.12,
    "doors": [
        {"u0": 0.06, "u1": 0.44, "z_bottom": 0.12, "z_top": 2.05},
        {"u0": 0.56, "u1": 0.94, "z_bottom": 0.12, "z_top": 2.05},
    ],
    "door_plane_y": 0.04,
    "door_plane_thickness": 0.03,
}

# Подмешивается при entrance_style=niche (kwargs перекрывают).
ENTRANCE_NICHE_PRESET: dict[str, Any] = {
    "width": 2.14,
    "depth": 1.28,
    "partitions_x": [],
    "has_left_wall": False,
    "has_right_wall": False,
    "right_support_pole": False,
}


def _quad_xyz(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    *,
    flip: bool = False,
) -> trimesh.Trimesh:
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in (bl, br, tr, tl)]
    f = [[0, 1, 2], [0, 2, 3]]
    if flip:
        f = [[0, 2, 1], [0, 3, 2]]
    return trimesh.Trimesh(vertices=[a, b, c, d], faces=f, process=False)


def _split_axis_aligned_rect(
    r: Tuple[float, float, float, float],
    h: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
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


def _rects_subtract_hole(
    rects: List[Tuple[float, float, float, float]],
    h: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for r in rects:
        out.extend(_split_axis_aligned_rect(r, h))
    return out


def _back_wall_patches_us(
    doors: List[dict],
    z_lo: float,
    z_hi: float,
) -> List[Tuple[float, float, float, float]]:
    span = max(z_hi - z_lo, 1e-9)
    rects: List[Tuple[float, float, float, float]] = [(0.0, 1.0, 0.0, 1.0)]
    for d in doors:
        u0 = float(np.clip(d["u0"], 0.0, 1.0))
        u1 = float(np.clip(d["u1"], 0.0, 1.0))
        zb, zt = float(d["z_bottom"]), float(d["z_top"])
        if u1 <= u0 + 1e-4 or zt <= zb + 1e-4:
            continue
        za = max(zb, z_lo)
        zb2 = min(zt, z_hi)
        if zb2 <= za + 1e-4:
            continue
        s0 = (za - z_lo) / span
        s1 = (zb2 - z_lo) / span
        rects = _rects_subtract_hole(rects, (u0, u1, s0, s1))
    return rects


def _point_back_wall(u: float, s: float, W: float, z_lo: float, z_hi: float) -> np.ndarray:
    x = -0.5 * W + float(u) * W
    z = z_lo + float(s) * (z_hi - z_lo)
    return np.array([x, 0.0, z], dtype=np.float64)


def _normalize_doors(raw: Any) -> List[dict]:
    if not raw or not isinstance(raw, list):
        return []
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                {
                    "u0": float(item["u0"]),
                    "u1": float(item["u1"]),
                    "z_bottom": float(item["z_bottom"]),
                    "z_top": float(item["z_top"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _band_patches_us(
    wall_z_lo: float,
    wall_z_hi: float,
    band_z_lo: float,
    band_z_hi: float,
    door_specs: List[dict],
) -> List[Tuple[float, float, float, float]]:
    span = max(wall_z_hi - wall_z_lo, 1e-9)
    z0 = max(band_z_lo, wall_z_lo)
    z1 = min(band_z_hi, wall_z_hi)
    if z1 <= z0 + 1e-6:
        return []
    s0 = (z0 - wall_z_lo) / span
    s1 = (z1 - wall_z_lo) / span
    rects: List[Tuple[float, float, float, float]] = [(0.0, 1.0, s0, s1)]
    for d in door_specs:
        u0 = float(np.clip(d["u0"], 0.0, 1.0))
        u1 = float(np.clip(d["u1"], 0.0, 1.0))
        zb, zt = float(d["z_bottom"]), float(d["z_top"])
        za = max(zb, z0)
        zb2 = min(zt, z1)
        if u1 <= u0 + 1e-4 or zb2 <= za + 1e-4:
            continue
        hs0 = (za - wall_z_lo) / span
        hs1 = (zb2 - wall_z_lo) / span
        rects = _rects_subtract_hole(rects, (u0, u1, hs0, hs1))
    return rects


def _point_niche_back(
    u: float, s: float, W: float, z_lo: float, z_hi: float, y_back: float
) -> np.ndarray:
    x = -0.5 * W + float(u) * W
    z = z_lo + float(s) * (z_hi - z_lo)
    return np.array([x, y_back, z], dtype=np.float64)


def build_niche_entrance_meshes(
    *,
    width: float,
    depth: float,
    niche_clear_height: float,
    niche_ceiling_thickness: float,
    niche_floor_z: float,
    step_front_depth: float,
    plinth_height: float,
    double_door: bool,
    niche_door_u0: float,
    niche_door_u1: float,
    niche_door_z_bottom: Optional[float],
    niche_door_z_top: Optional[float],
    door_frame_depth: float,
    door_frame_width: float,
    door_leaf_gap: float,
    door_midrail_z_frac: float,
    door_recess_y: float,
) -> List[Tuple[str, trimesh.Trimesh]]:
    W = max(float(width), 0.4)
    D = max(float(depth), 0.35)
    zf = max(float(niche_floor_z), 0.06)
    ch = max(float(niche_clear_height), 1.6)
    ct = max(float(niche_ceiling_thickness), 0.08)
    z_ceil_bot = zf + ch
    z_ceil_top = z_ceil_bot + ct
    ph = max(float(plinth_height), 0.05)
    z_pl0 = zf
    z_pl1 = zf + ph
    wall_z_lo = zf
    wall_z_hi = z_ceil_bot
    if wall_z_hi <= z_pl1 + 0.08:
        wall_z_hi = z_pl1 + 0.08

    u0 = float(np.clip(niche_door_u0, 0.0, 1.0))
    u1 = float(np.clip(niche_door_u1, 0.0, 1.0))
    if u1 <= u0 + 0.05:
        u0, u1 = 0.28, 0.72
    dz0 = niche_door_z_bottom
    dz1 = niche_door_z_top
    z_d0 = float(dz0) if dz0 is not None else zf + 0.04
    z_d1 = float(dz1) if dz1 is not None else z_ceil_bot - 0.1
    if z_d1 <= z_d0 + 0.2:
        z_d1 = z_d0 + 2.0

    door_spec = [{"u0": u0, "u1": u1, "z_bottom": z_d0, "z_top": z_d1}]
    parts: List[Tuple[str, trimesh.Trimesh]] = []

    # Ступенька снаружи (y < 0)
    sd = max(float(step_front_depth), 0.12)
    st = max(0.07, min(zf * 0.85, zf - 0.02))
    tread = trimesh.creation.box(extents=[W, sd, st])
    tread.apply_translation([0.0, -sd * 0.5, zf - st * 0.5])
    parts.append(("step_tread", tread))
    rh = max(zf - st, 0.06)
    riser = trimesh.creation.box(extents=[W, 0.08, rh])
    riser.apply_translation([0.0, -0.04, rh * 0.5])
    parts.append(("step_riser", riser))

    # Пол ниши
    fl = trimesh.creation.box(extents=[W, D, 0.06])
    fl.apply_translation([0.0, D * 0.5, zf - 0.03])
    parts.append(("floor", fl))

    # Потолок-плита
    ceil = trimesh.creation.box(extents=[W, D, ct])
    ceil.apply_translation([0.0, D * 0.5, z_ceil_bot + ct * 0.5])
    parts.append(("ceiling", ceil))

    yb = D
    # Задняя стена: цоколь + верх, с проёмом
    for band_lo, band_hi, name in (
        (z_pl0, z_pl1, "wall_plinth_back"),
        (z_pl1, wall_z_hi, "wall_upper_back"),
    ):
        for ua, ub, sa, sb in _band_patches_us(wall_z_lo, wall_z_hi, band_lo, band_hi, door_spec):
            if ub <= ua + 1e-6 or sb <= sa + 1e-6:
                continue
            bl = _point_niche_back(ua, sa, W, wall_z_lo, wall_z_hi, yb)
            br = _point_niche_back(ub, sa, W, wall_z_lo, wall_z_hi, yb)
            tr = _point_niche_back(ub, sb, W, wall_z_lo, wall_z_hi, yb)
            tl = _point_niche_back(ua, sb, W, wall_z_lo, wall_z_hi, yb)
            parts.append((name, _quad_xyz(bl, br, tr, tl, flip=False)))

    lx = -0.5 * W
    rx = 0.5 * W
    for band_lo, band_hi, name in (
        (z_pl0, z_pl1, "wall_plinth_left"),
        (z_pl1, wall_z_hi, "wall_upper_left"),
    ):
        bl = np.array([lx, 0.0, band_lo], dtype=np.float64)
        br = np.array([lx, D, band_lo], dtype=np.float64)
        tr = np.array([lx, D, band_hi], dtype=np.float64)
        tl = np.array([lx, 0.0, band_hi], dtype=np.float64)
        parts.append((name, _quad_xyz(bl, br, tr, tl, flip=False)))
    for band_lo, band_hi, name in (
        (z_pl0, z_pl1, "wall_plinth_right"),
        (z_pl1, wall_z_hi, "wall_upper_right"),
    ):
        bl = np.array([rx, D, band_lo], dtype=np.float64)
        br = np.array([rx, 0.0, band_lo], dtype=np.float64)
        tr = np.array([rx, 0.0, band_hi], dtype=np.float64)
        tl = np.array([rx, D, band_hi], dtype=np.float64)
        parts.append((name, _quad_xyz(bl, br, tr, tl, flip=False)))

    y_outer = max(D - float(door_recess_y), D * 0.5)
    dx0 = -0.5 * W + u0 * W
    dx1 = -0.5 * W + u1 * W
    if double_door:
        parts.extend(
            build_french_double_door_parts(
                x0=dx0,
                x1=dx1,
                z0=z_d0,
                z1=z_d1,
                y_outer=y_outer,
                frame_width=door_frame_width,
                frame_depth=door_frame_depth,
                leaf_gap=door_leaf_gap,
                midrail_z_frac=door_midrail_z_frac,
                niche_depth=D,
            )
        )
    else:
        parts.extend(
            build_simple_door_slab(
                x0=dx0,
                x1=dx1,
                z0=z_d0,
                z1=z_d1,
                y_outer=y_outer,
                depth=max(door_frame_depth, 0.04),
                niche_depth=D,
            )
        )

    return parts


def build_entrance_meshes(
    *,
    width: float,
    depth: float,
    canopy_thickness: float,
    canopy_z_bottom: float,
    platform_height: float,
    platform_depth: float,
    platform_width: Optional[float],
    has_left_wall: bool,
    has_right_wall: bool,
    wall_thickness: float,
    partition_thickness: float,
    partitions_x: List[float],
    right_support_pole: bool,
    pole_radius: float,
    pole_front_inset: float,
    pole_side_inset: float,
    doors: List[dict],
    door_plane_y: float,
    door_plane_thickness: float,
) -> List[Tuple[str, trimesh.Trimesh]]:
    W = max(float(width), 0.3)
    D = max(float(depth), 0.2)
    t_can = max(float(canopy_thickness), 0.06)
    z_cb = max(float(canopy_z_bottom), 0.5)
    z_ct = z_cb + t_can
    ph = max(float(platform_height), 0.02)
    pd = max(float(platform_depth), 0.05)
    pw = float(platform_width) if platform_width is not None else W + 0.2
    pw = max(pw, W)
    wt = max(float(wall_thickness), 0.05)
    pt = max(float(partition_thickness), 0.06)
    z_wall_lo = ph
    z_wall_hi = z_cb
    if z_wall_hi <= z_wall_lo + 0.05:
        z_wall_hi = z_wall_lo + 0.05

    door_list = _normalize_doors(doors)
    parts: List[Tuple[str, trimesh.Trimesh]] = []

    # --- площадка (ступень) ---
    plat = trimesh.creation.box(extents=[pw, pd, ph])
    plat.apply_translation([0.0, pd * 0.5, ph * 0.5])
    parts.append(("platform", plat))

    # --- козырёк (плита) ---
    canopy = trimesh.creation.box(extents=[W, D, t_can])
    canopy.apply_translation([0.0, D * 0.5, z_cb + t_can * 0.5])
    parts.append(("canopy", canopy))

    # --- задняя стена (куски с вырезами под двери), нормаль +Y ---
    for ua, ub, sa, sb in _back_wall_patches_us(door_list, z_wall_lo, z_wall_hi):
        if ub <= ua + 1e-6 or sb <= sa + 1e-6:
            continue
        bl = _point_back_wall(ua, sa, W, z_wall_lo, z_wall_hi)
        br = _point_back_wall(ub, sa, W, z_wall_lo, z_wall_hi)
        tr = _point_back_wall(ub, sb, W, z_wall_lo, z_wall_hi)
        tl = _point_back_wall(ua, sb, W, z_wall_lo, z_wall_hi)
        parts.append(("back_wall", _quad_xyz(bl, br, tr, tl, flip=False)))

    # --- плоскости «дверей» (тёмные прямоугольники чуть внутри проёма) ---
    dy = max(float(door_plane_y), 0.01)
    dt = max(float(door_plane_thickness), 0.008)
    for i, d in enumerate(door_list):
        u0, u1 = float(d["u0"]), float(d["u1"])
        zb, zt = float(d["z_bottom"]), float(d["z_top"])
        if u1 <= u0 + 1e-4 or zt <= zb + 1e-4:
            continue
        x0 = -0.5 * W + u0 * W
        x1 = -0.5 * W + u1 * W
        cx = 0.5 * (x0 + x1)
        cz = 0.5 * (zb + zt)
        ww = max(x1 - x0, 0.05)
        hh = max(zt - zb, 0.05)
        door = trimesh.creation.box(extents=[ww, dt, hh])
        door.apply_translation([cx, dy + dt * 0.5, cz])
        parts.append((f"door_{i}", door))

    # --- боковые стены (внутрь объёма нормаль ±X) ---
    if has_left_wall:
        lx = -0.5 * W
        bl = np.array([lx, 0.0, z_wall_lo], dtype=np.float64)
        br = np.array([lx, D, z_wall_lo], dtype=np.float64)
        tr = np.array([lx, D, z_wall_hi], dtype=np.float64)
        tl = np.array([lx, 0.0, z_wall_hi], dtype=np.float64)
        parts.append(("side_left", _quad_xyz(bl, br, tr, tl, flip=False)))
    if has_right_wall:
        rx = 0.5 * W
        bl = np.array([rx, D, z_wall_lo], dtype=np.float64)
        br = np.array([rx, 0.0, z_wall_lo], dtype=np.float64)
        tr = np.array([rx, 0.0, z_wall_hi], dtype=np.float64)
        tl = np.array([rx, D, z_wall_hi], dtype=np.float64)
        parts.append(("side_right", _quad_xyz(bl, br, tr, tl, flip=False)))

    # --- перегородки (параллель YZ, центр на partitions_x) ---
    for pi, px in enumerate(partitions_x):
        x_c = float(px)
        if abs(x_c) > 0.5 * W - 0.02:
            continue
        part = trimesh.creation.box(extents=[pt, D, z_wall_hi - z_wall_lo])
        part.apply_translation([x_c, D * 0.5, 0.5 * (z_wall_lo + z_wall_hi)])
        parts.append((f"partition_{pi}", part))

    # --- столб у переднего правого угла (если нет правой стены) ---
    if right_support_pole and not has_right_wall:
        pr = max(float(pole_radius), 0.02)
        h_pole = max(z_wall_hi - z_wall_lo, 0.1)
        cyl = trimesh.creation.cylinder(radius=pr, height=h_pole, sections=20)
        px = 0.5 * W - float(pole_side_inset) - pr
        py = D - float(pole_front_inset) - pr
        px = float(np.clip(px, -0.5 * W + pr, 0.5 * W - pr))
        py = float(np.clip(py, pr, D - pr))
        cyl.apply_translation([px, py, z_wall_lo + h_pole * 0.5])
        parts.append(("pole", cyl))

    return parts


def export_entrance(
    out_dir: Path | None = None,
    *,
    no_view: bool = False,
    **kwargs: Any,
) -> Path:
    out_dir = out_dir or (_REPO_ROOT / "data" / "entrance_export")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = {**USER_ENTRANCE, **kwargs}
    if str(merged.get("entrance_style", "canopy")).lower() == "niche":
        merged = {**USER_ENTRANCE, **ENTRANCE_NICHE_PRESET, **kwargs}
    p = merged

    style = str(p.get("entrance_style", "canopy")).lower()
    if style == "niche":
        dz0, dz1 = p.get("niche_door_z_bottom"), p.get("niche_door_z_top")
        parts = build_niche_entrance_meshes(
            width=float(p["width"]),
            depth=float(p["depth"]),
            niche_clear_height=float(p.get("niche_clear_height", 2.5)),
            niche_ceiling_thickness=float(p.get("niche_ceiling_thickness", 0.16)),
            niche_floor_z=float(p.get("niche_floor_z", 0.13)),
            step_front_depth=float(p.get("step_front_depth", 0.35)),
            plinth_height=float(p.get("plinth_height", 0.13)),
            double_door=bool(p.get("double_door", True)),
            niche_door_u0=float(p.get("niche_door_u0", 0.24)),
            niche_door_u1=float(p.get("niche_door_u1", 0.76)),
            niche_door_z_bottom=None if dz0 is None else float(dz0),
            niche_door_z_top=None if dz1 is None else float(dz1),
            door_frame_depth=float(p.get("door_frame_depth", 0.06)),
            door_frame_width=float(p.get("door_frame_width", 0.09)),
            door_leaf_gap=float(p.get("door_leaf_gap", 0.025)),
            door_midrail_z_frac=float(p.get("door_midrail_z_frac", 0.58)),
            door_recess_y=float(p.get("door_recess_y", 0.045)),
        )
    else:
        pw_raw = p.get("platform_width")
        parts = build_entrance_meshes(
            width=float(p["width"]),
            depth=float(p["depth"]),
            canopy_thickness=float(p["canopy_thickness"]),
            canopy_z_bottom=float(p["canopy_z_bottom"]),
            platform_height=float(p["platform_height"]),
            platform_depth=float(p["platform_depth"]),
            platform_width=None if pw_raw is None else float(pw_raw),
            has_left_wall=bool(p.get("has_left_wall", False)),
            has_right_wall=bool(p.get("has_right_wall", False)),
            wall_thickness=float(p.get("wall_thickness", 0.2)),
            partition_thickness=float(p.get("partition_thickness", 0.22)),
            partitions_x=list(p.get("partitions_x") or []),
            right_support_pole=bool(p.get("right_support_pole", True)),
            pole_radius=float(p.get("pole_radius", 0.055)),
            pole_front_inset=float(p.get("pole_front_inset", 0.12)),
            pole_side_inset=float(p.get("pole_side_inset", 0.12)),
            doors=p.get("doors") or [],
            door_plane_y=float(p.get("door_plane_y", 0.04)),
            door_plane_thickness=float(p.get("door_plane_thickness", 0.03)),
        )

    meshes = [m for _, m in parts if len(m.vertices) and len(m.faces)]
    if not meshes:
        raise RuntimeError("entrance: empty mesh")
    combined = trimesh.util.concatenate(meshes)
    obj_path = out_dir / "entrance.obj"
    combined.export(str(obj_path))
    print(f"[OK] Entrance export: {obj_path}")

    if not no_view:
        preview_entrance_obj_open3d(obj_path, niche=(style == "niche"))
    return obj_path


def _parse_door_cli(s: str) -> dict:
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if len(toks) != 4:
        raise argparse.ArgumentTypeError("door: нужно u0,u1,z_bottom,z_top")
    return {
        "u0": float(toks[0]),
        "u1": float(toks[1]),
        "z_bottom": float(toks[2]),
        "z_top": float(toks[3]),
    }


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Экспорт процедурного подъезда (OBJ): canopy или niche.")
    ap.add_argument("command", nargs="?", default="export", choices=("export",), help="Команда")
    ap.add_argument("-o", "--output", type=str, default=None, help="Папка вывода")
    ap.add_argument("--no-view", action="store_true", help="Не открывать Open3D")
    ap.add_argument(
        "--style",
        type=str,
        default=None,
        choices=("canopy", "niche"),
        help="canopy — козырёк; niche — углублённая ниша с потолком и двустворчатой дверью",
    )
    ap.add_argument("--clear-height", type=float, default=None, help="(niche) высота под потолком, м")
    ap.add_argument("--niche-floor-z", type=float, default=None, help="(niche) уровень пола ниши по Z")
    ap.add_argument("--plinth-height", type=float, default=None, help="(niche) высота цоколя")
    ap.add_argument("--step-depth", type=float, default=None, help="(niche) глубина наружной ступеньки по Y")
    ap.add_argument("--ceiling-thickness", type=float, default=None, help="(niche) толщина плиты потолка")
    ap.add_argument("--no-double-door", action="store_true", help="(niche) простая плоскость вместо рамы/стёкол")
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--canopy-thickness", type=float, default=None)
    ap.add_argument("--canopy-z-bottom", type=float, default=None, help="Низ плиты козырька (м над z=0)")
    ap.add_argument("--platform-height", type=float, default=None)
    ap.add_argument("--platform-depth", type=float, default=None)
    ap.add_argument("--platform-width", type=float, default=None)
    ap.add_argument("--left-wall", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--right-wall", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--partition-thickness", type=float, default=None)
    ap.add_argument(
        "--partition",
        action="append",
        default=None,
        metavar="X",
        type=float,
        help="X центра перегородки (м); можно несколько раз",
    )
    ap.add_argument("--no-partitions", action="store_true", help="Убрать все перегородки")
    ap.add_argument("--pole", action=argparse.BooleanOptionalAction, default=None, help="Столб справа впереди")
    ap.add_argument("--pole-radius", type=float, default=None)
    ap.add_argument(
        "--door",
        action="append",
        default=None,
        metavar="U0,U1,ZB,ZT",
        help="Проём двери: доли u0,u1 по ширине и z_bottom,z_top (м)",
    )
    return ap


def main(argv: List[str] | None = None) -> None:
    ap = _build_cli()
    args = ap.parse_args(argv)
    kw: dict[str, Any] = {}
    if args.style is not None:
        kw["entrance_style"] = args.style
    if args.clear_height is not None:
        kw["niche_clear_height"] = args.clear_height
    if args.niche_floor_z is not None:
        kw["niche_floor_z"] = args.niche_floor_z
    if args.plinth_height is not None:
        kw["plinth_height"] = args.plinth_height
    if args.step_depth is not None:
        kw["step_front_depth"] = args.step_depth
    if args.ceiling_thickness is not None:
        kw["niche_ceiling_thickness"] = args.ceiling_thickness
    if args.no_double_door:
        kw["double_door"] = False
    if args.width is not None:
        kw["width"] = args.width
    if args.depth is not None:
        kw["depth"] = args.depth
    if args.canopy_thickness is not None:
        kw["canopy_thickness"] = args.canopy_thickness
    if args.canopy_z_bottom is not None:
        kw["canopy_z_bottom"] = args.canopy_z_bottom
    if args.platform_height is not None:
        kw["platform_height"] = args.platform_height
    if args.platform_depth is not None:
        kw["platform_depth"] = args.platform_depth
    if args.platform_width is not None:
        kw["platform_width"] = args.platform_width
    if args.left_wall is not None:
        kw["has_left_wall"] = args.left_wall
    if args.right_wall is not None:
        kw["has_right_wall"] = args.right_wall
    if args.partition_thickness is not None:
        kw["partition_thickness"] = args.partition_thickness
    if args.no_partitions:
        kw["partitions_x"] = []
    elif args.partition is not None:
        kw["partitions_x"] = list(args.partition)
    if args.pole is not None:
        kw["right_support_pole"] = args.pole
    if args.pole_radius is not None:
        kw["pole_radius"] = args.pole_radius
    if args.door is not None:
        kw["doors"] = [_parse_door_cli(s) for s in args.door]

    out = Path(args.output).resolve() if args.output else None
    export_entrance(out, no_view=args.no_view, **kw)


if __name__ == "__main__":
    main()
