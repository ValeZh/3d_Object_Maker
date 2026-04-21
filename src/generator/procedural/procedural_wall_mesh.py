"""
Процедурная стена с прямоугольным проёмом (без окна).

Оси: X — вдоль стены, Y — толщина (лицевая +Y), Z — вверх от пола (низ z = 0).
Центр стены по X — 0.

Используется из procedural_wall_window (стена + окно); можно вызывать отдельно.
Развёртка стены под текстуру: src.generator.procedural.unfolding.wall_mesh_expanded_uv.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import trimesh


def _append_quad(
    verts: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
) -> None:
    """Два треугольника, порядок p0→p1→p2→p3 — CCW снаружи тела."""
    i = len(verts)
    verts.extend([p0, p1, p2, p3])
    faces.append((i, i + 1, i + 2))
    faces.append((i, i + 2, i + 3))


def build_wall_mesh_rect_opening(
    wall_length: float,
    wall_thickness: float,
    wall_height: float,
    opening_xmin: float,
    opening_xmax: float,
    opening_zmin: float,
    opening_zmax: float,
    *,
    clearance: float = 0.004,
) -> trimesh.Trimesh:
    """
    Стена: z ∈ [0, wall_height], x ∈ [-wall_length/2, wall_length/2], y ∈ [-T/2, T/2].
    Проём — призма на всю толщину; сетка — 16 выпуклых четырёхугольников (32 треугольника),
    без concat(box) и «звёздной» триангуляции на больших гранях.
    """
    L = max(float(wall_length), 0.05)
    T = max(float(wall_thickness), 0.02)
    H = max(float(wall_height), 0.05)
    eps = float(max(clearance, 0.001))
    x0 = opening_xmin - eps
    x1 = opening_xmax + eps
    z0 = opening_zmin - eps
    z1 = opening_zmax + eps
    hx = L * 0.5
    yp = T * 0.5
    yn = -yp

    if not (-hx + 1e-4 < x0 < x1 < hx - 1e-4 and 0.0 + 1e-4 < z0 < z1 < H - 1e-4):
        raise ValueError(
            "Проём выходит за границы стены или вырожден. "
            f"Стена x∈[-{hx:.4f},{hx:.4f}] z∈[0,{H:.4f}], проём x∈[{x0:.4f},{x1:.4f}] z∈[{z0:.4f},{z1:.4f}]"
        )

    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    # Лицевая +Y
    if z0 > 1e-6:
        _append_quad(verts, faces, (-hx, yp, 0.0), (hx, yp, 0.0), (hx, yp, z0), (-hx, yp, z0))
    if z1 < H - 1e-6:
        _append_quad(verts, faces, (-hx, yp, z1), (hx, yp, z1), (hx, yp, H), (-hx, yp, H))
    _append_quad(verts, faces, (-hx, yp, z0), (x0, yp, z0), (x0, yp, z1), (-hx, yp, z1))
    _append_quad(verts, faces, (x1, yp, z0), (hx, yp, z0), (hx, yp, z1), (x1, yp, z1))

    # Тыл -Y (вид с -Y: CCW противоположен лицевой в xz)
    if z0 > 1e-6:
        _append_quad(verts, faces, (-hx, yn, 0.0), (-hx, yn, z0), (hx, yn, z0), (hx, yn, 0.0))
    if z1 < H - 1e-6:
        _append_quad(verts, faces, (-hx, yn, z1), (-hx, yn, H), (hx, yn, H), (hx, yn, z1))
    _append_quad(verts, faces, (-hx, yn, z0), (-hx, yn, z1), (x0, yn, z1), (x0, yn, z0))
    _append_quad(verts, faces, (x1, yn, z0), (x1, yn, z1), (hx, yn, z1), (hx, yn, z0))

    # Наружные грани по периметру (проём не режет эти плоскости); везде порядок (x, y, z).
    _append_quad(verts, faces, (-hx, yn, 0.0), (-hx, yp, 0.0), (-hx, yp, H), (-hx, yn, H))
    _append_quad(verts, faces, (hx, yp, 0.0), (hx, yn, 0.0), (hx, yn, H), (hx, yp, H))
    _append_quad(verts, faces, (-hx, yn, 0.0), (hx, yn, 0.0), (hx, yp, 0.0), (-hx, yp, 0.0))
    _append_quad(verts, faces, (-hx, yp, H), (hx, yp, H), (hx, yn, H), (-hx, yn, H))

    # Внутренние грани проёма (туннель вдоль Y)
    _append_quad(verts, faces, (x0, yn, z0), (x0, yp, z0), (x0, yp, z1), (x0, yn, z1))
    _append_quad(verts, faces, (x1, yp, z0), (x1, yn, z0), (x1, yn, z1), (x1, yp, z1))
    _append_quad(verts, faces, (x0, yn, z0), (x1, yn, z0), (x1, yp, z0), (x0, yp, z0))
    _append_quad(verts, faces, (x0, yp, z1), (x1, yp, z1), (x1, yn, z1), (x0, yn, z1))

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    wall = trimesh.Trimesh(vertices=v, faces=f, process=False)
    wall.remove_unreferenced_vertices()
    wall.merge_vertices(merge_tex=False)
    wall.fix_normals()
    rgba = np.tile(np.array([175.0, 168.0, 158.0, 255.0], dtype=np.float64), (len(wall.faces), 1))
    wall.visual = trimesh.visual.ColorVisuals(face_colors=rgba.astype(np.uint8))
    return wall
