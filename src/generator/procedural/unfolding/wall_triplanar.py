"""Triplanar UV для стены с проёмом (развёртка под OBJ vt)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import trimesh


def _uv_triplanar_wall(p: np.ndarray, n: np.ndarray, hx: float, L: float, T: float, H: float) -> Tuple[float, float]:
    """Плитка UV по доминирующей оси нормали (для текстурированной стены в OBJ)."""
    ax, ay, az = abs(float(n[0])), abs(float(n[1])), abs(float(n[2]))
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if ay >= ax and ay >= az:
        return ((x + hx) / max(L, 1e-6), z / max(H, 1e-6))
    if ax >= az:
        return ((y + T * 0.5) / max(T, 1e-6), z / max(H, 1e-6))
    return ((x + hx) / max(L, 1e-6), (y + T * 0.5) / max(T, 1e-6))


def wall_mesh_expanded_uv(
    wall: trimesh.Trimesh,
    *,
    hx: float,
    L: float,
    T: float,
    H: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """По одному набору (p, uv) на вершину треугольника — корректные vt в OBJ."""
    v = np.asarray(wall.vertices, dtype=np.float64)
    f = np.asarray(wall.faces, dtype=np.int64)
    fn = np.asarray(wall.face_normals, dtype=np.float64)
    out_v: List[np.ndarray] = []
    out_uv: List[Tuple[float, float]] = []
    out_f: List[Tuple[int, int, int]] = []
    k = 0
    for fi, tri in enumerate(f):
        n = fn[fi]
        for j in range(3):
            out_v.append(v[tri[j]])
            out_uv.append(_uv_triplanar_wall(v[tri[j]], n, hx, L, T, H))
        out_f.append((k, k + 1, k + 2))
        k += 3
    return np.array(out_v, dtype=np.float64), np.array(out_f, dtype=np.int64), np.array(out_uv, dtype=np.float64)
