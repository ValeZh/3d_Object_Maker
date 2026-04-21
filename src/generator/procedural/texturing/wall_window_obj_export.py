"""OBJ/MTL экспорт стены с окном (vt, map_Kd)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def write_wall_window_obj(
    obj_path: Path,
    mtl_name: str,
    wall_v: np.ndarray,
    wall_f: np.ndarray,
    wall_uv: np.ndarray | None,
    win_v: np.ndarray,
    win_f: np.ndarray,
    win_uv: np.ndarray,
) -> None:
    """OBJ: стена с usemtl wall (без map_Kd, без vt) или с vt; окно с отдельным usemtl window + vt."""
    lines: List[str] = ["# wall + window (procedural_wall_window)", f"mtllib {mtl_name}", ""]
    nv_wall = int(len(wall_v))

    lines.append("o wall")
    lines.append("usemtl wall")
    for row in wall_v:
        lines.append(f"v {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")
    if wall_uv is not None:
        for row in wall_uv:
            lines.append(f"vt {row[0]:.8f} {row[1]:.8f}")
        for tri in wall_f:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            ta, tb, tc = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
    else:
        for tri in wall_f:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            lines.append(f"f {a} {b} {c}")

    lines.append("")
    lines.append("o window")
    lines.append("usemtl window")
    base_v = nv_wall
    for row in win_v:
        lines.append(f"v {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")
    for row in win_uv:
        lines.append(f"vt {row[0]:.8f} {row[1]:.8f}")
    vt_base = int(len(wall_uv)) if wall_uv is not None else 0
    for tri in win_f:
        a, b, c = int(tri[0]) + base_v + 1, int(tri[1]) + base_v + 1, int(tri[2]) + base_v + 1
        ta, tb, tc = int(tri[0]) + vt_base + 1, int(tri[1]) + vt_base + 1, int(tri[2]) + vt_base + 1
        lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wall_window_mtl(
    mtl_path: Path,
    *,
    window_atlas: str,
    wall_tex: str | None,
) -> None:
    lines = [
        "# wall: diffuse only (no map) unless wall_tex set — avoids atlas on wall without vt",
        "newmtl wall",
        "Ka 1 1 1",
        "Kd 0.69 0.66 0.62",
        "Ks 0 0 0",
    ]
    if wall_tex:
        lines.append(f"map_Kd {wall_tex}")
    lines.extend(
        [
            "",
            "newmtl window",
            "Ka 1 1 1",
            "Kd 1 1 1",
            "Ks 0 0 0",
            f"map_Kd {window_atlas}",
            "",
        ]
    )
    mtl_path.write_text("\n".join(lines), encoding="utf-8")
