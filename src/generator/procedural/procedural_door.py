"""
Процедурные двери (trimesh): рама, импосты, тонкие «стекла» — для подъездов и простых сцен.

Система координат (ниша подъезда): плоскость двери **XZ**, глубина рамы вдоль **Y**.
Внешняя грань рамы — y = y_outer (ближе к задней стене); рама и стекло уходят в сторону **уменьшения Y**.

  build_french_double_door_parts — двустворчатая дверь с центральным импостом и средней перекладиной.
  build_simple_door_slab — одна глухая плоскость в проёме.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import trimesh


def build_french_double_door_parts(
    *,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    y_outer: float,
    frame_width: float,
    frame_depth: float,
    leaf_gap: float,
    midrail_z_frac: float,
    glass_thickness: float = 0.014,
    niche_depth: Optional[float] = None,
    y_clip_min: float = 0.04,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Двустворчатая дверь: периметр рамы, центральный штапик, горизонтальная перекладина на двух полотнах,
    четыре тонких блока «стекло».

    x0,x1 — границы проёма по X; z0,z1 — по Z; y_outer — Y внешней грани рамы (клип внутрь ниши при niche_depth).
    """
    parts: List[Tuple[str, trimesh.Trimesh]] = []
    xa, xb = (x0, x1) if x0 < x1 else (x1, x0)
    za, zb = (z0, z1) if z0 < z1 else (z1, z0)
    fw = max(float(frame_width), 0.025)
    fd = max(float(frame_depth), 0.02)
    g = max(float(leaf_gap), 0.012)
    tg = max(float(glass_thickness), 0.006)

    y_hi = (niche_depth - y_clip_min) if niche_depth is not None else y_outer
    y_hi = max(y_hi, y_clip_min + 0.02)
    y_pl = float(np.clip(y_outer, y_clip_min, y_hi))

    spanz = max(zb - za, 0.15)
    z_mid = za + float(np.clip(midrail_z_frac, 0.2, 0.82)) * spanz
    rail_h = max(0.045 * spanz, 0.055)
    x_mid = 0.5 * (xa + xb)
    cy = y_pl - fd * 0.5

    def bx(tag: str, cx: float, cz: float, ex: float, ez: float) -> None:
        b = trimesh.creation.box(extents=[ex, fd, ez])
        b.apply_translation([cx, cy, cz])
        parts.append((tag, b))

    bx("door_frame", 0.5 * (xa + xb), zb - fw * 0.5, (xb - xa), fw)
    bx("door_frame", 0.5 * (xa + xb), za + fw * 0.5, (xb - xa), fw)
    bx("door_frame", xa + fw * 0.5, 0.5 * (za + zb), fw, (zb - za))
    bx("door_frame", xb - fw * 0.5, 0.5 * (za + zb), fw, (zb - za))
    bx("door_jamb", x_mid, 0.5 * (za + zb), g, (zb - za))

    xl = 0.5 * (xa + fw + x_mid - g * 0.5)
    wl = max(x_mid - g * 0.5 - xa - fw, 0.0)
    xr = 0.5 * (x_mid + g * 0.5 + xb - fw)
    wr = max(xb - fw - x_mid - g * 0.5, 0.0)

    bx("door_rail", xl, z_mid, wl, rail_h)
    bx("door_rail", xr, z_mid, wr, rail_h)

    gy = y_pl - fd - tg * 0.5
    z_up0 = z_mid + rail_h * 0.5
    z_up1 = zb - fw
    h_up = z_up1 - z_up0
    z_lo0 = za + fw
    z_lo1 = z_mid - rail_h * 0.5
    h_lo = z_lo1 - z_lo0

    if wl > 0.03 and h_up > 0.03:
        m = trimesh.creation.box(extents=[wl, tg, h_up])
        m.apply_translation([xl, gy, z_up0 + h_up * 0.5])
        parts.append(("door_glass", m))
    if wl > 0.03 and h_lo > 0.03:
        m = trimesh.creation.box(extents=[wl, tg, h_lo])
        m.apply_translation([xl, gy, z_lo0 + h_lo * 0.5])
        parts.append(("door_glass", m))
    if wr > 0.03 and h_up > 0.03:
        m = trimesh.creation.box(extents=[wr, tg, h_up])
        m.apply_translation([xr, gy, z_up0 + h_up * 0.5])
        parts.append(("door_glass", m))
    if wr > 0.03 and h_lo > 0.03:
        m = trimesh.creation.box(extents=[wr, tg, h_lo])
        m.apply_translation([xr, gy, z_lo0 + h_lo * 0.5])
        parts.append(("door_glass", m))

    return parts


def build_simple_door_slab(
    *,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    y_outer: float,
    depth: float = 0.06,
    niche_depth: Optional[float] = None,
    y_clip_min: float = 0.04,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Одна коробка в проёме (глухая дверь)."""
    xa, xb = (x0, x1) if x0 < x1 else (x1, x0)
    za, zb = (z0, z1) if z0 < z1 else (z1, z0)
    dd = max(float(depth), 0.04)
    y_hi = (niche_depth - y_clip_min) if niche_depth is not None else y_outer
    y_hi = max(y_hi, y_clip_min + 0.02)
    y_pl = float(np.clip(y_outer, y_clip_min, y_hi))
    door = trimesh.creation.box(extents=[max(xb - xa, 0.1), dd, max(zb - za, 0.1)])
    door.apply_translation([0.5 * (xa + xb), y_pl - dd * 0.5, 0.5 * (za + zb)])
    return [("door_fill", door)]
