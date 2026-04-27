from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.generator.procedural.procedural_wall_mesh import build_solid_wall_mesh
from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint
from src.generator.procedural.texturing.window_texture_assets import resolve_texture_path
from src.generator.procedural.unfolding.wall_triplanar import wall_mesh_expanded_uv

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WALL_DIR = _REPO_ROOT / "data" / "wall_export"


def _write_wall_obj(
    obj_path: Path,
    mtl_name: str,
    v: np.ndarray,
    f: np.ndarray,
    uv: np.ndarray,
) -> None:
    lines: list[str] = ["# wall (procedural_wall)", f"mtllib {mtl_name}", "", "o wall", "usemtl wall"]
    for row in v:
        lines.append(f"v {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")
    for row in uv:
        lines.append(f"vt {row[0]:.8f} {row[1]:.8f}")
    for tri in f:
        a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_wall_mtl(mtl_path: Path, *, wall_tex: str | None) -> None:
    lines = [
        "newmtl wall",
        "Ka 1 1 1",
        "Kd 0.69 0.66 0.62",
        "Ks 0 0 0",
    ]
    if wall_tex:
        lines.append(f"map_Kd {wall_tex}")
    lines.append("")
    mtl_path.write_text("\n".join(lines), encoding="utf-8")


def _write_wall_mtl_with_maps(
    mtl_path: Path,
    *,
    wall_tex: str | None,
    wall_normal_tex: str | None,
    wall_roughness_tex: str | None,
    bump_strength: float = 0.7,
) -> None:
    bump_scale = float(max(0.0, bump_strength))
    lines = [
        "newmtl wall",
        "Ka 1 1 1",
        "Kd 0.69 0.66 0.62",
        "Ks 0 0 0",
    ]
    if wall_tex:
        lines.append(f"map_Kd {wall_tex}")
    if wall_normal_tex:
        # Common OBJ/MTL normal/bump spellings used by viewers.
        lines.append(f"map_Bump -bm {bump_scale:.3f} {wall_normal_tex}")
        lines.append(f"bump -bm {bump_scale:.3f} {wall_normal_tex}")
    if wall_roughness_tex:
        # Non-standard extension used by PBR-capable importers.
        lines.append(f"map_Pr {wall_roughness_tex}")
    lines.append("")
    mtl_path.write_text("\n".join(lines), encoding="utf-8")


def _infer_companion_map(base: Path, suffix: str) -> Path | None:
    """
    For names like ``wall_cracked_albedo.png`` infer ``wall_cracked_normal.png`` / ``..._roughness.png``.
    """
    stem = base.stem
    if stem.endswith("_albedo"):
        c = base.with_name(stem[: -len("_albedo")] + f"_{suffix}" + base.suffix)
        return c if c.is_file() else None
    return None


def export_wall(
    out_dir: Path | None = None,
    *,
    wall_length: float,
    wall_thickness: float,
    wall_height: float,
    wall_texture: str | Path | None = None,
    wall_normal_texture: str | Path | None = None,
    wall_roughness_texture: str | Path | None = None,
    wall_texture_color: Any = None,
    bump_strength: float = 0.7,
) -> Path:
    out_dir = Path(out_dir or _DEFAULT_WALL_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    L = max(float(wall_length), 0.05)
    T = max(float(wall_thickness), 0.02)
    H = max(float(wall_height), 0.05)
    hx = L * 0.5

    wall = build_solid_wall_mesh(L, T, H)

    v, f, uv = wall_mesh_expanded_uv(wall, hx=hx, L=L, T=T, H=H)

    wp = resolve_texture_path(wall_texture)
    wn = resolve_texture_path(wall_normal_texture)
    wr = resolve_texture_path(wall_roughness_texture)
    if wn is None and wp is not None:
        wn = _infer_companion_map(wp, "normal")
    if wr is None and wp is not None:
        wr = _infer_companion_map(wp, "roughness")
    if wall_texture is not None and wp is None:
        print(f"[warn] wall_texture missing, wall without map: {wall_texture}")
    if wall_normal_texture is not None and wn is None:
        print(f"[warn] wall_normal_texture missing: {wall_normal_texture}")
    if wall_roughness_texture is not None and wr is None:
        print(f"[warn] wall_roughness_texture missing: {wall_roughness_texture}")
    wall_tex_name: str | None = None
    wall_normal_name: str | None = None
    wall_roughness_name: str | None = None
    if wp is not None:
        wall_tex_name = f"wall_diffuse{wp.suffix.lower()}"
        wt = parse_texture_color_tint(wall_texture_color)
        wim = Image.open(wp).convert("RGB")
        if wt is not None:
            wim = apply_texture_color_tint(wim, wt)
        wim.save(out_dir / wall_tex_name)
    if wn is not None:
        wall_normal_name = f"wall_normal{wn.suffix.lower()}"
        Image.open(wn).convert("RGB").save(out_dir / wall_normal_name)
    if wr is not None:
        wall_roughness_name = f"wall_roughness{wr.suffix.lower()}"
        Image.open(wr).convert("RGB").save(out_dir / wall_roughness_name)

    mtl_name = "wall.mtl"
    mtl_path = out_dir / mtl_name
    obj_path = out_dir / "wall.obj"
    _write_wall_mtl_with_maps(
        mtl_path,
        wall_tex=wall_tex_name,
        wall_normal_tex=wall_normal_name,
        wall_roughness_tex=wall_roughness_name,
        bump_strength=bump_strength,
    )
    _write_wall_obj(obj_path, mtl_name, v, f, uv)
    print(f"[OK] Wall export: {obj_path}")
    return obj_path

