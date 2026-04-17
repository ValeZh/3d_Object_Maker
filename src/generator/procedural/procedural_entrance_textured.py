"""
Текстурированный экспорт подъезда (procedural_entrance): один атлас 3×1 — стены | крыша | дверь.

Классификация по имени части (как в build_entrance_meshes / build_niche_entrance_meshes):
  • дверь — имя содержит «door»;
  • крыша/перекрытие — ceiling, canopy;
  • стены — остальное (стены, пол, площадка, ступени, столб, перегородки …).

UV: faceted triplanar (как у окон), каждый тайл атласа — своя текстура.

Пример:
  python -m src.generator.procedural.procedural_entrance_textured export -o data/entrance_tex \\
    --style niche --wall-tex path/wall.png --roof-tex path/roof.png --door-tex path/door.png
  Без путей к файлам — процедурные заглушки в атласе.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

from src.generator.procedural.procedural_entrance import (
    ENTRANCE_NICHE_PRESET,
    USER_ENTRANCE,
    build_entrance_meshes,
    build_niche_entrance_meshes,
)
from src.generator.procedural.run_window_demo import _faceted_triplanar_uv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ENTRANCE_ATLAS_NUM_TILES = 3
TILE_WALL = 0
TILE_ROOF = 1
TILE_DOOR = 2


def _entrance_part_tile_index(part_name: str) -> int:
    n = part_name.lower()
    if "door" in n:
        return TILE_DOOR
    if n.startswith("ceiling") or n == "canopy":
        return TILE_ROOF
    return TILE_WALL


def _scale_uv_to_atlas_tile(uv: np.ndarray, tile_index: int, n_tiles: int = ENTRANCE_ATLAS_NUM_TILES) -> np.ndarray:
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
) -> Image.Image:
    """Горизонтальный атлас: [стена | крыша | дверь]."""
    t = max(int(tile), 64)
    paths = (wall_tex, roof_tex, door_tex)
    defaults = ((150, 142, 132), (120, 128, 140), (92, 72, 58))
    tiles: List[Image.Image] = []
    for i, (p, rgb) in enumerate(zip(paths, defaults)):
        if p is not None:
            pp = Path(p).expanduser().resolve()
            if pp.is_file():
                tiles.append(_open_rgb_resize(pp, t))
                continue
        tiles.append(_proc_tile_rgb(i, rgb, t))
    w = t * ENTRANCE_ATLAS_NUM_TILES
    out = Image.new("RGB", (w, t))
    for i, im in enumerate(tiles):
        out.paste(im, (i * t, 0))
    return out


def _concatenate_uv_meshes(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
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
            raise RuntimeError("entrance textured: submesh without per-vertex uv")
        u = np.asarray(u, dtype=np.float64)
        if len(u) != len(vv):
            raise RuntimeError("entrance textured: uv/vertex count mismatch")
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


def _collect_entrance_named_parts(merged: dict[str, Any]) -> List[Tuple[str, trimesh.Trimesh]]:
    """Те же ветки, что export_entrance в procedural_entrance."""
    p = merged
    style = str(p.get("entrance_style", "canopy")).lower()
    if style == "niche":
        dz0, dz1 = p.get("niche_door_z_bottom"), p.get("niche_door_z_top")
        return build_niche_entrance_meshes(
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
    pw_raw = p.get("platform_width")
    return build_entrance_meshes(
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


def export_entrance_textured(
    out_dir: Path | None = None,
    *,
    no_view: bool = False,
    atlas_tile: int = 256,
    wall_tex: str | Path | None = None,
    roof_tex: str | Path | None = None,
    door_tex: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """
    Экспорт OBJ+MTL+PNG: атлас из трёх текстур (стены / крыша / дверь).
    ``**kwargs`` — те же параметры, что у ``export_entrance`` (width, depth, entrance_style, …).
    """
    out_dir = Path(out_dir or (_REPO_ROOT / "data" / "entrance_textured_export"))
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = {**USER_ENTRANCE, **kwargs}
    if str(merged.get("entrance_style", "canopy")).lower() == "niche":
        merged = {**USER_ENTRANCE, **ENTRANCE_NICHE_PRESET, **kwargs}

    parts = _collect_entrance_named_parts(merged)
    mesh_blocks: List[trimesh.Trimesh] = []
    for name, m in parts:
        if m is None or len(m.faces) == 0:
            continue
        m2, uv = _faceted_triplanar_uv(m)
        ti = _entrance_part_tile_index(name)
        uv_t = _scale_uv_to_atlas_tile(uv, ti)
        m2.visual = trimesh.visual.texture.TextureVisuals(uv=uv_t)
        mesh_blocks.append(m2)

    if not mesh_blocks:
        raise RuntimeError("entrance_textured: empty mesh")

    atlas_img = make_entrance_atlas(
        tile=atlas_tile,
        wall_tex=wall_tex,
        roof_tex=roof_tex,
        door_tex=door_tex,
    )
    work = _concatenate_uv_meshes(mesh_blocks)
    uv_all = np.asarray(work.visual.uv, dtype=np.float64)
    work.visual = trimesh.visual.texture.TextureVisuals(uv=uv_all, image=atlas_img)

    tex_name = "entrance_atlas.png"
    tex_path = out_dir / tex_name
    atlas_img.save(str(tex_path))

    obj_path = out_dir / "entrance.obj"
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

    print(f"[OK] Entrance (textured): {obj_path}")
    print(f"     Atlas: {tex_path}")

    if not no_view:
        _preview_entrance_textured_open3d(obj_path)
    return obj_path


def _preview_entrance_textured_open3d(obj_path: Path) -> None:
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        print("pip install open3d for interactive preview.")
        return
    mesh = o3d.io.read_triangle_mesh(str(obj_path.resolve()), enable_post_processing=False)
    if len(mesh.vertices) and mesh.has_triangle_uvs():
        mesh.compute_vertex_normals()
    lookat = np.array([0.0, 0.65, 1.05], dtype=np.float64)
    eye = np.array([0.0, -3.2, 1.35], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    o3d.visualization.draw(mesh, title="Entrance (textured)", lookat=lookat, eye=eye, up=up, field_of_view=58.0)


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


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Экспорт подъезда с атласом: стены | крыша | дверь.")
    ap.add_argument("command", nargs="?", default="export", choices=("export",), help="Команда")
    ap.add_argument("-o", "--output", type=str, default=None, help="Папка вывода")
    ap.add_argument("--no-view", action="store_true", help="Не открывать Open3D")
    ap.add_argument("--atlas-tile", type=int, default=256, help="Размер одного тайла атласа (px)")
    ap.add_argument("--wall-tex", type=str, default=None, help="PNG/JPG текстуры стен")
    ap.add_argument("--roof-tex", type=str, default=None, help="PNG/JPG крыши / потолка / козырька")
    ap.add_argument("--door-tex", type=str, default=None, help="PNG/JPG двери")
    ap.add_argument(
        "--style",
        type=str,
        default=None,
        choices=("canopy", "niche"),
        help="canopy — козырёк; niche — ниша",
    )
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--canopy-thickness", type=float, default=None)
    ap.add_argument("--canopy-z-bottom", type=float, default=None)
    ap.add_argument("--platform-height", type=float, default=None)
    ap.add_argument("--platform-depth", type=float, default=None)
    ap.add_argument("--platform-width", type=float, default=None)
    ap.add_argument("--left-wall", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--right-wall", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--partition-thickness", type=float, default=None)
    ap.add_argument("--partition", action="append", default=None, metavar="X", type=float)
    ap.add_argument("--no-partitions", action="store_true")
    ap.add_argument("--pole", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--pole-radius", type=float, default=None)
    ap.add_argument("--door", action="append", default=None, metavar="U0,U1,ZB,ZT")
    # niche
    ap.add_argument("--clear-height", type=float, default=None)
    ap.add_argument("--niche-floor-z", type=float, default=None)
    ap.add_argument("--plinth-height", type=float, default=None)
    ap.add_argument("--step-depth", type=float, default=None)
    ap.add_argument("--ceiling-thickness", type=float, default=None)
    ap.add_argument("--no-double-door", action="store_true")
    return ap


def main(argv: List[str] | None = None) -> None:
    ap = _build_argparser()
    args = ap.parse_args(argv)
    kw: dict[str, Any] = {}
    if args.style is not None:
        kw["entrance_style"] = args.style
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

    out = Path(args.output).resolve() if args.output else None
    export_entrance_textured(
        out,
        no_view=args.no_view,
        atlas_tile=int(args.atlas_tile),
        wall_tex=args.wall_tex,
        roof_tex=args.roof_tex,
        door_tex=args.door_tex,
        **kw,
    )


if __name__ == "__main__":
    main()
