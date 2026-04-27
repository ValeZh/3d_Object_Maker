"""
Стена с процедурным окном: вертикальный параллелепипед с прямоугольным проёмом
по AABB рамы окна и то же окно, что в procedural_window.

Оси (как у окна): X — вдоль стены, Y — толщина (лицевая сторона окна в +Y), Z — вверх от пола.
Низ стены z = 0. Центр стены по X — 0.

Позиция окна:
  window_center_x — сместить центр окна по X относительно центра стены;
  window_sill_z — подоконник (низ внешней рамы), метры от пола.

Проём — вертильная прямоугольная «дыра» по габаритам рамы (mf) с небольшим зазором.
Для profile arch / round вырез остаётся прямоугольным по bounding box рамы (в углах свода
могут остаться куски стены; для точного контура используйте rect).

Геометрия стены: procedural_wall_mesh.build_wall_mesh_rect_opening; UV — unfolding.wall_mesh_expanded_uv; OBJ/MTL — texturing.

Запуск (одна строка — так надёжно в PowerShell; символ ^ из cmd не использовать в PS):
  python -m src.generator.procedural.procedural_wall_window export -o data/wall_win --wall-length 4 --wall-thickness 0.35 --wall-height 3 --window-center-x 0.8 --window-sill-z 0.95

В PowerShell перенос строки — обратная кавычка ` в конце строки, не ^.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import trimesh
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generator.procedural.open3d_preview import preview_window_obj_open3d
from src.generator.procedural.procedural_wall_mesh import build_wall_mesh_rect_opening
from src.generator.procedural.procedural_window import (
    build_window_frame_glass_meshes,
    resolve_window_frame_glass_params,
    _add_window_build_args,
    _frame_thickness,
    _parse_partial_h_tokens,
)
from src.generator.procedural.texturing import (
    ensure_window_textures,
    make_atlas_from_sources,
    make_window_roughness_atlas,
    resolve_texture_path,
    write_wall_window_mtl,
    write_wall_window_obj,
)
from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint
from src.generator.procedural.texturing.pbr_map_utils import make_normal_map_from_albedo, make_roughness_map_from_albedo
from src.generator.procedural.unfolding import frame_glass_atlas_uv_mesh, wall_mesh_expanded_uv

_DEFAULT_WALL_WIN_DIR = _REPO_ROOT / "data" / "wall_window_export"


@dataclass
class WallWindowBuild:
    """Результат сборки: window_mesh + window_uv — окно на месте (текстура задаётся при экспорте)."""

    wall: trimesh.Trimesh
    window_frame: trimesh.Trimesh
    window_glass: trimesh.Trimesh
    window_mesh: trimesh.Trimesh
    window_uv: np.ndarray


def build_wall_with_window(
    *,
    wall_length: float,
    wall_thickness: float,
    wall_height: float,
    window_center_x: float,
    window_sill_z: float,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    profile: str | None = None,
    kind: str | None = None,
    mullions_vertical: int | None = None,
    mullions_horizontal: int | None = None,
    mullion_offset_x: float | None = None,
    mullion_offset_z: float | None = None,
    partial_horizontal_bars: List[Tuple[int, float]] | None = None,
) -> WallWindowBuild:
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
        with_glass=True,
    )

    tz = float(window_sill_z) + p.height * 0.5
    tvec = np.array([float(window_center_x), 0.0, tz], dtype=np.float64)
    mf_w = mf.copy()
    mg_w = mg.copy()
    mf_w.apply_translation(tvec)
    mg_w.apply_translation(tvec)

    bounds = mf_w.bounds
    opening_xmin, opening_xmax = float(bounds[0, 0]), float(bounds[1, 0])
    opening_zmin, opening_zmax = float(bounds[0, 2]), float(bounds[1, 2])

    wall_m = build_wall_mesh_rect_opening(
        wall_length,
        wall_thickness,
        wall_height,
        opening_xmin,
        opening_xmax,
        opening_zmin,
        opening_zmax,
    )

    wt, uv = frame_glass_atlas_uv_mesh(mf, mg)
    wt.apply_translation(tvec)

    return WallWindowBuild(
        wall=wall_m,
        window_frame=mf_w,
        window_glass=mg_w,
        window_mesh=wt,
        window_uv=uv,
    )


def export_wall_with_window(
    out_dir: Path | None = None,
    *,
    wall_length: float,
    wall_thickness: float,
    wall_height: float,
    window_center_x: float,
    window_sill_z: float,
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
    wall_texture: str | Path | None = None,
    wall_normal_texture: str | Path | None = None,
    wall_roughness_texture: str | Path | None = None,
    frame_normal_texture: str | Path | None = None,
    glass_normal_texture: str | Path | None = None,
    frame_texture_color: Any = None,
    glass_texture_color: Any = None,
    wall_texture_color: Any = None,
    atlas_half_size: int = 512,
    generate_normal_maps: bool = True,
    generate_roughness_maps: bool = True,
) -> Path:
    """Экспорт wall_window.obj + wall_window.mtl + window_atlas.png (стена без атласа; окно с атласом)."""
    out_dir = Path(out_dir or _DEFAULT_WALL_WIN_DIR).resolve()
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
        make_atlas_from_sources(
            frame_path=fp,
            glass_path=gp,
            half_size=max(atlas_half_size, 64),
            frame_color=frame_texture_color,
            glass_color=glass_texture_color,
        ).save(tex_path)
    else:
        paths = ensure_window_textures(_REPO_ROOT / "data" / "textures")
        shutil.copyfile(paths["atlas"], tex_path)

    window_normal_name: str | None = None
    window_roughness_name: str | None = None
    if generate_normal_maps:
        window_normal_name = "window_normal_atlas.png"
        fn = resolve_texture_path(frame_normal_texture)
        gn = resolve_texture_path(glass_normal_texture)
        wn_path = out_dir / window_normal_name
        if fn is not None or gn is not None:
            from src.generator.procedural.texturing import make_normal_atlas_from_sources

            make_normal_atlas_from_sources(frame_path=fn, glass_path=gn, half_size=max(atlas_half_size, 64)).save(wn_path)
        else:
            make_normal_map_from_albedo(Image.open(tex_path).convert("RGB"), strength=3.6).save(wn_path)
    if generate_roughness_maps:
        window_roughness_name = "window_roughness_atlas.png"
        wr_path = out_dir / window_roughness_name
        if fp is not None or gp is not None:
            make_roughness_map_from_albedo(Image.open(tex_path).convert("RGB"), min_roughness=0.25, max_roughness=0.9).save(
                wr_path
            )
        else:
            make_window_roughness_atlas(max(atlas_half_size, 64)).save(wr_path)

    b = build_wall_with_window(
        wall_length=wall_length,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        window_center_x=window_center_x,
        window_sill_z=window_sill_z,
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

    win_export = b.window_mesh.copy()
    win_export.visual = trimesh.visual.texture.TextureVisuals(uv=b.window_uv, image=Image.open(tex_path))

    hx = float(wall_length) * 0.5
    wp = resolve_texture_path(wall_texture)
    if wall_texture is not None and wp is None:
        print(f"[warn] wall_texture missing, wall without map: {wall_texture}")
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
        wv, wf, wuv = wall_mesh_expanded_uv(
            b.wall,
            hx=hx,
            L=float(wall_length),
            T=float(wall_thickness),
            H=float(wall_height),
        )
        wn = resolve_texture_path(wall_normal_texture)
        wr = resolve_texture_path(wall_roughness_texture)
        if wn is None and wp.stem.endswith("_albedo"):
            cand = wp.with_name(wp.stem[: -len("_albedo")] + "_normal" + wp.suffix)
            wn = cand if cand.is_file() else None
        if wr is None and wp.stem.endswith("_albedo"):
            cand = wp.with_name(wp.stem[: -len("_albedo")] + "_roughness" + wp.suffix)
            wr = cand if cand.is_file() else None
        if generate_normal_maps and wn is not None:
            wall_normal_name = f"wall_normal{wn.suffix.lower()}"
            Image.open(wn).convert("RGB").save(out_dir / wall_normal_name)
        elif generate_normal_maps:
            wall_normal_name = "wall_normal.png"
            make_normal_map_from_albedo(wim, strength=3.2).save(out_dir / wall_normal_name)
        if generate_roughness_maps and wr is not None:
            wall_roughness_name = f"wall_roughness{wr.suffix.lower()}"
            Image.open(wr).convert("RGB").save(out_dir / wall_roughness_name)
        elif generate_roughness_maps:
            wall_roughness_name = "wall_roughness.png"
            make_roughness_map_from_albedo(wim, min_roughness=0.35, max_roughness=0.92).save(out_dir / wall_roughness_name)
    else:
        wv = np.asarray(b.wall.vertices, dtype=np.float64)
        wf = np.asarray(b.wall.faces, dtype=np.int64)
        wuv = None

    win_v = np.asarray(win_export.vertices, dtype=np.float64)
    win_f = np.asarray(win_export.faces, dtype=np.int64)
    win_uv = np.asarray(win_export.visual.uv, dtype=np.float64)

    mtl_name = "wall_window.mtl"
    mtl_path = out_dir / mtl_name
    obj_path = out_dir / "wall_window.obj"
    write_wall_window_mtl(
        mtl_path,
        window_atlas=tex_name,
        wall_tex=wall_tex_name,
        wall_normal_tex=wall_normal_name,
        wall_roughness_tex=wall_roughness_name,
        window_normal_tex=window_normal_name,
        window_roughness_tex=window_roughness_name,
    )
    write_wall_window_obj(obj_path, mtl_name, wv, wf, wuv, win_v, win_f, win_uv)

    stale = out_dir / "material.mtl"
    if stale.is_file():
        try:
            stale.unlink()
        except OSError:
            pass

    print(f"[OK] Wall+window: {obj_path}")
    print(f"     Atlas: {tex_path}")
    return obj_path


def _parse_optional_tex_rgb_cli(value: str | None) -> Any:
    if value is None or not str(value).strip():
        return None
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ожидается r,g,b из трёх чисел через запятую")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _cli_export(args: Any) -> None:
    partial_kw: List[Tuple[int, float]] | None = None
    if args.partial_h is not None:
        partial_kw = _parse_partial_h_tokens(args.partial_h)

    path = export_wall_with_window(
        Path(args.output).resolve() if args.output else None,
        wall_length=args.wall_length,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height,
        window_center_x=args.window_center_x,
        window_sill_z=args.window_sill_z,
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
        wall_texture=getattr(args, "wall_texture", None),
        frame_texture_color=getattr(args, "frame_texture_color", None),
        glass_texture_color=getattr(args, "glass_texture_color", None),
        wall_texture_color=getattr(args, "wall_texture_color", None),
        atlas_half_size=max(getattr(args, "texture_half_size", 512), 64),
    )
    if not args.no_view:
        preview_window_obj_open3d(path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Процедурная стена с окном (OBJ+MTL+атлас).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
В PowerShell не используйте ^ (это не перенос строки — символы уйдут в argparse и дадут ошибку).
Одна строка или перенос через обратную кавычку ` в конце строки.

Пример (одной строкой):
  python -m src.generator.procedural.procedural_wall_window export -o ./out --wall-length 5 --wall-thickness 0.32 --wall-height 3.2 --window-center-x 0 --window-sill-z 1.0 --width 1.2 --height 1.5 --mullions-vertical 1 --partial-h 1:0.5
""".strip(),
    )
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="wall_window.obj + текстуры")
    ex.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help="Папка вывода (по умолчанию: data/wall_window_export)",
    )
    ex.add_argument("--wall-length", type=float, required=True, help="Длина стены по X (м)")
    ex.add_argument("--wall-thickness", type=float, required=True, help="Толщина стены по Y (м)")
    ex.add_argument("--wall-height", type=float, required=True, help="Высота стены по Z, низ z=0 (м)")
    ex.add_argument(
        "--window-center-x",
        type=float,
        default=0.0,
        help="Смещение центра окна по X от центра стены (м)",
    )
    ex.add_argument(
        "--window-sill-z",
        type=float,
        required=True,
        metavar="Z",
        help="Высота подоконника (низ рамы), от пола z=0 (м)",
    )
    _add_window_build_args(ex)
    ex.add_argument(
        "--frame-tex",
        type=str,
        default=None,
        metavar="PATH",
        dest="frame_texture",
        help="Текстура рамы (jpg/png)",
    )
    ex.add_argument(
        "--glass-tex",
        type=str,
        default=None,
        metavar="PATH",
        dest="glass_texture",
        help="Текстура стекла",
    )
    ex.add_argument(
        "--wall-tex",
        type=str,
        default=None,
        metavar="PATH",
        dest="wall_texture",
        help="Опционально: текстура стены (копируется как wall_diffuse.*; triplanar UV)",
    )
    ex.add_argument(
        "--frame-tex-color",
        type=_parse_optional_tex_rgb_cli,
        default=None,
        metavar="R,G,B",
        dest="frame_texture_color",
        help="Тинт diffuse для текстуры рамы",
    )
    ex.add_argument(
        "--glass-tex-color",
        type=_parse_optional_tex_rgb_cli,
        default=None,
        metavar="R,G,B",
        dest="glass_texture_color",
        help="Тинт для текстуры стекла",
    )
    ex.add_argument(
        "--wall-tex-color",
        type=_parse_optional_tex_rgb_cli,
        default=None,
        metavar="R,G,B",
        dest="wall_texture_color",
        help="Тинт для текстуры стены",
    )
    ex.add_argument(
        "--texture-size",
        type=int,
        default=512,
        metavar="N",
        dest="texture_half_size",
        help="Половина стороны атласа для пользовательских карт",
    )
    ex.add_argument("--no-view", action="store_true", help="Не открывать Open3D после экспорта")
    ex.set_defaults(_handler=_cli_export)
    return p


def main(argv: List[str] | None = None) -> None:
    p = _build_parser()
    args = p.parse_args(argv)
    h = getattr(args, "_handler", None)
    if callable(h):
        h(args)


if __name__ == "__main__":
    main()
