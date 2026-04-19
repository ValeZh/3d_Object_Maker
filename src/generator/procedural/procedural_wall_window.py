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

Геометрия стены с проёмом: модуль procedural_wall_mesh (build_wall_mesh_rect_opening, wall_mesh_expanded_uv).

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

from src.generator.procedural.procedural_window import (
    build_window_frame_glass_meshes,
    frame_glass_atlas_uv_mesh,
    resolve_window_frame_glass_params,
    _add_window_build_args,
    _frame_thickness,
    _parse_partial_h_tokens,
)
from src.generator.procedural.open3d_preview import preview_window_obj_open3d
from src.generator.procedural.window_texture_assets import resolve_texture_path
from src.generator.procedural.window_texture_assets import ensure_window_textures, make_atlas_from_sources
from src.generator.procedural.procedural_wall_mesh import build_wall_mesh_rect_opening, wall_mesh_expanded_uv

_DEFAULT_WALL_WIN_DIR = _REPO_ROOT / "data" / "wall_window_export"


def _write_wall_window_obj(
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
        n_vt_wall = int(len(wall_uv))
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


def _write_wall_window_mtl(
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
    atlas_half_size: int = 512,
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
        make_atlas_from_sources(frame_path=fp, glass_path=gp, half_size=max(atlas_half_size, 64)).save(tex_path)
    else:
        paths = ensure_window_textures(_REPO_ROOT / "data" / "textures")
        shutil.copyfile(paths["atlas"], tex_path)

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
    if wp is not None:
        wall_tex_name = f"wall_diffuse{wp.suffix.lower()}"
        shutil.copyfile(wp, out_dir / wall_tex_name)
        wv, wf, wuv = wall_mesh_expanded_uv(
            b.wall,
            hx=hx,
            L=float(wall_length),
            T=float(wall_thickness),
            H=float(wall_height),
        )
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
    _write_wall_window_mtl(mtl_path, window_atlas=tex_name, wall_tex=wall_tex_name)
    _write_wall_window_obj(obj_path, mtl_name, wv, wf, wuv, win_v, win_f, win_uv)

    stale = out_dir / "material.mtl"
    if stale.is_file():
        try:
            stale.unlink()
        except OSError:
            pass

    print(f"[OK] Wall+window: {obj_path}")
    print(f"     Atlas: {tex_path}")
    return obj_path


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
