"""
Процедурная крыша здания.

Оси: X — длина, Y — ширина, Z — высота. Основание лежит в плоскости z=0,
центр основания в (0, 0, 0).

Параметры:
  length, width, height — размеры крыши (м).
  roof_type — форма:
    flat    — плоская плита (параллелепiped);
    pyramid — четырёхгранная пирамида с прямоугольным основанием;
    gable   — двускатная: конёк параллелен оси X (длине).

Запуск:
  python -m src.generator.procedural.procedural_roof export -o data/roof_export
  python -m src.generator.procedural.procedural_roof export --type pyramid --length 8 --width 6 --height 2.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from PIL import Image

from src.generator.procedural.procedural_texture_maps.procedural_color_texture import (
    make_plaster_facade_texture,
    make_uniform_noise_texture,
)
from src.generator.procedural.procedural_texture_maps.normal_map import make_stucco_like_normal_map
from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint
from src.generator.procedural.texturing.pbr_map_utils import make_roughness_map_from_albedo
from src.generator.procedural.texturing.surface_texture_assets import make_roof_shingles_pack
from src.generator.procedural.texturing.window_texture_assets import resolve_texture_path
from src.generator.procedural.unfolding import faceted_triplanar_uv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_ROOF_DIR = _REPO_ROOT / "data" / "roof_export"

RoofType = Literal["flat", "pyramid", "gable"]

USER_ROOF: dict[str, Any] = {
    "length": 10.0,
    "width": 8.0,
    "height": 2.2,
    "roof_type": "gable",
    "closed": True,
}


def _quad(
    bl: np.ndarray,
    br: np.ndarray,
    tr: np.ndarray,
    tl: np.ndarray,
    *,
    flip: bool = False,
) -> trimesh.Trimesh:
    a, b, c, d = [np.asarray(p, dtype=np.float64) for p in (bl, br, tr, tl)]
    faces = [[0, 1, 2], [0, 2, 3]]
    if flip:
        faces = [[0, 2, 1], [0, 3, 2]]
    return trimesh.Trimesh(vertices=[a, b, c, d], faces=faces, process=False)


def _tri(a: np.ndarray, b: np.ndarray, c: np.ndarray, *, flip: bool = False) -> trimesh.Trimesh:
    va, vb, vc = [np.asarray(p, dtype=np.float64) for p in (a, b, c)]
    faces = [[0, 1, 2]] if not flip else [[0, 2, 1]]
    return trimesh.Trimesh(vertices=[va, vb, vc], faces=faces, process=False)


def build_flat_roof_mesh(length: float, width: float, height: float) -> trimesh.Trimesh:
    """Плоская плита: параллелепiped L×W×H, низ на z=0."""
    L = max(float(length), 0.05)
    W = max(float(width), 0.05)
    H = max(float(height), 0.02)
    mesh = trimesh.creation.box(extents=[L, W, H])
    mesh.apply_translation([0.0, 0.0, H * 0.5])
    return mesh


def build_pyramid_roof_mesh(
    length: float,
    width: float,
    height: float,
    *,
    closed: bool = True,
) -> trimesh.Trimesh:
    """Пирамида с прямоугольным основанием; вершина в центре основания на z=height."""
    L = max(float(length), 0.05)
    W = max(float(width), 0.05)
    H = max(float(height), 0.05)
    hx, hy = L * 0.5, W * 0.5
    apex = np.array([0.0, 0.0, H], dtype=np.float64)
    bl = np.array([-hx, -hy, 0.0], dtype=np.float64)
    br = np.array([hx, -hy, 0.0], dtype=np.float64)
    tr_ = np.array([hx, hy, 0.0], dtype=np.float64)
    tl = np.array([-hx, hy, 0.0], dtype=np.float64)

    parts = [
        _tri(bl, br, apex),
        _tri(br, tr_, apex),
        _tri(tr_, tl, apex),
        _tri(tl, bl, apex),
    ]
    if closed:
        parts.append(_quad(bl, br, tr_, tl))
    return trimesh.util.concatenate(parts)


def build_gable_roof_mesh(
    length: float,
    width: float,
    height: float,
    *,
    closed: bool = True,
) -> trimesh.Trimesh:
    """
    Двускатная крыша: конёк параллелен оси X (длине).
    Скаты смотрят на ±Y, фронтоны — треугольники на ±X.
    """
    L = max(float(length), 0.05)
    W = max(float(width), 0.05)
    H = max(float(height), 0.05)
    hx, hy = L * 0.5, W * 0.5

    bl = np.array([-hx, -hy, 0.0], dtype=np.float64)
    br = np.array([hx, -hy, 0.0], dtype=np.float64)
    fr = np.array([hx, hy, 0.0], dtype=np.float64)
    fl = np.array([-hx, hy, 0.0], dtype=np.float64)
    rl = np.array([-hx, 0.0, H], dtype=np.float64)
    rr = np.array([hx, 0.0, H], dtype=np.float64)

    parts = [
        _quad(bl, br, rr, rl),   # скат −Y
        _quad(fl, fr, rr, rl, flip=True),  # скат +Y
        _tri(bl, fl, rl),
        _tri(br, rr, fr),
    ]
    if closed:
        parts.append(_quad(bl, br, fr, fl))
    return trimesh.util.concatenate(parts)


def build_roof_mesh(
    length: float,
    width: float,
    height: float,
    *,
    roof_type: RoofType | str = "gable",
    closed: bool = True,
) -> trimesh.Trimesh:
    """Собрать меш крыши по типу."""
    kind = str(roof_type).strip().lower()
    if kind in ("flat", "plate", "slab", "плоская", "плоская крыша"):
        return build_flat_roof_mesh(length, width, height)
    if kind in ("pyramid", "пирамида", "пирамидальная"):
        return build_pyramid_roof_mesh(length, width, height, closed=closed)
    if kind in ("gable", "двускатная", "двускатная крыша", "shed"):
        return build_gable_roof_mesh(length, width, height, closed=closed)
    raise ValueError(f"Unknown roof_type: {roof_type!r} (expected flat | pyramid | gable)")


def _write_roof_obj(
    obj_path: Path,
    mtl_name: str,
    v: np.ndarray,
    f: np.ndarray,
    uv: np.ndarray,
) -> None:
    lines: list[str] = ["# roof (procedural_roof)", f"mtllib {mtl_name}", "", "o roof", "usemtl roof"]
    for row in v:
        lines.append(f"v {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")
    for row in uv:
        lines.append(f"vt {row[0]:.8f} {row[1]:.8f}")
    for tri in f:
        a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_roof_mtl(
    mtl_path: Path,
    *,
    roof_tex: str | None,
    roof_normal_tex: str | None = None,
    roof_roughness_tex: str | None = None,
    bump_strength: float = 0.7,
) -> None:
    bump_scale = float(max(0.0, bump_strength))
    lines = [
        "newmtl roof",
        "Ka 1 1 1",
        "Kd 0.48 0.32 0.24",
        "Ks 0 0 0",
    ]
    if roof_tex:
        lines.append(f"map_Kd {roof_tex}")
    if roof_normal_tex:
        lines.append(f"map_Bump -bm {bump_scale:.3f} {roof_normal_tex}")
        lines.append(f"bump -bm {bump_scale:.3f} {roof_normal_tex}")
    if roof_roughness_tex:
        lines.append(f"map_Pr {roof_roughness_tex}")
    lines.append("")
    mtl_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_roof_textures(
    out_dir: Path,
    *,
    roof_texture: str | Path | None,
    roof_normal_texture: str | Path | None,
    roof_roughness_texture: str | Path | None,
    roof_texture_color: Any,
    use_procedural_maps: bool,
    roof_color_preset: str,
    bump_strength: float,
) -> tuple[str | None, str | None, str | None]:
    wp = resolve_texture_path(roof_texture)
    wn = resolve_texture_path(roof_normal_texture)
    wr = resolve_texture_path(roof_roughness_texture)

    if use_procedural_maps and wp is None:
        s = 1024
        ck = str(roof_color_preset).strip().lower()
        if ck in ("plaster", "uniform_noise", "noise"):
            wim = (
                make_uniform_noise_texture(s, base_rgb=(140, 132, 124), noise_sigma=4.0, seed=51)
                if ck != "plaster"
                else make_plaster_facade_texture(s, base_rgb=(140, 132, 124), seed=51)
            )
            nim = make_stucco_like_normal_map(s, strength=2.8, coarse_grid=16, seed=53)
        else:
            pack = make_roof_shingles_pack(s, seed=404)
            wim = pack["albedo"]
            nim = pack["normal"]
            if wr is None:
                wr_path = out_dir / "_proc_roof_roughness.png"
                pack["roughness"].save(wr_path)
                wr = wr_path

        diffuse_path = out_dir / "_proc_roof_diffuse.png"
        wim.save(diffuse_path)
        wp = diffuse_path
        if wn is None:
            normal_path = out_dir / "_proc_roof_normal.png"
            nim.save(normal_path)
            wn = normal_path

    roof_tex_name: str | None = None
    roof_normal_name: str | None = None
    roof_roughness_name: str | None = None

    if wp is not None:
        roof_tex_name = f"roof_diffuse{wp.suffix.lower()}"
        wt = parse_texture_color_tint(roof_texture_color)
        wim = Image.open(wp).convert("RGB")
        if wt is not None:
            wim = apply_texture_color_tint(wim, wt)
        wim.save(out_dir / roof_tex_name)
    if wn is not None:
        roof_normal_name = f"roof_normal{wn.suffix.lower()}"
        Image.open(wn).convert("RGB").save(out_dir / roof_normal_name)
    if wr is None and wp is not None and use_procedural_maps:
        wr_path = out_dir / "_proc_roof_roughness.png"
        make_roughness_map_from_albedo(Image.open(wp).convert("RGB"), min_roughness=0.45, max_roughness=0.95).save(
            wr_path
        )
        wr = wr_path
    if wr is not None:
        roof_roughness_name = f"roof_roughness{wr.suffix.lower()}"
        Image.open(wr).convert("RGB").save(out_dir / roof_roughness_name)

    _ = bump_strength  # used in _write_roof_mtl by caller
    return roof_tex_name, roof_normal_name, roof_roughness_name


def export_roof(
    out_dir: Path | None = None,
    *,
    length: float | None = None,
    width: float | None = None,
    height: float | None = None,
    roof_type: RoofType | str | None = None,
    closed: bool | None = None,
    roof_texture: str | Path | None = None,
    roof_normal_texture: str | Path | None = None,
    roof_roughness_texture: str | Path | None = None,
    roof_texture_color: Any = None,
    use_procedural_maps: bool = True,
    roof_color_preset: str = "roof_shingles",
    bump_strength: float = 0.7,
    **kwargs: Any,
) -> Path:
    """Экспорт roof.obj + roof.mtl + текстуры в out_dir."""
    _ = kwargs
    cfg = dict(USER_ROOF)
    if length is not None:
        cfg["length"] = float(length)
    if width is not None:
        cfg["width"] = float(width)
    if height is not None:
        cfg["height"] = float(height)
    if roof_type is not None:
        cfg["roof_type"] = roof_type
    if closed is not None:
        cfg["closed"] = bool(closed)

    out_dir = Path(out_dir or _DEFAULT_ROOF_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_roof_mesh(
        cfg["length"],
        cfg["width"],
        cfg["height"],
        roof_type=cfg["roof_type"],
        closed=bool(cfg.get("closed", True)),
    )
    mesh_uv, uv = faceted_triplanar_uv(mesh)

    roof_tex_name, roof_normal_name, roof_roughness_name = _resolve_roof_textures(
        out_dir,
        roof_texture=roof_texture,
        roof_normal_texture=roof_normal_texture,
        roof_roughness_texture=roof_roughness_texture,
        roof_texture_color=roof_texture_color,
        use_procedural_maps=use_procedural_maps,
        roof_color_preset=roof_color_preset,
        bump_strength=bump_strength,
    )

    mtl_name = "roof.mtl"
    obj_path = out_dir / "roof.obj"
    mtl_path = out_dir / mtl_name
    _write_roof_mtl(
        mtl_path,
        roof_tex=roof_tex_name,
        roof_normal_tex=roof_normal_name,
        roof_roughness_tex=roof_roughness_name,
        bump_strength=bump_strength,
    )
    _write_roof_obj(obj_path, mtl_name, mesh_uv.vertices, mesh_uv.faces, uv)
    return obj_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Процедурная крыша здания")
    sub = p.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("export", help="Сохранить roof.obj")
    ex.add_argument("-o", "--output", type=str, default=None, help="Папка вывода")
    ex.add_argument("--length", type=float, default=None, help="Длина по X (м)")
    ex.add_argument("--width", type=float, default=None, help="Ширина по Y (м)")
    ex.add_argument("--height", type=float, default=None, help="Высота крыши (м)")
    ex.add_argument(
        "--type",
        dest="roof_type",
        type=str,
        default=None,
        choices=["flat", "pyramid", "gable"],
        help="Тип крыши",
    )
    ex.add_argument("--open-base", action="store_true", help="Без нижней грани (только для pyramid/gable)")
    ex.add_argument("--roof-tex", type=str, default=None, help="Diffuse-текстура")
    ex.add_argument("--roof-tex-color", type=str, default=None, metavar="R,G,B", help="Тинт текстуры")
    ex.add_argument("--no-procedural-maps", action="store_true", help="Без процедурных карт")
    ex.add_argument(
        "--roof-color-preset",
        type=str,
        default="roof_shingles",
        choices=["roof_shingles", "plaster", "uniform_noise"],
    )
    ex.add_argument("--no-view", action="store_true", help="Не открывать превью после экспорта")
    ex.set_defaults(_handler=_cli_export)

    return p


def _cli_export(args: argparse.Namespace) -> None:
    out = Path(args.output).resolve() if args.output else None
    tint = parse_texture_color_tint(args.roof_tex_color) if args.roof_tex_color else None
    obj_path = export_roof(
        out,
        length=args.length,
        width=args.width,
        height=args.height,
        roof_type=args.roof_type,
        closed=False if args.open_base else None,
        roof_texture=args.roof_tex,
        roof_texture_color=tint,
        use_procedural_maps=not args.no_procedural_maps,
        roof_color_preset=args.roof_color_preset,
    )
    print(f"[ok] {obj_path}")
    if not args.no_view:
        from src.generator.procedural.open3d_preview import preview_window_obj_open3d

        preview_window_obj_open3d(obj_path)


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args._handler(args)


if __name__ == "__main__":
    main()
