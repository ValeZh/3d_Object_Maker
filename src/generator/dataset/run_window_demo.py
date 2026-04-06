"""
Тест процедурного окна: экспорт OBJ+MTL+атлас текстур и просмотр в Open3D (окно приложения).

По умолчанию файлы пишутся в data/window_export/. После экспорта открывается интерактивный
просмотр (нужен пакет open3d). Без просмотра: python -m src.generator.dataset.run_window_demo --no-view

Размеры и тип — из procedural_window.USER_WINDOW_MESH.

Запуск из корня репозитория:
  python -m src.generator.dataset.run_window_demo

CLI с флагами (превью нескольких форм / экспорт в произвольную папку):
  python -m src.generator.dataset.procedural_window preview --help
  python -m src.generator.dataset.procedural_window export -o ./out
  python -m src.generator.dataset.procedural_window export --frame-tex wood.png --glass-tex glass.png -o ./out

Свои текстуры (рама / стекло): флаги --frame-tex и --glass-tex (можно указать только одно).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

# корень репозитория (…/3d_Object_Maker)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generator.dataset.procedural_window import (
    USER_WINDOW_MESH,
    build_window_frame_glass_meshes,
    _frame_thickness,
    _normalize_partial_horizontal_bars,
    _pick_float_param,
    _pick_kind,
    _pick_nonneg_int,
    _pick_profile,
)
from src.generator.dataset.window_texture_assets import ensure_window_textures, make_atlas_from_sources

_DEFAULT_EXPORT_DIR = _REPO_ROOT / "data" / "window_export"


def _faceted_triplanar_uv(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    UV по нормалям граней (вершины дублируются по граням): для бокса нет «уголковых»
    усреднённых нормалей и артефактов освещения/текстуры на тонком стекле.
    """
    mesh = mesh.copy()
    if len(mesh.faces) == 0:
        return mesh, np.zeros((0, 2), dtype=np.float64)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v_exp = verts[faces].reshape(-1, 3)
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fn_exp = np.repeat(fn, 3, axis=0)

    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    eps = 1e-9
    dx = max(xmax - xmin, eps)
    dy = max(ymax - ymin, eps)
    dz = max(zmax - zmin, eps)

    dom = np.argmax(np.abs(fn_exp), axis=1)
    u = np.zeros(len(v_exp), dtype=np.float64)
    vv = np.zeros(len(v_exp), dtype=np.float64)
    mx = dom == 0
    my = dom == 1
    mz = dom == 2
    u[mx] = (v_exp[mx, 1] - ymin) / dy
    vv[mx] = (v_exp[mx, 2] - zmin) / dz
    u[my] = (v_exp[my, 0] - xmin) / dx
    vv[my] = (v_exp[my, 2] - zmin) / dz
    u[mz] = (v_exp[mz, 0] - xmin) / dx
    vv[mz] = (v_exp[mz, 1] - ymin) / dy

    u = np.clip(u, 0.0, 1.0)
    vv = np.clip(vv, 0.0, 1.0)
    uv = np.stack([u, vv], axis=1)
    new_faces = np.arange(len(v_exp), dtype=np.int64).reshape(-1, 3)
    out = trimesh.Trimesh(vertices=v_exp, faces=new_faces, process=False)
    out.fix_normals()
    return out, uv


def preview_window_obj_open3d(obj_path: Path) -> None:
    """Показать экспортированное окно с текстурами (Open3D). Без open3d — только сообщение в консоль."""
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        print(
            "Для интерактивного просмотра установите: pip install open3d\n"
            "Меш уже сохранён на диск; превью нескольких профилей: "
            "python -m src.generator.dataset.procedural_window preview",
        )
        return
    obj_path = obj_path.resolve()
    if not obj_path.is_file():
        return
    try:
        model = o3d.io.read_triangle_model(str(obj_path))
    except Exception as e:
        print(f"Open3D не удалось открыть {obj_path}: {e}")
        return
    o3d.visualization.draw(model, title="Procedural window")


def _resolve_texture_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    r = Path(p).expanduser().resolve()
    return r if r.is_file() else None


def export_window_demo(
    out_dir: Path | None = None,
    *,
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
    atlas_half_size: int = 512,
) -> Path:
    u = USER_WINDOW_MESH
    w = float(width if width is not None else u["width"])
    h = float(height if height is not None else u["height"])
    d = float(depth if depth is not None else u["depth"])
    prof = profile if profile is not None else str(u["profile"])
    knd = kind if kind is not None else str(u["kind"])

    out_dir = out_dir or _DEFAULT_EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_name = "window_atlas.png"
    tex_path = out_dir / tex_name
    fp = _resolve_texture_path(frame_texture)
    gp = _resolve_texture_path(glass_texture)
    if frame_texture is not None and fp is None:
        print(f"[warn] frame_texture missing, procedural frame: {frame_texture}")
    if glass_texture is not None and gp is None:
        print(f"[warn] glass_texture missing, procedural glass: {glass_texture}")

    if fp is not None or gp is not None:
        atlas_img = make_atlas_from_sources(
            frame_path=fp,
            glass_path=gp,
            half_size=max(atlas_half_size, 64),
        )
        atlas_img.save(tex_path)
        src_note = "custom image(s) + procedural fallback if side omitted"
    else:
        tex_dir = _REPO_ROOT / "data" / "textures"
        paths = ensure_window_textures(tex_dir)
        shutil.copyfile(paths["atlas"], tex_path)
        src_note = f"{paths['frame'].name} + {paths['glass'].name} (data/textures)"

    ft = _frame_thickness(w, h)
    glass_t = max(d * 0.12, 0.004)
    nv = _pick_nonneg_int(mullions_vertical, u.get("mullions_vertical", 0))
    nh = _pick_nonneg_int(mullions_horizontal, u.get("mullions_horizontal", 0))
    ox = _pick_float_param(mullion_offset_x, u.get("mullion_offset_x", 0.0))
    oz = _pick_float_param(mullion_offset_z, u.get("mullion_offset_z", 0.0))
    ph_raw = partial_horizontal_bars if partial_horizontal_bars is not None else u.get("partial_horizontal_bars")
    partial_bars = _normalize_partial_horizontal_bars(ph_raw)
    profile = _pick_profile(prof, str(u.get("profile", "rect")))
    kind = _pick_kind(knd, str(u.get("kind", "fixed")))

    mf, mg = build_window_frame_glass_meshes(
        width=w,
        height=h,
        depth=d,
        profile=profile,
        kind=kind,
        mullions_vertical=nv,
        mullions_horizontal=nh,
        mullion_offset_x=ox,
        mullion_offset_z=oz,
        partial_horizontal_bars=partial_bars,
        ft=ft,
        glass_t=glass_t,
        glass_y=0.0,
    )

    mf_uv, uv_f = _faceted_triplanar_uv(mf)
    mg_uv, uv_g = _faceted_triplanar_uv(mg)
    uv_f = np.asarray(uv_f, dtype=np.float64).copy()
    uv_g = np.asarray(uv_g, dtype=np.float64).copy()
    uv_f[:, 0] = uv_f[:, 0] * 0.5
    uv_g[:, 0] = uv_g[:, 0] * 0.5 + 0.5
    uv = np.vstack([uv_f, uv_g]) if len(uv_f) + len(uv_g) else np.zeros((0, 2))

    if len(mf_uv.faces) == 0 and len(mg_uv.faces) == 0:
        work = trimesh.Trimesh()
    elif len(mf_uv.faces) == 0:
        work = mg_uv
        uv = uv_g
    elif len(mg_uv.faces) == 0:
        work = mf_uv
        uv = uv_f
    else:
        work = trimesh.util.concatenate([mf_uv, mg_uv])

    img = Image.open(tex_path)
    work.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)

    obj_path = out_dir / "window.obj"
    work.export(str(obj_path), include_texture=True)

    mtl_path = out_dir / "material.mtl"
    if mtl_path.is_file():
        txt = mtl_path.read_text(encoding="utf-8")
        txt = txt.replace("map_Kd material_0.png", f"map_Kd {tex_name}")
        txt = txt.replace("map_Kd material_0.jpg", f"map_Kd {tex_name}")
        # trimesh ставит Kd=0.4 — diffuse затемняет map_Kd, в Open3D меш выглядит серым
        txt = re.sub(r"(?m)^Ka\s+.*$", "Ka 1 1 1", txt)
        txt = re.sub(r"(?m)^Kd\s+.*$", "Kd 1 1 1", txt)
        txt = re.sub(r"(?m)^Ks\s+.*$", "Ks 0 0 0", txt)
        mtl_path.write_text(txt, encoding="utf-8")

    print(f"[OK] Window export: {obj_path}")
    print(f"     Atlas (frame|glass): {tex_path}")
    print(f"     Textures: {src_note}")
    return obj_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Экспорт процедурного окна + опционально свои текстуры.")
    parser.add_argument("--no-view", action="store_true", help="Не открывать Open3D после экспорта")
    parser.add_argument("-o", "--output", type=str, default=None, help="Папка вывода (по умолчанию data/window_export)")
    parser.add_argument("--frame-tex", type=str, default=None, metavar="PATH", help="Файл текстуры рамы")
    parser.add_argument("--glass-tex", type=str, default=None, metavar="PATH", help="Файл текстуры стекла")
    parser.add_argument(
        "--texture-size",
        type=int,
        default=512,
        metavar="N",
        help="Сторона квадрата половины атласа при сборке из файлов",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[warn] unknown args ignored:", unknown)

    out = Path(args.output).resolve() if args.output else None
    obj = export_window_demo(
        out,
        frame_texture=args.frame_tex,
        glass_texture=args.glass_tex,
        atlas_half_size=args.texture_size,
    )
    if not args.no_view:
        preview_window_obj_open3d(obj)


if __name__ == "__main__":
    main()
