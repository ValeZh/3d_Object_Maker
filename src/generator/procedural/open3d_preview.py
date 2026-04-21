"""
Превью экспортированных OBJ/MTL в Open3D (общий код для procedural_*).

Не импортирует procedural_window — чтобы избежать циклических импортов.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import trimesh


def try_import_open3d():
    try:
        import open3d as o3d

        return o3d
    except ModuleNotFoundError:
        return None


def require_open3d():
    """Импорт open3d для CLI preview; без пакета — сообщение в stderr и SystemExit(1)."""
    o3d = try_import_open3d()
    if o3d is None:
        print(
            "Команда preview требует пакет open3d.\n"
            "  pip install open3d\n"
            "Без Open3D можно только экспортировать меш:\n"
            "  python -m src.generator.procedural.procedural_window export -o ./out",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return o3d


def _open3d_mesh_preview_camera(mesh: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Центр меша, позиция камеры и up для превью окон/стен с UV."""
    bbox = mesh.get_axis_aligned_bounding_box()
    ctr = np.asarray(bbox.get_center(), dtype=np.float64)
    ext = np.asarray(bbox.get_extent(), dtype=np.float64)
    r = float(max(float(np.linalg.norm(ext)), 0.15))
    eye = ctr + np.array([1.15 * r, -0.92 * r, 0.42 * r], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return ctr, eye, up


def trimesh_to_open3d_mesh(
    mesh: trimesh.Trimesh,
    color_rgb: Tuple[float, float, float] | List[float] | None = None,
):
    """Конвертация trimesh → o3d.TriangleMesh с нормалями и цветом."""
    o3d = require_open3d()

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(v)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d_mesh.compute_vertex_normals()
    if color_rgb is None:
        color_rgb = (0.65, 0.72, 1.0)
    o3d_mesh.paint_uniform_color(list(color_rgb))
    return o3d_mesh


def preview_window_obj_open3d(obj_path: Path | str) -> None:
    """
    Окно / wall_window: ``defaultLit`` + ``window_atlas.png`` и при наличии ``window_normal_atlas.png``.

    По умолчанию **без skybox**, два шага PBR (albedo+normal → только albedo при ошибке) — меньше сбоев Filament на Windows.
    """
    o3d = try_import_open3d()
    if o3d is None:
        print(
            "Для интерактивного просмотра установите: pip install open3d\n"
            "Меш уже сохранён на диск; превью нескольких профилей: "
            "python -m src.generator.procedural.procedural_window preview",
        )
        return
    path = Path(obj_path).resolve()
    if not path.is_file():
        return

    # Стена+окно: в MTL два материала (map_Kd стены + атлас окна). Ветка ниже с одним albedo
    # window_atlas.png на весь TriangleMesh красит стену в цвета атласа — только модель с материалами корректна.
    if path.name.lower() == "wall_window.obj":
        try:
            model = o3d.io.read_triangle_model(str(path))
            mesh_cam = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
            if len(mesh_cam.vertices) > 0:
                mesh_cam.compute_vertex_normals()
                lookat, eye, up = _open3d_mesh_preview_camera(mesh_cam)
                o3d.visualization.draw(
                    model,
                    title="Wall + window",
                    show_skybox=False,
                    lookat=lookat,
                    eye=eye,
                    up=up,
                    field_of_view=58.0,
                )
            else:
                o3d.visualization.draw(model, title="Wall + window", show_skybox=False)
            return
        except Exception as e:
            print(f"[warn] Open3D wall_window (multi-material): {e}")

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) == 0:
        try:
            model = o3d.io.read_triangle_model(str(path))
            o3d.visualization.draw(model, title="Procedural window", show_skybox=False)
        except Exception as e:
            print(f"Open3D не удалось открыть {path}: {e}")
        return

    mesh.compute_vertex_normals()
    lookat, eye, up = _open3d_mesh_preview_camera(mesh)
    draw_base = dict(
        title="Procedural window",
        lookat=lookat,
        eye=eye,
        up=up,
        field_of_view=58.0,
        ibl_intensity=1.0,
        show_skybox=False,
    )
    if os.environ.get("OPEN3D_WIN_PREVIEW_SAFE", "").strip().lower() in ("1", "true", "yes"):
        safe_kw = dict(draw_base)
        safe_kw["title"] = "Procedural window (safe)"
        o3d.visualization.draw(mesh, **safe_kw)
        return

    out_dir = path.parent
    atlas_path = out_dir / "window_atlas.png"
    normal_path = out_dir / "window_normal_atlas.png"

    if atlas_path.is_file():
        from open3d.visualization import rendering

        try:
            albedo = o3d.io.read_image(str(atlas_path))
        except Exception as e:
            albedo = None
            print(f"[warn] Open3D read_image(window_atlas.png): {e}")

        normal_img = None
        if albedo is not None:
            if normal_path.is_file():
                try:
                    normal_img = o3d.io.read_image(str(normal_path))
                except Exception as e:
                    print(f"[warn] Open3D read_image(normal atlas): {e}")

        if albedo is not None:
            attempts: List[Tuple[str, Any]] = []
            m0 = rendering.MaterialRecord()
            m0.shader = "defaultLit"
            m0.albedo_img = albedo
            m0.base_color = (1.0, 1.0, 1.0, 1.0)
            m0.base_metallic = 0.0
            m0.base_roughness = 0.58
            attempts.append(("lit+albedo", m0))

            if normal_img is not None:
                m1 = rendering.MaterialRecord()
                m1.shader = "defaultLit"
                m1.albedo_img = albedo
                m1.normal_img = normal_img
                m1.base_color = (1.0, 1.0, 1.0, 1.0)
                m1.base_metallic = 0.0
                m1.base_roughness = 0.58
                attempts.append(("lit+albedo+normal", m1))

            for label, mat in reversed(attempts):
                try:
                    o3d.visualization.draw(
                        [{"name": "Window", "geometry": mesh, "material": mat}],
                        **draw_base,
                    )
                    return
                except Exception as e:
                    print(f"[warn] Open3D preview ({label}): {e}")

    try:
        o3d.visualization.draw(mesh, **draw_base)
    except Exception as e:
        print(f"[warn] Open3D mesh-only preview: {e}")
        try:
            model = o3d.io.read_triangle_model(str(path))
            o3d.visualization.draw(model, title="Procedural window", show_skybox=False)
        except Exception as e2:
            print(f"[warn] Open3D read_triangle_model fallback: {e2}")


def preview_entrance_obj_open3d(obj_path: Path | str, *, niche: bool = False) -> None:
    """Подъезд без атласа: read_triangle_mesh с постобработкой."""
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    path = Path(obj_path).resolve()
    if niche:
        lookat = np.array([0.0, 0.65, 1.05], dtype=np.float64)
        eye = np.array([0.0, -2.85, 1.32], dtype=np.float64)
    else:
        lookat = np.array([0.0, 0.9, 1.0], dtype=np.float64)
        eye = np.array([0.0, -4.5, 1.45], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices):
        mesh.compute_vertex_normals()
        o3d.visualization.draw(
            mesh,
            title="Entrance",
            lookat=lookat,
            eye=eye,
            up=up,
            field_of_view=58.0,
        )


def preview_entrance_textured_obj_open3d(obj_path: Path | str) -> None:
    """Подъезд с атласом (UV в MTL)."""
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    path = Path(obj_path).resolve()
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) and mesh.has_triangle_uvs():
        mesh.compute_vertex_normals()
    lookat = np.array([0.0, 0.65, 1.05], dtype=np.float64)
    eye = np.array([0.0, -3.2, 1.35], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    o3d.visualization.draw(mesh, title="Entrance (textured)", lookat=lookat, eye=eye, up=up, field_of_view=58.0)


def preview_balcony_obj_open3d(obj_path: Path | str) -> None:
    """
    Балкон: TriangleMesh + UV чаще корректнее тянет map_Kd из MTL, чем read_triangle_model;
    без UV — TriangleMeshModel. Для mesh enable_post_processing=False, чтобы не сшивать вершины на кромках.
    """
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    path = Path(obj_path).resolve()
    lookat = np.array([0.0, 0.65, 1.05], dtype=np.float64)
    eye = np.array([3.6, -4.0, 1.35], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) and mesh.has_triangle_uvs():
        mesh.compute_vertex_normals()
        o3d.visualization.draw(
            mesh,
            title="Balcony",
            lookat=lookat,
            eye=eye,
            up=up,
            field_of_view=58.0,
            ibl_intensity=1.15,
        )
    else:
        model = o3d.io.read_triangle_model(str(path))
        o3d.visualization.draw(
            model,
            title="Balcony",
            lookat=lookat,
            eye=eye,
            up=up,
            field_of_view=58.0,
        )
