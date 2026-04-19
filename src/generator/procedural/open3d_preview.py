"""
Превью экспортированных OBJ/MTL в Open3D (общий код для procedural_*).

Не импортирует procedural_window — чтобы избежать циклических импортов.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

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
    """Экспортированное окно или wall_window с текстурами (read_triangle_model)."""
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
    try:
        model = o3d.io.read_triangle_model(str(path))
    except Exception as e:
        print(f"Open3D не удалось открыть {path}: {e}")
        return
    o3d.visualization.draw(model, title="Procedural window")


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
