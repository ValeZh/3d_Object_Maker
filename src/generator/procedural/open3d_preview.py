"""
Превью экспортированных OBJ/MTL в Open3D (общий код для procedural_*).

Альтернатива Open3D (без Filament / нативных вылетов): переменная окружения ``PROCEDURAL_MESH_PREVIEW``:

- ``plotly`` — интерактивный меш в браузере (WebGL), нужны ``plotly`` и ``trimesh`` (уже в проекте).
- ``system`` — открыть ``.obj`` приложением по умолчанию ОС (Windows: «Просмотр 3D» / Mixed Reality Viewer).

Иначе: Open3D — ``read_triangle_model`` + Filament там, где включено; балкон на Windows по умолчанию
``draw_geometries``; Filament для балкона: ``OPEN3D_BALCONY_FILAMENT_PREVIEW=1``.
Текстурированный подъезд на Windows: по умолчанию меш + ``entrance_atlas.png`` (без пустого Filament+MTL);
Filament+MTL: ``OPEN3D_ENTRANCE_FILAMENT_PREVIEW=1``.

Не импортирует procedural_window — чтобы избежать циклических импортов.
"""
from __future__ import annotations

import os
import platform
import subprocess
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


def _mesh_preview_env_kind() -> str:
    """
    ``PROCEDURAL_MESH_PREVIEW``: пусто / неизвестно → ``open3d``;
    ``plotly`` / ``browser`` / ``html`` → Plotly в браузере;
    ``system`` / ``external`` / ``os`` / ``default_app`` → приложение ОС по умолчанию для ``.obj``.
    """
    v = os.environ.get("PROCEDURAL_MESH_PREVIEW", "").strip().lower()
    if v in ("plotly", "browser", "html"):
        return "plotly"
    if v in ("system", "external", "os", "default_app"):
        return "system"
    return "open3d"


def _trimesh_load_mesh_union(path: Path) -> trimesh.Trimesh | None:
    """OBJ/MTL → один ``Trimesh`` (сцена склеивается)."""
    try:
        loaded = trimesh.load(str(path), force=None)
    except Exception as e:
        print(f"[warn] trimesh.load({path}): {e}")
        return None
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return trimesh.util.concatenate(parts)
    return None


def preview_obj_plotly(obj_path: Path | str, *, title: str = "Mesh") -> bool:
    """Интерактивное превью в браузере (Plotly). Возвращает True при успехе."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        print("PROCEDURAL_MESH_PREVIEW=plotly требует пакет plotly (pip install plotly).")
        return False
    path = Path(obj_path).resolve()
    if not path.is_file():
        return False
    mesh = _trimesh_load_mesh_union(path)
    if mesh is None or len(mesh.vertices) == 0:
        print(f"[warn] Plotly preview: пустой или нечитаемый меш {path}")
        return False
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if f.size == 0:
        print(f"[warn] Plotly preview: нет граней (faces) в {path}")
        return False
    mesh.fix_normals()
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    light = np.array([0.55, -0.75, 0.45], dtype=np.float64)
    light /= max(float(np.linalg.norm(light)), 1e-9)
    intensity = np.clip((vn @ light) * 0.42 + 0.58, 0.0, 1.0)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                intensity=intensity,
                colorscale=[[0.0, "#2a3f5f"], [0.5, "#6b8cae"], [1.0, "#d4e4f2"]],
                showscale=False,
                flatshading=False,
                lighting=dict(ambient=0.35, diffuse=0.85, specular=0.25, roughness=0.45),
            )
        ]
    )
    fig.update_layout(
        title=title[:120],
        scene=dict(aspectmode="data", bgcolor="#1a1a1e"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()
    return True


def preview_obj_system_default(obj_path: Path | str) -> bool:
    """Открыть файл в приложении по умолчанию (для ``.obj`` — встроенный 3D-просмотрщик Windows и т.п.)."""
    path = Path(obj_path).resolve()
    if not path.is_file():
        return False
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except OSError as e:
        print(f"[warn] system preview: {e}")
        return False


def _alternative_preview_if_requested(obj_path: Path | str, *, title: str) -> bool:
    """Если задан альтернативный бэкенд — показать и вернуть True (вызывающий код выходит)."""
    kind = _mesh_preview_env_kind()
    if kind == "plotly":
        return preview_obj_plotly(obj_path, title=title)
    if kind == "system":
        if preview_obj_system_default(obj_path):
            print(f"Открыт файл в приложении по умолчанию: {Path(obj_path).resolve()}")
            return True
        return False
    return False


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


def _filament_viewer_triangle_model(
    o3d: Any,
    path: Path,
    *,
    title: str,
    mesh_post_processing: bool = False,
) -> bool:
    """
    Один режим просмотра, как у «Wall + window»: ``read_triangle_model`` + ``visualization.draw``
    (меню File / Actions / Help, текстуры из MTL). Камера — из ``read_triangle_mesh`` того же OBJ.

    Возвращает True, если ``draw`` вызван без исключения.
    """
    try:
        model = o3d.io.read_triangle_model(str(path))
    except Exception as e:
        print(f"[warn] Open3D read_triangle_model ({title}): {e}")
        return False
    mesh_cam = o3d.io.read_triangle_mesh(str(path), enable_post_processing=mesh_post_processing)
    kw: dict[str, Any] = dict(
        title=title[:127],
        show_skybox=False,
        field_of_view=58.0,
    )
    if len(mesh_cam.vertices) > 0:
        mesh_cam.compute_vertex_normals()
        la, eye, upv = _open3d_mesh_preview_camera(mesh_cam)
        kw["lookat"] = la
        kw["eye"] = eye
        kw["up"] = upv
    try:
        o3d.visualization.draw(model, **kw)
        return True
    except TypeError:
        kw.pop("show_skybox", None)
        try:
            o3d.visualization.draw(model, **kw)
            return True
        except Exception as e:
            print(f"[warn] Open3D draw ({title}): {e}")
    except Exception as e:
        print(f"[warn] Open3D draw ({title}): {e}")
    return False


def _preview_mesh_with_atlas(
    o3d: Any,
    path: Path,
    *,
    title: str,
    atlas_filename: str,
    normal_filename: str | None = None,
) -> bool:
    """
    Filament ``read_triangle_model`` + MTL (map_Bump / map_Pr) на Windows часто даёт пустое белое окно.
    Запасной путь: меш + ``entrance_atlas`` / ``window_atlas`` и shader ``defaultLit``.
    """
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) == 0:
        return False
    mesh.compute_vertex_normals()
    lookat, eye, up = _open3d_mesh_preview_camera(mesh)
    draw_base: dict[str, Any] = dict(
        title=title[:127],
        lookat=lookat,
        eye=eye,
        up=up,
        field_of_view=58.0,
        ibl_intensity=1.0,
        show_skybox=False,
    )
    out_dir = path.parent
    atlas_path = out_dir / atlas_filename
    if not atlas_path.is_file():
        return False
    from open3d.visualization import rendering

    try:
        albedo = o3d.io.read_image(str(atlas_path))
    except Exception as e:
        print(f"[warn] Open3D read_image({atlas_filename}): {e}")
        return False
    normal_img = None
    if normal_filename:
        normal_path = out_dir / normal_filename
        if normal_path.is_file():
            try:
                normal_img = o3d.io.read_image(str(normal_path))
            except Exception as e:
                print(f"[warn] Open3D read_image({normal_filename}): {e}")
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
                [{"name": title[:64], "geometry": mesh, "material": mat}],
                **draw_base,
            )
            return True
        except Exception as e:
            print(f"[warn] Open3D preview ({title}, {label}): {e}")
    return False


def _draw_geometries_safe(o3d: Any, mesh: Any, *, window_name: str) -> None:
    """Старый визуализатор Open3D (без Filament) — стабильнее для тяжёлых балконов на Windows."""
    try:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=window_name[:127],
            width=1280,
            height=800,
            mesh_show_back_face=True,
        )
    except TypeError:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=window_name[:127],
            width=1280,
            height=800,
        )


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
    Окно / wall_window: сначала тот же просмотр, что «Wall + window» — ``read_triangle_model`` + Filament UI.

    Если модель из OBJ не читается — запасной путь: ``window_atlas`` + ``defaultLit`` / меш без атласа.
    """
    path = Path(obj_path).resolve()
    if not path.is_file():
        return
    title = "Wall + window" if path.name.lower() == "wall_window.obj" else "Procedural window"
    if _alternative_preview_if_requested(path, title=title):
        return
    o3d = try_import_open3d()
    if o3d is None:
        print(
            "Для интерактивного просмотра установите: pip install open3d\n"
            "Меш уже сохранён на диск; превью нескольких профилей: "
            "python -m src.generator.procedural.procedural_window preview",
        )
        return

    if _filament_viewer_triangle_model(o3d, path, title=title, mesh_post_processing=False):
        return

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
    """Подъезд без атласа: тот же режим, что «Wall + window» (triangle model + Filament UI)."""
    path = Path(obj_path).resolve()
    if not path.is_file():
        return
    title = "Entrance (niche)" if niche else "Entrance"
    if _alternative_preview_if_requested(path, title=title):
        return
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    if _filament_viewer_triangle_model(o3d, path, title=title, mesh_post_processing=True):
        return
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices):
        mesh.compute_vertex_normals()
        lookat, eye, up = _open3d_mesh_preview_camera(mesh)
        try:
            o3d.visualization.draw(
                mesh,
                title=title,
                lookat=lookat,
                eye=eye,
                up=up,
                field_of_view=58.0,
                show_skybox=False,
            )
        except TypeError:
            o3d.visualization.draw(mesh, title=title, lookat=lookat, eye=eye, up=up, field_of_view=58.0)


def preview_entrance_textured_obj_open3d(obj_path: Path | str) -> None:
    """
    Подъезд с атласом.

    На Windows по умолчанию — меш + ``entrance_atlas.png`` (``defaultLit``), без ``read_triangle_model``:
    Filament+MTL с ``map_Bump``/``map_Pr`` часто открывает пустое белое окно.
    Filament+MTL: ``OPEN3D_ENTRANCE_FILAMENT_PREVIEW=1``.
    """
    path = Path(obj_path).resolve()
    if not path.is_file():
        return
    if _alternative_preview_if_requested(path, title="Entrance (textured)"):
        return
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    win = platform.system() == "Windows"
    force_filament = os.environ.get("OPEN3D_ENTRANCE_FILAMENT_PREVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if win and not force_filament:
        if _preview_mesh_with_atlas(
            o3d,
            path,
            title="Entrance (textured)",
            atlas_filename="entrance_atlas.png",
            normal_filename="entrance_normal_atlas.png",
        ):
            return
    elif not win:
        if _preview_mesh_with_atlas(
            o3d,
            path,
            title="Entrance (textured)",
            atlas_filename="entrance_atlas.png",
            normal_filename="entrance_normal_atlas.png",
        ):
            return
    if _filament_viewer_triangle_model(o3d, path, title="Entrance (textured)", mesh_post_processing=False):
        return
    if _preview_mesh_with_atlas(
        o3d,
        path,
        title="Entrance (textured)",
        atlas_filename="entrance_atlas.png",
        normal_filename="entrance_normal_atlas.png",
    ):
        return
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) and mesh.has_triangle_uvs():
        mesh.compute_vertex_normals()
    lookat, eye, up = _open3d_mesh_preview_camera(mesh) if len(mesh.vertices) else (
        np.array([0.0, 0.65, 1.05], dtype=np.float64),
        np.array([0.0, -3.2, 1.35], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    try:
        o3d.visualization.draw(
            mesh,
            title="Entrance (textured)",
            lookat=lookat,
            eye=eye,
            up=up,
            field_of_view=58.0,
            show_skybox=False,
        )
    except TypeError:
        o3d.visualization.draw(mesh, title="Entrance (textured)", lookat=lookat, eye=eye, up=up, field_of_view=58.0)


def preview_balcony_obj_open3d(obj_path: Path | str) -> None:
    """
    Балкон: на **Windows** по умолчанию ``draw_geometries`` (Filament при ``visualization.draw`` часто роняет процесс).

    Полноценный Filament как у wall_window: ``OPEN3D_BALCONY_FILAMENT_PREVIEW=1`` в окружении перед запуском.
    На Linux/macOS сначала пробуется ``read_triangle_model`` + Filament, при ошибке — ``draw_geometries``.

    Альтернатива без Open3D: ``PROCEDURAL_MESH_PREVIEW=plotly`` или ``system`` (см. модульный docstring).
    """
    path = Path(obj_path).resolve()
    if not path.is_file():
        return
    if _alternative_preview_if_requested(path, title="Balcony"):
        return
    o3d = try_import_open3d()
    if o3d is None:
        print("pip install open3d for interactive preview.")
        return
    force_filament = os.environ.get("OPEN3D_BALCONY_FILAMENT_PREVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    win = platform.system() == "Windows"
    try_filament_first = (not win) or force_filament

    if try_filament_first and _filament_viewer_triangle_model(
        o3d, path, title="Balcony", mesh_post_processing=False
    ):
        return

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0:
        print(f"[warn] Open3D: пустой меш для {path}")
        return
    mesh.compute_vertex_normals()
    _draw_geometries_safe(o3d, mesh, window_name="Balcony")