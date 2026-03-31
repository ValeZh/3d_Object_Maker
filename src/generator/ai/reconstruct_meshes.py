#!/usr/bin/env python3
# reconstruct_meshes.py — GAN → примитивы → OBJ/PLY/GLB

import torch
import numpy as np
import open3d as o3d
from pathlib import Path

# ==========================================================
# 🧠 Импорты из CGAN (НЕ ДУБЛИРУЕМ КОД!)
# ==========================================================
from src.generator.ai.pointcloud_cgan import (
    Generator,
    LATENT_DIM,
    COND_DIM,
    NUM_POINTS,
    DEVICE,
    DB_PATH    # путь к базе, пригодится позже
)

# ==========================================================
# 📌 Пути
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = DATA_DIR / "point_cgan_pointnet_output"
MODEL_PATH   = OUTPUT_DIR / "model_final.pt"

SAVE_DIR = DATA_DIR / "generated_obj"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 📌 Загрузка классов из БД, как в генераторе
# ==========================================================
def load_classes():
    # id, name → упорядочены
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name FROM shapes ORDER BY id ASC"
    ).fetchall()
    conn.close()

    classes = {i: row[1] for i, row in enumerate(rows)}
    inverse = {v: k for k, v in classes.items()}
    return classes, inverse

CLASSES, INV_CLASSES = load_classes()
print("✔ Классы:", CLASSES)

# ==========================================================
# 📌 Фиттеры примитивов
# ==========================================================
def fit_cube(points):
    center = np.mean(points, axis=0)
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    size = float(np.mean(extent))

    mesh = o3d.geometry.TriangleMesh.create_box(size, size, size)
    mesh.translate(center - np.array([size/2]*3))
    mesh.compute_vertex_normals()
    return mesh


def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=42)
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def fit_cylinder(points):
    center = np.mean(points, axis=0)
    height = np.ptp(points[:,2])
    radius = np.mean(np.linalg.norm(points[:, :2] - center[:2], axis=1))

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=40)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0]))
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def fit_cone(points):
    center = np.mean(points, axis=0)
    height = np.ptp(points[:, 2])
    radius = np.max(np.linalg.norm(points[:, :2] - center[:2], axis=1))

    mesh = o3d.geometry.TriangleMesh.create_cone(radius, height, resolution=40)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0]))
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def fit_torus(points):
    center = np.mean(points, axis=0)

    R = np.mean(np.linalg.norm(points[:, :2] - center[:2], axis=1))
    r = np.std(np.linalg.norm(points - center, axis=1))

    if r < 0.1 or R < r:
        R, r = 0.8, 0.3

    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=R,
        tube_radius=r,
        radial_resolution=42,
        tubular_resolution=24
    )
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def fit_pyramid(points):
    base_z = np.min(points[:, 2])
    base = points[np.abs(points[:,2] - base_z) < 0.1]

    apex = points[np.argmax(points[:,2])]
    center = np.mean(base[:, :2], axis=0)

    size = np.mean(np.max(base[:, :2], axis=0) - np.min(base[:, :2], axis=0))
    height = apex[2] - base_z

    mesh = o3d.geometry.TriangleMesh()
    v = np.array([
        [center[0]-size/2, center[1]-size/2, base_z],
        [center[0]+size/2, center[1]-size/2, base_z],
        [center[0]+size/2, center[1]+size/2, base_z],
        [center[0]-size/2, center[1]+size/2, base_z],
        [center[0],        center[1],        base_z+height]
    ])
    t = np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    mesh.vertices  = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(t)
    mesh.compute_vertex_normals()
    return mesh


FITTERS = {
    "cube": fit_cube,
    "sphere": fit_sphere,
    "cylinder": fit_cylinder,
    "cone": fit_cone,
    "torus": fit_torus,
    "pyramid": fit_pyramid,
}

# ==========================================================
# 📌 Экспорт OBJ/PLY/GLB
# ==========================================================
def export_mesh(mesh: o3d.geometry.TriangleMesh, name: str):
    out = {
        "obj": SAVE_DIR / f"gan_{name}.obj",
        "ply": SAVE_DIR / f"gan_{name}.ply",
        "glb": SAVE_DIR / f"gan_{name}.glb",
    }
    for k, p in out.items():
        o3d.io.write_triangle_mesh(str(p), mesh)
    print(f"  ✔ {name}: OBJ/PLY/GLB saved")


# ==========================================================
# 📌 Main
# ==========================================================
def main():
    if not MODEL_PATH.exists():
        print("❌ CGAN модель не найдена!")
        return

    print(f"✔ Load GAN from {MODEL_PATH.name}")
    generator = Generator(
        LATENT_DIM,
        COND_DIM,
        NUM_POINTS,
        len(CLASSES)
    ).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["G"])
    generator.eval()

    meshes = []
    colors = [
        [0.6, 0.7, 1.0],
        [1.0, 0.6, 0.6],
        [0.6, 1.0, 0.6],
        [1.0, 0.8, 0.4],
        [0.8, 0.5, 1.0],
        [1.0, 0.6, 0.8],
    ]

    for i, (cid, name) in enumerate(CLASSES.items()):
        print(f"\n[{cid}] Generate: {name}")

        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        label = torch.tensor([cid], device=DEVICE)

        with torch.no_grad():
            pts = generator(z, label).cpu().numpy()[0]

        if np.isnan(pts).any():
            print("⚠️ INVALID POINTS — SKIP")
            continue

        mesh = FITTERS[name](pts)
        mesh.paint_uniform_color(colors[i])
        mesh.translate([i*3, 0, 0])

        export_mesh(mesh, name)
        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes)


if __name__ == "__main__":
    main()
