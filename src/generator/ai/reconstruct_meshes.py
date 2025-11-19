#!/usr/bin/env python3
# reconstruct_meshes.py — GAN + ИДЕАЛЬНЫЕ ФИГУРЫ + .obj + .ply + .glb

import torch
import numpy as np
import open3d as o3d
from pathlib import Path

# -------------------------------
# ПУТИ
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "point_cgan_pointnet_output"
MODEL_PATH = OUTPUT_DIR / "model_final.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = DATA_DIR / "generated_obj"
# -------------------------------
# ПАРАМЕТРЫ
# -------------------------------
LATENT_DIM = 128
COND_DIM = 64
NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {0: "cube", 1: "sphere", 2: "cylinder", 3: "cone", 4: "torus", 5: "pyramid"}

# -------------------------------
# GENERATOR
# -------------------------------
class Generator(torch.nn.Module):
    def __init__(self, z_dim, cond_dim, num_points, num_classes):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_classes, cond_dim)
        h = 512
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim + cond_dim, h),
            torch.nn.ReLU(True),
            torch.nn.LayerNorm(h),
            torch.nn.Linear(h, h * 2),
            torch.nn.ReLU(True),
            torch.nn.LayerNorm(h * 2),
            torch.nn.Linear(h * 2, h),
            torch.nn.ReLU(True),
            torch.nn.LayerNorm(h),
            torch.nn.Linear(h, num_points * 3),
            torch.nn.Tanh()
        )
        self.num_points = num_points

    def forward(self, z, labels):
        cond = self.label_emb(labels)
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        return out.view(-1, self.num_points, 3)

# -------------------------------
# ФИТТИНГ
# -------------------------------
def fit_cube(points):
    center = np.mean(points, axis=0)
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    size = float(np.mean(extent) * 0.9)

    # создаём solid cube
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size,
        height=size,
        depth=size
    )

    # преобразуем в настоящий solid
    mesh = mesh.merge_close_vertices(0.0001)
    mesh.compute_vertex_normals()

    # перемещаем в правильное место
    mesh.translate(center - np.array([size/2, size/2, size/2]))

    return mesh

def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1)) * 0.9
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)
    mesh.translate(center)
    return mesh

def fit_cylinder(points):
    center = np.mean(points, axis=0)
    height = np.ptp(points[:, 2])
    radius = np.mean(np.linalg.norm(points[:, :2] - center[:2], axis=1)) * 0.9
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=30, split=4)
    mesh.translate(center)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0]), center=(0,0,0))
    return mesh

def fit_cone(points):
    center = np.mean(points, axis=0)
    height = np.ptp(points[:, 2])
    radius = np.max(np.linalg.norm(points[:, :2] - center[:2], axis=1)) * 0.9
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=30, split=4)
    mesh.translate(center)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]), center=(0,0,0))
    return mesh

def fit_torus(points):
    center = np.mean(points, axis=0)
    R = np.mean(np.linalg.norm(points[:, :2] - center[:2], axis=1))
    distances = np.linalg.norm(points - center, axis=1)
    r = np.std(distances) * 0.7

    if r < 0.01 or R <= r:
        r = 0.3
        R = 0.8

    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=R,
        tube_radius=r,
        radial_resolution=30,
        tubular_resolution=20
    )
    mesh.translate(center)
    return mesh

def fit_pyramid(points):
    base_z = np.min(points[:, 2])
    base = points[np.abs(points[:, 2] - base_z) < 0.1]
    apex = points[np.argmax(points[:, 2])]
    base_center = np.mean(base[:, :2], axis=0)
    size = np.max(base[:, :2], axis=0) - np.min(base[:, :2], axis=0)
    size = np.mean(size) * 0.9
    height = apex[2] - base_z

    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array([
        [base_center[0]-size/2, base_center[1]-size/2, 0],
        [base_center[0]+size/2, base_center[1]-size/2, 0],
        [base_center[0]+size/2, base_center[1]+size/2, 0],
        [base_center[0]-size/2, base_center[1]+size/2, 0],
        [base_center[0], base_center[1], height]
    ]) + [0, 0, base_z]
    triangles = np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

FITTERS = {
    "cube": fit_cube,
    "sphere": fit_sphere,
    "cylinder": fit_cylinder,
    "cone": fit_cone,
    "torus": fit_torus,
    "pyramid": fit_pyramid
}

# -------------------------------
# ЭКСПОРТ: .obj + .ply + .glb
# -------------------------------
def export_mesh(mesh, name):
    obj_path = SAVE_DIR / f"gan_fitted_{name}.obj"
    ply_path = SAVE_DIR / f"gan_fitted_{name}.ply"
    glb_path = SAVE_DIR / f"gan_fitted_{name}.glb"

    if mesh.has_triangle_normals():
        mesh.triangle_normals = o3d.utility.Vector3dVector()

    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    o3d.io.write_triangle_mesh(str(glb_path), mesh)

    print(f"    Saved: {obj_path.name} | {ply_path.name} | {glb_path.name}")

# -------------------------------
# ОСНОВНАЯ ЛОГИКА
# -------------------------------
def main():
    if not MODEL_PATH.exists():
        print(f"Error: Нет модели: {MODEL_PATH}")
        return

    generator = Generator(LATENT_DIM, COND_DIM, NUM_POINTS, len(CLASSES)).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    generator.eval()

    meshes = []
    combined = o3d.geometry.TriangleMesh()

    colors = [
        [0.5, 0.7, 1.0], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5],
        [1.0, 0.8, 0.3], [0.8, 0.5, 1.0], [1.0, 0.6, 0.8]
    ]

    for i, (class_id, name) in enumerate(CLASSES.items()):
        print(f"\n  [{i+1}/6] {name}")

        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        label = torch.tensor([class_id], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pts = generator(z, label).cpu().numpy()[0]

        if len(pts) == 0 or np.any(np.isnan(pts)):
            print("    Error: Пустое облако")
            continue

        try:
            mesh = FITTERS[name](pts)
            mesh.paint_uniform_color(colors[i])
            mesh.translate([i * 3.0, 0, 0])
            meshes.append(mesh)
            combined += mesh

            export_mesh(mesh, name)
        except Exception as e:
            print(f"    Error: Фиттинг не удался: {e}")

    # Сохраняем сцену
    scene_obj = SAVE_DIR / "gan_fitted_scene.obj"
    scene_ply = SAVE_DIR / "gan_fitted_scene.ply"
    scene_glb = SAVE_DIR / "gan_fitted_scene.glb"
    o3d.io.write_triangle_mesh(str(scene_obj), combined)
    o3d.io.write_triangle_mesh(str(scene_ply), combined)
    o3d.io.write_triangle_mesh(str(scene_glb), combined)
    print(f"\nSuccess: СЦЕНА: {scene_obj.name} | {scene_ply.name} | {scene_glb.name}")

    o3d.visualization.draw_geometries(meshes + [combined])

def main_test():
    if not MODEL_PATH.exists():
        print(f"Ошибка: нет модели: {MODEL_PATH}")
        return

    print("Загрузка генератора...")
    generator = Generator(LATENT_DIM, COND_DIM, NUM_POINTS, len(CLASSES)).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    generator.eval()

    print("\n=== Генерация OBJ моделей ===")

    for class_id, name in CLASSES.items():
        print(f"\n[{class_id}] Генерируем: {name}")

        # ----------------------------
        # 1. Генерация облака точек
        # ----------------------------
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        label = torch.tensor([class_id], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            pts = generator(z, label).cpu().numpy()[0]

        if pts.size == 0 or np.isnan(pts).any():
            print("Ошибка: GAN вернул пустое облако точек!")
            continue

        # ----------------------------
        # 2. Фиттинг → меш
        # ----------------------------
        try:
            mesh = FITTERS[name](pts)
            mesh.compute_vertex_normals()
        except Exception as e:
            print(f"Ошибка фиттинга фигуры {name}: {e}")
            continue

        # ----------------------------
        # 3. Сохранение OBJ
        # ----------------------------
        out_obj = OUTPUT_DIR / f"test_{name}.obj"
        o3d.io.write_triangle_mesh(str(out_obj), mesh)
        print(f"✔ Сохранён OBJ: {out_obj}")

        # ----------------------------
        # 4. ПОКАЗАТЬ МЕШ В ОКНЕ
        # ----------------------------
        print("Открываю предпросмотр окна...")
        o3d.visualization.draw_geometries([mesh])

    print("\n=== Готово ===")

if __name__ == "__main__":
    main()