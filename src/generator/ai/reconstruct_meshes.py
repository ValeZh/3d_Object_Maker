#!/usr/bin/env python3
# reconstruct_meshes.py — GAN → ідеальні примітиви → OBJ/PLY/GLB

import torch
import numpy as np
import open3d as o3d
from pathlib import Path


# ==========================================================
# 📌 ШЛЯХИ
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "point_cgan_pointnet_output"
MODEL_PATH = OUTPUT_DIR / "model_final.pt"

SAVE_DIR = DATA_DIR / "generated_obj"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 📌 ПАРАМЕТРИ (повтор з CGAN)
# ==========================================================
LATENT_DIM = 128
COND_DIM = 64
NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = {
    0: "cube",
    1: "sphere",
    2: "cylinder",
    3: "cone",
    4: "torus",
    5: "pyramid"
}


# ==========================================================
# 📌 Генератор (ідентичний pointcloud_cgan)
# ==========================================================
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
        return self.net(x).view(-1, self.num_points, 3)


# ==========================================================
# 📌 ФІТТЕРИ ПРИМІТИВІВ
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
    height = np.ptp(points[:, 2])
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
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]))
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
    base = points[np.abs(points[:, 2] - base_z) < 0.1]

    apex = points[np.argmax(points[:, 2])]
    center = np.mean(base[:, :2], axis=0)

    size = np.mean(np.max(base[:, :2], axis=0) - np.min(base[:, :2], axis=0))
    height = apex[2] - base_z

    mesh = o3d.geometry.TriangleMesh()
    v = np.array([
        [center[0]-size/2, center[1]-size/2, base_z],
        [center[0]+size/2, center[1]-size/2, base_z],
        [center[0]+size/2, center[1]+size/2, base_z],
        [center[0]-size/2, center[1]+size/2, base_z],
        [center[0],        center[1],      base_z+height]
    ])
    t = np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(t)
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


# ==========================================================
# 📌 ЕКСПОРТ
# ==========================================================
def export_mesh(mesh: o3d.geometry.TriangleMesh, name: str):
    out = {
        "obj": SAVE_DIR / f"gan_{name}.obj",
        "ply": SAVE_DIR / f"gan_{name}.ply",
        "glb": SAVE_DIR / f"gan_{name}.glb",
    }
    for k, p in out.items():
        o3d.io.write_triangle_mesh(str(p), mesh)
    print(f"  ✔ Saved {name}: OBJ/PLY/GLB")


# ==========================================================
# 📌 Головний запуск
# ==========================================================
def main():
    if not MODEL_PATH.exists():
        print("\n❌ No CGAN model found!")
        return

    generator = Generator(LATENT_DIM, COND_DIM, NUM_POINTS, len(CLASSES)).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["G"])
    generator.eval()

    combined = o3d.geometry.TriangleMesh()
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
        print(f"\n[{cid}] Generate {name}…")

        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        label = torch.tensor([cid], device=DEVICE)

        pts = generator(z, label).detach().cpu().numpy()[0]
        if not len(pts) or np.isnan(pts).any():
            print("❌ Empty cloud")
            continue

        mesh = FITTERS[name](pts)
        mesh.paint_uniform_color(colors[i])
        mesh.translate([i * 3.0, 0, 0])

        combined += mesh
        meshes.append(mesh)

        export_mesh(mesh, name)

    o3d.visualization.draw_geometries(meshes + [combined])


if __name__ == "__main__":
    main()
