# gan_mesh_factory.py
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

from reconstruct_meshes import Generator, FITTERS, LATENT_DIM, COND_DIM, NUM_POINTS, DEVICE, CLASSES, MODEL_PATH


# ===============================
# Функция генерации точек + фиттинг
# ===============================
def generate_mesh_from_points(shape_name: str):
    """
    shape_name: 'cube', 'sphere', 'torus' и т.д.
    Возвращает: open3d.geometry.TriangleMesh
    """

    # ищем id класса
    class_id = None
    for k, n in CLASSES.items():
        if n == shape_name:
            class_id = k
            break

    if class_id is None:
        raise ValueError(f"[GAN] Нет класса: {shape_name}")

    # загружаем модель
    generator = Generator(LATENT_DIM, COND_DIM, NUM_POINTS, len(CLASSES)).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    generator.eval()

    # 1️⃣ генерируем точки
    z = torch.randn(1, LATENT_DIM, device=DEVICE)
    label = torch.tensor([class_id], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pts = generator(z, label).cpu().numpy()[0]

    if pts.size == 0 or np.isnan(pts).any():
        raise RuntimeError("[GAN] Пустое облако точек!")

    # 2️⃣ фиттинг → меш
    mesh = FITTERS[shape_name](pts)
    mesh.compute_vertex_normals()

    return mesh


generate_mesh_from_points("cube")