# gan_mesh_factory.py
import torch
import numpy as np
import open3d as o3d

# Импортируем из ai
from config.paths import OUTPUT_DIR, TEXTURES_DIR
from pathlib import Path

# ===============================
# Функция генерации точек + фиттинг
# ===============================
def generate_mesh_from_points(shape_name: str):
    shape_name = shape_name.lower()

    """
    shape_name: 'cube', 'sphere', 'torus' и т.д.
    Возвращает: open3d.geometry.TriangleMesh
    """
    from generator.ai.reconstruct_meshes import Generator, FITTERS, LATENT_DIM, COND_DIM, NUM_POINTS, DEVICE, CLASSES, MODEL_PATH
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

def create_shape(shape: str, color: str | list, texture: str | None):
    """
    shape: 'cube', 'sphere', ...
    color: строка из COLOR_MAP или RGB [0..1]
    texture: имя текстуры в папке textures, либо None
    """
    from generator.ai.gan_object_factory import create_gan_object

    # Генерация и сохранение obj/mtl
    result = create_gan_object(
        shape=shape,
        color=color,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )
    return result

if __name__ == "__main__":
    generate_mesh_from_points("cube")