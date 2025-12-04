#!/usr/bin/env python3
# pointcloud_generator.py
# Генерация 3D-хмар точок з Conditional GAN

import torch
import sqlite3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Импорт моделей и констант из тренера ---
from pointcloud_cgan import (
    Generator,
    LATENT_DIM,
    COND_DIM,
    NUM_POINTS,
    DEVICE,
    save_pointcloud_ply,   # если нужен экспорт
    save_pc_image,         # визуализация в PNG
)

# ==============================
# 1. Пути
# ==============================
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "data" / "point_cgan_pointnet_output" / "model_final.pt"
DB_PATH = ROOT / "data" / "objects_data.db"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❗ Модель не найдена: {MODEL_PATH}")


# ==============================
# 2. Классы из БД
# ==============================
def load_classes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    rows = c.execute(
        "SELECT id, name FROM shapes ORDER BY id ASC"
    ).fetchall()
    conn.close()

    classes = {i: row[1] for i, row in enumerate(rows)}
    inv = {v: k for k, v in classes.items()}
    return classes, inv


CLASSES, INV_CLASSES = load_classes()
print("✔ Классы из БД:", CLASSES)


# ==============================
# 3. Загружаем GAN
# ==============================
generator = Generator(
    z_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    num_points=NUM_POINTS,
    num_classes=len(CLASSES)
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt["G"] if "G" in ckpt else ckpt
generator.load_state_dict(state)
generator.eval()

print(f"✔ Генератор загружен из {MODEL_PATH.name}")


# ==============================
# 4. Генерация облака точек
# ==============================
def generate_pointcloud(shape_name: str) -> np.ndarray:
    """
    shape_name — строка ("cube", "sphere" ...)
    Возвращает numpy array N×3
    """
    if shape_name not in INV_CLASSES:
        raise ValueError(
            f"Форма '{shape_name}' не найдена.\n"
            f"Доступно: {list(INV_CLASSES.keys())}"
        )

    cid = INV_CLASSES[shape_name]
    z = torch.randn(1, LATENT_DIM, device=DEVICE)
    label = torch.tensor([cid], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pts = generator(z, label).cpu().numpy()[0]

    return pts


# ==============================
# 5. Показ как в оригинале
# ==============================
def show_all_classes():
    """
    Визуализация всех классов одной фигурой (2x3 или 3xX)
    Каждая хмара точек — scatter plot
    """
    fig = plt.figure(figsize=(15, 10))

    for i, (cid, name) in enumerate(CLASSES.items()):
        pts = generate_pointcloud(name)

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=3,
            c=pts[:, 2],
            cmap="viridis"
        )
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ==============================
# 6. CLI демо
# ==============================
if __name__ == "__main__":
    show_all_classes()
