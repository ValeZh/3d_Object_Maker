#!/usr/bin/env python3
# pointcloud_generator.py — ГОТОВЫЙ К ИМПОРТУ МОДУЛЬ
# Теперь можно: from pointcloud_generator import generator, CLASSES

import torch
import numpy as np
import open3d as o3d
from pathlib import Path

# -------------------------------
# КОНФИГ (дублируем только нужное)
# -------------------------------
from pointcloud_cgan import (
    Generator,
    LATENT_DIM,
    COND_DIM,
    NUM_POINTS,
    DEVICE,
)

# -------------------------------
# ПУТЬ К МОДЕЛИ
# -------------------------------

MODEL_PATH = Path(__file__).resolve().parents[3]/"data/point_cgan_pointnet_output/model_final.pt"

# -------------------------------
# КЛАССЫ
# -------------------------------
CLASSES = {0: "cube", 1: "sphere", 2: "cylinder", 3: "cone", 4: "torus", 5: "pyramid"}
INV_CLASSES = {v: k for k, v in CLASSES.items()}

# -------------------------------
# СОЗДАЁМ И ЗАГРУЖАЕМ ГЕНЕРАТОР (экспортируем!)
# -------------------------------
generator = Generator(
    z_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    num_points=NUM_POINTS,
    num_classes=len(CLASSES)
).to(DEVICE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
generator.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
generator.eval()

print(f"Генератор загружен из: {MODEL_PATH.name}")

# -------------------------------
# ВИЗУАЛИЗАЦИЯ (по желанию, можно оставить)
# -------------------------------
def show_all_classes():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 10))
    for i, (class_id, name) in enumerate(CLASSES.items()):
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        label = torch.tensor([class_id], device=DEVICE)
        with torch.no_grad():
            pts = generator(z, label).cpu().numpy()[0]
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, c=pts[:,2], cmap='viridis')
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_all_classes()