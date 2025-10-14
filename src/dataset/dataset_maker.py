import trimesh
import os
import sqlite3
import random

# Папка для сохранения obj
OUTPUT_DIR = r"C:\Users\lasta\Projekt3d\3d_Object_Maker\data\simple_shapes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Путь к БД
DB_PATH = r"D:\4course_1sem\semestr_project_experiments\experimenrs\db\objects.db"

COLORS = ["red", "blue", "green", "yellow", "purple"]
TEXTURES = ["glossy", "matte", "metallic"]

# Цвета для материала (RGB, 0-1)
COLOR_MAP = {
    "red": [1.0, 0.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 0.5]
}


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape TEXT NOT NULL,
            color TEXT,
            texture TEXT,
            file_name TEXT NOT NULL,
            obj_data BLOB NOT NULL,
            mtl_data BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_mesh_with_mtl(mesh, name, color, texture):
    """Сохраняем OBJ и создаём соответствующий MTL для корректного отображения цвета в Blender"""
    file_name = f"{name}_{color}_{texture}.obj"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    mtl_name = f"{name}_{color}_{texture}.mtl"
    mtl_path = os.path.join(OUTPUT_DIR, mtl_name)

    # Экспортируем OBJ без vertex_colors
    mesh.export(file_path)

    # Создаём MTL
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {color}_{texture}\n")
        f.write(f"Kd {COLOR_MAP[color][0]} {COLOR_MAP[color][1]} {COLOR_MAP[color][2]}\n")
        f.write("Ka 0 0 0\n")
        f.write("Ks 0 0 0\n")
        f.write("Ns 10\n")

    # Добавляем ссылку на MTL в OBJ
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"usemtl {color}_{texture}\n")
        f.writelines(lines)

    # Читаем бинарные данные OBJ и MTL
    with open(file_path, "rb") as f:
        obj_data = f.read()
    with open(mtl_path, "rb") as f:
        mtl_data = f.read()

    return file_name, obj_data, mtl_data


def add_object_to_db(shape, color, texture, file_name, obj_data, mtl_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO objects (shape, color, texture, file_name, obj_data, mtl_data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (shape, color, texture, file_name, obj_data, mtl_data))
    conn.commit()
    conn.close()


def generate_primitives():
    primitives = [
        ("cube", lambda: trimesh.creation.box([1, 1, 1])),
        ("sphere", lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.0)),
        ("cylinder", lambda: trimesh.creation.cylinder(radius=0.5, height=1.5, sections=32)),
        ("cone", lambda: trimesh.creation.cone(radius=0.5, height=1.5, sections=32)),
        ("torus",
         lambda: trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_segments=32, minor_segments=16)),
        ("pyramid", lambda: trimesh.creation.cone(radius=0.8, height=1.2, sections=4))
    ]

    for name, mesh_func in primitives:
        color = random.choice(COLORS)
        texture = random.choice(TEXTURES)

        mesh = mesh_func()

        file_name, obj_data, mtl_data = save_mesh_with_mtl(mesh, name, color, texture)
        add_object_to_db(name, color, texture, file_name, obj_data, mtl_data)
        print(f"[OK] {name} | Color: {color} | Texture: {texture}")


if __name__ == "__main__":
    init_db()
    generate_primitives()
    print("✅ Примитивы с материалами сгенерированы и добавлены в БД.")
