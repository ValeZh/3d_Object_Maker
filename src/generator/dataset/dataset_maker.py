import trimesh
import os
import sqlite3
import random
import glob
from PIL import Image
import numpy as np
import uuid
import shutil

# === Базовые пути ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# === Папки ===
OUTPUT_DIR = os.path.join(DATA_DIR, "obj_for_learn")
TEXTURES_DIR = os.path.join(DATA_DIR, "textures")

# === Пути к БД ===
DB_PATH = os.path.join(DATA_DIR, "objects_data.db")

# === Проверка папок ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXTURES_DIR, exist_ok=True)

# === Цвета и материалы ===
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "cyan"]
TEXTURES = ["wood", "stone", "metallic"]

COLOR_MAP = {
    "red": [1.0, 0.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "orange": [1.0, 0.5, 0.0],
    "pink": [1.0, 0.4, 0.7],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
}


# === Инициализация БД ===
def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEXTURES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Создание таблиц
    c.execute("""
        CREATE TABLE IF NOT EXISTS shapes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS textures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_id INTEGER NOT NULL,
            color TEXT NOT NULL,
            texture_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            obj_data BLOB NOT NULL,
            mtl_data BLOB NOT NULL,
            description TEXT NOT NULL,
            FOREIGN KEY(shape_id) REFERENCES shapes(id) ON DELETE CASCADE,
            FOREIGN KEY(texture_id) REFERENCES textures(id) ON DELETE CASCADE
        )
    """)

    # Заполнение shapes
    shapes = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid"]
    existing_shapes = [row[0] for row in c.execute("SELECT name FROM shapes").fetchall()]
    for s in shapes:
        if s not in existing_shapes:
            c.execute("INSERT INTO shapes (name) VALUES (?)", (s,))

    # Заполнение textures
    existing_textures = [row[0] for row in c.execute("SELECT name FROM textures").fetchall()]
    for t in TEXTURES:
        if t not in existing_textures:
            c.execute("INSERT INTO textures (name) VALUES (?)", (t,))

    conn.commit()
    conn.close()


# === Вспомогательная функция для выбора текстуры ===
def get_texture_file(texture_name):
    matches = glob.glob(os.path.join(TEXTURES_DIR, f"{texture_name}*.jpg")) + \
              glob.glob(os.path.join(TEXTURES_DIR, f"{texture_name}*.png"))
    return random.choice(matches) if matches else None


# === Генерация UV для корректного отображения (на уровне вершин) ===
def generate_uv_for_mesh(mesh, shape_name="generic"):
    """
    Возвращает uv массив формы (n_vertices, 2).
    Для простых примитивов используем разные проекции.
    """
    verts = np.asarray(mesh.vertices)  # (N,3)
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # Prepare u,v arrays
    u = np.zeros(len(verts), dtype=float)
    v = np.zeros(len(verts), dtype=float)

    eps = 1e-9

    if shape_name == "sphere":
        # Spherical coords
        r = np.linalg.norm(verts, axis=1)
        r = np.where(r == 0, eps, r)
        u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
        v = 0.5 - np.arcsin(y / r) / np.pi

    elif shape_name == "torus":
        # approximate torus mapping: angle around Y -> u, around minor circle -> v
        # estimate major radius as mean of sqrt(x^2+z^2)
        radial = np.sqrt(x * x + z * z)
        major = np.mean(radial)
        # u: angle around Y
        u = (np.arctan2(z, x) + np.pi) / (2 * np.pi)
        # v: angle around minor circle (approx)
        minor = np.sqrt((radial - major) ** 2 + y * y)
        # avoid division by zero
        minor = np.where(minor == 0, eps, minor)
        v = 0.5 + np.arctan2(y, radial - major) / (2 * np.pi)

    elif shape_name in ("cylinder", "cone"):
        # Cylinder: u by angle around Y, v by height (Y)
        u = (np.arctan2(z, x) + np.pi) / (2 * np.pi)
        ymin = y.min()
        ymax = y.max()
        denom = (ymax - ymin) if (ymax - ymin) > eps else 1.0
        v = (y - ymin) / denom

    else:
        # Default: planar projection (XZ)
        xmin, xmax = x.min(), x.max()
        zmin, zmax = z.min(), z.max()
        denom_x = (xmax - xmin) if (xmax - xmin) > eps else 1.0
        denom_z = (zmax - zmin) if (zmax - zmin) > eps else 1.0
        u = (x - xmin) / denom_x
        v = (z - zmin) / denom_z

    # ensure u/v in [0,1]
    u = np.mod(u, 1.0)
    v = np.clip(v, 0.0, 1.0)

    uv = np.stack([u, v], axis=1)
    return uv


# === Сохранение сетки с MTL (уникальные материалы и копии текстур) ===
def save_mesh_with_mtl(mesh, shape_name, color, texture_name, return_texture_file=False):
    # Если texture_name задан, ищем файл текстуры
    texture_file = get_texture_file(texture_name) if texture_name else None
    uv = generate_uv_for_mesh(mesh, shape_name=shape_name)

    # Уникальные имена материалов и текстур
    unique_id = uuid.uuid4().hex[:8]
    material_name = f"{shape_name}_{color}_{texture_name}_{unique_id}" if texture_name else f"{shape_name}_{color}_{unique_id}"

    # Копирование текстуры только если она есть
    texture_file_rel = None
    if texture_file and os.path.exists(texture_file):
        ext = os.path.splitext(texture_file)[1]
        texture_copy_name = f"{texture_name}_{unique_id}{ext}"
        texture_copy_path = os.path.join(OUTPUT_DIR, texture_copy_name)
        shutil.copyfile(texture_file, texture_copy_path)
        texture_file_rel = texture_copy_name

    # Визуальные данные
    if texture_file_rel:
        img = Image.open(os.path.join(OUTPUT_DIR, texture_file_rel))
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)
    else:
        rgba = [int(c * 255) for c in COLOR_MAP[color]] + [255]
        face_colors = np.tile(rgba, (len(mesh.faces), 1))
        mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)

    # Пути файлов
    file_name = f"{shape_name}_{color}_{texture_name}_{unique_id}.obj" if texture_name else f"{shape_name}_{color}_{unique_id}.obj"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    mtl_name = f"{shape_name}_{color}_{texture_name}_{unique_id}.mtl" if texture_name else f"{shape_name}_{color}_{unique_id}.mtl"
    mtl_path = os.path.join(OUTPUT_DIR, mtl_name)

    mesh.export(file_path)

    # Создание MTL
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Kd {COLOR_MAP[color][0]} {COLOR_MAP[color][1]} {COLOR_MAP[color][2]}\n")
        f.write("Ka 0 0 0\nKs 0 0 0\nNs 10\n")
        if texture_file_rel:
            f.write(f"map_Kd {texture_file_rel}\n")

    # Ссылка на MTL в OBJ
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\nusemtl {material_name}\n")
        f.writelines(lines)

    # Чтение бинарных данных
    with open(file_path, "rb") as f:
        obj_data = f.read()
    with open(mtl_path, "rb") as f:
        mtl_data = f.read()

    if return_texture_file:
        return file_name, obj_data, mtl_data, texture_file
    else:
        return file_name, obj_data, mtl_data


# === Описание для БД ===
def create_description(shape_name, color, texture_name, texture_file):
    if texture_file:
        return f"{shape_name}, {color}, с текстурой {texture_name}"
    else:
        return f"{shape_name}, {color}, без текстуры"


# === Добавление объекта в БД ===
def add_object_to_db(shape_id, color, texture_id, file_name, obj_data, mtl_data, description):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO objects (shape_id, color, texture_id, file_name, obj_data, mtl_data, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (shape_id, color, texture_id, file_name, obj_data, mtl_data, description))
    conn.commit()
    conn.close()


# === Генерация примитивов ===
def generate_primitives():
    conn = sqlite3.connect(DB_PATH)
    shapes = conn.execute("SELECT id, name FROM shapes").fetchall()
    textures = conn.execute("SELECT id, name FROM textures").fetchall()
    conn.close()

    shape_generators = {
        "cube": lambda: trimesh.creation.box([1, 1, 1]),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.0),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1.5, sections=64),
        "cone": lambda: trimesh.creation.cone(radius=0.5, height=1.5, sections=64),
        "torus": lambda: trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_segments=64, minor_segments=32),
        "pyramid": lambda: trimesh.creation.cone(radius=0.8, height=1.2, sections=4)  # pyramid as 4-sided cone
    }

    for shape_id, shape_name in shapes:
        color = random.choice(COLORS)
        texture_id, texture_name = random.choice(textures)
        mesh = shape_generators[shape_name]()

        file_name, obj_data, mtl_data = save_mesh_with_mtl(mesh, shape_name, color, texture_name)
        texture_file = get_texture_file(texture_name)
        description = create_description(shape_name, color, texture_name, texture_file)

        add_object_to_db(shape_id, color, texture_id, file_name, obj_data, mtl_data, description)
        print(f"[OK] {shape_name} | Color: {color} | Texture: {texture_name} | Description: {description}")

def create_single_object(attributes):
    shape_name = attributes.get("shape")
    color = attributes.get("color", random.choice(COLORS))
    texture_name = attributes.get("texture")  # <-- убрали random.choice(TEXTURES)

    # Получаем ID фигуры
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT id FROM shapes WHERE name=?", (shape_name,)).fetchone()
    conn.close()
    shape_id = row[0]

    # Получаем ID текстуры (если texture_name задан)
    texture_id = 0
    if texture_name:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        row = c.execute("SELECT id FROM textures WHERE name=?", (texture_name,)).fetchone()
        conn.close()
        if not row:
            raise ValueError(f"Texture '{texture_name}' not found in DB")
        texture_id = row[0]

    # Создание примитива
    shape_generators = {
        "cube": lambda: trimesh.creation.box([1, 1, 1]),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.0),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1.5, sections=64),
        "cone": lambda: trimesh.creation.cone(radius=0.5, height=1.5, sections=64),
        "torus": lambda: trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_segments=64, minor_segments=32),
        "pyramid": lambda: trimesh.creation.cone(radius=0.8, height=1.2, sections=4)
    }

    if shape_name not in shape_generators:
        raise ValueError(f"Shape '{shape_name}' not supported")

    mesh = shape_generators[shape_name]()

    # Сохраняем mesh
    file_name, obj_data, mtl_data, used_texture_file = save_mesh_with_mtl(
        mesh, shape_name, color, texture_name, return_texture_file=True
    )
    description = create_description(shape_name, color, texture_name, used_texture_file)

    # Добавление объекта
    add_object_to_db(shape_id, color, texture_id, file_name, obj_data, mtl_data, description)

    print(f"[SINGLE OK] Created {shape_name} | Color: {color} | Texture: {texture_name}")


# === Запуск ===
if __name__ == "__main__":
    print("🔧 Initializing DB…")
    init_db()
    print("🎨 Generating initial dataset…")
    generate_primitives()
