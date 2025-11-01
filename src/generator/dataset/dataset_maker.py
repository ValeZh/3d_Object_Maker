import trimesh
import os
import sqlite3
import random
import glob
from PIL import Image

# === Базовые пути ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# === Папки ===
OUTPUT_DIR = os.path.join(DATA_DIR, "obj_for_learn")
TEXTURES_DIR = os.path.join(DATA_DIR, "textures")

# === Пути к БД ===
OBJECTS_DB = os.path.join(DATA_DIR, "objects.db")
SHAPES_DB = os.path.join(DATA_DIR, "shapes.db")
TEXTURES_DB = os.path.join(DATA_DIR, "textures.db")

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

    # shapes.db
    conn = sqlite3.connect(SHAPES_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS shapes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)""")
    shapes = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid"]
    existing = [row[0] for row in c.execute("SELECT name FROM shapes").fetchall()]
    for s in shapes:
        if s not in existing:
            c.execute("INSERT INTO shapes (name) VALUES (?)", (s,))
    conn.commit()
    conn.close()

    # textures.db
    conn = sqlite3.connect(TEXTURES_DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS textures (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)""")
    existing = [row[0] for row in c.execute("SELECT name FROM textures").fetchall()]
    for t in TEXTURES:
        if t not in existing:
            c.execute("INSERT INTO textures (name) VALUES (?)", (t,))
    conn.commit()
    conn.close()

    # objects.db
    conn = sqlite3.connect(OBJECTS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_id INTEGER NOT NULL,
            color TEXT NOT NULL,
            texture_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            obj_data BLOB NOT NULL,
            mtl_data BLOB NOT NULL,
            description TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# === Вспомогательная функция для выбора текстуры ===
def get_texture_file(texture_name):
    matches = glob.glob(os.path.join(TEXTURES_DIR, f"{texture_name}*.jpg")) + \
              glob.glob(os.path.join(TEXTURES_DIR, f"{texture_name}*.png"))
    return random.choice(matches) if matches else None

def save_mesh_with_mtl(mesh, shape_name, color, texture_name):
    texture_file = get_texture_file(texture_name)  # ищем файл текстуры

    # Добавляем UV и текстуру, если есть
    if texture_file and os.path.exists(texture_file):
        img = Image.open(texture_file)
        # Простая UV проекция
        uv = mesh.vertices[:, :2]
        uv = (uv - uv.min(axis=0)) / (uv.max(axis=0) - uv.min(axis=0) + 1e-9)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)
    else:
        # Если текстуры нет, просто цвет
        mesh.visual = trimesh.visual.ColorVisuals(face_colors=[int(COLOR_MAP[color][0]*255),
                                                              int(COLOR_MAP[color][1]*255),
                                                              int(COLOR_MAP[color][2]*255), 255])

    file_name = f"{shape_name}_{color}_{texture_name}.obj"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    mtl_name = f"{shape_name}_{color}_{texture_name}.mtl"
    mtl_path = os.path.join(OUTPUT_DIR, mtl_name)

    # Экспортируем объект
    mesh.export(file_path)

    # Создаём .mtl
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {color}_{texture_name}\n")
        f.write(f"Kd {COLOR_MAP[color][0]} {COLOR_MAP[color][1]} {COLOR_MAP[color][2]}\n")
        f.write("Ka 0 0 0\nKs 0 0 0\nNs 10\n")
        if texture_file and os.path.exists(texture_file):
            texture_file_rel = os.path.relpath(texture_file, OUTPUT_DIR).replace("\\", "/")
            f.write(f"map_Kd {texture_file_rel}\n")

    # Добавляем ссылку на материал в .obj
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        f.write(f"mtllib {mtl_name}\nusemtl {color}_{texture_name}\n")
        f.writelines(lines)

    # Чтение бинарных данных
    with open(file_path, "rb") as f:
        obj_data = f.read()
    with open(mtl_path, "rb") as f:
        mtl_data = f.read()

    return file_name, obj_data, mtl_data

# === Сохранение сетки с MTL ===
def save_mesh_with_mtl(mesh, shape_name, color, texture_name):
    # Добавляем UV, если их нет
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        # Простейшая UV-развёртка на основе проекции
        uv = mesh.vertices[:, :2]  # используем X и Y как UV
        uv = (uv - uv.min(axis=0)) / (uv.max(axis=0) - uv.min(axis=0) + 1e-9)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)

    file_name = f"{shape_name}_{color}_{texture_name}.obj"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    mtl_name = f"{shape_name}_{color}_{texture_name}.mtl"
    mtl_path = os.path.join(OUTPUT_DIR, mtl_name)

    # Экспортируем геометрию
    mesh.export(file_path)

    # Путь к текстуре
    texture_file = get_texture_file(texture_name)

    # Создаём .mtl
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {color}_{texture_name}\n")
        f.write(f"Kd {COLOR_MAP[color][0]} {COLOR_MAP[color][1]} {COLOR_MAP[color][2]}\n")
        f.write("Ka 0 0 0\nKs 0 0 0\nNs 10\n")

        if texture_file and os.path.exists(texture_file):
            texture_file_rel = os.path.relpath(texture_file, OUTPUT_DIR).replace("\\", "/")
            f.write(f"map_Kd {texture_file_rel}\n")

    # Добавляем ссылку на материал в .obj
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        f.write(f"mtllib {mtl_name}\nusemtl {color}_{texture_name}\n")
        f.writelines(lines)

    # Чтение бинарных данных
    with open(file_path, "rb") as f:
        obj_data = f.read()
    with open(mtl_path, "rb") as f:
        mtl_data = f.read()

    return file_name, obj_data, mtl_data

def create_description(shape_name, color, texture_name, texture_file):
    if texture_file:
        return f"{shape_name}, {color}, с текстурой {texture_name}"
    else:
        return f"{shape_name}, {color}, без текстуры"

# === Добавление объекта в БД ===
def add_object_to_db(shape_id, color, texture_id, file_name, obj_data, mtl_data, description):
    conn = sqlite3.connect(OBJECTS_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO objects (shape_id, color, texture_id, file_name, obj_data, mtl_data, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (shape_id, color, texture_id, file_name, obj_data, mtl_data, description))
    conn.commit()
    conn.close()


# === Генерация примитивов ===
def generate_primitives():
    conn_shapes = sqlite3.connect(SHAPES_DB)
    shapes = conn_shapes.execute("SELECT id, name FROM shapes").fetchall()
    conn_shapes.close()

    conn_textures = sqlite3.connect(TEXTURES_DB)
    textures = conn_textures.execute("SELECT id, name FROM textures").fetchall()
    conn_textures.close()

    shape_generators = {
        "cube": lambda: trimesh.creation.box([1, 1, 1]),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=1.0),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1.5, sections=32),
        "cone": lambda: trimesh.creation.cone(radius=0.5, height=1.5, sections=32),
        "torus": lambda: trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_segments=32, minor_segments=16),
        "pyramid": lambda: trimesh.creation.cone(radius=0.8, height=1.2, sections=4)
    }

    for shape_id, shape_name in shapes:
        color = random.choice(COLORS)
        texture_id, texture_name = random.choice(textures)
        mesh = shape_generators[shape_name]()

        file_name, obj_data, mtl_data = save_mesh_with_mtl(mesh, shape_name, color, texture_name)

        # Проверяем, есть ли текстура для описания
        texture_file = get_texture_file(texture_name)
        description = create_description(shape_name, color, texture_name, texture_file)

        # Добавляем объект в БД с описанием
        add_object_to_db(shape_id, color, texture_id, file_name, obj_data, mtl_data, description)

        print(f"[OK] {shape_name} | Color: {color} | Texture: {texture_name} | Description: {description}")


# === Запуск ===
if __name__ == "__main__":
    print("🔧 Инициализация базы данных...")
    init_db()
    print("🎨 Генерация примитивов...")
    generate_primitives()
    print("✅ Примитивы успешно сгенерированы и добавлены в объекты.")
