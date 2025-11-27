import sys
from pathlib import Path

# ======================
# Пути проекта
# ======================
ROOT = Path(__file__).resolve().parents[1]   # 3d_Object_Maker/
SRC = ROOT / "src"
sys.path.append(str(SRC))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

# Импорт функции генерации 3D объекта
from generator.ai.gan_mesh_factory import create_shape

# Опции
from api.options import (
    get_available_shapes,
    get_available_textures,
    get_available_colors
)

# Генерация 3D
from generator.ai.gan_object_factory import create_gan_object

# ZIP
from zipper.zipper import make_zip

# Пути
from config.paths import (
    ROOT as PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    TEXTURES_DIR
)

app = FastAPI()


# ============================================
# OPTIONS ENDPOINT — фронтенд забирает формы, цвета, текстуры
# ============================================
@app.get("/api/options")
async def api_options():
    return {
        "shapes": get_available_shapes(),
        "textures": get_available_textures(),
        "colors": get_available_colors()
    }


# ============================================
# TEXT → SHAPE/COLOR/TEXTURE
# ============================================
@app.post("/api/generate-from-text")
async def generate_from_text(payload: dict):

    text = payload.get("text", "").lower()

    # shape
    shape = next((s for s in get_available_shapes() if s.lower() in text), "cube")

    # color
    color = next((c for c in get_available_colors() if c.lower() in text), "white")

    # texture
    texture = next((t for t in get_available_textures() if t.lower() in text), None)

    # генерируем объект
    result = create_shape(shape, color, texture)


    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    # текстуры, если есть
    tex_paths = []
    for ext in ["jpg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    # ZIP
    zip_path = obj_path.parent / f"{shape}_{color}.zip"
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    return JSONResponse({
        "shape": shape,
        "color": color,
        "texture": texture,
        "zip_url": f"/files/{zip_path.relative_to(PROJECT_ROOT)}",
        "obj_url": f"/files/{obj_path.relative_to(PROJECT_ROOT)}",
        "mtl_url": f"/files/{mtl_path.relative_to(PROJECT_ROOT)}",
        "textures": [f"/files/{p.relative_to(PROJECT_ROOT)}" for p in tex_paths]
    })


# ============================================
# GENERATE OBJECT из параметров shape+color+texture
# ============================================
@app.post("/api/generate-object")
async def generate_object(payload: dict):

    shape = payload.get("shape")
    color = payload.get("color")
    texture = payload.get("texture")

    result = create_gan_object(
        shape=shape,
        color=color,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    tex_paths = []
    for ext in ["jpg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    zip_path = obj_path.parent / f"{shape}_{color}.zip"
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    return JSONResponse({
        "zip_url": f"/files/{zip_path.relative_to(PROJECT_ROOT)}",
        "obj_url": f"/files/{obj_path.relative_to(PROJECT_ROOT)}",
        "mtl_url": f"/files/{mtl_path.relative_to(PROJECT_ROOT)}",
        "textures": [f"/files/{p.relative_to(PROJECT_ROOT)}" for p in tex_paths]
    })


# ============================================
# FILE SERVER
# ============================================
@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    file_location = PROJECT_ROOT / file_path
    return FileResponse(file_location)
