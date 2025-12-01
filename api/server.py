from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

# Настройка логов
logger = logging.getLogger("uvicorn.info")

# === FastAPI app ===
app = FastAPI()

# === Импорт после app ===
from src.generator.ai.gan_mesh_factory import create_shape
from src.generator.ai.gan_object_factory import create_gan_object
from zipper.zipper import make_zip
# from api.options import get_available_shapes, get_available_textures, get_available_colors
from src.config.paths import ROOT as PROJECT_ROOT, OUTPUT_DIR, TEXTURES_DIR
from generator.dataset.ai_text import extract_attributes
from fastapi.responses import JSONResponse
from src.generator.dataset.db_editing import get_shapes, get_textures


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]


# === API маршруты ===
@app.get("/api/options")
async def api_options():
    logger.info("Options requested")
    shapes = get_shapes()      # список форм из БД
    textures = get_textures()  # список текстур из БД

    # Цвета можно оставить статичными или добавить отдельную таблицу в БД, если нужно
    colors = ["red", "green", "blue", "yellow", "black", "white", "brown", "cyan"]

    return JSONResponse({
        "shapes": shapes,
        "textures": textures,
        "colors": colors
    })


@app.get("/api/available-options")
async def available_options():
    """
    Возвращает список доступных форм и текстур из базы данных
    """
    shapes = get_shapes()
    textures = get_textures()

    return JSONResponse({
        "shapes": shapes,
        "textures": textures
    })

@app.post("/api/generate-object")
async def generate_object(payload: dict):
    shape = payload.get("shape")
    color = payload.get("color")
    texture = payload.get("texture")

    if isinstance(color, str) and color.startswith("#"):
        color_rgb = hex_to_rgb(color)
    else:
        color_rgb = color

    logger.info(f"Generate button pressed")
    logger.info(f"Shape selected: {shape}")
    logger.info(f"Texture selected: {texture}")
    logger.info(f"Color selected: {color}")

    # Генерация объекта через GAN
    result = create_gan_object(
        shape=shape,
        color=color_rgb,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    # Ищем текстуры
    tex_paths = []
    for ext in ["jpg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    # Создаем ZIP
    zip_path = obj_path.parent / f"{shape}_{texture}_{color.lstrip('#')}.zip"
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    logger.info(f"Object generated and zipped: {zip_path.name}")

    return JSONResponse({
        "zip_url": f"/files/{zip_path.relative_to(PROJECT_ROOT)}",
        "obj_url": f"/files/{obj_path.relative_to(PROJECT_ROOT)}",
        "mtl_url": f"/files/{mtl_path.relative_to(PROJECT_ROOT)}",
        "textures": [f"/files/{p.relative_to(PROJECT_ROOT)}" for p in tex_paths]
    })


@app.post("/api/generate-from-text")
async def generate_from_text(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    logger.info(f"Generate button pressed (TEXT MODE)")
    logger.info(f"Text description received: \"{text}\"")

    # --- Извлекаем shape, color, texture ---
    attrs = extract_attributes(text)
    logger.info(f"Extracted attributes from text: {attrs}")

    shape = attrs.get("shape")
    color = attrs.get("color")
    texture = attrs.get("texture")

    if not shape and not color and not texture:
        return JSONResponse(
            {"error": "Could not extract any attributes from text"},
            status_code=400
        )

    # --- Генерируем объект ---
    result = create_gan_object(
        shape=shape,
        color=color,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    # Ищем текстуры
    tex_paths = []
    texture_name = result.get("texture")
    if texture_name:
        for ext in ["jpg", "png"]:
            p = obj_path.parent / f"{texture_name}.{ext}"
            if p.exists():
                tex_paths.append(p)

    # Создаем ZIP
    zip_path = obj_path.parent / f"text_{obj_path.stem}.zip"
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    logger.info(f"Object generated from text and zipped: {zip_path.name}")

    return JSONResponse({
        "zip_url": f"/files/{zip_path.relative_to(PROJECT_ROOT)}",
        "obj_url": f"/files/{obj_path.relative_to(PROJECT_ROOT)}",
        "mtl_url": f"/files/{mtl_path.relative_to(PROJECT_ROOT)}",
        "textures": [f"/files/{p.relative_to(PROJECT_ROOT)}" for p in tex_paths]
    })


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    file_location = PROJECT_ROOT / file_path
    return FileResponse(file_location)

@app.post("/api/log-shape")
async def log_shape(payload: dict):
    shape = payload.get("shape")
    logger.info(f"Shape selected: {shape}")
    return {"status": "ok"}

@app.post("/api/log-texture")
async def log_texture(payload: dict):
    texture = payload.get("texture")
    logger.info(f"Texture selected: {texture}")
    return {"status": "ok"}

@app.post("/api/log-color")
async def log_color(payload: dict):
    color = payload.get("color")
    logger.info(f"Color selected: {color}")
    return {"status": "ok"}


# === Статика фронтенда ===
ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "3d frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

logger.info("Server started and serving frontend")
