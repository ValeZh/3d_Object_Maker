# api/server.py — Финальная версия (декабрь 2025)
import logging
import json
import requests

deepseek_URL = "http://127.0.0.1:11434/v1/completions"  # на сервере deepseek слушает локально

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.ai_parser.parser import send_text_to_deepseek
from src.generator.ai.gan_object_factory import create_gan_object
from src.zipper.zipper import make_zip

# Логи, чтобы видеть всё в консоли
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS — чтобы фронтенд мог стучаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= ПУТИ =======================
from src.config.paths import ROOT as PROJECT_ROOT, OUTPUT_DIR, TEXTURES_DIR

# Убедимся, что папки существуют
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= ОПЦИИ ИЗ БД ИЛИ ФОЛБЭК =======================
try:
    from src.generator.dataset.db_editing import get_shapes, get_textures
    DB_WORKS = True
    logger.info("База данных подключена успешно")
except Exception as e:
    logger.warning(f"БД не подключилась ({e}), будут использованы fallback-опции")
    DB_WORKS = False

FALLBACK_SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid"]
FALLBACK_TEXTURES = ["wood", "stone", "metallic", "none"]

@app.get("/api/options")
async def get_options():
    if DB_WORKS:
        try:
            shapes = [str(s) for s in get_shapes()]
            textures = [str(t) for t in get_textures()]
            if shapes and textures:
                return {"shapes": shapes, "textures": textures}
        except Exception as e:
            logger.error(f"Ошибка чтения БД: {e}")

    logger.info("Возвращаем fallback-опции")
    return {"shapes": FALLBACK_SHAPES, "textures": FALLBACK_TEXTURES}

# ======================= ЛОГИ (по желанию) =======================
@app.post("/api/log-shape")
async def log_shape(request: Request): return JSONResponse({"status": "ok"})

@app.post("/api/log-texture")
async def log_texture(request: Request): return JSONResponse({"status": "ok"})

@app.post("/api/log-color")
async def log_color(request: Request): return JSONResponse({"status": "ok"})

# ======================= hex_to_rgb ==============================
def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

# ======================== ЕНДПОИНТ ДЛЯ СЕРВЕРА ====================
@app.post("/api/process-text")
async def process_text(request: Request):
    """
    Параметры запроса JSON:
    {
        "text": "описание объекта",
        "generate": true/false   # если false, генерировать объект не нужно
    }
    """

    payload = await request.json()
    text = payload.get("text", "").strip()
    generate_obj = payload.get("generate", True)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    logger.info(f"Получен текст: '{text}'")

    # === 1. Анализ текста через DeepSeek ===
    attrs = send_text_to_deepseek(text)
    if not attrs:
        logger.warning(f"DeepSeek вернул None для текста: '{text}'")
        attrs = {
            "shape": "sphere",
            "color": "#ff00ff",
            "texture": "wood",
            "additional_features": ""
        }

    logger.info(f"JSON от DeepSeek: {attrs}")

    result_data = {"attributes": attrs}

    # === 2. Генерация объекта, если нужно ===
    if generate_obj:
        shape = attrs.get("shape", "sphere")
        texture = attrs.get("texture", "metallic")
        color = attrs.get("color", "#6952BE")
        color_rgb = hex_to_rgb(color) if color.startswith("#") else [0.4,0.3,0.7]

        logger.info(f"Генерация объекта: shape={shape}, texture={texture}, color={color}")

        obj_result = create_gan_object(
            shape=shape,
            color=color_rgb,
            texture=texture,
            output_dir=OUTPUT_DIR,
            textures_dir=TEXTURES_DIR
        )

        obj_path = Path(obj_result["obj_path"])
        mtl_path = Path(obj_result["mtl_path"])

        logger.info(f"obj_path exists: {obj_path.exists()}, mtl_path exists: {mtl_path.exists()}")

        tex_paths = []
        for ext in ["jpg","jpeg","png"]:
            p = obj_path.parent / f"{texture}.{ext}"
            if p.exists():
                tex_paths.append(p)

        logger.info(f"Найдено текстур: {[str(p) for p in tex_paths]}")

        zip_name = f"{shape}_{texture}_{color.lstrip('#')}.zip"
        zip_path = obj_path.parent / zip_name
        make_zip(obj_path, mtl_path, tex_paths, zip_path)

        logger.info(f"ZIP создан: {zip_path}, exists={zip_path.exists()}")

        result_data["zip_url"] = f"/files/{zip_path.name}"
        logger.info(f"ZIP создан: {zip_path}")

    return result_data


@app.post("/api/generate-from-text")
async def generate_from_text(request: Request):
    payload = await request.json()
    text = payload.get("text", "").strip()

    if not text:
        logger.warning("Пустой текст получен от фронтенда")
        return JSONResponse({"error": "Пустой текст"}, status_code=400)
    logger.info(f"Получен текст с фронтенда: '{text}'")

    # Используем ИИ вместо фильтра
    attrs = send_text_to_deepseek(text)
    if not attrs:
        logger.warning(f"Deepseek вернул None для текста: '{text}'")
        attrs = {"shape": "sphere",
                 "color": "#ff00ff",
                 "texture": "metallic",
                 "additional_features": ""}

    logger.info(f"JSON, полученный от deepseek: {attrs}")

    return attrs  # JSON возвращается напрямую фронтенду


# ======================= СТАТИКА =======================
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

FRONTEND_DIR = PROJECT_ROOT / "3d frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="3d frontend")
    logger.info(f"[OK] Frontend подключён → {FRONTEND_DIR}")
else:
    logger.warning(f"[NO FRONTEND] Папка не найдена → {FRONTEND_DIR}")