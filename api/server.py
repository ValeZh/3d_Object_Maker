# api/server.py — Финальная версия (декабрь 2025)
import logging
import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/v1/completions"  # на сервере Ollama слушает локально

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.ai_parser.parser import send_text_to_ollama
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
logger = logging.getLogger(__name__)


def extract_attributes_with_ollama(text: str) -> dict:
    prompt = f"""
    Извлеки цвет, форму и дополнительные детали из текста в формате JSON.
    Текст: "{text}"
    Вывод должен быть в виде:
    {{ "color": "Red", "shape": "Cube", "additional": "Red top" }}
    """

    data = {
        "model": "llama2",  # или ваша модель
        "prompt": prompt,
        "max_tokens": 50
    }

    try:
        response = requests.post(OLLAMA_URL, json=data)
        response.raise_for_status()
        completion_text = response.json().get("completion", "")
        attrs = json.loads(completion_text)
        return attrs
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        # Возврат fallback
        return {"color": None, "shape": None, "additional": None}
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

# ======================== ЕНДПОИНТ ДЛЯ СЕРВЕРА ====================
@app.post("/api/analyze-text")
async def analyze_text(request: Request):
    payload = await request.json()
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    # Используем ИИ для извлечения атрибутов
    attrs = send_text_to_ollama(text)
    if not attrs:
        # fallback
        logger.warning(f"Ollama вернул None для текста: '{text}'")
        attrs = {"shape": "sphere", "color": "#ff00ff", "additional_features": "", "texture": "metallic"}

    # Логируем для дебага
    logger.info(f"AI JSON: {attrs}")

    return attrs  # JSON возвращается напрямую фронтенду

# ======================= ГЕНЕРАЦИЯ =======================
from src.generator.ai.gan_object_factory import create_gan_object
from src.zipper.zipper import make_zip

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

@app.post("/api/generate-object")
async def generate_object(request: Request):
    payload = await request.json()
    shape = payload.get("shape", "cube")
    texture = payload.get("texture", "wood")
    color = payload.get("color", "#6952BE")
    color_rgb = hex_to_rgb(color) if color.startswith("#") else [0.4, 0.3, 0.7]

    logger.info(f"Генерация: {shape=}, {texture=}, {color=}")

    result = create_gan_object(
        shape=shape,
        color=color_rgb,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR  # ← КРИТИЧЕСКИ ВАЖНО!
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    # Собираем все текстуры с нужным именем
    tex_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    # Имя ZIP-файла
    zip_name = f"{shape}_{texture}_{color.lstrip('#')}.zip"
    zip_path = obj_path.parent / zip_name
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    # ← ВОТ ГДЕ БЫЛА ОШИБКА НА WINDOWS! Используем .as_posix()
    rel_path = zip_path.relative_to(OUTPUT_DIR).as_posix()

    logger.info(f"Готово! ZIP: {rel_path}")
    return {"zip_url": f"/files/{rel_path}"}


@app.post("/api/generate-from-text")
async def generate_from_text(request: Request):
    payload = await request.json()
    text = payload.get("text", "").strip()

    if not text:
        logger.warning("Пустой текст получен от фронтенда")
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    logger.info(f"Получен текст с фронтенда: '{text}'")

    # Отправка в Ollama
    attrs = send_text_to_ollama(text)
    if not attrs:
        logger.warning(f"Ollama вернул None для текста: '{text}'")
        attrs = {"shape": "sphere", "color": "#ff00ff", "additional_features": "", "texture": "metallic"}

    logger.info(f"JSON, полученный от Ollama: {attrs}")
    logger.info("Теперь фронтенд может использовать этот JSON для локальной генерации 3D объекта")

    return attrs  # JSON возвращается напрямую фронтенду


# ======================= СТАТИКА =======================
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

FRONTEND_DIR = PROJECT_ROOT / "3d frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="3d frontend")
    logger.info(f"[OK] Frontend подключён → {FRONTEND_DIR}")
else:
    logger.warning(f"[NO FRONTEND] Папка не найдена → {FRONTEND_DIR}")