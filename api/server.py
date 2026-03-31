# api/server.py — Финальная версия (декабрь 2025)
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

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
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    from src.generator.dataset.ai_text import extract_attributes
    attrs = extract_attributes(text)

    shape = attrs.get("shape") or "sphere"
    texture = attrs.get("texture") or "metallic"
    color = attrs.get("color") or "#ff00ff"
    color_rgb = hex_to_rgb(color) if isinstance(color, str) and color.startswith("#") else [1, 0, 1]

    logger.info(f"Генерация из текста: '{text}' → {shape=}, {texture=}, {color=}")

    result = create_gan_object(
        shape=shape, color=color_rgb, texture=texture,
        output_dir=OUTPUT_DIR, textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    tex_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    zip_name = f"text_{obj_path.stem}.zip"
    zip_path = obj_path.parent / zip_name
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    rel_path = zip_path.relative_to(OUTPUT_DIR).as_posix()
    return {"zip_url": f"/files/{rel_path}"}


# ======================= СТАТИКА =======================
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

FRONTEND_DIR = PROJECT_ROOT / "3d frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="3d frontend")
    logger.info(f"[OK] Frontend подключён → {FRONTEND_DIR}")
else:
    logger.warning(f"[NO FRONTEND] Папка не найдена → {FRONTEND_DIR}")