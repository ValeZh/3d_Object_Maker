"""
api/server.py — Интегрированный сервер со всеми компонентами
Вызывает процедурные генераторы через subprocess (CLI)
"""

import logging
import json
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Импортируем свои модули
import sys

PROJECT_ROOT = Path(__file__).parent.parent  # Поднимаемся в корень проекта
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_parser.parser import extract_building_parameters
from src.ai_parser.nlp_parser import BuildingTextParser
from src.generator.assembler import assemble_building

# ======================= КОНФИГУРАЦИЯ =======================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= ПУТИ =======================

OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = OUTPUT_DIR / "config"
MODELS_DIR = OUTPUT_DIR / "models"
BUILDINGS_DIR = OUTPUT_DIR / "buildings"
TEXTURES_DIR = OUTPUT_DIR / "textures"
FRONTEND_DIR = PROJECT_ROOT / "3d frontend"

# Создаем папки
for d in [OUTPUT_DIR, CONFIG_DIR, MODELS_DIR, BUILDINGS_DIR, TEXTURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Папка создана/проверена: {d}")


# ======================= ФУНКЦИИ ГЕНЕРАЦИИ =======================

def generate_windows(count: int, config: Dict[str, Any], output_dir: Path) -> bool:
    """
    Генерирует окна через procedural_window.py

    Args:
        count: Количество окон для генерации
        config: Конфиг окна {width, height, depth, kind, mullions_vertical, ...}
        output_dir: Папка для сохранения
    """
    try:
        logger.info(f"🪟 Генерация {count} окон...")

        cmd = [
            "python", "-m", "src.generator.procedural.procedural_window", "export",
            "--width", str(config.get("width", 1.5)),
            "--height", str(config.get("height", 1.2)),
            "--depth", str(config.get("depth", 0.14)),
            "--profile", config.get("profile", "rect"),
            "--kind", config.get("kind", "fixed"),
            "--mullions-vertical", str(config.get("mullions_vertical", 2)),
            "--mullions-horizontal", str(config.get("mullions_horizontal", 1)),
            "-o", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            logger.error(f"Ошибка генерации окон: {result.stderr}")
            return False

        logger.info(f"✓ Окна сгенерированы")
        return True

    except Exception as e:
        logger.error(f"Ошибка при генерации окон: {e}")
        return False


def generate_doors(count: int, config: Dict[str, Any], output_dir: Path) -> bool:
    """
    Генерирует двери через procedural_door.py
    """
    try:
        logger.info(f"🚪 Генерация {count} дверей...")

        cmd = [
            "python", "-m", "src.generator.procedural.procedural_door", "export",
            "--height", str(config.get("height", 2.1)),
            "--width", str(config.get("width", 0.9)),
            "--door-type", config.get("type", "standard"),
            "-o", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            logger.error(f"Ошибка генерации дверей: {result.stderr}")
            return False

        logger.info(f"✓ Двери сгенерированы")
        return True

    except Exception as e:
        logger.error(f"Ошибка при генерации дверей: {e}")
        return False


def generate_balconies(count: int, config: Dict[str, Any], output_dir: Path) -> bool:
    """
    Генерирует балконы через procedural_balcony.py
    """
    try:
        logger.info(f"🏠 Генерация {count} балконов...")

        cmd = [
            "python", "-m", "src.generator.procedural.procedural_balcony", "export",
            "--width-back", str(config.get("width_back", 1.6)),
            "--width-front", str(config.get("width_front", 2.0)),
            "--depth", str(config.get("depth", 1.15)),
            "--height", str(config.get("height", 2.15)),
            "--floor-thickness", str(config.get("floor_thickness", 0.14)),
            "--window-mode", config.get("window_mode", "with_glass"),
            "-o", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            logger.error(f"Ошибка генерации балконов: {result.stderr}")
            return False

        logger.info(f"✓ Балконы сгенерированы")
        return True

    except Exception as e:
        logger.error(f"Ошибка при генерации балконов: {e}")
        return False


def generate_entrances(count: int, config: Dict[str, Any], output_dir: Path) -> bool:
    """
    Генерирует входы/подъезды через procedural_entrance.py
    """
    try:
        logger.info(f"🚶 Генерация {count} подъездов...")

        cmd = [
            "python", "-m", "src.generator.procedural.procedural_entrance", "export",
            "--platform-height", "0.5",
            "--platform-width", "1.5",
            "--platform-depth", "1.5",
            "-o", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            logger.error(f"Ошибка генерации подъездов: {result.stderr}")
            return False

        logger.info(f"✓ Подъезды сгенерированы")
        return True

    except Exception as e:
        logger.error(f"Ошибка при генерации подъездов: {e}")
        return False


def generate_all_components(building_params: Dict[str, Any]) -> bool:
    """
    Генерирует все компоненты дома
    """

    # Конфиги для каждого компонента
    window_config = {
        "width": building_params.get("window_width", 1.5),
        "height": building_params.get("window_height", 1.2),
        "depth": 0.14,
        "profile": "rect",
        "kind": "fixed",
        "mullions_vertical": 2,
        "mullions_horizontal": 1
    }

    door_config = {
        "height": building_params.get("door_height", 2.1),
        "width": building_params.get("door_width", 0.9),
        "type": "standard"
    }

    balcony_config = {
        "width_back": 1.6,
        "width_front": 2.0,
        "depth": building_params.get("balcony_depth", 1.15),
        "height": 2.15,
        "floor_thickness": 0.14,
        "window_mode": "with_glass"
    }

    entrance_config = {
        "steps_count": 5,
        "landing_width": 1.5,
        "landing_depth": 1.5,
        "ramp_enabled": True
    }

    # Вычисляем количество каждого элемента
    total_windows = building_params["floors"] * building_params["windows_per_floor"]
    total_balconies = building_params["floors"] * building_params["balconies_per_floor"]
    total_entrances = building_params["entrance_count"]
    total_doors = building_params["entrance_count"]

    logger.info(f"""
    📊 ПЛАН ГЕНЕРАЦИИ:
    - Окон: {total_windows} ({building_params['windows_per_floor']} × {building_params['floors']} этажей)
    - Балконов: {total_balconies} ({building_params['balconies_per_floor']} × {building_params['floors']} этажей)
    - Входов: {total_doors}
    - Подъездов: {total_entrances}
    """)

    # Генерируем все компоненты
    success = True

    if total_windows > 0:
        success &= generate_windows(total_windows, window_config, MODELS_DIR)

    if total_doors > 0:
        success &= generate_doors(total_doors, door_config, MODELS_DIR)

    if total_balconies > 0:
        success &= generate_balconies(total_balconies, balcony_config, MODELS_DIR)

    if total_entrances > 0:
        success &= generate_entrances(total_entrances, entrance_config, MODELS_DIR)

    return success


def assemble_building_components(building_params: Dict[str, Any]) -> Path:
    """
    Собирает полный дом из сгенерированных компонентов
    """
    logger.info("🔧 Сборка дома из компонентов...")

    building_path = MODELS_DIR / "building.obj"

    # Используем assembler.py
    success = assemble_building(
        params=building_params,
        models_dir=MODELS_DIR,
        output_path=building_path
    )

    if success:
        logger.info(f"✓ Дом собран: {building_path}")
        return building_path
    else:
        logger.error("❌ Ошибка сборки дома")
        return None


def create_zip_archive(params: Dict[str, Any]) -> Path:
    """
    Создает ZIP архив со всеми сгенерированными файлами
    """

    logger.info("📦 Упаковка в ZIP...")

    zip_filename = f"building_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = BUILDINGS_DIR / zip_filename

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            # Добавляем все OBJ и MTL файлы
            for obj_file in MODELS_DIR.glob("*.obj"):
                z.write(obj_file, arcname=obj_file.name)

            for mtl_file in MODELS_DIR.glob("*.mtl"):
                z.write(mtl_file, arcname=mtl_file.name)

            # Добавляем текстуры если есть
            for tex_file in TEXTURES_DIR.glob("*"):
                if tex_file.is_file():
                    z.write(tex_file, arcname=f"textures/{tex_file.name}")

            # Добавляем манифест
            manifest = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "building": params,
                "components": {
                    "windows": params["floors"] * params["windows_per_floor"],
                    "doors": params["entrance_count"],
                    "balconies": params["floors"] * params["balconies_per_floor"],
                    "entrances": params["entrance_count"]
                }
            }
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        logger.info(f"✓ ZIP создан: {zip_path}")
        return zip_path

    except Exception as e:
        logger.error(f"Ошибка при упаковке ZIP: {e}")
        return None


# ======================= API ENDPOINTS =======================

@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(OUTPUT_DIR)
    }


@app.post("/api/generate-building")
async def generate_building(request: Request):
    """
    Основной эндпоинт для генерации здания из текста

    Поток:
    1. Получить текст
    2. DeepSeek: исправить и извлечь параметры
    3. NLP парсер: парсить в параметры здания
    4. Генерировать компоненты (окна, двери, балконы, входы)
    5. Собрать дом
    6. Упаковать в ZIP
    7. Вернуть ссылку на ZIP
    """

    try:
        payload = await request.json()
        text = payload.get("text", "").strip()

        if not text:
            return JSONResponse(
                {"error": "Пустой текст"},
                status_code=400
            )

        logger.info(f"📝 Получен текст: '{text}'")

        # === 1️⃣ DEEPSEEK: Исправление и извлечение параметров ===
        logger.info("1️⃣ DeepSeek обработка...")
        ai_params = extract_building_parameters(text)
        logger.info(f"DeepSeek результат: {ai_params}")

        # === 2️⃣ NLP ПАРСЕР: Дополнительный парсинг ===
        logger.info("2️⃣ NLP парсер...")
        nlp_parser = BuildingTextParser()
        nlp_params = nlp_parser.parse(text)
        logger.info(f"NLP результат: {nlp_params.__dict__}")

        # Объединяем результаты (DeepSeek + NLP)
        building_params = {**nlp_params.__dict__, **ai_params}
        logger.info(f"Финальные параметры: {building_params}")

        # === 3️⃣ ГЕНЕРАЦИЯ КОМПОНЕНТОВ ===
        logger.info("3️⃣ Генерация компонентов...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        success = generate_all_components(building_params)
        if not success:
            logger.warning("⚠️ Некоторые компоненты не сгенерировались (может быть OK)")

        # === 4️⃣ СБОРКА ДОМА ===
        logger.info("4️⃣ Сборка дома...")
        building_path = assemble_building_components(building_params)

        # === 5️⃣ УПАКОВКА В ZIP ===
        logger.info("5️⃣ Упаковка в ZIP...")
        zip_path = create_zip_archive(building_params)

        if not zip_path:
            return JSONResponse(
                {"error": "Ошибка при упаковке ZIP"},
                status_code=500
            )

        logger.info("✓ Генерация успешно завершена!")

        return {
            "status": "success",
            "message": "Дом успешно сгенерирован",
            "parameters": building_params,
            "zip_url": f"/files/{zip_path.name}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        return JSONResponse(
            {"error": f"Внутренняя ошибка: {str(e)}"},
            status_code=500
        )


@app.get("/files/{filename}")
async def download_file(filename: str):
    """Скачивание сгенерированного ZIP файла"""
    file_path = BUILDINGS_DIR / filename

    if not file_path.exists():
        return JSONResponse(
            {"error": "Файл не найден"},
            status_code=404
        )

    return FileResponse(file_path, media_type="application/zip")


@app.get("/api/options")
async def get_options():
    """Возвращает доступные опции"""
    return {
        "shapes": ["building"],
        "profiles": ["rect", "arch", "round"],
        "window_kinds": ["fixed", "double_hung", "casement", "french"],
        "door_types": ["standard", "glass", "metal"]
    }


# ======================= СТАТИКА =======================

if BUILDINGS_DIR.exists():
    app.mount("/files", StaticFiles(directory=BUILDINGS_DIR), name="files")
    logger.info(f"✓ Файлы доступны по /files (папка: {BUILDINGS_DIR})")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"✓ Фронтенд подключен (папка: {FRONTEND_DIR})")
else:
    logger.warning(f"⚠️ Фронтенд папка не найдена: {FRONTEND_DIR}")

# ======================= ЗАПУСК =======================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Запуск сервера на http://localhost:8000")
    logger.info(f"📁 Output папка: {OUTPUT_DIR}")
    logger.info(f"🌐 Фронтенд: {FRONTEND_DIR}")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)