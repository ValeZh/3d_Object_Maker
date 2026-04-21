"""
api/server.py — Переделанный сервер для модульной системы
Три вкладки: Module Generator → Module Library → House Builder
"""

import logging
import json
import subprocess
import zipfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Импортируем свои модули
import sys

PROJECT_ROOT = Path(__file__).parent.parent  # Поднимаемся в корень проекта
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_parser.parser import extract_module_parameters
from src.ai_parser.nlp_parser import ModuleTextParser
from src.generator.assembler import assemble_building
from src.generator.procedural.procedural_batch_runner import run_all_generators
from src.generator.procedural.procedural_batch_json_parser import parse_and_run

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
MODULES_DIR = OUTPUT_DIR / "modules"  # 🆕 Папка для модулей
TEXTURES_DIR = OUTPUT_DIR / "textures"
FRONTEND_DIR = PROJECT_ROOT / "3d frontend"

# Реестр модулей (JSON файл со списком всех созданных модулей)
MODULES_REGISTRY_FILE = OUTPUT_DIR / "modules_registry.json"

# Реестр домов (JSON файл со списком всех созданных домов)
HOUSES_REGISTRY_FILE = OUTPUT_DIR / "houses_registry.json"

# Создаем папки
for d in [OUTPUT_DIR, CONFIG_DIR, MODELS_DIR, BUILDINGS_DIR, MODULES_DIR, TEXTURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Папка создана/проверена: {d}")

# Инициализируем реестры если их нет
def ensure_registry_exists(registry_file: Path):
    """Создает пустой реестр если его нет"""
    if not registry_file.exists():
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Реестр создан: {registry_file}")

ensure_registry_exists(MODULES_REGISTRY_FILE)
ensure_registry_exists(HOUSES_REGISTRY_FILE)


# ======================= ФУНКЦИИ РЕЕСТРА =======================

def load_modules_registry() -> List[Dict[str, Any]]:
    """Загружает список модулей из реестра"""
    try:
        with open(MODULES_REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки реестра модулей: {e}")
        return []

def save_modules_registry(modules: List[Dict[str, Any]]):
    """Сохраняет список модулей в реестр"""
    try:
        with open(MODULES_REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(modules, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Реестр сохранен ({len(modules)} модулей)")
    except Exception as e:
        logger.error(f"Ошибка сохранения реестра: {e}")

def load_houses_registry() -> List[Dict[str, Any]]:
    """Загружает список домов из реестра"""
    try:
        with open(HOUSES_REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки реестра домов: {e}")
        return []

def save_houses_registry(houses: List[Dict[str, Any]]):
    """Сохраняет список домов в реестр"""
    try:
        with open(HOUSES_REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(houses, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Реестр домов сохранен ({len(houses)} домов)")
    except Exception as e:
        logger.error(f"Ошибка сохранения реестра домов: {e}")


# ======================= ФУНКЦИИ ГЕНЕРАЦИИ МОДУЛЕЙ =======================

def generate_module_obj(module_type: str, params: Dict[str, Any], module_id: str) -> Optional[Path]:
    """Генерирует модуль через batch JSON parser"""
    try:
        output_dir = MODULES_DIR / module_type / module_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔨 Генерация {module_type}_{module_id}...")

        # Конфиг для batch генератора
        config = {}

        if module_type == "wall":
            config["wall"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "wall_length": params.get("width", 2.0),
                "wall_height": params.get("height", 3.0),
                "wall_thickness": params.get("thickness", 0.3),
            }

        elif module_type == "window":  # ← ОКНО = СТЕНА С ОКНОМ
            config["wall_window"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "wall_length": params.get("width", 2.0),
                "wall_height": params.get("height", 3.0),
                "wall_thickness": params.get("thickness", 0.3),
                "window_center_x": params.get("width", 2.0) / 2,
                "window_sill_z": params.get("height", 3.0) / 3,
            }

        elif module_type == "door":
            config["entrance"] = {
                "enabled": True,
                "out_dir": str(output_dir),
            }

        elif module_type == "balcony":  # ← БАЛКОН + ОКНО
            config["balcony"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "width_front": params.get("width", 2.0),
                "depth": params.get("depth", 1.15),
            }
            config["window"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "width": params.get("width", 1.5),
                "height": 1.2,
            }

        elif module_type == "entrance":
            config["entrance_textured"] = {
                "enabled": True,
                "out_dir": str(output_dir),
            }

        # Вызов batch генератора
        results = parse_and_run(config, output_dir)

        for key, path in results.items():
            if path and path.exists():
                logger.info(f"✓ Модуль сгенерирован: {path}")
                return path

        return None

    except Exception as e:
        logger.error(f"Ошибка генерации модуля: {e}")
        return None

def create_module_zip(module_id: str, module_type: str, params: Dict[str, Any], obj_path: Optional[Path]) -> Optional[Path]:
    """
    Создает ZIP архив модуля
    """
    try:
        zip_filename = f"{module_type}_{module_id}.zip"
        zip_path = MODULES_DIR / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            # Добавляем OBJ файл если существует
            if obj_path and obj_path.exists():
                z.write(obj_path, arcname=obj_path.name)

            # Добавляем конфиг параметров
            config = {
                "module_id": module_id,
                "module_type": module_type,
                "params": params,
                "created_at": datetime.now().isoformat()
            }
            z.writestr("config.json", json.dumps(config, indent=2, ensure_ascii=False))

        logger.info(f"✓ ZIP модуля создан: {zip_path}")
        return zip_path

    except Exception as e:
        logger.error(f"Ошибка создания ZIP модуля: {e}")
        return None


# ======================= API ENDPOINTS =======================

@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "modules_count": len(load_modules_registry()),
        "houses_count": len(load_houses_registry())
    }


# ======================= 1️⃣ MODULE GENERATOR ENDPOINTS =======================

@app.post("/api/parse-module")
async def parse_module(request: Request):
    """
    🔹 ВКЛАДКА 1: ПАРСИНГ МОДУЛЯ

    Входные данные:
    {
        "text": "стена 3м высота, 2м ширина",
        "module_type": "wall" (опционально)
    }

    Выходные данные:
    {
        "status": "success",
        "module_type": "wall",
        "params": {...},
        "confidence": 0.95
    }
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()
        module_type = payload.get("module_type")

        if not text:
            return JSONResponse(
                {"error": "Пустой текст"},
                status_code=400
            )

        logger.info(f"📝 Парсинг модуля: '{text}' (тип: {module_type})")

        # Используем локальный NLP парсер (рекомендуется)
        parser = ModuleTextParser()
        result = parser.parse(text)

        logger.info(f"✓ Параметры извлечены: {result.to_dict()}")

        return {
            "status": "success",
            "module_type": result.module_type.value,
            "module_name": result.module_name,
            "params": result.params,
            "confidence": result.confidence
        }

    except Exception as e:
        logger.error(f"Ошибка парсинга: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/api/generate-module")
async def generate_module(request: Request):
    """
    🔹 ВКЛАДКА 1: ГЕНЕРАЦИЯ МОДУЛЯ (текст → 3D → сохранение)

    Входные данные:
    {
        "text": "стена 3м высота, 2м ширина, бетон",
        "module_type": "wall"
    }

    Выходные данные:
    {
        "status": "success",
        "module_id": "uuid",
        "module_type": "wall",
        "params": {...},
        "zip_url": "/files/wall_uuid.zip"
    }
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()
        module_type = payload.get("module_type")

        if not text:
            return JSONResponse(
                {"error": "Пустой текст"},
                status_code=400
            )

        logger.info(f"🔨 Генерация модуля: '{text}'")

        # === 1️⃣ Парсинг ===
        parser = ModuleTextParser()
        parse_result = parser.parse(text)

        module_type = parse_result.module_type.value
        params = parse_result.params

        # === 2️⃣ Генерация OBJ ===
        module_id = str(uuid.uuid4())[:8]
        obj_path = generate_module_obj(module_type, params, module_id)

        # === 3️⃣ Упаковка в ZIP ===
        zip_path = create_module_zip(module_id, module_type, params, obj_path)

        if not zip_path:
            return JSONResponse(
                {"error": "Ошибка создания ZIP"},
                status_code=500
            )

        # === 4️⃣ Сохранение в реестр ===
        module_record = {
            "module_id": module_id,
            "module_type": module_type,
            "module_name": parse_result.module_name,
            "params": params,
            "zip_file": zip_path.name,
            "created_at": datetime.now().isoformat()
        }

        modules = load_modules_registry()
        modules.append(module_record)
        save_modules_registry(modules)

        logger.info(f"✓ Модуль сохранен: {module_id}")

        return {
            "status": "success",
            "module_id": module_id,
            "module_type": module_type,
            "module_name": parse_result.module_name,
            "params": params,
            "zip_url": f"/api/modules/{module_id}/download",
            "confidence": parse_result.confidence
        }

    except Exception as e:
        logger.error(f"Ошибка генерации модуля: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# ======================= 2️⃣ MODULE LIBRARY ENDPOINTS =======================

@app.get("/api/modules")
async def get_all_modules():
    """
    🔹 ВКЛАДКА 2: БИБЛИОТЕКА - ВСЕ МОДУЛИ

    Возвращает все модули, отсортированные по типам
    """
    try:
        modules = load_modules_registry()

        # Группируем по типам
        by_type = {}
        for module in modules:
            mtype = module["module_type"]
            if mtype not in by_type:
                by_type[mtype] = []
            by_type[mtype].append(module)

        return {
            "status": "success",
            "total": len(modules),
            "by_type": by_type,
            "modules": modules
        }

    except Exception as e:
        logger.error(f"Ошибка получения модулей: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/modules/{module_type}")
async def get_modules_by_type(module_type: str):
    """
    🔹 ВКЛАДКА 2: БИБЛИОТЕКА - МОДУЛИ КОНКРЕТНОГО ТИПА

    module_type: wall, window, door, balcony, entrance
    """
    try:
        modules = load_modules_registry()
        filtered = [m for m in modules if m["module_type"] == module_type]

        return {
            "status": "success",
            "module_type": module_type,
            "count": len(filtered),
            "modules": filtered
        }

    except Exception as e:
        logger.error(f"Ошибка получения модулей типа {module_type}: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/modules/{module_id}/download")
async def download_module(module_id: str):
    """
    🔹 ВКЛАДКА 2: СКАЧИВАНИЕ МОДУЛЯ ZIP
    """
    try:
        modules = load_modules_registry()
        module = next((m for m in modules if m["module_id"] == module_id), None)

        if not module:
            return JSONResponse(
                {"error": "Модуль не найден"},
                status_code=404
            )

        zip_file = MODULES_DIR / module["zip_file"]

        if not zip_file.exists():
            return JSONResponse(
                {"error": "ZIP файл не найден"},
                status_code=404
            )

        return FileResponse(zip_file, media_type="application/zip")

    except Exception as e:
        logger.error(f"Ошибка скачивания модуля: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.delete("/api/modules/{module_id}")
async def delete_module(module_id: str):
    """
    🔹 ВКЛАДКА 2: УДАЛЕНИЕ МОДУЛЯ
    """
    try:
        modules = load_modules_registry()
        module = next((m for m in modules if m["module_id"] == module_id), None)

        if not module:
            return JSONResponse(
                {"error": "Модуль не найден"},
                status_code=404
            )

        # Удаляем ZIP файл
        zip_file = MODULES_DIR / module["zip_file"]
        if zip_file.exists():
            zip_file.unlink()
            logger.info(f"✓ ZIP удален: {zip_file}")

        # Удаляем из реестра
        modules = [m for m in modules if m["module_id"] != module_id]
        save_modules_registry(modules)

        logger.info(f"✓ Модуль удален: {module_id}")

        return {"status": "success", "message": "Модуль удален"}

    except Exception as e:
        logger.error(f"Ошибка удаления модуля: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# ======================= 3️⃣ HOUSE BUILDER ENDPOINTS =======================

@app.post("/api/generate-house")
async def generate_house(request: Request):
    """
    🔹 ВКЛАДКА 3: СБОРКА ДОМА ИЗ МОДУЛЕЙ

    Входные данные:
    {
        "house_name": "Мой дом",
        "floors": 5,
        "sections": 3,
        "width": 18,
        "depth": 20,
        "wall_module_id": "uuid",
        "window_module_id": "uuid",
        "door_module_id": "uuid",
        "balcony_module_id": "uuid",
        "entrance_module_id": "uuid"
    }
    """
    try:
        payload = await request.json()
        house_name = payload.get("house_name", "Дом")

        logger.info(f"🏗️ Создание дома: {house_name}")

        # Получаем модули
        modules = load_modules_registry()

        house_params = {
            "house_name": house_name,
            "floors": payload.get("floors", 5),
            "sections": payload.get("sections", 3),
            "width": payload.get("width", 18),
            "depth": payload.get("depth", 20),
            "modules": {
                "wall": payload.get("wall_module_id"),
                "window": payload.get("window_module_id"),
                "door": payload.get("door_module_id"),
                "balcony": payload.get("balcony_module_id"),
                "entrance": payload.get("entrance_module_id"),
            }
        }

        logger.info(f"Параметры дома: {house_params}")

        # TODO: Интегрировать с assembler.py для сборки дома из модулей

        # Сохраняем в реестр домов
        house_id = str(uuid.uuid4())[:8]
        house_record = {
            "house_id": house_id,
            "house_name": house_name,
            "params": house_params,
            "created_at": datetime.now().isoformat()
        }

        houses = load_houses_registry()
        houses.append(house_record)
        save_houses_registry(houses)

        logger.info(f"✓ Дом сохранен: {house_id}")

        return {
            "status": "success",
            "house_id": house_id,
            "house_name": house_name,
            "message": "Дом создан (интеграция с assembler.py в разработке)"
        }

    except Exception as e:
        logger.error(f"Ошибка создания дома: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/houses")
async def get_all_houses():
    """
    🔹 ВКЛАДКА 3: ВСЕ СОХРАНЕННЫЕ ДОМА
    """
    try:
        houses = load_houses_registry()

        return {
            "status": "success",
            "count": len(houses),
            "houses": houses
        }

    except Exception as e:
        logger.error(f"Ошибка получения домов: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/houses/{house_id}")
async def get_house(house_id: str):
    """
    🔹 ВКЛАДКА 3: ДЕТАЛИ ДОМА
    """
    try:
        houses = load_houses_registry()
        house = next((h for h in houses if h["house_id"] == house_id), None)

        if not house:
            return JSONResponse(
                {"error": "Дом не найден"},
                status_code=404
            )

        return {
            "status": "success",
            "house": house
        }

    except Exception as e:
        logger.error(f"Ошибка получения дома: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# ======================= СТАРЫЙ ENDPOINT (для совместимости) =======================

@app.post("/api/generate-building")
async def generate_building_legacy(request: Request):
    """
    ⚠️ СТАРЫЙ ENDPOINT (для совместимости)

    Полный поток: текст → дом (без модульной системы)
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()

        if not text:
            return JSONResponse(
                {"error": "Пустой текст"},
                status_code=400
            )

        logger.info(f"📝 [LEGACY] Генерация дома: '{text}'")

        return JSONResponse(
            {"warning": "Используется старый endpoint. Перейдите на новую модульную систему"},
            status_code=501
        )

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# ======================= СТАТИКА =======================

if MODULES_DIR.exists():
    app.mount("/modules", StaticFiles(directory=MODULES_DIR), name="modules")
    logger.info(f"✓ Модули доступны по /modules")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"✓ Фронтенд подключен: {FRONTEND_DIR}")
else:
    logger.warning(f"⚠️ Папка фронтенда не найдена: {FRONTEND_DIR}")

# ======================= ЗАПУСК =======================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("🚀 Запуск сервера на http://localhost:8000")
    logger.info("📋 Модульная система активирована")
    logger.info("=" * 70)
    logger.info(f"📁 Модули: {MODULES_DIR}")
    logger.info(f"📁 Дома: {BUILDINGS_DIR}")
    logger.info(f"📁 Фронтенд: {FRONTEND_DIR}")
    logger.info("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)