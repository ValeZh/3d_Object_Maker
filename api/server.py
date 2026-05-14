"""
api/server.py — Переделанный сервер для модульной системы
Три вкладки: Module Generator → Module Library → House Builder
"""

import base64
import copy
import logging
import json
import subprocess
import threading
import zipfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, Response
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

# Single-pixel PNG — satisfies browser /favicon.ico requests without a static file
_FAVICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@app.get("/favicon.ico")
async def favicon_ico():
    return Response(content=_FAVICON_PNG, media_type="image/png")


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

# Per-file threading locks prevent concurrent in-process corruption.
# Advisory file locks (fcntl) cover multi-process scenarios (e.g. uvicorn --workers N).
_modules_lock = threading.Lock()
_houses_lock  = threading.Lock()

try:
    import fcntl as _fcntl
    def _flock(f, exclusive: bool):
        op = _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH
        _fcntl.flock(f, op)
    def _funlock(f):
        _fcntl.flock(f, _fcntl.LOCK_UN)
except ImportError:
    # Windows fallback — threading.Lock alone is still sufficient for a
    # single uvicorn worker, which is the default dev configuration.
    def _flock(f, exclusive: bool):  # noqa: F811
        pass
    def _funlock(f):                 # noqa: F811
        pass


@contextmanager
def _registry_read(path: Path, thread_lock: threading.Lock):
    with thread_lock:
        with open(path, 'r', encoding='utf-8') as f:
            _flock(f, exclusive=False)
            try:
                yield f
            finally:
                _funlock(f)


@contextmanager
def _registry_write(path: Path, thread_lock: threading.Lock):
    with thread_lock:
        with open(path, 'w', encoding='utf-8') as f:
            _flock(f, exclusive=True)
            try:
                yield f
            finally:
                _funlock(f)


def load_modules_registry() -> List[Dict[str, Any]]:
    try:
        with _registry_read(MODULES_REGISTRY_FILE, _modules_lock) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки реестра модулей: {e}")
        return []

def save_modules_registry(modules: List[Dict[str, Any]]):
    try:
        with _registry_write(MODULES_REGISTRY_FILE, _modules_lock) as f:
            json.dump(modules, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Реестр сохранен ({len(modules)} модулей)")
    except Exception as e:
        logger.error(f"Ошибка сохранения реестра: {e}")

def load_houses_registry() -> List[Dict[str, Any]]:
    try:
        with _registry_read(HOUSES_REGISTRY_FILE, _houses_lock) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки реестра домов: {e}")
        return []

def save_houses_registry(houses: List[Dict[str, Any]]):
    try:
        with _registry_write(HOUSES_REGISTRY_FILE, _houses_lock) as f:
            json.dump(houses, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Реестр домов сохранен ({len(houses)} домов)")
    except Exception as e:
        logger.error(f"Ошибка сохранения реестра домов: {e}")


# ======================= ФУНКЦИИ ГЕНЕРАЦИИ МОДУЛЕЙ =======================

def generate_module_obj(module_type: str, params: Dict[str, Any], module_id: str) -> Optional[Path]:
    """Генерирует модуль с процедурными текстурами через procedural_batch_runner"""
    try:
        output_dir = MODULES_DIR / module_type / module_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔨 Генерация {module_type}_{module_id}...")

        # Загружаем JSON конфиг
        config_file = PROJECT_ROOT / "scripts" / "balcony_examples" / "batch_generators_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Обновляем конфиг в зависимости от типа модуля
        if module_type == "wall":
            tex_block: Dict[str, Any] = {
                "use_procedural_maps": True,
                "wall_color_preset": "plaster",
                "generate_normal": True,
                "generate_roughness": True,
            }
            hex_col = params.get("color")
            if isinstance(hex_col, str) and hex_col.strip().startswith("#"):
                tex_block["wall_tex_color"] = hex_to_rgb(hex_col.strip())

            config["wall"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "wall_length": params.get("width", 2.0),
                "wall_thickness": params.get("thickness", 0.3),
                "wall_height": params.get("height", 3.0),
                "texture": tex_block,
                "no_view": True,
            }
            # Отключаем остальные
            for key in ["window", "wall_window", "balcony", "entrance", "entrance_textured"]:
                if key in config:
                    config[key]["enabled"] = False

        elif module_type == "window":
            tex_block = {
                "use_procedural_maps": True,
                "frame_color_preset": "plaster",
                "glass_color_preset": "uniform_noise",
                "frame_normal_preset": "fine_noise",
                "generate_normal": True,
                "generate_roughness": True,
            }
            hex_frame = params.get("color") or params.get("frame_color")
            if isinstance(hex_frame, str) and hex_frame.strip().startswith("#"):
                tex_block["frame_tex_color"] = hex_to_rgb(hex_frame.strip())
            hex_glass = params.get("glass_color")
            if isinstance(hex_glass, str) and hex_glass.strip().startswith("#"):
                tex_block["glass_tex_color"] = hex_to_rgb(hex_glass.strip())
            config["window"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "width": params.get("width", 1.5),
                "height": params.get("height", 1.2),
                "depth": params.get("depth", 0.12),
                "profile": "rect",
                "kind": "fixed",
                "mullions_vertical": 1,
                "mullions_horizontal": 0,
                "atlas_half_size": 256,
                "texture": tex_block,
                "no_view": True,
            }
            # Отключаем остальные
            for key in ["wall", "wall_window", "balcony", "entrance", "entrance_textured"]:
                if key in config:
                    config[key]["enabled"] = False

        elif module_type == "door":
            config["entrance"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "entrance_style": "canopy",
                "width": params.get("width", 2.0),
                "depth": params.get("depth", 1.75),
                "has_left_wall": True,
                "has_right_wall": True,
                "doors": [
                    {"u0": 0.1, "u1": 0.9, "z_bottom": 0.12, "z_top": 2.05}
                ],
                # === ДОБАВЛЕНЫ ТЕКСТУРЫ ===
                "texture": {
                    "use_procedural_maps": True,
                    "wall_color_preset": "plaster",
                    "door_color_preset": "wood",
                    "generate_normal": True,
                    "generate_roughness": True,
                },
                "no_view": True,
            }
            # Отключаем остальные
            for key in ["wall", "window", "wall_window", "balcony", "entrance_textured"]:
                if key in config:
                    config[key]["enabled"] = False

        elif module_type == "balcony":
            ceramic_file = (
                PROJECT_ROOT / "scripts" / "balcony_examples" / "batch_balcony_ceramic_tile.json"
            )
            if not ceramic_file.exists():
                raise FileNotFoundError(f"Ceramic balcony config not found: {ceramic_file}")
            with open(ceramic_file, "r", encoding="utf-8") as bf:
                ceramic_cfg = json.load(bf)

            tpl = copy.deepcopy(ceramic_cfg.get("balcony") or {})
            for key in ("entrance", "entrance_textured", "window", "wall", "wall_window"):
                if key in ceramic_cfg:
                    config[key] = copy.deepcopy(ceramic_cfg[key])

            tpl["enabled"] = True
            tpl["out_dir"] = str(output_dir)
            tpl["no_view"] = True

            wf = tpl.get("width_front", 2.0)
            wb = tpl.get("width_back", wf)
            if params.get("width") is not None:
                wf = wb = float(params["width"])
            tpl["width_front"] = wf
            tpl["width_back"] = wb
            if params.get("depth") is not None:
                tpl["depth"] = float(params["depth"])
            if params.get("height") is not None:
                tpl["height"] = float(params["height"])

            hex_col = params.get("color")
            if isinstance(hex_col, str) and hex_col.strip().startswith("#"):
                rgb = hex_to_rgb(hex_col.strip())
                for k in (
                    "wall_lower_tex_color",
                    "wall_upper_tex_color",
                    "side_jamb_tex_color",
                    "side_separator_tex_color",
                    "side_basket_tex_color",
                ):
                    tpl[k] = rgb

            config["balcony"] = tpl
            # Отключаем остальные
            for key in ["wall", "window", "wall_window", "entrance", "entrance_textured"]:
                if key in config:
                    config[key]["enabled"] = False

        elif module_type == "entrance":
            config["entrance_textured"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "atlas_tile": 256,
                # === ДОБАВЛЕНЫ ТЕКСТУРЫ ===
                "texture": {
                    "use_procedural_maps": True,
                    "wall_color_preset": "plaster",
                    "door_color_preset": "wood",
                    "generate_normal": True,
                    "generate_roughness": True,
                },
                "no_view": True,
            }
            # Отключаем остальные
            for key in ["wall", "window", "wall_window", "balcony", "entrance"]:
                if key in config:
                    config[key]["enabled"] = False

        else:
            raise ValueError(f"Unknown module type: {module_type}")

        logger.info(
            f"📋 Config для {module_type}: {json.dumps({k: v for k, v in config.items() if isinstance(v, dict) and v.get('enabled')}, indent=2)}")

        # Вызываем batch генератор
        results = run_all_generators(config, default_out_root=output_dir)

        # Возвращаем первый найденный результат
        if results:
            for key, path in results.items():
                if path and path.exists():
                    logger.info(f"✓ Модуль сгенерирован: {path}")

                    if module_type == "door" and path.name == "entrance.obj":
                        new_path = path.parent / "door.obj"
                        path.rename(new_path)
                        logger.info(f"✓ Переименовано: entrance.obj → door.obj")
                        _inject_door_material(new_path, params.get("color"))
                        return new_path

                    elif module_type == "entrance" and path.name == "entrance_textured.obj":
                        new_path = path.parent / "entrance.obj"
                        path.rename(new_path)
                        logger.info(f"✓ Переименовано: entrance_textured.obj → entrance.obj")
                        return new_path

                    return path

        logger.warning(f"⚠️ Генератор не вернул файлы для {module_type}")
        return None

    except Exception as e:
        logger.error(f"Ошибка генерации модуля: {e}", exc_info=True)
        return None


def hex_to_rgb(hex_color: str) -> list:
    """Конвертирует HEX в RGB (0-255)"""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]


def _normalise_hex(color: Optional[str]) -> Optional[str]:
    """Return #RRGGBB or None.  Accepts upper/lower-case with or without '#'."""
    if not isinstance(color, str):
        return None
    c = color.strip().lstrip('#')
    if len(c) == 6 and all(ch in '0123456789abcdefABCDEF' for ch in c):
        return f'#{c.upper()}'
    return None


def _inject_door_material(obj_path: Path, color_hex: Optional[str]) -> None:
    """
    Post-process a trimesh-exported door OBJ (no material) to add a basic MTL.
    Writes door.mtl alongside the OBJ and prepends mtllib + usemtl lines.
    """
    try:
        hex_norm = _normalise_hex(color_hex) or "#8B6914"   # wood brown default
        r, g, b  = hex_to_rgb(hex_norm)
        rd, gd, bd = r / 255.0, g / 255.0, b / 255.0

        mtl_path = obj_path.parent / "door.mtl"
        mtl_path.write_text(
            f"newmtl door\nKa 1 1 1\nKd {rd:.4f} {gd:.4f} {bd:.4f}\nKs 0 0 0\n",
            encoding="utf-8"
        )

        obj_text = obj_path.read_text(encoding="utf-8")
        lines    = obj_text.splitlines(keepends=True)

        # Inject mtllib after the leading comment block; add usemtl before first face.
        new_lines: list[str] = []
        mtllib_added  = False
        usemtl_added  = False
        for line in lines:
            stripped = line.lstrip()
            if not mtllib_added and not stripped.startswith('#'):
                new_lines.append("mtllib door.mtl\n")
                mtllib_added = True
            if not usemtl_added and stripped.startswith('f '):
                new_lines.append("usemtl door\n")
                usemtl_added = True
            new_lines.append(line)

        obj_path.write_text("".join(new_lines), encoding="utf-8")
        logger.info(f"✓ door.mtl injected (color {hex_norm})")
    except Exception as exc:
        logger.warning(f"_inject_door_material failed: {exc}")

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
        "dimensions": {"width": 2.0, "height": 3.0},
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

        # Color from the UI color-picker overrides any color the NLP parser extracted
        # from the text description.  Normalise both paths to #RRGGBB before storing.
        picker_color = _normalise_hex(payload.get("color"))
        if picker_color:
            params["color"] = picker_color
        elif "color" in params:
            params["color"] = _normalise_hex(params["color"]) or params["color"]

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

        # === 4️⃣ СОХРАНЯЕМ РАЗМЕРЫ МОДУЛЯ ===
        module_width = params.get("width", 4.0)
        module_height = params.get("height", 3.0)

        # === 5️⃣ Сохранение в реестр ===
        module_record = {
            "module_id": module_id,
            "module_type": module_type,
            "module_name": parse_result.module_name,
            "params": params,
            "zip_file": zip_path.name,
            "created_at": datetime.now().isoformat(),
            "dimensions": {  # ← ДОБАВЛЕНО
                "width": module_width,
                "height": module_height
            }
        }

        modules = load_modules_registry()
        modules.append(module_record)
        save_modules_registry(modules)

        logger.info(f"✓ Модуль сохранен: {module_id}")
        logger.info(f"✓ Размеры: {module_width}м (ширина) × {module_height}м (высота)")

        return {
            "status": "success",
            "module_id": module_id,
            "module_type": module_type,
            "module_name": parse_result.module_name,
            "params": params,
            "dimensions": {  # ← ВОЗВРАЩАЕМ
                "width": module_width,
                "height": module_height
            },
            "obj_url": f"/modules/{module_type}/{module_id}/{module_type}.obj",
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


@app.patch("/api/modules/{module_id}")
async def rename_module(module_id: str, request: Request):
    """
    🔹 ВКЛАДКА 2: ПЕРЕИМЕНОВАНИЕ МОДУЛЯ

    Входные данные:
    {
        "name": "Новое имя"
    }
    """
    try:
        payload = await request.json()
        new_name = payload.get("name", "").strip()

        if not new_name:
            return JSONResponse({"error": "Имя не может быть пустым"}, status_code=400)

        modules = load_modules_registry()
        module = next((m for m in modules if m["module_id"] == module_id), None)

        if not module:
            return JSONResponse({"error": "Модуль не найден"}, status_code=404)

        module["module_name"] = new_name
        save_modules_registry(modules)
        logger.info(f"✓ Модуль переименован: {module_id} → '{new_name}'")

        return {"status": "success", "module_id": module_id, "module_name": new_name}

    except Exception as e:
        logger.error(f"Ошибка переименования модуля: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/facade-textures")
async def get_facade_textures():
    """
    🔹 ВКЛАДКА 3: СПИСОК ТЕКСТУР ФАСАДА

    Сканирует TEXTURES_DIR и возвращает список доступных текстур.
    Фронтенд использует этот список для заполнения <select>.
    """
    try:
        if not TEXTURES_DIR.exists():
            return []

        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        textures = []

        for f in sorted(TEXTURES_DIR.iterdir()):
            if f.suffix.lower() in extensions:
                textures.append({
                    "name": f.stem,
                    "url": f"/textures/{f.name}",
                })

        logger.info(f"Текстур найдено: {len(textures)}")
        return textures

    except Exception as e:
        logger.error(f"Ошибка получения текстур: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze-building-text")
async def analyze_building_text(request: Request):
    """
    🔹 ВКЛАДКА 3: ПАРСИНГ ТЕКСТОВОГО ОПИСАНИЯ ДОМА

    Входные данные:
    {
        "text": "9-этажный дом, 3 секции, с балконами"
    }

    Выходные данные (формат совместим с applyHouseParams на фронтенде):
    {
        "house": {
            "floors": 9,
            "sections": 3,
            "width": 18,
            "depth": 2,
            "has_balconies": true,
            "balcony_rate": 0.3,
            "window_cols": 8,
            "facade": {"texture_url": "", "texture_scale": 3}
        }
    }
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()

        if not text:
            return JSONResponse({"error": "Пустой текст"}, status_code=400)

        logger.info(f"📝 Анализ текста дома: '{text}'")

        import re

        t = text.lower()

        def _int(pattern, default):
            m = re.search(pattern, t)
            return int(m.group(1)) if m else default

        def _float(pattern, default):
            m = re.search(pattern, t)
            return float(m.group(1).replace(",", ".")) if m else default

        floors = _int(r"(\d+)\s*(?:этаж|floor|storey|story)", 9)
        sections = _int(r"(\d+)\s*(?:секци|подъезд|entranc|section)", 3)
        width = _int(r"(?:ширин[аы]|width)\s*(\d+)", 18)
        depth = _int(r"(?:глубин[аы]|depth)\s*(\d+)", 2)
        window_cols = _int(r"(\d+)\s*(?:окон|window)", 8)
        balcony_rate = _float(r"балкон[ыа]?\s*(\d+(?:[.,]\d+)?)", 0.3)
        if balcony_rate > 1.0:
            balcony_rate = balcony_rate / 100.0

        has_balconies = bool(
            re.search(r"балкон|balcon", t) and
            not re.search(r"без балкон|no balcon", t)
        )

        result = {
            "house": {
                "floors": max(1, min(floors, 25)),
                "sections": max(1, min(sections, 10)),
                "width": max(6, min(width, 30)),
                "depth": max(1, min(depth, 6)),
                "has_balconies": has_balconies,
                "balcony_rate": round(max(0.0, min(balcony_rate, 1.0)), 2),
                "window_cols": max(2, min(window_cols, width)),
                "facade": {
                    "texture_url": "",
                    "texture_scale": 3,
                },
            }
        }

        logger.info(f"✓ Параметры дома извлечены: {result}")
        return result

    except Exception as e:
        logger.error(f"Ошибка анализа текста дома: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ======================= 3️⃣ HOUSE BUILDER ENDPOINTS =======================
def create_wall_window_module(wall_params: Dict[str, Any], window_params: Dict[str, Any]) -> str:
    """
    Builds a wall_window module from wall + window params via procedural_batch_runner.
    Returns the module_id saved in the registry (type "window", assembler picks it up).
    """
    try:
        module_id = str(uuid.uuid4())[:8]
        output_dir = MODULES_DIR / "window" / module_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the wall_window config section for run_all_generators
        wall_window_cfg: Dict[str, Any] = {
            "enabled": True,
            "out_dir": str(output_dir),
            # Wall geometry from wall module params
            "wall_length": float(wall_params.get("width", 3.0)),
            "wall_height": float(wall_params.get("height", 3.0)),
            "wall_thickness": float(wall_params.get("thickness", 0.25)),
            # Window position (centred by default)
            "window_center_x": 0.0,
            "window_sill_z": 1.0,
            # Window geometry from window module params
            "width": float(window_params.get("width", 1.1)),
            "height": float(window_params.get("height", 1.4)),
            "mullions_vertical": int(window_params.get("mullions_vertical", 1)),
            "no_view": True,
            # PBR maps
            "texture": {
                "generate_normal": True,
                "generate_roughness": True,
            },
        }

        # Pass wall colour tint directly (top-level, not nested – forwarded as-is to exporter)
        wall_color = wall_params.get("color")
        if isinstance(wall_color, str) and wall_color.strip().startswith("#"):
            wall_window_cfg["wall_texture_color"] = hex_to_rgb(wall_color.strip())

        frame_color = window_params.get("color") or window_params.get("frame_color")
        if isinstance(frame_color, str) and frame_color.strip().startswith("#"):
            wall_window_cfg["frame_texture_color"] = hex_to_rgb(frame_color.strip())

        glass_color = window_params.get("glass_color")
        if isinstance(glass_color, str) and glass_color.strip().startswith("#"):
            wall_window_cfg["glass_texture_color"] = hex_to_rgb(glass_color.strip())

        logger.info(f"🔨 Generating wall_window via batch runner for module {module_id}")
        results = run_all_generators({"wall_window": wall_window_cfg}, default_out_root=output_dir)

        obj_path = results.get("wall_window")
        if not obj_path or not obj_path.exists():
            raise Exception("wall_window generator returned no output file")

        # Assembler looks for {type}.obj = window.obj; rename so it can find it
        if obj_path.name != "window.obj":
            new_path = obj_path.parent / "window.obj"
            obj_path.rename(new_path)
            obj_path = new_path
            logger.info("✓ Renamed wall_window.obj → window.obj")

        combined_params = {
            "wall_length": wall_window_cfg["wall_length"],
            "wall_height": wall_window_cfg["wall_height"],
            "wall_thickness": wall_window_cfg["wall_thickness"],
            "window_center_x": wall_window_cfg["window_center_x"],
            "window_sill_z": wall_window_cfg["window_sill_z"],
            "width": wall_window_cfg["width"],
            "height": wall_window_cfg["height"],
            "mullions_vertical": wall_window_cfg["mullions_vertical"],
        }

        zip_path = create_module_zip(module_id, "window", combined_params, obj_path)
        if not zip_path:
            raise Exception("Failed to create ZIP for wall_window")

        module_record = {
            "module_id": module_id,
            "module_type": "window",
            "module_name": (
                f"Wall+Window "
                f"{window_params.get('width', 1.1):.1f}×{window_params.get('height', 1.4):.1f}"
            ),
            "params": combined_params,
            "zip_file": zip_path.name,
            "created_at": datetime.now().isoformat(),
            "dimensions": {
                "width": wall_params.get("width", 3.0),
                "height": wall_params.get("height", 3.0),
            },
        }

        modules = load_modules_registry()
        modules.append(module_record)
        save_modules_registry(modules)

        logger.info(f"✓ Wall_window saved to registry: {module_id}")
        return module_id

    except Exception as e:
        logger.error(f"❌ Error creating wall_window: {e}", exc_info=True)
        raise Exception(f"Failed to create wall_window: {str(e)}")

@app.post("/api/generate-house")
async def generate_house(request: Request):
    try:
        payload = await request.json()
        house_name = payload.get("house_name", "Дом")

        # === ПОЛУЧАЕМ ПАРАМЕТРЫ WALL И WINDOW ===
        wall_module_id = payload.get("wall_module_id")
        window_module_id = payload.get("window_module_id")
        wall_dimensions = {"width": 4.0, "height": 3.0}

        wall_params = None
        window_params = None
        modules_registry = load_modules_registry()

        # Ищем wall и window в реестре
        for module in modules_registry:
            module_id = module.get("module_id")

            if module_id == wall_module_id:
                wall_params = module.get("params", {})
                if "dimensions" in module:
                    wall_dimensions = module["dimensions"]
                logger.info(f"✓ Wall параметры: {wall_params}")

            if module_id == window_module_id:
                window_params = module.get("params", {})
                logger.info(f"✓ Window параметры: {window_params}")

        # === REQUIRE BOTH WALL AND WINDOW ===
        if not wall_params:
            return JSONResponse({"error": "Wall module not found in registry"}, status_code=400)
        if not window_params:
            return JSONResponse({"error": "Window module not found in registry"}, status_code=400)

        # === COMBINE WALL + WINDOW → WALL_WINDOW via procedural_batch_runner ===
        logger.info("🔗 Combining wall + window → wall_window via batch runner...")
        window_module_id = create_wall_window_module(wall_params, window_params)
        logger.info(f"✓ Wall_window created: {window_module_id}")

        house_id = str(uuid.uuid4())[:8]
        house_dir = MODULES_DIR / "houses" / house_id
        house_dir.mkdir(parents=True, exist_ok=True)

        # === ПАРАМЕТРЫ ДЛЯ АССЕМБЛЕРА ===
        from src.generator.assembler import assemble_building

        building_params = {
            "floors": payload.get("floors", 5),
            "columns": payload.get("width", 18),
            "sections": payload.get("sections", 3),
            "module_width": wall_dimensions.get("width", 4.0),
            "module_height": wall_dimensions.get("height", 3.0),
            "depth": payload.get("depth", 2),
            "texture_scale": payload.get("texture_scale", 1),
            "balcony_rate": payload.get("balcony_rate", 0.25),
            # Specific UUIDs so the assembler loads the freshly generated composite,
            # not an arbitrary first match from the output directory.
            "wall_module_id":   wall_module_id,
            "window_module_id": window_module_id,
        }

        logger.info(f"🏗️ Параметры здания: {building_params}")

        # Вызываем ассемблер
        output_path = house_dir / "house.obj"
        success = assemble_building(
            building_params,
            MODULES_DIR,
            output_path
        )

        if not success:
            raise Exception("Ошибка сборки дома")

        logger.info(f"✓ Дом собран: {house_id}")

        house_record = {
            "house_id": house_id,
            "house_name": house_name,
            "params": building_params,
            "obj_url": f"/modules/houses/{house_id}/house.obj",
            "created_at": datetime.now().isoformat()
        }

        houses = load_houses_registry()
        houses.append(house_record)
        save_houses_registry(houses)

        return {
            "status": "success",
            "house_id": house_id,
            "house_name": house_name,
            "obj_url": f"/modules/houses/{house_id}/house.obj"
        }

    except Exception as e:
        logger.error(f"Ошибка создания дома: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

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

if TEXTURES_DIR.exists():
    app.mount("/textures", StaticFiles(directory=TEXTURES_DIR), name="textures")
    logger.info(f"✓ Текстуры доступны по /textures")

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