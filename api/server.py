"""
api/server.py — Переделанный сервер для модульной системы
Три вкладки: Module Generator → Module Library → House Builder
"""

import base64
import copy
import logging
import json
import re
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

from src.ai_parser.parser import extract_module_parameters, parse_building_text, parse_roof_text
from src.generator.procedural.procedural_roof import export_roof
from src.ai_parser.nlp_parser import ModuleTextParser, BuildingTextParser, ModuleType, ModuleParams
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

_BEAUTY_DESC_RE = re.compile(
    r"color|цвет|material|материал|wood|дерев|plaster|штукатур|ceramic|плит|tile|"
    r"double|двойн|glass|стекл|style|стиль|shingle|черепиц|mullion|расклад|"
    r"railing|перил|niche|ниш",
    re.IGNORECASE,
)

_BEAUTY_MODULE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "wall": {
        "width": 2.0,
        "height": 3.0,
        "thickness": 0.3,
        "color": "#C9B28F",
        "material": "plaster",
        "_tex_preset": "plaster",
    },
    "window": {
        "width": 1.5,
        "height": 1.2,
        "depth": 0.12,
        "style": "double",
        "frame_color": "#5C4A3A",
        "glass_color": "#87CEEB",
        "color": "#5C4A3A",
        "_frame_preset": "wood",
        "_mullions_v": 1,
        "_mullions_h": 1,
    },
    "door": {
        "width": 0.9,
        "height": 2.1,
        "depth": 1.75,
        "style": "standard",
        "material": "wood",
        "color": "#6B4A33",
        "_door_preset": "wood",
    },
    "balcony": {
        "width": 2.0,
        "height": 2.15,
        "depth": 1.15,
        "style": "open",
        "color": "#B8B0A8",
        "has_roof": True,
    },
    "roof": {
        "length": 3.0,
        "width": 3.0,
        "height": 0.45,
        "overhang": 0.4,
        "roof_type": "flat",
        "color": "#7A523E",
        "_roof_preset": "roof_shingles",
    },
}


def _is_sparse_module_description(text: str) -> bool:
    """True when the user only gave dimensions (or nothing meaningful for materials)."""
    return _BEAUTY_DESC_RE.search(text or "") is None


def _enrich_sparse_module_params(module_type: str, params: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Apply visual-quality defaults when description has no material/color/style hints."""
    if not _is_sparse_module_description(text):
        return params

    preset = _BEAUTY_MODULE_DEFAULTS.get(module_type, {})
    if not preset:
        return params

    out = dict(params)
    for key, value in preset.items():
        if key.startswith("_"):
            continue
        out.setdefault(key, value)

    logger.info(f"✨ Enriched sparse {module_type} params: {out}")
    return out


MODULE_SCRIPT_EXAMPLES: Dict[str, str] = {
    "wall": "batch_wall_only.json",
    "window": "batch_window_only.json",
    "door": "batch_entrance_textured_niche.json",
    "balcony": "beautiful_balcony_loggia.json",
}


def _example_dims_from_config(module_type: str, config: Dict[str, Any]) -> Dict[str, float]:
    if module_type == "wall":
        w = config.get("wall", {})
        return {
            "width": float(w.get("wall_length", 2.0)),
            "height": float(w.get("wall_height", 3.0)),
            "depth": float(w.get("wall_thickness", 0.3)),
        }
    if module_type == "window":
        w = config.get("window", {})
        return {
            "width": float(w.get("width", 1.4)),
            "height": float(w.get("height", 1.6)),
            "depth": float(w.get("depth", 0.12)),
        }
    if module_type == "door":
        e = config.get("entrance_textured", {})
        z0 = float(e.get("niche_door_z_bottom", 0.12))
        z1 = float(e.get("niche_door_z_top", z0 + 2.0))
        return {
            "width": float(e.get("width", 3.2)),
            "height": float(e.get("niche_clear_height", z1 - z0)),
            "depth": float(e.get("depth", 1.5)),
        }
    if module_type == "balcony":
        b = config.get("balcony", {})
        return {
            "width": float(b.get("width_front", b.get("width_back", 2.0))),
            "height": float(b.get("height", 2.48)),
            "depth": float(b.get("depth", 1.4)),
        }
    return {}


def _finalize_module_obj_path(module_type: str, path: Path, params: Dict[str, Any]) -> Path:
    if module_type == "door" and path.name == "entrance.obj":
        new_path = path.parent / "door.obj"
        path.rename(new_path)
        logger.info("✓ Переименовано: entrance.obj → door.obj")
        if not (new_path.parent / "material.mtl").exists():
            _inject_door_material(new_path, params.get("color"))
        return new_path
    if module_type == "entrance" and path.name == "entrance_textured.obj":
        new_path = path.parent / "entrance.obj"
        path.rename(new_path)
        logger.info("✓ Переименовано: entrance_textured.obj → entrance.obj")
        return new_path
    return path


def _apply_script_example_color_tint(
    module_type: str, config: Dict[str, Any], hex_col: Optional[str]
) -> None:
    if not hex_col:
        return
    rgb = hex_to_rgb(hex_col.lstrip("#"))
    if module_type == "balcony":
        bal = config.get("balcony")
        if isinstance(bal, dict):
            bal["wall_lower_tex_color"] = rgb
    elif module_type == "wall":
        wall = config.get("wall")
        if isinstance(wall, dict):
            tex = wall.setdefault("texture", {})
            tex["wall_tex_color"] = rgb
    elif module_type == "window":
        win = config.get("window")
        if isinstance(win, dict):
            tex = win.setdefault("texture", {})
            tex["frame_tex_color"] = rgb
    elif module_type == "door":
        ent = config.get("entrance_textured")
        if isinstance(ent, dict):
            ent["door_tex_color"] = rgb


def _generate_from_script_example(
    module_type: str,
    params: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    example_name = MODULE_SCRIPT_EXAMPLES.get(module_type)
    if not example_name:
        return None

    example_path = PROJECT_ROOT / "scripts" / "balcony_examples" / example_name
    if not example_path.exists():
        logger.warning("Script example not found: %s", example_path)
        return None

    with open(example_path, "r", encoding="utf-8") as f:
        config = copy.deepcopy(json.load(f))

    for _section_key, section in config.items():
        if not isinstance(section, dict):
            continue
        if section.get("enabled"):
            section["out_dir"] = str(output_dir)
            section["no_view"] = True
        elif "enabled" in section:
            section["enabled"] = False

    _apply_script_example_color_tint(module_type, config, _normalise_hex(params.get("color")))
    params.update(_example_dims_from_config(module_type, config))

    logger.info("📜 Script example %s → %s", example_name, module_type)
    results = run_all_generators(config, default_out_root=output_dir)
    if not results:
        return None

    for _key, path in results.items():
        if path and path.exists():
            logger.info("✓ Модуль сгенерирован (example): %s", path)
            return _finalize_module_obj_path(module_type, path, params)
    return None


def generate_module_obj(module_type: str, params: Dict[str, Any], module_id: str) -> Optional[Path]:
    """Генерирует модуль с процедурными текстурами через procedural_batch_runner"""
    try:
        output_dir = MODULES_DIR / module_type / module_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔨 Генерация {module_type}_{module_id}...")

        if params.get("_use_script_example"):
            result = _generate_from_script_example(module_type, params, output_dir)
            if result:
                return result
            logger.warning("Script example failed for %s, using procedural path", module_type)

        # Загружаем JSON конфиг
        config_file = PROJECT_ROOT / "scripts" / "balcony_examples" / "batch_generators_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Обновляем конфиг в зависимости от типа модуля
        source_text = str(params.get("_source_text", ""))
        sparse = _is_sparse_module_description(source_text)

        if module_type == "wall":
            tex_block: Dict[str, Any] = {
                "use_procedural_maps": True,
                "wall_color_preset": "plaster",
                "generate_normal": True,
                "generate_roughness": True,
                "bump_strength": 0.75,
            }
            mat = str(params.get("material", "")).lower()
            if "ceramic" in mat or "tile" in mat or "плит" in source_text.lower():
                tex_block["wall_color_preset"] = "ceramic_tile"
                tex_block["wall_normal_preset"] = "ceramic_tile"
                tex_block["tiles_per_side"] = 9
                tex_block["grout_width"] = 0.055
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
            win_preset = _BEAUTY_MODULE_DEFAULTS.get("window", {})
            tex_block = {
                "use_procedural_maps": True,
                "frame_color_preset": win_preset.get("_frame_preset", "wood") if sparse else "plaster",
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
                "mullions_vertical": int(win_preset.get("_mullions_v", 1)) if sparse else 1,
                "mullions_horizontal": int(win_preset.get("_mullions_h", 0)) if sparse else 0,
                "atlas_half_size": 256,
                "texture": tex_block,
                "no_view": True,
            }
            # Отключаем остальные
            for key in ["wall", "wall_window", "balcony", "entrance", "entrance_textured"]:
                if key in config:
                    config[key]["enabled"] = False

        elif module_type == "door":
            total_width = float(params.get("width", 2.0))
            door_height = float(params.get("height", 2.0))
            niche_z_bottom = 0.12
            niche_z_top = niche_z_bottom + door_height
            # Center door panel within entrance width; panel defaults to 80% of total.
            panel_width = float(params.get("door_width", total_width * 0.8))
            u0 = max(0.0, (1.0 - panel_width / total_width) / 2)
            u1 = min(1.0, 1.0 - u0)
            door_cfg: Dict[str, Any] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "entrance_style": "niche",
                "width": total_width,
                "depth": float(params.get("depth", 1.75)),
                "has_left_wall": True,
                "has_right_wall": True,
                "niche_clear_height": door_height,
                "niche_door_z_bottom": niche_z_bottom,
                "niche_door_z_top": niche_z_top,
                "niche_door_u0": round(u0, 4),
                "niche_door_u1": round(u1, 4),
                "double_door": True,
                "atlas_tile": 256,
                "texture": {
                    "use_procedural_maps": True,
                    "wall_color_preset": "plaster",
                    "door_color_preset": "wood",
                    "generate_normal": True,
                    "generate_roughness": True,
                },
                "no_view": True,
            }
            hex_col = params.get("color")
            if isinstance(hex_col, str) and hex_col.strip().startswith("#"):
                rgb = hex_to_rgb(hex_col.strip())
                door_cfg["door_tex_color"] = rgb
                door_cfg["wall_tex_color"] = rgb
            config["entrance_textured"] = door_cfg
            # Отключаем остальные
            for key in ["wall", "window", "wall_window", "balcony", "entrance"]:
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
                logger.info(f"[balcony] height from params: {tpl['height']}")
            else:
                logger.info(f"[balcony] height NOT in params → generator will use USER_BALCONY default (2.15m); params keys: {list(params.keys())}")
            if params.get("has_roof") is not None:
                tpl["has_roof"] = bool(params["has_roof"])
            elif str(params.get("style", "")).lower() in ("enclosed", "closed", "лоджия", "закрытый"):
                tpl["has_roof"] = True
            if params.get("roof_thickness") is not None:
                tpl["roof_thickness"] = float(params["roof_thickness"])
            if params.get("roof_overhang") is not None:
                tpl["roof_overhang"] = float(params["roof_overhang"])
            roof_hex = params.get("roof_color")
            if isinstance(roof_hex, str) and roof_hex.strip().startswith("#"):
                tpl["roof_tex_color"] = hex_to_rgb(roof_hex.strip())

            # Scale inner windows/doors z-coords proportionally to the actual height.
            # Template was authored at 2.15 m (USER_BALCONY default).
            _DEFAULT_BALCONY_H = 2.15
            _actual_h = float(tpl.get("height", _DEFAULT_BALCONY_H))
            _ratio = _actual_h / _DEFAULT_BALCONY_H
            if abs(_ratio - 1.0) > 0.001:
                for item in tpl.get("inner_wall_windows", []):
                    if "z_bottom" in item:
                        item["z_bottom"] = round(item["z_bottom"] * _ratio, 4)
                    if "z_top" in item:
                        item["z_top"] = round(item["z_top"] * _ratio, 4)
                for item in tpl.get("inner_wall_doors", []):
                    if "z_bottom" in item:
                        item["z_bottom"] = round(item["z_bottom"] * _ratio, 4)
                    if "z_top" in item:
                        item["z_top"] = round(item["z_top"] * _ratio, 4)
                logger.info(
                    f"[balcony] scaled inner windows/doors by ratio {_ratio:.3f} "
                    f"(height {_DEFAULT_BALCONY_H}→{_actual_h})"
                )

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

        elif module_type == "roof":
            roof_type = str(params.get("roof_type") or params.get("type") or "flat").strip().lower()
            default_height = 0.45 if roof_type == "flat" else 2.2
            tex_block = {
                "use_procedural_maps": True,
                "roof_color_preset": "roof_shingles",
                "generate_normal": True,
                "generate_roughness": True,
            }
            hex_col = params.get("color") or params.get("roof_color")
            if isinstance(hex_col, str) and hex_col.strip().startswith("#"):
                tex_block["roof_tex_color"] = hex_to_rgb(hex_col.strip())
            config["roof"] = {
                "enabled": True,
                "out_dir": str(output_dir),
                "length": float(params.get("length", params.get("width", 3.0))),
                "width": float(params.get("width", params.get("depth", 3.0))),
                "height": max(0.35, float(params.get("height", default_height))),
                "overhang": float(params.get("overhang", 0.4)),
                "roof_type": roof_type,
                "texture": tex_block,
                "no_view": True,
            }
            for key in ["wall", "window", "wall_window", "balcony", "entrance", "entrance_textured"]:
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

                    return _finalize_module_obj_path(module_type, path, params)

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
        use_script_example = bool(payload.get("use_script_example"))
        text = payload.get("text", "").strip()
        module_type_hint = str(payload.get("module_type", "") or "").strip().lower()

        if use_script_example:
            try:
                mt = ModuleType(module_type_hint or text or "wall")
            except ValueError:
                mt = ModuleType.WALL
            module_type = mt.value
            params: Dict[str, Any] = {"_source_text": ""}
            if module_type in MODULE_SCRIPT_EXAMPLES:
                params["_use_script_example"] = True
            params = _enrich_sparse_module_params(module_type, params, "")
            parse_result = ModuleParams(
                module_type=mt,
                module_name=f"{module_type} (preset)",
                params=params,
                confidence=1.0,
            )
            logger.info("🔨 Генерация модуля (preset example): %s", module_type)
        else:
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
            params = _enrich_sparse_module_params(
                parse_result.module_type.value,
                parse_result.params,
                text,
            )
            params["_source_text"] = text

        # Color from the UI color-picker overrides any color the NLP parser extracted
        # from the text description.  Normalise both paths to #RRGGBB before storing.
        picker_color = _normalise_hex(payload.get("color"))
        if picker_color:
            params["color"] = picker_color
        elif "color" in params:
            params["color"] = _normalise_hex(params["color"]) or params["color"]

        # === 2️⃣ Генерация OBJ ===
        module_id = str(uuid.uuid4())[:8]
        params_for_store = {k: v for k, v in params.items() if not str(k).startswith("_")}
        obj_path = generate_module_obj(module_type, params, module_id)

        # === 3️⃣ Упаковка в ZIP ===
        zip_path = create_module_zip(module_id, module_type, params_for_store, obj_path)

        if not zip_path:
            return JSONResponse(
                {"error": "Ошибка создания ZIP"},
                status_code=500
            )

        # === 4️⃣ СОХРАНЯЕМ РАЗМЕРЫ МОДУЛЯ ===
        module_width = params_for_store.get("width", 4.0)
        module_height = params_for_store.get("height", 3.0)

        # === 5️⃣ Сохранение в реестр ===
        module_record = {
            "module_id": module_id,
            "module_type": module_type,
            "module_name": parse_result.module_name,
            "params": params_for_store,
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
            "params": params_for_store,
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


_building_text_parser = BuildingTextParser()


@app.post("/api/analyze-building-text")
async def analyze_building_text(request: Request):
    """
    ВКЛАДКА 3: ПАРСИНГ ТЕКСТОВОГО ОПИСАНИЯ ДОМА
    AI (DeepSeek через parser.py), fallback — regex (nlp_parser.py).
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()

        if not text:
            return JSONResponse({"error": "Пустой текст"}, status_code=400)

        logger.info(f"Анализ текста дома: '{text}'")

        import asyncio
        loop = asyncio.get_event_loop()
        ai_result = await loop.run_in_executor(None, parse_building_text, text)
        if ai_result is not None:
            logger.info(f"AI parse succeeded: {ai_result}")
            return ai_result

        logger.info("AI parse failed — using regex fallback")
        return _building_text_parser.parse(text)

    except Exception as e:
        logger.error(f"Ошибка анализа текста дома: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ======================= 3️⃣ HOUSE BUILDER ENDPOINTS =======================
def _recreate_wall_with_color(wall_params: Dict[str, Any], new_color: str) -> tuple[str, Dict[str, Any]]:
    """
    Clone wall_params, override color, regenerate OBJ + ZIP, save to registry.
    Returns (new_module_id, updated_params).
    """
    module_id = str(uuid.uuid4())[:8]
    updated_params = {**wall_params, "color": new_color}

    obj_path = generate_module_obj("wall", updated_params, module_id)
    if not obj_path or not obj_path.exists():
        raise Exception(f"Wall OBJ generation failed for color-synced module {module_id}")

    zip_path = create_module_zip(module_id, "wall", updated_params, obj_path)
    if not zip_path:
        raise Exception(f"ZIP creation failed for color-synced wall module {module_id}")

    module_record = {
        "module_id": module_id,
        "module_type": "wall",
        "module_name": f"Wall (color {new_color})",
        "params": updated_params,
        "zip_file": zip_path.name,
        "created_at": datetime.now().isoformat(),
        "dimensions": {
            "width": float(updated_params.get("width", 4.0)),
            "height": float(updated_params.get("height", 3.0)),
        },
    }
    modules = load_modules_registry()
    modules.append(module_record)
    save_modules_registry(modules)

    logger.info(f"✓ Color-synced wall created: {module_id} (color={new_color})")
    return module_id, updated_params


def create_wall_window_module(wall_params: Dict[str, Any], window_params: Dict[str, Any]) -> str:
    """
    Builds a wall_window module from wall + window params via procedural_batch_runner.
    Returns the module_id saved in the registry (type "wall_window").
    """
    try:
        module_id = str(uuid.uuid4())[:8]
        output_dir = MODULES_DIR / "wall_window" / module_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the wall_window config section for run_all_generators
        wall_height = float(wall_params.get("height", 3.0))
        win_height  = float(window_params.get("height", 1.4))
        # Centre the window vertically within the wall, leaving at least 0.15m
        # headroom above and 0.2m sill below so the wall mesh doesn't degenerate.
        sill_z = max(0.2, (wall_height - win_height) / 2)
        if sill_z + win_height > wall_height - 0.12:
            sill_z = wall_height - win_height - 0.12
            if sill_z < 0.15:  # если слишком низко, оставить центровано
                sill_z = max(0.2, (wall_height - win_height) / 2)

        wall_window_cfg: Dict[str, Any] = {
            "enabled": True,
            "out_dir": str(output_dir),
            # Wall geometry from wall module params
            "wall_length": float(wall_params.get("width", 3.0)),
            "wall_height": wall_height,
            "wall_thickness": float(wall_params.get("thickness", 0.25)),
            # Window position (centred vertically; sill computed above)
            "window_center_x": 0.0,
            "window_sill_z": sill_z,
            # Window geometry from window module params
            "width": float(window_params.get("width", 1.1)),
            "height": win_height,
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
            logger.info(f"wall_window_cfg = {wall_window_cfg}")

        frame_color = window_params.get("color") or window_params.get("frame_color")
        if isinstance(frame_color, str) and frame_color.strip().startswith("#"):
            wall_window_cfg["frame_texture_color"] = hex_to_rgb(frame_color.strip())

        glass_color = window_params.get("glass_color")
        if isinstance(glass_color, str) and glass_color.strip().startswith("#"):
            wall_window_cfg["glass_texture_color"] = hex_to_rgb(glass_color.strip())
        else:
            wall_window_cfg["glass_texture_color"] = [135, 206, 235]  # default sky blue

        logger.info(f"🔨 Generating wall_window via batch runner for module {module_id}")
        results = run_all_generators({"wall_window": wall_window_cfg}, default_out_root=output_dir)

        obj_path = results.get("wall_window")
        if not obj_path or not obj_path.exists():
            raise Exception("wall_window generator returned no output file")

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

        zip_path = create_module_zip(module_id, "wall_window", combined_params, obj_path)
        if not zip_path:
            raise Exception("Failed to create ZIP for wall_window")

        module_record = {
            "module_id": module_id,
            "module_type": "wall_window",
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


def create_roof_module(roof_type: str) -> str:
    """Generate a 3×3m roof module, save to registry, return module_id."""
    module_id = str(uuid.uuid4())[:8]
    height = 0.45 if roof_type == "flat" else 1.5
    params = {"roof_type": roof_type, "length": 3.0, "width": 3.0, "height": height}
    obj_path = generate_module_obj("roof", params, module_id)
    if not obj_path or not obj_path.exists():
        raise Exception(f"Roof OBJ generation failed for module {module_id}")
    zip_path = create_module_zip(module_id, "roof", params, obj_path)
    if not zip_path:
        raise Exception(f"ZIP creation failed for roof module {module_id}")
    module_record = {
        "module_id": module_id,
        "module_type": "roof",
        "module_name": f"Roof ({roof_type})",
        "params": params,
        "zip_file": zip_path.name,
        "created_at": datetime.now().isoformat(),
        "dimensions": {"width": params["length"], "height": params["height"]},
    }
    modules = load_modules_registry()
    modules.append(module_record)
    save_modules_registry(modules)
    logger.info(f"✓ Roof module created: {module_id} (type={roof_type})")
    return module_id


@app.post("/api/generate-roof-module")
async def generate_roof_module_endpoint(request: Request):
    """
    Create a roof module from text or explicit roof_type.
    Body: {"text": "gable roof"} or {"roof_type": "pyramid"}
    """
    try:
        payload = await request.json()
        text = payload.get("text", "").strip()
        roof_type = payload.get("roof_type", "").strip().lower()

        if text and not roof_type:
            parsed = parse_roof_text(text)
            roof_type = parsed.get("roof_type", "flat")

        if roof_type not in ("flat", "gable", "pyramid"):
            roof_type = "flat"

        module_id = create_roof_module(roof_type)
        return JSONResponse({
            "status": "success",
            "module_id": module_id,
            "module_type": "roof",
            "params": {"roof_type": roof_type},
            "zip_url": f"/api/modules/{module_id}/download",
        })
    except Exception as e:
        logger.error(f"❌ generate_roof_module_endpoint: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate-house")
async def generate_house(request: Request):
    try:
        payload = await request.json()
        house_name = payload.get("house_name", "Дом")

        # === ПОЛУЧАЕМ ПАРАМЕТРЫ WALL, WINDOW И BALCONY ===
        wall_module_id = payload.get("wall_module_id")
        window_module_id = payload.get("window_module_id")
        door_module_id = payload.get("door_module_id")
        balcony_module_id = payload.get("balcony_module_id") or None
        roof_module_id = payload.get("roof_module_id") or None
        wall_dimensions = {"width": 4.0, "height": 3.0}

        wall_params = None
        window_params = None
        roof_params = {
            "roof_type": str(payload.get("roof_type", "flat")).strip().lower() or "flat",
            "height": 0.45,
            "overhang": 0.4,
        }
        modules_registry = load_modules_registry()

        # Ищем wall, window и balcony в реестре
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

            if module_id == balcony_module_id:
                logger.info(f"✓ Balcony module найден в реестре: {module_id}")

            if module_id == roof_module_id:
                mod_params = module.get("params", {})
                house_roof_type = str(payload.get("roof_type", "flat")).strip().lower() or "flat"
                roof_params.update(mod_params)
                roof_params["roof_type"] = house_roof_type
                logger.info(f"✓ Roof параметры: {roof_params}")

        # === REQUIRE BOTH WALL AND WINDOW ===
        if not wall_params:
            return JSONResponse({"error": "Wall module not found in registry"}, status_code=400)
        if not window_params:
            return JSONResponse({"error": "Window module not found in registry"}, status_code=400)

        # === COLOR SYNC: if house_color overrides the wall module color, regenerate wall ===
        logger.info(f"DEBUG: payload keys = {list(payload.keys())}")
        logger.info(f"DEBUG: house_color raw = {payload.get('house_color')!r}")
        house_color = _normalise_hex(payload.get("house_color"))
        logger.info(f"DEBUG: house_color normalised = {house_color!r}")
        if house_color:
            current_wall_color = _normalise_hex(wall_params.get("color"))
            if house_color != current_wall_color:
                logger.info(
                    f"🎨 house_color={house_color} differs from wall color={current_wall_color}; "
                    "regenerating wall module"
                )
                wall_module_id, wall_params = _recreate_wall_with_color(wall_params, house_color)
            else:
                logger.info(f"🎨 house_color={house_color} matches wall module color, no regeneration needed")

        # === COMBINE WALL + WINDOW → WALL_WINDOW via procedural_batch_runner ===
        logger.info("🔗 Combining wall + window → wall_window via batch runner...")
        try:
            wall_window_module_id = create_wall_window_module(wall_params, window_params)
            logger.info(f"✓ Wall_window created: {wall_window_module_id}")
        except Exception as ww_err:
            logger.warning(
                f"⚠️ Wall_window generation failed: {ww_err}. "
                "Assembly will proceed using plain walls for window cells."
            )
            wall_window_module_id = None

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
            # has_balcony: True if the user selected a balcony module
            "has_balcony": bool(payload.get("has_balconies", False)) and balcony_module_id is not None,
            # Specific UUIDs so the assembler loads freshly generated modules
            # rather than an arbitrary first alphabetical match from disk.
            "wall_module_id":         wall_module_id,
            "wall_window_module_id":  wall_window_module_id,
            "entrance_module_id":     door_module_id,
            "door_module_id":         door_module_id,
            "balcony_module_id":      balcony_module_id,
            "roof_module_id":         roof_module_id,
            "roof_params":            roof_params,
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