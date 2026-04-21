from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.generator.procedural.open3d_preview import preview_window_obj_open3d
from src.generator.procedural.procedural_balcony import export_balcony
from src.generator.procedural.procedural_entrance import export_entrance, export_entrance_textured
from src.generator.procedural.procedural_wall import export_wall
from src.generator.procedural.procedural_wall_window import export_wall_with_window
from src.generator.procedural.procedural_window import export_window_demo


def _no_view_from_json(raw: Any, *, default: bool = True) -> bool:
    """Безопасный разбор no_view (поддержка bool/str/число)."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return default
    if isinstance(raw, (int, float)):
        return bool(raw)
    return bool(raw)


def _prepare_call(
    payload: Dict[str, Any],
    *,
    default_out_root: Path,
    default_name: str,
) -> tuple[Path, dict[str, Any]]:
    cfg = dict(payload)
    out_dir_raw = cfg.pop("out_dir", None) or cfg.pop("output", None)
    out_dir = Path(out_dir_raw) if out_dir_raw else (default_out_root / default_name)
    return out_dir, cfg


def run_all_generators(config: Dict[str, Any], *, default_out_root: Path) -> dict[str, Path]:
    """
    Оркестратор: только вызовы экспорт-функций процедурных генераторов.
    Ожидает словарь конфигурации с ключами:
      balcony, entrance, entrance_textured, window, wall, wall_window

    Секции window и wall_window: опционально no_view (по умолчанию true — без Open3D).
    При no_view: false после экспорта вызывается preview_window_obj_open3d (нужен pip install open3d).
    """
    out: dict[str, Path] = {}

    balcony_cfg = config.get("balcony")
    if isinstance(balcony_cfg, dict) and balcony_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(balcony_cfg, default_out_root=default_out_root, default_name="balcony")
        kwargs.pop("enabled", None)
        out["balcony"] = export_balcony(out_dir=out_dir, **kwargs)

    entrance_cfg = config.get("entrance")
    if isinstance(entrance_cfg, dict) and entrance_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(entrance_cfg, default_out_root=default_out_root, default_name="entrance")
        kwargs.pop("enabled", None)
        out["entrance"] = export_entrance(out_dir=out_dir, **kwargs)

    entrance_textured_cfg = config.get("entrance_textured")
    if isinstance(entrance_textured_cfg, dict) and entrance_textured_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(
            entrance_textured_cfg,
            default_out_root=default_out_root,
            default_name="entrance_textured",
        )
        kwargs.pop("enabled", None)
        out["entrance_textured"] = export_entrance_textured(out_dir=out_dir, **kwargs)

    window_cfg = config.get("window")
    if isinstance(window_cfg, dict) and window_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(window_cfg, default_out_root=default_out_root, default_name="window")
        kwargs.pop("enabled", None)
        no_view = _no_view_from_json(kwargs.pop("no_view", True))
        obj_path = export_window_demo(out_dir=out_dir, **kwargs)
        out["window"] = obj_path
        if not no_view:
            preview_window_obj_open3d(obj_path)

    wall_window_cfg = config.get("wall_window")
    if isinstance(wall_window_cfg, dict) and wall_window_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(wall_window_cfg, default_out_root=default_out_root, default_name="wall_window")
        kwargs.pop("enabled", None)
        no_view = _no_view_from_json(kwargs.pop("no_view", True))
        obj_path = export_wall_with_window(out_dir=out_dir, **kwargs)
        out["wall_window"] = obj_path
        if not no_view:
            preview_window_obj_open3d(obj_path)

    wall_cfg = config.get("wall")
    if isinstance(wall_cfg, dict) and wall_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(wall_cfg, default_out_root=default_out_root, default_name="wall")
        kwargs.pop("enabled", None)
        no_view = _no_view_from_json(kwargs.pop("no_view", True))
        obj_path = export_wall(out_dir=out_dir, **kwargs)
        out["wall"] = obj_path
        if not no_view:
            preview_window_obj_open3d(obj_path)

    return out
