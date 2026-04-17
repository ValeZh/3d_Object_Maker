from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.generator.procedural.procedural_balcony import export_balcony
from src.generator.procedural.procedural_entrance import export_entrance
from src.generator.procedural.procedural_entrance_textured import export_entrance_textured
from src.generator.procedural.procedural_wall_window import export_wall_with_window
from src.generator.procedural.run_window_demo import export_window_demo


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
      balcony, entrance, entrance_textured, window, wall_window
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
        out["window"] = export_window_demo(out_dir=out_dir, **kwargs)

    wall_window_cfg = config.get("wall_window")
    if isinstance(wall_window_cfg, dict) and wall_window_cfg.get("enabled", True):
        out_dir, kwargs = _prepare_call(wall_window_cfg, default_out_root=default_out_root, default_name="wall_window")
        kwargs.pop("enabled", None)
        out["wall_window"] = export_wall_with_window(out_dir=out_dir, **kwargs)

    return out
