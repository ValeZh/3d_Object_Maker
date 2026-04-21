"""
Каталоги и разрешение путей к картам высот (height / displacement) для текстур OBJ/MTL.

Основное хранилище в репозитории: ``assets/height_maps`` (удобно коммитить небольшие PNG).

Дополнительно можно класть файлы в ``data/textures/height_maps`` (часто в .gitignore у проекта)
и указывать абсолютный или относительный путь при экспорте.
"""
from __future__ import annotations

from pathlib import Path

from src.generator.procedural.texturing.window_texture_assets import resolve_texture_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_height_maps_dir() -> Path:
    """Корневая папка карт высот в репозитории (рядом с кодом, не под игнорируемым data/)."""
    return _repo_root() / "assets" / "height_maps"


def data_height_maps_dir() -> Path:
    """Локальная папка под data/textures (как у color-текстур окон), если используете игнорируемый data/."""
    return _repo_root() / "data" / "textures" / "height_maps"


def resolve_height_map_path(path: str | Path | None) -> Path | None:
    """Существующий файл или None (те же правила, что у resolve_texture_path для PNG/JPG)."""
    return resolve_texture_path(path)


def resolve_height_map_in_defaults(name: str) -> Path | None:
    """
    Ищет файл по имени в ``default_height_maps_dir()``, затем в ``data_height_maps_dir()``.
    Принимается только одно компонентное имя (без путей), чтобы не уходить за пределы каталогов.
    """
    p = Path(name.strip())
    if p.is_absolute() or len(p.parts) != 1:
        return None
    base = p.name
    if not base or base in (".", ".."):
        return None
    for d in (default_height_maps_dir(), data_height_maps_dir()):
        cand = (d / base).resolve()
        if cand.is_file():
            return cand
    return None
