import sqlite3
from typing import List, Optional
from pathlib import Path

# === Путь к БД ===
# если этот файл лежит рядом с paths.py:
from src.config.paths import DB_PATH


# -------------------------
# 🔥 1. Универсальный helper
# -------------------------

def _execute(query: str, params: tuple = (), fetch: bool = False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(query, params)

    result = None
    if fetch:
        result = c.fetchall()

    conn.commit()
    conn.close()
    return result


# ======================================================
# 🔹 Работа с таблицей SHAPES
# ======================================================

def add_shape(name: str) -> bool:
    """
    Добавляет новую форму (shape) в БД.
    Возвращает:
        True — если добавлено,
        False — если уже есть такая форма.
    """
    name = name.strip().lower()

    existing = _execute(
        "SELECT id FROM shapes WHERE name=?",
        (name,),
        fetch=True
    )

    if existing:
        return False

    _execute(
        "INSERT INTO shapes (name) VALUES (?)",
        (name,)
    )
    return True


def get_shapes() -> List[str]:
    """ Возвращает список всех форм из таблицы shapes """
    rows = _execute(
        "SELECT name FROM shapes",
        fetch=True
    )
    return [r[0] for r in rows]


def shape_exists(name: str) -> bool:
    """ Проверяет, существует ли shape """
    row = _execute(
        "SELECT id FROM shapes WHERE name=?",
        (name,),
        fetch=True
    )
    return bool(row)


# ======================================================
# 🔹 Работа с таблицей TEXTURES
# ======================================================

def add_texture(name: str) -> bool:
    """
    Добавляет новую текстуру.
    Всегда добавляет текстуру "none" (без текстуры) один раз.
    Возвращает:
        True — если добавлено,
        False — если уже есть.
    """
    # --- Сначала добавляем "none" если ещё нет ---
    if not texture_exists("none"):
        _execute(
            "INSERT INTO textures (name) VALUES (?)",
            ("none",)
        )

    name = name.strip().lower()

    existing = _execute(
        "SELECT id FROM textures WHERE name=?",
        (name,),
        fetch=True
    )

    if existing:
        return False

    _execute(
        "INSERT INTO textures (name) VALUES (?)",
        (name,)
    )
    return True

def get_textures() -> List[str]:
    """ Возвращает список всех текстур """
    rows = _execute(
        "SELECT name FROM textures",
        fetch=True
    )
    return [r[0] for r in rows]


def texture_exists(name: str) -> bool:
    """ Проверяет, существует ли текстура """
    row = _execute(
        "SELECT id FROM textures WHERE name=?",
        (name,),
        fetch=True
    )
    return bool(row)


# ======================================================
# 🔹 Получение ID — чтобы использовать в других файлах
# ======================================================

def get_shape_id(name: str) -> Optional[int]:
    """  Возвращает id формы или None """
    row = _execute(
        "SELECT id FROM shapes WHERE name=?",
        (name,),
        fetch=True
    )
    if row:
        return row[0][0]
    return None


def get_texture_id(name: str) -> Optional[int]:
    """ Возвращает id текстуры или None """
    row = _execute(
        "SELECT id FROM textures WHERE name=?",
        (name,),
        fetch=True
    )
    if row:
        return row[0][0]
    return None
