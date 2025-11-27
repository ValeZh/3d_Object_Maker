from pathlib import Path

# Корень проекта — подняться на одну директорию от папки api/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папка с текстурами — должна существовать: /textures/
TEXTURES_DIR = PROJECT_ROOT / "textures"


def get_available_shapes():
    """
    Возвращает список фигур, которые доступны для генерации.
    Эти названия должны совпадать с теми, которые использует frontend.
    """
    return [
        "Cube",
        "Sphere",
        "Pyramid",
        "Prism",
        "Cylinder",
        "Cone",
        "Torus",
    ]


def get_available_textures():
    """
    Возвращает список текстур, основанный на файлах в /textures/.
    Например: textures/wood.png → "wood"
    """
    textures = []

    if TEXTURES_DIR.exists():
        for file in TEXTURES_DIR.glob("*.*"):
            if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                textures.append(file.stem)

    # Если папка пустая — пусть будут дефолтные
    if not textures:
        textures = ["stone", "metal", "wood"]

    return textures


def get_available_colors():
    """
    Просто список готовых цветов — можешь менять на любые.
    """
    return [
        "#6952BE",
        "#FF4B4B",
        "#FFD700",
        "#00C976",
        "#4B69FF",
    ]
