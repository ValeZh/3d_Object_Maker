from pathlib import Path

# Корень проекта — подняться на одну директорию от папки api/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папка с текстурами — должна существовать: /textures/
TEXTURES_DIR = PROJECT_ROOT / "textures"

