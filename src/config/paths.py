from pathlib import Path

# Корень проекта (3d_Object_Maker)
ROOT = Path(__file__).resolve().parents[2]

# Папки для данных и вывода
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TEXTURES_DIR = DATA_DIR / "textures"
SIMPLE_SHAPES_DIR = DATA_DIR / "simple_shapes"
POINTCLOUD_MODEL_DIR = DATA_DIR / "point_cgan_pointnet_output"
DB_PATH = DATA_DIR / "objects_data.db"