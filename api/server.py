from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Пути
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

# === FastAPI app ===
app = FastAPI()

# === Импорт после app ===
from generator.ai.gan_mesh_factory import create_shape
from generator.ai.gan_object_factory import create_gan_object
from zipper.zipper import make_zip
from api.options import get_available_shapes, get_available_textures, get_available_colors
from config.paths import ROOT as PROJECT_ROOT, OUTPUT_DIR, TEXTURES_DIR

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

# === API маршруты ===
@app.get("/api/options")
async def api_options():
    return {
        "shapes": get_available_shapes(),
        "textures": get_available_textures(),
        "colors": get_available_colors()
    }

@app.post("/api/generate-object")
async def generate_object(payload: dict):
    shape = payload.get("shape")
    color = payload.get("color")
    texture = payload.get("texture")

    # --- Конвертируем HEX в RGB, если нужно ---
    if isinstance(color, str) and color.startswith("#"):
        color = hex_to_rgb(color)

    result = create_gan_object(
        shape=shape,
        color=color,
        texture=texture,
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    tex_paths = []
    for ext in ["jpg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    zip_path = obj_path.parent / f"{shape}_{color}.zip"
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    return JSONResponse({
        "zip_url": f"/files/{zip_path.relative_to(PROJECT_ROOT)}",
        "obj_url": f"/files/{obj_path.relative_to(PROJECT_ROOT)}",
        "mtl_url": f"/files/{mtl_path.relative_to(PROJECT_ROOT)}",
        "textures": [f"/files/{p.relative_to(PROJECT_ROOT)}" for p in tex_paths]
    })

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    file_location = PROJECT_ROOT / file_path
    return FileResponse(file_location)

# === Статика фронтенда в самом конце ===
FRONTEND_DIR = ROOT / "3d frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
