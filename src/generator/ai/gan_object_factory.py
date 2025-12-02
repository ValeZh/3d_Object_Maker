import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
# SRC — папка src, чтобы Python видел модули generator.ai, config и т.д.
SRC = ROOT / "src"
sys.path.append(str(SRC))

from src.generator.ai.materials import apply_material
from src.config.paths import  OUTPUT_DIR, TEXTURES_DIR



COLOR_MAP = {
    "red":     [1, 0, 0],
    "blue":    [0, 0, 1],
    "green":   [0, 1, 0],
    "yellow":  [1, 1, 0],
    "purple":  [0.5, 0, 0.5],
    "orange":  [1, 0.5, 0],
    "pink":    [1, 0.4, 0.7],
    "white":   [1, 1, 1],
    "black":   [0, 0, 0],
    "cyan":    [0, 1, 1],
}


def create_gan_object(shape, color, texture, output_dir, textures_dir):
    """
    color:
        'red' | 'cyan' | ...
        или [0.2,0.7,1.0]
    """
    from src.generator.ai.gan_mesh_factory import generate_mesh_from_points
    # === Определяем имя и RGB ===
    color_name = None
    color_rgb  = None

    # если RGB
    if isinstance(color, (list, tuple)):
        if len(color) != 3:
            raise ValueError("RGB должен быть длиной 3 (0..1)")
        color_rgb  = list(map(float, color))
        color_name = None

    # если строка
    elif isinstance(color, str):
        cname = color.lower()
        if cname not in COLOR_MAP:
            raise ValueError(f"Неизвестный цвет: {color}")
        color_name = cname
        color_rgb  = COLOR_MAP[cname]

    # если ничего
    elif color is None:
        color_name = "white"
        color_rgb  = COLOR_MAP["white"]

    else:
        raise ValueError(f"color: ожидаю строку или RGB список/tuple, получил {type(color)}")


    print(f"🧠 GAN генерирует меш: {shape}")
    mesh = generate_mesh_from_points(shape)

    folder_name = f"{shape}_{texture}"
    if color_name:
        folder_name += f"_{color_name}"

    obj_dir = Path(output_dir) / folder_name
    obj_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Сохраняем в {obj_dir}")

    obj_path, mtl_path, obj_bin, mtl_bin = apply_material(
        mesh=mesh,
        output_dir=obj_dir,
        base_name=folder_name,
        color_rgb=color_rgb,
        texture_dir=textures_dir,
        texture_name=texture,
    )

    return {
        "shape": shape,
        "color_name": color_name,  # название, если было
        "color_rgb": color_rgb,    # всегда RGB массив
        "texture": texture,
        "obj_path": obj_path,
        "mtl_path": mtl_path,
        "obj_bin": obj_bin,
        "mtl_bin": mtl_bin,
    }


if __name__ == "__main__":
    data = create_gan_object(
        shape="cube",
        color="cyan",
        texture="wood",
        output_dir=OUTPUT_DIR,
        textures_dir=TEXTURES_DIR
    )

