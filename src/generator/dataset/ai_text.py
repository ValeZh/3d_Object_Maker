from . import dataset_maker

SHAPES = ["cube", "sphere", "pyramid", "cylinder", "cone"]
COLORS = ["green", "bluish", "blue", "red", "yellow", "black", "white", "brown"]
TEXTURES = ["wood", "metal", "stone", "plastic", "glass"]


# === Функция извлечения атрибутов ===
def extract_attributes(text):
    text = text.lower()

    found_shape = next((s for s in SHAPES if s in text), None)
    found_color = next((c for c in COLORS if c in text), None)
    found_texture = next((t for t in TEXTURES if t in text), None)

    result = {}
    if found_shape:
        result["shape"] = found_shape
    if found_color:
        result["color"] = found_color
    if found_texture:
        result["texture"] = found_texture

    return result

