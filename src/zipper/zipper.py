import zipfile
from pathlib import Path

def make_zip(obj_path, mtl_path, texture_paths, output_zip):
    obj_path = Path(obj_path)
    mtl_path = Path(mtl_path)
    output_zip = Path(output_zip)

    with zipfile.ZipFile(output_zip, 'w') as zipf:
        # OBJ
        zipf.write(obj_path, arcname='model.obj')

        # MTL
        zipf.write(mtl_path, arcname='model.mtl')

        # Текстуры
        for t in texture_paths:
            t = Path(t)
            zipf.write(t, arcname=f"textures/{t.name}")

    return str(output_zip)
