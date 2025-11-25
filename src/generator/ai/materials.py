# materials.py
import os
import glob
import random
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image
import shutil


def find_texture(texture_dir, texture_name):
    matches = glob.glob(os.path.join(texture_dir, f"{texture_name}*.jpg")) + \
              glob.glob(os.path.join(texture_dir, f"{texture_name}*.png"))
    return random.choice(matches) if matches else None


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False
    )


def apply_material(mesh, output_dir, base_name, color_rgb, texture_dir, texture_name):
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh = o3d_to_trimesh(mesh)

    os.makedirs(output_dir, exist_ok=True)

    obj_path = os.path.join(output_dir, base_name + ".obj")
    mtl_path = os.path.join(output_dir, base_name + ".mtl")

    # --- Generate UV for trimesh ---
    need_uv = True
    if mesh.visual:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            need_uv = False

    if need_uv:
        uv = mesh.vertices[:, :2]
        uv = (uv - uv.min(0)) / (uv.max(0) - uv.min(0) + 1e-9)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)

    mesh.export(
        obj_path,
        file_type="obj",
        include_normals=True,
        include_color=False,
        include_texture=False
    )

    # ------------------------------
    # COPY TEXTURE INTO OBJ FOLDER
    # ------------------------------
    tex_src = find_texture(texture_dir, texture_name)
    tex_rel = None

    if tex_src:
        tex_name = os.path.basename(tex_src)
        tex_copy = os.path.join(output_dir, tex_name)
        shutil.copyfile(tex_src, tex_copy)
        tex_rel = tex_name

    # ------------------------------
    # WRITE MTL
    # ------------------------------
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {base_name}\n")
        f.write(f"Kd {color_rgb[0]} {color_rgb[1]} {color_rgb[2]}\n")
        f.write("Ka 0 0 0\nKs 0.1 0.1 0.1\nNs 20\n")
        if tex_rel:
            f.write(f"map_Kd {tex_rel}\n")

    # ------------------------------
    # PATCH OBJ MATERIAL BLOCK
    # ------------------------------
    with open(obj_path, "r") as f:
        content = f.readlines()

    with open(obj_path, "w") as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.writelines(content)
        f.write(f"\nusemtl {base_name}\n")

    return obj_path, mtl_path, open(obj_path, "rb").read(), open(mtl_path, "rb").read()
