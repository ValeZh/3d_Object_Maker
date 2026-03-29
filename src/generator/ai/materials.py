import os, glob, shutil
import numpy as np


# ==========================================================
# apply_material
# ==========================================================
def apply_material(mesh,
                   output_dir,
                   base_name,
                   color_rgb,
                   texture_dir,
                   texture_name):




    os.makedirs(output_dir, exist_ok=True)

    obj_path = os.path.join(output_dir, base_name + ".obj")
    mtl_path = os.path.join(output_dir, base_name + ".mtl")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    uvs   = np.asarray(mesh.triangle_uvs)

    # ------------------------------------
    # copy texture
    # ------------------------------------
    tex_rel = None
    if texture_name:
        matches = glob.glob(os.path.join(texture_dir, f"{texture_name}*"))
        if matches:
            tex_rel = os.path.basename(matches[0])
            shutil.copyfile(matches[0], os.path.join(output_dir, tex_rel))

    # ------------------------------------
    # write .mtl
    # ------------------------------------
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {base_name}\n")
        f.write(f"Kd {color_rgb[0]} {color_rgb[1]} {color_rgb[2]}\n")
        if tex_rel:
            f.write(f"map_Kd {tex_rel}\n")

    # ------------------------------------
    # write .obj
    # ------------------------------------
    with open(obj_path, "w") as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.write(f"usemtl {base_name}\n\n")

        # vertices
        for vx,vy,vz in verts:
            f.write(f"v {vx} {vy} {vz}\n")

        # UV — сохраняем как у O3D (per-corner!)
        for u,v in uvs:
            f.write(f"vt {u} {v}\n")

        # faces
        for i,(a,b,c) in enumerate(faces):
            ua = i*3 + 0
            ub = i*3 + 1
            uc = i*3 + 2
            f.write(f"f {a+1}/{ua+1} {b+1}/{ub+1} {c+1}/{uc+1}\n")

    return obj_path, mtl_path, open(obj_path, "rb").read(), open(mtl_path, "rb").read()
