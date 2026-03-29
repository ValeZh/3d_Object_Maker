#!/usr/bin/env python3
# uv_export.py — створення UV unwrap для fit-мешів

import numpy as np
import open3d as o3d
from src.config.paths import OUTPUT_DIR
SAVE = OUTPUT_DIR/"uv_output"
SAVE.mkdir(exist_ok=True)

# -------------------------
#  Utility: write OBJ UV
# -------------------------
def export_mesh_uv(mesh, uvs, faces, name):
    """
    mesh.vertices : Nx3
    uvs           : Nx2
    faces         : Mx3   (індекси)
    """
    obj_path = SAVE / f"{name}.obj"
    with open(obj_path, "w") as f:
        # vertices
        for v in np.asarray(mesh.vertices):
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # uv coords
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        # triangles
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    print("✔ Saved:", obj_path)


# ============================
# UV GENERATORS
# ============================

# -----------------
# 1️⃣ Cube UV Atlas
# -----------------
def uv_cube(mesh):
    verts = np.asarray(mesh.vertices)
    # Проєкція по напрямку максимального компоненту
    # Але... ми використаємо готове АТЛАС Мапу (2×3 квадрата).
    uvs = []
    for v in verts:
        x,y,z = v
        # Проста параметрична мапа — cube cross
        u = (x+1)/2
        v = (z+1)/2
        uvs.append([u,v])
    return np.array(uvs)


# -----------------
# 2️⃣ Sphere UV
# equirectangular
# -----------------
def uv_sphere(mesh):
    verts = np.asarray(mesh.vertices)
    uvs=[]
    for v in verts:
        x,y,z = v
        theta = np.arctan2(x,z)       # longitude
        phi   = np.arccos(y)          # latitude
        u = (theta/(2*np.pi)) + 0.5
        v = phi / np.pi
        uvs.append([u,v])
    return np.array(uvs)


# -----------------
# 3️⃣ Cylinder UV
# unwrap side + caps
# -----------------
def uv_cylinder(mesh):
    verts = np.asarray(mesh.vertices)
    uvs=[]
    for v in verts:
        x,y,z=v
        theta=np.arctan2(x,y)
        u=(theta/(2*np.pi))+0.5
        v=(z+1)/2
        uvs.append([u,v])
    return np.array(uvs)


# -----------------
# 4️⃣ Cone
# -----------------
def uv_cone(mesh):
    verts=np.asarray(mesh.vertices)
    center=np.mean(verts,axis=0)
    uvs=[]
    for v in verts:
        x,y,z=v-center
        ang=np.arctan2(x,y)/(2*np.pi)+0.5
        h=(z-np.min(verts[:,2]))/(np.max(verts[:,2])-np.min(verts[:,2]))
        uvs.append([ang,h])
    return np.array(uvs)


# -----------------
# 5️⃣ Torus
# donut parametric UV
# -----------------
def uv_torus(mesh):
    verts=np.asarray(mesh.vertices)
    uvs=[]
    for x,y,z in verts:
        θ=np.arctan2(x,y)/(2*np.pi)+0.5
        φ=np.arctan2(z,np.sqrt(x*x+y*y))/(2*np.pi)+0.5
        uvs.append([θ,φ])
    return np.array(uvs)


# -----------------
# 6️⃣ Pyramid
# planar base + angular sides
# -----------------
def uv_pyramid(mesh):
    verts=np.asarray(mesh.vertices)
    zmin=np.min(verts[:,2])
    zmax=np.max(verts[:,2])
    uvs=[]
    for v in verts:
        x,y,z=v
        if abs(z-zmin)<1e-4: # base
            u=(x-np.min(verts[:,0]))/(np.ptp(verts[:,0]))
            v=(y-np.min(verts[:,1]))/(np.ptp(verts[:,1]))
            uvs.append([u,v])
        else:                # sides
            r=np.linalg.norm([x,y])
            ang=np.arctan2(x,y)/(2*np.pi)+0.5
            h=(z-zmin)/(zmax-zmin)
            uvs.append([ang,h])
    return np.array(uvs)


UV_MAP = {
    "cube":     uv_cube,
    "sphere":   uv_sphere,
    "cylinder": uv_cylinder,
    "cone":     uv_cone,
    "torus":    uv_torus,
    "pyramid":  uv_pyramid,
}

def process(mesh, name):
    verts=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)

    uv_gen = UV_MAP[name]
    uvs = uv_gen(mesh)

    export_mesh_uv(mesh, uvs, faces, f"{name}_uv")
def apply_uv(mesh, shape_name: str):
    if shape_name not in UV_MAP:
        return mesh

    verts_uv = UV_MAP[shape_name](mesh)
    tris     = np.asarray(mesh.triangles)

    expanded=[]
    for a,b,c in tris:
        expanded.append(verts_uv[a])
        expanded.append(verts_uv[b])
        expanded.append(verts_uv[c])

    mesh.triangle_uvs = o3d.utility.Vector2dVector(expanded)
    return mesh
