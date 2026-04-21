"""UV по граням (faceted triplanar) и сборка рама+стекло под атлас [0,0.5]|[0.5,1]."""
from __future__ import annotations

import numpy as np
import trimesh


def faceted_triplanar_uv(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    UV по нормалям граней (вершины дублируются по граням): для бокса нет «уголковых»
    усреднённых нормалей и артефактов освещения/текстуры на тонком стекле.
    """
    mesh = mesh.copy()
    if len(mesh.faces) == 0:
        return mesh, np.zeros((0, 2), dtype=np.float64)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v_exp = verts[faces].reshape(-1, 3)
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fn_exp = np.repeat(fn, 3, axis=0)

    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    eps = 1e-9
    dx = max(xmax - xmin, eps)
    dy = max(ymax - ymin, eps)
    dz = max(zmax - zmin, eps)

    dom = np.argmax(np.abs(fn_exp), axis=1)
    u = np.zeros(len(v_exp), dtype=np.float64)
    vv = np.zeros(len(v_exp), dtype=np.float64)
    mx = dom == 0
    my = dom == 1
    mz = dom == 2
    u[mx] = (v_exp[mx, 1] - ymin) / dy
    vv[mx] = (v_exp[mx, 2] - zmin) / dz
    u[my] = (v_exp[my, 0] - xmin) / dx
    vv[my] = (v_exp[my, 2] - zmin) / dz
    u[mz] = (v_exp[mz, 0] - xmin) / dx
    vv[mz] = (v_exp[mz, 1] - ymin) / dy

    u = np.clip(u, 0.0, 1.0)
    vv = np.clip(vv, 0.0, 1.0)
    uv = np.stack([u, vv], axis=1)
    new_faces = np.arange(len(v_exp), dtype=np.int64).reshape(-1, 3)
    out = trimesh.Trimesh(vertices=v_exp, faces=new_faces, process=False)
    out.fix_normals()
    return out, uv


def frame_glass_atlas_uv_mesh(
    frame_mesh: trimesh.Trimesh,
    glass_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Один меш рама+стекло: левая половина U атласа — рама, правая — стекло (как в OBJ-экспорте)."""
    mf_uv, uv_f = faceted_triplanar_uv(frame_mesh)
    mg_uv, uv_g = faceted_triplanar_uv(glass_mesh)
    uv_f = np.asarray(uv_f, dtype=np.float64).copy()
    uv_g = np.asarray(uv_g, dtype=np.float64).copy()
    uv_f[:, 0] = uv_f[:, 0] * 0.5
    uv_g[:, 0] = uv_g[:, 0] * 0.5 + 0.5
    uv = np.vstack([uv_f, uv_g]) if len(uv_f) + len(uv_g) else np.zeros((0, 2))

    if len(mf_uv.faces) == 0 and len(mg_uv.faces) == 0:
        wt = trimesh.Trimesh()
    elif len(mf_uv.faces) == 0:
        wt = mg_uv
        uv = uv_g
    elif len(mg_uv.faces) == 0:
        wt = mf_uv
        uv = uv_f
    else:
        wt = trimesh.util.concatenate([mf_uv, mg_uv])
    return wt, uv
