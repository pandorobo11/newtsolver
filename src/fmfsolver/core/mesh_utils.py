from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import trimesh


@dataclass
class MeshData:
    mesh: trimesh.Trimesh
    centers_m: np.ndarray
    normals_out: np.ndarray
    areas_m2: np.ndarray


def load_meshes(stl_paths: List[str], scale_m_per_unit: float, logfn) -> MeshData:
    meshes = []
    for p in stl_paths:
        m = trimesh.load_mesh(Path(p).expanduser(), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        meshes.append(m)

    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    mesh.vertices = mesh.vertices.astype(float) * float(scale_m_per_unit)

    # Always fix/check normals
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception as e:
        logfn(f"[WARN] fix_normals failed: {e}")

    if mesh.is_watertight:
        try:
            if mesh.volume < 0:
                logfn("[WARN] Mesh volume negative -> inverting orientation (normals).")
                mesh.invert()
        except Exception as e:
            logfn(f"[WARN] volume/orientation check failed: {e}")
    else:
        logfn("[WARN] Mesh is not watertight (trimesh). Continuing anyway.")

    centers = mesh.triangles_center.astype(float)
    normals = mesh.face_normals.astype(float)
    areas = mesh.area_faces.astype(float)

    return MeshData(mesh=mesh, centers_m=centers, normals_out=normals, areas_m2=areas)
