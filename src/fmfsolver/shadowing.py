from __future__ import annotations

import numpy as np
import trimesh

def compute_shadow_mask(mesh: trimesh.Trimesh, centers_m: np.ndarray, Vhat: np.ndarray) -> np.ndarray:
    """Compute shadowed faces by casting rays from each face center along -V direction.

    `rtree` is REQUIRED (declared as dependency) for trimesh ray acceleration.
    """
    Vhat = np.asarray(Vhat, dtype=float)
    Vn = float(np.linalg.norm(Vhat))
    if Vn == 0.0:
        raise ValueError("Vhat has zero norm.")
    d = -Vhat / Vn

    bbox = mesh.bounds
    L = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(1e-9, 1e-6 * L)

    origins = centers_m + d[None, :] * eps
    directions = np.repeat(d[None, :], len(centers_m), axis=0)

    tri_idx, ray_idx, _loc = mesh.ray.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
        return_locations=True,
    )

    shadowed = np.zeros(len(centers_m), dtype=bool)
    for t, r in zip(tri_idx, ray_idx):
        if int(t) != int(r):
            shadowed[int(r)] = True
    return shadowed
