"""Writers for mesh and array artifacts produced by the solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def export_vtp(
    out_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    cell_data: dict,
    field_data: dict | None = None,
):
    """Write face-based solver outputs as a VTP PolyData file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    faces = faces.astype(np.int64)
    verts = vertices.astype(float)

    f = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(verts, f)

    for k, v in cell_data.items():
        poly.cell_data[k] = np.asarray(v)

    if field_data:
        for k, v in field_data.items():
            poly.field_data[k] = np.asarray([v])

    poly.save(str(out), binary=True)


def export_npz(out_path: str, **arrays):
    """Write multiple named arrays into a compressed NPZ file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **arrays)
