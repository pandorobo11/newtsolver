from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

_US1976_CACHE: pd.DataFrame | None = None

def _find_header_row(path: Path, max_scan_rows: int = 50) -> int:
    preview = pd.read_excel(path, header=None, nrows=max_scan_rows, engine="openpyxl")
    needed = {"Z", "T", "ρ", "c", "V", "M"}

    def norm(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and math.isnan(x):
            return ""
        return str(x).strip().replace(" ", "")

    for r in range(len(preview)):
        row = {norm(v) for v in preview.iloc[r].tolist()}
        if needed.issubset(row):
            return r
    raise ValueError("Failed to locate header row in US1976.xlsx (need Z,T,ρ,c,V,M).")

def load_us1976_fixed(path_str: str = "US1976.xlsx") -> pd.DataFrame:
    global _US1976_CACHE
    if _US1976_CACHE is not None:
        return _US1976_CACHE

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"US1976 file not found: {path.resolve()}")

    header_row = _find_header_row(path)
    df = pd.read_excel(path, header=header_row, engine="openpyxl")
    df.columns = [str(c).strip().replace(" ", "") for c in df.columns]

    for col in ["Z", "T", "ρ", "c", "V", "M"]:
        if col not in df.columns:
            raise ValueError(f"US1976.xlsx missing required column '{col}'.")

    out = df[["Z", "T", "ρ", "c", "V", "M"]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Z", "T", "ρ", "c", "V", "M"]).sort_values("Z").reset_index(drop=True)
    _US1976_CACHE = out
    return out

def sample_at_altitude_km(alt_km: float, path_str: str = "US1976.xlsx") -> dict:
    df = load_us1976_fixed(path_str)
    Z = df["Z"].to_numpy(dtype=float)

    if alt_km < Z.min() or alt_km > Z.max():
        raise ValueError(f"Altitude_km={alt_km} out of range [{Z.min()}, {Z.max()}]")

    def interp(col: str) -> float:
        y = df[col].to_numpy(dtype=float)
        return float(np.interp(alt_km, Z, y))

    return {
        "T_K": interp("T"),
        "rho_kgm3": interp("ρ"),
        "c_ms": interp("c"),
        "Vmean_ms": interp("V"),
        "M_g_per_mol": interp("M"),
    }

def mean_to_most_probable_speed(v_mean: float) -> float:
    return (math.sqrt(math.pi) / 2.0) * float(v_mean)
