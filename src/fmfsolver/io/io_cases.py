from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = [
    "case_id",
    "stl_path",
    "stl_scale_m_per_unit",
    "alpha_deg",
    "beta_deg",
    "Tw_K",
    "ref_x_m",
    "ref_y_m",
    "ref_z_m",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
]

DEFAULTS = {
    "shielding_on": 0,
    "save_vtp_on": 1,
    "save_npz_on": 0,
    "out_dir": "outputs",
}


def read_cases(path: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(p, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported input format: {p.suffix}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for k, v in DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
        else:
            df[k] = df[k].fillna(v)

    return df
