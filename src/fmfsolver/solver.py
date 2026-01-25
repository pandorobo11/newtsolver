from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

from .mesh_utils import load_meshes
from .shadowing import compute_shadow_mask
from .exporters import export_vtp, export_npz
from .us1976 import sample_at_altitude_km, mean_to_most_probable_speed
from .sentman_core import (
    vhat_from_alpha_beta_stl,
    sentman_dC_dA_vector,
    stl_to_body,
    rot_y,
    R_air_default,
    R_from_mol_weight_g_per_mol,
)

def _is_filled(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return str(x).strip() != ""

def _mode_from_row(row: dict) -> str:
    A_ok = _is_filled(row.get("S")) and _is_filled(row.get("Ti_K"))
    B_ok = _is_filled(row.get("Mach")) and _is_filled(row.get("Altitude_km"))
    if A_ok and B_ok:
        raise ValueError(f"Case '{row.get('case_id')}' has BOTH Mode A and Mode B inputs.")
    if (not A_ok) and (not B_ok):
        raise ValueError(f"Case '{row.get('case_id')}' has NEITHER complete Mode A nor Mode B inputs.")
    return "A" if A_ok else "B"

def _compute_S_Ti_R(row: dict) -> tuple[float, float, float, str]:
    mode = _mode_from_row(row)

    if mode == "A":
        S = float(row["S"])
        Ti = float(row["Ti_K"])
        R = R_air_default()
        return S, Ti, R, "A"

    Mach = float(row["Mach"])
    alt = float(row["Altitude_km"])

    atm = sample_at_altitude_km(alt, path_str="US1976.xlsx")
    Ti = float(atm["T_K"])
    c = float(atm["c_ms"])
    v_mean = float(atm["Vmean_ms"])
    v_mp = mean_to_most_probable_speed(v_mean)

    V_bulk = Mach * c
    S = V_bulk / v_mp
    R = R_from_mol_weight_g_per_mol(float(atm["M_g_per_mol"]))
    return S, Ti, R, "B"

def run_case(row: dict, logfn) -> dict:
    case_id = str(row["case_id"])
    stl_paths = [p.strip() for p in str(row["stl_path"]).split(";") if p.strip()]
    scale = float(row["stl_scale_m_per_unit"])

    Aref = float(row["Aref_m2"])
    Lref_Cl = float(row["Lref_Cl_m"])
    Lref_Cm = float(row["Lref_Cm_m"])
    Lref_Cn = float(row["Lref_Cn_m"])

    Tw = float(row["Tw_K"])
    ref_body = np.array([float(row["ref_x_m"]), float(row["ref_y_m"]), float(row["ref_z_m"])], dtype=float)

    alpha_deg = float(row["alpha_deg"])
    beta_deg = float(row["beta_deg"])

    shielding_on = bool(int(row.get("shielding_on", 0)))
    save_vtp = bool(int(row.get("save_vtp_on", 1)))
    save_npz = bool(int(row.get("save_npz_on", 0)))
    out_dir = Path(str(row.get("out_dir", "outputs"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    S, Ti, R, mode = _compute_S_Ti_R(row)

    Vhat = vhat_from_alpha_beta_stl(alpha_deg, beta_deg)

    md = load_meshes(stl_paths, scale, logfn)
    mesh = md.mesh
    centers_stl = md.centers_m
    normals_out_stl = md.normals_out
    areas = md.areas_m2

    if shielding_on:
        shadowed = compute_shadow_mask(mesh, centers_stl, Vhat)
    else:
        shadowed = np.zeros(len(areas), dtype=bool)

    C_face_stl = np.zeros((len(areas), 3), dtype=float)
    Cp_n = np.zeros(len(areas), dtype=float)
    eta_arr = np.zeros(len(areas), dtype=float)
    gamma_arr = np.zeros(len(areas), dtype=float)
    theta_deg = np.zeros(len(areas), dtype=float)

    C_force_stl = np.zeros(3, dtype=float)

    for i in range(len(areas)):
        dot_nv = float(np.dot(normals_out_stl[i], Vhat))
        dot_nv = max(-1.0, min(1.0, dot_nv))
        theta_deg[i] = math.degrees(math.acos(dot_nv))

        if shadowed[i]:
            continue

        dC_dA, eta, gam = sentman_dC_dA_vector(
            Vhat=Vhat,
            n_out=normals_out_stl[i],
            S=S,
            Ti=Ti,
            Tw=Tw,
            Aref=Aref,
        )
        eta_arr[i] = eta
        gamma_arr[i] = gam

        Ci = dC_dA * areas[i]
        C_face_stl[i] = Ci
        C_force_stl += Ci

        Cp_n[i] = -(Aref / areas[i]) * float(np.dot(Ci, normals_out_stl[i]))

    C_force_body = stl_to_body(C_force_stl)

    CA = -float(C_force_body[0])
    CY =  float(C_force_body[1])
    CN = -float(C_force_body[2])

    C_M_body = np.zeros(3, dtype=float)
    for i in range(len(areas)):
        if shadowed[i]:
            continue
        center_body = stl_to_body(centers_stl[i])
        r = center_body - ref_body
        C_face_body = stl_to_body(C_face_stl[i])
        C_M_body += np.cross(r, C_face_body)

    Cl = float(C_M_body[0]) / Lref_Cl
    Cm = float(C_M_body[1]) / Lref_Cm
    Cn = float(C_M_body[2]) / Lref_Cn

    Ry = rot_y(math.radians(alpha_deg))
    C_stab = Ry @ C_force_body

    CD = -float(C_stab[0])
    CL = -float(C_stab[2])

    vtp_path = out_dir / f"{case_id}.vtp"
    npz_path = out_dir / f"{case_id}.npz"

    if save_vtp:
        cell_data = {
            "area_m2": areas,
            "shadowed": shadowed.astype(np.uint8),
            "Cp_n": Cp_n,
            "theta_deg": theta_deg,
            "eta": eta_arr,
            "gamma": gamma_arr,
            "C_face_stl": C_face_stl,
            # NEW: face center scalars (STL axes, meters)
            "center_x_stl_m": centers_stl[:, 0],
            "center_y_stl_m": centers_stl[:, 1],
            "center_z_stl_m": centers_stl[:, 2],
        }
        export_vtp(
            str(vtp_path),
            mesh.vertices,
            mesh.faces,
            cell_data,
            field_data={"case_id": case_id},
        )

    if save_npz:
        export_npz(
            str(npz_path),
            vertices=mesh.vertices,
            faces=mesh.faces,
            centers_stl_m=centers_stl,
            normals_out_stl=normals_out_stl,
            areas_m2=areas,
            shadowed=shadowed,
            Vhat_stl=Vhat,
            S=S,
            Ti_K=Ti,
            Tw_K=Tw,
            Aref_m2=Aref,
            C_force_stl=C_force_stl,
            C_force_body=C_force_body,
            C_M_body=C_M_body,
            CA=CA, CY=CY, CN=CN, Cl=Cl, Cm=Cm, Cn=Cn, CD=CD, CL=CL,
            Cp_n=Cp_n,
        )

    return {
        "case_id": case_id,
        "mode": mode,
        "S": float(S),
        "Ti_K": float(Ti),
        "Tw_K": float(Tw),
        "CA": CA,
        "CY": CY,
        "CN": CN,
        "Cl": Cl,
        "Cm": Cm,
        "Cn": Cn,
        "CD": CD,
        "CL": CL,
        "faces": int(len(mesh.faces)),
        "shadowed_faces": int(shadowed.sum()),
        "vtp_path": str(vtp_path) if save_vtp else "",
        "npz_path": str(npz_path) if save_npz else "",
    }

def run_cases(df: pd.DataFrame, logfn) -> pd.DataFrame:
    rows = []
    total = len(df)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        logfn(f"[RUN] ({i}/{total}) case_id={r['case_id']}")
        rows.append(run_case(r.to_dict(), logfn))
    return pd.DataFrame(rows)
