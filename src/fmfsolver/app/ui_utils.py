from __future__ import annotations

def format_case_text(row: dict) -> str:
    parts: list[str] = []

    def add(k, v):
        if v is None:
            return
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return
        parts.append(f"{k}={s}")

    add("case_id", row.get("case_id"))
    add("stl", row.get("stl_path"))
    add("scale", row.get("stl_scale_m_per_unit"))
    add("alpha", row.get("alpha_deg"))
    add("beta", row.get("beta_deg"))
    add("Tw", row.get("Tw_K"))
    add("Aref", row.get("Aref_m2"))
    add("Lcl", row.get("Lref_Cl_m"))
    add("Lcm", row.get("Lref_Cm_m"))
    add("Lcn", row.get("Lref_Cn_m"))
    add("ref", f"({row.get('ref_x_m')},{row.get('ref_y_m')},{row.get('ref_z_m')})")

    # Mode A/B inputs (as entered)
    if str(row.get("S")).strip() not in ("", "nan", "None") and str(row.get("Ti_K")).strip() not in ("", "nan", "None"):
        add("mode", "A")
        add("S", row.get("S"))
        add("Ti", row.get("Ti_K"))
    elif str(row.get("Mach")).strip() not in ("", "nan", "None") and str(row.get("Altitude_km")).strip() not in ("", "nan", "None"):
        add("mode", "B")
        add("Mach", row.get("Mach"))
        add("Alt_km", row.get("Altitude_km"))

    add("shield", row.get("shielding_on"))
    return " | ".join(parts)
