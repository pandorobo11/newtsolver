from __future__ import annotations

"""Case signature hashing for output validation and cache identity."""

import hashlib
import json
import math

from ..surface_equations import (
    count_semicolon_entries,
    expand_equations_for_components,
    normalize_leeward_equation,
    normalize_windward_equation,
    split_semicolon_tokens,
)


def _canonical_surface_equation_value(
    raw_value,
    *,
    field_name: str,
    default_value: str,
    n_components: int,
    resolver,
) -> str:
    """Return canonical one-or-many equation selector string for signatures."""
    try:
        _eq_list, canonical = expand_equations_for_components(
            raw_value,
            default_value=default_value,
            resolver=resolver,
            n_components=n_components,
            field_name=field_name,
        )
        return canonical
    except Exception:
        raw = str(raw_value or "").strip().lower()
        return raw or default_value


def build_case_signature(row: dict) -> str:
    """Build a stable signature hash for case identity and cache validation.

    The signature is stored in VTP metadata and used to verify that a VTP file
    matches the currently selected case parameters.
    """
    keys = [
        "case_id",
        "stl_path",
        "stl_scale_m_per_unit",
        "Mach",
        "gamma",
        "windward_eq",
        "leeward_eq",
        "alpha_deg",
        "beta_or_bank_deg",
        "attitude_input",
        "ref_x_m",
        "ref_y_m",
        "ref_z_m",
        "Aref_m2",
        "Lref_Cl_m",
        "Lref_Cm_m",
        "Lref_Cn_m",
        "shielding_on",
        "ray_backend",
    ]
    numeric_keys = {
        "stl_scale_m_per_unit",
        "Mach",
        "gamma",
        "alpha_deg",
        "beta_or_bank_deg",
        "ref_x_m",
        "ref_y_m",
        "ref_z_m",
        "Aref_m2",
        "Lref_Cl_m",
        "Lref_Cm_m",
        "Lref_Cn_m",
        "shielding_on",
    }

    def norm_value(k, v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if k in numeric_keys:
            try:
                return float(v)
            except Exception:
                return str(v)
        return str(v)

    n_components = max(count_semicolon_entries(row.get("stl_path")), 1)
    stl_tokens = [t for t in split_semicolon_tokens(row.get("stl_path")) if t]
    canonical_stl = ";".join(stl_tokens) if stl_tokens else ""

    data = {k: norm_value(k, row.get(k)) for k in keys}
    data["stl_path"] = canonical_stl
    data["windward_eq"] = _canonical_surface_equation_value(
        row.get("windward_eq"),
        field_name="windward_eq",
        default_value="newtonian",
        n_components=n_components,
        resolver=normalize_windward_equation,
    )
    data["leeward_eq"] = _canonical_surface_equation_value(
        row.get("leeward_eq"),
        field_name="leeward_eq",
        default_value="shield",
        n_components=n_components,
        resolver=normalize_leeward_equation,
    )
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
