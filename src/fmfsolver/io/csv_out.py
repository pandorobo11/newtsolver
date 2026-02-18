"""CSV output writer for aggregated case results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_cases import INPUT_COLUMN_ORDER


def _ordered_input_columns(df_in: pd.DataFrame) -> list[str]:
    """Return input-side columns in canonical order with unknown extras appended."""
    ordered = [c for c in INPUT_COLUMN_ORDER if c in df_in.columns]
    extras = [c for c in df_in.columns if c not in ordered]
    return ordered + extras


def _merge_input_output(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """Merge input/result rows while keeping input columns first."""
    df_in2 = df_in.reset_index(drop=True).copy()
    input_cols = _ordered_input_columns(df_in2)
    df_in2 = df_in2[input_cols]
    if "case_id" in df_in2.columns:
        df_in2["case_id"] = df_in2["case_id"].astype(str)

    df_out2 = df_out.reset_index(drop=True).copy()
    if "case_id" in df_out2.columns:
        df_out2["case_id"] = df_out2["case_id"].astype(str)

    overlap_cols = [c for c in df_out2.columns if c in df_in2.columns and c != "case_id"]
    if overlap_cols:
        df_out2.rename(columns={c: f"out_{c}" for c in overlap_cols}, inplace=True)

    if "case_id" in df_out2.columns and "case_id" in df_in2.columns:
        combined = df_out2.merge(df_in2, on="case_id", how="left", sort=False)
        out_cols = [c for c in combined.columns if c not in input_cols]
        combined = combined[input_cols + out_cols]

        # Keep final output ordered by the input case order regardless of
        # parallel completion order in upstream computation.
        case_order = {cid: i for i, cid in enumerate(df_in2["case_id"].astype(str).tolist())}
        combined["_case_order"] = (
            combined["case_id"].astype(str).map(case_order).fillna(len(case_order)).astype(int)
        )
        if "scope" in combined.columns:
            combined["_scope_order"] = combined["scope"].map({"total": 0, "component": 1}).fillna(2)
        else:
            combined["_scope_order"] = 0
        if "component_id" in combined.columns:
            combined["_component_order"] = (
                pd.to_numeric(combined["component_id"], errors="coerce").fillna(-1).astype(int)
            )
        else:
            combined["_component_order"] = -1
        combined = combined.sort_values(
            by=["_case_order", "_scope_order", "_component_order"],
            kind="mergesort",
        ).drop(columns=["_case_order", "_scope_order", "_component_order"])
        return combined
    return pd.concat([df_in2, df_out2], axis=1)


def write_results_csv(out_path: str, df_in: pd.DataFrame, df_out: pd.DataFrame):
    """Write merged input/result rows to a single CSV file.

    Result-side columns that overlap selected input names are prefixed with
    ``out_`` to avoid collisions.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = _merge_input_output(df_in, df_out)
    combined.to_csv(out, index=False)


def append_results_csv(out_path: str, df_in: pd.DataFrame, df_out: pd.DataFrame):
    """Append merged input/result rows to an existing CSV.

    If ``out_path`` does not exist, this writes a new CSV including headers.
    """
    if df_out is None or len(df_out) == 0:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = _merge_input_output(df_in, df_out)
    write_header = (not out.exists()) or out.stat().st_size == 0
    mode = "w" if write_header else "a"
    combined.to_csv(out, mode=mode, header=write_header, index=False)
