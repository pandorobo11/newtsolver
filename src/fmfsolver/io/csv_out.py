"""CSV output writer for aggregated case results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_results_csv(out_path: str, df_in: pd.DataFrame, df_out: pd.DataFrame):
    """Write merged input/result rows to a single CSV file.

    Result-side columns that overlap selected input names are prefixed with
    ``out_`` to avoid collisions.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_in2 = df_in.reset_index(drop=True).copy()
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
        input_cols = list(df_in2.columns)
        out_cols = [c for c in combined.columns if c not in input_cols]
        combined = combined[input_cols + out_cols]
    else:
        combined = pd.concat([df_in2, df_out2], axis=1)

    combined.to_csv(out, index=False)
