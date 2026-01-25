from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_results_excel(out_path: str, df_in: pd.DataFrame, df_out: pd.DataFrame):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_out2 = df_out.copy()
    for col in ["mode", "S", "Ti_K"]:
        if col in df_in.columns and col in df_out2.columns:
            df_out2.rename(columns={col: f"out_{col}"}, inplace=True)

    combined = pd.concat([df_in.reset_index(drop=True), df_out2.reset_index(drop=True)], axis=1)

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        combined.to_excel(w, index=False, sheet_name="summary")
