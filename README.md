# fmfsolver (uv project)

## Install (uv)
```bash
uv sync
```

## Run
```bash
uv run fmfsolver
```

## Notes
- Input is an Excel file (header row + cases rows).
- Output summary is written as `<input_stem>_result.xlsx` in the same folder as the input.

## Data source
US1976 atmospheric tables (Table 1 and Table 2) are sourced from PDAS Big Tables:
```
https://www.pdas.com/bigtables.html
```


### Optional/Required
- `rtree` is included as a dependency (required) for fast ray casting.
