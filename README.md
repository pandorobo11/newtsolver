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
- Place `US1976.xlsx` in the working directory where you run the program (fixed filename).
- Input is an Excel file (header row + cases rows).
- Output summary is written as `<input_stem>_result.xlsx` in the same folder as the input.


### Optional/Required
- `rtree` is included as a dependency (required) for fast ray casting.
