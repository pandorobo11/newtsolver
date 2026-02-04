from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..core.solver import build_case_signature, run_cases
from ..io.excel_out import write_results_excel
from ..io.io_excel import read_cases


class CasesPanel(QtWidgets.QWidget):
    vtp_loaded = QtCore.Signal(str, object)
    cases_updated = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.xlsx_label = QtWidgets.QLabel("Excel: (not selected)")
        self.btn_pick_xlsx = QtWidgets.QPushButton("Select Excel Input")
        self.btn_run = QtWidgets.QPushButton("Run Selected Cases")
        self.btn_run.setEnabled(False)

        self.case_table = QtWidgets.QTableWidget()
        self.case_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.case_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.case_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.case_table.setAlternatingRowColors(True)
        self.case_table.horizontalHeader().setStretchLastSection(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)

        layout.addWidget(self.xlsx_label)
        layout.addWidget(self.btn_pick_xlsx)
        layout.addWidget(QtWidgets.QLabel("Cases:"))
        layout.addWidget(self.case_table, 3)
        layout.addWidget(self.btn_run)
        layout.addWidget(QtWidgets.QLabel("Log:"))
        layout.addWidget(self.log, 2)

        self.df_cases: pd.DataFrame | None = None
        self.xlsx_path: str | None = None

        self.btn_pick_xlsx.clicked.connect(self.pick_xlsx)
        self.btn_run.clicked.connect(self.run_selected)
        self.case_table.itemSelectionChanged.connect(self.on_case_selection_changed)

    def logln(self, s: str):
        self.log.appendPlainText(s)

    def pick_xlsx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Excel Input", str(Path.cwd()), "Excel (*.xlsx *.xlsm)"
        )
        if not path:
            return
        self.xlsx_path = path
        self.xlsx_label.setText(f"Excel: {path}")

        try:
            self.df_cases = read_cases(path)
        except Exception as e:
            self.logln(f"[ERROR] Failed to read Excel: {e}")
            self.df_cases = None
            self.btn_run.setEnabled(False)
            return

        self._populate_case_table()
        self.btn_run.setEnabled(True)
        self.logln(f"[OK] Loaded {len(self.df_cases)} case(s). Select and run.")
        self.cases_updated.emit(self.df_cases)

    def _populate_case_table(self):
        if self.df_cases is None:
            self.case_table.clear()
            self.case_table.setRowCount(0)
            return

        def _mode_from_row(r: dict) -> str:
            if pd.notna(r.get("S")) and pd.notna(r.get("Ti_K")):
                return "A"
            if pd.notna(r.get("Mach")) and pd.notna(r.get("Altitude_km")):
                return "B"
            return "?"

        cols = ["case_id", "mode"] + [c for c in self.df_cases.columns if c != "case_id"]
        self.case_table.clear()
        self.case_table.setColumnCount(len(cols))
        self.case_table.setRowCount(len(self.df_cases))
        self.case_table.setHorizontalHeaderLabels(cols)

        for row_idx, (_, row) in enumerate(self.df_cases.iterrows()):
            row_dict = row.to_dict()
            row_dict["mode"] = _mode_from_row(row_dict)
            for col_idx, col in enumerate(cols):
                val = row_dict.get(col, "")
                text = "" if pd.isna(val) else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, int(row_idx))
                self.case_table.setItem(row_idx, col_idx, item)

        self.case_table.resizeColumnsToContents()

    def on_case_selection_changed(self):
        if self.df_cases is None:
            return
        sel = self.case_table.selectionModel().selectedRows()
        if not sel:
            return
        row_idx = sel[0].row()
        item = self.case_table.item(row_idx, 0)
        if item is None:
            return
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        row = self.df_cases.loc[int(idx)].to_dict()
        case_id = str(row.get("case_id", "")).strip()
        out_dir = Path(str(row.get("out_dir", "outputs"))).expanduser()
        if not case_id:
            return
        vtp_path = out_dir / f"{case_id}.vtp"
        if not vtp_path.exists():
            return
        try:
            poly = pv.read(str(vtp_path))
        except Exception as e:
            self.logln(f"[ERROR] Failed to read VTP: {e}")
            return

        expected = build_case_signature(row)
        actual = None
        try:
            if "case_signature" in poly.field_data:
                actual = str(poly.field_data["case_signature"][0])
        except Exception:
            actual = None

        if actual == expected:
            self.vtp_loaded.emit(str(vtp_path), poly)

    def _confirm_overwrite(self, out_path: Path) -> bool:
        if not out_path.exists():
            return True
        resp = QtWidgets.QMessageBox.question(
            self,
            "Overwrite?",
            f"Result file already exists:\n{out_path}\n\nOverwrite it?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return resp == QtWidgets.QMessageBox.StandardButton.Yes

    def run_selected(self):
        if self.df_cases is None or self.xlsx_path is None:
            return

        sel = self.case_table.selectionModel().selectedRows()
        if sel:
            idxs = []
            for s in sel:
                item = self.case_table.item(s.row(), 0)
                if item is not None:
                    idxs.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
            df_sel = self.df_cases.loc[idxs].reset_index(drop=True)
        else:
            df_sel = self.df_cases.copy().reset_index(drop=True)

        self.logln(f"[RUN] Running {len(df_sel)} case(s)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = run_cases(df_sel, self.logln)

            out_summary = Path(self.xlsx_path)
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{out_summary.stem}_result.xlsx"

            if self._confirm_overwrite(out_path):
                write_results_excel(str(out_path), df_sel, res)
                self.logln(f"[OK] Wrote results: {out_path}")
            else:
                self.logln(f"[SKIP] Not overwriting existing file: {out_path}")

            if len(res) and str(res.loc[0, "vtp_path"]).strip():
                vtp_path = str(res.loc[0, "vtp_path"])
                try:
                    poly = pv.read(vtp_path)
                except Exception as e:
                    self.logln(f"[ERROR] Failed to read VTP: {e}")
                    return
                self.vtp_loaded.emit(vtp_path, poly)
        except Exception as e:
            self.logln(f"[ERROR] {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
