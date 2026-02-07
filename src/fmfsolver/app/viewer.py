"""VTP viewer panel and visualization controls for the GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from .ui_utils import format_case_text


class ViewerPanel(QtWidgets.QWidget):
    """Right-side panel that renders VTP results and camera controls."""

    log_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        right_layout = QtWidgets.QVBoxLayout(self)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        try:
            self.plotter.enable_parallel_projection()
        except Exception:
            try:
                self.plotter.camera.parallel_projection = True
            except Exception:
                pass
        right_layout.addWidget(self.plotter.interactor, 6)

        ctrl = QtWidgets.QVBoxLayout()
        ctrl.setSpacing(4)
        ctrl.setContentsMargins(0, 0, 0, 0)

        self.cmb_scalar = QtWidgets.QComboBox()
        self.cmb_scalar.addItems(
            [
                "Cp_n",
                "shielded",
                "theta_deg",
                "area_m2",
                "center_x_stl_m",
                "center_y_stl_m",
                "center_z_stl_m",
            ]
        )

        self.chk_edges = QtWidgets.QCheckBox("Show edges")
        self.chk_edges.setChecked(True)

        self.chk_shield_transparent = QtWidgets.QCheckBox("Shielded transparent")
        self.chk_shield_transparent.setChecked(True)

        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["jet", "viridis", "bwr"])
        self.cmb_cmap.setCurrentText("jet")

        self.edit_vmin = QtWidgets.QLineEdit()
        self.edit_vmax = QtWidgets.QLineEdit()
        self.edit_vmin.setPlaceholderText("vmin (blank=auto)")
        self.edit_vmax.setPlaceholderText("vmax (blank=auto)")
        self.btn_auto_range = QtWidgets.QPushButton("Auto range")

        self.btn_open_vtp = QtWidgets.QPushButton("Open VTP...")
        self.btn_view_xp = QtWidgets.QPushButton("+X")
        self.btn_view_xn = QtWidgets.QPushButton("-X")
        self.btn_view_yp = QtWidgets.QPushButton("+Y")
        self.btn_view_yn = QtWidgets.QPushButton("-Y")
        self.btn_view_zp = QtWidgets.QPushButton("+Z")
        self.btn_view_zn = QtWidgets.QPushButton("-Z")
        self.btn_view_iso_1 = QtWidgets.QPushButton("-X -Y +Z")
        self.btn_view_iso_2 = QtWidgets.QPushButton("+X -Y -Z")
        self.btn_save_image = QtWidgets.QPushButton("Save Image...")

        self.lbl_scalar = QtWidgets.QLabel("Scalar:")
        self.lbl_colorbar = QtWidgets.QLabel("Colorbar range:")
        self.lbl_camera = QtWidgets.QLabel("Camera:")
        for lbl in (self.lbl_scalar, self.lbl_colorbar, self.lbl_camera):
            lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

        max_label_width = max(
            self.lbl_scalar.sizeHint().width(),
            self.lbl_colorbar.sizeHint().width(),
            self.lbl_camera.sizeHint().width(),
        )
        for lbl in (self.lbl_scalar, self.lbl_colorbar, self.lbl_camera):
            lbl.setFixedWidth(max_label_width)

        ref_height = self.btn_open_vtp.sizeHint().height()
        camera_buttons = [
            self.btn_view_xp,
            self.btn_view_xn,
            self.btn_view_yp,
            self.btn_view_yn,
            self.btn_view_zp,
            self.btn_view_zn,
            self.btn_view_iso_1,
            self.btn_view_iso_2,
            self.btn_save_image,
        ]
        max_width = max(b.sizeHint().width() for b in camera_buttons)
        for b in camera_buttons:
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            b.setFixedWidth(max_width)
            b.setMinimumHeight(ref_height)

        scalar_layout = QtWidgets.QHBoxLayout()
        scalar_layout.setSpacing(6)
        scalar_layout.setContentsMargins(0, 0, 0, 0)
        scalar_layout.addWidget(self.lbl_scalar)
        scalar_layout.addWidget(self.cmb_scalar)
        scalar_layout.addWidget(self.chk_edges)
        scalar_layout.addWidget(self.chk_shield_transparent)
        scalar_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        scalar_layout.addWidget(self.cmb_cmap)
        scalar_layout.addWidget(self.btn_open_vtp)
        scalar_layout.addStretch(1)

        colorbar_layout = QtWidgets.QHBoxLayout()
        colorbar_layout.setSpacing(6)
        colorbar_layout.setContentsMargins(0, 0, 0, 0)
        colorbar_layout.addWidget(self.lbl_colorbar)
        colorbar_layout.addWidget(self.edit_vmin)
        colorbar_layout.addWidget(self.edit_vmax)
        colorbar_layout.addWidget(self.btn_auto_range)
        colorbar_layout.addStretch(1)

        camera_block = QtWidgets.QVBoxLayout()
        camera_block.setSpacing(4)
        camera_block.setContentsMargins(0, 0, 0, 0)

        camera_row1 = QtWidgets.QHBoxLayout()
        camera_row1.setSpacing(6)
        camera_row1.setContentsMargins(0, 0, 0, 0)
        camera_row1.addWidget(self.lbl_camera)
        camera_row1.addWidget(self.btn_view_xp)
        camera_row1.addWidget(self.btn_view_xn)
        camera_row1.addWidget(self.btn_view_yp)
        camera_row1.addWidget(self.btn_view_yn)
        camera_row1.addWidget(self.btn_view_zp)
        camera_row1.addWidget(self.btn_view_zn)
        camera_row1.addStretch(1)

        camera_row2 = QtWidgets.QHBoxLayout()
        camera_row2.setSpacing(6)
        camera_row2.setContentsMargins(0, 0, 0, 0)
        spacer = QtWidgets.QLabel("")
        spacer.setFixedWidth(max_label_width)
        camera_row2.addWidget(spacer)
        camera_row2.addWidget(self.btn_view_iso_1)
        camera_row2.addWidget(self.btn_view_iso_2)
        camera_row2.addWidget(self.btn_save_image)
        camera_row2.addStretch(1)

        camera_block.addLayout(camera_row1)
        camera_block.addLayout(camera_row2)

        ctrl.addLayout(scalar_layout)
        ctrl.addLayout(colorbar_layout)
        ctrl.addLayout(camera_block)
        right_layout.addLayout(ctrl)

        self.df_cases = None
        self._poly: pv.PolyData | None = None
        self._display_case_row: dict | None = None
        self._overlay_actor = None
        self._default_view_vec = (-1, -1, 1)

        self.btn_open_vtp.clicked.connect(self.open_vtp)
        self.cmb_scalar.currentTextChanged.connect(self.update_view)
        self.chk_edges.toggled.connect(self.update_view)
        self.chk_shield_transparent.toggled.connect(self.update_view)
        self.cmb_cmap.currentTextChanged.connect(self.update_view)
        self.edit_vmin.editingFinished.connect(self.update_view)
        self.edit_vmax.editingFinished.connect(self.update_view)
        self.btn_auto_range.clicked.connect(self.clear_range)
        self.btn_view_xp.clicked.connect(lambda: self.set_view_vector((1, 0, 0)))
        self.btn_view_xn.clicked.connect(lambda: self.set_view_vector((-1, 0, 0)))
        self.btn_view_yp.clicked.connect(lambda: self.set_view_vector((0, 1, 0)))
        self.btn_view_yn.clicked.connect(lambda: self.set_view_vector((0, -1, 0)))
        self.btn_view_zp.clicked.connect(lambda: self.set_view_vector((0, 0, 1)))
        self.btn_view_zn.clicked.connect(lambda: self.set_view_vector((0, 0, -1)))
        self.btn_view_iso_1.clicked.connect(self.set_view_iso_1)
        self.btn_view_iso_2.clicked.connect(self.set_view_iso_2)
        self.btn_save_image.clicked.connect(self.save_view_image)

    def logln(self, s: str):
        """Emit a message to the shared GUI log."""
        self.log_message.emit(s)

    def set_cases_df(self, df):
        """Store loaded case table for overlay lookup by case_id."""
        self.df_cases = df

    def clear_range(self):
        """Reset scalar colorbar limits to automatic range."""
        self.edit_vmin.setText("")
        self.edit_vmax.setText("")
        self.update_view()

    def set_view_vector(self, vec):
        """Set camera direction to a Cartesian view vector."""
        self.plotter.view_vector(vec)
        self.plotter.render()

    def set_view_iso_1(self):
        """Set camera to ISO view ``(-X, -Y, +Z)``."""
        self.plotter.view_vector((-1, -1, 1))
        self.plotter.render()

    def set_view_iso_2(self):
        """Set camera to ISO view ``(+X, -Y, -Z)``."""
        self.plotter.view_vector((1, -1, -1))
        self.plotter.render()

    def save_view_image(self):
        """Save the current viewport image to a user-selected file."""
        if self.plotter is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save View Image",
            str(Path.cwd()),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            self.plotter.screenshot(path)
            self.logln(f"[OK] Saved image: {path}")
        except Exception as e:
            self.logln(f"[ERROR] Failed to save image: {e}")

    # -------------------------
    # VTP view
    # -------------------------
    def open_vtp(self):
        """Open a VTP file from disk and display it."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open VTP", str(Path.cwd()), "VTK PolyData (*.vtp)"
        )
        if not path:
            return
        self.load_vtp(path)

    def load_vtp(self, path: str, poly: pv.PolyData | None = None):
        """Load VTP data, resolve case context, and refresh rendering."""
        if poly is None:
            try:
                self._poly = pv.read(path)
            except Exception as e:
                self.logln(f"[ERROR] Failed to read VTP: {e}")
                self._poly = None
                return
        else:
            self._poly = poly

        # Determine which case this VTP corresponds to (for overlay)
        self._display_case_row = None
        case_id = None
        try:
            if "case_id" in self._poly.field_data:
                case_id = str(self._poly.field_data["case_id"][0])
        except Exception:
            case_id = None

        if case_id and self.df_cases is not None:
            m = self.df_cases[self.df_cases["case_id"].astype(str) == str(case_id)]
            if len(m):
                self._display_case_row = m.iloc[0].to_dict()

        self.logln(f"[VIEW] Loaded VTP: {path}")
        self.update_view()

    def _get_clim(self, poly: pv.PolyData, scalar: str):
        """Determine colorbar limits from UI inputs and scalar data."""
        if scalar == "shielded":
            return (0.0, 1.0)
        if scalar not in poly.cell_data:
            return None

        arr = np.asarray(poly.cell_data[scalar])

        vmin_txt = self.edit_vmin.text().strip()
        vmax_txt = self.edit_vmax.text().strip()
        if vmin_txt == "" and vmax_txt == "":
            return (float(np.nanmin(arr)), float(np.nanmax(arr)))

        try:
            vmin = float(vmin_txt) if vmin_txt != "" else float(np.nanmin(arr))
            vmax = float(vmax_txt) if vmax_txt != "" else float(np.nanmax(arr))
            if vmin == vmax:
                vmax = vmin + 1e-12
            return (vmin, vmax)
        except Exception:
            return (float(np.nanmin(arr)), float(np.nanmax(arr)))

    def _update_overlay(self):
        """Update the corner text overlay with the active case summary."""
        if self._overlay_actor is not None:
            try:
                self.plotter.remove_actor(self._overlay_actor)
            except Exception:
                pass
            self._overlay_actor = None

        if self._display_case_row is not None:
            txt = format_case_text(self._display_case_row)
        else:
            txt = ""
            if self._poly is not None:
                try:
                    if "case_id" in self._poly.field_data:
                        txt = f"case_id={self._poly.field_data['case_id'][0]}"
                except Exception:
                    txt = ""
            if txt == "":
                txt = "(no case info for displayed VTP)"

        self._overlay_actor = self.plotter.add_text(txt, position="upper_left", font_size=10)

    def update_view(self):
        """Redraw the current mesh with selected scalar, style, and camera."""
        if self._poly is None:
            return

        scalar = self.cmb_scalar.currentText()
        cmap = self.cmb_cmap.currentText()
        show_edges = self.chk_edges.isChecked()
        shield_transparent = self.chk_shield_transparent.isChecked()

        self.plotter.clear()
        poly = self._poly

        sh = (
            np.asarray(poly.cell_data["shielded"]).astype(int)
            if "shielded" in poly.cell_data
            else None
        )
        clim = self._get_clim(poly, scalar)

        if sh is not None:
            exposed_ids = np.where(sh == 0)[0]
            shielded_ids = np.where(sh == 1)[0]

            exposed = poly.extract_cells(exposed_ids) if len(exposed_ids) else None
            shielded = poly.extract_cells(shielded_ids) if len(shielded_ids) else None

            if exposed is not None and exposed.n_cells > 0:
                self.plotter.add_mesh(
                    exposed,
                    scalars=scalar if scalar in exposed.cell_data else None,
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    opacity=1.0,
                )
            if shielded is not None and shielded.n_cells > 0:
                opacity = 0.30 if shield_transparent else 1.0
                self.plotter.add_mesh(
                    shielded,
                    scalars=scalar if scalar in shielded.cell_data else None,
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    opacity=opacity,
                )
        else:
            self.plotter.add_mesh(
                poly,
                scalars=scalar if scalar in poly.cell_data else None,
                cmap=cmap,
                clim=clim,
                show_edges=show_edges,
                opacity=1.0,
            )

        self._update_overlay()
        self.plotter.add_axes()
        self.plotter.reset_camera()
        if self._default_view_vec is not None:
            self.plotter.view_vector(self._default_view_vec)
        self.plotter.render()
