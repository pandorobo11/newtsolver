"""Top-level GUI window wiring cases and viewer panels."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .ui_cases import CasesPanel
from .viewer import ViewerPanel


class MainWindow(QtWidgets.QMainWindow):
    """Main application window for the FMF solver GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentman FMF Solver (GUI)")
        self.resize(1480, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.cases_panel = CasesPanel()
        self.viewer_panel = ViewerPanel()

        self.viewer_panel.log_message.connect(self.cases_panel.logln)
        self.cases_panel.vtp_loaded.connect(self.viewer_panel.load_vtp)
        self.cases_panel.cases_updated.connect(self.viewer_panel.set_cases_df)

        splitter.addWidget(self.cases_panel)
        splitter.addWidget(self.viewer_panel)
        splitter.setStretchFactor(1, 4)
