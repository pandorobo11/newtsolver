"""GUI application bootstrap."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .main_window import MainWindow


def main():
    """Launch the Qt-based FMF solver GUI."""
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
