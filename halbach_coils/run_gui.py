r"""
Launch the pyCoilGen Tkinter GUI.

    .\.venv\Scripts\python.exe halbach_coils\run_gui.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gui.main_window import main

if __name__ == '__main__':
    main()
