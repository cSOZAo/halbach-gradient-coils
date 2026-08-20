"""Make the project root importable so ``coilgen`` / ``gui`` resolve in tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
