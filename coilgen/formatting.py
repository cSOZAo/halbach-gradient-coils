"""Number formatting shared by the metrics and sweep reports."""

from __future__ import annotations

from typing import Any

import numpy as np


def fmt_value(value: Any, fmt: str = '.6g') -> str:
    """Format *value* with *fmt*, rendering None / NaN / inf as ``n/a``."""
    if value is None:
        return 'n/a'
    try:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return 'n/a'
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)
