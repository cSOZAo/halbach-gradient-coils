"""
Entry parsing helpers for the GUI.

Every numeric field is a ``tk.StringVar``, so a typo yields a ``ValueError``
that must reach the panel's validation dialog instead of being dropped (which
would silently run with a default the user never chose). These helpers raise
``ValueError`` naming the offending field.
"""

from __future__ import annotations

from .units import mm_to_m


def parse_float(value: str, label: str) -> float:
    try:
        return float(str(value).strip().replace(',', '.'))
    except (TypeError, ValueError):
        raise ValueError(f"'{label}': '{value}' no es un numero valido.") from None


def parse_int(value: str, label: str) -> int:
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValueError(f"'{label}': '{value}' no es un entero valido.") from None


def parse_mm(value: str, label: str) -> float:
    """Parse a millimetre entry and return metres."""
    return mm_to_m(parse_float(value, label))
