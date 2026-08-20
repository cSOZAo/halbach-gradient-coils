"""GUI length units: users enter millimetres; Config / pyCoilGen use metres."""

MM_PER_M = 1000.0


def mm_to_m(value_mm: float) -> float:
    return float(value_mm) / MM_PER_M


def m_to_mm(value_m: float) -> float:
    return float(value_m) * MM_PER_M
