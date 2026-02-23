"""Beat curve functions.

Each function maps a beat phase value in [0, 1] to a modulation
intensity in [0, 1].  Phase 0.0 means "on the beat" and 1.0 means
"just before the next beat".
"""

from __future__ import annotations

import math
from typing import Callable


def pulse(phase: float, decay: float = 5.0) -> float:
    """Sharp attack at phase=0, exponential decay toward 0."""
    return math.exp(-decay * phase)


def sine(phase: float) -> float:
    """Smooth cosine wave.  Peak (1.0) at phase=0, trough (0.0) at phase=0.5."""
    return (math.cos(2.0 * math.pi * phase) + 1.0) / 2.0


def square(phase: float, duty: float = 0.5) -> float:
    """Hard on/off.  1.0 for phase < *duty*, 0.0 otherwise."""
    return 1.0 if phase < duty else 0.0


def sawtooth(phase: float) -> float:
    """Linear ramp down.  1.0 at phase=0, 0.0 at phase≈1.0."""
    return 1.0 - phase


def triangle(phase: float) -> float:
    """Symmetric triangle.  1.0 at phase=0, 0.0 at phase=0.5, back to 1.0."""
    return 1.0 - 2.0 * abs(phase - 0.5)


CURVES: dict[str, Callable[[float], float]] = {
    "pulse": pulse,
    "sine": sine,
    "square": square,
    "sawtooth": sawtooth,
    "triangle": triangle,
}


def get_curve_value(curve_name: str, phase: float) -> float:
    """Look up *curve_name* and evaluate at *phase* (0–1)."""
    fn = CURVES.get(curve_name, sine)
    return fn(phase)
