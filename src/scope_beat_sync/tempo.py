from __future__ import annotations

import time


class TapTempo:
    """Calculates BPM from toggle-transitions on a boolean parameter.

    Each transition (True→False *or* False→True) counts as one tap.
    After two or more taps the BPM is derived from the mean inter-tap
    interval.  Falls back to manual BPM after *timeout* seconds of
    inactivity.
    """

    def __init__(self, max_taps: int = 8, timeout: float = 10.0) -> None:
        self.max_taps = max_taps
        self.timeout = timeout
        self._tap_times: list[float] = []
        self._prev_value: bool | None = None

    def update(self, tap_value: bool, now: float | None = None) -> None:
        """Call every frame with the current ``tap`` kwarg value."""
        if now is None:
            now = time.time()

        if self._prev_value is not None and tap_value != self._prev_value:
            self._tap_times.append(now)
            # Keep only the most recent taps
            if len(self._tap_times) > self.max_taps:
                self._tap_times = self._tap_times[-self.max_taps :]

        self._prev_value = tap_value

    def get_bpm(self, manual_bpm: float, now: float | None = None) -> float:
        """Return tap-derived BPM if active, otherwise *manual_bpm*."""
        if now is None:
            now = time.time()

        if not self.is_active(now):
            return manual_bpm

        if len(self._tap_times) < 2:
            return manual_bpm

        intervals = [
            self._tap_times[i] - self._tap_times[i - 1]
            for i in range(1, len(self._tap_times))
        ]
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval <= 0:
            return manual_bpm

        return 60.0 / mean_interval

    def is_active(self, now: float | None = None) -> bool:
        """True if taps have occurred within the timeout window."""
        if now is None:
            now = time.time()
        if not self._tap_times:
            return False
        return (now - self._tap_times[-1]) < self.timeout
