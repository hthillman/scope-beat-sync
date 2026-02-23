from __future__ import annotations

import time


class TapTempo:
    """Calculates BPM from an incrementing tap counter.

    The UI renders an integer field with a +/− stepper. The user clicks +
    on each beat. Each increment (or decrement) of ``tap_count`` registers
    as one tap.

    If multiple taps land in the same pipeline chunk (user clicked fast
    between inference cycles), the taps are spaced evenly across the time
    since the last registered tap — giving a reasonable BPM estimate even
    when individual timestamps aren't available.

    Falls back to manual BPM after *timeout* seconds of inactivity.
    """

    def __init__(self, max_taps: int = 8, timeout: float = 10.0) -> None:
        self.max_taps = max_taps
        self.timeout = timeout
        self._tap_times: list[float] = []
        self._prev_count: int | None = None

    def update(self, tap_count: int, now: float | None = None) -> None:
        """Call every chunk with the current ``tap_count`` value."""
        if now is None:
            now = time.time()

        if self._prev_count is None:
            # First call — just record baseline, don't register taps
            self._prev_count = tap_count
            return

        delta = abs(tap_count - self._prev_count)
        if delta == 0:
            return

        self._prev_count = tap_count

        if delta == 1:
            # Single tap — straightforward
            self._tap_times.append(now)
        else:
            # Multiple taps arrived in one chunk — space them evenly
            # between the last known tap time and now
            last = self._tap_times[-1] if self._tap_times else now
            span = now - last
            if span > 0:
                step = span / delta
                for i in range(delta):
                    self._tap_times.append(last + step * (i + 1))
            else:
                # No time reference — just register them all at now
                for _ in range(delta):
                    self._tap_times.append(now)

        # Keep only the most recent taps
        if len(self._tap_times) > self.max_taps:
            self._tap_times = self._tap_times[-self.max_taps :]

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
