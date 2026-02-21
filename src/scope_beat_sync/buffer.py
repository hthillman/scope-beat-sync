from __future__ import annotations

import time
from bisect import bisect_left

import torch


class TimestampedBuffer:
    """Ring buffer that stores frames with arrival timestamps for time-based lookup.

    Frames are stored on CPU to preserve VRAM. They are moved to the target
    device only when retrieved for output.
    """

    def __init__(self, max_frames: int = 480) -> None:
        self.max_frames = max_frames
        self._timestamps: list[float] = []
        self._frames: list[torch.Tensor] = []

    def push(self, frame: torch.Tensor, timestamp: float) -> None:
        """Store a frame with its arrival timestamp (CPU-side)."""
        self._timestamps.append(timestamp)
        self._frames.append(frame.detach().cpu())

        # Trim oldest if over capacity
        while len(self._timestamps) > self.max_frames:
            self._timestamps.pop(0)
            self._frames.pop(0)

    def get_frame(self, target_time: float, device: torch.device) -> torch.Tensor | None:
        """Nearest-neighbor lookup: return the frame closest to *target_time*."""
        if not self._timestamps:
            return None

        idx = self._nearest_index(target_time)
        return self._frames[idx].to(device=device, dtype=torch.float32)

    def interpolate_frame(
        self, target_time: float, device: torch.device
    ) -> torch.Tensor | None:
        """Linearly blend the two frames bracketing *target_time*."""
        n = len(self._timestamps)
        if n == 0:
            return None
        if n == 1:
            return self._frames[0].to(device=device, dtype=torch.float32)

        # Find insertion point
        pos = bisect_left(self._timestamps, target_time)

        if pos == 0:
            return self._frames[0].to(device=device, dtype=torch.float32)
        if pos >= n:
            return self._frames[-1].to(device=device, dtype=torch.float32)

        t0 = self._timestamps[pos - 1]
        t1 = self._timestamps[pos]
        dt = t1 - t0
        alpha = (target_time - t0) / dt if dt > 0 else 0.0

        f0 = self._frames[pos - 1].to(device=device, dtype=torch.float32)
        f1 = self._frames[pos].to(device=device, dtype=torch.float32)
        return f0 * (1.0 - alpha) + f1 * alpha

    def cleanup(self, oldest_allowed: float) -> None:
        """Remove frames older than *oldest_allowed*."""
        while self._timestamps and self._timestamps[0] < oldest_allowed:
            self._timestamps.pop(0)
            self._frames.pop(0)

    @property
    def oldest_time(self) -> float | None:
        """Timestamp of the oldest frame, or None if empty."""
        return self._timestamps[0] if self._timestamps else None

    @property
    def newest_time(self) -> float | None:
        """Timestamp of the newest frame, or None if empty."""
        return self._timestamps[-1] if self._timestamps else None

    def __len__(self) -> int:
        return len(self._timestamps)

    # ------------------------------------------------------------------

    def _nearest_index(self, target_time: float) -> int:
        """Return the index of the frame closest to *target_time*."""
        pos = bisect_left(self._timestamps, target_time)
        if pos == 0:
            return 0
        if pos >= len(self._timestamps):
            return len(self._timestamps) - 1
        # Compare neighbours
        if (target_time - self._timestamps[pos - 1]) <= (
            self._timestamps[pos] - target_time
        ):
            return pos - 1
        return pos


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
