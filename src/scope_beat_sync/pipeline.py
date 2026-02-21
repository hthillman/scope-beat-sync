from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .buffer import TapTempo, TimestampedBuffer
from .schema import BeatSyncConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class BeatSyncPipeline(Pipeline):
    """Beat-locked frame buffer that absorbs inference jitter.

    Incoming frames are timestamped and stored in a ring buffer on CPU.
    Each output frame is selected (or interpolated) from the buffer at a
    time offset equal to ``beat_delay × 60 / bpm`` seconds behind the
    current moment, producing smooth, beat-aligned playback.
    """

    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig]:
        return BeatSyncConfig

    def __init__(self, device: torch.device | None = None, **kwargs) -> None:
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.buffer = TimestampedBuffer(max_frames=480)
        self.tap_tempo = TapTempo(max_taps=8, timeout=10.0)

    def prepare(self, **kwargs) -> Requirements:
        """Request exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        now = time.time()

        # --- Normalise input ------------------------------------------------
        video = kwargs.get("video")
        if video is None:
            raise ValueError("BeatSyncPipeline requires video input")

        frame = torch.stack([f.squeeze(0) for f in video], dim=0)
        frame = frame.to(device=self.device, dtype=torch.float32) / 255.0

        # --- Read runtime params from kwargs --------------------------------
        bpm = kwargs.get("bpm", 120.0)
        tap = kwargs.get("tap", False)
        beat_delay = kwargs.get("beat_delay", 1.0)
        interpolate = kwargs.get("interpolate", True)
        show_overlay = kwargs.get("show_overlay", True)

        # --- Tap tempo ------------------------------------------------------
        self.tap_tempo.update(tap, now)
        effective_bpm = self.tap_tempo.get_bpm(bpm, now)

        # --- Push frame into buffer -----------------------------------------
        self.buffer.push(frame, now)

        # --- Calculate target time ------------------------------------------
        delay_seconds = beat_delay * 60.0 / max(effective_bpm, 1.0)
        target_time = now - delay_seconds

        # --- Retrieve output frame ------------------------------------------
        oldest = self.buffer.oldest_time
        filling = oldest is None or target_time < oldest

        if filling or beat_delay == 0.0:
            # Still filling or delay disabled — passthrough
            output = frame
        elif interpolate:
            result = self.buffer.interpolate_frame(target_time, self.device)
            output = result if result is not None else frame
        else:
            result = self.buffer.get_frame(target_time, self.device)
            output = result if result is not None else frame

        # --- Cleanup old frames ---------------------------------------------
        self.buffer.cleanup(target_time - 1.0)

        # --- Overlay status bar ---------------------------------------------
        if show_overlay:
            output = self._draw_overlay(output, filling, delay_seconds, effective_bpm)

        return {"video": output.clamp(0, 1)}

    # ------------------------------------------------------------------

    @staticmethod
    def _draw_overlay(
        frame: torch.Tensor,
        filling: bool,
        delay_seconds: float,
        bpm: float,
    ) -> torch.Tensor:
        """Draw a 4px status bar at the top of the frame.

        Yellow while filling, green when synced.  Bar width encodes the
        current delay in seconds (capped at a visible maximum).
        """
        _T, _H, W, C = frame.shape
        bar_h = 4

        if filling:
            # Yellow: RGB (1, 0.85, 0)
            color = torch.tensor([1.0, 0.85, 0.0], device=frame.device)
            bar_w = W // 3  # partial bar while filling
        else:
            # Green: RGB (0, 0.9, 0.3)
            color = torch.tensor([0.0, 0.9, 0.3], device=frame.device)
            bar_w = W  # full width when synced

        result = frame.clone()
        result[:, :bar_h, :bar_w, :C] = color
        return result
