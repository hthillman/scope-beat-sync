"""Beat Sync preprocessor pipeline.

Chains after a conditioning preprocessor (depth, edge, flow) and
modulates the conditioning frames based on BPM.  Uses input_size=12
to match the main pipeline's chunk size so that beat-phase modulation
is applied to the exact frames that will be used for VACE inference.

Cross-chunk timing: the default "clock" mode derives phase from wall
time at every chunk boundary, so the visual never drifts from the
beat even when inference speed varies.  The within-chunk phase spread
uses an EMA of recent chunk durations for smooth frame-to-frame
progression.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .curves import get_curve_value
from .effects import (
    apply_blur,
    apply_contrast,
    apply_intensity,
    apply_invert,
)
from .schema import BeatSyncConfig
from .tempo import TapTempo

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Match the main VACE pipeline chunk size so that our per-frame beat
# modulation lands on the frames that actually reach inference.
_CHUNK_SIZE = 12

# EMA smoothing factor for chunk duration estimation (0–1, higher = more responsive)
_EMA_ALPHA = 0.3


class BeatSyncPipeline(Pipeline):
    """Beat-reactive conditioning preprocessor.

    Receives a chunk of conditioning frames (depth maps, edges, etc.)
    from an upstream preprocessor, modulates each frame according to
    its beat phase, and returns the modulated chunk.  Because
    input_size matches the downstream VACE chunk size, the per-frame
    beat pattern is preserved through to inference.
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
        self.tap_tempo = TapTempo(max_taps=8, timeout=10.0)

        # --- Timing state ---
        self._last_time: float | None = None
        self._chunk_dt_ema: float | None = None  # smoothed chunk duration
        self._frame_counter: int = 0

    def prepare(self, **kwargs) -> Requirements:
        """Request a full chunk so beat phasing survives subsampling."""
        return Requirements(input_size=_CHUNK_SIZE)

    # ------------------------------------------------------------------
    # Phase computation
    # ------------------------------------------------------------------

    def _compute_chunk_phases(
        self,
        now: float,
        num_frames: int,
        effective_bpm: float,
        phase_offset: float,
        timing_mode: str,
        target_fps: float,
    ) -> list[float]:
        """Return a beat phase [0, 1) for each frame in the chunk.

        **clock** (default) — phase is derived from wall time at each
        chunk boundary, so the visual never accumulates drift.  The
        within-chunk spread uses an EMA of recent chunk durations to
        keep frame-to-frame progression smooth.

        **counter** — deterministic: phase is a pure function of frame
        count and target FPS.  Smooth but not locked to real time.
        """
        beat_period = 60.0 / max(effective_bpm, 1.0)

        if timing_mode == "counter":
            phases: list[float] = []
            for i in range(num_frames):
                idx = self._frame_counter + i
                raw = (idx * effective_bpm) / (60.0 * max(target_fps, 1.0))
                phases.append((raw + phase_offset) % 1.0)
            self._last_time = now
            self._frame_counter += num_frames
            return phases

        # --- Clock mode (default) -------------------------------------------

        # 1. Snap start phase to wall clock — no drift, ever
        start_phase = (now / beat_period) % 1.0

        # 2. Estimate how long this chunk will play back for.
        #    Use EMA of measured inter-chunk dt for smooth progression.
        if self._last_time is not None:
            raw_dt = now - self._last_time
            raw_dt = max(min(raw_dt, 3.0), 0.01)  # clamp outliers

            if self._chunk_dt_ema is None:
                self._chunk_dt_ema = raw_dt
            else:
                self._chunk_dt_ema += _EMA_ALPHA * (raw_dt - self._chunk_dt_ema)
        else:
            # First chunk: fall back to target FPS estimate
            self._chunk_dt_ema = num_frames / max(target_fps, 1.0)

        chunk_phase_span = self._chunk_dt_ema / beat_period

        # 3. Spread phase evenly across frames
        phases = []
        for i in range(num_frames):
            frac = i / max(num_frames - 1, 1)
            p = start_phase + frac * chunk_phase_span
            phases.append((p + phase_offset) % 1.0)

        self._last_time = now
        self._frame_counter += num_frames
        return phases

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, **kwargs) -> dict:
        now = time.time()

        # --- Normalise input ------------------------------------------------
        video = kwargs.get("video")
        if video is None:
            raise ValueError("BeatSyncPipeline requires video input")

        if isinstance(video, list):
            video = normalize_frame_sizes(video)
            frames = torch.stack([f.squeeze(0) for f in video], dim=0)
        elif isinstance(video, torch.Tensor):
            frames = video if video.dim() == 4 else video.unsqueeze(0)
        else:
            raise TypeError(f"Unexpected video type: {type(video)}")

        frames = frames.to(device=self.device, dtype=torch.float32)

        if frames.max() > 1.5:
            frames = frames / 255.0

        T, H, W, C = frames.shape

        # --- Read runtime params from kwargs --------------------------------
        bpm = kwargs.get("bpm", 120.0)
        tap = kwargs.get("tap", False)
        phase_offset = kwargs.get("beat_phase_offset", 0.0)
        curve_name = kwargs.get("beat_curve", "pulse")
        timing_mode = kwargs.get("timing_mode", "clock")
        target_fps = kwargs.get("target_fps", 15.0)
        reset_phase = kwargs.get("reset_phase", False)

        intensity_on = kwargs.get("intensity_enabled", True)
        intensity_amt = kwargs.get("intensity_amount", 0.5)
        blur_on = kwargs.get("blur_enabled", False)
        blur_amt = kwargs.get("blur_amount", 0.5)
        invert_on = kwargs.get("invert_enabled", False)
        invert_amt = kwargs.get("invert_amount", 0.3)
        contrast_on = kwargs.get("contrast_enabled", False)
        contrast_amt = kwargs.get("contrast_amount", 0.5)

        # --- Phase reset ----------------------------------------------------
        if reset_phase:
            self._chunk_dt_ema = None
            self._frame_counter = 0
            self._last_time = None

        # --- Tap tempo ------------------------------------------------------
        self.tap_tempo.update(tap, now)
        effective_bpm = self.tap_tempo.get_bpm(bpm, now)

        # --- Per-frame beat values for the chunk ----------------------------
        phases = self._compute_chunk_phases(
            now, T, effective_bpm, phase_offset, timing_mode, target_fps
        )
        beat_vals = torch.tensor(
            [get_curve_value(curve_name, p) for p in phases],
            device=self.device,
        )
        beat_vals_4d = beat_vals.view(T, 1, 1, 1)

        # --- Apply effects --------------------------------------------------
        modulated = frames.clone()

        if intensity_on and intensity_amt > 0:
            modulated = apply_intensity(modulated, intensity_amt, beat_vals_4d)

        if blur_on and blur_amt > 0:
            modulated = apply_blur(modulated, blur_amt, beat_vals)

        if invert_on and invert_amt > 0:
            modulated = apply_invert(modulated, invert_amt, beat_vals_4d)

        if contrast_on and contrast_amt > 0:
            modulated = apply_contrast(modulated, contrast_amt, beat_vals_4d)

        modulated = modulated.clamp(0, 1)

        return {"video": modulated}
