"""Beat Sync preprocessor pipeline.

Chains after a conditioning preprocessor (depth, edge, flow) and
modulates the conditioning frames based on BPM.  Returns modulated
video frames — the pipeline processor handles collecting frames into
chunks and constructing VACE inputs.
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


class BeatSyncPipeline(Pipeline):
    """Beat-reactive conditioning preprocessor.

    Receives conditioning frames (depth maps, edges, flow, etc.) from
    an upstream preprocessor, modulates them per-frame according to BPM
    and beat curve, and returns the modulated video.  The pipeline
    processor collects frames into chunks and builds VACE inputs.
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

        # --- Phase tracking state ---
        self._phase: float = 0.0
        self._last_time: float | None = None
        self._frame_counter: int = 0

    def prepare(self, **kwargs) -> Requirements:
        """Accept any number of input frames from the upstream preprocessor."""
        return Requirements(input_size=1)

    # ------------------------------------------------------------------
    # Phase computation
    # ------------------------------------------------------------------

    def _advance_phase(
        self,
        now: float,
        effective_bpm: float,
        phase_offset: float,
        timing_mode: str,
        target_fps: float,
    ) -> float:
        """Advance the internal phase and return the current beat phase [0, 1).

        **accumulator** — phase += measured_dt * bpm / 60.  Adapts to the
        actual preprocessor call rate instead of assuming a fixed FPS.

        **counter** — phase is a pure function of frame count and target FPS.
        Deterministic visual rhythm; perceived BPM scales with actual
        output FPS relative to *target_fps*.
        """
        beat_period = 60.0 / max(effective_bpm, 1.0)

        if timing_mode == "counter":
            raw_phase = (self._frame_counter * effective_bpm) / (
                60.0 * max(target_fps, 1.0)
            )
            phase = (raw_phase + phase_offset) % 1.0
        else:
            # accumulator (default)
            if self._last_time is not None:
                dt = now - self._last_time
                # Clamp to avoid huge jumps from pauses / stalls
                dt = min(dt, 0.5)
                self._phase += dt / beat_period
            phase = (self._phase + phase_offset) % 1.0

        self._last_time = now
        self._frame_counter += 1
        return phase

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
        timing_mode = kwargs.get("timing_mode", "accumulator")
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
            self._phase = 0.0
            self._frame_counter = 0
            self._last_time = None

        # --- Tap tempo ------------------------------------------------------
        self.tap_tempo.update(tap, now)
        effective_bpm = self.tap_tempo.get_bpm(bpm, now)

        # --- Compute beat value (T is always 1 with input_size=1) -----------
        phase = self._advance_phase(
            now, effective_bpm, phase_offset, timing_mode, target_fps
        )
        beat_val = get_curve_value(curve_name, phase)
        beat_vals = torch.tensor([beat_val], device=self.device)
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

        # Return modulated video only — the pipeline processor collects
        # frames into chunks and builds vace_input_frames / vace_input_masks.
        return {"video": modulated}
