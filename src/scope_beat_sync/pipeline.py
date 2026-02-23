"""Beat Sync preprocessor pipeline.

Chains after a conditioning preprocessor (depth, edge, flow) and
modulates the conditioning frames based on BPM.  The modulated frames
are returned both as canvas video *and* as VACE conditioning so the
generated output inherits the beat reactivity.
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
    compute_mask_value,
)
from .schema import BeatSyncConfig
from .tempo import TapTempo

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Assumed output FPS for per-frame time estimation
_FPS = 20.0


class BeatSyncPipeline(Pipeline):
    """Beat-reactive conditioning preprocessor.

    Receives conditioning frames (depth maps, edges, flow, etc.) from
    an upstream preprocessor, modulates them per-frame according to BPM
    and beat curve, and returns both modulated canvas video and
    VACE-formatted conditioning outputs.
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

    def prepare(self, **kwargs) -> Requirements:
        """Accept any number of input frames from the upstream preprocessor."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        now = time.time()

        # --- Normalise input ------------------------------------------------
        video = kwargs.get("video")
        if video is None:
            raise ValueError("BeatSyncPipeline requires video input")

        # Normalise frame sizes to handle resolution changes
        if isinstance(video, list):
            video = normalize_frame_sizes(video)
            frames = torch.stack([f.squeeze(0) for f in video], dim=0)  # [T, H, W, C]
        elif isinstance(video, torch.Tensor):
            frames = video if video.dim() == 4 else video.unsqueeze(0)
        else:
            raise TypeError(f"Unexpected video type: {type(video)}")

        frames = frames.to(device=self.device, dtype=torch.float32)

        # If frames came in [0, 255] normalise to [0, 1]
        if frames.max() > 1.5:
            frames = frames / 255.0

        T, H, W, C = frames.shape

        # --- Read runtime params from kwargs --------------------------------
        bpm = kwargs.get("bpm", 120.0)
        tap = kwargs.get("tap", False)
        phase_offset = kwargs.get("beat_phase_offset", 0.0)
        curve_name = kwargs.get("beat_curve", "pulse")

        intensity_on = kwargs.get("intensity_enabled", True)
        intensity_amt = kwargs.get("intensity_amount", 0.5)
        blur_on = kwargs.get("blur_enabled", False)
        blur_amt = kwargs.get("blur_amount", 0.5)
        invert_on = kwargs.get("invert_enabled", False)
        invert_amt = kwargs.get("invert_amount", 0.3)
        contrast_on = kwargs.get("contrast_enabled", False)
        contrast_amt = kwargs.get("contrast_amount", 0.5)
        mask_on = kwargs.get("mask_pulse_enabled", False)
        mask_amt = kwargs.get("mask_pulse_amount", 0.5)

        # --- Tap tempo ------------------------------------------------------
        self.tap_tempo.update(tap, now)
        effective_bpm = self.tap_tempo.get_bpm(bpm, now)
        beat_period = 60.0 / max(effective_bpm, 1.0)

        # --- Per-frame beat values ------------------------------------------
        # A 12-frame chunk at 20 FPS spans 0.6s.  At 120 BPM (0.5s/beat)
        # that's more than a full beat, so each frame needs its own phase.
        beat_vals = torch.zeros(T, device=self.device)
        for i in range(T):
            frame_time = now - (T - 1 - i) * (1.0 / _FPS)
            phase = ((frame_time / beat_period) + phase_offset) % 1.0
            beat_vals[i] = get_curve_value(curve_name, phase)

        # Broadcastable shape for per-frame element-wise ops: [T, 1, 1, 1]
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

        # --- Build VACE conditioning output ---------------------------------
        # vace_input_frames: list of [1, H, W, C] float32 in [0, 255]
        # When the downstream PreprocessVideoBlock sees a list it runs
        # preprocess_chunk() which stacks to BCTHW and normalises to [-1, 1].
        vace_frames = [(modulated[i : i + 1] * 255.0) for i in range(T)]

        # vace_input_masks: [1, 1, T, H, W] float32
        # 1.0 = generate from conditioning, 0.0 = preserve previous
        if mask_on and mask_amt > 0:
            mask_vals = [compute_mask_value(mask_amt, bv.item()) for bv in beat_vals]
        else:
            mask_vals = [1.0] * T
        masks = torch.tensor(mask_vals, device=self.device, dtype=torch.float32)
        masks = masks.view(1, 1, T, 1, 1).expand(1, 1, T, H, W)

        return {
            "video": modulated,  # [T, H, W, C] float32 [0, 1] — canvas display
            "vace_input_frames": vace_frames,  # list of [1, H, W, C] — VACE conditioning
            "vace_input_masks": masks,  # [1, 1, T, H, W] — VACE masks
        }
