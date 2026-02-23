"""Beat Sync preprocessor — parameter forwarder.

This preprocessor does NOT modulate pixels itself.  Instead it:
1. Resolves tap tempo into an effective BPM
2. Forwards all beat params as extra kwargs so they reach
   ``block_state`` in the main LongLive pipeline
3. Sets ``beat_sync_enabled = True`` to activate the beat
   modulation hook inside ``VaceEncodingBlock``

The actual modulation happens at VACE encode time — right before
``vace_encode_frames()`` — where the latency to display is just
the forward pass + denoise + decode (consistent and predictable).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import BeatSyncConfig
from .tempo import TapTempo

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Match the main VACE pipeline chunk size so conditioning frames
# pass through 1:1 without subsampling disruption.
_CHUNK_SIZE = 12


class BeatSyncPipeline(Pipeline):
    """Beat-reactive parameter forwarder.

    Chains after a conditioning preprocessor (depth, edge, flow).
    Passes conditioning frames through unmodified and forwards beat
    configuration params to the main pipeline where modulation is
    applied at VACE encode time for tight beat lock.
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
        """Request a full chunk so frames pass through 1:1."""
        return Requirements(input_size=_CHUNK_SIZE)

    def __call__(self, **kwargs) -> dict:
        now = time.time()

        # --- Normalise input --------------------------------------------------
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

        # --- Tap tempo --------------------------------------------------------
        tap = kwargs.get("tap", False)
        bpm = kwargs.get("bpm", 120.0)
        self.tap_tempo.update(tap, now)
        effective_bpm = self.tap_tempo.get_bpm(bpm, now)

        # --- Forward params + unmodified frames -------------------------------
        return {
            "video": frames,
            # Gate for the VACE encoding block hook
            "beat_sync_enabled": True,
            # BPM (resolved from tap tempo)
            "bpm": effective_bpm,
            # Curve / timing
            "beat_curve": kwargs.get("beat_curve", "pulse"),
            "beat_phase_offset": kwargs.get("beat_phase_offset", 0.0),
            "timing_mode": kwargs.get("timing_mode", "clock"),
            "target_fps": kwargs.get("target_fps", 15.0),
            "reset_phase": kwargs.get("reset_phase", False),
            # Effects
            "intensity_enabled": kwargs.get("intensity_enabled", True),
            "intensity_amount": kwargs.get("intensity_amount", 0.5),
            "blur_enabled": kwargs.get("blur_enabled", False),
            "blur_amount": kwargs.get("blur_amount", 0.5),
            "invert_enabled": kwargs.get("invert_enabled", False),
            "invert_amount": kwargs.get("invert_amount", 0.3),
            "contrast_enabled": kwargs.get("contrast_enabled", False),
            "contrast_amount": kwargs.get("contrast_amount", 0.5),
        }
