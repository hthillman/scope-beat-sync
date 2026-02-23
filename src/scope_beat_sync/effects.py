"""Beat-reactive modulation effects for conditioning frames.

Each effect is a pure function that takes frames in THWC float32 [0, 1]
format and returns modulated frames in the same format.  The *amount*
parameter controls effect depth (0 = off, 1 = full) and *beat_val* is
the current beat curve value (0 = off-beat, 1 = on-beat).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_intensity(
    frames: torch.Tensor, amount: float, beat_val: torch.Tensor
) -> torch.Tensor:
    """Scale brightness by beat.  On-beat = full brightness, off-beat = dimmed.

    *beat_val* is a [T, 1, 1, 1] tensor so it broadcasts per-frame.
    """
    scale = 1.0 - amount * (1.0 - beat_val)
    return frames * scale


def apply_blur(
    frames: torch.Tensor, amount: float, beat_val: torch.Tensor
) -> torch.Tensor:
    """Blur off-beat, sharp on-beat.  Uses downscale/upscale for speed.

    *beat_val* is a [T] tensor (we index per-frame because kernel varies).
    """
    T, H, W, C = frames.shape
    result = frames.clone()

    for i in range(T):
        bv = beat_val[i].item() if beat_val.dim() > 0 else beat_val.item()
        blur_strength = amount * (1.0 - bv)
        if blur_strength < 0.01:
            continue

        # Downscale factor: 1.0 (no blur) to 0.25 (heavy blur)
        scale = max(1.0 - blur_strength * 0.75, 0.25)
        small_h = max(int(H * scale), 1)
        small_w = max(int(W * scale), 1)

        # THWC -> CHW for interpolate
        f = frames[i].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        f = F.interpolate(f, size=(small_h, small_w), mode="bilinear", align_corners=False)
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        result[i] = f.squeeze(0).permute(1, 2, 0)  # back to HWC

    return result


def apply_invert(
    frames: torch.Tensor, amount: float, beat_val: torch.Tensor
) -> torch.Tensor:
    """Invert conditioning on-beat.  Useful for depth map inversion.

    *beat_val* is a [T, 1, 1, 1] tensor.
    """
    mix = amount * beat_val
    return torch.lerp(frames, 1.0 - frames, mix)


def apply_contrast(
    frames: torch.Tensor, amount: float, beat_val: torch.Tensor
) -> torch.Tensor:
    """Boost contrast on-beat.

    *beat_val* is a [T, 1, 1, 1] tensor.
    """
    gain = 1.0 + 2.0 * amount * beat_val
    return ((frames - 0.5) * gain + 0.5).clamp(0, 1)


def compute_mask_value(amount: float, beat_val: float) -> float:
    """Return VACE mask value modulated by beat.

    1.0 = generate from conditioning (on-beat), lower = preserve previous.
    """
    return 1.0 - amount * (1.0 - beat_val)
