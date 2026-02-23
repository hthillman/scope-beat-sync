from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class BeatSyncConfig(BasePipelineConfig):
    """Configuration for the Beat Sync preprocessor.

    Chains after a conditioning preprocessor (depth, edge, flow, etc.)
    and modulates the conditioning frames based on BPM so the generated
    output inherits beat-reactive visuals.
    """

    pipeline_id = "beat-sync"
    pipeline_name = "Beat Sync"
    pipeline_description = (
        "Beat-reactive conditioning preprocessor. Modulates depth maps, "
        "edges, or flow fields with BPM-synced effects so VACE output "
        "pulses to the beat."
    )

    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # --- BPM Control ---

    bpm: float = Field(
        default=120.0,
        ge=30.0,
        le=300.0,
        description="Manual BPM. Overridden by tap tempo when active.",
        json_schema_extra=ui_field_config(order=10, label="BPM"),
    )

    tap: bool = Field(
        default=False,
        description="Toggle on each beat to set BPM via tap tempo. Each flip counts as one tap.",
        json_schema_extra=ui_field_config(order=11, label="Tap Tempo"),
    )

    beat_phase_offset: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Phase shift. 0 = downbeat, 0.5 = upbeat.",
        json_schema_extra=ui_field_config(order=12, label="Phase Offset"),
    )

    # --- Timing ---

    timing_mode: Literal["clock", "counter"] = Field(
        default="clock",
        description=(
            "Phase timing strategy. "
            "'clock' snaps to wall-clock phase every chunk — stays locked to BPM, "
            "never drifts (best for live VJ / audio sync). "
            "'counter' uses a deterministic frame counter (smooth but not real-time locked)."
        ),
        json_schema_extra=ui_field_config(order=13, label="Timing Mode"),
    )

    target_fps: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description=(
            "Expected output FPS (counter mode only). "
            "Perceived BPM = actual_output_fps * set_BPM / target_fps. "
            "Set to your observed output FPS for accurate visual tempo."
        ),
        json_schema_extra=ui_field_config(order=14, label="Target FPS"),
    )

    reset_phase: bool = Field(
        default=False,
        description="Toggle to reset beat phase to zero. Flip on then off to trigger.",
        json_schema_extra=ui_field_config(order=15, label="Reset Phase"),
    )

    # --- Curve ---

    beat_curve: Literal["pulse", "sine", "square", "sawtooth", "triangle"] = Field(
        default="pulse",
        description="Shape of the beat modulation curve.",
        json_schema_extra=ui_field_config(order=30, label="Curve"),
    )

    # --- Effects ---

    intensity_enabled: bool = Field(
        default=True,
        description="Modulate brightness with beat.",
        json_schema_extra=ui_field_config(order=40, label="Intensity"),
    )
    intensity_amount: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Depth of brightness modulation.",
        json_schema_extra=ui_field_config(order=41, label="Intensity Amount"),
    )

    blur_enabled: bool = Field(
        default=False,
        description="Blur off-beat, sharp on-beat.",
        json_schema_extra=ui_field_config(order=42, label="Blur"),
    )
    blur_amount: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Depth of blur modulation.",
        json_schema_extra=ui_field_config(order=43, label="Blur Amount"),
    )

    invert_enabled: bool = Field(
        default=False,
        description="Invert conditioning on-beat (e.g. depth flip).",
        json_schema_extra=ui_field_config(order=44, label="Invert"),
    )
    invert_amount: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Depth of inversion modulation.",
        json_schema_extra=ui_field_config(order=45, label="Invert Amount"),
    )

    contrast_enabled: bool = Field(
        default=False,
        description="Boost contrast on-beat.",
        json_schema_extra=ui_field_config(order=46, label="Contrast"),
    )
    contrast_amount: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Depth of contrast modulation.",
        json_schema_extra=ui_field_config(order=47, label="Contrast Amount"),
    )

    # NOTE: Mask pulse removed — preprocessors are called per-frame so
    # they can't construct the full-chunk vace_input_masks tensor.
    # Mask modulation would need to live in the main pipeline.
