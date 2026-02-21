from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class BeatSyncConfig(BasePipelineConfig):
    """Configuration for the Beat Sync pipeline."""

    pipeline_id = "beat-sync"
    pipeline_name = "Beat Sync"
    pipeline_description = (
        "Beat-locked frame buffer for VJ latency compensation. "
        "Absorbs inference jitter and outputs smooth, beat-aligned video "
        "with a configurable beat delay."
    )

    supports_prompts = False

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
        json_schema_extra=ui_field_config(order=20, label="Tap Tempo"),
    )

    # --- Buffer ---

    beat_delay: float = Field(
        default=1.0,
        ge=0.0,
        le=16.0,
        description="Number of beats to delay output. Higher values absorb more jitter but add latency.",
        json_schema_extra=ui_field_config(order=30, label="Beat Delay"),
    )

    # --- Output ---

    interpolate: bool = Field(
        default=True,
        description="Blend between frames for smoother output. Disable for nearest-neighbor (sharper transitions).",
        json_schema_extra=ui_field_config(order=40, label="Interpolate"),
    )

    show_overlay: bool = Field(
        default=True,
        description="Show a thin status bar at the top of the frame (yellow = filling, green = synced).",
        json_schema_extra=ui_field_config(order=50, label="Status Bar"),
    )
