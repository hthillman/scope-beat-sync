# scope-beat-sync

Beat-reactive VACE conditioning preprocessor for [Daydream Scope](https://github.com/daydreamlive/scope).

## What it does

Scope generates AI video in 12-frame chunks. You can't beat-lock chunked output with a frame buffer. But VACE conditioning frames map 1:1 to output frames within a chunk — if the conditioning pulses to the beat, so does the generated output.

This plugin is a **preprocessor** that chains after a conditioning preprocessor (depth, edge, flow, etc.) and modulates the conditioning frames based on BPM before they reach the VACE pipeline.

```
Camera -> [Depth/Edge] -> conditioning -> [Beat Sync] -> beat-modulated conditioning -> [VACE] -> beat-reactive output
```

**Key properties:**
- **Per-frame beat phase** — each frame in a chunk gets its own beat phase, so modulation is smooth even across chunk boundaries
- **Multiple effects** — intensity, blur, invert, contrast, mask pulse
- **Multiple curves** — pulse, sine, square, sawtooth, triangle
- **Tap tempo** — set BPM by toggling a switch on each beat
- **VACE mask pulse** — modulate the VACE inpainting mask for more dramatic beat-reactive generation

## Installation

**Local path** (for development):
1. Open Scope Settings -> Plugins
2. Click Browse and select the `scope-beat-sync` folder
3. Click Install — Scope will restart

## Usage

1. Select any VACE pipeline as main (e.g. Wan2.1)
2. Select a conditioning preprocessor first (e.g. Video Depth Anything)
3. Select **Beat Sync** as the second preprocessor (it chains after the first)
4. Connect camera/video and enable VACE
5. Set BPM to match your music
6. Enable effects (Intensity is on by default)
7. The conditioning frames will visibly pulse in the canvas, and generated output inherits the beat reactivity

### Tap tempo

Toggle the **Tap Tempo** switch in rhythm with the music. After 2+ taps the plugin calculates BPM from the intervals. Falls back to the manual BPM slider after 10 seconds of inactivity.

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| BPM | float | 30-300 | 120 | Manual BPM (overridden by tap tempo) |
| Tap Tempo | toggle | - | off | Toggle on each beat to auto-detect BPM |
| Phase Offset | float | 0-1 | 0 | Phase shift (0 = downbeat, 0.5 = upbeat) |
| Curve | select | 5 options | pulse | Beat curve shape |
| Intensity | toggle | - | on | Brightness modulation |
| Intensity Amount | float | 0-1 | 0.5 | Brightness modulation depth |
| Blur | toggle | - | off | Blur off-beat, sharp on-beat |
| Blur Amount | float | 0-1 | 0.5 | Blur modulation depth |
| Invert | toggle | - | off | Invert conditioning on-beat |
| Invert Amount | float | 0-1 | 0.3 | Inversion depth |
| Contrast | toggle | - | off | Contrast boost on-beat |
| Contrast Amount | float | 0-1 | 0.5 | Contrast boost depth |
| Mask Pulse | toggle | - | off | Pulse VACE mask with beat |
| Mask Pulse Amount | float | 0-1 | 0.5 | VACE mask pulse depth |

## Development

Edit code, then click **Reload** next to the plugin in Settings -> Plugins. Changes take effect immediately.
