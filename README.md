# scope-beat-sync

Beat-locked frame buffer for [Daydream Scope](https://github.com/daydreamlive/scope). Designed for VJs who need consistent latency compensation when running AI-generated visuals to displays.

## What it does

AI inference takes variable time per frame, causing output jitter. Display systems add their own latency. This plugin sits after the generation pipeline and absorbs that timing inconsistency by buffering frames and playing them back with a smooth, beat-synchronized delay.

Frames generated for beat N play back on beat N+X. The visuals still react to beats — just different beats than they were made for. As long as the delay is consistent, it's musically usable.

**Key properties:**
- **Timestamped buffer** — frames are stored with arrival times, output is selected by time-based lookup
- **Frame interpolation** — optionally blends between frames for sub-frame smooth output
- **Tap tempo** — set BPM by toggling a switch on each beat (no external clock needed)
- **CPU-side storage** — buffer lives in system RAM to preserve VRAM for AI pipelines

## Installation

**Local path** (for development):
1. Open Scope → Settings → Plugins
2. Click Browse and select the `scope-beat-sync` folder
3. Click Install — Scope will restart

**Git URL** (for sharing):
```
git+https://github.com/YOUR_USER/scope-beat-sync.git
```

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| BPM | float | 30–300 | 120 | Manual BPM (overridden by tap tempo when active) |
| Tap Tempo | toggle | — | off | Flip on each beat to auto-detect BPM |
| Beat Delay | float | 0–16 | 1.0 | Number of beats to delay output |
| Interpolate | toggle | — | on | Blend between frames for smoother output |
| Status Bar | toggle | — | on | Show coloured bar (yellow = filling, green = synced) |

## Usage

1. Run your AI generation pipeline (or connect a video source)
2. Select **Beat Sync** from the pipeline selector
3. Set BPM to match your music
4. Adjust **Beat Delay** to compensate for your display system's latency
5. The status bar turns green when the buffer is full and output is synced

### Tap tempo

Instead of manually entering BPM, toggle the **Tap Tempo** switch in rhythm with the music. After 2+ taps the plugin calculates BPM from the intervals. Falls back to the manual BPM slider after 10 seconds of inactivity.

## Development

Edit code, then click **Reload** next to the plugin in Settings → Plugins. Changes take effect immediately — no reinstall needed.

## Memory

Frames are stored on CPU (system RAM), not GPU VRAM. A 1080p frame is ~24 MB. At 120 BPM / 30 fps / 1 beat delay the buffer holds ~15 frames (~360 MB). The hard cap is 480 frames.
