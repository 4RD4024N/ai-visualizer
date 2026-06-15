# AI Music Visualizer

A real-time music visualizer that captures system audio and uses a local AI model to dynamically generate color palettes and visual parameters in response to what is playing.

## How It Works

The visualizer captures audio from your system output using WASAPI loopback (Windows). It performs FFT analysis across 8 frequency bands (sub-bass to air) and continuously sends audio snapshots to a locally running language model via LM Studio. The AI returns a color palette, mood tag, and parameters that drive the visuals — such as mandala rotation speed, particle rate, and wave amplitude.

Visuals update in real time at 60 FPS using OpenGL via pyglet.

## Requirements

- Windows (WASAPI loopback is Windows-only)
- Python 3.10 or newer
- [LM Studio](https://lmstudio.ai) running locally on port 1234 with a model loaded

Install Python dependencies:

```
pip install -r requirements.txt
```

## Setup

1. Open LM Studio, load any chat model, and start the local server on `http://localhost:1234`.
2. In your Windows sound settings, enable **Stereo Mix** or use a virtual audio cable (e.g., VB-Cable) so the visualizer can capture system audio.
3. Run the visualizer:

```
python visualizer.py
```

## Audio Sources

The visualizer automatically detects the best loopback device in this priority order:

1. Virtual cable output (e.g., VB-Cable)
2. Stereo Mix / What U Hear
3. Microphone fallback (if no loopback device is found)

## Frequency Bands

| Band | Range |
|---|---|
| Sub Bass | 20 - 60 Hz |
| Bass | 60 - 250 Hz |
| Low Mid | 250 - 500 Hz |
| Mid | 500 - 2000 Hz |
| High Mid | 2000 - 4000 Hz |
| Presence | 4000 - 6000 Hz |
| Brilliance | 6000 - 12000 Hz |
| Air | 12000 - 20000 Hz |

## AI Behavior

Every 3 seconds, a snapshot of the current audio energy per band, RMS level, and estimated BPM is sent to the LM Studio API. The model responds with:

- A 4-color palette (RGB values 0.0 to 1.0)
- Background color (kept near black)
- Mood label (e.g., energetic, calm, dark, ethereal)
- Mandala rotation speed
- Particle spawn rate
- Wave amplitude

High bass energy drives faster mandala rotation. High treble increases particle density. High RMS produces brighter, more vivid colors.

## Dependencies

| Package | Purpose |
|---|---|
| pyglet | OpenGL window and rendering |
| sounddevice | WASAPI audio capture |
| numpy | FFT and signal processing |
| scipy | Additional DSP utilities |

## Notes

- The `anthropic` package is not required. All AI calls go through LM Studio's local OpenAI-compatible API.
- If LM Studio is not running, the visualizer falls back to a default color palette and fixed visual parameters.
