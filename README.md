# AI Music Visualizer

A real-time music visualizer that captures system audio and uses a local AI model to dynamically generate color palettes and visual parameters in response to what is playing.

## How It Works

The visualizer captures audio from your system output using WASAPI loopback (Windows). It performs FFT analysis across 8 frequency bands (sub-bass to air) and a built-in director turns those statistics into a scene, color palette, mood tag and animation parameters. No external AI service is needed: the only learned model is the project's own audio model (see below).

Visuals update in real time at 60 FPS using OpenGL via pyglet.

## Requirements

- Windows (WASAPI loopback is Windows-only)
- Python 3.10 or newer

Install Python dependencies:

```
pip install -r requirements.txt
```

## Setup

1. In your Windows sound settings, enable **Stereo Mix** or use a virtual audio cable (e.g., VB-Cable) so the visualizer can capture system audio.
2. Run the visualizer:

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

## Scene Director

Every 3 seconds `LocalDirector` reads an audio snapshot (RMS, onset activity, dynamics, BPM, bass/treble balance) and computes an energy score. Levels are normalized against a slowly decaying peak, so it adapts to any volume or device. From that it picks:

- A scene: `minimal`, `cosmic`, `flow`, `web`, `galaxy`, `full` or `storm`. A new scene must win two checks in a row before it switches.
- A 4-color palette: bass-heavy music shifts warm, treble-heavy shifts cool. Higher energy gives more saturated, more contrasting colors.
- A background color, a mood label, complexity, animation speed, and shape-rain bursts on very intense moments.

## Neural Background (CPPN + Generative Shader)

The default visual mode is a neural background rendered entirely on the GPU (`neural_bg.py`):

1. **CPPN pass:** a small MLP (12→16→16→16→4) runs per pixel. Its inputs are the pixel coordinates (after kaleidoscope folding and domain warping), an 8-dimensional latent vector, and time.
2. **Composite pass:** the network output is mapped to the current palette gradient, a mask that leaves negative space, and contour lines. The previous frame is fed back with zoom, rotation and flow trails.
3. **Display pass:** mipmap bloom, chromatic aberration, ACES tone mapping, vignette and film grain.

A *style* is a set of network weights plus shader parameters. Styles change automatically every 22–38 seconds, and instantly when a drop is detected. During a change the weights are interpolated, so one world morphs into the next.

The 8 latent dimensions come from the trained audio model (see below), which sets the song's overall character. Deviations in the 8 frequency bands are added on top for fast reactions. Without a model, the bands drive the latent on their own.

Styles saved with **K** go to `neural_styles/` and are mixed back into the rotation.

| Key | Action |
|---|---|
| G | Toggle neural background |
| B | Toggle preset shape layers |
| V | New style |
| K | Save current style to favorites |
| L | Lock / unlock current style |
| X | Toggle audio model |
| C | Toggle live feature capture (for training) |

## Audio Model (Spectrogram → Latent)

A small CNN (~1.2M parameters) maps a 2-second log-mel spectrogram (16 kHz, 64 mel bands, 128 frames) to an embedding. The 8 principal components of that embedding, whitened, become the CPPN latent. As a result, similar music produces similar visual worlds and different music produces different ones.

**Training is self-supervised (time-contrastive).** Two moments from the same track, 0.5–4 seconds apart, form a positive pair, and the other tracks in the batch act as negatives. The loss is symmetric InfoNCE, so no labels are needed. Augmentations (gain, EQ tilt, noise floor, SpecAugment) together with per-window mean removal make the model robust to system volume and to the playback chain.

Feature extraction ([audio_features.py](audio_features.py)) is shared between training and runtime, so the features always match. At runtime the model runs at 10 Hz on CPU via ONNX Runtime.

```
# Training environment (kept outside the repo, e.g. D:\ai-visualizer-work)
python -m venv D:\ai-visualizer-work\venv
D:\ai-visualizer-work\venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
D:\ai-visualizer-work\venv\Scripts\pip install -r train/requirements.txt

# 1) Features from a dataset (FMA-small: 8000 CC-licensed 30 s tracks), read straight from the zip
python train/prepare.py --zip D:\ai-visualizer-work\data\fma_small.zip --out D:\ai-visualizer-work\features\fma

# 2) Optional: press C in the visualizer while listening to music.
#    Only log-mel features are stored, never audio. Tracks are split on silence.

# 3) Train → models/audio_latent.onnx
python train/train.py --steps 8000
python train/emb_stats.py --data D:i-visualizer-workeaturesma
```

`emb_stats.py` saves whitening statistics for the model's 128-dim embedding (`models/audio_latent_emb.npz`), which dream mode needs.

### Emotion (DEAM + CLIP)

Dream mode turns the music's emotion into visuals with two more learned parts:

```
python train/prepare.py --zip D:i-visualizer-work\data\deam\DEAM_audio.zip --out D:i-visualizer-workeatures\deam
python train/emotion.py      # emotion head: embedding -> valence/arousal (DEAM human annotations)
python train/visual.py       # visual generator: (valence, arousal, music latent) -> CPPN world, guided by CLIP
```

- **Emotion head** (`models/emotion_head.npz`): a small MLP on top of the frozen audio model, trained on the DEAM dataset's per-half-second valence/arousal annotations (1802 songs). On 270 unseen songs: arousal R² 0.61, valence R² 0.27 (song level 0.71 / 0.42).
- **Visual generator** (`models/visual_gen.npz`): an MLP that outputs every CPPN weight, activation mix, visual parameter and the palette. It is trained through a differentiable torch copy of the shader; CLIP scores how well each image expresses the target emotion ("a sad, melancholic abstract painting" ... "a joyful, happy abstract painting", blended continuously by valence/arousal). A diversity term makes different music produce different images within the same emotion. No colors or shapes are hand-picked.

At runtime both run in numpy; CLIP and torch are only needed for training. The HUD shows the detected mood.

### Dream mode

When the model is loaded, the neural background runs in dream mode (toggle with **X**). Every hand-designed element is switched off: kaleidoscope folding, rotation around the center, the distance-from-center input, zoom/rotate feedback trails and the random style timer. Instead, `DreamMapper` (in `neural_bg.py`) whitens the model's embedding and maps it through a fixed random projection to every CPPN weight, activation mix, visual parameter and the color palette. The same musical moment always produces the same image; different music produces a different network.

The work directory can be changed with the `AIVIS_WORK` environment variable.

## Dependencies

| Package | Purpose |
|---|---|
| pyglet | OpenGL window and rendering |
| sounddevice | WASAPI audio capture |
| numpy | FFT and signal processing |
| scipy | Resampling and DSP utilities |
| onnxruntime | Runs the trained audio model |

## Notes

- No LLM or network service is used. Everything runs locally.
