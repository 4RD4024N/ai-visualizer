# PyInstaller tarifi — tek klasörlük Windows sürümü.
#   pyinstaller visualizer.spec --noconfirm
# Çıktı: dist/AIVisualizer/AIVisualizer.exe  (klasörü zip'leyip paylaşın)
#
# Difüzyon katmanı (torch + diffusers, birkaç GB) pakete alınmaz; .exe onsuz çalışır.
# Eğitim kodu (train/) da dışarıda; sadece canlıda gereken train/visual.py difüzyonla
# birlikte kullanıldığından o da gerekmez.

a = Analysis(
    ["visualizer.py"],
    pathex=["."],
    datas=[
        ("models/audio_latent.onnx", "models"),
        ("models/audio_latent.json", "models"),
        ("models/audio_latent_emb.npz", "models"),
        ("models/emotion_head.npz", "models"),
        ("models/visual_gen.npz", "models"),
    ],
    hiddenimports=["pyglet.gl", "pyglet.window.win32", "pyglet.media.drivers.directsound"],
    excludes=["torch", "torchvision", "torchaudio", "diffusers", "transformers", "accelerate",
              "tensorflow", "matplotlib", "pandas", "IPython", "tkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIVisualizer",
    console=True,          # log satırları (cihaz, model, kayıt) görünsün
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AIVisualizer", upx=False)
