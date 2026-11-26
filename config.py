"""
Ayarlar — config.json (uygulama klasöründe; .exe'de .exe'nin yanında).

Bilinmeyen anahtarlar yok sayılır, eksikler varsayılanla doldurulur, tipler
varsayılanın tipine dönüştürülür. Dosya bozuksa varsayılanlarla devam edilir.
Ayarlar menüsü (S) değişiklikleri buraya kaydeder.
"""

import json
import os
import sys

DEFAULTS = {
    # Pencere
    "width": 1280,
    "height": 720,
    "fullscreen": False,
    # Ses
    "device": "",                 # boş = otomatik; dolu = cihaz adında geçen metin (ör. "CABLE")
    "sensitivity": 1.0,
    # Görsel
    "beat_sync": True,            # flash / zoom / sahne ve stil geçişleri beat ızgarasına otursun
    "section_detection": True,    # bölüm değişince (nakarat, drop…) yeni stil + sahne
    "flash": 1.0,
    "zoom_pulse": 1.0,
    "palette_drift": 0.025,
    "scene_duration": 22.0,
    "show_hud": True,
    "diff_bars": 1,               # difüzyon: kaç ölçüde bir yeni sahne (beat kilitliyken)
    "diff_valence_bias": 0.0,     # difüzyon: pozitiflik eğilimi (−1 karamsar … +1 neşeli)
    "lyrics": True,               # şarkı sözlerini dinle (Whisper) ve sahneye kat
    "lyrics_model": "openai/whisper-medium",   # ya da openai/whisper-large-v3 / whisper-small
    "lyrics_window": 4.0,         # Whisper penceresi (sn) — kısa = daha az token, daha hızlı
    "lyrics_step": 0.5,           # pencere adımı (sn) — çıkarım daha uzunsa döngü beklemeden sürer
    "lyrics_translate_first": True,   # Türkçe: SD'ye giden çeviri tek generate ile önce
    "lyrics_hud_original": True,  # Türkçe orijinali (HUD) yeni satırlarda sonradan üret
    "lyrics_reuse_encoder": True, # dil tespitinin encoder çıktısını generate'e ver (hata → eski yol)
    "lyric_priority": True,       # yeni söz satırı MIN_SCENE'i beklemeden sahne kursun
    "lyric_fade": 0.06,           # söz satırının ilk karesinde çapraz geçiş (sn, 0 = anında)
    "latency_log": True,          # her söz satırı için [Gecikme] logu + HUD
    "diff_fast": True,            # channels_last + cudnn.benchmark + inference_mode (hata → eski yol)
    "diff_compile": False,        # torch.compile (Windows'ta triton gerekir; hata → eski yol)
    # Kayıt
    "record_fps": 30,
    "record_crf": 18,             # x264 kalite: düşük = daha iyi (18 ≈ görsel kayıpsız)
    "record_dir": "recordings",
    "record_hud": False,
}


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(app_dir(), "config.json")
# Veri, feature, HF önbelleği, checkpoint'ler (git'e girmez). AIVIS_WORK ile değiştirilebilir.
WORK_DIR = os.environ.get("AIVIS_WORK") or os.path.join(app_dir(), "work")


def _coerce(value, default):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "evet")
        return bool(value)
    if isinstance(default, int):
        return int(round(float(value)))
    if isinstance(default, float):
        return float(value)
    return str(value)


def load(path=CONFIG_PATH):
    cfg = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return cfg
    except (OSError, ValueError) as e:
        print(f"[Config] {path} okunamadı ({e}) — varsayılanlar kullanılıyor")
        return cfg
    if not isinstance(data, dict):
        return cfg
    for k, v in data.items():
        if k in DEFAULTS:
            try:
                cfg[k] = _coerce(v, DEFAULTS[k])
            except (TypeError, ValueError):
                print(f"[Config] '{k}' geçersiz ({v!r}) — varsayılan: {DEFAULTS[k]!r}")
    return cfg


def save(cfg, path=CONFIG_PATH):
    data = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        print(f"[Config] Kaydedilemedi: {e}")


def resolve_dir(p):
    """Göreli yolları uygulama klasörüne göre çözer."""
    return p if os.path.isabs(p) else os.path.join(app_dir(), p)
