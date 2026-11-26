"""
AI Music Visualizer
WASAPI loopback → DSP + beat/bölüm analizi → neural arka plan + preset katmanlar (pyglet 2.x).

Modüller:
  audio_analyzer  ses yakalama, FFT bantları, beat takibi, bölüm algılama
  director        ses istatistiklerinden sahne / palet / ruh hali
  palette         renk paleti geçişleri
  layers/         preset görsel katmanları
  neural_bg       CPPN + üretken shader arka planı
  recorder        MP4 kaydı (ffmpeg)
  settings_menu   ekran üstü ayarlar (config.json)
"""

import math
import os
import queue
import random

import numpy as np
import pyglet
import pyglet.shapes as shapes

import config
import diffusion_layer
import lyrics as lyrics_mod
import latency
from audio_analyzer import AudioAnalyzer, BANDS
from audio_model import AudioTap, AudioLatentModel, MODEL_PATH
from director import LocalDirector
from layers import (WINDOW_W, WINDOW_H, clamp8, ShockwaveLayer, ConstellationLayer, ShapeRainLayer,
                    ParticleLayer, LissajousLayer, RibbonLayer, SpiralLayer, HelixLayer, TunnelLayer,
                    BurstRaysLayer, GridWaveLayer, OrbitsLayer)
from neural_bg import NeuralBackground
from palette import Palette
from recorder import VideoRecorder
from settings_menu import SettingsMenu

TARGET_FPS = 60
SCENE_APPLY_TIMEOUT = 2.5     # beat senkronunda sahne değişimi en fazla bu kadar bekler (sn)

# ── Ana Pencere ───────────────────────────────────────────────────────────────
class VisualizerWindow(pyglet.window.Window):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.load()
        super().__init__(
            width=self.cfg["width"], height=self.cfg["height"],
            caption="AI Music Visualizer",
            resizable=True, vsync=True,
        )
        self.set_minimum_size(640, 360)
        if self.cfg["fullscreen"]:
            self.set_fullscreen(True)

        # Blending aktif — additive glow için
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)

        self.audio       = AudioAnalyzer(device_hint=self.cfg["device"])
        self.dir_queue   = queue.Queue(maxsize=8)
        self.ai = LocalDirector(self.audio, self.dir_queue)
        self.palette     = Palette()
        self.palette.drift_speed = self.cfg["palette_drift"]
        self.particle_rate  = 1.0
        self._tgt_particle_rate = 1.0

        # Ritim durumu (beat ızgarası + bölümler)
        self.beat_phase   = 0.0
        self.beat_conf    = 0.0
        self.beat_count   = 0         # ölçü içindeki konum (0 = ölçü başı)
        self.section      = "intro"
        self._pending_dir = None      # beat senkronunda bir sonraki ölçü başını bekleyen sahne
        self._diff_bar    = 0         # difüzyon sahnesi için ölçü sayacı
        self._pending_t   = 0.0

        # Kayıt + ayarlar menüsü
        self.recorder = VideoRecorder(config.resolve_dir(self.cfg["record_dir"]),
                                      fps=self.cfg["record_fps"], crf=self.cfg["record_crf"])
        self.settings = SettingsMenu(self.cfg, self._on_setting)
        self._rec_msg_t = 0.0

        self.batch  = pyglet.graphics.Batch()
        self.batch2 = pyglet.graphics.Batch()  # HUD

        # Neural arka plan (CPPN + üretken shader). Başarısız olursa eski moda düşer.
        try:
            self.neural = NeuralBackground(self)
        except Exception as e:
            print(f"[Neural] Başlatılamadı, preset moda dönülüyor: {e}")
            self.neural = None
        self.neural_on    = self.neural is not None
        self.show_presets = self.neural is None   # neural açıkken presetler varsayılan kapalı

        # Difüzyon katmanı (rüya modu → nesneli sahneler). torch/CUDA yoksa kapalı.
        self.diff = None
        self.diff_on = False
        # Çapraz geçiş: yeni kare bir öncekinin üzerine opaklığı artarak gelir (iki sprite)
        self._diff_sprites = []       # [eski, yeni]
        self._diff_seen = -1
        self._diff_t = 0.0            # yeni karenin geldiği an
        self._diff_fade = 0.25        # geçiş süresi (difüzyon hızına göre ayarlanır)
        # Söz → ekran gecikmesi: Whisper, SD ve çizim damgaları (latency.py)
        self.latency = latency.LatencyTracker(enabled=self.cfg["latency_log"])
        if self.neural is not None and diffusion_layer.available():
            self.diff = diffusion_layer.DiffusionLayer(tap=None, tracker=self.latency,
                                                       lyric_priority=self.cfg["lyric_priority"],
                                                       fast=self.cfg["diff_fast"],
                                                       compile=self.cfg["diff_compile"])
            self.diff.valence_bias = self.cfg["diff_valence_bias"]
            self.diff.start()
            self.diff_on = True
            print("[Difüzyon] Arka planda yükleniyor...")
        else:
            print("[Difüzyon] Kapalı (torch + diffusers + CUDA gerekli — venv ile çalıştır)")

        # Ses modeli (spektrogram → latent) + canlı özellik kaydı
        self.tap = AudioTap()
        self.audio.tap = self.tap
        self.tap.start()
        self.audio_model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.audio_model = AudioLatentModel(self.tap)
                self.audio_model.start()
                print(f"[Model] Yüklendi: {MODEL_PATH}")
            except Exception as e:
                print(f"[Model] Yüklenemedi: {e}")
        else:
            print("[Model] models/audio_latent.onnx yok — bant tabanlı latent kullanılıyor")
        self.use_model = self.audio_model is not None
        self.lyrics = None
        if self.diff is not None:
            self.diff.tap = self.tap          # CLAP kavram seçimi için ham ses
            # Şarkı sözü dinleyici (Whisper) — difüzyonla aynı ortamı ister
            # transformers import'u thread-safe değil: difüzyon yüklenmeden başlatılmaz (bkz. update)
            self.lyrics = lyrics_mod.LyricsListener(
                self.tap, model=self.cfg["lyrics_model"], window=self.cfg["lyrics_window"],
                step=self.cfg["lyrics_step"], translate_first=self.cfg["lyrics_translate_first"],
                hud_original=self.cfg["lyrics_hud_original"],
                reuse_encoder=self.cfg["lyrics_reuse_encoder"], tracker=self.latency)
            self.lyrics.enabled = self.cfg["lyrics"]
            self.lyrics.on_line.append(self.diff.wake)    # yeni satır → SD thread'i hemen uyanır
            self.diff.lyrics = self.lyrics

        # Fade overlay
        self.fade = shapes.Rectangle(0, 0, WINDOW_W, WINDOW_H,
                                     color=(0, 0, 0, 35))

        # Katmanlar (çizim sırasına göre — arka → ön)
        self.layer_grid    = GridWaveLayer(self.batch)
        self.layer_tunnel  = TunnelLayer(self.batch)
        self.layer_ribbon  = RibbonLayer(self.batch)
        self.layer_spiral  = SpiralLayer(self.batch)
        self.layer_orbits  = OrbitsLayer(self.batch)
        self.layer_rays    = BurstRaysLayer(self.batch)
        self.layer_liss    = LissajousLayer(self.batch)
        self.layer_helix   = HelixLayer(self.batch)
        self.layer_const   = ConstellationLayer(self.batch)
        self.layer_shock   = ShockwaveLayer(self.batch)
        self.layer_parts   = ParticleLayer(self.batch)

        # ShapeRain (sürpriz katman)
        self.layer_rain = ShapeRainLayer(self.batch)

        # Beat flash overlay (additive)
        self.flash = shapes.Rectangle(0, 0, WINDOW_W, WINDOW_H,
                                      color=(255, 255, 255, 0), batch=None)
        self._flash_intensity = 0.0

        # AI kreatif direktör state
        self.ai_driving      = False
        self.complexity      = 0.6
        self.anim_speed      = 1.0
        self._tgt_complexity = 0.6
        self._tgt_speed      = 1.0
        self._cur_scene_name = "full"
        # AI sahne adı → aktif katmanlar
        self.ai_scene_map = {
            "flow":    ["ribbon","liss","parts","helix"],
            "galaxy":  ["spiral","rays","shock","parts"],
            "web":     ["const","liss","shock","grid"],
            "cosmic":  ["orbits","const","helix","parts"],
            "storm":   ["grid","rays","spiral","shock","parts"],
            "minimal": ["liss","orbits"],
            "full":    ["tunnel","ribbon","spiral","rays","liss","const","shock","parts"],
        }

        # ── Sahne sistemi ──────────────────────────────────────────
        # Aktif katmanlar (sahne sistemi bunu yönetir)
        self.layers_on    = {"grid":False, "tunnel":False, "ribbon":True,
                             "spiral":True, "orbits":False, "rays":False,
                             "liss":True, "helix":False, "const":True,
                             "shock":True, "parts":True}
        # Yumuşak geçiş: her katmanın anlık opaklığı + nefes fazı
        self.layer_op     = {k: 1.0 for k in self.layers_on}
        self.layer_phase  = {k: random.uniform(0, math.tau) for k in self.layers_on}
        # Her sahne = aktif katman kombinasyonu. Otomatik döner.
        self.scenes = [
            {"name":"Akis",      "on":["ribbon","liss","parts"]},
            {"name":"Galaksi",   "on":["spiral","rays","shock","parts"]},
            {"name":"Ag",        "on":["const","liss","shock"]},
            {"name":"Helix",     "on":["helix","const","parts"]},
            {"name":"Tunel",     "on":["tunnel","rays","parts"]},
            {"name":"Synthwave", "on":["grid","rays","shock"]},
            {"name":"Yorunge",   "on":["orbits","const","parts"]},
            {"name":"Patlama",   "on":["rays","spiral","shock","parts"]},
            {"name":"Tam",       "on":["ribbon","spiral","liss","const","shock","parts","rays"]},
            {"name":"Minimal",   "on":["liss","parts"]},
            {"name":"Kozmik",    "on":["spiral","const","helix"]},
        ]
        self.auto_scenes   = True
        self.scene_idx     = 3          # "Tam" ile başla
        self.scene_timer   = 0.0
        self.scene_dur     = self.cfg["scene_duration"]   # sahne süresi (sn)
        self._apply_scene(self.scene_idx)

        # ── Drop algılama ──────────────────────────────────────────
        self._energy_avg   = 0.0        # uzun ortalama enerji
        self._drop_cooldown = 0.0

        # Etkileşim state
        self.mouse_pos    = None
        self.mouse_mode   = 0       # 0=off, 1=cek, 2=it
        self.show_hud     = self.cfg["show_hud"]
        self.show_help    = False
        self.sensitivity  = self.cfg["sensitivity"]
        self._prev_bass   = 0.0

        # HUD
        self.fps_lbl  = pyglet.text.Label("", x=14, y=self.height-24,
                                          font_name="Consolas", font_size=11,
                                          color=(200,200,210,150), batch=self.batch2)
        self.mood_lbl = pyglet.text.Label("Baslatiliyor...", x=14, y=14,
                                          font_name="Consolas", font_size=14,
                                          color=(150,200,255,210), batch=self.batch2)
        self.hint_lbl = pyglet.text.Label("[H] yardım  [S] ayarlar  [F9] kayıt  [TAB] HUD",
                                          x=14, y=40, font_name="Consolas",
                                          font_size=10, color=(120,130,150,140),
                                          batch=self.batch2)
        self.scene_lbl = pyglet.text.Label("", x=14, y=66,
                                           font_name="Consolas", font_size=12,
                                           color=(255,200,120,200), batch=self.batch2)
        # Yardım overlay (ayrı batch, iki sütun)
        self.help_batch = pyglet.graphics.Batch()
        help_left = ("GENEL\n\n"
                     "S      Ayarlar menüsü\n"
                     "F9     Video kaydı başlat/durdur\n"
                     "F      Tam ekran\n"
                     "TAB    HUD göster/gizle\n"
                     "+/-    Hassasiyet\n"
                     "SPACE  Renk paletini karıştır\n"
                     "M      Fare modu (kapalı/çek/it)\n"
                     "H      Bu menü\n"
                     "ESC/Q  Çıkış\n\n"
                     "SAHNE / KATMAN\n\n"
                     "A      Otomatik sahne aç/kapa\n"
                     "N      Sonraki sahne\n"
                     "R      Şekil yağmuru\n"
                     "1-0,O  Katmanları aç/kapa")
        help_right = ("NEURAL / MODEL\n\n"
                      "G      Neural arka plan aç/kapa\n"
                      "B      Preset katmanları aç/kapa\n"
                      "V      Yeni neural stil\n"
                      "K      Stili favorilere kaydet\n"
                      "L      Stili kilitle/çöz\n"
                      "X      Ses modeli (rüya modu)\n"
                      "D      Difüzyon katmanı aç/kapa\n"
                      "[ / ]  Difüzyon gürültüsü\n"
                      "C      Canlı özellik kaydı (eğitim)")
        self.help_bg = shapes.Rectangle(0, 0, 780, 380,
                                        color=(10, 12, 22, 230), batch=self.help_batch)
        self.help_lbls = [pyglet.text.Label(txt, x=0, y=0, font_name="Consolas", font_size=12,
                                            color=(210, 220, 240, 255), multiline=True,
                                            width=370, batch=self.help_batch)
                          for txt in (help_left, help_right)]

        self.t       = 0.0
        self._fps_dt = 0.0; self._fps_fr = 0; self._fps_v = 0.0
        # Görsel yumuşatma + zoom pulse efekt state
        self.vis_bands   = np.zeros(BANDS)
        self._zoom       = 1.0
        self._zoom_pulse = 0.0

        self.audio.start()
        self.ai.start()
        pyglet.clock.schedule_interval(self.update, 1.0 / TARGET_FPS)
        print("[App] Hazir!")

    def update(self, dt):
        self.t += dt

        # AI direktifi — kreatif direktör modu
        try:
            while True:
                d = self.dir_queue.get_nowait()
                self.palette.set(d)
                self.mood_lbl.text = f"Mood: {d.get('mood','?')}"
                # Sahne değişimi: beat kilitliyse bir sonraki ölçü başını bekler
                scene = d.get("scene")
                if scene and scene in self.ai_scene_map and scene != self._cur_scene_name:
                    if self._beat_locked():
                        if self._pending_dir is None:
                            self._pending_t = self.t
                        self._pending_dir = scene
                    else:
                        self._set_director_scene(scene)
                # Karmaşıklık & hız hedefleri
                self._tgt_complexity = float(d.get("complexity", 0.6))
                self._tgt_speed      = float(d.get("speed", 1.0))
                self._tgt_particle_rate = 0.5 + self._tgt_complexity * 2.0
                if d.get("shape_rain"):
                    self.layer_rain.burst(n=22)
        except queue.Empty:
            pass
        # Smooth interpolate
        spd = dt * 1.2
        self.complexity    += (self._tgt_complexity - self.complexity) * min(1.0, spd)
        self.anim_speed    += (self._tgt_speed      - self.anim_speed) * min(1.0, spd)
        self.particle_rate += (self._tgt_particle_rate - self.particle_rate) * min(1.0, spd)

        self.palette.update(dt)

        bands_raw, rms, peak = self.audio.get()
        bands_raw = bands_raw * self.sensitivity
        rms_s = rms * self.sensitivity

        # ── Görsel yumuşatma: envelope follower ────────────────────
        # Hızlı atak (vuruşu yakala) + yavaş bırakma (akıcı iz) → pürüzsüz hareket
        for i in range(BANDS):
            target = float(bands_raw[i])
            if target > self.vis_bands[i]:
                self.vis_bands[i] += (target - self.vis_bands[i]) * min(1.0, dt * 22)  # atak
            else:
                self.vis_bands[i] += (target - self.vis_bands[i]) * min(1.0, dt * 7)   # bırakma
        bands = self.vis_bands

        # ── Ritim: beat ızgarası + bölümler ───────────────────────
        beats, sections, self.beat_phase, self.beat_conf, bpm, self.section = self.audio.rhythm()
        downbeat = False
        bass = float(bands_raw[1])
        fl, zp = self.cfg["flash"], self.cfg["zoom_pulse"]
        if self._beat_locked():
            # Izgaradan gelen beat'ler: vuruşla aynı anda, ölçü başı daha güçlü
            for _, idx, down in beats:
                self.beat_count = 0 if down else self.beat_count + 1
                k = (1.0 if down else 0.55) * (0.5 + min(1.0, bass))
                self._flash_intensity = min(1.0, self._flash_intensity + 0.45 * k * fl)
                self._zoom_pulse = min(0.10, self._zoom_pulse + 0.045 * k * zp)
                downbeat |= down
        else:
            # Kilit yok: bas yükselişi (eski yöntem)
            if bass - self._prev_bass > 0.25 and bass > 0.4:
                self._flash_intensity = min(1.0, self._flash_intensity + bass * 0.6 * fl)
                self._zoom_pulse = min(0.10, self._zoom_pulse + bass * 0.06 * zp)
        self._prev_bass = bass

        # Bekleyen sahne: ölçü başında (ya da zaman aşımında) uygula
        if self._pending_dir is not None and (downbeat or not self._beat_locked()
                                              or self.t - self._pending_t > SCENE_APPLY_TIMEOUT):
            self._set_director_scene(self._pending_dir)
            self._pending_dir = None

        if self.cfg["section_detection"]:
            for _, label, _ in sections:
                self._on_section(label)

        # Söz dinleyici: difüzyon yüklendikten (ya da düştükten) sonra başlar
        if self.lyrics is not None and not self.lyrics.running and (self.diff.ready or self.diff.failed):
            self.lyrics.start()

        # Difüzyon: bölüm etiketi + her beat'te nabız + her N ölçüde yeni sahne
        # (kilit yoksa katman kendi 3 sn zamanlayıcısını kullanır)
        if self.diff is not None:
            self.diff.section = self.section
            if beats and self.diff_on and self.diff.ready and self._beat_locked():
                self.diff.pulse()
        if downbeat and self.diff_on and self.diff is not None and self.diff.ready:
            self._diff_bar += 1
            if self._diff_bar >= self.cfg["diff_bars"]:
                self._diff_bar = 0
                self.diff.request_scene()
        self._flash_intensity *= (1 - dt * 5.0)
        # Zoom pulse yumuşak sönüm (yaylı geri dönüş)
        self._zoom_pulse *= (1 - dt * 4.0)
        self._zoom = 1.0 + self._zoom_pulse

        # ── Sahne otomatik rotasyonu ───────────────────────────────
        if self.auto_scenes:
            self.scene_timer += dt
            if self.scene_timer >= self.scene_dur:
                self.scene_timer = 0.0
                self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
                self._apply_scene(self.scene_idx)

        # ── Drop algılama → özel an ────────────────────────────────
        cur_energy = float(np.mean(bands))
        self._energy_avg = self._energy_avg * 0.99 + cur_energy * 0.01
        self._drop_cooldown -= dt
        # Ani büyük enerji sıçraması = drop
        if (cur_energy > self._energy_avg * 2.2 and cur_energy > 0.5
                and self._drop_cooldown <= 0):
            self._trigger_drop(cur_energy)
            self._drop_cooldown = 4.0   # 4sn boşluk

        # ShapeRain her zaman güncellenir (aktif şekiller düşsün)
        self.layer_rain.update(dt, bands, self.palette, self.t)

        # Fade alpha: sessizken daha koyu
        fade_a = max(12, min(50, 32 - int(rms_s * 120)))
        bg = self.palette.bg_gl()
        pyglet.gl.glClearColor(bg[0], bg[1], bg[2], 1.0)
        self.fade.color = (
            clamp8(bg[0]*255), clamp8(bg[1]*255), clamp8(bg[2]*255), fade_a
        )

        # Flash rengi — paletin en parlak rengiyle
        fc = self.palette.get(0, alpha=1.0)
        self.flash.color = (fc[0], fc[1], fc[2],
                            clamp8(self._flash_intensity * 60))

        # Neural arka plan: bantlar → latent, palet → renk gradyanı
        if self.neural_on:
            self.neural.master = 0.7 if self.show_presets else 1.0
            if self.use_model and self.audio_model.ready:
                # Rüya modu: görüntü tamamen modelin o anki ham çıktısından türetilir
                z, e = self.audio_model.raw()
                self.neural.set_external_latent(z, mix=1.0)
                self.neural.set_dream(e)
            else:
                self.neural.set_external_latent(None)
                self.neural.set_dream(None)
            self.neural.update(dt, bands, rms_s, self._flash_intensity,
                               palette=self.palette._draw, bg=self.palette.bg,
                               downbeat=downbeat, quantize=self._beat_locked())
            n = self.neural
            if self.diff_on and self.diff.ready and n.dreaming and hasattr(n.dream, "va"):
                self.diff.submit(n.cur, n.latent, n.dream.va)
            if n.dreaming and hasattr(n.dream, "mood"):
                v, a = n.dream.va
                txt = f"RÜYA · {n.dream.mood()}  (pozitiflik {v:+.2f}  enerji {a:+.2f})"
                if self.diff_on and self.diff is not None and self.diff.va_eff is not None:
                    dv, da = self.diff.va_eff                # difüzyonun kullandığı: model + CLAP
                    txt += f"  dif:{dv:+.2f}/{da:+.2f}"
                if self.diff_on and self.diff is not None and self.diff.concepts:
                    txt += (f"  · {self.diff.genre_name} · {self.diff.mood_name} · "
                            + ", ".join(c for c, _ in self.diff.concepts))
                    lo = self.lyrics.current_original() if self.lyrics is not None else None
                    if self.diff.lyric_used and lo:
                        txt += f'  ♪[{lo[1]}] "{" ".join(lo[0].split()[:14])}"'
                self.scene_lbl.text = txt
            elif n.dreaming:
                self.scene_lbl.text = "Neural RÜYA — model ne duyuyorsa o  [X: kapat]"
            else:
                tag = " [KILIT]" if n.locked else ""
                self.scene_lbl.text = f"Neural #{n.cur['seed']}{tag}  fav:{len(n.favorites)}"

        # ── Katman opaklıkları: yumuşak fade + organik nefes ───────
        for k in self.layers_on:
            tgt = 1.0 if self.layers_on[k] else 0.0
            # ~0.6sn'de fade (in: hızlı, out: biraz daha yavaş hisset)
            rate = 3.2 if tgt > self.layer_op[k] else 2.0
            self.layer_op[k] += (tgt - self.layer_op[k]) * min(1.0, dt * rate)
            # Yavaş nefes — transparanlık ara ara dalgalanır
            self.layer_phase[k] += dt * 0.35
        def op_of(k):
            breath = 0.82 + 0.18 * math.sin(self.layer_phase[k])
            return max(0.0, self.layer_op[k]) * breath

        # Katman güncellemeleri — complexity + speed + opacity ile
        cx_v = self.complexity
        sp_v = self.anim_speed
        dt_s = dt * sp_v
        VIS = 0.01
        P = self.palette
        m, mm = self.mouse_pos, self.mouse_mode

        def up(k, layer, **kw):
            if self.layer_op[k] > VIS:
                layer.update(dt_s, bands, P, self.t, opacity=op_of(k), **kw)
                setattr(self, f"_cleared_{k}", False)
            elif not getattr(self, f"_cleared_{k}", False):
                self.layer_op[k] = 0.0
                setattr(self, f"_cleared_{k}", True)
                self._clear_layer(layer)

        up("grid",   self.layer_grid,   complexity=cx_v)
        up("tunnel", self.layer_tunnel, complexity=cx_v)
        up("ribbon", self.layer_ribbon, complexity=cx_v)
        up("spiral", self.layer_spiral, complexity=cx_v)
        up("orbits", self.layer_orbits, complexity=cx_v)
        up("rays",   self.layer_rays,   complexity=cx_v)
        up("liss",   self.layer_liss,   complexity=cx_v)
        up("helix",  self.layer_helix,  complexity=cx_v)
        up("const",  self.layer_const,  complexity=cx_v, mouse=m, mouse_mode=mm)
        up("shock",  self.layer_shock)
        up("parts",  self.layer_parts,  rate_mult=self.particle_rate, mouse=m, mouse_mode=mm)

        # FPS
        self._fps_dt += dt; self._fps_fr += 1
        if self._fps_dt >= 0.5:
            self._fps_v  = self._fps_fr / self._fps_dt
            self._fps_dt = 0.0; self._fps_fr = 0
        mode_txt = ["fare:kapali", "fare:cek", "fare:it"][self.mouse_mode]
        extra = ""
        if self.audio_model is not None:
            extra += (f"  MODEL:{self.audio_model.infer_ms:.0f}ms" if self.use_model
                      else "  MODEL:kapali")
        if self.diff is not None:
            if not self.diff_on:
                extra += "  DIF:kapali"
            elif self.diff.failed:
                extra += "  DIF:hata"
            elif not self.diff.ready:
                extra += "  DIF:yukleniyor"
            else:
                extra += f"  DIF:{self.diff.fps:.0f}fps gurultu:{self.diff.noise}"
                if self.latency.last is not None and self.cfg["latency_log"]:
                    extra += f"  söz→ekran:{self.latency.last[0]:.2f}sn"
        if self.tap.recorder is not None:
            extra += f"  ●ÖZELLİK({self.tap.recorder.saved_tracks})"
        rec = self.recorder
        if rec.active:
            m, sec = divmod(int(rec.elapsed), 60)
            extra += f"  ● KAYIT {m:02d}:{sec:02d}"
        elif rec.status and self.t - self._rec_msg_t < 6:
            extra += f"  [{rec.status}]"
        if self._beat_locked():
            dots = "".join("●" if i == self.beat_count % 4 else "○" for i in range(4))
            beat_txt = f"BPM:{bpm:.0f} {dots}"
        else:
            beat_txt = f"BPM:{bpm:.0f}?"
        self.fps_lbl.text = (f"FPS:{self._fps_v:.0f}  RMS:{rms:.3f}  {beat_txt}  "
                             f"bölüm:{self.section}  x{self.sensitivity:.1f}  {mode_txt}{extra}")
        self.fps_lbl.y = self.height - 24

    def on_draw(self):
        gl = pyglet.gl
        self.clear()

        if self.neural_on and self._draw_diffusion():
            pass                          # 1) Difüzyon sahnesi (CPPN'den hayal edilen)
        elif self.neural_on:
            # 1) Neural arka plan (kendi feedback izlerini kendisi yönetir)
            self.neural.draw()
        else:
            # 1) Fade overlay — normal blend (trail efekti)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            self.fade.width  = self.width
            self.fade.height = self.height
            self.fade.draw()

        # 2) Tüm görsel katmanlar — ADDITIVE blend + zoom pulse
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        z = self._zoom
        if not self.show_presets:
            pass
        elif abs(z - 1.0) > 0.001:
            from pyglet.math import Mat4, Vec3
            cx, cy = self.width / 2, self.height / 2
            # Merkez etrafında ölçekle: T(c) · S(z) · T(-c)
            self.view = (Mat4.from_translation(Vec3(cx, cy, 0)) @
                         Mat4.from_scale(Vec3(z, z, 1)) @
                         Mat4.from_translation(Vec3(-cx, -cy, 0)))
            self.batch.draw()
            self.view = Mat4()   # sıfırla (HUD/flash etkilenmesin)
        else:
            self.batch.draw()

        # 3) Beat flash — additive
        if self._flash_intensity > 0.01:
            self.flash.width  = self.width
            self.flash.height = self.height
            self.flash.draw()

        # Kayıt: HUD'suz kare (ayarlardan HUD dahil edilebilir)
        if self.recorder.active and not self.cfg["record_hud"]:
            self._capture()

        # 4) HUD — normal blend
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        if self.show_hud:
            self.batch2.draw()

        # 5) Yardım menüsü
        if self.show_help:
            self.help_bg.x  = self.width // 2 - 390
            self.help_bg.y  = self.height // 2 - 190
            for i, lbl in enumerate(self.help_lbls):
                lbl.x = self.width // 2 - 370 + i * 390
                lbl.y = self.height // 2 + 165
            self.help_batch.draw()

        # 6) Ayarlar menüsü
        self.settings.draw(self.width, self.height)

        if self.recorder.active and self.cfg["record_hud"]:
            self._capture()

    def _capture(self):
        w, h = self.get_framebuffer_size()
        if (w - w % 2, h - h % 2) != tuple(self.recorder.size):
            print("[Kayıt] Pencere boyutu değişti — kayıt durduruldu")
            self._toggle_recording()
            return
        self.recorder.capture(pyglet.gl)

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self.fps_lbl.y = h - 24

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, mod):
        self.mouse_pos = (x, y)

    def on_mouse_leave(self, x, y):
        self.mouse_pos = None

    def on_key_press(self, sym, mod):
        key = pyglet.window.key
        if self.settings.on_key(sym):
            return pyglet.event.EVENT_HANDLED
        if sym == key.S:
            self.settings.toggle()
        elif sym == key.F9:
            self._toggle_recording()
        elif sym in (key.ESCAPE, key.Q):
            self.close()
        elif sym == key._1: self.layers_on["ribbon"] = not self.layers_on["ribbon"]
        elif sym == key._2: self.layers_on["spiral"] = not self.layers_on["spiral"]
        elif sym == key._3: self.layers_on["liss"]   = not self.layers_on["liss"]
        elif sym == key._4: self.layers_on["const"]  = not self.layers_on["const"]
        elif sym == key._5: self.layers_on["shock"]  = not self.layers_on["shock"]
        elif sym == key._6: self.layers_on["parts"]  = not self.layers_on["parts"]
        elif sym == key._7: self.layers_on["helix"]  = not self.layers_on["helix"]
        elif sym == key._8: self.layers_on["tunnel"] = not self.layers_on["tunnel"]
        elif sym == key._9: self.layers_on["rays"]   = not self.layers_on["rays"]
        elif sym == key._0: self.layers_on["grid"]   = not self.layers_on["grid"]
        elif sym == key.O:  self.layers_on["orbits"] = not self.layers_on["orbits"]
        elif sym == key.M:
            self.mouse_mode = (self.mouse_mode + 1) % 3
        elif sym in (key.PLUS, key.EQUAL, key.NUM_ADD):
            self.sensitivity = round(min(3.0, self.sensitivity + 0.1), 2)
            self.cfg["sensitivity"] = self.sensitivity
        elif sym in (key.MINUS, key.NUM_SUBTRACT):
            self.sensitivity = round(max(0.2, self.sensitivity - 0.1), 2)
            self.cfg["sensitivity"] = self.sensitivity
        elif sym == key.F:
            self.set_fullscreen(not self.fullscreen)
        elif sym == key.SPACE:
            self.palette.randomize()
        elif sym == key.TAB:
            self.show_hud = not self.show_hud
            self.cfg["show_hud"] = self.show_hud
        elif sym == key.H:
            self.show_help = not self.show_help
        elif sym == key.A:
            self.auto_scenes = not self.auto_scenes
            self._apply_scene(self.scene_idx)
        elif sym == key.N:
            self.scene_timer = 0.0
            self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
            self._apply_scene(self.scene_idx)
        elif sym == key.R:
            self.layer_rain.burst(28)  # manuel şekil yağmuru
        elif sym == key.G and self.neural:
            self.neural_on = not self.neural_on
            if not self.neural_on:
                self.show_presets = True
                self._apply_scene(self.scene_idx)
        elif sym == key.B:
            self.show_presets = not self.show_presets
        elif sym == key.V and self.neural_on:
            self.neural.new_style()
        elif sym == key.K and self.neural_on:
            self.neural.save_current()
        elif sym == key.L and self.neural_on:
            self.neural.locked = not self.neural.locked
        elif sym == key.X and self.audio_model is not None:
            self.use_model = not self.use_model
        elif sym == key.C:
            self.tap.toggle_recording()
        elif sym == key.D and self.diff is not None:
            self.diff_on = not self.diff_on
        elif sym == key.BRACKETLEFT and self.diff is not None:
            self.diff.noise = max(300, self.diff.noise - 50)
        elif sym == key.BRACKETRIGHT and self.diff is not None:
            self.diff.noise = min(950, self.diff.noise + 50)

    def _draw_diffusion(self):
        """Son difüzyon karesini ekranı kaplayacak şekilde çizer; yeni kare öncekinin
        üzerine çapraz geçişle gelir (difüzyon ~5-15 FPS, ekran 60 FPS → takılma olmaz).
        Kare yoksa False."""
        n = self.neural
        if not (self.diff_on and self.diff is not None and self.diff.ready and n.dreaming):
            return False
        fid, frame, lyric_id = self.diff.get_frame_info()
        if frame is None:
            return False
        if fid != self._diff_seen:
            h, w = frame.shape[:2]
            img = pyglet.image.ImageData(w, h, "RGB", frame.tobytes(), pitch=-w * 3)
            spr = pyglet.sprite.Sprite(img.get_texture())
            if len(self._diff_sprites) == 2:
                self._diff_sprites[0].delete()
            self._diff_sprites = (self._diff_sprites + [spr])[-2:]
            self._diff_seen = fid
            self._diff_t = self.t
            # Geçiş ≈ iki kare arası süre: sürekli akış, ama yavaş üretimde de bekletmez
            self._diff_fade = min(0.5, max(0.08, 1.0 / max(self.diff.fps, 1e-3)))
            if lyric_id is not None:
                # Söz satırının ilk karesi: neredeyse anında (≤ lyric_fade) — gecikmeye geçiş eklenir
                self._diff_fade = max(1e-3, self.cfg["lyric_fade"])
                self.latency.finish(lyric_id, extra=self.cfg["lyric_fade"])
        a = min(1.0, (self.t - self._diff_t) / self._diff_fade)
        if a >= 1.0 and len(self._diff_sprites) == 2:
            self._diff_sprites[0].delete()
            self._diff_sprites = self._diff_sprites[1:]
        two = len(self._diff_sprites) == 2
        for i, spr in enumerate(self._diff_sprites):
            tex = spr.image
            sc = max(self.width / tex.width, self.height / tex.height)   # ekranı kapla (kenarlar kırpılır)
            spr.update(x=(self.width - tex.width * sc) / 2, y=(self.height - tex.height * sc) / 2, scale=sc)
            spr.opacity = int(255 * a) if (two and i == 1) else 255
        for spr in self._diff_sprites:
            spr.draw()
        return True

    def _clear_layer(self, layer):
        """Sönen katmanın tüm şekillerini görünmez yap (kalıntı kalmasın)."""
        def hide(o):
            try: o.color = (0,0,0,0)
            except: pass
        try:
            for attr in ("lines", "rays", "rungs", "a", "b", "hlines", "vlines", "orbits", "planets"):
                if hasattr(layer, attr):
                    for o in getattr(layer, attr): hide(o)
            if hasattr(layer, "sun"): hide(layer.sun)
            if hasattr(layer, "dots"):
                for d in layer.dots:
                    hide(d["obj"] if isinstance(d, dict) else d)
            if hasattr(layer, "rings"):
                for r in layer.rings:
                    hide(r["obj"] if isinstance(r, dict) else r)
            if hasattr(layer, "ribbons"):
                for rb in layer.ribbons:
                    for l in rb["lines"]: hide(l)
            if hasattr(layer, "_pool"):
                for p in layer._pool:
                    hide(p["obj"]); p["alive"] = False
        except Exception:
            pass

    # ── Ritim yardımcıları ──
    def _beat_locked(self):
        return self.cfg["beat_sync"] and self.beat_conf >= 0.4

    def _set_director_scene(self, scene):
        """Direktörün seçtiği sahneyi uygula (otomatik rotasyonu devralır)."""
        self.ai_driving = True
        self.auto_scenes = False
        layers = self.ai_scene_map[scene]
        for key in self.layers_on:
            self.layers_on[key] = (key in layers)
        self._cur_scene_name = scene
        if not (self.neural_on and self.neural.dreaming):
            self.scene_lbl.text = f"Sahne: {scene} [AUTO]"

    def _on_section(self, label):
        """Şarkıda yeni bölüm: drop → özel an, diğerleri → yeni neural stil + sahne."""
        print(f"[Bölüm] {label}")
        if self.diff_on and self.diff is not None and self.diff.ready:
            self.diff.request_scene()
        if label == "intro":
            return
        if label == "drop":
            if self._drop_cooldown <= 0:
                self._trigger_drop(max(0.6, float(np.mean(self.vis_bands))))
                self._drop_cooldown = 4.0
            return
        if self.neural_on and not self.neural.locked and not self.neural.dreaming:
            self.neural.request_style(fast=(label == "breakdown"))
        if self.auto_scenes:
            self.scene_timer = 0.0
            self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
            self._apply_scene(self.scene_idx)

    # ── Ayarlar / kayıt ──
    def _on_setting(self, key, value):
        if key == "sensitivity":
            self.sensitivity = value
        elif key == "palette_drift":
            self.palette.drift_speed = value
        elif key == "scene_duration":
            self.scene_dur = value
        elif key == "show_hud":
            self.show_hud = value
        elif key == "lyrics" and self.lyrics is not None:
            self.lyrics.enabled = value
        elif key == "diff_valence_bias" and self.diff is not None:
            self.diff.valence_bias = value
            self.diff.request_scene()
        elif key == "record_fps" and not self.recorder.active:
            self.recorder.fps = value
        elif key == "record_crf" and not self.recorder.active:
            self.recorder.crf = value

    def _toggle_recording(self):
        rec = self.recorder
        if rec.active:
            self.audio.recorder = None
            rec.stop()
        else:
            rec.fps, rec.crf = self.cfg["record_fps"], self.cfg["record_crf"]
            w, h = self.get_framebuffer_size()
            if rec.start(w, h, self.audio.sample_rate, self.audio.channels):
                self.audio.recorder = rec
        self._rec_msg_t = self.t

    def _apply_scene(self, idx):
        """Sahnenin katmanlarını aç, diğerlerini kapat."""
        scene = self.scenes[idx]
        on = set(scene["on"])
        for key in self.layers_on:
            self.layers_on[key] = (key in on)
        if hasattr(self, "scene_lbl"):
            mode = "AUTO" if self.auto_scenes else "MANUEL"
            self.scene_lbl.text = f"Sahne: {scene['name']} [{mode}]"

    def _trigger_drop(self, energy):
        """Müzik patladığında özel an: şekil yağmuru + güçlü flash + palet."""
        self.layer_rain.burst(n=int(18 + energy * 20))
        self._flash_intensity = min(1.2, self._flash_intensity + 0.9)
        if self.neural_on:
            self.neural.on_drop()   # drop anında hızlı stil geçişi
        # Birden fazla şok dalgası
        if "shock" in self.layers_on:
            cx, cy = self.width//2, self.height//2
            for _ in range(3):
                self.layer_shock._spawn(self.palette, energy,
                                        cx + random.uniform(-120,120),
                                        cy + random.uniform(-120,120))
        print(f"[Event] DROP! enerji={energy:.2f}")

    def on_close(self):
        print("[App] Kapatiliyor...")
        if self.recorder.active:
            self._toggle_recording()
        if not self.fullscreen:
            self.cfg["width"], self.cfg["height"] = self.width, self.height
        config.save(self.cfg)
        self.ai.stop()
        self.audio.stop()
        self.tap.stop()          # açık kayıt varsa son parçayı kaydeder
        if self.audio_model is not None:
            self.audio_model.stop()
        if self.diff is not None:
            self.diff.stop()
        if self.lyrics is not None:
            self.lyrics.stop()
        self.recorder.wait()
        pyglet.app.exit()


def main():
    print("=" * 52)
    print("  AI Music Visualizer")
    print("  Beat/bölüm analizi + neural arka plan + pyglet 2.x")
    print("=" * 52)
    VisualizerWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()