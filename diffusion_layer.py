"""
Difüzyon katmanı — rüya modundaki CPPN görüntüsünden nesneli sahneler hayal eder.

Zincir: ses → ses modeli → duygu başlığı → görsel üretici (CPPN) → SD-Turbo img2img
  • Başlangıç resmi : o anki CPPN dünyası (torch kopyasıyla çizilir: renk, kompozisyon)
  • Metin temsili   : dört duygu çapasının (hüzünlü / gergin / neşeli / huzurlu)
                      valence–arousal'a göre sürekli karışımı
                      + RUH HALİ: CLAP 30 ruh haliyle karşılaştırır → üslup (destansı, romantik…)
                      + İÇERİK: CLAP 169 kavramla karşılaştırır → konu (savaşçı, yağmur, kalabalık…)
                      SAHNE SAHNE: müzik belirgin değişince ya da sahne ~8 sn sürünce yeni
                      sahne — müziğe en uyan 12 kavramdan 2'si, en uyan 3 ruh halinden 1'i
                      CLAP olasılıklarıyla ÇEKİLİR (hep aynı en-iyi-2 değil → çeşitlilik) ve
                      gerçek bir cümle olarak kodlanır ("a hero raising a sword, a mighty
                      warrior, cinematic dramatic artwork"). Her sahnede yeni tohum + kadraj.
  • CPPN girişi    : kontur çizgileri kapalı + bulanık — sadece renk ve kaba kompozisyon
                      verir (ince şeritler difüzyonda zebra desenine dönüşüyordu)
  • Hafıza         : her kare = önceki hayal × MEMORY + CPPN × (1 − MEMORY) başlangıcından
                      üretilir; model sahneyi sürdürür ve dönüştürür, her karede baştan kurmaz
  • Gürültü         : sabit tohum — girdi az değişince sahne de az değişir (akıcılık)
  • Gürültü seviyesi: düşük = CPPN'e sadık, yüksek = model daha özgür hayal eder

Söz önceliği : yeni söz satırı MIN_SCENE'i beklemez (LYRIC_MIN_SCENE yeter); Whisper satırı
               yayınlayınca SD thread'i bir threading.Event ile hemen uyanır. Satırın ilk
               karesi `frame_lyric` ile işaretlenir → ekranda kısa geçişle çizilir ve
               gecikme ölçülür (latency.py).
Hız          : channels_last + cudnn.benchmark + inference_mode (diff_fast), isteğe bağlı
               torch.compile (diff_compile, varsayılan kapalı). Yüklemede ısınma kareleri.

Ayrı thread'de çalışır; pencere / ses akışı beklemez. torch + diffusers yoksa
(ör. sistem Python'u) katman sessizce devre dışı kalır.
Modeller: stabilityai/sd-turbo (ticari olmayan lisans), madebyollin/taesd (hızlı çözücü),
          laion/larger_clap_music_and_speech (Apache-2.0, kavram seçimi).
"""

import collections
import importlib.util
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
from config import WORK_DIR as WORK  # noqa: E402
# Modeller her zaman projenin kendi önbelleğinde (sistemdeki HF_HOME başka bir projeyi gösterebilir).
# torch / transformers / diffusers import edilmeden önce ayarlanmalı.
os.environ["HF_HOME"] = os.path.join(WORK, "hf")

# CLAP metin uzayında duygu eksenleri: kavram / ruh hali / müzik bu eksenlere izdüşürülür
AXIS_TEXT = {
    "pos": ["happy, joyful, cheerful music", "uplifting, euphoric, celebratory music", "bright, positive, feel-good music"],
    "neg": ["sad, dark, melancholic music", "gloomy, mournful, depressing music", "bleak, painful, hopeless music"],
    "hi": ["energetic, intense, powerful music", "loud, fast, exciting music"],
    "lo": ["calm, quiet, relaxing music", "slow, soft, gentle music"],
}
# Prompt'a eklenen duygu ifadesi (valence, arousal işaretine göre)
EMO_PHRASE = {
    (1, 1): "joyful euphoric atmosphere, vibrant saturated colors, bright sunlight, celebration",
    (1, -1): "serene happy atmosphere, soft warm light, gentle pastel colors",
    (-1, 1): "intense dark atmosphere, harsh contrast, stormy",
    (-1, -1): "melancholic gloomy atmosphere, muted colors, dim light",
}

ANCHORS = {
    (-1, -1): ["a sad, melancholic artwork", "sorrow, loneliness and grief", "a gloomy, mournful, heartbroken scene"],
    (-1, +1): ["an artwork full of anger and rage", "chaos, violence and danger", "a furious, menacing, dark and intense scene"],
    (+1, +1): ["a joyful, happy artwork", "euphoria, cheer and celebration", "a bright, exciting, festive scene"],
    (+1, -1): ["a calm, peaceful artwork", "serenity, gentleness and rest", "a tranquil, dreamy, soothing scene"],
}


def available():
    try:
        if importlib.util.find_spec("diffusers") is None:
            return False
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def is_new_line(lyric, lid, seen):
    """Yeni söz satırı mı? Geri çekilen (uydurma) satırdan sonra görünür hale gelen
    ESKİ satır yeni sayılmaz: id'ler yalnızca artar."""
    return lyric is not None and lid is not None and lid > seen


def scene_due(age, changed, want_scene, new_line, min_scene, scene_sec,
              lyric_priority=True, lyric_min_scene=0.3):
    """Yeni sahne kurulmalı mı? (torch'suz, birim testli)
    changed       : müzik belirgin değişti (yaştan bağımsız)
    want_scene    : ana thread ölçü başında istedi → en az min_scene
    new_line      : yeni söz satırı → söz önceliği açıksa sadece lyric_min_scene
                    (art arda iki satırda titreme olmasın diye küçük bir taban)
    age≥scene_sec : sahne zaten uzun sürdü"""
    if changed or age >= scene_sec:
        return True
    if want_scene and age >= min_scene:
        return True
    if new_line:
        return age >= (lyric_min_scene if lyric_priority else min_scene)
    return False


class DiffusionLayer:
    CONCEPT_EVERY = 0.4        # sn — CLAP thread'inin analiz aralığı
    CLAP_SEC = 5.0             # CLAP'ın baktığı son ses (sn): kısa = müziğe daha çabuk tepki
    N_CONCEPTS = 2             # cümleye giren kavram sayısı
    POOL, MOOD_POOL = 8, 3     # çekilişin yapıldığı en uyan kavram / ruh hali sayısı (dar = şarkıya sadık)
    SCENE_SEC = 3.0            # bir sahnenin en uzun süresi (beat kilidi yokken)
    MIN_SCENE = 1.2            # ölçü başı isteğinde bile en kısa sahne
    LYRIC_MIN_SCENE = 0.3      # söz önceliği: yeni söz satırı için en kısa sahne
    CHANGE_DIST = 0.12         # müzik değişimi eşiği (CLAP temsilinde kosinüs uzaklığı)
    RECENT = 4                 # son kaç kavram tekrar seçilmez (ruh hali: son 1)
    GENRE_EMA = 0.15           # tür olasılığının yumuşatması (her ~0,4 sn) — tür sahne sahne zıplamasın
    PULSE_NOISE = 120          # beat anındaki karede gürültü artışı (sahne içinde vuruşa tepki)
    LYRIC_W = 1.5              # söz satırına anlamca yakın kavramların seçimdeki ek ağırlığı
    LYRIC_WORDS = 12           # prompt'a giren söz satırının en fazla kelime sayısı
    FRESH_NOISE = 950          # yeni sahnenin ilk karesi: önceki hayalden bağımsız, yüksek gürültü
    EMO_WEIGHT_C = 1.2         # kavram seçiminde duygu uyumunun ağırlığı (özgüllük z-skoruna eklenir)
    EMO_WEIGHT_M = 1.8         # ruh hali seçiminde duygu uyumunun ağırlığı
    EMO_PHRASE_MIN = 0.25      # |valence| ya da |arousal| bunu aşarsa prompt'a duygu ifadesi eklenir
    FRAMINGS = ["", "close-up, ", "wide shot, ", "aerial view, ", "silhouette, ", "panoramic landscape, ",
                "low angle, ", "detailed scene, "]
    MEMORY = 0.55              # başlangıç resminde önceki hayalin payı

    def __init__(self, tap=None, size=(640, 384), noise=800, seed=42, tracker=None,
                 lyric_priority=True, fast=True, compile=False):
        self.tap = tap
        self.tracker = tracker         # latency.LatencyTracker (ya da None)
        self.lyric_priority = lyric_priority
        self.fast = fast               # channels_last + cudnn.benchmark + inference_mode
        self.compile = compile         # torch.compile (UNet) — Windows'ta triton gerekir, riskli
        self._wake = threading.Event() # yeni söz satırı → SD thread'i beklemeden uyanır
        self._lyric_pending = None     # sahnesi kuruldu, ilk karesi henüz üretilmedi (line_id)
        self.frame_lyric = None        # son karenin "söz satırının ilk karesi" etiketi (line_id)
        self._unet_eager = None
        self.concepts = []             # [(kavram, ağırlık)] — HUD için
        self.mood_name = None          # baskın ruh hali — HUD için
        self.va_eff = None             # sahnede kullanılan (valence, arousal) — model + CLAP
        self.valence_bias = 0.0        # kullanıcı ayarı: okumaya eklenir
        self._prev = None              # önceki hayal (hafıza)
        self._scene_t = 0.0            # sahnenin başladığı an
        self._scene_a = None           # sahnenin başındaki CLAP temsili
        self._rng = np.random.default_rng()
        self._recent_c = collections.deque(maxlen=self.RECENT)
        self._recent_m = collections.deque(maxlen=1)
        self.section = "verse"         # ana thread yazar (structure.py etiketi)
        self.genre_name = None         # HUD için
        self._genre_p = None           # yumuşatılmış tür olasılıkları
        self._pulse = False            # ana thread: beat anı
        self.lyrics = None             # LyricsListener (varsa)
        self.lyric_used = None         # sahnede kullanılan söz satırı — HUD için
        self._lyric_seen = 0
        self._lyric_emb = (None, None) # (metin, CLAP metin temsili) önbelleği
        self._want_scene = False       # ana thread: ölçü başında yeni sahne iste
        self._fresh = True             # yeni sahnenin ilk karesi mi
        self._flip = (False, False)    # sahneye özgü kadraj dönüşümü
        self._shift = (0.0, 0.0)
        self.w, self.h = size
        self.noise = int(noise)
        self.seed = seed
        self.ready = False
        self.failed = None
        self.running = False
        self.fps = 0.0
        self.frame = None              # (h, w, 3) uint8 — son üretilen kare
        self.frame_id = 0
        self._inp = None               # (style, latent, va) — son gönderilen girdi
        self._inp_id = 0
        self._lock = threading.Lock()
        self._clap = None              # CLAP thread'inin son çıktısı: (a, z, zm, tür_indeksi)
        self._clap_lock = threading.Lock()

    # ── Ana thread ──
    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def submit(self, style, latent, va):
        with self._lock:
            self._inp = ({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in style.items()},
                         np.asarray(latent, np.float32).copy(), np.asarray(va, np.float32).copy())
            self._inp_id += 1

    def pulse(self):
        """Ana thread (her beat): bir sonraki kare daha serbest üretilsin → vuruşta görünür değişim."""
        self._pulse = True

    def request_scene(self):
        """Ana thread (ölçü başı): yeni, öncekine benzemeyen bir sahne iste."""
        self._want_scene = True

    def get_frame(self):
        with self._lock:
            return self.frame_id, self.frame

    def get_frame_info(self):
        """(kare no, kare, söz satırı id | None) — id yalnızca o satırın İLK karesinde dolu."""
        with self._lock:
            return self.frame_id, self.frame, self.frame_lyric

    def wake(self, *_):
        """Söz dinleyicinin geri çağrısı: yeni satır geldi."""
        self._wake.set()

    def _nograd(self):
        return self.torch.inference_mode() if self.fast else self.torch.no_grad()

    # ── Difüzyon thread'i ──
    def _load(self):
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
        sys.path.insert(0, os.path.join(ROOT, "train"))
        import visual as V
        self.torch, self.V = torch, V
        self.dev = torch.device("cuda")
        pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                          variant="fp16")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
        pipe = pipe.to(self.dev)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self._speedups()
        with torch.no_grad():
            self.anc = {q: pipe.encode_prompt(t, self.dev, 1, False)[0].mean(0, keepdim=True)
                        for q, t in ANCHORS.items()}
        self._load_clap()
        self._warmup()

    def _speedups(self):
        """Kaliteyi değiştirmeyen hızlandırmalar. Her biri ayrı ayrı denenir; hata → eski yol + log."""
        torch, pipe = self.torch, self.pipe
        if self.fast:
            try:
                torch.backends.cudnn.benchmark = True          # sabit boyut (640×384): en hızlı conv algoritması
                pipe.unet.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)
                print("[Difüzyon] Hızlandırma: channels_last + cudnn.benchmark + inference_mode")
            except Exception as e:
                self.fast = False
                print(f"[Difüzyon] Hızlandırma kapalı ({e})")
        if self.compile:
            try:
                self._unet_eager = pipe.unet
                pipe.unet = torch.compile(pipe.unet, dynamic=False)   # CUDA graph yok: çoklu stream ile güvenli
                print("[Difüzyon] torch.compile açık (ilk kareler derleme yüzünden yavaş)")
            except Exception as e:
                self._uncompile(e)

    def _uncompile(self, err):
        if self._unet_eager is not None:
            self.pipe.unet = self._unet_eager
            self._unet_eager = None
        self.compile = False
        print(f"[Difüzyon] torch.compile kapatıldı, eski yola dönüldü ({err})")

    def _warmup(self, n=2):
        """İlk gerçek kare (çoğu zaman bir söz sahnesi) cudnn / derleme ısınmasını beklemesin."""
        torch = self.torch
        try:
            init = torch.full((1, 3, self.h, self.w), 0.5, device=self.dev)
            pe = next(iter(self.anc.values()))
            for _ in range(n):
                with self._nograd():
                    self.pipe(prompt_embeds=pe, image=init, strength=1.0, timesteps=[self.FRESH_NOISE],
                              guidance_scale=0.0, output_type="pt")
            torch.cuda.synchronize()
        except Exception as e:
            if self.compile:
                self._uncompile(e)
            else:
                print(f"[Difüzyon] Isınma atlandı ({e})")

    def _load_clap(self):
        """CLAP + kavram istatistikleri. Yoksa sadece duygu kullanılır."""
        self.clap = None
        self._prompt = None            # o anki cümle
        self._prompt_pe = None         # kodlanmış hali
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
            sys.path.insert(0, ROOT)
            from concepts import CONCEPTS, MOODS, MOOD_SD_TEXT, GENRES, GENRE_CLAP_TEXT, GENRE_SD_TEXT, SECTION_SD_TEXT
            st = np.load(os.path.join(ROOT, "models", "clap_concepts.npz"))
            name = str(st["model"])
            self.clap = ClapModel.from_pretrained(name).to(self.dev).eval()
            self.clap_proc = ClapProcessor.from_pretrained(name)
            self.c_names = CONCEPTS
            self.c_text = torch.tensor(st["text"], device=self.dev)
            self.c_mean = torch.tensor(st["mean"], device=self.dev)
            self.c_std = torch.tensor(st["std"], device=self.dev)
            self.c_phrase, self.m_phrase = CONCEPTS, MOOD_SD_TEXT
            self.m_names = [m[0] for m in MOODS]
            self.m_text = torch.tensor(st["m_text"], device=self.dev)
            self.m_mean = torch.tensor(st["m_mean"], device=self.dev)
            self.m_std = torch.tensor(st["m_std"], device=self.dev)
            self._load_axes()
            with torch.no_grad():
                gi = self.clap_proc(text=GENRE_CLAP_TEXT, return_tensors="pt", padding=True).to(self.dev)
                gt = self.clap.get_text_features(**gi)
                self.g_text = torch.nn.functional.normalize(getattr(gt, "pooler_output", gt), dim=-1)
            self.g_names, self.g_phrase, self.sec_phrase = [g[0] for g in GENRES], GENRE_SD_TEXT, SECTION_SD_TEXT
            print(f"[Difüzyon] Kavram seçimi açık: {len(CONCEPTS)} kavram + {len(MOODS)} ruh hali (CLAP)")
        except Exception as e:
            self.clap = None
            print(f"[Difüzyon] Kavram seçimi kapalı ({e}) — sadece duygu")

    def _load_axes(self):
        """Duygu eksenleri (CLAP metin uzayı) ve her kavram / ruh halinin bu eksenlerdeki yeri."""
        torch = self.torch

        def emb(texts):
            inp = self.clap_proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
            t = self.clap.get_text_features(**inp)
            t = torch.nn.functional.normalize(getattr(t, "pooler_output", t), dim=-1)
            return torch.nn.functional.normalize(t.mean(0), dim=-1)

        with torch.no_grad():
            e = {k: emb(v) for k, v in AXIS_TEXT.items()}
        self.ax_v = torch.nn.functional.normalize(e["pos"] - e["neg"], dim=-1)
        self.ax_a = torch.nn.functional.normalize(e["hi"] - e["lo"], dim=-1)

        def coords(T):                      # kavramlar arasında standartlaştırılmış eksen koordinatı
            out = []
            for ax in (self.ax_v, self.ax_a):
                c = T @ ax
                out.append((c - c.mean()) / (c.std() + 1e-6))
            return out

        self.c_v, self.c_a = coords(self.c_text)
        self.m_v, self.m_a = coords(self.m_text)

    def _clap_emotion(self, zm):
        """Müziğin ruh hali profili (35 ruh haline özgüllük) ile ruh hallerinin duygu
        koordinatları arasındaki korelasyon → CLAP'ın (valence, arousal) okuması, −1…1."""
        def corr(x, y):
            x, y = x - x.mean(), y - y.mean()
            return float((x * y).sum() / (x.norm() * y.norm() + 1e-6))
        return (float(np.clip(2.0 * corr(zm, self.m_v), -1, 1)),
                float(np.clip(2.0 * corr(zm, self.m_a), -1, 1)))

    def _clap_step(self):
        """CLAP thread'i: son ~5 sn müzik → temsil, kavram / ruh hali özgüllüğü, tür.
        Sonuç önbelleğe yazılır; sahne kurma (SD thread'i) beklemeden en sonuncuyu kullanır."""
        if self.clap is None or self.tap is None:
            return
        torch = self.torch
        x, sr = self.tap.latest_raw()
        if sr is None or len(x) < sr:
            return
        x = x[-int(self.CLAP_SEC * sr):]
        if sr != 48000:
            from scipy.signal import resample_poly
            g = np.gcd(int(sr), 48000)
            x = resample_poly(x, 48000 // g, int(sr) // g).astype(np.float32)
        with torch.no_grad():
            inp = self.clap_proc(audio=[x], sampling_rate=48000, return_tensors="pt").to(self.dev)
            a = self.clap.get_audio_features(**inp)
            a = torch.nn.functional.normalize(getattr(a, "pooler_output", a), dim=-1)[0]
            z = (a @ self.c_text.T - self.c_mean) / self.c_std          # o müziğe özgüllük
            zm = (a @ self.m_text.T - self.m_mean) / self.m_std
        # Tür: sıfır-atış CLAP (logit ölçeğiyle softmax), zaman içinde yumuşatılmış
        with torch.no_grad():
            scale = float(self.clap.logit_scale_a.exp()) if hasattr(self.clap, "logit_scale_a") else 30.0
            gp = torch.softmax(scale * (a @ self.g_text.T), 0).cpu().numpy()
        self._genre_p = gp if self._genre_p is None else (1 - self.GENRE_EMA) * self._genre_p + self.GENRE_EMA * gp
        gi = int(np.argmax(self._genre_p))
        self.genre_name = self.g_names[gi]
        torch.cuda.current_stream().synchronize()                       # diğer stream'den güvenle okunsun
        with self._clap_lock:
            self._clap = (a, z, zm, gi)

    def _update_concepts(self):
        """SD thread'i, her kare öncesi: sahne değişmeli mi? Değişecekse yeni cümle kur."""
        if self.clap is None:
            return
        with self._clap_lock:
            cs = self._clap
        if cs is None:
            return
        a, z, zm, gi = cs
        torch = self.torch
        now = time.time()
        age = now - self._scene_t
        changed = self._scene_a is None or float(1 - (a @ self._scene_a)) > self.CHANGE_DIST
        lyric, lid = None, None
        if self.lyrics is not None and self.lyrics.enabled:
            if hasattr(self.lyrics, "current_with_id"):
                lyric, lid = self.lyrics.current_with_id()
            else:
                lyric, lid = self.lyrics.current(), self.lyrics.line_id
        new_line = is_new_line(lyric, lid, self._lyric_seen)
        if not scene_due(age, changed, self._want_scene, new_line, self.MIN_SCENE, self.SCENE_SEC,
                         self.lyric_priority, self.LYRIC_MIN_SCENE):
            return                                                       # sahne sürüyor
        self._want_scene = False
        rng = self._rng

        # Duygu: duygu başlığı (DEAM) + CLAP'ın kendi okuması — ikisinin ortalaması
        with self._lock:
            va_m = None if self._inp is None else self._inp[2]
        v_c, a_c = self._clap_emotion(zm)
        if va_m is None:
            ev, ea = v_c, a_c
        else:
            ev, ea = 0.5 * float(va_m[0]) + 0.5 * v_c, 0.5 * float(va_m[1]) + 0.5 * a_c
        ev = float(np.clip(ev + self.valence_bias, -1.0, 1.0))
        self.va_eff = (ev, ea)
        # Seçim skoru = müziğe özgüllük + duygu uyumu (pozitif şarkı → pozitif kavramlar öne çıkar)
        z = z + self.EMO_WEIGHT_C * (ev * self.c_v + ea * self.c_a)
        zm_sel = zm + self.EMO_WEIGHT_M * (ev * self.m_v + ea * self.m_a)
        # Söz: satıra anlamca yakın kavramlar öne çıkar (CLAP metin uzayında benzerlik)
        if lyric is not None:
            self._lyric_seen = lid
            if self._lyric_emb[0] != lyric:
                with self._nograd():
                    li = self.clap_proc(text=[lyric], return_tensors="pt", padding=True).to(self.dev)
                    lt = self.clap.get_text_features(**li)
                    lt = torch.nn.functional.normalize(getattr(lt, "pooler_output", lt), dim=-1)[0]
                self._lyric_emb = (lyric, lt)
            ls = self.c_text @ self._lyric_emb[1]
            z = z + self.LYRIC_W * (ls - ls.mean()) / (ls.std() + 1e-6)

        def draw(scores, pool, k, temp, recent=()):
            scores = scores.clone()
            if recent:                                                   # yakın zamanda kullanılanlar dışarıda
                scores[list(recent)] = -1e9
            top = torch.topk(scores, pool)
            p = torch.softmax(top.values / temp, 0).cpu().numpy().astype(np.float64)
            pick = rng.choice(pool, size=k, replace=False, p=p / p.sum())
            return [int(top.indices[j]) for j in pick], [float(top.values[j]) for j in pick]

        ci, cv = draw(z, self.POOL, self.N_CONCEPTS, 0.4, self._recent_c)
        (mi,), _ = draw(zm_sel, self.MOOD_POOL, 1, 0.3, self._recent_m)
        self._recent_c.extend(ci)
        self._recent_m.append(mi)
        framing = self.FRAMINGS[rng.integers(len(self.FRAMINGS))]
        # Sıra = önem (CLIP 77 token'dan sonrasını keser): konu, tür üslubu, duygu, ruh hali, bölüm anı
        parts = [self.c_phrase[i] for i in ci] + [self.g_phrase[gi]]
        self.lyric_used = None
        if lyric is not None:                                            # söz en başta: en güçlü etki
            short = " ".join(lyric.split()[:self.LYRIC_WORDS])
            parts.insert(0, short)
            self.lyric_used = short
        if max(abs(ev), abs(ea)) >= self.EMO_PHRASE_MIN:
            parts.append(EMO_PHRASE[(1 if ev >= 0 else -1, 1 if ea >= 0 else -1)])
        parts.append(self.m_phrase[mi])
        sec = self.sec_phrase.get(self.section, "")
        if sec:
            parts.append(sec)
        prompt = framing + ", ".join(parts)
        with self._nograd():
            self._prompt_pe = self.pipe.encode_prompt(prompt, self.dev, 1, False)[0]
        if new_line:                                                     # gecikme ölçümü: sahne kuruldu
            self._lyric_pending = lid
            if self.tracker is not None:
                self.tracker.stamp(lid, "scene")
        self._prompt = prompt
        self.seed = int(rng.integers(1 << 31))                            # yeni kompozisyon
        self._scene_t, self._scene_a = now, a
        # Önceki sahneyle bağı kopar: hafıza sıfır, ilk kare yüksek gürültü, farklı kadraj
        self._prev = None
        self._fresh = True
        self._flip = (bool(rng.integers(2)), bool(rng.integers(2)))
        self._shift = (float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
        self.concepts = [(self.c_names[i], w) for i, w in zip(ci, cv)]
        self.mood_name = self.m_names[mi]
        vm = "—" if va_m is None else f"{float(va_m[0]):+.2f}/{float(va_m[1]):+.2f}"
        print(f"[Difüzyon] sahne  tür {self.genre_name} ({self._genre_p[gi]:.2f})  bölüm {self.section}  v/a {ev:+.2f}/{ea:+.2f}  (model {vm}, CLAP {v_c:+.2f}/{a_c:+.2f})  → {prompt}")

    def _blend(self, va):
        v, a = (float(va[0]) + 1) / 2, (float(va[1]) + 1) / 2
        emo = sum((v if sv > 0 else 1 - v) * (a if sa > 0 else 1 - a) * e for (sv, sa), e in self.anc.items())
        return emo if self._prompt_pe is None else self._prompt_pe

    def _style_tensors(self, st):
        torch, V = self.torch, self.V
        d = {"W": st["W"][None], "B": st["B"][None], "acts": st["acts"][None],
             "pal": np.asarray(st["palette"], np.float32)[None]}
        d.update({k: np.array([st.get(k, (lo + hi) / 2)], np.float32) for k, lo, hi in V.SC})
        return {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=self.dev) for k, v in d.items()}

    def _generate(self, st, z, va):
        """Bir kare: CPPN (+ önceki hayal) → SD-Turbo. (h, w, 3) uint8 döndürür."""
        torch = self.torch
        with self._nograd():
            zt = torch.tensor(np.clip(z, -3, 3), device=self.dev)[None]
            st = dict(st, lines=0.0)                                    # kontur çizgileri yok
            init = self.V.render(self._style_tensors(st), zt, self.h, self.w)
            small = torch.nn.functional.interpolate(init, scale_factor=1 / 8, mode="area")
            init = torch.nn.functional.interpolate(small, size=(self.h, self.w), mode="bilinear",
                                                   align_corners=False)      # sadece renk + kaba kompozisyon
            # Sahneye özgü kadraj: aynı CPPN renkleri her sahnede aynı kompozisyonu vermesin
            fy, fx = self._flip
            dims = [d for d, f in ((2, fy), (3, fx)) if f]
            if dims:
                init = torch.flip(init, dims)
            sy, sx = int(self._shift[0] * self.h), int(self._shift[1] * self.w)
            init = torch.roll(init, shifts=(sy, sx), dims=(2, 3))
            if self._prev is not None:                                  # hafıza: sahne içinde akıcılık
                init = self.MEMORY * self._prev + (1 - self.MEMORY) * init
            noise = max(self.noise, self.FRESH_NOISE) if self._fresh else self.noise
            if self._pulse and not self._fresh:
                noise = min(990, noise + self.PULSE_NOISE)
            self._fresh = self._pulse = False
            gen = torch.Generator(self.dev).manual_seed(self.seed)
            out = self.pipe(prompt_embeds=self._blend(va), image=init, strength=1.0,
                            timesteps=[noise], guidance_scale=0.0, generator=gen,
                            output_type="pt").images[0]
            self._prev = out.float().clamp(0, 1)[None]
            return (self._prev[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()

    def _loop(self):
        try:
            t0 = time.time()
            self._load()
            print(f"[Difüzyon] Hazır ({time.time() - t0:.0f} sn) — SD-Turbo {self.w}×{self.h}")
            self.ready = True
        except Exception as e:
            self.failed = str(e)
            print(f"[Difüzyon] Yüklenemedi: {e}")
            return
        torch = self.torch
        # GPU önceliği: SD yüksek öncelikli stream'de; CLAP ve Whisper düşük öncelikte.
        # Whisper çalışırken SD'nin kare hızı yarıya düşüyordu (47 → 115 ms).
        hi = torch.cuda.Stream.priority_range()[1]
        self._sd_stream = torch.cuda.Stream(priority=hi)
        threading.Thread(target=self._clap_loop, daemon=True).start()
        last_id, times = -1, []
        torch.cuda.set_stream(self._sd_stream)
        while self.running:
            try:
                self._update_concepts()
            except Exception as e:
                print(f"[Difüzyon] Kavram hatası: {e}")
            with self._lock:
                inp, iid = self._inp, self._inp_id
            if inp is None or iid == last_id:
                # Yeni girdi yok: kısa bekle, ama yeni söz satırı gelirse hemen uyan
                if self._wake.wait(0.005):
                    self._wake.clear()
                continue
            last_id = iid
            st, z, va = inp
            pending = self._lyric_pending
            t1 = time.perf_counter()
            try:
                img = self._generate(st, z, va)
            except Exception as e:
                if self.compile:                                         # derlenmiş UNet hata verdi → eski yol
                    self._uncompile(e)
                    continue
                print(f"[Difüzyon] Hata: {e}")
                time.sleep(0.5)
                continue
            if pending is not None and self.tracker is not None:
                self.tracker.stamp(pending, "frame")
            with self._lock:
                self.frame, self.frame_id = img, self.frame_id + 1
                self.frame_lyric = pending
            if pending is not None and self._lyric_pending == pending:
                self._lyric_pending = None
            times.append(time.perf_counter() - t1)
            if len(times) >= 10:
                self.fps = len(times) / sum(times)
                times.clear()

    def _clap_loop(self):
        torch = self.torch
        torch.cuda.set_stream(torch.cuda.Stream(priority=0))           # düşük öncelik
        while self.running:
            t0 = time.time()
            try:
                self._clap_step()
            except Exception as e:
                print(f"[Difüzyon] CLAP hatası: {e}")
                time.sleep(0.5)
            time.sleep(max(0.02, self.CONCEPT_EVERY - (time.time() - t0)))
