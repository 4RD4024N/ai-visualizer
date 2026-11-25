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

Ayrı thread'de çalışır; pencere / ses akışı beklemez. torch + diffusers yoksa
(ör. sistem Python'u) katman sessizce devre dışı kalır.
Modeller: stabilityai/sd-turbo (ticari olmayan lisans), madebyollin/taesd (hızlı çözücü),
          laion/larger_clap_music_and_speech (Apache-2.0, kavram seçimi).
"""

import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("AIVIS_WORK", r"D:\ai-visualizer-work")

ANCHORS = {
    (-1, -1): ["a sad, melancholic artwork", "sorrow, loneliness and grief", "a gloomy, mournful, heartbroken scene"],
    (-1, +1): ["an artwork full of anger and rage", "chaos, violence and danger", "a furious, menacing, dark and intense scene"],
    (+1, +1): ["a joyful, happy artwork", "euphoria, cheer and celebration", "a bright, exciting, festive scene"],
    (+1, -1): ["a calm, peaceful artwork", "serenity, gentleness and rest", "a tranquil, dreamy, soothing scene"],
}


def available():
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


class DiffusionLayer:
    CONCEPT_EVERY = 1.0        # sn — CLAP 10 sn'lik pencereye bakar, saniyede bir yeter
    N_CONCEPTS = 2             # cümleye giren kavram sayısı
    POOL, MOOD_POOL = 12, 3    # çekilişin yapıldığı en uyan kavram / ruh hali sayısı
    SCENE_SEC = 8.0            # bir sahnenin en uzun süresi
    CHANGE_DIST = 0.12         # müzik değişimi eşiği (CLAP temsilinde kosinüs uzaklığı)
    FRAMINGS = ["", "close-up, ", "wide shot, ", "aerial view, ", "silhouette, ", "panoramic landscape, ",
                "low angle, ", "detailed scene, "]
    MEMORY = 0.55              # başlangıç resminde önceki hayalin payı

    def __init__(self, tap=None, size=(640, 384), noise=800, seed=42):
        self.tap = tap
        self.concepts = []             # [(kavram, ağırlık)] — HUD için
        self.mood_name = None          # baskın ruh hali — HUD için
        self._prev = None              # önceki hayal (hafıza)
        self._scene_t = 0.0            # sahnenin başladığı an
        self._scene_a = None           # sahnenin başındaki CLAP temsili
        self._rng = np.random.default_rng()
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

    def get_frame(self):
        with self._lock:
            return self.frame_id, self.frame

    # ── Difüzyon thread'i ──
    def _load(self):
        os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf"))
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
        with torch.no_grad():
            self.anc = {q: pipe.encode_prompt(t, self.dev, 1, False)[0].mean(0, keepdim=True)
                        for q, t in ANCHORS.items()}
        self._load_clap()

    def _load_clap(self):
        """CLAP + kavram istatistikleri. Yoksa sadece duygu kullanılır."""
        self.clap = None
        self._prompt = None            # o anki cümle
        self._prompt_pe = None         # kodlanmış hali
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
            sys.path.insert(0, ROOT)
            from concepts import CONCEPTS, MOODS, MOOD_SD_TEXT
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
            print(f"[Difüzyon] Kavram seçimi açık: {len(CONCEPTS)} kavram + {len(MOODS)} ruh hali (CLAP)")
        except Exception as e:
            self.clap = None
            print(f"[Difüzyon] Kavram seçimi kapalı ({e}) — sadece duygu")

    def _update_concepts(self):
        """Son ~10 sn müzik → en özgü 3 kavram → içerik metin temsili."""
        if self.clap is None or self.tap is None:
            return
        torch = self.torch
        x, sr = self.tap.latest_raw()
        if sr is None or len(x) < sr:
            return
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
        now = time.time()
        changed = self._scene_a is None or float(1 - (a @ self._scene_a)) > self.CHANGE_DIST
        if not (changed or now - self._scene_t >= self.SCENE_SEC):
            return                                                       # sahne sürüyor
        rng = self._rng

        def draw(scores, pool, k, temp):
            top = torch.topk(scores, pool)
            p = torch.softmax(top.values / temp, 0).cpu().numpy().astype(np.float64)
            pick = rng.choice(pool, size=k, replace=False, p=p / p.sum())
            return [int(top.indices[j]) for j in pick], [float(top.values[j]) for j in pick]

        ci, cv = draw(z, self.POOL, self.N_CONCEPTS, 0.5)
        (mi,), _ = draw(zm, self.MOOD_POOL, 1, 0.3)
        framing = self.FRAMINGS[rng.integers(len(self.FRAMINGS))]
        prompt = framing + ", ".join([self.c_phrase[i] for i in ci] + [self.m_phrase[mi]])
        with torch.no_grad():
            self._prompt_pe = self.pipe.encode_prompt(prompt, self.dev, 1, False)[0]
        self._prompt = prompt
        self.seed = int(rng.integers(1 << 31))                            # yeni kompozisyon
        self._scene_t, self._scene_a = now, a
        self.concepts = [(self.c_names[i], v) for i, v in zip(ci, cv)]
        self.mood_name = self.m_names[mi]

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
        with torch.no_grad():
            zt = torch.tensor(np.clip(z, -3, 3), device=self.dev)[None]
            st = dict(st, lines=0.0)                                    # kontur çizgileri yok
            init = self.V.render(self._style_tensors(st), zt, self.h, self.w)
            small = torch.nn.functional.interpolate(init, scale_factor=1 / 8, mode="area")
            init = torch.nn.functional.interpolate(small, size=(self.h, self.w), mode="bilinear",
                                                   align_corners=False)      # sadece renk + kaba kompozisyon
            if self._prev is not None:                                  # hafıza: sahneyi sürdür
                init = self.MEMORY * self._prev + (1 - self.MEMORY) * init
            gen = torch.Generator(self.dev).manual_seed(self.seed)
            out = self.pipe(prompt_embeds=self._blend(va), image=init, strength=1.0,
                            timesteps=[self.noise], guidance_scale=0.0, generator=gen,
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
        last_id, times, last_c = -1, [], 0.0
        while self.running:
            if time.time() - last_c >= self.CONCEPT_EVERY:
                last_c = time.time()
                try:
                    self._update_concepts()
                except Exception as e:
                    print(f"[Difüzyon] Kavram hatası: {e}")
            with self._lock:
                inp, iid = self._inp, self._inp_id
            if inp is None or iid == last_id:
                time.sleep(0.005)
                continue
            last_id = iid
            st, z, va = inp
            t1 = time.perf_counter()
            try:
                img = self._generate(st, z, va)
            except Exception as e:
                print(f"[Difüzyon] Hata: {e}")
                time.sleep(0.5)
                continue
            with self._lock:
                self.frame, self.frame_id = img, self.frame_id + 1
            times.append(time.perf_counter() - t1)
            if len(times) >= 10:
                self.fps = len(times) / sum(times)
                times.clear()
