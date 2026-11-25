"""
Şarkı sözü dinleyici — canlı sesten vokali yazıya döker, İngilizceye çevirir.

  • Model   : openai/whisper-medium (config.json → lyrics_model ile değiştirilebilir).
  • Dil     : sadece Türkçe ve İngilizce. Whisper'ın dil olasılıklarından ikisi arasında
              seçilir (müzik altında serbest tahmin Türkçeyi Azerice vb. sanabiliyor);
              "ses yok" olasılığı yüksekse pencere atlanır.
  • Çıktı   : orijinal satır (HUD / log) + İngilizcesi (SD'nin metin kodlayıcısı yalnızca
              İngilizce anlar): Türkçe söz → Whisper çevirisi, İngilizce söz → kendisi.
  • Döngü   : ~1,5 sn'de bir son 6 sn'lik ses (AudioTap.latest_raw) → 16 kHz → Whisper.
              Düşük öncelikli CUDA stream'inde çalışır; SD'nin kare hızını düşürmez.
  • Eleme   : Whisper müzik altında sık sık metin uydurur ("thank you for watching",
              "altyazı M.K."…). Ortalama log-olasılık eşiği + kara liste + kısa / tekrarlı
              metin elemesi. Şüpheli satır hiç kullanılmaz.
  • Satırlar: pencereler örtüştüğü için aynı söz art arda gelir; kelimeleri büyük ölçüde
              örtüşen satır öncekinin yerine geçer, yeni satır sayılmaz.

torch + transformers + CUDA gerekir (difüzyon ortamı). Yoksa sessizce kapalı kalır.
"""

import collections
import os
import re
import threading
import time

import numpy as np

WORK = os.environ.get("AIVIS_WORK", r"D:\ai-visualizer-work")
MODEL = "openai/whisper-medium"

BLACKLIST = [
    "thank you for watching", "thanks for watching", "subscribe", "subtitles", "subtitle",
    "altyazı", "altyazi", "amara", "copyright", "please like", "see you", "bye bye",
    "music", "♪", "applause", "laughter", "transcribed", "translated by", "caption",
    "www.", ".com", "m.k.", "dvd", "the end",
    # Türkçe uydurmalar (orijinal metinde aranır)
    "izlediğiniz için", "teşekkür ederim", "abone ol", "beğenmeyi unutmayın", "altyazılar",
    "sesli betimleme", "müzik", "alkış",
]
LANGS = ("tr", "en")
STOP = set("""a an the and or but if then so of to in on at by for with from as is are was were be been
being i me my you your he she it we they them his her its our their this that these those do does did
not no yes oh ooh yeah hey la na da uh ah mm just like get got go gonna wanna can will would""".split())


def clean_line(text):
    """Söz satırı mı? Değilse None. Satırsa sadeleştirilmiş hali."""
    t = " ".join(text.replace("\n", " ").split()).strip(" .,-–—\"'")
    low = t.lower()
    if len(t) < 6 or any(b in low for b in BLACKLIST):
        return None
    words = re.findall(r"[a-zA-Z']+", low)
    if len(words) < 3:
        return None
    if len(set(words)) / len(words) < 0.4:              # "la la la la", aynı kelime tekrarı
        return None
    content = [w for w in words if w not in STOP]
    if len(content) < 2:
        return None
    return t


def overlap(a, b):
    wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


class LyricsListener:
    EVERY = 1.5            # sn
    WINDOW = 6.0           # sn
    MIN_LOGPROB = -0.9     # ortalama token log-olasılığı bunun altındaysa uydurma say
    MIN_RMS = 0.01
    KEEP = 15.0            # bir satır kaç sn "güncel" sayılır

    def __init__(self, tap, model=None):
        self.tap = tap
        self.model_name = model or MODEL
        self.ready = False
        self.failed = None
        self.running = False
        self.enabled = True
        self.lines = collections.deque(maxlen=6)     # [(zaman, ingilizce, orijinal, dil)]
        self.line_id = 0                             # yeni satır geldikçe artar
        self.last_raw = ""                           # elenmiş olsa da son çıktı (hata ayıklama)
        self.infer_ms = 0.0
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _latest(self, max_age):
        max_age = self.KEEP if max_age is None else max_age
        with self._lock:
            if self.lines and time.time() - self.lines[-1][0] <= max_age:
                return self.lines[-1]
        return None

    def current(self, max_age=None):
        """En son güncel satırın İngilizcesi (SD için). Yoksa None."""
        ln = self._latest(max_age)
        return ln[1] if ln else None

    def current_original(self, max_age=None):
        """(orijinal satır, dil) — HUD için. Yoksa None."""
        ln = self._latest(max_age)
        return (ln[2], ln[3]) if ln else None

    # ── Thread ──
    def _load(self):
        os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf"))
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()          # her pencerede tekrarlanan üretim uyarıları
        self.torch = torch
        self.dev = torch.device("cuda")
        self.proc = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16).to(self.dev).eval()
        tok = self.proc.tokenizer
        self.lang_ids = {l: tok.convert_tokens_to_ids(f"<|{l}|>") for l in LANGS}
        self.sot = self.model.generation_config.decoder_start_token_id
        self.nospeech_id = None
        for t in ("<|nospeech|>", "<|nocaptions|>"):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i != tok.unk_token_id:
                self.nospeech_id = i
                break

    def _detect(self, feats):
        """(dil, Türkçe/İngilizce içindeki güven, ses-yok olasılığı) — ilk çözücü adımından."""
        torch = self.torch
        with torch.no_grad():
            enc = self.model.model.encoder(feats).last_hidden_state
            dec = torch.tensor([[self.sot]], device=self.dev)
            logits = self.model(encoder_outputs=(enc,), decoder_input_ids=dec).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        nospeech = float(p[self.nospeech_id]) if self.nospeech_id is not None else 0.0
        pl = {l: float(p[i]) for l, i in self.lang_ids.items()}
        lang = max(pl, key=pl.get)
        conf = pl[lang] / (sum(pl.values()) + 1e-9)
        return lang, conf, nospeech, enc

    def _generate(self, feats, lang, task):
        torch = self.torch
        with torch.no_grad():
            out = self.model.generate(feats, language=lang, task=task, max_new_tokens=60,
                                      return_dict_in_generate=True, output_scores=True)
            sc = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        lp = sc[0].float()
        lp = lp[torch.isfinite(lp)]
        avg = float(lp.mean()) if len(lp) else -9.0
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        return " ".join(text.split()), avg

    def _transcribe(self, x16):
        """→ (ingilizce, orijinal, dil, ortalama log-olasılık) ya da None (vokal yok)."""
        torch = self.torch
        feats = self.proc(x16, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.dev, torch.float16)
        lang, conf, nospeech, _ = self._detect(feats)
        if nospeech > 0.6:
            self.last_raw = f"(ses yok {nospeech:.2f})"
            return None
        orig, avg = self._generate(feats, lang, "transcribe")
        if lang == "en":
            return orig, orig, lang, avg
        en, avg_en = self._generate(feats, lang, "translate")
        return en, orig, lang, min(avg, avg_en)

    def _push(self, text, orig, lang):
        now = time.time()
        with self._lock:
            if self.lines and now - self.lines[-1][0] < self.KEEP and overlap(text, self.lines[-1][1]) >= 0.6:
                # Aynı satırın örtüşen penceresi: daha uzun olanı tut, yeni satır sayma
                last = self.lines[-1]
                self.lines[-1] = (now, text, orig, lang) if len(text) > len(last[1]) else (now,) + tuple(last[1:])
                return
            self.lines.append((now, text, orig, lang))
            self.line_id += 1
        extra = f"  →  {text}" if lang != "en" else ""
        print(f"[Söz] [{lang}] » {orig}{extra}")

    def _loop(self):
        try:
            t0 = time.time()
            self._load()
            self.ready = True
            print(f"[Söz] Hazır ({time.time() - t0:.0f} sn) — {self.model_name}, Türkçe + İngilizce")
        except Exception as e:
            self.failed = str(e)
            print(f"[Söz] Yüklenemedi: {e}")
            return
        from scipy.signal import resample_poly
        self.torch.cuda.set_stream(self.torch.cuda.Stream(priority=0))  # düşük öncelik: SD önde
        while self.running:
            t_start = time.time()
            if self.enabled:
                try:
                    x, sr = self.tap.latest_raw()
                    if sr and len(x) >= sr * 3:
                        x = x[-int(self.WINDOW * sr):]
                        if float(np.sqrt(np.mean(x ** 2))) >= self.MIN_RMS:
                            g = np.gcd(int(sr), 16000)
                            x16 = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
                            t1 = time.perf_counter()
                            res = self._transcribe(x16)
                            self.infer_ms = (time.perf_counter() - t1) * 1000
                            if res is not None:
                                en, orig, lang, avg = res
                                self.last_raw = f"[{lang}] {orig} ({avg:.2f})"
                                bad_orig = any(b in orig.lower() for b in BLACKLIST)
                                line = clean_line(en) if (avg >= self.MIN_LOGPROB and not bad_orig) else None
                                if line:
                                    self._push(line, orig, lang)
                except Exception as e:
                    print(f"[Söz] Hata: {e}")
                    time.sleep(1.0)
            time.sleep(max(0.05, self.EVERY - (time.time() - t_start)))
