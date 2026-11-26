"""
Şarkı sözü dinleyici — canlı sesten vokali yazıya döker, İngilizceye çevirir.

  • Model   : openai/whisper-medium (config.json → lyrics_model ile değiştirilebilir).
  • Dil     : sadece Türkçe ve İngilizce. Whisper'ın dil olasılıklarından ikisi arasında
              seçilir (müzik altında serbest tahmin Türkçeyi Azerice vb. sanabiliyor);
              "ses yok" olasılığı yüksekse pencere atlanır.
  • Çıktı   : orijinal satır (HUD / log) + İngilizcesi (SD'nin metin kodlayıcısı yalnızca
              İngilizce anlar): Türkçe söz → Whisper çevirisi, İngilizce söz → kendisi.
  • Döngü   : kayan pencere — her ~0,5 sn'de son 4 sn'lik ses (AudioTap.latest_raw) →
              16 kHz → Whisper; çıkarım adımdan uzunsa beklemeden sıradaki pencere (sürekli
              döngü). Düşük öncelikli CUDA stream'inde çalışır; SD'nin kare hızını düşürmez.
              (Eski: 1,5 sn'de bir 6 sn → ortalama 0,75 sn bekleme + daha uzun metin üretimi.)
  • Hız     : Türkçede SD'ye giden İngilizce çeviri (task=translate) TEK generate ile önce
              üretilir ve satır hemen yayınlanır; Türkçe orijinal (HUD / log) yalnızca YENİ
              satırlar için sonradan üretilir (lyrics_hud_original). Dil tespitinde hesaplanan
              encoder çıktısı generate'e yeniden verilir (encoder iki kez koşmaz).
  • Gecikme : latency.LatencyTracker'a pencere / Whisper / kabul damgaları yazılır.
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

import latency

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


def blacklisted(text):
    low = text.lower()
    return any(b in low for b in BLACKLIST)


def overlap(a, b):
    wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


class LyricsListener:
    EVERY = 0.5            # sn — pencere adımı (lyrics_step)
    WINDOW = 4.0           # sn — pencere boyu (lyrics_window)
    MIN_AUDIO = 3.0        # sn — bundan kısa ses birikmişse bekle
    MAX_TOKENS = 40        # 4 sn'lik söz ≈ 10–30 token; uydurma döngüleri ("la la…") erken kesilir
    MIN_LOGPROB = -0.9     # ortalama token log-olasılığı bunun altındaysa uydurma say
    MIN_RMS = 0.01
    NOSPEECH = 0.6         # "ses yok" olasılığı bunu aşarsa pencere atlanır (generate hiç koşmaz)
    KEEP = 15.0            # bir satır kaç sn "güncel" sayılır

    def __init__(self, tap, model=None, window=None, step=None, translate_first=True,
                 hud_original=True, reuse_encoder=True, tracker=None):
        self.tap = tap
        self.model_name = model or MODEL
        if window:
            self.WINDOW = float(window)
        if step:
            self.EVERY = float(step)
        self.translate_first = translate_first   # Türkçe: önce (ve tek) çeviri, orijinal sonra
        self.hud_original = hud_original         # Türkçe orijinali ikinci generate ile üret
        self.reuse_encoder = reuse_encoder       # _detect'in encoder çıktısını generate'e ver
        self.tracker = tracker                   # latency.LatencyTracker (ya da None)
        self.on_line = []                        # yeni satır geri çağrıları: f(line_id) — SD'yi uyandırır
        self.ready = False
        self.failed = None
        self.running = False
        self.enabled = True
        self.lines = collections.deque(maxlen=6)     # [(zaman, ingilizce, orijinal, dil, line_id)]
        self.line_id = 0                             # yeni satır geldikçe artar
        self.last_raw = ""                           # elenmiş olsa da son çıktı (hata ayıklama)
        self.infer_ms = 0.0
        self._lock = threading.Lock()
        self._prev_window_t = None                   # bir önceki pencerenin alındığı an
        self._prev_tail = None                       # ses akışı durduysa aynı pencereyi tekrar işleme
        self._feat_device = True                     # öznitelik çıkarımını GPU'da dene

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

    def current_with_id(self, max_age=None):
        """(İngilizce satır, line_id) — ikisi aynı kilit altında okunur (SD thread'i için)."""
        ln = self._latest(max_age)
        return (ln[1], ln[4]) if ln else (None, self.line_id)

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

    def _features(self, x16):
        """16 kHz ses → log-mel (fp16, GPU). Önce öznitelik çıkarımını doğrudan GPU'da dener
        (torch STFT; CPU numpy STFT'den hızlı); desteklenmezse bir kez log yazıp CPU'ya döner."""
        torch = self.torch
        if self._feat_device:
            try:
                f = self.proc.feature_extractor(x16, sampling_rate=16000, return_tensors="pt",
                                                device="cuda").input_features
                return f.to(self.dev, torch.float16)
            except Exception as e:
                self._feat_device = False
                print(f"[Söz] GPU öznitelik çıkarımı kapalı ({e}) — CPU yolu")
        feats = self.proc(x16, sampling_rate=16000, return_tensors="pt").input_features
        return feats.to(self.dev, torch.float16)

    def _detect(self, feats):
        """(dil, Türkçe/İngilizce içindeki güven, ses-yok olasılığı, encoder çıktısı) — ilk çözücü adımından."""
        torch = self.torch
        with torch.inference_mode():
            enc = self.model.model.encoder(feats).last_hidden_state
            dec = torch.tensor([[self.sot]], device=self.dev)
            logits = self.model(encoder_outputs=(enc,), decoder_input_ids=dec).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        nospeech = float(p[self.nospeech_id]) if self.nospeech_id is not None else 0.0
        pl = {l: float(p[i]) for l, i in self.lang_ids.items()}
        lang = max(pl, key=pl.get)
        conf = pl[lang] / (sum(pl.values()) + 1e-9)
        return lang, conf, nospeech, enc

    def _generate(self, feats, lang, task, enc=None):
        torch = self.torch
        kw = dict(language=lang, task=task, max_new_tokens=self.MAX_TOKENS,
                  return_dict_in_generate=True, output_scores=True)
        with torch.inference_mode():
            out = None
            if enc is not None and self.reuse_encoder:
                try:
                    from transformers.modeling_outputs import BaseModelOutput
                    out = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc), **kw)
                except Exception as e:
                    self.reuse_encoder = False
                    print(f"[Söz] Encoder yeniden kullanımı kapalı ({e}) — eski yol")
                    out = None
            if out is None:
                out = self.model.generate(feats, **kw)
            sc = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        lp = sc[0].float()
        lp = lp[torch.isfinite(lp)]
        avg = float(lp.mean()) if len(lp) else -9.0
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        return " ".join(text.split()), avg

    def _transcribe(self, x16):
        """→ (ingilizce, orijinal | None, dil, ortalama log-olasılık, ctx) ya da None (vokal yok).
        orijinal None ise Türkçe metin henüz üretilmedi (translate-first); ctx ile sonradan
        _original(ctx) çağrılabilir."""
        feats = self._features(x16)
        lang, conf, nospeech, enc = self._detect(feats)
        if nospeech > self.NOSPEECH:
            self.last_raw = f"(ses yok {nospeech:.2f})"
            return None
        ctx = (feats, lang, enc)
        if lang == "en":
            orig, avg = self._generate(feats, lang, "transcribe", enc)
            return orig, orig, lang, avg, ctx
        if self.translate_first:
            en, avg_en = self._generate(feats, lang, "translate", enc)
            return en, None, lang, avg_en, ctx
        orig, avg = self._generate(feats, lang, "transcribe", enc)
        en, avg_en = self._generate(feats, lang, "translate", enc)
        return en, orig, lang, min(avg, avg_en), ctx

    def _original(self, ctx):
        feats, lang, enc = ctx
        return self._generate(feats, lang, "transcribe", enc)

    def _push(self, text, orig, lang, stamps=None):
        """Satırı kaydet. Yeni satırsa line_id'yi, örtüşen tekrar ise None döndürür."""
        now = time.time()
        with self._lock:
            if self.lines and now - self.lines[-1][0] < self.KEEP and overlap(text, self.lines[-1][1]) >= 0.6:
                # Aynı satırın örtüşen penceresi: daha uzun olanı tut, yeni satır sayma
                last = self.lines[-1]
                if len(text) > len(last[1]):
                    self.lines[-1] = (now, text, orig if orig is not None else last[2], lang, last[4])
                else:
                    self.lines[-1] = (now,) + tuple(last[1:])
                return None
            nid = self.line_id + 1
            if self.tracker is not None and stamps:
                for k, t in stamps.items():                  # SD thread'i satırı görmeden önce
                    self.tracker.stamp(nid, k, t)
                self.tracker.stamp(nid, "accept")
            self.lines.append((now, text, orig if orig is not None else text, lang, nid))
            self.line_id = nid
        for cb in list(self.on_line):
            try:
                cb(nid)
            except Exception as e:
                print(f"[Söz] Geri çağrı hatası: {e}")
        return nid

    def _set_original(self, lid, orig):
        with self._lock:
            for i, ln in enumerate(self.lines):
                if ln[4] == lid:
                    self.lines[i] = ln[:2] + (orig,) + ln[3:]

    def _retract(self, lid):
        """Sonradan uydurma olduğu anlaşılan satırı geri çek (bir sonraki sahnede kullanılmaz)."""
        with self._lock:
            keep = [ln for ln in self.lines if ln[4] != lid]
            self.lines.clear()
            self.lines.extend(keep)

    def _log(self, orig, en, lang):
        extra = f"  →  {en}" if lang != "en" and en != orig else ""
        print(f"[Söz] [{lang}] » {orig}{extra}")

    def _accept(self, res, stamps):
        """Whisper sonucu → filtre → satır. Yeni satırın id'si ya da None."""
        en, orig, lang, avg, ctx = res
        self.last_raw = f"[{lang}] {orig if orig is not None else en} ({avg:.2f})"
        bad_orig = orig is not None and blacklisted(orig)
        line = clean_line(en) if (avg >= self.MIN_LOGPROB and not bad_orig) else None
        if not line:
            return None
        lid = self._push(line, orig, lang, stamps)
        if lid is None:
            return None
        if orig is None and self.hud_original:
            # Satır SD'ye çoktan gitti; Türkçe orijinal yalnızca HUD / log ve kara liste için
            # Eski filtrenin gücü korunur: orijinal kara listede ya da log-olasılığı düşükse
            # (eski yolda min(transcribe, translate) eşiği) satır geri çekilir
            orig2, avg2 = self._original(ctx)
            if blacklisted(orig2) or avg2 < self.MIN_LOGPROB:
                self._retract(lid)
                print(f"[Söz] [{lang}] geri çekildi (uydurma): {orig2}")
                return None
            self._set_original(lid, orig2)
            orig = orig2
        self._log(orig if orig is not None else line, line, lang)
        return lid

    def _step(self, resample):
        """Bir pencere. Testlerde sahte tap / _transcribe ile çağrılır."""
        x, sr = self.tap.latest_raw()
        t_win = latency.now()
        prev, self._prev_window_t = self._prev_window_t, t_win
        if not sr or len(x) < sr * min(self.MIN_AUDIO, self.WINDOW):
            return None
        x = x[-int(self.WINDOW * sr):]
        tail = x[-256:]
        if self._prev_tail is not None and np.array_equal(tail, self._prev_tail):
            return None                                  # ses akışı durmuş: aynı pencere
        self._prev_tail = tail.copy()
        if float(np.sqrt(np.mean(x ** 2))) < self.MIN_RMS:
            return None
        stamps = {"heard": latency.heard_estimate(t_win, prev, self.WINDOW), "window": t_win}
        x16 = resample(x, sr)
        t1 = time.perf_counter()
        res = self._transcribe(x16)
        self.infer_ms = (time.perf_counter() - t1) * 1000
        if res is None:
            return None
        stamps["whisper"] = latency.now()
        return self._accept(res, stamps)

    def _loop(self):
        try:
            t0 = time.time()
            self._load()
            self.ready = True
            print(f"[Söz] Hazır ({time.time() - t0:.0f} sn) — {self.model_name}, Türkçe + İngilizce, "
                  f"pencere {self.WINDOW:g} sn / adım {self.EVERY:g} sn")
        except Exception as e:
            self.failed = str(e)
            print(f"[Söz] Yüklenemedi: {e}")
            return
        from scipy.signal import resample_poly

        def resample(x, sr):
            g = np.gcd(int(sr), 16000)
            return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)

        self.torch.cuda.set_stream(self.torch.cuda.Stream(priority=0))  # düşük öncelik: SD önde
        while self.running:
            t_start = time.time()
            if self.enabled:
                try:
                    self._step(resample)
                except Exception as e:
                    print(f"[Söz] Hata: {e}")
                    time.sleep(1.0)
            else:
                self._prev_window_t = None
            # Sürekli döngü: çıkarım adımdan uzun sürdüyse beklemeden sıradaki pencere
            time.sleep(max(0.01, self.EVERY - (time.time() - t_start)))
