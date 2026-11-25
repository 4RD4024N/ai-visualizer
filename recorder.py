"""
Video kaydı — ekran + sistem sesi → MP4 (ffmpeg).

  • Görüntü : her karede (hedef FPS'e göre) glReadPixels → kuyruk → yazıcı thread →
              ffmpeg stdin (rawvideo → libx264). Render thread'i diske beklemez.
  • Senkron : kare sayısı duvar saatine göre tutulur; render yavaşsa son kare
              tekrarlanır, hızlıysa kare atlanır → video süresi = gerçek süre.
  • Ses     : ses callback'inden gelen bloklar ayrı bir WAV'a yazılır.
  • Bitiş   : video + ses arka planda tek MP4'te birleştirilir (video kopyalanır,
              ses AAC). ffmpeg yoksa kayıt devre dışıdır.
"""

import ctypes
import os
import queue
import shutil
import subprocess
import threading
import time
import wave

import numpy as np


def find_ffmpeg():
    return shutil.which("ffmpeg")


class VideoRecorder:
    def __init__(self, out_dir, fps=30, crf=18):
        self.out_dir = out_dir
        self.fps = int(fps)
        self.crf = int(crf)
        self.ffmpeg = find_ffmpeg()
        self.active = False
        self.size = (0, 0)
        self.last_output = None
        self.status = ""
        self._frames = 0
        self._dropped = 0

    @property
    def available(self):
        return self.ffmpeg is not None

    @property
    def elapsed(self):
        return time.perf_counter() - self._t0 if self.active else 0.0

    # ── Başlat / durdur (ana thread) ──
    def start(self, width, height, sample_rate, channels):
        if not self.available:
            self.status = "ffmpeg bulunamadı"
            print("[Kayıt] ffmpeg PATH'te yok — kurulum: winget install Gyan.FFmpeg")
            return False
        w, h = width - width % 2, height - height % 2           # x264 çift boyut ister
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.final = os.path.join(self.out_dir, f"visualizer_{stamp}.mp4")
        self._vid = os.path.join(self.out_dir, f".tmp_{stamp}_video.mp4")
        self._wav_path = os.path.join(self.out_dir, f".tmp_{stamp}_audio.wav")
        cmd = [self.ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
               "-vf", "vflip", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
               "-pix_fmt", "yuv420p", self._vid]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.status = f"ffmpeg başlatılamadı: {e}"
            print(f"[Kayıt] {self.status}")
            return False
        self._wav = wave.open(self._wav_path, "wb")
        self._wav.setnchannels(int(channels))
        self._wav.setsampwidth(2)
        self._wav.setframerate(int(sample_rate))
        self._channels = int(channels)
        self._audio_q = queue.Queue(maxsize=512)
        self._video_q = queue.Queue(maxsize=6)
        self._buf = (ctypes.c_ubyte * (w * h * 3))()
        self.size = (w, h)
        self._frames = self._dropped = 0
        self._t0 = time.perf_counter()
        self.active = True
        self.status = "kaydediliyor"
        self._threads = [threading.Thread(target=self._video_writer, daemon=True),
                         threading.Thread(target=self._audio_writer, daemon=True)]
        for t in self._threads:
            t.start()
        print(f"[Kayıt] Başladı: {w}x{h} @ {self.fps} FPS → {self.final}")
        return True

    def stop(self):
        if not self.active:
            return
        self.active = False
        self._video_q.put(None)
        self._audio_q.put(None)
        self.status = "birleştiriliyor"
        self._finisher = threading.Thread(target=self._finish, daemon=True)
        self._finisher.start()

    def wait(self, timeout=120):
        """Uygulama kapanırken: birleştirme bitene kadar bekle (yoksa MP4 yarım kalır)."""
        t = getattr(self, "_finisher", None)
        if t is not None and t.is_alive():
            print("[Kayıt] MP4 birleştiriliyor, bekleniyor...")
            t.join(timeout)

    # ── Besleme ──
    def push_audio(self, block):
        """Ses thread'i: (n, kanal) float32 blok."""
        if not self.active:
            return
        try:
            self._audio_q.put_nowait(np.array(block[:, :self._channels], copy=True))
        except queue.Full:
            pass

    def capture(self, gl):
        """Render thread'i, sahne çizildikten sonra (flip'ten önce) çağırır."""
        if not self.active:
            return
        due = int(self.elapsed * self.fps) + 1                  # şu ana kadar olması gereken kare
        repeat = due - self._frames
        if repeat <= 0:
            return                                              # hedef FPS'ten hızlıyız: bu kareyi atla
        w, h = self.size
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self._buf)
        try:
            self._video_q.put_nowait((bytes(self._buf), repeat))
            self._frames += repeat
        except queue.Full:
            self._dropped += 1                                  # yazıcı yetişemiyor; sonraki kare telafi eder

    # ── Thread'ler ──
    def _video_writer(self):
        stdin = self._proc.stdin
        while True:
            item = self._video_q.get()
            if item is None:
                break
            data, repeat = item
            try:
                for _ in range(repeat):
                    stdin.write(data)
            except (BrokenPipeError, OSError) as e:
                self.status = f"ffmpeg hatası: {e}"
                print(f"[Kayıt] {self.status}")
                self.active = False
                break
        try:
            stdin.close()
        except OSError:
            pass

    def _audio_writer(self):
        while True:
            block = self._audio_q.get()
            if block is None:
                break
            pcm = (np.clip(block, -1.0, 1.0) * 32767).astype("<i2")
            self._wav.writeframes(pcm.tobytes())
        self._wav.close()

    def _finish(self):
        for t in self._threads:
            t.join(timeout=30)
        self._proc.wait(timeout=60)
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-i", self._vid, "-i", self._wav_path,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", self.final]
        r = subprocess.run(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0:
            for p in (self._vid, self._wav_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            self.last_output = self.final
            self.status = f"kaydedildi: {os.path.basename(self.final)}"
            extra = f" ({self._dropped} kare yetişemedi)" if self._dropped else ""
            print(f"[Kayıt] Kaydedildi: {self.final}{extra}")
        else:
            self.status = "birleştirme başarısız — geçici dosyalar duruyor"
            print(f"[Kayıt] Birleştirme başarısız. Video: {self._vid}  Ses: {self._wav_path}")
