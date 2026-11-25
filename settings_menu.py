"""
Ekran üstü ayarlar menüsü (S).

  ↑ ↓      seçim
  ← →      değeri değiştir (açık/kapalı ayarlarda da çalışır)
  ENTER    açık/kapalı
  R        seçili ayarı varsayılana döndür
  S / ESC  kapat

Her değişiklik hemen uygulanır ve config.json'a kaydedilir.
"""

import pyglet
import pyglet.shapes as shapes

import config

# (anahtar, etiket, tür, min, maks, adım)
ITEMS = [
    ("sensitivity", "Hassasiyet", "float", 0.2, 3.0, 0.1),
    ("beat_sync", "Beat senkronu", "bool", None, None, None),
    ("section_detection", "Bölüm algılama", "bool", None, None, None),
    ("flash", "Flash gücü", "float", 0.0, 2.0, 0.1),
    ("zoom_pulse", "Zoom darbesi", "float", 0.0, 2.0, 0.1),
    ("palette_drift", "Renk kayması", "float", 0.0, 0.2, 0.005),
    ("scene_duration", "Sahne süresi (sn)", "float", 6.0, 90.0, 2.0),
    ("show_hud", "HUD", "bool", None, None, None),
    ("diff_bars", "Difüzyon: kaç ölçüde yeni sahne", "int", 1, 8, 1),
    ("diff_valence_bias", "Difüzyon: pozitiflik eğilimi", "float", -1.0, 1.0, 0.1),
    ("lyrics", "Difüzyon: şarkı sözü etkisi", "bool", None, None, None),
    ("record_fps", "Kayıt FPS", "int", 15, 60, 5),
    ("record_crf", "Kayıt kalitesi (CRF, düşük=iyi)", "int", 12, 32, 1),
    ("record_hud", "Kayda HUD dahil", "bool", None, None, None),
]


class SettingsMenu:
    W, ROW = 520, 30

    def __init__(self, cfg, on_change):
        self.cfg = cfg
        self.on_change = on_change
        self.visible = False
        self.sel = 0
        self.batch = pyglet.graphics.Batch()
        h = 90 + self.ROW * len(ITEMS)
        self.bg = shapes.Rectangle(0, 0, self.W, h, color=(10, 12, 22, 235), batch=self.batch)
        self.hl = shapes.Rectangle(0, 0, self.W - 24, self.ROW - 4, color=(60, 90, 160, 160), batch=self.batch)
        self.title = pyglet.text.Label("AYARLAR", font_name="Consolas", font_size=15, weight="bold",
                                       color=(230, 235, 250, 255), batch=self.batch)
        self.foot = pyglet.text.Label("↑↓ seç   ←→ değiştir   R varsayılan   S kapat",
                                      font_name="Consolas", font_size=10, color=(150, 160, 185, 230),
                                      batch=self.batch)
        self.rows = [(pyglet.text.Label("", font_name="Consolas", font_size=12, color=(210, 220, 240, 255),
                                        batch=self.batch),
                      pyglet.text.Label("", font_name="Consolas", font_size=12, color=(255, 210, 130, 255),
                                        anchor_x="right", batch=self.batch)) for _ in ITEMS]
        self._h = h
        self._refresh()

    def toggle(self):
        self.visible = not self.visible

    def _fmt(self, kind, v):
        if kind == "bool":
            return "AÇIK" if v else "kapalı"
        if kind == "int":
            return str(int(v))
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:.1f}"

    def _refresh(self):
        for (key, label, kind, *_), (l_lbl, l_val) in zip(ITEMS, self.rows):
            l_lbl.text = label
            l_val.text = self._fmt(kind, self.cfg[key])

    def _change(self, direction):
        key, _, kind, lo, hi, step = ITEMS[self.sel]
        if kind == "bool":
            self.cfg[key] = not self.cfg[key]
        else:
            v = self.cfg[key] + direction * step
            v = min(hi, max(lo, v))
            self.cfg[key] = int(round(v)) if kind == "int" else round(v, 4)
        self._commit(key)

    def _commit(self, key):
        self._refresh()
        config.save(self.cfg)
        self.on_change(key, self.cfg[key])

    def on_key(self, sym):
        """Menü açıkken tuşu tüketir → True."""
        k = pyglet.window.key
        if not self.visible:
            return False
        if sym in (k.S, k.ESCAPE):
            self.visible = False
        elif sym == k.UP:
            self.sel = (self.sel - 1) % len(ITEMS)
        elif sym == k.DOWN:
            self.sel = (self.sel + 1) % len(ITEMS)
        elif sym == k.LEFT:
            self._change(-1)
        elif sym in (k.RIGHT, k.ENTER, k.RETURN):
            self._change(+1)
        elif sym == k.R:
            key = ITEMS[self.sel][0]
            self.cfg[key] = config.DEFAULTS[key]
            self._commit(key)
        return True

    def draw(self, win_w, win_h):
        if not self.visible:
            return
        x0, y0 = win_w // 2 - self.W // 2, win_h // 2 - self._h // 2
        self.bg.x, self.bg.y = x0, y0
        top = y0 + self._h
        self.title.x, self.title.y = x0 + 18, top - 34
        for i, (l_lbl, l_val) in enumerate(self.rows):
            y = top - 70 - i * self.ROW
            l_lbl.x, l_lbl.y = x0 + 22, y
            l_val.x, l_val.y = x0 + self.W - 22, y
            if i == self.sel:
                self.hl.x, self.hl.y = x0 + 12, y - 8
        self.foot.x, self.foot.y = x0 + 18, y0 + 14
        self.batch.draw()
