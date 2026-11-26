"""
Neural Background — CPPN + üretken shader katmanı

Üç GPU geçişi:
  1) CPPN      : her piksel için küçük bir MLP (12→16→16→16→4) çalışır.
                 Girdi = koordinat (kaleydoskop + domain warp sonrası) + 8 boyutlu latent + zaman.
  2) Composite : CPPN çıktısı → palet gradyanı + maske + kontur çizgileri,
                 önceki kareyle feedback (zoom / dönüş / akış izleri).
  3) Display   : bloom (mipmap glow), kromatik sapma, ACES ton eşleme, vinyet, grain.

Çeşitlilik "stil"lerden gelir: stil = ağ ağırlıkları + shader parametreleri.
Stiller arası geçişte ağırlıklar interpole edilir → bir dünya diğerine akarak dönüşür.

Latent vektör ses bantlarından + yavaş gürültüden üretilir; ses modeli varsa
`set_external_latent()` ile modelin 8 boyutlu latent'i karışır.

Rüya modu (`set_dream()`): görüntü modelin o anki çıktısının saf bir
fonksiyonudur. Zaman sayacı, yumuşatma, feedback izi, gürültü dokusu, film
grain'i, kaleydoskop, dönüş, bant/vuruş tepkisi yok. Ağın tüm ağırlıkları,
aktivasyonları, görsel parametreleri ve renk paleti ses modelinin 128 boyutlu
embedding'inden sabit bir izdüşümle (DreamMapper) türetilir. Hareket sadece
modelin çıktısı değiştikçe olur — sessizlikte de model çalışır.
"""

import os
import glob
import math
import random
import ctypes

import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

from config import app_dir

ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_DIR = os.path.join(app_dir(), "neural_styles")   # .exe'de .exe'nin yanında (yazılabilir)
EMB_STATS = os.path.join(ROOT, "models", "audio_latent_emb.npz")
EMO_HEAD = os.path.join(ROOT, "models", "emotion_head.npz")
VIS_GEN = os.path.join(ROOT, "models", "visual_gen.npz")

N_W = 48   # mat4 blokları: 12 (giriş) + 16 + 16 (gizli) + 4 (çıkış)
N_B = 13   # vec4 bias blokları: 4 + 4 + 4 + 1
LATENT = 8

# ── Shader kaynakları ───────────────────────────────────────────────────────

VERT_SRC = """#version 330 core
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
"""

CPPN_FRAG = """#version 330 core
out vec4 frag;
uniform vec2  u_res;
uniform float u_time;
uniform mat4  u_W[48];
uniform vec4  u_B[13];
uniform vec4  u_z0;
uniform vec4  u_z1;
uniform vec3  u_act1;
uniform vec3  u_act2;
uniform vec3  u_act3;
uniform float u_sym;
uniform float u_warp;
uniform float u_scale;
uniform float u_rot;

const float TAU = 6.28318530718;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return s;
}
vec4 act(vec4 v, vec3 w) {
    return w.x * sin(v) + w.y * tanh(v) + w.z * (exp(-v * v) * 2.0 - 1.0);
}

void main() {
    vec2 p = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;
    float c = cos(u_rot), s = sin(u_rot);
    p = mat2(c, -s, s, c) * p;

    if (u_sym > 0.5) {                       // kaleydoskop katlaması
        float a = atan(p.y, p.x);
        float r = length(p);
        float seg = TAU / u_sym;
        a = mod(a, seg);
        a = abs(a - 0.5 * seg);
        p = r * vec2(cos(a), sin(a));
    }
    vec2 q = vec2(fbm(p * 2.0 + vec2(0.0, u_time * 0.07)),
                  fbm(p * 2.0 + vec2(5.2, -u_time * 0.05)));
    p += u_warp * (q - 0.5);                 // domain warp
    p *= u_scale;

    vec4 x[3];
    x[0] = vec4(p, length(p), 0.6 * sin(u_time * 0.21));
    x[1] = u_z0;
    x[2] = u_z1;

    vec4 h[4];
    vec4 g[4];
    for (int i = 0; i < 4; i++) {
        vec4 a = u_B[i];
        for (int j = 0; j < 3; j++) a += u_W[i * 3 + j] * x[j];
        h[i] = act(a, u_act1);
    }
    for (int i = 0; i < 4; i++) {
        vec4 a = u_B[4 + i];
        for (int j = 0; j < 4; j++) a += u_W[12 + i * 4 + j] * h[j];
        g[i] = act(a, u_act2);
    }
    for (int i = 0; i < 4; i++) {
        vec4 a = u_B[8 + i];
        for (int j = 0; j < 4; j++) a += u_W[28 + i * 4 + j] * g[j];
        h[i] = act(a, u_act3);
    }
    vec4 o = u_B[12];
    for (int j = 0; j < 4; j++) o += u_W[44 + j] * h[j];
    frag = 0.5 + 0.5 * tanh(o);
}
"""

COMPOSITE_FRAG = """#version 330 core
out vec4 frag;
uniform sampler2D u_cppn;
uniform sampler2D u_prev;
uniform vec2  u_res;
uniform float u_time;
uniform vec3  u_c0;
uniform vec3  u_c1;
uniform vec3  u_c2;
uniform vec3  u_c3;
uniform float u_fb;
uniform float u_fbzoom;
uniform float u_fbrot;
uniform float u_flow;
uniform float u_energy;
uniform float u_beat;
uniform float u_high;
uniform float u_thresh;
uniform float u_lines;
uniform float u_linefreq;
uniform float u_huecycle;

vec3 grad4(float t) {
    t = fract(t) * 4.0;
    float f = fract(t);
    f = f * f * (3.0 - 2.0 * f);
    int i = int(floor(t));
    vec3 a = (i == 0) ? u_c0 : (i == 1) ? u_c1 : (i == 2) ? u_c2 : u_c3;
    vec3 b = (i == 0) ? u_c1 : (i == 1) ? u_c2 : (i == 2) ? u_c3 : u_c0;
    return mix(a, b, f);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_res;
    vec2 px = 2.0 / u_res;
    vec4 c = texture(u_cppn, uv);
    float gx = texture(u_cppn, uv + vec2(px.x, 0.0)).x - texture(u_cppn, uv - vec2(px.x, 0.0)).x;
    float gy = texture(u_cppn, uv + vec2(0.0, px.y)).x - texture(u_cppn, uv - vec2(0.0, px.y)).x;

    float v = smoothstep(0.05, 0.95, c.x);
    vec3 col = grad4(v * 0.75 + c.z * 0.5 + u_time * u_huecycle);
    col *= 0.35 + 0.65 * smoothstep(0.0, 0.7, v);          // iç gölgelendirme → derinlik

    // Maske: negatif alan bırakır (estetik için kritik); enerji arttıkça dolar
    float th = u_thresh - u_energy * 0.18;
    float mask = smoothstep(th - 0.07, th + 0.07, c.y);

    // Kontur çizgileri: keskin detay, tiz seslerle parlar
    float lf = c.w * u_linefreq;
    float fw = min(fwidth(lf), 0.25);
    float d = abs(fract(lf) - 0.5);
    float line = 1.0 - smoothstep(0.0, fw * 1.4 + 0.004, d);

    // Dolgu düz değil: tepeye doğru parlayan gölgelendirme (sulu boya hissi)
    float shade = 0.3 + 0.7 * smoothstep(th, th + 0.4, c.y);
    vec3 neu = col * mask * shade * (0.35 + 0.8 * u_energy);
    neu += col * line * u_lines * (0.45 + 1.4 * u_high);
    vec3 flash = col * u_beat * 0.35 * (0.2 + mask);

    // Feedback: önceki kare zoom + dönüş + CPPN gradyanı boyunca girdap akışı
    float asp = u_res.x / u_res.y;
    vec2 dd = uv - 0.5;
    dd.x *= asp;
    float cs = cos(u_fbrot), sn = sin(u_fbrot);
    dd = mat2(cs, -sn, sn, cs) * dd / u_fbzoom;
    dd.x /= asp;
    vec2 uvp = dd + 0.5 + vec2(gy, -gx) * u_flow;
    vec3 prev = texture(u_prev, uvp).rgb;

    // max harmanlama: yeni kare keskin kalır, izler arkada söner
    vec3 outc = max(prev * u_fb, neu) + flash;
    frag = vec4(outc, 1.0);
}
"""

DISPLAY_FRAG = """#version 330 core
out vec4 frag;
uniform sampler2D u_tex;
uniform vec2  u_res;
uniform float u_time;
uniform float u_aberr;
uniform float u_glow;
uniform float u_master;
uniform vec3  u_bg;
uniform float u_grain;

void main() {
    vec2 uv = gl_FragCoord.xy / u_res;
    vec2 dir = uv - 0.5;
    float ab = u_aberr * (0.4 + dot(dir, dir) * 2.0);
    vec3 col;
    col.r = texture(u_tex, uv + dir * ab).r;
    col.g = texture(u_tex, uv).g;
    col.b = texture(u_tex, uv - dir * ab).b;

    vec3 glow = textureLod(u_tex, uv, 3.0).rgb * 0.5
              + textureLod(u_tex, uv, 5.0).rgb * 0.5;
    col += glow * u_glow;
    col = col * u_master + u_bg;

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = max(mix(vec3(lum), col, 1.3), 0.0);                                  // doygunluk

    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);   // ACES

    float vig = smoothstep(1.15, 0.25, length(dir * vec2(u_res.x / u_res.y, 1.0)));
    col *= mix(0.55, 1.0, vig);
    vec3 p3 = fract(vec3(gl_FragCoord.xy, u_time * 37.0) * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    float n = fract((p3.x + p3.y) * p3.z);
    col += (n - 0.5) * 0.012 * u_grain;
    frag = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# ── Stil ────────────────────────────────────────────────────────────────────

SCALAR_KEYS = ["warp", "scale", "rot_speed", "thresh", "lines", "linefreq",
               "huecycle", "fb", "fbzoom", "fbrot", "flow", "aberr", "glow"]


def random_style(seed=None):
    """Rastgele bir 'dünya': CPPN ağırlıkları + görsel parametreler."""
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = np.random.default_rng(seed)

    W = np.zeros((N_W, 16), np.float32)
    s_in = rng.uniform(1.2, 2.8)
    s_lat = rng.uniform(0.5, 1.4)
    for i in range(4):
        W[i * 3 + 0] = rng.normal(0, s_in, 16)     # koordinat girdisi
        W[i * 3 + 1] = rng.normal(0, s_lat, 16)    # latent 0-3
        W[i * 3 + 2] = rng.normal(0, s_lat, 16)    # latent 4-7
    W[12:28] = rng.normal(0, rng.uniform(0.28, 0.55), (16, 16))
    W[28:44] = rng.normal(0, rng.uniform(0.28, 0.55), (16, 16))
    W[44:48] = rng.normal(0, rng.uniform(0.4, 0.9), (4, 16))
    B = rng.normal(0, 0.5, (N_B, 4)).astype(np.float32)

    acts = np.array([rng.dirichlet([0.6, 0.6, 0.6]) for _ in range(3)], np.float32)

    zoom_dir = rng.choice([-1.0, 1.0, 1.0])
    return {
        "seed": int(seed),
        "W": W, "B": B, "acts": acts,
        "sym": float(rng.choice([0, 0, 0, 3, 4, 5, 6, 8])),
        "warp": rng.uniform(0.0, 0.9),
        "scale": rng.uniform(1.5, 3.5),
        "rot_speed": rng.uniform(-0.12, 0.12),
        "thresh": rng.uniform(0.44, 0.64),
        "lines": 0.0 if rng.random() < 0.25 else rng.uniform(0.4, 1.0),
        "linefreq": rng.uniform(3.0, 12.0),
        "huecycle": rng.uniform(0.004, 0.025),
        "fb": rng.uniform(0.80, 0.92),
        "fbzoom": 1.0 + zoom_dir * rng.uniform(0.001, 0.006),
        "fbrot": rng.uniform(-0.004, 0.004),
        "flow": rng.uniform(0.0, 0.15),
        "aberr": rng.uniform(0.001, 0.006),
        "glow": rng.uniform(0.15, 0.5),
    }


def lerp_style(a, b, t):
    out = {"seed": b["seed"] if t >= 0.5 else a["seed"]}
    out["W"] = a["W"] + (b["W"] - a["W"]) * t
    out["B"] = a["B"] + (b["B"] - a["B"]) * t
    out["acts"] = a["acts"] + (b["acts"] - a["acts"]) * t
    out["sym"] = b["sym"] if t >= 0.5 else a["sym"]   # tam sayı: yarıda geçiş
    for k in SCALAR_KEYS:
        out[k] = a[k] + (b[k] - a[k]) * t
    return out


def save_style(style):
    os.makedirs(STYLE_DIR, exist_ok=True)
    path = os.path.join(STYLE_DIR, f"fav_{style['seed']}.npz")
    np.savez(path, W=style["W"], B=style["B"], acts=style["acts"],
             sym=style["sym"], seed=style["seed"],
             **{k: style[k] for k in SCALAR_KEYS})
    return path


def load_styles():
    styles = []
    for p in glob.glob(os.path.join(STYLE_DIR, "fav_*.npz")):
        try:
            d = np.load(p)
            s = {"W": d["W"].astype(np.float32), "B": d["B"].astype(np.float32),
                 "acts": d["acts"].astype(np.float32), "sym": float(d["sym"]),
                 "seed": int(d["seed"])}
            for k in SCALAR_KEYS:
                s[k] = float(d[k])
            styles.append(s)
        except Exception as e:
            print(f"[Neural] Stil okunamadı {p}: {e}")
    return styles


# ── Rüya modu: embedding → ağ ──────────────────────────────────────────────

class DreamMapper:
    """Ses modelinin embedding'ini doğrudan bir CPPN 'dünyasına' çevirir.

    Sabit tohumlu rastgele bir izdüşüm matrisi kullanılır: aynı müzik anı her
    zaman aynı ağı (aynı görüntüyü) üretir, farklı müzik farklı ağ üretir.
    Embedding önce PCA ile beyazlatılır; en çok varyans taşıyan K yön kullanılır.
    Kararlılık eğitimli modelde bilginin ~%97'si ilk 8 yönde; geri kalanı çoğunlukla
    pencere kaymasından gelen gürültü (titreme yapar).
    """
    SEED = 20261125
    K = 8

    def __init__(self, stats_path=EMB_STATS):
        d = np.load(stats_path)
        self.mu = d["mu"].astype(np.float32)
        self.proj = np.ascontiguousarray(d["white"][:, ::-1][:, :self.K], np.float32)
        rng = np.random.default_rng(self.SEED)
        k = self.K
        g = lambda *shape: rng.normal(0, 1.0 / math.sqrt(k), shape + (k,)).astype(np.float32)

        # Ağırlık izdüşümleri — her blok random_style ile aynı ölçekte
        row_std = np.zeros((N_W, 16), np.float32)
        for i in range(4):
            row_std[i * 3 + 0] = 2.0
            row_std[i * 3 + 0, 8:12] = 0.0     # merkeze uzaklık (length(p)) girdisi kapalı
            row_std[i * 3 + 1] = 0.9
            row_std[i * 3 + 2] = 0.9
        row_std[12:44] = 0.42
        row_std[44:48] = 0.65
        self.P_W = g(N_W * 16) * row_std.reshape(-1, 1)
        self.P_B = g(N_B * 4) * 0.5
        self.P_act = g(9) * 2.5
        self.P_sc = g(len(self.SC))
        self.P_hue = g(4)
        self.P_sv = g(8)

    # (anahtar, alt sınır, üst sınır)
    SC = [("scale", 1.5, 3.5), ("thresh", 0.30, 0.64), ("lines", 0.0, 1.0),
          ("linefreq", 3.0, 12.0), ("aberr", 0.0, 0.005), ("glow", 0.15, 0.5),
          ("gain", 0.3, 1.2), ("hi", 0.0, 1.0)]

    def whiten(self, e):
        return np.clip((np.asarray(e, np.float32) - self.mu) @ self.proj, -3.0, 3.0)

    def style(self, e, z_lat=None):
        import colorsys
        z = self.whiten(e)
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        a = (self.P_act @ z).reshape(3, 3)
        a = np.exp(a - a.max(1, keepdims=True))
        st = {
            "seed": self.SEED,
            "W": (self.P_W @ z).reshape(N_W, 16),
            "B": (self.P_B @ z).reshape(N_B, 4),
            "acts": (a / a.sum(1, keepdims=True)).astype(np.float32),
            # Zaman / gürültü / feedback kaynakları rüya modunda yok
            "sym": 0.0, "rot_speed": 0.0, "fbzoom": 1.0, "fbrot": 0.0,
            "warp": 0.0, "huecycle": 0.0, "fb": 0.0, "flow": 0.0,
        }
        sc = sig(self.P_sc @ z * 1.5)
        for (k, lo, hi), v in zip(self.SC, sc):
            st[k] = float(lo + (hi - lo) * v)
        hues = (0.5 + 0.35 * (self.P_hue @ z)) % 1.0
        sv = sig(self.P_sv @ z * 1.5)
        st["palette"] = [colorsys.hsv_to_rgb(float(hues[i]), 0.5 + 0.5 * float(sv[i]),
                                             0.75 + 0.25 * float(sv[4 + i])) for i in range(4)]
        return st


class EmotionMapper:
    """Duygu tabanlı rüya: embedding → (valence, arousal) → görsel üretici → CPPN dünyası.

    İki öğrenilmiş parça (train/emotion.py, train/visual.py):
      • duygu başlığı : DEAM'deki insan duygu etiketleriyle eğitilmiş MLP
      • görsel üretici: CLIP denetimiyle "bu duygu neye benzer" öğrenmiş MLP
    Müziğin 8 boyutlu latent'i üreticiye ikinci girdi olarak gider: aynı duyguda
    farklı müzik farklı görüntü üretir. Renk / şekil kuralı yok.
    """
    SEED = 20261126
    MOODS = {(-1, -1): "hüzünlü", (-1, 1): "gergin", (1, 1): "neşeli", (1, -1): "huzurlu"}

    def __init__(self, head_path=EMO_HEAD, gen_path=VIS_GEN):
        from scipy.special import erf
        self._erf = erf
        self.h = {k: v.astype(np.float32) for k, v in np.load(head_path).items()}
        self.g = {k: v.astype(np.float32) for k, v in np.load(gen_path).items()}
        self.va = np.zeros(2, np.float32)

    # SC sırası train/visual.py ile aynı olmalı
    SC = [("scale", 1.5, 3.5), ("thresh", 0.30, 0.64), ("lines", 0.0, 1.0),
          ("linefreq", 3.0, 12.0), ("aberr", 0.0, 0.005), ("glow", 0.15, 0.5),
          ("gain", 0.3, 1.2), ("hi", 0.0, 1.0)]

    def emotion(self, e):
        h, gelu = self.h, (lambda x: 0.5 * x * (1 + self._erf(x / math.sqrt(2))))
        x = (np.asarray(e, np.float32) - h["mu"]) / h["sd"]
        x = gelu(x @ h["W1"].T + h["b1"])
        x = gelu(x @ h["W2"].T + h["b2"])
        va = x @ h["W3"].T + h["b3"]
        # Genel müzikteki tahmin dağılımına göre [−1, 1]: ±1.5σ kenarlar
        return np.clip((va - h["p_mean"]) / (1.5 * h["p_std"]), -1.0, 1.0).astype(np.float32)

    def mood(self):
        v, a = self.va
        if abs(v) < 0.2 and abs(a) < 0.2:
            return "nötr"
        return self.MOODS[(1 if v >= 0 else -1, 1 if a >= 0 else -1)]

    def style(self, e, z_lat):
        g = self.g
        self.va = self.emotion(e)
        z = np.zeros(LATENT, np.float32) if z_lat is None else np.clip(np.asarray(z_lat, np.float32), -3, 3)
        silu = lambda x: x / (1.0 + np.exp(-x))
        x = silu(np.concatenate([self.va, z]) @ g["W1"].T + g["b1"])
        x = silu(x @ g["W2"].T + g["b2"])
        o = x @ g["W3"].T + g["b3"]
        i = 0
        W = (o[i:i + N_W * 16] * g["rs"]).reshape(N_W, 16); i += N_W * 16
        B = (o[i:i + N_B * 4] * 0.5).reshape(N_B, 4); i += N_B * 4
        a = o[i:i + 9].reshape(3, 3); i += 9
        a = np.exp(a - a.max(1, keepdims=True))
        sc = 1.0 / (1.0 + np.exp(-o[i:i + len(self.SC)])); i += len(self.SC)
        pal = 1.0 / (1.0 + np.exp(-o[i:i + 12].reshape(4, 3) * 1.5))
        st = {"seed": self.SEED, "W": W.astype(np.float32), "B": B.astype(np.float32),
              "acts": (a / a.sum(1, keepdims=True)).astype(np.float32),
              "sym": 0.0, "rot_speed": 0.0, "fbzoom": 1.0, "fbrot": 0.0,
              "warp": 0.0, "huecycle": 0.0, "fb": 0.0, "flow": 0.0,
              "palette": [tuple(float(c) for c in row) for row in pal]}
        for (k, lo, hi), v in zip(self.SC, sc):
            st[k] = float(lo + (hi - lo) * v)
        return st


# ── Render katmanı ──────────────────────────────────────────────────────────

class NeuralBackground:
    def __init__(self, window, cppn_scale=1.0):
        self.win = window
        self.cppn_scale = cppn_scale

        self.p_cppn = self._program(CPPN_FRAG)
        self.p_comp = self._program(COMPOSITE_FRAG)
        self.p_disp = self._program(DISPLAY_FRAG)
        quad = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
        self.q_cppn = self.p_cppn.vertex_list(6, gl.GL_TRIANGLES, position=("f", quad))
        self.q_comp = self.p_comp.vertex_list(6, gl.GL_TRIANGLES, position=("f", quad))
        self.q_disp = self.p_disp.vertex_list(6, gl.GL_TRIANGLES, position=("f", quad))
        self._loc_W = gl.glGetUniformLocation(self.p_cppn.id, b"u_W")
        self._loc_B = gl.glGetUniformLocation(self.p_cppn.id, b"u_B")

        self._size = (0, 0)
        self._fbos = []

        # Stil durumu
        self.favorites = load_styles()
        self.style_a = self._pick_style()
        self.style_b = self.style_a
        self.morph = 1.0
        self.morph_dur = 5.0
        self.cur = self.style_a
        self.locked = False
        self.style_timer = 0.0
        self.style_dur = random.uniform(22, 38)
        self._want_style = None        # None | "slow" | "fast" — ölçü başını bekleyen stil isteği
        self._want_t = 0.0
        self.time_real = 0.0

        # Animasyon durumu
        self.time = 0.0
        self.rot = 0.0
        self.latent = np.zeros(LATENT, np.float32)
        self._band_avg = np.zeros(LATENT, np.float32)
        self._noise_f = np.random.uniform(0.05, 0.25, LATENT)
        self._noise_p = np.random.uniform(0, math.tau, LATENT)
        self._external = None          # ileride ses modeli buraya yazacak
        self._external_mix = 0.0
        self.energy = 0.0
        self.high = 0.0
        self.beat = 0.0
        self.master = 1.0
        self.palette = [(0.2, 0.4, 1.0), (0.9, 0.1, 0.8), (1.0, 0.5, 0.1), (0.1, 0.9, 0.5)]
        self.bg = (0.02, 0.0, 0.08)

        # Rüya modu
        self.dreaming = False
        self._dream_target = None
        self._dream_cache = None      # (hedef, latent, stil) — aynı girdide MLP tekrar koşmaz
        self.dream = None
        try:
            self.dream = EmotionMapper()
            print("[Neural] Rüya modu: duygu → görsel üretici (DEAM + CLIP)")
        except Exception as e:
            print(f"[Neural] Duygu modeli yok ({e}) — rastgele izdüşüm kullanılacak")
            try:
                self.dream = DreamMapper()
            except Exception as e2:
                print(f"[Neural] Rüya modu kapalı ({os.path.basename(EMB_STATS)} yok: train/emb_stats.py çalıştır) — {e2}")
        print(f"[Neural] Hazır — stil #{self.cur['seed']}  (favori: {len(self.favorites)})")

    # ── GL yardımcıları ──
    @staticmethod
    def _program(frag_src):
        return ShaderProgram(Shader(VERT_SRC, "vertex"), Shader(frag_src, "fragment"))

    @staticmethod
    def _set(prog, name, value):
        try:
            prog[name] = value
        except Exception:
            pass   # sürücü kullanılmayan uniform'u optimize etmiş olabilir

    def _make_target(self, w, h, mip):
        minf = gl.GL_LINEAR_MIPMAP_LINEAR if mip else gl.GL_LINEAR
        tex = pyglet.image.Texture.create(w, h, internalformat=gl.GL_RGBA16F,
                                          min_filter=minf, mag_filter=gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        if mip:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        fbo = pyglet.image.Framebuffer()
        fbo.attach_texture(tex, attachment=gl.GL_COLOR_ATTACHMENT0)
        fbo.bind()
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        fbo.unbind()
        return tex, fbo

    def _ensure_targets(self):
        w, h = self.win.get_framebuffer_size()
        w, h = max(1, w), max(1, h)
        if (w, h) == self._size:
            return
        for tex, fbo in self._fbos:
            fbo.delete()
            tex.delete()
        cw, ch = max(1, int(w * self.cppn_scale)), max(1, int(h * self.cppn_scale))
        self.cppn_t = self._make_target(cw, ch, mip=False)
        self.fb_a = self._make_target(w, h, mip=True)
        self.fb_b = self._make_target(w, h, mip=True)
        self._fbos = [self.cppn_t, self.fb_a, self.fb_b]
        self._size = (w, h)
        self._cppn_size = (cw, ch)

    # ── Stil yönetimi ──
    def _pick_style(self):
        if self.favorites and random.random() < 0.5:
            return random.choice(self.favorites)
        return random_style()

    def new_style(self, fast=False):
        self.style_a = self.cur
        self.style_b = self._pick_style()
        self.morph = 0.0
        self.morph_dur = 1.4 if fast else random.uniform(4.0, 7.0)
        self.style_timer = 0.0
        self.style_dur = random.uniform(22, 38)
        self._want_style = None
        print(f"[Neural] Yeni stil #{self.style_b['seed']}")

    def request_style(self, fast=False):
        """Yeni stil iste; beat kilitliyse bir sonraki ölçü başında uygulanır (bkz. update)."""
        if self._want_style is None:
            self._want_t = self.time_real
        if fast or self._want_style is None:
            self._want_style = "fast" if fast else "slow"

    def save_current(self):
        path = save_style(self.cur)
        self.favorites.append(self.cur)
        print(f"[Neural] Favori kaydedildi: {path}")
        return path

    def on_drop(self):
        if not self.locked:
            self.new_style(fast=True)
        self.beat = min(1.5, self.beat + 1.0)

    def set_external_latent(self, z, mix=1.0):
        """Kendi ses modelimizin ürettiği 8 boyutlu latent. None → devre dışı."""
        if z is None:
            self._external, self._external_mix = None, 0.0
            return
        self._external = np.asarray(z, np.float32)[:LATENT]
        self._external_mix = float(mix)

    def set_dream(self, emb):
        """Modelin 128 boyutlu embedding'i. None → rüya modu kapalı."""
        self._dream_target = None if (emb is None or self.dream is None) else np.asarray(emb, np.float32)

    # ── Her kare ──
    def update(self, dt, bands, rms, beat, palette=None, bg=None, downbeat=False, quantize=False):
        """quantize=True iken stil geçişleri ölçü başına (downbeat) oturur; en fazla 3 sn bekler."""
        self.time_real += dt
        bands = np.asarray(bands, np.float32)[:LATENT]
        if palette is not None:
            self.palette = [tuple(c) for c in palette]
        if bg is not None:
            self.bg = tuple(bg)

        dreaming = self._dream_target is not None
        if self.dreaming and not dreaming:          # rüyadan çıkış: rastgele stile akarak dön
            self.style_a, self.style_b = self.cur, self._pick_style()
            self.morph, self.morph_dur, self.style_timer = 0.0, 3.0, 0.0
        self.dreaming = dreaming

        # Stil zamanlayıcı + morph
        if dreaming:
            # Yumuşatma yok: modelin son çıktısı doğrudan ağ olur.
            # Ses modeli ~kare hızından seyrek güncellenir: girdi aynıysa MLP'yi tekrar koşma
            # (deterministik → aynı çıktı; 60 FPS döngüde gereksiz numpy / dict üretimi olmasın)
            dc = self._dream_cache
            ext = self._external
            if not (dc is not None and np.array_equal(dc[0], self._dream_target)
                    and ((ext is None and dc[1] is None)
                         or (ext is not None and dc[1] is not None and np.array_equal(dc[1], ext)))):
                st = self.dream.style(self._dream_target, ext)
                self._dream_cache = (self._dream_target.copy(), None if ext is None else ext.copy(), st)
            self.cur = self._dream_cache[2]
            self.palette = self.cur["palette"]
            c0 = self.palette[0]
            self.bg = (c0[0] * 0.03, c0[1] * 0.03, c0[2] * 0.03)
        elif not self.locked:
            self.style_timer += dt
            if self.style_timer >= self.style_dur and self.morph >= 1.0 and self._want_style is None:
                self.request_style()
        if self._want_style is not None:
            if dreaming or self.locked:
                self._want_style = None
            elif not quantize or downbeat or self.time_real - self._want_t > 3.0:
                self.new_style(fast=self._want_style == "fast")
        if dreaming:
            pass
        elif self.morph < 1.0:
            self.morph = min(1.0, self.morph + dt / self.morph_dur)
            t = self.morph * self.morph * (3 - 2 * self.morph)
            self.cur = lerp_style(self.style_a, self.style_b, t)
        else:
            self.cur = self.style_b

        # Enerji / tiz / beat yumuşatma
        e = float(np.clip(np.mean(bands) * 1.2, 0, 1.5))
        self.energy += (e - self.energy) * min(1.0, dt * (8 if e > self.energy else 2.5))
        hi = float(np.clip(np.mean(bands[5:8]) * 1.6, 0, 1.5))
        self.high += (hi - self.high) * min(1.0, dt * (10 if hi > self.high else 3))
        self.beat = max(self.beat * (1 - dt * 4.5), float(beat))

        # Müziğe bağlı zaman akışı (rüya modunda kullanılmaz)
        self.time += dt * (0.5 + self.energy * 1.3)
        self.rot += dt * self.cur["rot_speed"] * (0.6 + self.energy)

        # Latent: yavaş gürültü + bant sapmaları (ses modelinin yerine geçici)
        self._band_avg += (bands - self._band_avg) * min(1.0, dt * 0.4)
        noise = 0.7 * np.sin(self.time * self._noise_f * 2.0 + self._noise_p)
        audio = np.clip((bands - self._band_avg) * 2.5, -1.2, 1.2)
        if dreaming:
            if self._external is not None:
                self.latent = np.clip(self._external, -3.0, 3.0)  # sadece model, yumuşatma yok
            return
        if self._external is not None:
            # Model = şarkının karakteri (yavaş), bant sapmaları = an (hızlı)
            m = self._external_mix
            z = self._external * m + noise * (1 - 0.7 * m) + audio * (1 - 0.4 * m)
        else:
            z = noise + audio
        self.latent += (z.astype(np.float32) - self.latent) * min(1.0, dt * 6)

    def draw(self):
        self._ensure_targets()
        w, h = self._size
        cw, ch = self._cppn_size
        s = self.cur
        gl.glDisable(gl.GL_BLEND)

        # 1) CPPN
        tex_c, fbo_c = self.cppn_t
        fbo_c.bind()
        gl.glViewport(0, 0, cw, ch)
        p = self.p_cppn
        p.use()
        W = np.ascontiguousarray(s["W"], np.float32)
        B = np.ascontiguousarray(s["B"], np.float32)
        gl.glUniformMatrix4fv(self._loc_W, N_W, gl.GL_FALSE,
                              W.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))
        gl.glUniform4fv(self._loc_B, N_B, B.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))
        dream = self.dreaming
        t = 0.0 if dream else self.time
        self._set(p, "u_res", (float(cw), float(ch)))
        self._set(p, "u_time", t)
        self._set(p, "u_z0", tuple(float(v) for v in self.latent[:4]))
        self._set(p, "u_z1", tuple(float(v) for v in self.latent[4:8]))
        for i in range(3):
            self._set(p, f"u_act{i+1}", tuple(float(v) for v in s["acts"][i]))
        self._set(p, "u_sym", s["sym"])
        self._set(p, "u_warp", 0.0 if dream else s["warp"] * (0.7 + self.energy * 0.6))
        self._set(p, "u_scale", s["scale"])
        self._set(p, "u_rot", 0.0 if self.dreaming else self.rot)
        self.q_cppn.draw(gl.GL_TRIANGLES)
        p.stop()
        fbo_c.unbind()

        # 2) Composite + feedback (a → b)
        tex_a, _ = self.fb_a
        tex_b, fbo_b = self.fb_b
        fbo_b.bind()
        gl.glViewport(0, 0, w, h)
        p = self.p_comp
        p.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_c.id)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_a.id)
        self._set(p, "u_cppn", 0)
        self._set(p, "u_prev", 1)
        self._set(p, "u_res", (float(w), float(h)))
        self._set(p, "u_time", t)
        for i in range(4):
            self._set(p, f"u_c{i}", tuple(float(v) for v in self.palette[i % len(self.palette)]))
        self._set(p, "u_fb", s["fb"])
        self._set(p, "u_fbzoom", 1.0 + (s["fbzoom"] - 1.0) * (0.6 + self.energy * 1.6))
        self._set(p, "u_fbrot", s["fbrot"] * (0.6 + self.energy))
        self._set(p, "u_flow", s["flow"])
        self._set(p, "u_energy", s["gain"] if dream else self.energy)
        self._set(p, "u_beat", 0.0 if dream else self.beat)
        self._set(p, "u_high", s["hi"] if dream else self.high)
        self._set(p, "u_thresh", s["thresh"])
        self._set(p, "u_lines", s["lines"])
        self._set(p, "u_linefreq", s["linefreq"])
        self._set(p, "u_huecycle", s["huecycle"])
        self.q_comp.draw(gl.GL_TRIANGLES)
        p.stop()
        fbo_b.unbind()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_b.id)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # 3) Ekrana
        gl.glViewport(0, 0, w, h)
        p = self.p_disp
        p.use()
        self._set(p, "u_tex", 0)
        self._set(p, "u_res", (float(w), float(h)))
        self._set(p, "u_time", t)
        self._set(p, "u_aberr", s["aberr"] if dream else s["aberr"] * (1.0 + self.beat * 1.5))
        self._set(p, "u_glow", s["glow"])
        self._set(p, "u_master", self.master)
        self._set(p, "u_bg", tuple(float(v) for v in self.bg))
        self._set(p, "u_grain", 0.0 if dream else 1.0)
        self.q_disp.draw(gl.GL_TRIANGLES)
        p.stop()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.fb_a, self.fb_b = self.fb_b, self.fb_a
        gl.glEnable(gl.GL_BLEND)
