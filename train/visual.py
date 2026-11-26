"""
Görsel üretici: (duygu, müzik karakteri) → CPPN dünyası.  CLIP denetimiyle eğitilir.

Girdi : va (valence, arousal ∈ [−1, 1], duygu başlığından)
        z  (8 boyutlu ses latent'i, müziğin karakteri → aynı duygu içinde çeşitlilik)
Çıktı : shader'ın CPPN ağırlıkları, aktivasyon karışımı, görsel parametreler, palet.

Renkleri / şekilleri kimse elle seçmez. Eğitim sırasında üretilen görüntüleri
CLIP (milyonlarca görüntü–metin çiftinden öğrenmiş model) puanlar: görüntü,
hedef duyguyu ("hüzünlü, melankolik soyut resim" ↔ "neşeli, coşkulu soyut
resim" …) ne kadar anlatıyor? Hedef metin va'ya göre dört duygu çapası
arasında sürekli karışır. Çeşitlilik terimi aynı duyguda farklı müziklerin
farklı görüntü üretmesini sağlar.

Renderer, neural_bg.py'deki CPPN + composite + display shader'larının
türevlenebilir torch kopyasıdır (rüya modundaki hali: zaman/feedback yok).

Kullanım:
  python train/visual.py --steps 2500
Çıktı: models/visual_gen.npz (numpy MLP — canlıda torch/CLIP gerekmez),
       models/visual_gen_preview.png
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("AIVIS_WORK") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")
os.environ["HF_HOME"] = os.path.join(WORK, "hf")   # sistem HF_HOME'u başka projeyi gösterebilir

N_W, N_B, LAT = 48, 13, 8

# Dört duygu çapası (valence, arousal işaretleri) — her biri birkaç ifadenin ortalaması
ANCHORS = {
    (-1, -1): ["a sad, melancholic abstract painting", "sorrowful lonely grief, abstract art",
               "a gloomy, mournful, heartbroken abstract image"],
    (-1, +1): ["an angry, aggressive abstract painting", "chaotic violent tension, abstract art",
               "a furious, menacing, intense abstract image"],
    (+1, +1): ["a joyful, happy abstract painting", "euphoric cheerful energy, abstract art",
               "a bright, exciting, celebratory abstract image"],
    (+1, -1): ["a calm, peaceful abstract painting", "serene gentle relaxation, abstract art",
               "a tranquil, dreamy, soothing abstract image"],
}
QUAD = list(ANCHORS.keys())

# (anahtar, alt, üst) — DreamMapper ile aynı anlamlar
SC = [("scale", 1.5, 3.5), ("thresh", 0.30, 0.64), ("lines", 0.0, 1.0),
      ("linefreq", 3.0, 12.0), ("aberr", 0.0, 0.005), ("glow", 0.15, 0.5),
      ("gain", 0.3, 1.2), ("hi", 0.0, 1.0)]


def row_std():
    rs = np.zeros((N_W, 16), np.float32)
    for i in range(4):
        rs[i * 3 + 0] = 2.0
        rs[i * 3 + 0, 8:12] = 0.0          # merkeze uzaklık girdisi kapalı (kaleydoskop eğilimi yok)
        rs[i * 3 + 1] = 0.9
        rs[i * 3 + 2] = 0.9
    rs[12:44] = 0.42
    rs[44:48] = 0.65
    return rs


class Generator(nn.Module):
    N_OUT = N_W * 16 + N_B * 4 + 9 + len(SC) + 12

    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + LAT, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, self.N_OUT))
        self.register_buffer("rs", torch.from_numpy(row_std()).reshape(-1))
        with torch.no_grad():                                   # başlangıç ≈ rastgele CPPN
            self.net[-1].weight.mul_(0.3)
            self.net[-1].bias.normal_(0, 1.0)

    def forward(self, va, z):
        o = self.net(torch.cat([va, z], 1))
        i = 0
        W = o[:, i:i + N_W * 16] * self.rs; i += N_W * 16
        B = o[:, i:i + N_B * 4] * 0.5; i += N_B * 4
        acts = torch.softmax(o[:, i:i + 9].reshape(-1, 3, 3), -1); i += 9
        sc = torch.sigmoid(o[:, i:i + len(SC)]); i += len(SC)
        pal = torch.sigmoid(o[:, i:i + 12].reshape(-1, 4, 3) * 1.5)
        scal = {k: lo + (hi - lo) * sc[:, n] for n, (k, lo, hi) in enumerate(SC)}
        return {"W": W.reshape(-1, N_W, 16), "B": B.reshape(-1, N_B, 4), "acts": acts, "pal": pal, **scal}


# ── Türevlenebilir renderer (shader ile birebir) ─────────────────────────────

def _sstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _act(v, w):                       # v: (B, P, 4·k), w: (B, 3)
    w = w[:, None, None, :]
    return w[..., 0] * torch.sin(v) + w[..., 1] * torch.tanh(v) + w[..., 2] * (torch.exp(-v * v) * 2 - 1)


def _block_mat(W, base, n_in, n_out, stride):
    """mat4 blokları (GL column-major) → (B, 4·n_in, 4·n_out) matris.
    Blok (i çıkış, j giriş) = W[base + i*stride + j]; out[r] = Σ_c M[c*4+r]·x[c]."""
    Bn = W.shape[0]
    M = W.new_zeros(Bn, 4 * n_in, 4 * n_out)
    for i in range(n_out):
        for j in range(n_in):
            M[:, j * 4:(j + 1) * 4, i * 4:(i + 1) * 4] = W[:, base + i * stride + j].reshape(Bn, 4, 4)
    return M


def render(st, z, H=224, Wd=400):
    Bn, dev = z.shape[0], z.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(Wd, device=dev), indexing="ij")
    p = torch.stack([(xs + 0.5 - Wd / 2) / H, (ys + 0.5 - H / 2) / H], -1).reshape(1, -1, 2)
    p = p * st["scale"][:, None, None]
    P = p.shape[1]
    x = torch.cat([p, p.new_zeros(Bn, P, 2), z[:, None, :].expand(Bn, P, LAT)], -1)   # (B, P, 12)
    W, Bb, A = st["W"], st["B"], st["acts"]
    h = _act(x @ _block_mat(W, 0, 3, 4, 3) + Bb[:, None, 0:4].reshape(Bn, 1, 16), A[:, 0])
    g = _act(h @ _block_mat(W, 12, 4, 4, 4) + Bb[:, None, 4:8].reshape(Bn, 1, 16), A[:, 1])
    h = _act(g @ _block_mat(W, 28, 4, 4, 4) + Bb[:, None, 8:12].reshape(Bn, 1, 16), A[:, 2])
    M4 = W[:, 44:48].reshape(Bn, 4, 4, 4).reshape(Bn, 16, 4)        # (j,c) × r
    o = h @ M4 + Bb[:, None, 12]
    c = (0.5 + 0.5 * torch.tanh(o)).reshape(Bn, H, Wd, 4)

    # Composite (rüya modu: huecycle/feedback/beat yok)
    pal = st["pal"]
    v = _sstep(0.05, 0.95, c[..., 0])
    t = torch.frac(v * 0.75 + c[..., 2] * 0.5) * 4.0
    i0 = torch.floor(t).long().clamp(0, 3)
    f = _sstep(0.0, 1.0, t - torch.floor(t))[..., None]
    gi = lambda idx: torch.gather(pal, 1, idx.reshape(Bn, -1, 1).expand(-1, -1, 3)).reshape(Bn, H, Wd, 3)
    col = gi(i0) * (1 - f) + gi((i0 + 1) % 4) * f
    col = col * (0.35 + 0.65 * _sstep(0.0, 0.7, v))[..., None]
    gain, hi = st["gain"][:, None, None], st["hi"][:, None, None]
    th = st["thresh"][:, None, None] - gain * 0.18
    mask = _sstep(th - 0.07, th + 0.07, c[..., 1])
    lf = c[..., 3] * st["linefreq"][:, None, None]
    dx = F.pad(lf[:, :, 1:] - lf[:, :, :-1], (0, 1)).abs()
    dy = F.pad(lf[:, 1:, :] - lf[:, :-1, :], (0, 0, 0, 1)).abs()
    fw = ((dx + dy) * (H / 720.0)).clamp(max=0.25)                 # ekran çözünürlüğündeki fwidth
    d = (torch.frac(lf) - 0.5).abs()
    line = 1 - _sstep(torch.zeros_like(fw), fw * 1.4 + 0.004, d)
    shade = 0.3 + 0.7 * _sstep(th, th + 0.4, c[..., 1])
    neu = col * (mask * shade * (0.35 + 0.8 * gain))[..., None] \
        + col * (line * st["lines"][:, None, None] * (0.45 + 1.4 * hi))[..., None]

    # Display: glow + arka plan + doygunluk + ACES + vinyet
    img = neu.permute(0, 3, 1, 2)
    glow = 0.5 * F.interpolate(F.avg_pool2d(img, 8), size=(H, Wd), mode="bilinear", align_corners=False) \
         + 0.5 * F.interpolate(F.avg_pool2d(img, 32), size=(H, Wd), mode="bilinear", align_corners=False)
    img = img + glow * st["glow"][:, None, None, None] + pal[:, 0, :, None, None] * 0.03
    lum = (img * torch.tensor([0.2126, 0.7152, 0.0722], device=dev)[None, :, None, None]).sum(1, keepdim=True)
    img = (lum + (img - lum) * 1.3).clamp(min=0)
    img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)
    vx = (xs + 0.5 - Wd / 2) / H
    vy = (ys + 0.5 - H / 2) / H
    vig = _sstep(torch.tensor(1.15), torch.tensor(0.25), torch.sqrt(vx ** 2 + vy ** 2))
    img = img * (0.55 + 0.45 * vig)[None, None]
    return img.clamp(0, 1)                                          # (B, 3, H, Wd)


# ── Eğitim ──────────────────────────────────────────────────────────────────

def quad_weights(va):
    """va ∈ [−1,1]² → dört çapanın bilineer ağırlıkları (QUAD sırasıyla)."""
    v, a = (va[:, 0] + 1) / 2, (va[:, 1] + 1) / 2
    ws = []
    for (sv, sa) in QUAD:
        ws.append((v if sv > 0 else 1 - v) * (a if sa > 0 else 1 - a))
    return torch.stack(ws, 1)


def cutouts(img, n, size=224):
    B, _, H, Wd = img.shape
    out = []
    for _ in range(n):
        s = int(np.random.uniform(0.6, 1.0) * H)
        y0 = np.random.randint(0, H - s + 1)
        x0 = np.random.randint(0, Wd - s + 1)
        c = img[:, :, y0:y0 + s, x0:x0 + s]
        if np.random.rand() < 0.5:
            c = c.flip(3)
        out.append(F.interpolate(c, size=(size, size), mode="bilinear", align_corners=False))
    return torch.stack(out, 1)                                      # (B, n, 3, 224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--cuts", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--div", type=float, default=0.3, help="çeşitlilik ağırlığı")
    ap.add_argument("--clip", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")

    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(args.clip, pretrained=args.pretrained, device=dev)
    clip.eval().requires_grad_(False)
    tok = open_clip.get_tokenizer(args.clip)
    with torch.no_grad():
        anc = []
        for q in QUAD:
            t = F.normalize(clip.encode_text(tok(ANCHORS[q]).to(dev)).float(), dim=-1).mean(0)
            anc.append(F.normalize(t, dim=-1))
        anc = torch.stack(anc)                                      # (4, D)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[None, :, None, None]

    def embed(img):
        B, n = img.shape[:2]
        x = (img.reshape(B * n, 3, 224, 224) - mean) / std
        return F.normalize(clip.encode_image(x).float(), dim=-1).reshape(B, n, -1)

    G = Generator().to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    print(f"Üretici: {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M parametre · CLIP {args.clip}")
    t0 = time.time()
    half = args.batch // 2
    for step in range(args.steps):
        # Yarım batch: farklı duygular; her duygu iki farklı müzik karakteriyle (çeşitlilik çifti)
        va = torch.rand(half, 2, device=dev) * 2 - 1
        va = torch.cat([va, va])
        z = torch.randn(args.batch, LAT, device=dev) * 0.9
        st = G(va, z)
        img = render(st, z)
        e = embed(cutouts(img, args.cuts))                          # (B, n, D)
        w = quad_weights(va)                                        # (B, 4)
        target = F.normalize(w @ anc, dim=-1)
        sim = (e * target[:, None]).sum(-1).mean(1)                 # hedef metne benzerlik
        logits = 100.0 * e.mean(1) @ anc.T                          # dört çapaya göre sınıflama
        l_emo = -(w * F.log_softmax(logits, -1)).sum(1).mean()
        em = F.normalize(e.mean(1), dim=-1)
        same = (em[:half] * em[half:]).sum(-1)                      # aynı duygu, farklı müzik
        l_div = F.relu(same - 0.8).mean()
        loss = -sim.mean() * 10 + l_emo + args.div * 10 * l_div
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (logits.argmax(1) == w.argmax(1)).float().mean().item()
            print(f"adım {step + 1:5d}/{args.steps}  benzerlik {sim.mean().item():.3f}  duygu-isabet {acc:.2f}  "
                  f"aynı-duygu-benzerliği {same.mean().item():.3f}  {(time.time() - t0) / 60:.1f} dk", flush=True)

    # ── Kaydet (numpy MLP) ──
    sd = {k: v.detach().cpu().numpy() for k, v in G.state_dict().items()}
    out = os.path.join(ROOT, "models", "visual_gen.npz")
    np.savez(out, W1=sd["net.0.weight"], b1=sd["net.0.bias"], W2=sd["net.2.weight"], b2=sd["net.2.bias"],
             W3=sd["net.4.weight"], b3=sd["net.4.bias"], rs=sd["rs"])
    print(f"Kaydedildi: {out}")

    # ── Önizleme: satır = duygu çapası, sütun = farklı müzik karakteri ──
    with torch.no_grad():
        rows = []
        zz = torch.randn(4, LAT, device=dev) * 0.9
        for q in QUAD + [(0, 0)]:
            va = torch.tensor([q], device=dev, dtype=torch.float32).expand(4, 2) * 0.8
            rows.append(torch.cat(list(render(G(va, zz), zz, 180, 320)), 2))
        grid = (torch.cat(rows, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    from PIL import Image
    pth = os.path.join(ROOT, "models", "visual_gen_preview.png")
    Image.fromarray(grid).save(pth)
    print(f"Önizleme: {pth}  (satırlar: hüzünlü, öfkeli, neşeli, huzurlu, nötr)")


if __name__ == "__main__":
    main()
