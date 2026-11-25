"""
Ses dosyalarını log-mel özelliklere çevirir (ses kaydedilmez, sadece özellikler).

Çıktı formatı (bir klasör):
  feats.bin   float16, satır = zaman karesi, sütun = 64 mel bandı
  index.json  {"n_mels": 64, "frames": toplam, "tracks": [{"name", "start", "length"}, ...]}

Kullanım:
  python train/prepare.py --zip D:/ai-visualizer-work/data/fma_small.zip --out D:/ai-visualizer-work/features/fma
  python train/prepare.py --dir D:/muziklerim --out D:/ai-visualizer-work/features/benim
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio_features import SR, N_MELS, FRAMES, logmel  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac")
MIN_SECONDS = 5.0


def _decode(data_or_path):
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0" if isinstance(data_or_path, bytes) else data_or_path,
           "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    p = subprocess.run(cmd, input=data_or_path if isinstance(data_or_path, bytes) else None,
                       capture_output=True, timeout=120)
    if not p.stdout:
        return None
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


_ZIP = None


def _init_worker(zip_path):
    global _ZIP
    if zip_path:
        _ZIP = zipfile.ZipFile(zip_path)   # her worker zip'i kendisi açar → bellekte birikme yok


def _work(item):
    name, payload = item
    try:
        if payload is None:
            payload = _ZIP.read(name)
        audio = _decode(payload)
        if audio is None or len(audio) < MIN_SECONDS * SR:
            return name, None
        return name, logmel(audio).T.astype(np.float16)   # (T, 64)
    except Exception:
        return name, None


def _iter_zip(path):
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist()
                 if i.filename.lower().endswith(AUDIO_EXT) and i.file_size > 0]
    for n in names:
        yield n, None


def _iter_dir(path):
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(AUDIO_EXT):
                full = os.path.join(root, f)
                yield os.path.relpath(full, path), full


def _count(args):
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            return sum(1 for i in z.infolist() if i.filename.lower().endswith(AUDIO_EXT))
    return sum(1 for _ in _iter_dir(args.dir))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip")
    src.add_argument("--dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, "feats.bin")
    idx_path = os.path.join(args.out, "index.json")
    total = _count(args)
    items = _iter_zip(args.zip) if args.zip else _iter_dir(args.dir)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    tracks, start, skipped = [], 0, 0
    with open(bin_path, "wb") as fbin, Pool(args.workers, _init_worker, (args.zip,)) as pool:
        for name, feats in tqdm(pool.imap_unordered(_work, items, chunksize=4), total=total, unit="parça"):
            if feats is None or len(feats) < FRAMES + 64:
                skipped += 1
                continue
            fbin.write(feats.tobytes())
            tracks.append({"name": name, "start": start, "length": int(len(feats))})
            start += len(feats)

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"n_mels": N_MELS, "frames": start, "tracks": tracks}, f)
    hours = start * 256 / SR / 3600
    print(f"Bitti: {len(tracks)} parça, {hours:.1f} saat, atlanan {skipped} → {args.out}")


if __name__ == "__main__":
    main()
