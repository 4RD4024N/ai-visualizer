durum: bitti
son_güncelleme: 2026-11-26T01:11:31Z
pr: https://github.com/4RD4024N/ai-visualizer/pull/1

# Gece ilerleme notları (perf/lyric-latency)

Sonraki oturumlar: durum "bitti" — yalnızca PR #1'in CI'ını kontrol et, kırmızıysa düzelt.

## Yapılanlar (oturum 1, 01:00–01:30 UTC)
- latency.py: LatencyTracker (heard/window/whisper/accept/scene/frame/draw), [Gecikme] logu, HUD değeri.
- lyrics.py: translate-first (Türkçe: çeviri tek generate → satır hemen yayınlanır; orijinal yalnız yeni
  satırlarda sonra, kara liste / düşük logprob → satır geri çekilir), pencere 4 sn / adım 0,5 sn sürekli
  döngü, encoder yeniden kullanımı (fallback'li), GPU öznitelik çıkarımı (fallback'li), MAX_TOKENS 40,
  duran ses akışında aynı pencereyi atlama, on_line geri çağrısı.
- diffusion_layer.py: scene_due() + is_new_line() (söz önceliği, LYRIC_MIN_SCENE 0,3), wake Event,
  ilk söz karesi etiketi (get_frame_info), diff_fast (channels_last+cudnn.benchmark+inference_mode),
  diff_compile (varsayılan kapalı, hata → eski UNet), yüklemede ısınma kareleri.
- visualizer.py: tracker bağlandı; söz karesinde lyric_fade (0,06 sn); HUD söz→ekran.
- neural_bg.py: rüya modunda girdi değişmediyse stil MLP'si tekrar koşmuyor.
- config.py yeni anahtarlar; README (Diffusion layer, Settings, Development); tools/bench_latency.py
  (--wav, --legacy, --compile; wav yoksa uyarı + sadece SD ölçümü).
- tests/test_latency.py (18 test), CI lint listesine latency/lyrics/diffusion_layer eklendi. 58 test yeşil.
- PR #1 açıklaması Türkçe yazıldı (tahmin bütçesi, komutlar, riskler / config anahtarları).

## Ölçüm / tahmin notları
- Hiçbir süre GPU'da ölçülmedi. Tahmini yeni bütçe: bekleme ~0,25–0,4 + Whisper ~0,5–0,9 + sahne
  ~0,01–0,1 + kare ~0,05–0,09 + çizim ~0,08 ≈ 0,9–1,6 sn (eski ≈ 2,5–4 sn).
- Kullanıcı sabah: tools/bench_latency.py --wav <şarkı> (ve --legacy) ile doğrulamalı.

## Olası sonraki işler (kullanıcı ölçümünden sonra)
- Bench Whisper'ı hâlâ >1 sn gösterirse: SD'yi Whisper çalışırken kısa süre yavaşlatma seçeneği,
  ya da Whisper decoder için statik KV cache.
- visualizer._draw_diffusion: her karede yeni GL dokusu yerine doku havuzu (GL'siz test edilemedi).
