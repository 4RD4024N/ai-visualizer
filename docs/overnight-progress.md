durum: çalışıyor
son_güncelleme: 2026-11-26T01:08:11Z
pr: https://github.com/4RD4024N/ai-visualizer/pull/1

# Gece ilerleme notları (perf/lyric-latency)

## Yapılanlar (oturum 1)
- latency.py: LatencyTracker (heard/window/whisper/accept/scene/frame/draw), [Gecikme] logu, HUD.
- lyrics.py: translate-first (Türkçe tek generate → satır hemen yayınlanır, orijinal sonra),
  pencere 4 sn / adım 0,5 sn sürekli döngü, encoder yeniden kullanımı (fallback'li), GPU öznitelik
  çıkarımı (fallback'li), MAX_TOKENS 40, duran ses akışında aynı pencereyi atlama, on_line geri çağrısı.
- diffusion_layer.py: scene_due() (söz önceliği, LYRIC_MIN_SCENE 0,3), wake Event, ilk söz karesi
  etiketi (get_frame_info), diff_fast (channels_last+cudnn.benchmark+inference_mode), diff_compile, ısınma.
- visualizer.py: tracker bağlandı, söz karesinde lyric_fade (0,06 sn) geçiş, HUD söz→ekran.
- config.py: yeni anahtarlar. tests/test_latency.py (15 test). CI lint listesine yeni modüller.
- PR #1 açıldı.

## Sıradakiler
- (bitti) tools/bench_latency.py
- README güncellemesi, PR açıklaması
- (isteğe bağlı) 60 FPS ana döngüde mikro optimizasyonlar
