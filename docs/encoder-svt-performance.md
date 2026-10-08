# Encoder comparison with SVT-AV1

This October 8, 2026 diagnostic compares the final Go 1.27.2 SIMD build with
SVT-AV1 4.0.1 preset 13 on an Apple M4 Max. Go uses one encoder thread and
GOMAXPROCS=1; SVT uses LP=1, with measured CPU/wall parallelism close to one.
LP is an SVT parallelism level, not a literal thread count. Both use low-delay
CBR at 60 fps with no lookahead, scene-cut detection or temporal filtering.

The two clips are the first 120 frames of Xiph FourPeople and Johnny, decoded
from their public WebM files to 1280×720 8-bit I420. Each is only two seconds
long and both are conference content. This is not a publish-quality corpus or
a general ranking of encoder speed. The host is not an isolated benchmark
machine; small timing differences need confirmation.

The table reports median process-inclusive encode throughput from three
measured runs after a separate warmup, with shuffled encoder/rate ordering.
Raw loading, encoder setup, output writes and process startup are included;
reference decode and quality measurement are outside timing. All runs completed
successfully. Quality is measured against the raw input with FFmpeg/libdav1d.
The two encoders have different PSNR/SSIM tradeoffs, so equal target bitrate
does not establish equal visual quality.

| Clip | Target Mbps | Go fps | SVT fps | PSNR dB, Go / SVT | SSIM, Go / SVT |
| --- | ---: | ---: | ---: | ---: | ---: |
| FourPeople | 3 | 98.39 | 119.57 | 43.6694 / 43.4732 | 0.9841 / 0.9853 |
| FourPeople | 6 | 86.15 | 106.46 | 45.4854 / 45.2783 | 0.9868 / 0.9887 |
| FourPeople | 9 | 79.83 | 90.14 | 46.3141 / 45.9792 | 0.9877 / 0.9898 |
| FourPeople | 12 | 72.58 | 91.92 | 47.0016 / 46.4360 | 0.9890 / 0.9903 |
| Johnny | 3 | 96.73 | 126.79 | 45.2945 / 44.7765 | 0.9863 / 0.9866 |
| Johnny | 6 | 85.40 | 103.64 | 46.9032 / 46.2228 | 0.9890 / 0.9896 |
| Johnny | 9 | 72.66 | 88.48 | 47.7472 / 46.8638 | 0.9901 / 0.9907 |
| Johnny | 12 | 69.57 | 86.22 | 48.4096 / 47.2124 | 0.9911 / 0.9912 |

Go takes approximately 13–31% more encode time in these samples. Go has higher
PSNR, while SVT has higher SSIM; neither is an equal-quality speed comparison.
The final Go encoded and decoded SHA-256 values match baseline `855b80f4` at
all eight clip/rate points, including across the Go 1.27.1 to 1.27.2 update.

## Work removed

The encoder retains the selected MDS0 luma predictor instead of rebuilding it
before residual formation. It tracks which motion vector owns the retained
pixels, restores overwritten cached winners, and reuses them only when they
match the final coded interpolation filters. Fused and split encoder paths use
the same rule. Direct tests cover ME, fresh-candidate and duplicate-vector
winners; serial and four-thread tests pin baseline bitstream and reconstruction
hashes.

The byte-residual transform path now uses its proven [-255,255] range instead
of scanning every residual again. The checked full-int16 API and CPU feature
gates are unchanged. An earlier matched Go 1.27.1 trial of these two changes
measured 0–7.6% less process-inclusive encode time across these eight points
(median reduction about 4.3%); the final table above is a separate Go 1.27.2
measurement, not a cross-version speedup claim. The final SAD SIMD exact-window
fix also avoids forming unused pointers past the last row.

## Reproduce

Use `cmd/qualitybench` with the official Go release and `GOEXPERIMENT=simd`.
A manifest records raw input hashes and the following source URLs:

- [FourPeople source](https://media.xiph.org/video/derf/webm/FourPeople_1280x720_60.webm), raw 120-frame SHA-256 `a022f06b5751735dc9f21d4706a73b6ccbdc4706ac34da4789b5891d48f82f17`.
- [Johnny source](https://media.xiph.org/video/derf/webm/Johnny_1280x720_60.webm), raw 120-frame SHA-256 `c11fc6faff36f2daecc4b4b3bc522de118d31e97af3ca0ea42061fefc316b2cf`.

SVT executable SHA-256:
`db044194faf94d553c2509c8e033ecd05a373ba54d919d0fb49a3a13149d1687`.
Use the existing harness's metadata JSON to retain exact external commands,
versions, binary/input/output hashes, and all individual samples.

~~~sh
qualitybench -manifest /path/to/manifest.csv \
  -encoders goav1,svt-av1 -require-encoders all \
  -bitrates 3000000,6000000,9000000,12000000 \
  -fps 60 -layers 1 -tiles 0 -golden 0 -keyint 0 \
  -gomaxprocs 1 -goav1-max-threads 1 -goav1-effort 0 \
  -goav1-scene-cut=false -goav1-process-timing=true -timing-mode e2e \
  -svt-preset 13 -svt-lp 1 -svt-asm max \
  -svt-bin /path/to/SvtAv1EncApp -go-bin /path/to/official/go \
  -ffmpeg-bin /path/to/ffmpeg -ffmpeg-av1-decoder libdav1d \
  -runs 3 -warmup-runs 1 -run-order shuffle -shuffle-seed 1 \
  -require-metrics psnr,ssim -csv results.csv -summary-csv summary.csv \
  -stats-csv stats.csv -frame-metrics-csv frames.csv -metadata-json metadata.json
~~~

The recorded run was split into one-clip, two-bitrate chunks to bound temporary
raw-frame disk usage. It did not use publish mode or claim the publish corpus
gates passed. For publication, use the larger, diverse corpus and controlled
machine requirements in [quality and parity](quality-and-parity.md).
