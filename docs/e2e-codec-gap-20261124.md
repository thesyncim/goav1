# End-to-end codec gap: November 24, 2026

Measured at `5ceb50f0` (main after #339) with official Go 1.27.4 and
`GOEXPERIMENT=simd` on Apple M4 Max (12 P + 4 E cores, 64 GB) / macOS 26.3.1,
against dav1d 1.5.4 (Homebrew) and SVT-AV1 4.1.0 built from the vendored
source (SHA-256 `b4d8c5ed...`). Every Go binary was built with
`-pgo=cmd/qualitybench/default.pgo`. PRs #342 (itx dispatch, coefficient
clears) and #343 (intra edge prep) landed after this measurement point and
are **not** included. This supersedes the [November 23 report](e2e-codec-gap-20261123.md).

Process-inclusive CLI runs under `/usr/bin/time -l`, with no shell wrapper
(the command is the measured process), both decoders writing YUV to
`/dev/null`, `GOMAXPROCS` equal to the worker count. Each number is the
minimum of five interleaved repetitions. Wall time is taken from a monotonic
clock around the process, so it includes process start-up. Before each
repetition the harness waited, up to a cap, for a one-minute load average of 10
or less. Other agents were running builds and tests during the session. Of
1,740 recorded runs, 1,728 started at a load of 5.4 to 10.0 and 12 at 10.0 to
10.4. Blocks whose first pass ran above 10.5 were discarded and re-run in a
quiet window.

Compare cycles, not seconds, with earlier reports. dav1d, the same binary in
both sessions, used 68.6G corpus cycles on November 23 and 69.0G now, but
17.2 CPU-seconds now against 25.8 then: cycles per CPU-second, derived from the
two reports, are about 4.0 G now and about 2.7 G then. Cycles include kernel
time.

| Measure | 20261123 (`9451293c`) | 20261124 (`5ceb50f0`) |
| --- | ---: | ---: |
| Decoder, 42-stream corpus, 1 thread, cycles: Go / dav1d | 1.322× (CPU 1.329×) | **1.206×** |
| Decoder, 42-stream corpus, 1 thread, instructions: Go / dav1d | 1.853× | **1.692×** |
| Decoder, corpus, 8 threads, wall: Go / dav1d | 1.158× | **1.058×** (1.142× without `goav1_crowdrun`) |
| Decoder, corpus, 16 threads, wall: Go / dav1d | not measured | **0.972×** (1.063× without `goav1_crowdrun`) |
| Encoder, RTC p12, 1 thread, cycles: Go / SVT | 1.04–1.06× | **0.94–0.99×** (user CPU 1.10–1.14×) |
| Encoder, RTC p12, default threads, wall: Go / SVT | 0.55× (1 Mbps) | **0.50–0.61×** |
| Encoder quality, BD-rate (PSNR-Y) Go vs SVT, same recipe | not measured | **−4.0%** (Go better) |

The decoder's single-thread gap went from 32% to 21% more cycles than dav1d. On
one thread the encoder now needs fewer total cycles than SVT-AV1, but SVT
spends about 0.2 s of each run in the kernel; on user CPU alone goav1 is still
10-14% behind. With threads, the encoder finishes in 0.50-0.61 times SVT-AV1's
wall time. On the corpus the decoder is 1.06× dav1d's wall time at 8 threads
and 0.97× at 16; without the 30-tile `goav1_crowdrun` stream, which the
November 23 report called unrepresentative, it is 1.14× and 1.06×.

## Decoder: real-content corpus

The 42 streams of `testdata/realcorpus/streams` (31) and
`testdata/realcorpus/aq0/streams` (11): aomenc good and realtime, SVT-AV1
realtime and random access, and goav1 itself, 360p to 1080p, 300 frames each.
Each cell is the sum over streams of the per-stream minimum. Every stream
matches its `.md5` (the dav1d MD5) at 1, 8 and 16 workers (126/126).
`-framedelay 1` / `--framedelay 1` at one worker, defaults otherwise.

| Config | Wall, s | CPU, s | Instr, G | Cycles, G |
| --- | ---: | ---: | ---: | ---: |
| goav1, 1 worker | 21.36 | 20.79 | 474.7 | 83.20 |
| dav1d, 1 thread | 17.70 | 17.18 | 280.6 | 68.99 |
| goav1, 4 workers | 7.22 | 22.95 | 492.4 | 88.26 |
| dav1d, 4 threads | 5.91 | 19.48 | 297.9 | 75.08 |
| goav1, 8 workers | 5.23 | 24.89 | 496.1 | 93.78 |
| dav1d, 8 threads | 4.94 | 24.79 | 305.7 | 91.88 |
| goav1, 16 workers | 4.80 | 26.77 | 499.8 | 98.98 |
| dav1d, 16 threads | 4.94 | 36.54 | 315.3 | 135.16 |

Wall time, Go / dav1d: 1.21× at 1 thread, 1.22× at 4, 1.06× at 8, 0.97× at 16.
Instructions, Go / dav1d: 1.69×, 1.65×, 1.62×, 1.59×. Without
`goav1_crowdrun_1080p_3000k` (30 tiles, where dav1d spends 0.53-0.57 s and
10.7-22 G cycles at 8-16 threads), wall is 1.25× at 4 threads, 1.14× at 8 and
1.06× at 16.

On this hybrid machine, multi-thread cycles depend on how much work lands on
efficiency cores, so the threaded rows are compared by wall time and
instructions. dav1d's 16-thread CPU time includes 14.8 s of system time.

## Decoder: per stream

Cycles and instruction ratios on one thread; wall time in milliseconds at 4, 8
and 16 threads (Go / dav1d).

| Stream | Cycles, G, Go / dav1d | Cycles | Instr | 4 thr wall, ms | 8 thr wall, ms | 16 thr wall, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aomgood_debugging_720p_600k_c6 | 0.26 / 0.17 | 1.51× | 2.14× | 64 / 35 | 60 / 39 | 33 / 25 |
| aomgood_johnny_360p_600k_c2 | 0.11 / 0.05 | 2.03× | 2.69× | 23 / 13 | 28 / 14 | 22 / 12 |
| aomgood_mtbike_720p_1500k_c6 | 0.88 / 0.63 | 1.40× | 1.87× | 102 / 65 | 76 / 49 | 75 / 47 |
| aomgood_parkjoy_360p_1000k_c4 | 0.45 / 0.23 | 2.01× | 2.76× | 78 / 36 | 65 / 31 | 60 / 30 |
| aomgood_testsrc2_360p_20f | 0.07 / 0.04 | 1.84× | 2.54× | 16 / 10 | 15 / 9 | 15 / 10 |
| aomgood_testsrc2_360p_still | 0.01 / 0.01 | 1.61× | 1.79× | 7 / 5 | 7 / 5 | 7 / 6 |
| aomgood_wikipedia_360p_400k_c4 | 0.17 / 0.10 | 1.74× | 2.36× | 30 / 19 | 27 / 18 | 28 / 20 |
| aomrt_crowdrun_1080p_4000k_s10 | 6.65 / 5.34 | 1.24× | 1.69× | 506 / 378 | 250 / 192 | 222 / 169 |
| aomrt_debugging_720p_800k_s11 | 0.43 / 0.34 | 1.27× | 1.87× | 51 / 40 | 49 / 44 | 51 / 50 |
| aomrt_fourpeople_360p_500k_s9 | 0.57 / 0.44 | 1.29× | 1.72× | 55 / 40 | 51 / 44 | 52 / 58 |
| aomrt_fourpeople_720p_1000k_s10 | 1.56 / 1.26 | 1.23× | 1.69× | 125 / 92 | 91 / 79 | 94 / 97 |
| aomrt_johnny_720p_1500k_s9 | 1.85 / 1.57 | 1.18× | 1.69× | 152 / 116 | 116 / 112 | 119 / 125 |
| aomrt_mtbike_1080p_3000k_s9 | 4.53 / 3.59 | 1.26× | 1.74× | 338 / 253 | 272 / 191 | 147 / 115 |
| aomrt_mtbike_360p_600k_s8 | 0.62 / 0.47 | 1.32× | 1.73× | 53 / 37 | 38 / 28 | 39 / 35 |
| aomrt_oldtown_720p_1500k_s7 | 2.01 / 1.71 | 1.17× | 1.64× | 169 / 131 | 135 / 126 | 138 / 137 |
| aomrt_parkjoy_720p_2500k_s10 | 3.87 / 3.39 | 1.14× | 1.61× | 291 / 241 | 208 / 199 | 215 / 207 |
| aomrt_vidyo1_720p_2500k_s8 | 2.57 / 2.14 | 1.20× | 1.68× | 199 / 155 | 162 / 143 | 166 / 153 |
| aomrt_wikipedia_1080p_1500k_s11 | 0.84 / 0.66 | 1.28× | 1.84× | 80 / 54 | 55 / 51 | 57 / 57 |
| goav1_crowdrun_1080p_3000k | 4.58 / 3.77 | 1.22× | 1.68× | 373 / 421 | 187 / 531 | 151 / 569 |
| goav1_debugging_720p_1000k | 0.36 / 0.31 | 1.15× | 1.63× | 38 / 34 | 32 / 36 | 31 / 38 |
| goav1_fourpeople_720p_1000k | 1.03 / 0.88 | 1.16× | 1.59× | 88 / 71 | 60 / 68 | 56 / 78 |
| goav1_johnny_360p_500k | 0.43 / 0.35 | 1.23× | 1.68× | 49 / 42 | 43 / 41 | 40 / 47 |
| goav1_parkjoy_720p_2000k | 2.41 / 2.17 | 1.11× | 1.64× | 182 / 152 | 120 / 124 | 116 / 126 |
| svtra_johnny_720p_crf35_p8 | 1.06 / 0.76 | 1.41× | 1.91× | 144 / 112 | 114 / 99 | 113 / 104 |
| svtra_mtbike_1080p_crf35_p8_2tile | 7.37 / 6.86 | 1.07× | 1.65× | 575 / 566 | 380 / 387 | 348 / 359 |
| svtra_oldtown_720p_crf35_p8_fgs8 | 2.46 / 1.91 | 1.29× | 1.89× | 276 / 218 | 225 / 176 | 207 / 170 |
| svtrtc_crowdrun_1080p_3000k_p11 | 4.86 / 4.15 | 1.17× | 1.65× | 475 / 420 | 353 / 335 | 333 / 320 |
| svtrtc_fourpeople_720p_1000k_p12 | 1.03 / 0.88 | 1.18× | 1.62× | 144 / 125 | 124 / 117 | 115 / 117 |
| svtrtc_johnny_360p_500k_p12 | 0.41 / 0.34 | 1.20× | 1.66× | 73 / 66 | 62 / 61 | 57 / 62 |
| svtrtc_parkjoy_720p_2000k_p10 | 2.89 / 2.38 | 1.22× | 1.74× | 387 / 371 | 334 / 329 | 327 / 322 |
| svtrtc_wikipedia_1080p_1000k_p12 | 0.55 / 0.49 | 1.14× | 1.68× | 76 / 68 | 70 / 67 | 64 / 64 |
| aomrtaq0_crowdrun_1080p_4000k_s10 | 6.87 / 5.53 | 1.24× | 1.68× | 489 / 373 | 273 / 209 | 221 / 171 |
| aomrtaq0_debugging_720p_800k_s11 | 0.52 / 0.43 | 1.21× | 1.74× | 55 / 43 | 52 / 44 | 51 / 49 |
| aomrtaq0_fourpeople_360p_500k_s9 | 0.60 / 0.46 | 1.30× | 1.71× | 57 / 40 | 52 / 44 | 52 / 59 |
| aomrtaq0_fourpeople_720p_1000k_s10 | 1.59 / 1.29 | 1.24× | 1.69× | 122 / 89 | 93 / 79 | 92 / 95 |
| aomrtaq0_johnny_720p_1500k_s9 | 1.96 / 1.66 | 1.18× | 1.68× | 152 / 118 | 126 / 120 | 125 / 131 |
| aomrtaq0_mtbike_1080p_3000k_s9 | 4.47 / 3.54 | 1.26× | 1.73× | 319 / 242 | 188 / 141 | 143 / 111 |
| aomrtaq0_mtbike_360p_600k_s8 | 0.60 / 0.46 | 1.30× | 1.73× | 52 / 36 | 38 / 28 | 38 / 34 |
| aomrtaq0_oldtown_720p_1500k_s7 | 2.20 / 1.87 | 1.18× | 1.66× | 187 / 137 | 147 / 131 | 147 / 143 |
| aomrtaq0_parkjoy_720p_2500k_s10 | 3.89 / 3.41 | 1.14× | 1.60× | 284 / 235 | 222 / 203 | 215 / 211 |
| aomrtaq0_vidyo1_720p_2500k_s8 | 2.60 / 2.18 | 1.19× | 1.66× | 191 / 150 | 165 / 145 | 162 / 154 |
| aomrtaq0_wikipedia_1080p_1500k_s11 | 1.01 / 0.80 | 1.27× | 1.75× | 90 / 60 | 63 / 55 | 59 / 59 |

One thread: 1.07× (svtra_mtbike 2-tile) to 1.41× on aomenc realtime, SVT-AV1
and goav1 content; 1.40× to 2.03× on the seven aomenc good-quality streams.
Those use warped motion, filter intra and compound tools that the realtime
streams barely touch (see the levers below). Eight threads, 720p and up:
0.35× (goav1_crowdrun) to 1.42× (aomrt_mtbike_1080p).

`aomrt_mtbike_1080p_3000k_s9` at 8 threads is unstable. Its first pass
(load 9.5-17) gave 171 / 131 ms; the re-run (load 5.6) gave 272 / 191 ms, with
CPU time up in both decoders. The table shows the re-run.

The six streams the November 23 report tracked, one-thread cycles, Go:
`aomrt_crowdrun` 7.27G → 6.65G (1.37× → 1.24×), `svtrtc_crowdrun` 5.11G →
4.86G (1.26× → 1.17×), `goav1_crowdrun` 4.96G → 4.58G (1.33× → 1.22×),
`aomrt_fourpeople_720p` 1.79G → 1.56G (1.43× → 1.23×), `svtra_mtbike` 8.00G →
7.37G (1.18× → 1.07×), `aomrt_johnny` 2.09G → 1.85G (1.35× → 1.18×). dav1d's
counts moved by +0.5..+2.5% over the same pair of sessions.

## Decoder: where the single-thread gap is

PMI cycle samples from `xctrace` (goav1) and `sample` (dav1d, three runs
summed). Each decoder's stage shares are scaled to its own `time -l` cycle
count. Mcycles, Go / dav1d:

| Stage | aomrt_crowdrun (8 tiles) | svtrtc_crowdrun (1 tile) |
| --- | ---: | ---: |
| parse chain: mode parse, block loop, coefficients, symbol decoder | 2302 / 1758 (+544) | 1760 / 1435 (+325) |
| reconstruction glue (recon_b; classifier split differs, see lever 2) | 464 / 105 (+359) | 385 / 137 (+249) |
| motion compensation | 969 / 738 (+231) | 593 / 504 (+90) |
| refmvs | 393 / 315 (+78) | 245 / 216 (+29) |
| loop-filter masks | 367 / 292 (+75) | 0 / 0 |
| loop filter | 746 / 676 (+70) | 0 / 4 |
| inverse transform | 112 / 72 (+40) | 296 / 222 (+74) |
| intra prediction | 27 / 5 (+22) | 90 / 16 (+74) |
| frame/decoder glue | 65 / 0 (+65) | 48 / 0 (+48) |
| CDEF | 1017 / 1126 (−110) | 1286 / 1419 (−133) |
| runtime / libc | 179 / 245 (−66) | 145 / 187 (−42) |
| total | 6653 / 5344 (+1309) | 4861 / 4145 (+716) |

Mode parse, coefficient decoding and the symbol decoder share one row. Their
split does not line up between the two profiles: #338 moved range decoding
into `DecodeCoefs`, goav1 inlines most mode-parse symbol reads into the
`tile` functions, dav1d's `msac` bucket holds all of them, and the
`CoefBlock.DecodeCoefs` wrapper classifies as mode parse. #338 itself measured
coefficient decode plus all entropy reads on `aomrt_crowdrun` going from 1770
to 1530 Mcycles.

Noise: three goav1 profiles of `aomrt_crowdrun` vary per stage by 50-110
Mcycles (MC 887-954, coefficients 1007-1123, mode parse 809-913, refmvs
328-387), while the parse-chain sum stays within 2267-2339. dav1d's shares vary
as much across its three `sample` runs: MC 11.0-13.2% (590-705 Mcycles) and
CDEF 20.1-22.5%. Differences under about 100 Mcycles in one row, such as
loop-filter masks or refmvs, are inside that spread.

svtrtc_crowdrun codes no deblocking, so it has no loop-filter or mask rows. The `sample` profile of dav1d resolves the
unexported loop-filter kernels to `ipred_cfl_ac_444_8bpc_neon`; they are
counted as loop filter here.

## Encoder: RTC preset 12 recipe

`cmd/encbench` against `SvtAv1EncApp --preset 12 --rtc 1 --pred-struct 1
--rc 2` (flags under Reproduce), on `rtc240`: 720p, 240 frames, 30 fps.
Minimum of five interleaved repetitions. At one thread, goav1 runs with
`GOMAXPROCS=1` and SVT with `--lp 1`. Default threads means goav1 with no
`GOMAXPROCS` or `-threads` setting, and SVT with `--lp 0`.

| Config | SVT wall, s | Go wall, s | SVT user / sys, s | Go user / sys, s | SVT cycles | Go cycles | Cycles Go/SVT | Instr Go/SVT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 thread, 500 kbps | 1.13 | 1.06 | 0.91 / 0.20 | 1.00 / 0.03 | 4.41G | 4.15G | 0.94× | 1.31× |
| 1 thread, 1000 kbps | 1.28 | 1.22 | 1.05 / 0.22 | 1.17 / 0.04 | 5.01G | 4.76G | 0.95× | 1.30× |
| 1 thread, 2000 kbps | 1.43 | 1.41 | 1.19 / 0.22 | 1.36 / 0.03 | 5.58G | 5.52G | 0.99× | 1.29× |
| default threads, 500 kbps | 0.88 | 0.44 | 1.23 / 0.45 | 1.48 / 0.27 | 6.16G | 6.64G | 1.08× | 1.42× |
| default threads, 1000 kbps | 0.96 | 0.52 | 1.37 / 0.47 | 1.67 / 0.30 | 6.76G | 7.47G | 1.10× | 1.43× |
| default threads, 2000 kbps | 1.05 | 0.64 | 1.53 / 0.49 | 1.94 / 0.31 | 7.50G | 8.58G | 1.14× | 1.44× |

On November 23 the 1 Mbps, one-thread row was 5.16G (Go) against 4.95G (SVT)
cycles, with an instruction ratio of 1.48×. goav1 has since dropped 7.8% of its
cycles and 12% of its instructions (#321, #323, #324, #327).

### Quality at matched rate

The encoded streams from the same commands, plus 700 and 1400 kbps, were
written to IVF, decoded with dav1d, and scored with FFmpeg against the source.
goav1's output is byte-identical at one thread and at default threads; so is
SVT-AV1's at `--lp 1` and `--lp 0`. The quality figures therefore apply to
every timed row.

| Target | Go kbps | SVT kbps | PSNR-Y Go / SVT, dB | PSNR-YUV (4:1:1) Go / SVT | SSIM-Y Go / SVT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 514.8 | 522.6 | 37.13 / 37.03 | 38.28 / 38.22 | 0.9499 / 0.9507 |
| 700 | 715.3 | 718.4 | 38.86 / 38.67 | 39.92 / 39.77 | 0.9632 / 0.9631 |
| 1000 | 1016.0 | 1015.3 | 40.85 / 40.58 | 41.80 / 41.57 | 0.9749 / 0.9744 |
| 1400 | 1418.7 | 1409.8 | 42.88 / 42.61 | 43.70 / 43.48 | 0.9829 / 0.9824 |
| 2000 | 2018.7 | 2003.1 | 44.70 / 44.50 | 45.40 / 45.24 | 0.9876 / 0.9873 |

BD-rate of goav1 against SVT-AV1, using a cubic fit over the five points:
−3.98% on PSNR-Y, −3.45% on PSNR-YUV, −2.62% on SSIM-Y. Negative means goav1
needs fewer bits for the same quality. goav1's chroma PSNR is lower by up to
0.2 dB at the lowest rate. This covers one 240-frame clip;
[encoder-svt-performance.md](encoder-svt-performance.md) has the broader
quality comparison.

### Where the encoder's cycles go

1 Mbps, one thread. goav1 was profiled with `xctrace` PMI samples. SVT-AV1 was
profiled with `sample`, three runs, dropping the 48,680 samples in which its
pipeline threads sat in `semaphore_wait`. Each encoder's shares are scaled to
its own `time -l` cycles (4.76G / 5.01G). `sample` has under-counted SVT's
short NEON kernels by up to 2× against direct timing before, so read the SVT
kernel rows as lower bounds. Mcycles, Go / SVT:

| Stage | Go | SVT | Gap |
| --- | ---: | ---: | ---: |
| HME + full-pel ME (SAD) | 963 | 518 | +445 |
| variance | 367 | 183 | +184 |
| quantization | 223 | 96 | +127 |
| CDEF search + apply | 707 | 617 | +90 |
| inverse transform + reconstruction | 239 | 151 | +88 |
| forward transform | 308 | 244 | +64 |
| entropy coding + rate estimation | 476 | 421 | +54 |
| sub-pel search | 111 | 75 | +36 |
| ref-MV lists | 125 | 154 | −30 |
| mode decision, intra and inter prediction | 1117 | 1382 | −265 |
| runtime / libc / kernel | 114 | 1165 | −1051 |

SVT's runtime row is mostly `__mmap` page faults, about 17% of its busy
samples, plus `memcpy`/`memset`. That kernel time is why total cycles favour
goav1 while user CPU does not. Leaving the runtime rows out, goav1 is about
0.81 G cycles behind SVT's own compute per 240 frames (4.65G against 3.84G). Three rows account for 0.76 G: SAD, variance and
quantization.

## Ranked levers

The levers are ranked roughly by the cycles still on the table, from the two profiles
above. Each carries a tag:

- **Walled**: blocked by an arm64 operation that Go 1.27.4's `simd/archsimd`
  does not expose: UABD, UABAL, UADALP/UADDLP, UDOT, SDOT, USMMLA, SQRDMULH or
  SQRSHRN.
- **ABI**: bounded by Go's calling convention, which has no callee-saved
  registers.
- **Addressable**: neither of the above.

Recorded dead ends are not listed again: entropy NEON, blanket BCE, coefficient
loop codegen variants, eob-threshold itx dispatch, the int16 column pipe, MC
codegen, the refmvs store variants, in-place CDEF staging and PGO
regeneration.

1. **Decoder parse chain: +544 / +325 Mcyc.** *Mostly ABI.* This is
   mode parse, the block loop, coefficients and the symbol decoder. The
   block loop no longer dispatches through a generic controller: as of
   `60fabd1f` the tile controller has one method, `ReconB`, called once per
   block (dav1d makes one `bd_fn.recon_b_*` call per block), the per-TXB
   calls are direct calls on concrete types, and `BlockLoopVisit` is 232
   bytes. The remaining mode-side gap is flat across
   `decodeBlockPredictionModeInto`, the refmvs mode reads and the partition
   walk, with no single leader, and is largely Go's call ABI (no
   callee-saved registers). The coefficient token loop is not the lever
   either: recorded variants there are within noise. This is also the serial
   chain that bounds threaded decode of single-tile streams.
2. **Decoder reconstruction glue: inter-prediction setup and intra edge
   prep.** *Addressable.* The +359 / +249 in the stage table overstates it:
   the two classifiers bucket differently (dav1d's `read_coef_tree` falls
   under coefficients, and the goav1 side of similar comparisons has also
   carried LF-mask building). On a matched set the TXB walk is about level:
   `ReconB` + `reconCoefTree` + `reconTXB` + `addResidualInWindow` are about
   233 Mcyc on `aomrt_crowdrun` against about 206 for dav1d's
   `recon_b_inter` + `read_coef_tree`, and about 121 against 113 on
   `svtrtc_crowdrun`. The real gaps are inter-prediction setup
   (`reconBInter` + `interTarget*`, about 110 Mcyc against about 82 for
   dav1d's `mc()` on `aomrt_crowdrun`) and intra edge prep (about 90 against
   39 on `svtrtc_crowdrun`, which #343 addresses).
3. **Encoder SAD, HME and variance: +445 and +184 Mcyc.** *Walled* (UABD,
   UDOT, UABAL, UADALP). SVT uses `*_neon_dotprod` (UABD + UDOT). goav1's
   SAD is at about 4 vector ops per 16 pixels per row against SVT's 2, after
   #327. Nothing more from source changes alone, so this needs the
   operations in archsimd.
4. **Decoder motion compensation: +231 / +90 Mcyc.** *Walled* (SDOT, USDOT,
   USMMLA; no SQRSHRUN or immediate-shift lowering). `Put8` and
   `mc8PutV6FromI32` are vector-ALU-throughput bound, and dav1d runs its I8MM
   path on the M4. The MC codegen branch is a recorded dead end.
5. **Encoder quantization: +127 Mcyc.** *Addressable.*
   `TryQuantizeBRowGroupsDiffTrusted` + `quantizeBVectors` take 168 Mcyc
   against `svt_aom_quantize_b_neon`'s 45 samples (about 75 Mcyc). An October
   note points at about 16 GPR-to-lane moves of glue per call.
6. **Decoder loop-filter masks: +75 Mcyc, inside the profile noise.**
   *Addressable.*
   `FrameWorkLoopFilterMasks.buildBlockWithLevels` (130) and `maskEdgesInter`
   (107) against dav1d's `dav1d_create_lf_mask_inter` path. This work is also
   on the parse chain at high thread counts.
7. **Decoder warped compound and the aomenc good-quality streams.**
   *Addressable.* These seven streams are 1.40-2.03× dav1d's cycles, the worst
   ratios in the corpus, and none is in the attribution above. On
   `aomgood_parkjoy_360p`, `motion.PredictWarpedCompoundToConvBuf` is 81 of
   434 Mcycles (19%). Its vertical 8-tap pass is a scalar per-pixel loop
   (`internal/av1/motion/compound.go`). The non-compound warp got NEON in
   #337 and earlier, and dav1d runs `warp_affine_8x8t` in NEON. Filter intra
   (`predictFilterIntraBlockDirect8SIMD`) and CDEF lead on
   `aomgood_mtbike_720p`. These streams are small, so this is a coverage
   lever rather than a corpus-total one.
8. **Encoder CDEF, inverse transform and reconstruction, forward transform:
   +90 / +88 / +64 Mcyc.** *Addressable.* The per-kernel forward DCTs are at
   SVT's cycles (#321, #323), so what is left is per-block glue: residual
   formation, the reconstruction copy and the CDEF search loop
   (`searchOneDual`, `searchBlocksU8SIMD`).
9. **Threaded decode of multi-tile streams.** *Addressable.* At 8 threads,
   the single-tile SVT-AV1 realtime streams are 1.02-1.06× dav1d's wall time,
   but `aomrt_crowdrun` (8 tiles) is 1.30× and `aomrt_mtbike_1080p` 1.30-1.42×.
   Most of that is the single-thread gap carried through: `aomrt_crowdrun`
   is 1.24× at one thread, and goav1 scales 6.7× from 1 to 8 threads against
   dav1d's 7.05×, a loss of about 5%. The port of dav1d's pull-model worker
   loop is parked in #308 behind a tile parse that can stop and resume per
   superblock row.
10. **Decoder refmvs: +78 / +29 Mcyc, inside the profile noise.** *ABI.*
    `refmvsTile.find` is a line-for-line port of dav1d's; what remains is stack traffic, about 430
    references in about 1,460 instructions, because Go has no callee-saved
    registers. The store-layout variants are recorded dead ends.

Landed after the measurement point, so not counted above: #342 (dav1d
`itxfm_add` dispatch, inline coefficient clears; the inverse-transform rows,
+40 / +74 Mcyc) and #343 (intra edge prep and ipred kernels).

## Reproduce

~~~sh
export GOEXPERIMENT=simd
go build -pgo=cmd/qualitybench/default.pgo -o aom-go-dec ./cmd/aom-go-dec
go build -pgo=cmd/qualitybench/default.pgo -o encbench ./cmd/encbench

# 1 worker; N = 4, 8, 16 drop the framedelay flags
GOMAXPROCS=1 /usr/bin/time -l ./aom-go-dec -quiet -workers 1 -framedelay 1 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 1 --framedelay 1 --muxer yuv -i X.ivf -o /dev/null
GOMAXPROCS=8 /usr/bin/time -l ./aom-go-dec -quiet -workers 8 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 8 --muxer yuv -i X.ivf -o /dev/null

GOMAXPROCS=1 ./encbench -input rtc240.yuv -frames 240 -width 1280 -height 720 \
  -fps 30 -bitrate 1000000 -nopsnr -o /dev/null
SvtAv1EncApp -i rtc240.yuv -w 1280 -h 720 --fps-num 30 --fps-denom 1 --frames 240 \
  --input-depth 8 --preset 12 --rc 2 --tbr 1000 --buf-sz 1000 \
  --buf-initial-sz 500 --buf-optimal-sz 600 --lookahead 0 --pred-struct 1 \
  --rtc 1 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1 --progress 0 \
  --lp 1 -b /dev/null

# quality: same commands with -o / -b out.ivf, then
dav1d -q --muxer yuv -i out.ivf -o - | ffmpeg -f rawvideo -pix_fmt yuv420p \
  -s 1280x720 -i - -f rawvideo -pix_fmt yuv420p -s 1280x720 -i rtc240.yuv \
  -lavfi "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr;[b][d]ssim" -f null -
~~~

Run the timed commands directly under `/usr/bin/time -l`, as above. Behind
`sh -c` the cycle and instruction counters cover only the shell.

Correctness of the measured decoder: all 42 corpus streams match their dav1d
MD5 at 1, 8 and 16 workers.
