# End-to-end codec gap: October 24, 2026

Measured `main` at `6d62eb88ae414c283be8fff35a1e864441af41fe` (PR #172), using
official Go 1.27.2 with `GOEXPERIMENT=simd` on Apple M4 Max (12 P + 4 E cores,
64 GB) / macOS 26.3.1. This supersedes the
[October 9 report](e2e-codec-gap-20261009.md), which stays as history.
These are exploratory measurements on a shared workstation, not an isolated
benchmark machine. Other agents ran Go test suites and encoder benchmarks
throughout (1-minute load average 2.5–34 across blocks, see
[Machine load](#machine-load)); every table therefore uses the minimum or the
median of interleaved repeats, and reports retired instructions next to time.
Do not read small differences as regressions or improvements.

| Measure | 20261009 (`0824b687`) | 20261024 (`6d62eb88`) |
| --- | ---: | ---: |
| Decoder, public API, 18 clips, 1 worker: Go / dav1d | 3.640–3.653× | **1.756–1.847×** (3 rounds) |
| Decoder, real-content corpus, 1 thread, CPU time: Go / dav1d | not measured | **1.595×** (42 streams) |
| Decoder, real-content corpus, 8 threads, wall: Go / dav1d | not measured | **1.868×** (CPU 1.439×) |
| Encoder, qualitybench preset 13, median Go / SVT time | 3.678× (3.419–4.857×) | **1.819×** (1.578–2.012×) |
| Encoder, RTC preset 12, 1 thread, CPU time: Go / SVT | not measured | **1.21–1.26×** (instructions 1.64×) |

The decoder gap is now about 1.8× on the matched API benchmark and about 1.6×
CPU on real content; the encoder gap is about 1.8× process time on the old
conference clips and about 1.2× CPU on the RTC recipe. **Neither is parity.**
The reference builds also changed between reports (dav1d 1.5.3 to 1.5.4, SVT
4.0.1 to 4.1.0, see below), and the old decoder run was not PGO, so the
cross-report ratio change is a statement about these two configurations, not a
clean attribution to code.

## Decoder: public API versus dav1d

Same protocol as the old report: the 18-clip, 864-visible-frame
`testdata/benchcorpus`, one decoder worker, one warmup and nine full-clip
samples per clip and path, aggregate = sum of per-clip medians. All 18 clips
matched their stream-MD5 sidecars in the Go sampler's preflight (both the IVF
and the pre-parsed-payload constructors) and in the C helper's preflight. Three
rounds were interleaved (dav1d helper, Go PGO, Go non-PGO, Homebrew-linked
helper; repeat).

| Path | Round 1, ms | Round 2, ms | Round 3, ms |
| --- | ---: | ---: | ---: |
| Go cold IVF constructor + decode + close (PGO) | 957.677 | 977.995 | 944.437 |
| Go cold pre-parsed payload constructor + decode + close (PGO) | 947.660 | 977.807 | 944.624 |
| Go warm reset + decode (PGO) | 921.105 | 940.580 | 911.615 |
| Go cold IVF, non-PGO control build | 974.632 | 978.612 | 972.946 |
| dav1d 1.5.4 fresh C API context + decode + close (local static) | 518.576 | 552.364 | 537.909 |
| dav1d 1.5.4 same helper linked to the Homebrew dylib | 560.410 | 542.510 | 539.280 |
| Go cold-IVF / dav1d time ratio | 1.847× | 1.771× | 1.756× |

Taking each clip's minimum across the three rounds, Go cold IVF is 938.0 ms
against dav1d 518.6 ms: **1.809×**, warm Go 1.749×, non-PGO Go 1.857×. Go would
need about 45% less time to match this dav1d baseline (it needed 72.5% less in
the old report). The old report's dav1d was 1.5.3: a separate
`DYLD_LIBRARY_PATH` run of the helper against the Homebrew 1.5.3 dylib gave
547.1–558.7 ms, within noise of the 1.5.4 numbers (539.2–555.1 ms in three
further static runs), so the version change does not move the ratio. Pre-parsing
payloads still does not matter (-1.0% to +0.02% by round). The PGO profile is
worth 0.1–3% on this corpus (938.0 vs 963.0 ms by per-clip minimum). The
Homebrew-linked helper differs from the static build by -2% to +8% by round,
which is the noise floor of this host.

By resolution (per-clip minimum across rounds, cold IVF): p144 2.26×, p288
1.90×, p360 1.99×, p720 1.61×. The 10-bit clip is the worst at 2.65× and the
p720 inter clips the best at 1.54–1.65×. Per-clip rounds are in the CSV.

Same exclusions and caveats as before: file reads, hash checks and MD5
preflight are excluded; Go's IVF constructor copies packet payloads while dav1d
references the preloaded bytes; no matched C warm-reset result exists. This is
in-process API timing, not CLI startup. The generated corpus has no source
manifest and is one source clip scaled and encoded only by aomenc.

## Decoder: process-inclusive, real content, 1 and 8 workers

The C helper hard-codes one thread, `max_frame_delay=1` and exactly the
18-clip/864-frame corpus, and the Go sampler hard-codes `WithWorkers(1)` and the
same 18-clip check. Neither can time the real-content corpus or 8 workers, and
the harness was not modified. Those two cases are therefore
**process-inclusive CLI comparisons** (`cmd/aom-go-dec` against the `dav1d`
CLI, both writing YUV to `/dev/null`, timed with `/usr/bin/time -l`), which
include process startup and file I/O and are **not comparable with the API
table above**. They are the numbers to use for the RTC question.

Corpora: `testdata/benchcorpus` (18 clips, same as above), `testdata/realcorpus/streams`
(31 streams, 300 frames each, from `scripts/gen_real_corpus.sh`: aomenc good
and realtime, SVT-AV1 realtime and random access, goav1) and
`testdata/realcorpus/aq0/streams` (11 libwebrtc-style aomenc realtime streams
with aq-mode 0). Streams are 360p–1080p, derived from nine Xiph sources
(conferencing, film, camera and screen content) plus a synthetic `testsrc2`
pattern; one SVT random-access stream has film grain and one is 2-tile.

**Verification.** Before timing, every stream was decoded once with Go at 1 and 8
workers and dav1d at 1 and 8 threads, and the MD5 of the output was compared with
the `.md5` sidecar: **all 60 streams match on all four decoders** (240 of 240).
`dav1d --muxer md5` skips film grain by default, so the verification pass uses
`--filmgrain 1`; the timed runs use the YUV muxer, which applies film grain by
default. A first verification pass without that flag reported the one
film-grain stream (`svtra_oldtown_720p_crf35_p8_fgs8`) as a dav1d mismatch; that
was the missing flag, not a decoder difference, and the pass was rerun. Every
timed run exited 0.

Each cell is the sum over streams of the per-stream minimum over six interleaved
repetitions (two batches of three). `time -l` resolves 10 ms, so for the small
benchcorpus clips wall and CPU are coarse; cycles and instructions are exact.

| Corpus | Config | Wall, s | CPU (user+sys), s | Instr, G | Cycles, G |
| --- | --- | ---: | ---: | ---: | ---: |
| real, 31 streams | Go 1 worker | 18.67 | 18.49 | 423.8 | 74.88 |
| real, 31 streams | dav1d 1 thread | 11.58 | 11.42 | 191.0 | 46.66 |
| real, 31 streams | Go 8 workers | 6.67 | 23.37 | 441.5 | 89.26 |
| real, 31 streams | dav1d 8 threads | 3.41 | 17.05 | 209.0 | 64.20 |
| aq0, 11 streams | Go 1 worker | 8.06 | 8.00 | 179.6 | 32.48 |
| aq0, 11 streams | dav1d 1 thread | 5.25 | 5.19 | 89.5 | 21.16 |
| aq0, 11 streams | Go 8 workers | 1.70 | 10.96 | 189.0 | 41.37 |
| aq0, 11 streams | dav1d 8 threads | 1.07 | 6.80 | 96.4 | 25.98 |
| benchcorpus, 18 clips | Go 1 worker | 0.95 | 0.89 | 21.4 | 4.07 |
| benchcorpus, 18 clips | dav1d 1 thread | 0.54 | 0.50 | 10.2 | 2.39 |
| benchcorpus, 18 clips | Go 8 workers | 0.43 | 1.18 | 23.3 | 5.17 |
| benchcorpus, 18 clips | dav1d 8 threads | 0.10 | 0.64 | 11.1 | 3.17 |

| Corpus | 1 thread: CPU / cycles / instr | 8 threads: wall / CPU / cycles / instr |
| --- | ---: | ---: |
| real, 31 | 1.619× / 1.605× / 2.219× | 1.956× / 1.371× / 1.390× / 2.112× |
| aq0, 11 | 1.541× / 1.535× / 2.006× | 1.589× / 1.612× / 1.592× / 1.960× |
| real + aq0, 42 | **1.595× / 1.583× / 2.151×** | **1.868× / 1.439× / 1.449× / 2.064×** |
| benchcorpus, 18 | 1.780× / 1.702× / 2.093× | 4.300× / 1.844× / 1.631× / 2.099× |

Findings, all ratios Go/dav1d (above 1 means Go took more):

- On real content Go needs about 1.6× the CPU of dav1d on one thread, but
  retires about 2.2× the instructions: Go runs at 5.6 instructions per cycle
  against dav1d's 4.1, so the remaining gap is instruction count, not stalls.
- With 8 workers Go's wall gap is 1.87× and its CPU cost is 1.44×; Go scales
  3.2× from 1 to 8 workers on these streams, dav1d 3.8×. The Go frame-threaded
  path uses about 30% more total CPU than its own single-thread path
  (34.3 vs 26.5 s over the 42 streams); dav1d uses 44% more (23.9 vs 16.6 s).
- The benchcorpus 8-thread wall ratio (4.3×) is dominated by 48-frame clips of
  tiny pictures where Go's startup and thread spin-up outweigh decoding; do not
  generalize it. The API sampler above is the right benchcorpus number.
- By encoder (cycles, 1 thread): aomenc realtime 1.53–1.58×, SVT realtime
  1.45×, goav1 1.48×, aomenc good mode 1.86×, SVT random access 1.86× (this
  group contains the film-grain stream). The worst stream is the **film-grain**
  one at 4.26× cycles (CPU 1.96 s against 0.45 s, instructions 7.3×); film
  grain is the single largest decoder outlier. The best stream is the 2-tile SVT
  random-access stream at 1.23×. Median per-stream cycle ratio is 1.59×.

## Encoder: process-inclusive versus SVT-AV1 (old protocol)

Same protocol and clips as the old report: FourPeople and Johnny, first 120
frames of the Xiph 1280×720 60 fps WebM, 8-bit I420 (raw SHA-256 identical to
the old report's, ffmpeg 8.0.1), at 3/6/9/12 Mbps, one warmup and three
measured process-inclusive runs per clip/rate/encoder, interleaved with shuffled
order and seed 1, two rounds. The Go executable (`qualitybench`) is prebuilt with
the repository's default PGO profile, `GOMAXPROCS=1`, `MaxThreads=1`, effort 0,
`-layers 1`. SVT-AV1 is **4.1.0** (the old report used 4.0.1), preset 13, `--lp 1`,
`--asm max`, same low-delay CBR settings, no lookahead, scene-cut detection or
temporal filtering. Metrics are PSNR and SSIM against the raw input via
FFmpeg/libdav1d.

| Clip | Target Mbps | Go fps, R1 / R2 | SVT fps, R1 / R2 | Go/SVT time, R1 / R2 |
| --- | ---: | ---: | ---: | ---: |
| FourPeople | 3 | 198.51 / 196.63 | 354.59 / 342.58 | 1.790× / 1.743× |
| FourPeople | 6 | 157.98 / 155.75 | 284.99 / 276.45 | 1.805× / 1.774× |
| FourPeople | 9 | 133.42 / 122.68 | 251.16 / 246.98 | 1.881× / 2.012× |
| FourPeople | 12 | 119.76 / 119.62 | 225.18 / 224.60 | 1.880× / 1.878× |
| Johnny | 3 | 184.63 / 192.45 | 291.34 / 333.42 | 1.578× / 1.733× |
| Johnny | 6 | 143.66 / 151.30 | 249.35 / 268.98 | 1.736× / 1.778× |
| Johnny | 9 | 128.02 / 127.63 | 234.94 / 244.67 | 1.834× / 1.918× |
| Johnny | 12 | 113.54 / 113.55 | 216.24 / 219.83 | 1.905× / 1.936× |

The median of these 16 time ratios is **1.819×**, range **1.578–2.012×** (old
report: 3.678×, 3.419–4.857×). CPU-time ratios agree (median 1.806×, range
1.732–1.948×; `observed_parallelism` is 0.89–1.00 for both encoders). Go's
absolute throughput roughly doubled (old 56–79 fps, now 114–199 fps) while SVT's
barely moved (different SVT version), which is the whole change in the ratio.

| Clip | Target Mbps | Actual Mbps, Go / SVT | PSNR dB, Go / SVT | SSIM, Go / SVT |
| --- | ---: | ---: | ---: | ---: |
| FourPeople | 3 | 3.262 / 3.039 | 44.0353 / 43.6197 | 0.9851 / 0.9856 |
| FourPeople | 6 | 6.406 / 6.018 | 46.0058 / 45.3188 | 0.9889 / 0.9887 |
| FourPeople | 9 | 9.503 / 8.909 | 46.9273 / 46.0852 | 0.9904 / 0.9899 |
| FourPeople | 12 | 12.483 / 11.604 | 47.4405 / 46.4897 | 0.9909 / 0.9903 |
| Johnny | 3 | 3.220 / 2.916 | 45.1637 / 44.8867 | 0.9866 / 0.9868 |
| Johnny | 6 | 6.390 / 5.921 | 46.9267 / 46.2498 | 0.9904 / 0.9897 |
| Johnny | 9 | 9.527 / 8.818 | 47.7749 / 46.9238 | 0.9916 / 0.9908 |
| Johnny | 12 | 12.484 / 11.458 | 48.2608 / 47.3082 | 0.9922 / 0.9913 |

Every measured encode succeeded. For each clip/rate/encoder the encoded and
decoded SHA-256 values are identical across its three samples and across both
rounds (all 16 combinations), so quality values are shown once. The encoded
hashes differ from the old report because the encoder changed substantially.

**Rate-control accuracy regressed in this configuration.** Go now overshoots the
target by **+4.0% to +8.7%** at these 60 fps, 120-frame, single-layer settings
(the old report: -0.4% to +0.4%), while SVT lands between -4.5% and +1.3%. Go's
PSNR advantage (+0.28 to +0.95 dB) therefore comes with 6–10% more bits than SVT,
and SVT has the higher SSIM at 3 Mbps on both clips.
Equal target bitrate is **not equal quality**; no equal-quality speed or BD-rate
claim is made. The RTC recipe below, at the target rates it was tuned for,
hits the target to within about 2% for both encoders.

## Encoder: RTC preset 12 recipe, CPU, cycles and instructions

The 720p real-content RTC case: 240 frames of 1280×720 at 30 fps, CBR low delay
at 1 and 2 Mbps. Go is `cmd/encbench` (PGO build) with its realtime defaults
(`-layers` default, one tile, effort default). SVT-AV1 4.1.0 runs preset 12
with `--rtc 1 --pred-struct 1 --lookahead 0 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1`
and the buffer sizes in the reproduce section. Both write an IVF during the
timed run; neither encoder's PSNR pass is timed (`-nopsnr`; both streams are
scored afterwards, identically: decode with dav1d, compare Y with the input).
The input is the 48-frame `p720_inter_q20` benchcorpus decode (lossy, so easier
than camera noise) ping-pong looped to 240 frames; its SHA-256 is
`63334b7d55c94588c1115756ee063ad7ab42aedb23df7885a9f8d6b12153de04`.

Twelve interleaved repetitions per configuration (two batches); each cell is the
minimum over the repetitions of `/usr/bin/time -l` wall, CPU (user+sys),
instructions retired and cycles. The first batch ran at load average 8–34, the
second at 5–6; the minimum is dominated by the second.

| Rate | Encoder | Wall, s | CPU, s | Instr, G | Cycles, G | Payload kbps | PSNR-Y, dB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 Mbps | Go, 1 thread | 1.49 | 1.47 | 32.75 | 5.84 | 1016.3 | 40.81 |
| 1 Mbps | SVT p12, lp 1 | 1.23 | 1.21 | 20.00 | 4.79 | 1015.3 | 40.79 |
| 1 Mbps | Go / SVT | 1.21× | 1.21× | 1.64× | 1.22× | | +0.014 |
| 2 Mbps | Go, 1 thread | 1.76 | 1.74 | 37.92 | 6.90 | 2021.5 | 44.66 |
| 2 Mbps | SVT p12, lp 1 | 1.39 | 1.38 | 23.18 | 5.45 | 2003.1 | 44.68 |
| 2 Mbps | Go / SVT | 1.27× | 1.26× | 1.64× | 1.27× | | -0.013 |
| 1 Mbps | Go, `-threads 8` | 0.61 | 2.18 | 37.99 | 8.30 | 1016.3 | 40.81 |
| 1 Mbps | SVT p12, lp 8 | 0.93 | 1.81 | 21.99 | 6.62 | 1015.3 | 40.79 |
| 2 Mbps | Go, `-threads 8` | 0.76 | 2.52 | 44.41 | 9.60 | 2021.5 | 44.66 |
| 2 Mbps | SVT p12, lp 8 | 1.01 | 1.98 | 25.18 | 7.33 | 2003.1 | 44.68 |

PSNR-Y is the mean of per-frame Y PSNR (the aggregate-MSE values are in the
CSV). Payload kbps excludes the 32-byte IVF header and 12-byte frame headers.
Go's stream is bit-identical at 1 and 8 threads; so is SVT's at lp 1 and 8. The
single-thread results are the robust ones: 161 fps (Go) against 195 fps (SVT) at
1 Mbps by minimum wall, equal PSNR-Y and bitrate within 0.1–1%. Go retires 1.64×
the instructions but needs only 1.2–1.3× the cycles. The 8-thread rows are the
noisiest (the other agents were also running 8-thread suites); the minimum
wall and minimum CPU in a row can come from different repetitions, so treat the
8-thread wall advantage as unconfirmed.

## Provenance

- **Go toolchain and PGO.** go1.27.2 `GOEXPERIMENT=simd`. The old run used
  `GOTOOLCHAIN=go1.27.2 ... /Users/thesyncim/.codex/bin/project-env go test -c`.
  That wrapper only sets a project `GOCACHE` and `GOTMPDIR`; it adds no `-pgo`,
  and test binaries do not auto-discover the profile, so by the commands that
  report records **the old decoder sampler binary was not PGO** while the old encoder (`qualitybench`, a main
  package that discovers `cmd/qualitybench/default.pgo`) was. This run does not
  use the wrapper: `GOEXPERIMENT=simd go test -c -pgo=cmd/qualitybench/default.pgo`
  with a private `GOCACHE` for the sampler, and `go build -pgo=...` for
  `aom-go-dec`, `encbench` and `qualitybench`. A non-PGO sampler binary was
  built too as a control (above). Profile SHA-256
  `4705be048559aca1b3dd5eecc486fb9ebe65354e5b51aba106e5af4bae90fb95`.
- **dav1d (decoder reference).** Source `third_party/upstream.lock` pin: tag 1.5.4,
  commit `54706fc6bc0cdecab7e9593974a4039cc038fca7`, runtime `1.5.4-0-g54706fc`.
  The helper was linked against a local static `libdav1d.a` built with
  `meson setup --buildtype=release -Ddefault_library=static -Dbitdepths=8,16
  -Dc_args=-O3 -Denable_tools=false -Denable_tests=false -Denable_examples=false`
  (meson 1.12.1, Apple clang 21.0.0;
  `config.h` shows `HAVE_ASM 1`, `ARCH_AARCH64 1`; no LTO, no debug info).
  Library SHA-256 `260eb74f7656e348656288055107f166842f625cacab2359b21b757e8903575d`,
  helper SHA-256 `192dcfc87e568ed5cb0b99a9cc3dc7116d27e7c8f994d8804f3984faa2cf1503`,
  compiled from `tools/c_api_decode_bench/bench.c` with the flags of
  `build.sh` (`-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wconversion`) but dav1d-only,
  because the pinned libaom build tree that `build.sh` requires no longer exists
  (libaom headers and `md5_utils.c` come from `third_party/upstream/libaom` at
  `047d8cf6`). The old report's build was commit `b546257f` (1.5.3) with `-g`.
  `third_party/upstream/dav1d` in the checkout is still a stripped 1.5.3 snapshot
  without `meson_options.txt`, so the 1.5.4 source came from a separate clone at
  the pinned commit. The CLI tables use the Homebrew `dav1d` 1.5.4 binary
  (SHA-256 `add73331b706cb655203c27f7ee21d091ccbbf65191a98e4e49d2d9391cc38e4`,
  dylib `60c3a42a79582241ac77bbbba4c0137c2209204aecbc59e5d6392c2dc0a5b439`);
  its API-helper timing is within 4% of the static build. Neither binary's
  build provenance is independently attested.
- **SVT-AV1.** `~/bin/svt410/SvtAv1EncApp`, `SVT-AV1 v4.1.0 (release)`, SHA-256
  `b4d8c5ed72671814f13559db312195414e26c02de878ebf18d7746d6e6f2b4f0`, built
  from the vendored `third_party/upstream/svt-av1` at tag v4.1.0, commit
  `c04f951541ad600e0d9c10836f2ab7b9bc69816d`. The scratch CMake trees of that
  source show Release, `-O3 -DNDEBUG`, NEON/DOTPROD/I8MM on, `NATIVE` off,
  `SVT_AV1_LTO` off; which tree produced the saved binary is not recorded. The
  old report used Homebrew 4.0.1 (SHA-256 `db044194...`); the two differ at RTC
  preset 12, so the encoder numbers are not comparable with the old SVT column.
- **Go binaries (SHA-256).** Sampler (PGO) `2a9678ad...`, sampler (non-PGO)
  `c472712d...`, `aom-go-dec` `de96bc8f...`, `qualitybench` `365b36d1...`,
  `encbench` `f3a34e0d...`; full hashes in the JSON.
- **Inputs.** FourPeople and Johnny raw 120-frame SHA-256 values equal the old
  report's (`a022f06b...`, `c11fc6fa...`); the real-content corpus generator
  recorded goav1 `26d62d52`, aomenc v12, SVT 4.1.0, dav1d 1.5.4 in
  `streams.tsv`. The real corpus and benchcorpus are gitignored local data.

## Machine load

One-minute load averages sampled at each measurement: decoder API rounds 2.5–6.4;
qualitybench 7.8–10.4; RTC batch one 8.0–33.9 and batch two 5.2–5.8; CLI repetitions
1–3 3.3–5.0 (mean 4.0) and repetitions 4–6 5.1–7.2 (mean 6.1). The 16 cores were
shared with concurrent Go test binaries and encoder benchmarks from other
agents. Instruction counts are insensitive to this; cycles and CPU time are not,
which is why minima over interleaved repetitions are reported. The 3-repetition
and 6-repetition CLI sums differ by at most 0.5% (cycles), but a quiet host
could still shift absolute times. The old report also ran beside another
project's test process, so the same "exploratory, shared workstation" reading
applies.

## Reproduce

Build outside timing with a private cache and the PGO flag:

~~~sh
export GOCACHE=/path/to/private-cache GOTOOLCHAIN=go1.27.2 GOEXPERIMENT=simd
go test -c -pgo=cmd/qualitybench/default.pgo -o public-e2e.test .
for c in aom-go-dec encbench qualitybench; do
  go build -pgo=cmd/qualitybench/default.pgo -o "$c" "./cmd/$c"
done
~~~

dav1d helper (static 1.5.4 build from the pinned commit; `$AOM` is the pinned
libaom checkout used for its headers and `common/md5_utils.c`):

~~~sh
meson setup d.build "$DAV1D_SRC" --buildtype=release -Ddefault_library=static \
  -Dbitdepths=8,16 -Dc_args=-O3 -Denable_tools=false -Denable_tests=false \
  -Denable_examples=false && ninja -C d.build
cc -O3 -DNDEBUG -std=c11 -Wall -Wextra -Wconversion \
  '-DGOAV1_BUILD_FLAGS="-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wconversion"' \
  '-DGOAV1_AOM_SOURCE_COMMIT="047d8cf6168feafe1300eb6902000dd1a03d5549"' \
  '-DGOAV1_DAV1D_REFERENCE_COMMIT="54706fc6bc0cdecab7e9593974a4039cc038fca7"' \
  -DGOAV1_DECODER_DAV1D -I"$AOM" -I"$DAV1D_SRC/include" -Id.build/include \
  -Id.build/include/dav1d tools/c_api_decode_bench/bench.c "$AOM/common/md5_utils.c" \
  d.build/src/libdav1d.a -pthread -o c-api-decode-bench-dav1d
~~~

Decoder API sampling, repeated and interleaved:

~~~sh
env GOAV1_PUBLIC_FULLCLIP_SAMPLES=1 GOAV1_BENCH_CORPUS_DIR=testdata/benchcorpus \
  GOMAXPROCS=1 GOGC=100 ./public-e2e.test \
  -test.run '^TestPublicDecoderFullClipCorpusSamples$' -test.count=1 -test.v
./c-api-decode-bench-dav1d --corpus testdata/benchcorpus
~~~

Process-inclusive decode (per stream and config, `time -l` reports wall, user,
sys, instructions retired, cycles; repeat six times, interleaved, keep the
minimum; verify MD5s first with `md5 -q` of the same decode against the sidecar):

~~~sh
/usr/bin/time -l ./aom-go-dec -quiet -workers 1 -framedelay 1 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 1 --framedelay 1 --muxer yuv -i X.ivf -o /dev/null
/usr/bin/time -l ./aom-go-dec -quiet -workers 8 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 8 --muxer yuv -i X.ivf -o /dev/null
dav1d -q --filmgrain 1 --muxer md5 -i X.ivf -o -     # verification only
~~~

Old-protocol encoder comparison: see
[encoder performance](encoder-svt-performance.md#reproduce); the manifest has
one row per clip (`clip,input,width,height,frames,fps,pix_fmt,sha256`) pointing at
the raw files, and the recorded run used the same flags with `-svt-bin`
`~/bin/svt410/SvtAv1EncApp`, `-go-bin` the go1.27.2 executable, and one-clip,
two-rate chunks (`-bitrates 3000000,6000000` then `9000000,12000000`).

RTC recipe (`KBPS` is 1000 or 2000, `LP` is 1 or 8; the Go bitrate flag is bits
per second; Go 1 thread runs under `GOMAXPROCS=1`, 8 threads uses `-threads 8`):

~~~sh
/usr/bin/time -l ./encbench -input clip.yuv -frames 240 -width 1280 -height 720 \
  -fps 30 -bitrate "${KBPS}000" -nopsnr -o go.ivf
/usr/bin/time -l SvtAv1EncApp -i clip.yuv -b svt.ivf -w 1280 -h 720 \
  --fps-num 30 --fps-denom 1 --frames 240 --input-depth 8 --color-format 1 \
  --profile 0 --level 0 --preset 12 --rc 2 --tbr "$KBPS" --buf-sz 1000 \
  --buf-initial-sz 500 --buf-optimal-sz 600 --lookahead 0 --pred-struct 1 \
  --rtc 1 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1 --progress 0 --lp "$LP"
dav1d -q --muxer yuv -i go.ivf -o go.yuv   # then compare the Y plane with clip.yuv
~~~

## What changed since 20261009

`git log --oneline 0824b687..origin/main` lists 215 commits, 161 of them
numbered pull requests (#8–#172; #6 was squash-merged, so the range also counts
the earlier history of the #8 branch). The decoder levers that matter, by
profile weight:

- **Fused dav1d-style inverse transforms (arm64).** Two-pass transform with the
  add fused and eob-bounded work: 32x32 DCT (#138, #141), 16x16 and 8x8
  (#144, #148), 64x64 and full 32x32 (#147), DCT8 low-4 (#162), 16x32 and
  32x16 (#163), int16 DCT16/DCT32 column kernels (#99, #116) and dav1d's DC-only
  reconstruction (#56, #140).
- **Frame and multi-tile threading.** dav1d's frame-threading port: single-tile
  frames (#71), parse off the pool lock plus `-framedelay` (#72), reference
  slots published at Begin (#78), overlapped entropy passes (#81), multi-tile
  frames (#150), retained CDFs published right after the update tile (#152) and
  a lock-free wavefront row-progress fast path (#153).
- **Block, coefficient and entropy ports.** dav1d's `decode_coefs`/`read_coef_tree`
  (#50), refmvs and `decode_b` mode decode (#43), `read_tx_tree` (#128),
  branch-free symbol/sign/CDF paths (#102, #109), branch-free DC-sign contexts
  (#169), palette indices per diagonal (#172) and the vector large-alphabet
  symbol decode, dav1d's `msac_decode_symbol_adapt16` (#171).
- **Post-filters.** dav1d deblocking structure (#48) and mask-only deblocking
  (#145); **loop-filter level-0 skip**: no masks built for frames with deblocking
  off (#157); CDEF and restoration apply structure (#52) and **CDEF interior
  units in byte lanes**, as dav1d's edged 8-bit path (#160).
- **PGO refresh.** A current decoder CPU profile merged into
  `cmd/qualitybench/default.pgo` (#135) after the realtime-encoder profile
  (#106). Measured effect on the decoder sampler here: about 1–3%.
- **Correctness fixes from the real-content corpus** (`scripts/gen_real_corpus.sh`,
  #118, #143): palette edge contexts and on-screen index reads (#70, #129),
  skip-intra per-transform-block prediction (#119), segmentation-map allocation
  and propagation (#120), compound interpolation kernel selection (#122), film
  grain on `show_existing_frame` (#127), frame-size changes with per-reference
  scaling (#133), super-res scratch sizing (#137), aomenc good-mode keyframes
  with still-picture headers and intrabc (#142) and per-frame side data across
  tile groups (#151). Real streams from four encoders now decode byte-exact.
- **Encoder.** A broad port of SVT-AV1 realtime preset 12: light-PD1
  decisions and inter path (#40, #47, #51, #53, #57, #82, #87–#93), RTC rate
  control and the 3-layer low-delay default (#59), p12 partitioning, deblock and
  restoration levels (#60, #61), the CDEF strength search (#66, #67, #110),
  open-loop ME and HME (#62, #73, #79, #80, #83, #86), RDOQ (#44, #45),
  `TX_MODE_LARGEST` (#96), one tile by default and keyframes on the SB-row
  wavefront (#154), OBU_FRAME packaging (#149) and screen-content tools
  (#126, #131, #134, #158). Project notes from 2026-10-21 (not committed)
  recorded about 2.6× SVT CPU and -1.06 dB luma PSNR at 2 Mbps; this report
  measures 1.26× CPU and -0.013 dB on a similar clip, with the caveats above.

## Where the remaining gap is

Real-content decode at one thread is instruction-bound: 2.15× dav1d's retired
instructions at a better IPC. The visible outliers are film grain (4.26×
cycles), 10-bit (2.46–2.65×) and the multi-thread CPU overhead (Go uses 1.44×
dav1d's CPU at 8 workers and scales 3.2× against dav1d's 3.8×). Encoder: Go
retires 1.64× the instructions of SVT p12 for equal PSNR-Y, and over-spends
bitrate under the old qualitybench settings. No profile was taken in this
round, so this list is derived from the ratios, not from a flat profile.

## Evidence

- [Decoder API per-clip rounds](benchmarks/e2e-20261024-decoder.csv): Go cold,
  payload, warm, non-PGO, dav1d static and Homebrew, three rounds.
- [Decoder CLI per-stream minima](benchmarks/e2e-20261024-decoder-cli.csv),
  [all 1,440 timed runs](benchmarks/e2e-20261024-decoder-cli-samples.csv) and
  [MD5 verification, 60 streams](benchmarks/e2e-20261024-decoder-cli-verify.csv).
- [Encoder results](benchmarks/e2e-20261024-encoder.csv): both rounds, attained
  bitrate, throughput, CPU time and quality.
- [RTC recipe timings](benchmarks/e2e-20261024-rtc.csv) (all 96 runs) and
  [quality](benchmarks/e2e-20261024-rtc-quality.csv).
- [Provenance, hashes and commands](benchmarks/e2e-20261024.json).

## Validation

The measured checkpoint is `main` after PR #172 (CI gates apply per PR); this
report ran no test suites, only the corpus MD5 checks above: 18 of 18
benchcorpus clips in the Go sampler and C helper preflights, and all 60 streams
in four decoder configurations. Encoder outputs were decoded by dav1d/libdav1d
for scoring. Native ARM64 measurement only; it does not establish AMD64 or
Linux performance. The corpora are exploratory: the benchcorpus is one scaled
source, the real corpus is nine Xiph sources plus a test pattern. The old
report's remaining limits (no equal-quality claim, two conference clips) still
apply to the old-protocol encoder table.
