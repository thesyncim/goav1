# End-to-end codec gap: November 23, 2026

Measured at `9451293c` (PR #307) with official Go 1.27.4 and
`GOEXPERIMENT=simd` on Apple M4 Max (12 P + 4 E cores, 64 GB) / macOS 26.3.1,
against dav1d 1.5.4 (Homebrew) and SVT-AV1 4.1.0 built from the vendored
source. Every binary was built with `-pgo=cmd/qualitybench/default.pgo`. This
supersedes the [October 24 report](e2e-codec-gap-20261024.md), which stays as
history and describes the protocols in full.

These are process-inclusive CLI measurements (`/usr/bin/time -l`, both
decoders writing YUV to `/dev/null`) on a shared workstation with a load
average of 3 to 6. Each number is a minimum over interleaved repetitions;
cycles and instructions are exact, wall and CPU time resolve 10 ms. Do not read
small differences as regressions or improvements.

| Measure | 20261024 (`6d62eb88`) | 20261123 (`9451293c`) |
| --- | ---: | ---: |
| Decoder, real-content corpus, 1 thread, CPU time: Go / dav1d | 1.595× (42 streams) | **1.329×** (42 streams) |
| Decoder, real-content corpus, 8 threads, wall: Go / dav1d | 1.868× (CPU 1.439×) | **1.158×** (CPU 1.204×) |
| Encoder, RTC preset 12, 1 thread, CPU time: Go / SVT | 1.21–1.26× (instructions 1.64×) | **1.04–1.06×** (instructions 1.44–1.53×) |
| Encoder, RTC preset 12, default threads, wall: Go / SVT | not measured | **0.55×** (cycles 1.21×) |

**Neither is ahead on one thread.** The decoder runs about 1.85 times dav1d's
instructions at a higher IPC, which comes to 1.2 to 1.4 times its cycles; the
encoder is within 6% of SVT-AV1's cycles. With threads the encoder finishes in
about half SVT-AV1's wall time, and the decoder is 16% behind dav1d on the
corpus at 8 threads.

## Decoder: real-content corpus

The 42 streams of `testdata/realcorpus/streams` and
`testdata/realcorpus/aq0/streams` (aomenc good and realtime, SVT-AV1 realtime
and random access, goav1; 360p to 1080p, 300 frames each). Each cell is the sum
over streams of the per-stream minimum over three interleaved repetitions.
Every stream matches its dav1d MD5 at 1 and 8 workers.

| Config | Wall, s | CPU, s | Instr, G | Cycles, G |
| --- | ---: | ---: | ---: | ---: |
| goav1, 1 worker (`-framedelay 1`) | 34.53 | 34.33 | 519.9 | 90.66 |
| dav1d, 1 thread (`--framedelay 1`) | 26.13 | 25.83 | 280.6 | 68.58 |
| goav1, 8 workers | 7.71 | 42.63 | 546.1 | 109.58 |
| dav1d, 8 threads | 6.66 | 35.41 | 305.9 | 90.28 |

## Decoder: per stream

Cycles on one thread, and wall time with both CLIs at their defaults (one
worker per logical CPU), minimum of five interleaved repetitions.

| Stream | Tiles | 1 thread, cycles Go / dav1d | Default threads, wall Go / dav1d |
| --- | ---: | ---: | ---: |
| aomrt_crowdrun_1080p_4000k_s10 | 8 | 7.27G / 5.31G (1.37×) | 0.48 s / 0.25 s |
| svtrtc_crowdrun_1080p_3000k_p11 | 1 | 5.11G / 4.05G (1.26×) | 0.53 s / 0.45 s |
| goav1_crowdrun_1080p_3000k | 30 | 4.96G / 3.74G (1.33×) | 0.20 s / 0.78 s |
| aomrt_fourpeople_720p_1000k_s10 | 4 | 1.79G / 1.25G (1.43×) | 0.18 s / 0.13 s |
| svtra_mtbike_1080p_crf35_p8_2tile | 2 | 8.00G / 6.80G (1.18×) | 0.55 s / 0.51 s |
| aomrt_johnny_720p_1500k_s9 | 4 | 2.09G / 1.54G (1.35×) | 0.22 s / 0.17 s |

`goav1_crowdrun_1080p_3000k` was encoded by an older goav1 with one tile per
superblock column (30 tiles); dav1d spends 8.6 CPU seconds on it at 16 threads.
It is not representative: the encoder now codes one tile by default.

Where the single-thread gap is, on `aomrt_crowdrun_1080p_4000k_s10` (PMI cycle
samples, Mcycles Go / dav1d): coefficient decoding about 1300 / 670, mode
parsing 860 / 450, reconstruction glue 600 / 210, motion compensation
880 / 630, inverse transform 270 / 75, loop filter 790 / 680, CDEF
980 / 1280 (Go ahead). Two things bound it:

- Go 1.27.4's arm64 `simd/archsimd` has no absolute-difference,
  widening-accumulate or saturating-doubling-multiply operations (UABD, UABAL,
  UADALP, SMLAL, SQRDMULH). Motion compensation, inverse transforms and SAD
  need about twice the vector operations of the C and assembly references.
- The scalar entropy and parse code compiles to more instructions than dav1d's
  C. All remaining bounds checks together are about 5% of them.

With threads, single-tile streams are bounded in both decoders by the serial
entropy chain between frames. Multi-tile streams lose to dav1d on scheduling:
goav1 keeps more goroutines running than it has workers, and with `GOMAXPROCS`
at the core count a share of its cycles lands on efficiency cores. A port of
dav1d's pull-model worker loop is parked in PR #308; it needs a tile parse that
can stop and resume per superblock row before it pays.

## Encoder: RTC preset 12 recipe

`cmd/encbench` against `SvtAv1EncApp --preset 12 --rtc 1 --pred-struct 1
--rc 2`, 720p, 240 frames, minimum of five interleaved repetitions. PR #307
leaves the encoder's output byte-identical, so it does not move the quality
figures in [encoder-svt-performance.md](encoder-svt-performance.md).

| Config | SVT wall, s | Go wall, s | SVT cycles | Go cycles | Cycles | Instr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 thread, 500 kbps | 1.70 | 1.76 | 4.45G | 4.61G | 1.04× | 1.53× |
| 1 thread, 1000 kbps | 1.90 | 1.98 | 4.95G | 5.16G | 1.04× | 1.48× |
| 1 thread, 2000 kbps | 2.16 | 2.30 | 5.63G | 5.99G | 1.06× | 1.44× |
| default threads, 1000 kbps | 1.40 | 0.77 | 6.71G | 8.14G | 1.21× | 1.59× |

## Allocation and GC

Measured on the public API after warm-up: decoding a stream a second time makes
0 allocations at one worker and about a dozen (1.3 kB) per 300 frames at eight;
encoding makes 0 at one thread and about 0.6 per frame with threads, all of
them the Go runtime's wait records for blocked goroutines. Both CLIs run two to
five GC cycles per process, during start-up.

## Reproduce

~~~sh
export GOEXPERIMENT=simd
go build -pgo=cmd/qualitybench/default.pgo -o aom-go-dec ./cmd/aom-go-dec
go build -pgo=cmd/qualitybench/default.pgo -o encbench ./cmd/encbench

/usr/bin/time -l ./aom-go-dec -quiet -workers 1 -framedelay 1 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 1 --framedelay 1 --muxer yuv -i X.ivf -o /dev/null
/usr/bin/time -l ./aom-go-dec -quiet -workers 8 -o /dev/null X.ivf
/usr/bin/time -l dav1d -q --threads 8 --muxer yuv -i X.ivf -o /dev/null

GOMAXPROCS=1 ./encbench -input clip.yuv -frames 240 -width 1280 -height 720 \
  -fps 30 -bitrate 1000000 -nopsnr -o /dev/null
SvtAv1EncApp -i clip.yuv -w 1280 -h 720 --fps-num 30 --fps-denom 1 --frames 240 \
  --input-depth 8 --preset 12 --rc 2 --tbr 1000 --buf-sz 1000 \
  --buf-initial-sz 500 --buf-optimal-sz 600 --lookahead 0 --pred-struct 1 \
  --rtc 1 --scd 0 --enable-tf 0 --irefresh-type 2 --keyint -1 --progress 0 \
  --lp 1 -b /dev/null
~~~

Correctness of the measured binaries: the 42 corpus streams against dav1d at
1 and 8 workers, and `make feature-sweep` (178 aomenc and SVT-AV1 clips
against dav1d through both decoders).
