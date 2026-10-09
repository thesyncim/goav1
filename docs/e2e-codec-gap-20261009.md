# End-to-end codec gap: October 9, 2026

Measured PR #6 implementation checkpoint `0824b687ea02c9b0f8bd49b327c8dd4acbfa805b`,
using official Go 1.27.2 with `GOEXPERIMENT=simd` on Apple M4 Max / macOS.
These are exploratory measurements on a shared workstation, not an isolated
benchmark machine. Another project's test process used several cores during
sampling. Do not compare these absolute times with earlier runs to infer a
regression or improvement.

## Decoder: public API versus dav1d

Both rounds independently verified all 18 clips / 864 visible frames against
the same stream-MD5 sidecars. The dav1d input SHA-256 values also matched the
Go corpus files. Each clip has 48 frames; each path uses one decoder worker,
one warmup and nine full-clip samples. Aggregate time is the sum of the
per-clip medians, not the median of whole-corpus runs.

| Path | Round 1, ms | Round 2, ms |
| --- | ---: | ---: |
| Go cold IVF constructor + decode + close | 2085.691 | 2173.100 |
| Go cold pre-parsed payload constructor + decode + close | 2085.559 | 2182.154 |
| Go warm reset + decode | 2046.862 | 2120.712 |
| dav1d fresh C API context + decode + close | 571.028 | 597.002 |
| Go cold-IVF / dav1d time ratio | 3.653× | 3.640× |

**The observed decoder gap is about 3.65× time, not parity.** Go would need
roughly 72.5% less time to reach this dav1d baseline. Pre-extracting packet
payloads does not materially close the gap. These measurements cover full
decode, not just a SIMD kernel.

File reads, hash checks and MD5 preflight are excluded. Go's IVF constructor
copies packet payloads; dav1d references the preloaded bytes. The pre-parsed
Go path excludes those copies. Warm Go reset has no matched C reset result.
This is in-process API end-to-end timing, not CLI process startup or disk I/O.
The generated decoder corpus has no source manifest and is not diverse enough
to support a general codec ranking.

dav1d is the local optimized static ARM64 build from pinned commit
`b546257f770768b2c88258c533da38b91a06f737`, runtime
`1.5.3-0-gb546257`, with assembly enabled and `-O3`. The helper SHA-256 is
`08f1350db97f0389bbff6f96623f11e189d9aeeefadc4a85ebb71924897a1a6d`.
The helper's generic JSON metadata marks binary provenance unverified;
the separate local build configuration records the source, archive hash,
compiler flags and executable hash. This is recorded provenance, not an
independently attested reproducible build.

## Reproduce decoder sampling

Build once outside timing, using the project cache wrapper. Run the existing
public sampler and pinned C helper sequentially; repeat for a second round.

~~~sh
env CODEX_AGENT_ID=e2e-gap GOTOOLCHAIN=go1.27.2 GOEXPERIMENT=simd \
  /Users/thesyncim/.codex/bin/project-env go test -c -o /tmp/public-e2e.test .
env GOAV1_PUBLIC_FULLCLIP_SAMPLES=1 GOAV1_BENCH_CORPUS_DIR="$CORPUS" \
  GOMAXPROCS=1 GOGC=100 /tmp/public-e2e.test \
  -test.run '^TestPublicDecoderFullClipCorpusSamples$' -test.count=1 -test.v
"$DAV1D_C_API_HELPER" --corpus "$CORPUS"
~~~

See `tools/c_api_decode_bench/README.md` for building the C helper. Preserve
the reference build configuration rather than substituting an arbitrary
installed decoder. Keep build activity and other benchmarks out of timed runs.
For strong claims, repeat on a quiet host and a source-manifested diverse corpus.

## Encoder: process-inclusive versus SVT-AV1

Two complete rounds covered FourPeople and Johnny: 1280×720, 120 frames,
60 fps, 8-bit I420, at 3/6/9/12 Mbps. Each clip/rate/encoder combination has
one warmup and three measured process-inclusive runs, interleaved by sample
pass with shuffled order and seed 1. The Go executable is prebuilt, using
the repository's default qualitybench PGO profile; compilation is not timed.
Raw loading, setup, encode, output writes and process startup are timed.
Reference decode and quality metrics are not timed. Go decision statistics
are enabled, as in the existing comparison protocol.

Go uses GOMAXPROCS=1, MaxThreads=1 and effort 0. SVT-AV1 4.0.1 uses preset 13,
LP=1 and assembly `max`. Both use low-delay CBR with no lookahead, scene-cut
detection or temporal filtering. LP is a parallelism level, not a thread
count; measured CPU/wall ratios are close to one for both encoders.

| Clip | Target Mbps | Go fps, R1 / R2 | SVT fps, R1 / R2 | Go/SVT time, R1 / R2 |
| --- | ---: | ---: | ---: | ---: |
| FourPeople | 3 | 75.98 / 78.58 | 369.00 / 317.73 | 4.857× / 4.043× |
| FourPeople | 6 | 77.16 / 69.76 | 285.75 / 259.90 | 3.703× / 3.726× |
| FourPeople | 9 | 70.85 / 62.74 | 257.42 / 220.22 | 3.633× / 3.510× |
| FourPeople | 12 | 65.35 / 58.11 | 231.95 / 204.91 | 3.549× / 3.526× |
| Johnny | 3 | 72.40 / 75.09 | 308.80 / 319.74 | 4.265× / 4.258× |
| Johnny | 6 | 65.80 / 66.47 | 252.56 / 249.51 | 3.838× / 3.754× |
| Johnny | 9 | 60.38 / 60.20 | 220.51 / 212.22 | 3.652× / 3.525× |
| Johnny | 12 | 56.07 / 56.07 | 193.00 / 191.70 | 3.442× / 3.419× |

The median of these 16 time ratios is **3.678×**, with a range of
**3.419–4.857×**. The visible round-to-round drift means these are approximate
workstation observations, not confidence-bounded speed claims. They do not
reproduce the smaller gap in the historical encoder table; exact encoder
settings and both encoded/reconstructed hashes match those earlier runs,
but the timing environment differs. Do not infer a code regression from
cross-session timings.

| Clip | Target Mbps | Actual Mbps, Go / SVT | PSNR dB, Go / SVT | SSIM, Go / SVT |
| --- | ---: | ---: | ---: | ---: |
| FourPeople | 3 | 2.996 / 3.019 | 43.6694 / 43.4732 | 0.9841 / 0.9853 |
| FourPeople | 6 | 6.000 / 6.002 | 45.4854 / 45.2783 | 0.9868 / 0.9887 |
| FourPeople | 9 | 9.003 / 8.807 | 46.3141 / 45.9792 | 0.9877 / 0.9898 |
| FourPeople | 12 | 12.050 / 11.486 | 47.0016 / 46.4360 | 0.9890 / 0.9903 |
| Johnny | 3 | 2.988 / 2.909 | 45.2945 / 44.7765 | 0.9863 / 0.9866 |
| Johnny | 6 | 5.976 / 5.909 | 46.9032 / 46.2228 | 0.9890 / 0.9896 |
| Johnny | 9 | 9.012 / 8.703 | 47.7472 / 46.8638 | 0.9901 / 0.9907 |
| Johnny | 12 | 12.045 / 11.334 | 48.4096 / 47.2124 | 0.9911 / 0.9912 |

Every measured encode succeeds with stable encoded and reconstructed SHA-256
values across its three samples and across both rounds. Go has higher PSNR;
SVT has higher SSIM and sometimes undershoots the target bitrate. Equal target
bitrate is **not equal quality**. No equal-quality speed or BD-rate claim is
made. These two short conference clips do not establish a general ranking.

Use the command in [encoder performance](encoder-svt-performance.md#reproduce),
with the newly built `qualitybench` binary. Split the manifest into one-clip,
two-rate chunks to limit disk usage. Repeat the four chunks for a second round.
SVT executable SHA-256 remains
`db044194faf94d553c2509c8e033ecd05a373ba54d919d0fb49a3a13149d1687`.

## Next measured targets

A separate public-decoder CPU profile covers both `p720_inter_q32` and
`p720_inter_q32_2tiles`, cold fresh-decoder paths, with GOMAXPROCS=1. Its 21.35
seconds of CPU samples show about 56% cumulative time in the tile block loop
and 33% in postfilters. Prediction-mode parsing accounts for about 12.5%
cumulative time; coefficient processing about 16.1%. These inclusive figures
overlap and must not be summed. Largest individual flat symbols include
Wiener horizontal filtering (3.3%), horizontal motion convolution (3.2%),
entropy CDF updates (2.3%) and DCT32 columns (2.0%). The shipped SIMD paths
appear in the profile; the gap is not just dispatch bypassing SIMD.

The fresh API paths also allocate roughly 39 MB and 203–206 allocations per
48-frame clip in this profile. This is not a zero-allocation kernel claim.
Prioritize block/entropy/coefficient overhead and the entire postfilter
pipeline, then verify each change with full-clip A/B timing and MD5 parity.
There is no single small kernel whose removal alone can close a 3.65× gap.

## Evidence

- [Decoder per-clip medians](benchmarks/e2e-20261009-decoder.csv): both rounds.
- [Encoder results](benchmarks/e2e-20261009-encoder.csv): both rounds, attained
  bitrate, throughput, CPU time and quality.
- [Raw samples and provenance](benchmarks/e2e-20261009.json): decoder samples,
  hashes, encoder commands/settings, all measured process samples and the
  decoder flat profile. Temporary full logs are in `/tmp/pr6/e2e-20261009`.

## Validation

The full SIMD and non-SIMD/default Go suites pass at the measured checkpoint.
The current benchmark binaries were built from that checkpoint before timing.
Motion and loop-filter parity, race, exact-window/checkptr checks and the
864-frame MD5 corpus check also passed in the preceding optimization round.
Native ARM64 measurement does not establish native AMD64 performance.

At the measured checkpoint, two AMD64 assembly files remained: four live
DCT32/DCT64 transform kernels and CPUID/XGETBV. Commit `d985372a` subsequently
replaced those final sources with Go SIMD and official CPU queries. This
report's E2E measurements remain pinned to `0824b687`; they are not new
performance measurements of the replacement. See the
[assembly-removal evidence](assembly-removal-20261009.md) for its separate
validation and emulated microbenchmark limitations. Native AMD64 execution
and competitive E2E performance remain distinct gates.
