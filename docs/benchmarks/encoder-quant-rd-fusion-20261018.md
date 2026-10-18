# Fuse quantization and skip statistics

The ARM64 row-mask quantizer now collects the immediately following AC distortion/rate statistics in its existing 8/16/32-wide groups. DC correction and the original int32-before-widening arithmetic remain exact. Unsupported layouts and aliases take the existing path before writes. Search, quantization decisions, generic kernels and entropy updates are unchanged. Implementation uses Go/archsimd.

One frozen screen on Darwin/ARM64 Apple M4 Max, Go 1.27.0 with GOEXPERIMENT=simd and default PGO, measured the complete encoding helper process. Each affected case encodes 120 frames of 1280×720 at 60 fps with effort 0, MaxThreads=1, GOMAXPROCS=1, GOGC=100, one layer, no tiles, scene-cut enabled and statistics disabled. Baseline is main c07ac861; the measured source is a2f88592. Three alternating AB/BA/AB pairs follow four warmup rows. All 18 rows, including the separate eight-keyframe constructor control, are retained in the evidence JSON.

| Case | Paired wall saved | Paired CPU saved | Faster pairs |
| --- | ---: | ---: | ---: |
| FourPeople 3 Mbps | 0.992053% | 0.703105% | 3/3 |
| Johnny 9 Mbps | 0.398880% | 0.321846% | 3/3 |
| Eight-keyframe control | -0.332791% | 0.391477% | 0/1 |

The predeclared gate required one affected case to save at least 0.5% paired geometric-mean wall time with two faster pairs, and the other to regress at most 1% wall and CPU. It passed. The statistic is `100*(1-exp(mean(log(candidate/baseline))))`; controls are descriptive. No row was excluded or unchanged candidate retried. All 18 emitted bitstreams independently rehash to the fixed prior hashes and sizes. Source/binary/protocol identities match before and after; root independently recomputed the statistics. Agents were idle and the host checks were quiet, but unrelated host activity is uncontrolled.

Independent numeric coverage includes 2027 cases, complete destination/scratch state, aliases, fallback and error behavior. SIMD and pure-Go gates, zero-allocation coverage, and public 20-frame L1T1/L1T2 bytes plus reconstruction passed before timing. Existing evidence was reused. Generated ARM64 shows inline per-group statistics with no loop calls or spills. The kernel frame is 112 bytes versus 128; the caller grows 176 to 192 bytes and retains one indirect dispatch plus guards. SVT still computes distortion separately: this removes a duplicate Go traversal rather than claiming all native math disappears.

The publication cherry-pick is based on main 0a47ce2b; all five production/test files are byte-identical to the timed candidate. The intervening changes affect decoder CDEF and its test-only size guard. No timing was repeated. [Evidence](encoder-quant-rd-fusion-20261018.json) includes raw rows and source identities.

The last full native comparison, before later optimizations, measured Go/dav1d decode 2.65255× and Go/SVT encode 2.85944× elapsed time. Encoder settings were matched, not exact quality/search; Go PSNR was higher and SSIM lower. Those ratios are historical and are not updated by this two-clip screen. Installed native binary provenance was not attested to the source audit pins. There is no matched native-stage timing attribution.
