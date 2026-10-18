# Fuse deblocking level and threshold setup

The arm64 Go SIMD 8-bit superblock filter now expands each live unit's E/I/H thresholds while resolving its level. This removes the temporary four-level array and a second four-lane walk. Zero-initialized lanes retain zero thresholds. Level fallback, edge masks, statistics, pixel kernels, and vertical-before-horizontal order are unchanged.

The source reference is dav1d `b546257f770768b2c88258c533da38b91a06f737`, `src/loopfilter_tmpl.c:163–240`: each superblock entry resolves `L`, skips zero, then derives `H = L >> 4`, `E = lut->e[L]`, and `I = lut->i[L]`. The Go high-bit-depth setup already uses this single-pass shape. The candidate changes only `internal/av1/loopfilter/sb_u8_arm64.go`. Threshold lookup happens earlier for scalar fallback groups, adding unused setup there; the fallback behavior remains exact.

The inspected compiler output inlines the helper and removes the second setup loop. The horizontal body falls from 325 to 297 instructions and its frame from 272 to 256 bytes; the vertical body falls from 400 to 372 instructions with the same 624-byte frame. These are static diagnostics, including cold paths, rather than speed predictions.

## Public screen

Baseline main: `10f4a6ae956e881e8d9e85a4797c55a441cd27c3`, tree `6c0ed30fcaa5d62627d262c93ea8960f50666932`. Both frozen binaries used Go `1.27.0-X:simd`, Darwin/arm64, `GOARM64=v8.0`, and `CGO_ENABLED=1` on Apple M4 Max. Runs used `GOMAXPROCS=1`, `GOGC=100`, one decoder worker, and warm `Reset` + `DecodeNext`, with 48 frames per operation. Every process used 20 operations. Coordinated agent CPU work was paused for each timing window; other host activity was not controlled.

The declared screen was three alternating baseline/candidate pairs for `p720_inter_q32` and the affected `p360_intra_q32` comparison case. Acceptance required at least 0.5% paired geometric-mean inter latency savings, at least two faster inter pairs, no more than 1% intra point regression, and zero warm allocations. All intended outcomes are retained.

| Case | Paired latency saved | Faster pairs | Interpretation |
| --- | ---: | ---: | --- |
| p720 inter q32 | 1.8908% | 3/3 | Passes the declared target gate |
| p360 intra q32 | −0.5273% | 1/3 | Within the declared 1% regression guard |
| p720 inter q32, two tiles | 0.9073% | 2/3 | Unplanned, descriptive only |

The reported statistic is `100 * (1 - exp(mean(log(candidate_ns / baseline_ns))))` within each case. All 18 actual benchmark rows reported zero allocations; B/op varied between zero and two bytes, so there is no memory-reduction claim. Process-end one-minute host load ranged from 14.70 to 19.58. Three pairs are a small screen with no confidence-interval claim. This supports a narrow inter improvement on the measured machine, not an updated whole-corpus dav1d gap or an all-workload gain.

**Selection deviation:** Go splits benchmark expressions at `/`. The clip component in `^BenchmarkPublicDecoderFullClip/p720_inter_q32/warm_reset_decode$` was not individually anchored, so each inter process also ran `p720_inter_q32_2tiles` afterward. There were 12 processes and 18 actual rows, rather than the intended 12 rows. Original manifests and raw logs are preserved, all extra rows are included in the companion JSON, and selection still uses only the original target/comparison gate. Future exact selection must anchor each component, for example `^BenchmarkPublicDecoderFullClip$/^p720_inter_q32$/^warm_reset_decode$`. No rerun or postselection was used to repair the deviation.

## Byte and allocation checks

- 4,000 dispatched superblock cases matched the scalar reference, including masks, levels, sharpness, boundary fallbacks, all filter widths, and statistics. Another 3,000 scalar cases matched the existing edge kernels.
- Focused decoder checks passed for vertical-before-horizontal order, pooled filtering, high-bit-depth chroma, and allocations. Public loopfilter and compact-decoder warm-reset allocation checks passed.
- The frozen candidate passed the independent MD5 sidecars for all 18 clips / 864 frames, with both IVF and pre-parsed-payload constructors, at one worker.
- The existing single-thread encoder fixture passed both encoded-stream and reconstructed-plane SHA-256 expectations over 48 frames.

No production tests or expected bytes were changed. This arm64-only setup change does not alter the pure-Go/amd64 or high-bit-depth implementations. No new race/full-worker matrix or native performance campaign was run.

## Rejected CDEF experiments

Two preceding CDEF variants are absent from this change. V1 hoisted fixed rounding shifts but added two vector spill slots: inter −4.5220%, intra +3.4065%. V2 phased secondary taps and removed all SIMD spills in both 8- and 4-wide kernels: inter −0.1225%, intra +1.3580%. Both failed the declared inter gate and were rejected without resampling. Their correctness checks passed. Their source patches, binary identities, all actual timing rows, and the same regex deviation are preserved in the companion evidence; no CDEF gain is claimed.

The companion JSON contains the original protocol, all process logs and timing rows, binary/source identities, gate outputs, codegen diagnostics, and rejected-variant records.
