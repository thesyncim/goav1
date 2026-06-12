# SIMD Coverage Against SVT-AV1

This ledger tracks whether goav1 has the same useful assembly coverage as the
SVT-AV1 build used for speed comparisons. It is intentionally profile-driven:
matching every upstream file name is not the goal; matching the active kernels
that matter for this encoder, and not giving SVT an unreported SIMD tier
advantage, is the goal.

## Source Truth

- Local SVT CLI: `/opt/homebrew/bin/SvtAv1EncApp`
- Version: `SVT-AV1 v4.0.1 (release)`
- Pinned source audit: `third_party/upstream/svt-av1`, tag `v4.1.0`, commit
  `c04f951541ad600e0d9c10836f2ab7b9bc69816d`
- Upstream repository: <https://gitlab.com/AOMediaCodec/SVT-AV1>
- Local platform for this audit: `darwin/arm64`, Apple M4 Max
- macOS-reported Arm features: `hw.optional.neon=1`,
  `hw.optional.arm.FEAT_DotProd=1`, `hw.optional.arm.FEAT_I8MM=1`

SVT v4.0.1 exposes `--asm` values
`c, neon, crc32, neon_dotprod, neon_i8mm, sve, sve2, max`; its default is
`max`. A goav1 vs SVT row with SVT default `max` is a best-SVT row, not a
baseline-NEON-equivalent row on this machine. Use `qualitybench -svt-asm neon`
for baseline-NEON rows.

SVT's `--lp` is not a processor/thread count. The installed CLI describes it as
an "Amount of parallelism" level with range `[0, 6]`, where `0` lets SVT choose
from the machine. Therefore `GOMAXPROCS=4` and `--lp 4` are not equivalent
knobs; fair reporting must include wall time, CPU time, observed parallelism,
`--lp`, and `--asm`.

## Current Speed Snapshot

Fresh synthetic 1080p/120-frame single-rate row on 2026-06-12 with
`GOMAXPROCS=4` for goav1 and SVT default `--asm max`:

| Encoder | FPS | Wall s | CPU s | Observed parallelism | Frames/CPU-s |
| --- | ---: | ---: | ---: | ---: | ---: |
| goav1 | 106.48 | 1.127 | 3.998 | 3.55x | 30.0 |
| SVT-AV1 `--lp 0 --asm max` | 196.81 | 0.610 | 1.520 | 2.49x | 78.9 |
| SVT-AV1 `--lp 4 --asm max` | 198.57 | 0.604 | 1.527 | 2.53x | 78.6 |

CPU-normalized gap: SVT is about 2.62x more CPU-efficient on this smoke row.
That gap should be reported by CPU seconds or frames/CPU-s, not only by wall
FPS. Wall FPS is noisier because the two encoders do not consume the same
parallelism, and `--lp 4` is not semantically equivalent to `GOMAXPROCS=4`.

Earlier same-shape control with SVT pinned to baseline NEON via
`-svt-asm neon`:

| Encoder | FPS | Wall s | CPU s | Observed parallelism | Frames/CPU-s |
| --- | ---: | ---: | ---: | ---: | ---: |
| goav1 | 105.32 | 1.139 | 4.018 | 3.53x | 29.9 |
| SVT-AV1 `--asm neon` | 178.34 | 0.673 | 1.626 | 2.42x | 73.8 |

Pinning SVT to baseline NEON still leaves SVT about 2.47x more CPU-efficient.
The remaining gap is therefore not only DOTPROD/I8MM.

## Coverage Ledger

| Area | SVT SIMD coverage | goav1 status | Decision |
| --- | --- | --- | --- |
| CPU feature tiers | `ASM_NEON`, `ASM_NEON_DOTPROD`, `ASM_NEON_I8MM`, `ASM_SVE`, `ASM_SVE2` | arm64 detects NEON plus Darwin DOTPROD/I8MM/SVE/SVE2 feature bits, but dispatch still uses only the baseline NEON tier | Do not claim max-tier parity until DOTPROD/I8MM kernels are wired and measured. Pin SVT with `-svt-asm neon` for baseline-tier rows. |
| TXB coefficient prep and contexts | `encodetxb_neon.c`: `svt_av1_txb_init_levels_neon`, `svt_av1_get_nz_map_contexts_neon`; `av1_quantize_neon.c`: `svt_av1_compute_cul_level_neon` | No assembly. Hot Go writer has a trusted 8x8 path, stack level buffer, fixed CDF storage, and branchless sign extraction. | High priority because profile points at coefficient/range coding. Prototype only narrow, measured kernels; previous nonzero-list and extra scan-table attempts regressed. |
| Range coder and CDF update | SVT does not make this a comparable named SIMD surface; arithmetic coding is serial | `WriteCDF4`, `normalize`, and `WriteBit` are top profile entries | Keep source-shaped Go unless a benchmark proves assembly beats call/setup cost. This is a hot scalar issue, not an SVT SIMD parity item. |
| SAD/search metrics | Broad SAD loops, PME SAD, external all/eight SAD, highbd SAD in `compute_sad_neon.c` and `sad_neon.c`; DOTPROD variants exist | `sad8x8`, `sad16x16`, `sad32x32`, `sad8x8Dual`, emitted rect sizes `16x8`, `8x16`, `32x16`, `16x32`, and the 8x8 compound-average precheck SAD have arm64 NEON coverage through direct or composed kernels; batched-candidate SAD remains scalar | High priority. Add measured batched-candidate kernels before more scheduler tuning; add DOTPROD only after runtime feature detection and profile proof. |
| Variance, SSE, block error, SATD, Hadamard | `variance_neon.c`, `sse_neon.c`, `block_error_neon.c`, `hadamard_path_neon.c`, plus DOTPROD SSE/variance | goav1 has residual/RD stats NEON, but not SVT's full metric surface | Medium-high. Implement only where the encoder actually uses the metric or where a mode-search change will use it. |
| Forward transforms | `highbd_fwd_txfm_neon.c` covers square, rectangular, N2/N4, and many tx types including ADST paths | Forward DCT 4/8/16/32 has NEON; forward ADST/other tx-type trial work is scalar | High for the current profile: `transform.fwdADST8` is visible in P-frame TX-type trials. |
| Quantize/dequant | FP/B quantize, 32x32/64x64 variants, highbd quantize | Quantize B/FP and dequant have NEON/AVX2 surfaces | Mostly covered for current 8-bit path; revisit after TXB/search gaps. |
| Inter prediction/convolve | Convolve, compound, joint compound, scale, warp, highbd, DOTPROD and I8MM variants | 8-bit and highbd X/Y/2D convolve have NEON/AVX2; compound paths reuse convolve/blend, no dotprod/i8mm tier | Baseline coverage is good, max-tier coverage is not equivalent. DOTPROD/I8MM should come after CPU feature detection and profiler confirmation. |
| Intra, CFL, blend, wedge, palette | Intra, CFL, blend, wedge, palette-related SIMD | Intra predictors, CFL, blend and min/max have NEON/AVX2 coverage; wedge/palette are feature-dependent | Low unless these paths become hot in the encoder mode set. |
| Loop filter, CDEF, restoration, superres, film grain | Broad NEON and highbd variants | goav1 has NEON/AVX2 coverage for these postfilter/decoder-style kernels | Mostly covered for current profile. Loopfilter still appears, so optimize only with a focused profile. |
| Temporal filtering, pic analysis, k-means, mem | SVT has NEON files for these encoder pipeline helpers | Not part of the current low-delay goav1 encode path | Do not chase for this benchmark until the feature exists and profiles hot. |
| SVE/SVE2 | SVT source has SVE/SVE2 directories | Apple M4 row does not report SVE/SVE2; goav1 has no SVE dispatch | Not relevant for this machine. Treat as future portability, not current fairness. |

## Go Hot-Path Hygiene Findings

- CDF state is fixed `uint16` storage, not oversized heap maps or slices in the
  hot writer path.
- Hot struct sizes are guarded by `TestHotStructSizes` and
  `TestWriterHotStructSize`, which catch accidental widening of
  tile/request/context carriers and the entropy writer.
- arm64 SAD hot call sites use build-tagged direct wrappers instead of calling
  through package-level function variables. The wrappers inline and compiler
  escape analysis reports their source/reference slices as non-escaping; the
  dispatch variables remain for backend parity tests and non-arm64 fallback.
- The trusted 8x8 coefficient writer stack-allocates its 144-byte level buffer.
  Larger scratch remains caller-owned or state-owned; forced reuse changes have
  already regressed and should not be repeated without benchmark proof.
- The 8x8 trusted coefficient count/write paths now use fixed stack arrays for
  trusted 64-coefficient input access and a 256-byte level buffer indexed by
  `uint8` padded offsets. Escape analysis still reports the hot inputs and
  scratch as non-escaping, and the compiler now elides the level-buffer bounds
  checks in the fill, lower-level context, BR context, and sign/golomb passes.
- Current coefficient writers accumulate `culLevel`, `dcValue`, and
  `maxScanLine` during level fill, then use branchless sign extraction in the
  sign/golomb pass.
- `WriteCDF4` uses a single symbol switch for the quaternary coefficient-base
  path and is guarded by `TestWriteCDF4MatchesWriteCDF`; the local writer
  stream benchmark is about `20.3-20.4 us` per 4096-symbol stream, zero
  allocations.
- Trial TXB pricing uses `entropy.NewCountingWriter`, preserving exact
  `Tell()` against the byte writer while skipping unused entropy byte
  materialization and removing the 16 KiB `trialBuf` scratch from lossy encoder
  state. `TestCountingWriterTellMatchesWriter` covers mixed bool/bit/CDF/symbol
  streams.
- The 8x8 inter TX-type trial path uses `entropy.BitCounter` through
  `tile.CountCoefficientsTXB8x8Y2DTrusted`, avoiding the generic writer carrier
  while preserving exact `Tell()` and CDF evolution. The tile microbench's
  count-only lane is now roughly `507-536 ns/op`, zero allocations on the local
  M4 Max; the sign/golomb pass avoids exact abs-level work unless the clamped
  level proves a Golomb tail is possible.
- TX-type trial snapshot restores are split by coefficient plane: luma-only
  candidates now restore only the 8x8 Y CDF subset, and UV CDFs are restored
  only after the candidate survives the luma rate bound. In the refreshed
  P-frame CPU profile, the hot pre-luma restore site is about `60 ms` cumulative
  versus the earlier monolithic restore line around `150 ms`; selector behavior
  is covered by the seeded reference tests and `BenchmarkChooseInter8x8TXType`.
- `motion.LumaSubpelProber` now indexes the regular 8-tap subpel tables
  directly for its 8-bit luma probes instead of re-entering the generic
  multi-filter selector per probe. `BenchmarkLumaSubpelProberPredict` improved
  in same-session A/B from roughly `64-77 ns/op` to `44-56 ns/op` for 8x8 and
  from roughly `161-189 ns/op` to `116-118 ns/op` for 16x16, with zero
  allocations.
- The 8x8 hybrid forward-transform path now dispatches directly to the
  array-backed DCT/ADST/IDTX kernels instead of copying through the generic
  slice helper for each row/column. `BenchmarkForwardBlockHybrid8x8` moved
  clean local ADST_DCT rows from roughly `630-738 ns/op` to `374-382 ns/op`,
  with ADST_ADST clean rows around `404-410 ns/op`; all rows remain zero
  allocations. This is scalar dispatch cleanup, not a replacement for forward
  ADST8 SIMD.
- Rectangular SAD for emitted inter block sizes now reuses existing 8x8/16x16
  NEON kernels. On the local M4 Max, `BenchmarkSADRect32x16` is about
  `14.5-14.7 ns/op` versus `239-244 ns/op` for the scalar reference, with zero
  allocations.
- The compound LAST/GOLDEN precheck now uses an arm64 NEON 8x8
  rounded-average SAD kernel (`urhadd` + widening absolute difference) instead
  of the scalar per-pixel average loop. `BenchmarkSAD8x8CompoundAvgBlock` is
  `6.36-6.41 ns/op` versus `51.70-52.58 ns/op` for the scalar reference, with
  zero allocations; randomized tests compare the dispatch kernel against the
  pure-Go reference across independent strides and offsets.
- Current P-frame diagnostics: the profiled
  `BenchmarkVideoEncoderPFrame1080p-4` sample is `92.3 ms/op`, with repeat
  rows around `88.4-92.1 ms/op` and zero or one tiny allocation. Allocation is
  not the dominant CPU gap; the CPU profile is.

## Next Implementation Order

1. Run every SVT comparison in two rows: best SVT (`-svt-asm max` or omitted)
   and baseline NEON (`-svt-asm neon`). Report both `--lp` and `--asm`.
2. Prototype a tiny TXB prep kernel only if it replaces work already proven hot:
   eob/level-buffer/stat extraction for the 8x8 luma path. Keep it only if both
   coefficient microbenchmarks and the fair row improve.
3. Add encoder search metric kernels with direct profile mapping:
   batched candidates.
4. Add forward ADST8/tx-type trial SIMD before broad transform-surface work.
5. Use the arm64 DOTPROD/I8MM feature metadata already detected by goav1 to
   decide whether convolve, SAD, or CDEF variants make sense relative to SVT
   `max` rows.

The rule for adding assembly is the same everywhere: pure-Go/reference parity
test first, zero allocations, focused benchmark, then fair `qualitybench` row.
