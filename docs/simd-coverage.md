# SIMD Coverage Against SVT-AV1

This ledger tracks whether goav1 has the same useful assembly coverage as the
SVT-AV1 build used for speed comparisons. It is intentionally profile-driven:
matching every upstream file name is not the goal; matching the active kernels
that matter for this encoder, and not giving SVT an unreported SIMD tier
advantage, is the goal.

## Source Truth

- Local SVT CLI: `/opt/homebrew/bin/SvtAv1EncApp`
- Version: `SVT-AV1 v4.0.1 (release)`
- Installed-binary source audit: SVT-AV1 tag `v4.0.1`, commit
  `4ae9272b588a05ee6e77a43e8dfdac05f54c4ff0`
- Repo architecture pin: `third_party/upstream/svt-av1`, tag `v4.1.0`, commit
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

## Coverage Policy

For the SVT comparison, "covered" means one of these must be true:

- goav1 has a same-tier assembly/SIMD kernel for the active encoder work.
- goav1 has a focused benchmark proving the scalar path beats or matches the
  assembly-shaped alternative on the target CPU.
- The SVT kernel belongs to a feature that this encoder does not execute in the
  measured mode; those entries stay marked out-of-scope, not covered.

SVT `--asm max` on this machine can select baseline NEON, DOTPROD, and I8MM
families. goav1 now detects the same arm64 feature bits and has measured
DOTPROD tiers for selected pixel-domain SSE/variance shapes and step-4 SAD
search groups, but convolve, CDEF, and other max-tier DOTPROD/I8MM surfaces
are still broader in SVT. Report max-tier SVT rows as best-SVT rows and
baseline `-svt-asm neon` rows as the closest assembly-tier control.

## SVT v4.0.1 ARM Inventory

The installed SVT binary matches tag `v4.0.1`, commit
`4ae9272b588a05ee6e77a43e8dfdac05f54c4ff0`. Its ARM SIMD source inventory is:

- Baseline NEON: transform, TXB/context prep, quantize/dequant, SAD/search,
  variance/SSE/block-error/Hadamard, convolve/inter prediction, intra/CFL,
  blend/wedge, CDEF, loop filter, restoration, superres-style resize helpers,
  temporal filtering, picture analysis, k-means, and memory helpers.
- DOTPROD: convolve, joint convolve, scaled convolve, CDEF, SAD/search, SSE,
  variance, corner matching, and picture analysis.
- I8MM: convolve, joint convolve, scaled convolve, intra prediction, and warp.
- SVE/SVE2: present in source, but not active for the local Apple M4 row.

The current high-priority coverage gaps for the measured goav1 encoder are
TXB prep/context extraction, forward ADST/hybrid transform trials, and any
search/RD metrics still proven hot after the existing SAD and residual-stat
NEON kernels. DOTPROD/I8MM convolve and CDEF are max-tier fairness gaps, but
they should follow a profile row that shows those paths are active enough to
matter in the measured encoder mode.

## Current Speed Snapshot

Fresh synthetic 1080p/120-frame single-rate rows on 2026-06-14 with
`GOMAXPROCS=4` for goav1. These rows include the SIMD safe points landed since
the older 2026-06-12 snapshot; do not attribute the full delta to any one
kernel. SVT `--lp` was swept from `0..6`; no max-tier row reached goav1's
observed `3.48x` CPU parallelism, so there is no true equal-CPU-budget row in
this sweep.

| Encoder | FPS | Wall s | CPU s | Observed parallelism | Frames/CPU-s |
| --- | ---: | ---: | ---: | ---: | ---: |
| goav1 | 137.48 | 0.873 | 3.037 | 3.48x | 39.5 |
| SVT-AV1 `--lp 0 --asm max` | 229.07 | 0.524 | 1.190 | 2.27x | 100.8 |
| SVT-AV1 `--lp 3 --asm max`, fastest max-tier row | 231.95 | 0.517 | 1.152 | 2.23x | 104.2 |
| SVT-AV1 `--lp 4 --asm max`, closest max-tier observed parallelism | 230.04 | 0.522 | 1.204 | 2.31x | 99.7 |

Best-SVT wall-clock gap is now about 1.69x in SVT's favor. The max-tier
CPU-normalized gap is still about 2.52x-2.64x by frames/CPU-s, depending on
whether the closest observed-parallelism row or fastest wall row is used. That
gap should be reported by CPU seconds or frames/CPU-s, not only by wall FPS.
Wall FPS is noisier because the two encoders do not consume the same
parallelism, and `--lp 4` is not semantically equivalent to `GOMAXPROCS=4`.

Same-shape control with SVT pinned to baseline NEON via `-svt-asm neon`:

| Encoder | FPS | Wall s | CPU s | Observed parallelism | Frames/CPU-s |
| --- | ---: | ---: | ---: | ---: | ---: |
| goav1 | 137.48 | 0.873 | 3.037 | 3.48x | 39.5 |
| SVT-AV1 `--lp 0 --asm neon` | 223.44 | 0.537 | 1.274 | 2.37x | 94.2 |

Pinning SVT to baseline NEON still leaves SVT about 1.63x faster by wall time
and about 2.38x more CPU-efficient. The remaining gap is therefore not only
DOTPROD/I8MM.

## Coverage Ledger

| Area | SVT SIMD coverage | goav1 status | Decision |
| --- | --- | --- | --- |
| CPU feature tiers | `ASM_NEON`, `ASM_NEON_DOTPROD`, `ASM_NEON_I8MM`, `ASM_SVE`, `ASM_SVE2` | arm64 detects NEON plus Darwin DOTPROD/I8MM/SVE/SVE2 feature bits. Metric dispatch selects a measured DOTPROD tier for winning pixel SSE/variance rows, and SAD/search dispatch selects DOTPROD for measured-winning step-4 x4 raster groups. The rest of the active arm64 dispatch remains baseline NEON unless another feature-tier kernel is proved. | Do not claim max-tier parity until DOTPROD/I8MM convolve, broader search, CDEF, and remaining metric surfaces are wired and measured. Pin SVT with `-svt-asm neon` for baseline-tier rows. |
| TXB coefficient prep and contexts | `encodetxb_neon.c`: `svt_av1_txb_init_levels_neon`, `svt_av1_get_nz_map_contexts_neon`; `av1_quantize_neon.c`: `svt_av1_compute_cul_level_neon` | No assembly. Hot Go writer has trusted 4x4/8x8/16x16/32x32 count-only trial paths, stack level buffers, fixed CDF storage, and recorded sign bits/nonzero bitsets on measured-winning count paths. | High priority because profile points at coefficient/range coding. Prototype only narrow, measured kernels; previous 8x8 extra scan-table and branchless sign rewrites regressed or tied after paired measurement. |
| Range coder and CDF update | SVT does not make this a comparable named SIMD surface; arithmetic coding is serial | `WriteBinaryCDFTrusted`, `WriteCDF4`/`WriteCDF5`/`WriteCDF7`, `normalize`, and `WriteBit` are top scalar cleanup entries. Fixed-arity writer/counter streams gate the exact count-only paths. | Keep source-shaped Go unless a benchmark proves assembly beats call/setup cost. This is a hot scalar issue, not an SVT SIMD parity item. |
| SAD/search metrics | Broad SAD loops, PME SAD, external all/eight SAD, highbd SAD in `compute_sad_neon.c` and `sad_neon.c`; DOTPROD variants exist | `sad8x8`, `sad16x16`, `sad32x32`, `sad8x8Dual`, `sad16x16Dual`, `sad32x32Dual`, emitted rect sizes `16x8`, `8x16`, `32x16`, `16x32`, the 8x8 compound-average precheck SAD, the current 8x8/16x16/32x32/64x64 full-pel raster x4 candidate groups, and generic four-reference 8x8/16x16/32x32/64x64 SAD counterparts to SVT's `sad8x8x4d`, `sad16x16x4d`, `sad32x32x4d`, and `sad64x64x4d` have arm64 NEON coverage through direct or composed kernels. DOTPROD now dispatches for measured-winning 16x16/32x32 step-4 x4 raster-search groups and generic four-reference 16x16/32x32 x4 groups; composed 64x64 x4 groups move through the active 32x32 x4 tier. Standalone 16x16/32x32 and dual-stride 16x16/32x32 DOTPROD SAD were tested but left undispatched because baseline NEON ties or wins after dispatch cost. | Baseline NEON coverage for current SAD/search probes is much closer, and active max-tier DOTPROD search surfaces are partially covered. Convolve, CDEF, and I8MM search-adjacent surfaces remain open until profile proof and measured kernels exist. |
| Variance, SSE, block error, SATD, Hadamard | `variance_neon.c`, `sse_neon.c`, `block_error_neon.c`, `hadamard_path_neon.c`, plus DOTPROD SSE/variance | goav1 has residual/RD stats NEON, baseline-NEON pixel SSE+variance stats for the practical 4-wide and width-multiple-of-8 square/rectangular low-bitdepth shapes through 64x64, a measured DOTPROD SSE/variance tier for winning width-multiple-of-8 rows, an arm64 NEON coefficient block-error reducer matching SVT's `svt_av1_block_error_neon`, an arm64 NEON coefficient SATD reducer matching SVT's `svt_aom_satd_neon`, and arm64 NEON 4x4/8x8/16x16/32x32 low-bitdepth Hadamard producers matching SVT's NEON order. These metric kernels are now dispatched by feature tier but still do not change encoder mode decisions unless that path is called. | DOTPROD SSE/variance is partially closed for measured-winning rows; small `8x4`/`8x8` variance remains baseline NEON because DOTPROD loses there. Baseline block-error is covered; broader max-tier metric/search coverage remains open. Only wire these into mode/search scoring after a focused profile proves they beat the current SAD/RD flow. |
| Forward transforms | `highbd_fwd_txfm_neon.c` covers square, rectangular, N2/N4, and many tx types including ADST paths | Forward DCT 4/8/16/32 has NEON; the active trusted 8x8 IDTX, ADST_DCT, DCT_ADST, and ADST_ADST tx-type trials now have arm64 NEON matching SVT's identity, ADST-column/DCT-row, DCT-column/ADST-row, and ADST-column/ADST-row surfaces. Larger/rectangular ADST and flip-ADST surfaces are still scalar or unsupported in this encoder mode set. | High for the current profile: the active 8x8 tx-type trial surface is covered, but SVT's broader forward-transform matrix still includes rectangular/N2/N4/flip ADST variants. |
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
- arm64 SAD/search dispatch now uses DOTPROD only for measured-winning
  full-pel step-4 x4 groups. Direct paired medians on Apple M4 Max were
  `16x16x4Step4` baseline NEON `14.03 ns/op` vs DOTPROD `13.26 ns/op`, and
  `32x32x4Step4` baseline NEON `52.15 ns/op` vs DOTPROD `47.26 ns/op`, all
  zero allocations. Standalone `16x16` DOTPROD tied baseline NEON and
  standalone `32x32` DOTPROD lost, so those direct SAD kernels remain on
  baseline NEON.
- Generic four-reference arm64 SAD x4 groups also have a DOTPROD tier now.
  Paired medians were `16x16x4` baseline NEON `16.93 ns/op` vs DOTPROD
  `16.72 ns/op`, and `32x32x4` baseline NEON `52.98 ns/op` vs DOTPROD
  `48.75 ns/op`, all zero allocations. The 16x16 win is small but repeatable
  over the longer 11-sample run; 32x32 is the clearer search-kernel win.
- Dual-stride arm64 SAD stays on baseline NEON. A DOTPROD prototype for
  `16x16Dual` and `32x32Dual` matched the pure-Go reference but did not justify
  dispatch: direct paired medians were roughly `6.62 ns/op` NEON vs
  `6.52 ns/op` DOTPROD for `16x16Dual`, while the active wrapper branch erased
  that tiny win; `32x32Dual` was a clear regression at roughly `16.89 ns/op`
  NEON vs `23.64 ns/op` DOTPROD. All rows were zero-allocation.
- The trusted 8x8 coefficient writer stack-allocates its 256-byte padded level buffer.
  Larger scratch remains caller-owned or state-owned; forced reuse changes have
  already regressed and should not be repeated without benchmark proof.
- The 8x8 trusted coefficient count/write paths now use fixed stack arrays for
  trusted 64-coefficient input access and a 256-byte level buffer indexed by
  `uint8` padded offsets. Escape analysis still reports the hot inputs and
  scratch as non-escaping; compiler BCE still leaves some scan/CDF/level checks
  in this path, so further changes need proof instead of assumption.
  The 8x8 trusted writer/counter now read a fixed packed scan table with
  `uint8` coefficient positions, `uint8` padded offsets, and `uint8` EOB
  lower/base-range contexts, so the trusted reverse pass no longer slices the
  generic scan metadata, reloads the position table, or branches through the
  generic EOB context helpers for the final coefficient.
- The trusted 8x8 coefficient APIs now expose pointer-shaped
  64-coefficient forms; the slice APIs remain compatibility wrappers. The 8x8
  TX-type selector and final luma writer use the pointer forms so the fixed
  shape is proven at the block owner instead of inside the tile helper on every
  trusted count/write call. Compiler reports keep inputs non-escaping and lower
  the generic inter TXB-pricing wrapper cost by isolating the 8x8 fast path;
  runtime rows remain pending.
- Final emitted 8x8 chroma TXBs now use a trusted Class2D writer with
  caller-derived txb-skip/dc-sign contexts, the packed 8x8 scan metadata, and a
  stack level buffer. The generic writer remains active for rectangular chroma
  and non-DCT paths; this is a scalar fixed-shape cleanup, not new ASM coverage.
- Final emitted 8x8 luma/chroma TXBs now cache exact absolute levels in a
  stack `[64]uint16` plus signs in a `uint64` bitset while filling the padded
  level window. The reverse base-range and sign/Golomb passes reuse that cache
  instead of reloading coefficient magnitudes/signs. Compiler reports lower the
  luma/chroma writer costs from `1654 -> 1648` and `1573 -> 1567`, keep hot
  inputs non-escaping, and remove two remaining bounds checks from each touched
  writer range. Focused runtime parity tests still stall locally, so this is not
  an fps claim.
- Final emitted 16x16 square luma/chroma TXBs now use trusted Class2D writers
  with caller-derived txb-skip/dc-sign contexts, packed 16x16 scan metadata, and
  stack level buffers. The luma path still writes the inter tx_type symbol at
  the same after-skip position when required; rectangular final writes remain on
  the generic writer.
- Trusted 16x16 luma/chroma count paths and the final 16x16 writer now cache
  exact absolute levels in a stack `[256]uint16` while filling the padded level
  window. The reverse base-range and Golomb passes reuse those magnitudes while
  signs are still read from the coefficient buffer; a wider abs+sign cache was
  rejected because it increased compiler cost. The kept magnitude-only form
  lowers compiler costs from `1721 -> 1707`, `1645 -> 1631`, and `1638 -> 1624`,
  keeps hot inputs non-escaping, and removes one/two remaining bounds checks in
  the touched ranges. This is scalar memory-traffic cleanup, not an fps claim.
- Trusted 4x4 luma/chroma count paths now use the same exact-level cache with a
  tiny stack `[16]uint16` plus a single sign bitset. Compiler reports lower both
  count-helper costs from `1645 -> 1639`, keep hot inputs non-escaping, and
  remove three remaining bounds checks from each touched range.
- Trusted 32x32 count and final writer reverse passes now use the existing
  padded/clamped level window for base-range coding instead of re-reading exact
  coefficient magnitudes. Exact magnitudes are still read for the Golomb tail,
  where they are required. This avoids a large `[1024]uint16` stack cache while
  removing one remaining bounds check in each touched range; compiler costs stay
  flat and hot inputs remain non-escaping.
- Final emitted 32x32 square luma/chroma TXBs now use trusted Class2D writers
  with caller-derived txb-skip/dc-sign contexts, packed 32x32 scan metadata, and
  stack level buffers. This covers 64x64 luma split children and 64x64 chroma
  square TXBs; rectangular final writes still use the generic writer.
- Current coefficient writers accumulate `culLevel`, `dcValue`, and
  `maxScanLine` during level fill, then use branchless sign extraction in the
  sign/golomb pass.
- P-frame full-pel motion grids now clear only the active frame rectangle for
  8x8/16x16/32x32/64x64 SAD caches, and 32x32 block coding uses the same
  sentinel guard as the other square sizes. This avoids clearing stale
  oversized backing capacity after resolution changes and prevents stale
  32x32 SAD reuse if a block reaches coding without a fresh partition score.
- Streaming keyframes now reuse the encoder's persistent tile-column workers
  and encoder-owned tile payload/error slices. This removes per-keyframe worker
  goroutine fanout and scratch allocation from periodic scene-cut/keyframe
  paths; the one-shot helper keeps its simple per-call path. This is a
  scheduler-overhead cleanup only, with no new FPS claim while local runtime
  binaries are still blocked before Go code starts.
- Persistent tile-column workers now grow only to the active job count for the
  current tile layout, so 2-/4-tile streams do not start the full 15-worker
  pool. HME pyramid rotation also uses an explicit queue swap instead of a
  per-frame defer. Both are scheduler/dispatch cleanups, not concurrency
  increases.
- A zero-input fast path for scalar `fwdADST8` was tested but not kept. SVT's
  `svt_av1_fwd_txfm2d_8x8_neon` batches ADST_DCT/DCT_ADST/ADST_ADST through
  `highbd_fadst8_xn_neon`; the Go scalar kernel has clean escape analysis but
  is too large to inline. The branch made synthetic zero rows much faster
  (`BenchmarkForwardADST8/zero` median `6.871 -> 1.773 ns/op`) but regressed
  dense ADST8 (`6.754 -> 7.050 ns/op`) and two of the three full trusted 8x8
  hybrid modes (`ADST_DCT` `267.5 -> 277.9 ns/op`, `ADST_ADST`
  `265.2 -> 273.0 ns/op`; `DCT_ADST` improved `295.2 -> 288.7 ns/op`).
  Because P-frame TX-type trials score nonzero residuals across all three ADST
  shapes, this stays scalar/source-shaped until a batched NEON-style 2D kernel
  is implemented and measured.
- `WriteCDF4` uses a single symbol switch for the quaternary coefficient-base
  path and is guarded by `TestWriteCDF4MatchesWriteCDF`; after byte-writer
  normalization moved to the 16-bit shift expression below, the specialized
  local writer stream benchmark is about `18.6-19.0 us` per 4096-symbol
  stream, zero allocations.
- Motion-vector writing now routes the four-symbol joint and fractional-pel
  CDFs through `WriteCDF4` instead of the generic adaptive CDF writer, matching
  SVT's same CDF source shape while using the local fixed-arity specialization.
  `BenchmarkWriteMotionVectorStream` moved from `173478 ns/op` mean /
  `172227 ns/op` median to `161569 ns/op` mean / `161805 ns/op` median for a
  2048-vector stream, zero allocations; the compiler reports the touched writer
  and CDF parameters as non-escaping and no `inter_write.go` bounds checks.
- `WriteCDF3` now covers the fixed three-symbol coefficient-base-EOB CDFs
  (`coeff_base_eob_cdf`, `CDF_SIZE(3)` in SVT). The stream microbench moved
  from `22633 ns/op` median generic writer to `18335 ns/op` specialized, and
  from `18331 ns/op` median generic counter to `17729 ns/op` specialized, zero
  allocations. On the 8x8 TXB bench, exact previous-commit A/B left the byte
  writer neutral/noisy (`trusted` median `500.0 ns/op` baseline versus
  `500.4 ns/op` patched) while the hot count-only lane improved from
  `422.9 ns/op` to `418.8 ns/op`.
- `WriteCDF5` now covers the fixed five-symbol 4x4 EOB-token CDF
  (`eob_flag_cdf16`, SVT writes it with symbol count `5`). The CDF5 stream
  microbench moved from median `39560 ns/op` generic writer to `18981 ns/op`
  specialized, and from `35058 ns/op` generic counter to `18268 ns/op`
  specialized, zero allocations. Same-session A/B on 4x4 coefficient benches
  kept both luma and chroma routes: luma trusted count moved from median
  `123.1 ns/op` to `120.4 ns/op`, chroma trusted count from `112.4 ns/op` to
  `110.3 ns/op`, and the luma final-write direct-tx path from `140.1 ns/op` to
  `137.7 ns/op`, all zero allocations. Compiler reports the new writer/counter
  receivers and CDF pointers as non-escaping; the 4x4 trusted wrappers keep
  their existing inline/non-escape shape.
- `WriteCDF7` now covers the fixed seven-symbol 8x8 luma EOB-token CDF
  (`eob_flag_cdf64`, SVT writes it with symbol count `7`). The CDF7 stream
  microbench moved from median `40876 ns/op` generic writer to `18997 ns/op`
  specialized, and from `36869 ns/op` generic counter to `18249 ns/op`
  specialized, zero allocations. Clean previous-commit A/B on
  `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` moved from median
  `421.1 ns/op` to `412.5 ns/op`, zero allocations; the compiler reports the
  new writer/counter receivers and CDF pointers as non-escaping. The matching
  UV route was tested but not kept because its trusted-count median moved from
  `439.2 ns/op` to `441.5 ns/op` in the all-plane patch.
- A nine-symbol specialization for the 16x16 EOB-token CDF was tested but not
  kept. SVT writes `eob_flag_cdf256` with symbol count `9`, and the isolated
  stream did improve (`BenchmarkWriterCDF9Stream` median `42510 -> 34143
  ns/op`, `BenchmarkBitCounterCDF9Stream` median `37841 -> 29474 ns/op`, zero
  allocations). The real 16x16 caller rows were mixed: all-route clean A/B
  regressed generic luma final write (`3015 -> 3038 ns/op`) and generic chroma
  count (`3140 -> 3151 ns/op`); after trimming to final trusted writes only,
  final-only A/B still moved luma trusted write from median `2757 -> 2770
  ns/op` while chroma improved `3096 -> 3063 ns/op`. The luma/count signal is
  too weak for a hot-path code-size increase, so `eob_flag_cdf256` remains on
  the generic CDF writer until a narrower caller shape proves out.
- `WriteBinaryCDFTrusted` specializes known two-symbol adaptive CDF writes for
  coefficient txb-skip/eob-extra/dc-sign syntax, inter reference bits, the
  single-reference inter-mode cascade, DRL bits, MV sign/class0/integer/HP bits,
  skip-transform, and tx-partition split bits. It is guarded by
  `TestWriteBinaryCDFTrustedMatchesWriteCDF`, and the count-only path is covered
  by the mixed-stream `TestCountingWriterTellMatchesWriter` gate.
- `BenchmarkWriterBitStream` now gates the direct equiprobable-bit writer and
  counter paths used by literal and coefficient-sign emission. The current
  source-shaped split expression measures about `15.04-15.32 us` for the byte
  writer and `14.33-14.47 us` for the bit counter per 4096-bit stream, zero
  allocations. A 16-bit masked split expression was rejected: paired rows were
  `15.16-15.76 us` for the byte writer and `14.31-14.54 us` for the bit
  counter, so it did not beat the existing formula.
- `BitCounter.normalize` keeps the count-only arithmetic source-shaped but now
  uses an `int32` ready-byte count and derives the post-flush bit count from the
  pre-flush count. It also computes the normalization shift with
  `bits.LeadingZeros16(uint16(rng))`, matching the 16-bit AV1 range width
  instead of using a 32-bit length expression. Compiler reports for entropy,
  tile, and encoder builds still inline it at all count-only call sites
  (`encodeQ15`, `WriteBoolQ15`, `WriteBit`, `WriteBinaryCDFTrusted`, and
  `WriteCDF4`) with no new heap escape.
- `Writer.normalize` now uses the same 16-bit normalization-shift expression
  for emitted bytes. Paired local repeats after temporarily restoring the old
  expression measured `BenchmarkWriterCDF4Stream/specialized` at
  `19.98-20.43 us` before and `18.60-19.01 us` after, zero allocations. The
  generic CDF4 lane traded off from `23.04-23.68 us` before to
  `23.70-24.16 us` after, so keep watching rectangular/generic writer use.
  `BenchmarkVideoEncoderPFrame1080p` stayed neutral to slightly faster:
  `74.90-75.37 ms/op` before and `74.56-75.01 ms/op` after, zero allocations.
- A source-shaped shift-before-cast CDF arithmetic variant matching SVT's
  `(fl >> EC_PROB_SHIFT)` spelling was retested but not kept. It helped the
  binary trusted stream only slightly (`19.04-19.23 us` versus `19.13-19.57 us`)
  while the hotter CDF4 writer and counter lanes were neutral to slower
  (`19.06-19.20 us` versus `18.94-19.08 us`, and `18.25-18.43 us` versus
  `18.25-18.42 us`).
- Single-reference and compound-reference frame selection now read their fixed
  write-side bit patterns from compact `uint8` tables and emit the symbols
  directly, avoiding the former single-ref per-call `[]refBit` construction and
  the compound-ref non-inlined helper-call ladder.
- Trial TXB pricing uses `entropy.NewCountingWriter`, preserving exact
  `Tell()` against the byte writer while skipping unused entropy byte
  materialization and removing the 16 KiB `trialBuf` scratch from lossy encoder
  state. `TestCountingWriterTellMatchesWriter` covers mixed bool/bit/CDF/symbol
  streams.
- 8x8 trial pricing uses `entropy.BitCounter` through
  `tile.CountCoefficientsTXB8x8Y2DTrusted` for luma and
  `tile.CountCoefficientsTXB8x8UV2DTrusted` for chroma, avoiding the generic
  writer carrier for the inter TX-type selector path, plain 8x8 Y trials, and
  8x8 UV trials from 16x16 block decisions. The TX-type luma path
  passes/restores the transform CDF; plain Y and UV trials pass no TX-type CDF.
  The tile microbench's count-only luma lane is roughly `507-536 ns/op`, zero
  allocations on the local M4 Max; the sign/golomb pass avoids exact abs-level
  work unless the clamped level proves a Golomb tail is possible.
- The hot 8x8 luma count-only sign/Golomb pass now records non-zero scan slots
  in a `uint64` while filling levels, then iterates those set bits in forward
  scan order. This keeps SVT's `av1_write_coeffs_txb_1d` symbol order but skips
  per-slot zero checks in sparse trial blocks. On the local M4 Max, a
  `BenchmarkVideoEncoderPFrame1080p-4` A/B run with `GOMAXPROCS=4`, `-cpu=4`,
  `-benchtime=8x`, and `-count=7` measured baseline `81.76/81.70 ms`
  mean/median and patched `80.31/80.23 ms`; the reversed order measured
  baseline `81.50/81.44 ms` and patched `80.60/80.00 ms`.
- Branchless sign-bit accumulation for the same 8x8 luma count-only level-fill
  pass was retested after the 16-bit `BitCounter.normalize` change. Early rows
  looked favorable (`420.1-428.4 ns/op` baseline versus `409.5-423.6 ns/op`
  branchless), but paired current-load repeats tied at `428.5 ns/op` baseline
  versus `428.4 ns/op` branchless. The branchless form was not kept.
- Fixed 4x4 trial TXBs now price through trusted count-only luma/chroma paths:
  `tile.CountCoefficientsTXB4x4Y2DTrusted` for luma trial costs and
  `tile.CountCoefficientsTXB4x4UV2DTrusted` for chroma. This keeps the common
  small-block trials out of the generic writer/scratch path while preserving
  exact CDF and `Tell()` evolution; runtime rows are still pending.
- 16x16 and 32x32 trial pricing now use trusted count-only luma/chroma paths:
  `tile.CountCoefficientsTXB16x16Y2DTrusted`,
  `tile.CountCoefficientsTXB16x16UV2DTrusted`,
  `tile.CountCoefficientsTXB32x32Y2DTrusted`, and
  `tile.CountCoefficientsTXB32x32UV2DTrusted`. This keeps larger split/merge
  decisions on the count-only path while preserving exact CDF and `Tell()`
  evolution against the generic writer. Square 16x16/32x32 inter TX-type trials
  also use trusted luma counters that write the after-skip transform-type symbol
  and restore the transform CDF, matching the existing 8x8 trial behavior.
  Larger non-square luma/chroma TXBs still use the generic counting writer until
  profiles justify the extra source-shaped specializations.
- The measured-winning 16x16 luma and 32x32 count-only sign/Golomb passes now
  record non-zero scan slots in stack bitsets while filling the padded level
  window, then iterate set bits in forward scan order. This mirrors the
  previously kept sparse 8x8 count shape without changing symbol order. On the
  local M4 Max, `BenchmarkCountCoefficientsTXB16x16Y2D/trusted-count` moved
  from `2404-2485 ns/op` to `2296-2399 ns/op`, `trusted-count-tx` moved from
  `2432-2520 ns/op` to `2369-2429 ns/op`,
  `BenchmarkCountCoefficientsTXB32x32Y2D/trusted-count` moved from
  `11330-11588 ns/op` to `10413-10438 ns/op`, `trusted-count-tx` moved from
  `11324-11468 ns/op` to `10425-10501 ns/op`, and
  `BenchmarkCountCoefficientsTXB32x32UV2D/trusted-count` moved from
  `11744-13189 ns/op` to `10776-10880 ns/op`; all rows stayed at
  `0 allocs/op`. The 16x16 UV bitset variant was tested and not kept because it
  moved from `2623-2671 ns/op` baseline to `2635-2717 ns/op`.
- P-frame 16x16 split trials and 8x8/16x16 merge pricing now call typed
  square-TXB rate helpers directly, avoiding the non-inlined generic
  `trialTXBBits(plane, []int16, n)` dispatcher on those known-shape luma/chroma
  decisions while preserving the same trusted coefficient counters.
- Merge pricing now splits the exact 8x8 child and 16x16 parent cost paths so
  their prediction, quantization, and coefficient-rate slices use fixed sizes.
  The compiler no longer reports the old `n*n` and `cn*cn` slice-bound checks
  in that exact-cost routine; the remaining checks there are motion-vector grid
  lookups for the four child blocks and one parent block.
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
- Subpel refine now has fixed 8x8, 16x16, and 32x32 paths for the common luma
  probes. The search order and quarter-pel acceptance rules remain the same,
  but the exact-probe SAD path no longer switches on `n` per candidate, and the
  compiler no longer reports the hot `sadScratch[:n*n]` bounds check on those
  sizes.
- The 8x8, 16x16, and 32x32 luma subpel prober entry points now bypass the
  generic `regularSubpelKernel(blockSize, phase)` branch on fractional probes
  and index the regular 8-tap kernel table directly. Compile diagnostics move
  the sized prober bodies to cost `421` versus the generic prober's `461` and no
  longer report the hot kernel-table bounds checks there; the cold integer-pel
  case still falls back to the generic copy path. The indirect convolve dispatch
  still limits escape precision, so this is a dispatch/BCE cleanup rather than a
  new allocation claim.
- The 8x8 hybrid forward-transform path now dispatches directly to the
  array-backed DCT/ADST/IDTX kernels instead of copying through the generic
  slice helper for each row/column. `BenchmarkForwardBlockHybrid8x8` moved
  clean local ADST_DCT rows from roughly `630-738 ns/op` to `374-382 ns/op`,
  with ADST_ADST clean rows around `404-410 ns/op`; all rows remain zero
  allocations. This is scalar dispatch cleanup, not a replacement for forward
  ADST8 SIMD.
- The trusted 8x8 hybrid path now switches once on the full tx type and runs
  concrete ADST_DCT, DCT_ADST, ADST_ADST, or IDTX loops. That removes the
  repeated per-row/per-column 1-D type switch while preserving the same scalar
  `fwdADST8`/`fwdDCT8` kernels.
- `fwdADST8` now keeps its staged butterfly in scalar temporaries with typed
  fixed cosine constants, avoiding the former intermediate `[8]int32` stage
  array and writes through the caller output buffer before the final
  permutation. The compiler now emits it as a leaf/no-frame function and
  reports no escaping inputs/outputs; this is scalar hygiene, not ADST SIMD
  parity with SVT's NEON forward-transform kernels.
- `BenchmarkForwardBlock8x8HybridTrusted` now gates the exact 8x8 trusted
  hybrid entry used under inter TX-type trials. On the local M4 Max, current
  rows are `270.6-280.4 ns/op` for ADST_DCT, `298.0-305.2 ns/op` for DCT_ADST,
  `271.3-279.1 ns/op` for ADST_ADST, and `25.85-28.57 ns/op` for IDTX, all zero
  allocations. SVT's matching 8x8 surface is still a real SIMD advantage:
  `svt_av1_fwd_txfm2d_8x8_neon` runs DCT/ADST/identity row and column kernels
  in vector batches, and the source also contains N2/N4 plus SSE4.1/AVX2
  variants for the same transform family.
- The encoder's 8x8 hybrid transform dispatch now switches once at the
  already-known 8x8 boundary and calls per-type trusted wrappers, avoiding the
  generic trusted full-type switch in the TX-type selector path. The wrappers
  inline into `forwardTransformBlock`, and escape analysis keeps
  coeff/residual/scratch non-escaping. `BenchmarkChooseInter8x8TXType` moved
  from `11287.6 ns/op` mean / `11315 ns/op` median to `11153.3 ns/op` mean /
  `11115 ns/op` median over seven local repeats, with zero allocations. This
  is a scalar dispatch cleanup, not a substitute for forward ADST8 SIMD.
- The encoder's 8x8 inter tx-type trial now calls a trusted 8x8 hybrid forward
  transform entry point after DCT_DCT has already taken the specialized DCT
  path. This skips the checked `ForwardBlock` shape/type dispatch on every
  ADST/IDTX trial while preserving the checked API and parity tests. Compiler
  reports still show the ADST body itself is too large to inline, so SVT's
  forward-transform SIMD gap remains open.
- The 8x8 IDTX forward-transform trial now writes the algebraic result
  (`residual << 3`) straight into the transposed coefficient layout while
  retaining the same caller validation. This removes the former identity
  column pass, round-shift, scratch writes, row pass, and scratch reads; the
  compiler now reports the IDTX helper as inlineable with non-escaping inputs.
- Rectangular SAD for emitted inter block sizes now reuses existing 8x8/16x16
  NEON kernels. On the local M4 Max, `BenchmarkSADRect32x16` is about
  `14.5-14.7 ns/op` versus `239-244 ns/op` for the scalar reference, with zero
  allocations.
- Square 64x64 SAD now composes four active 32x32 kernels instead of falling
  through the scalar square helper. This covers 64-tier merge scoring and rare
  fallback 64x64 search/zero probes with existing arm64 NEON coverage;
  `BenchmarkSAD64x64` rows on the local M4 Max sit around `84-87 ns/op`, zero
  allocations.
- Subpel-refine prediction scoring now has 16x16 and 32x32 dual-stride SAD
  surfaces, with arm64 NEON kernels for the source-vs-prediction-scratch case.
  The refine scorer dispatches 8x8/16x16/32x32 probes directly to the matching
  dual-SAD kernel, replacing four or sixteen 8x8 dual-SAD calls per 16x16/32x32
  subpel probe with one larger kernel call. It also hoists the source base and
  prediction scratch slice once per refine call, so each exact probe avoids
  recomputing the same geometry. Compile gates cover native arm64, linux/arm64,
  darwin/amd64, and pure-Go arm64; runtime benchmark rows are still pending
  because newly built local binaries currently hang before Go code starts.
- Inter predictor-swap scoring now uses the same square/rectangular dual-SAD
  helper for emitted block sizes instead of summing 8x8 dual-SAD tiles after
  materializing the candidate prediction. Larger square blocks hit one 16x16 or
  32x32 dual kernel, and rectangular halves compose the matching 8x8/16x16
  kernels.
- The compound LAST/GOLDEN precheck now uses an arm64 NEON 8x8
  rounded-average SAD kernel (`urhadd` + widening absolute difference) instead
  of the scalar per-pixel average loop. `BenchmarkSAD8x8CompoundAvgBlock` is
  `6.36-6.41 ns/op` versus `51.70-52.58 ns/op` for the scalar reference, with
  zero allocations; randomized tests compare the dispatch kernel against the
  pure-Go reference across independent strides and offsets.
- The 8x8, 16x16, 32x32, and 64x64 full-pel raster searches now group four
  horizontal step-4 candidates into one `sad8x8x4Step4`, `sad16x16x4Step4`,
  `sad32x32x4Step4`, or composed `sad64x64x4Step4` call. The arm64 NEON
  kernels follow SVT's multi-candidate SAD shape with shared source loads,
  widened reference windows, and four independent accumulators; compiler
  reports show the direct dispatch wrappers inline at the 8x8/16x16/32x32
  raster call sites and inside the 64x64 composition helper, while the
  stack-local NEON contexts remain non-escaping. `BenchmarkFullPelDiamondSearch8`
  currently sits around `89.9-98.7 ns/op` with zero allocations.
- A generic four-reference 8x8 SAD surface now mirrors SVT's `sad8x8x4d`
  primitive for non-raster candidate groups. The arm64 NEON kernel shares each
  source row load across four independent reference pointers and is covered by
  `TestSAD8x8x4ImplMatchesPureGo`; a 9-run `BenchmarkSAD8x8x4` repeat had a
  `12.56 ns/op` median versus `89.79 ns/op` for the scalar reference, zero
  allocations. Wiring it into the final 8x8 diamond refinement was tested but
  not kept: same-session `BenchmarkFullPelDiamondSearch8` moved from a clean
  HEAD median around `95 ns/op` to a patched median around `102 ns/op`, so the
  active search path stays on the four direct `sad8x8` probes until a grouped
  shape improves the full row.
- A matching generic four-reference 16x16 SAD surface now mirrors SVT's
  `sad16x16x4d`. The direct kernel is covered by
  `TestSAD16x16x4ImplMatchesPureGo`; a 7-run
  `BenchmarkSAD16x16x4` repeat had a `16.44 ns/op` median versus
  `310.2 ns/op` for scalar and `23.58 ns/op` for four separate active
  `sad16x16` calls, all zero allocations. Wiring it into the final 16x16
  diamond refinement was tested but not kept: same-session
  `BenchmarkFullPelDiamondSearch16` moved from a baseline median around
  `159.6 ns/op` to a patched median around `163.2 ns/op`.
- A generic four-reference 32x32 SAD surface now mirrors SVT's
  `sad32x32x4d`. The direct kernel is covered by
  `TestSAD32x32x4ImplMatchesPureGo`; a 7-run
  `BenchmarkSAD32x32x4` repeat after production wiring had a `56.16 ns/op`
  median versus `87.50 ns/op` for four separate active `sad32x32` calls and
  about `1245 ns/op` for scalar, all zero allocations. Unlike the 8x8/16x16
  generic x4 kernels, this one pays off in the full 32x32 search row:
  `BenchmarkFullPelDiamondSearch32` moved from a baseline median around
  `534.3 ns/op` to a patched median around `514.7 ns/op`.
- A composed generic four-reference 64x64 SAD surface now mirrors SVT's
  `sad64x64x4d` shape by calling the active `sad32x32x4` kernel on the four
  quadrants. SVT v4.0.1 has a direct AVX2 `sad64x64x4d` body, while its
  generated baseline ARM table keeps this surface on C, so goav1 does not need
  a separate arm64 asm body to be fair on the local NEON row. The helper is
  covered by `TestSAD64x64x4MatchesReference`; 7-run
  `BenchmarkSAD64x64x4` repeats landed around `204-235 ns/op` versus
  `334-417 ns/op` for four separate active `sad64x64` calls and about
  `4535 ns/op` for scalar, all zero allocations. This paid off in the full
  64x64 search row: `BenchmarkFullPelDiamondSearch64` moved from a baseline
  median around `1926 ns/op` to a patched median around `1779 ns/op`.
- A 2026-06-14 arm64 SAD wrapper cleanup that replaced `&slice[0]` pointer
  extraction with `unsafe.SliceData` was rejected. It removed 21 reported BCE
  sites in `sad_neon_arm64.go` and kept the same escape count, but paired local
  rows did not show a reliable runtime win: full-pel 8x8 search moved only
  slightly (`92.70 ns/op` baseline median versus `91.99 ns/op` patched), while
  `BenchmarkSAD8x8CompoundAvgBlock` regressed (`6.33 ns/op` versus
  `6.59 ns/op`). Keep the indexed form until a narrower wrapper change improves
  an end-to-end search row without regressing the direct SAD kernels.
- `BenchmarkSubpelRefine8x8` now gates the exact 8x8 subpel refine scorer. The
  initial rows are `151.5-159.8 ns/op`, zero allocations. A direct arm64 NEON
  helper experiment for the subpel probe was rejected: the same-checkout
  baseline probe was `38.0-39.3 ns/op` for 8x8, while the direct-helper variant
  measured `40.1-45.0 ns/op`.
- Residual extraction now has focused 8x8/16x16/32x32 scalar-vs-NEON benchmark
  rows for the RD metric surface. On the local M4 Max, `BenchmarkResidual8x8NEON`
  is about `6.65-8.71 ns/op` versus `39.3-40.2 ns/op` scalar,
  `BenchmarkResidual16x16NEON` is about `19.1-20.2 ns/op` versus
  `158-162 ns/op` scalar, and `BenchmarkResidual32x32NEON` is about
  `44.6-47.3 ns/op` versus `575-761 ns/op` scalar, all zero allocations. Wiring
  the lossy keyframe TXB residual loop through this dispatcher was tested but
  rejected: a clean detached baseline `BenchmarkEncodeKeyframe1080p` 7-run
  median was about `25.95 ms/op`, while the patched row was about
  `29.82 ms/op`. A direct scalar guard for sub-8-wide inter residuals was also
  rejected: `BenchmarkChooseInter8x8TXType` baseline rows were about
  `10.15-10.32 us/op`, while the guard measured `10.25-10.61 us/op`.
- RD transform-stat accumulation now has benchmark coverage through the full
  active 8x8/16x16/32x32 coefficient counts. `BenchmarkRDStats64NEON` is about
  `13.5-14.2 ns/op` versus `42.2-50.0 ns/op` scalar,
  `BenchmarkRDStats256NEON` is about `49.1-50.3 ns/op` versus
  `180-186 ns/op` scalar, and `BenchmarkRDStats1024NEON` is about
  `193.8-200.5 ns/op` versus `790-891 ns/op` scalar, all zero allocations.
- Coefficient block-error now has an arm64 NEON reducer mirroring SVT's
  `svt_av1_block_error_neon`: eight `TranLow` int32 lanes are narrowed to
  int16 like SVT's `load_tran_low_to_s16q`, then the kernel accumulates
  `sum((coeff-dqcoeff)^2)` and `sum(coeff^2)` in int64. The parity test
  includes values outside int16 range to pin the narrowing behavior. On the
  local M4 Max, final `BenchmarkBlockError` medians were 1024 coeffs
  `182.4 ns/op` versus `521.6 ns/op` scalar, 256 coeffs `46.63 ns/op` versus
  `136.2 ns/op` scalar, and 64 coeffs `12.59 ns/op` versus `33.21 ns/op`
  scalar, all zero allocations. Escape analysis reports the coeff/dqcoeff
  slices non-escaping, and BCE reports only the intentional entry guards.
- Pixel-domain SSE/variance now has a baseline arm64 NEON stats kernel for the
  practical low-bitdepth metric surface through 64x64:
  4x4/8x4/4x8/8x8/16x4/4x16/16x8/8x16/16x16/32x8/8x32/32x16/16x32/32x32/64x16/16x64/64x32/32x64,
  with 64x64 composed from four 32x32 stats calls. This matches the practical
  width>=4 portion of SVT's `sse_neon.c`/`variance_neon.c` surface without
  changing mode decisions; the width-4 rows use an exact four-byte load kernel
  so edge probes do not overread. On the local M4 Max, the square rows remain
  `BenchmarkPixelStats8x8NEON` about
  `7.12-7.42 ns/op` versus `29.69-31.06 ns/op` scalar,
  `BenchmarkPixelStats16x16NEON` about `25.64-26.06 ns/op` versus
  `126.2-127.3 ns/op` scalar, `BenchmarkPixelStats32x32NEON` about
  `97.73-102.5 ns/op` versus `482.0-497.2 ns/op` scalar, and composed
  `BenchmarkSSEVariance64x64NEON` about `400.2-417.8 ns/op` versus
  `1905-1942 ns/op` scalar, all zero allocations. Earlier rectangular median
  rows are `BenchmarkPixelStats16x8NEON` `12.41 ns/op` versus `64.12 ns/op`
  scalar, `BenchmarkPixelStats8x16NEON` `12.56 ns/op` versus `64.64 ns/op`
  scalar, `BenchmarkPixelStats32x16NEON` `45.10 ns/op` versus `237.4 ns/op`
  scalar, and `BenchmarkPixelStats16x32NEON` `46.23 ns/op` versus
  `239.0 ns/op` scalar; their matching variance rows are
  `BenchmarkSSEVariance16x8NEON` `12.79 ns/op` versus `58.57 ns/op` scalar,
  `BenchmarkSSEVariance8x16NEON` `12.64 ns/op` versus `65.08 ns/op` scalar,
  `BenchmarkSSEVariance32x16NEON` `45.55 ns/op` versus `231.6 ns/op` scalar,
  and `BenchmarkSSEVariance16x32NEON` `46.60 ns/op` versus `232.6 ns/op`
  scalar, all zero allocations. The wider rectangular median rows are
  `BenchmarkPixelStats32x8NEON` `23.12 ns/op` versus `120.8 ns/op` scalar,
  `BenchmarkPixelStats8x32NEON` `24.12 ns/op` versus `123.7 ns/op` scalar,
  `BenchmarkPixelStats64x16NEON` `88.61 ns/op` versus `476.2 ns/op` scalar,
  `BenchmarkPixelStats16x64NEON` `91.49 ns/op` versus `469.7 ns/op` scalar,
  `BenchmarkPixelStats64x32NEON` `177.3 ns/op` versus `947.2 ns/op` scalar,
  and `BenchmarkPixelStats32x64NEON` `179.2 ns/op` versus `931.7 ns/op`
  scalar. Their matching variance medians are `BenchmarkSSEVariance32x8NEON`
  `23.93 ns/op` versus `118.6 ns/op` scalar,
  `BenchmarkSSEVariance8x32NEON` `24.51 ns/op` versus `121.1 ns/op` scalar,
  `BenchmarkSSEVariance64x16NEON` `89.22 ns/op` versus `464.8 ns/op` scalar,
  `BenchmarkSSEVariance16x64NEON` `91.66 ns/op` versus `460.0 ns/op` scalar,
  `BenchmarkSSEVariance64x32NEON` `178.4 ns/op` versus `923.2 ns/op` scalar,
  and `BenchmarkSSEVariance32x64NEON` `179.8 ns/op` versus `910.0 ns/op`
  scalar, all zero allocations. The small-shape median rows are
  `BenchmarkPixelStats4x4NEON` `5.875 ns/op` versus `9.735 ns/op` scalar,
  `BenchmarkPixelStats8x4NEON` `5.734 ns/op` versus `14.90 ns/op` scalar,
  `BenchmarkPixelStats4x8NEON` `6.958 ns/op` versus `16.20 ns/op` scalar,
  `BenchmarkPixelStats16x4NEON` `7.301 ns/op` versus `29.27 ns/op` scalar,
  and `BenchmarkPixelStats4x16NEON` `12.68 ns/op` versus `30.65 ns/op`
  scalar. Their matching variance medians are `BenchmarkSSEVariance4x4NEON`
  `6.441 ns/op` versus `8.890 ns/op` scalar,
  `BenchmarkSSEVariance8x4NEON` `6.247 ns/op` versus `22.40 ns/op` scalar,
  `BenchmarkSSEVariance4x8NEON` `7.325 ns/op` versus `17.44 ns/op` scalar,
  `BenchmarkSSEVariance16x4NEON` `7.568 ns/op` versus `29.16 ns/op` scalar,
  and `BenchmarkSSEVariance4x16NEON` `12.78 ns/op` versus `30.98 ns/op`
  scalar, all zero allocations.
- Pixel-domain SSE/variance now has a runtime-gated DOTPROD tier for the
  width-multiple-of-8 rows where the local M4 Max comparison beats baseline
  NEON. The DOTPROD asm accumulates `src*src`, `ref*ref`, and `src*ref` with
  `UDOT`, keeps the signed residual sum with the baseline widening diff, and
  reconstructs SSE as `src2 + ref2 - 2*cross`. The measured-winning dispatched
  rows are 16x4, 16x8, 8x16, 16x16, 32x8, 8x32, 32x16, 16x32, 32x32, 64x16,
  16x64, 64x32, and 32x64; 64x64 benefits through its four 32x32 stats calls.
  `BenchmarkSSEVarianceDotProd` medians from the final comparison run are:
  16x4 `8.459 -> 8.251 ns/op`, 16x8 `13.30 -> 13.11 ns/op`, 8x16
  `12.67 -> 12.38 ns/op`, 16x16 `23.44 -> 22.09 ns/op`, 32x8
  `23.63 -> 16.09 ns/op`, 8x32 `24.23 -> 23.82 ns/op`, 32x16
  `44.93 -> 30.87 ns/op`, 16x32 `45.51 -> 41.74 ns/op`, 32x32
  `88.74 -> 61.11 ns/op`, 64x16 `87.01 -> 60.70 ns/op`, 16x64
  `90.49 -> 78.21 ns/op`, 64x32 `173.5 -> 120.4 ns/op`, and 32x64
  `175.0 -> 121.5 ns/op`, all zero allocations. Small 8x4 and 8x8 stay on
  baseline NEON because their variance medians were `7.139 -> 7.377 ns/op`
  and `8.117 -> 8.388 ns/op` under DOTPROD.
- Coefficient SATD reduction now has an arm64 NEON reducer mirroring SVT's
  `svt_aom_satd_neon` loop over 16 int32 coefficients per iteration. It covers
  the AV1 coefficient counts `{16,64,256,1024}` and is parity-tested against
  the source-shaped scalar reducer. On the local M4 Max,
  `BenchmarkSATDCoeffs16NEON` is about `2.58-2.68 ns/op` versus scalar
  `5.98-6.37 ns/op` excluding scalar noise spikes, `BenchmarkSATDCoeffs64NEON`
  is about `8.72-9.16 ns/op` versus `18.55-19.34 ns/op` scalar,
  `BenchmarkSATDCoeffs256NEON` is about `32.78-35.43 ns/op` versus
  `80.57-82.83 ns/op` scalar, and `BenchmarkSATDCoeffs1024NEON` is about
  `131.2-135.9 ns/op` versus `303.3-308.6 ns/op` scalar, all zero allocations.
  This does not claim full Hadamard-path parity by itself.
- Low-bitdepth 4x4 Hadamard production now has an arm64 NEON kernel mirroring
  SVT's `svt_aom_hadamard_4x4_neon` two-pass signed-halving butterfly,
  transpose, and sign-extension store. The test has an exact source-shaped
  NEON-order reference and also checks SATD equality against the portable C
  order. On the local M4 Max, `BenchmarkHadamard4x4NEON` is about
  `3.00-3.17 ns/op` versus `20.3-25.1 ns/op` scalar, with zero allocations.
- Low-bitdepth 8x8 Hadamard production now has an arm64 NEON kernel mirroring
  SVT's `svt_aom_hadamard_8x8_neon` two-pass butterfly and transpose. SVT's
  NEON coefficient order is the transpose of its portable C order; this is
  source-allowed because the consumer is SATD, so the test accepts exactly C
  order or exactly that transpose. On the local M4 Max,
  `BenchmarkHadamard8x8NEON` is about `7.10-7.25 ns/op` versus
  `62.7-63.9 ns/op` scalar, with zero allocations.
- Low-bitdepth 16x16 Hadamard production now mirrors SVT's shape: four 8x8
  NEON-order producers followed by the signed halving add/sub quadrant combine
  from `svt_aom_hadamard_16x16_neon`. The test has an exact source-shaped
  NEON-order reference and also checks SATD equality against the portable C
  order. On the local M4 Max, `BenchmarkHadamard16x16NEON` is about
  `47.7-48.7 ns/op` versus `307-312 ns/op` scalar, with zero allocations.
- Low-bitdepth 32x32 Hadamard production now mirrors SVT's shape: four 16x16
  NEON-order producers followed by the signed add/sub and arithmetic shift-by-2
  quadrant combine from `svt_aom_hadamard_32x32_neon`. The test has an exact
  source-shaped NEON-order reference and also checks SATD equality against the
  portable C order. On the local M4 Max, `BenchmarkHadamard32x32NEON` is about
  `246.8-248.9 ns/op` versus `1460-1488 ns/op` scalar, with zero allocations.
- `BenchmarkBitCounterCDF4Stream` now gates the count-only range-coder CDF4
  path that appears under TXB coefficient pricing. On a clean `70027abc`
  baseline worktree, the specialized `BitCounter.WriteCDF4` path measured
  `19.06-19.19 us` per 4096-symbol stream; final repeat rows after the 16-bit
  normalization-shift change measure `17.97-18.01 us`, zero allocations. The
  generic counter lane remains neutral/noisy at about `19.3-20.7 us`. The same
  change moves
  `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` from `447.3-451.4 ns/op`
  to final repeat rows of `427.5-435.5 ns/op`, zero allocations. A branchless
  saturated-count update and a trace-preserving `WriteBit` count wrapper were
  both rejected: neither improved the trusted-count path after focused
  measurement.
- Current P-frame diagnostics: the profiled
  `BenchmarkVideoEncoderPFrame1080p-4` sample is `79.4 ms/op`, and non-profiled
  repeat rows are around `78.0-78.8 ms/op` with zero steady-state allocations.
  Allocation is not the dominant CPU gap; the CPU profile is.
- Latest P-frame profile on 2026-06-13 after the trusted 8x8 hybrid-dispatch
  safe point measured `71.695 ms/op`, `4666 B/op`, and `1 alloc/op` with
  `GOMAXPROCS=4`, `-benchtime=40x`, and a CPU profile. The top local costs are
  still 8x8 TXB coefficient pricing (`CountCoefficientsTXB8x8Y2DTrustedArray`,
  `1.51 s` cumulative of `8.12 s`) and loop-filter postfilter planning
  (`loopFilterPostFilterPlanTrustedSweep`, `1.32 s` cumulative). SVT maps the
  TXB prep side to `svt_av1_txb_init_levels_neon`,
  `svt_av1_get_nz_map_contexts_neon`, and
  `svt_av1_compute_cul_level_neon`; its loop-filter edge decision maps to
  `deblocking_filter.c:set_lpf_parameters`, followed by NEON filter kernels.
- A zero-symbol split for `BitCounter.WriteCDF4` was rejected on 2026-06-13.
  It preserved CDF/Tell parity, but
  `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` moved from mean/median
  `414.75/410.65 ns/op` to `425.56/424.65 ns/op`, and
  `BenchmarkCountCoefficientsTXB8x8UV2D/trusted-count` moved from
  `438.61/437.60 ns/op` to `466.64/457.80 ns/op`, zero allocations. The
  caller-level `level == 0` branch did not beat the existing four-way CDF4
  switch.
- Direct luma transform-tree replay inside the loop-filter planner was also
  rejected on 2026-06-13. Removing the `ForEachLumaTXB` callback but passing the
  large context through recursion moved `BenchmarkLoopFilterPostFilterPlanTrusted`
  from mean/median `13.330/13.317 us` to `13.707/13.496 us`; a replay-object
  rewrite was worse at `17.612/17.597 us` and introduced `6432 B/op` with
  `3 allocs/op`. The existing callback shape remains the best measured Go
  form for that planner path.
- The luma loop-filter segment splitter now follows SVT's
  `deblocking_filter.c:set_lpf_parameters` non-zero-current-level shape more
  closely: when the current side already has a non-zero level, the per-MI-cell
  loop resolves only previous-side transform width and skips previous-level
  fallback checks. It also hoists the repeated frame color and map dimension
  conversions out of that loop. Same-session
  `BenchmarkLoopFilterPostFilterPlanTrusted` baseline at `50000x` had median
  `13778 ns/op`; the split path measured median `13624 ns/op` at `50000x` and
  `13504 ns/op` at `200000x`, zero allocations. Compiler reports keep the
  previous-record lookup helpers inlineable at cost `78`, with `record`,
  `plan`, and `edges` non-escaping; the follow-up value-cache entry below clears
  the cached-record map-slice escape report.
- The loop-filter planner previous-record caches now store compact
  `FrameWorkLoopFilterBlockRecord` values instead of retaining pointers into
  `FrameWorkLoopFilterMap.Records`, matching the stack-owned hot-path shape used
  elsewhere in the planner. This clears the cached-record escape source for both
  luma and chroma: `-gcflags='all=-m=2 -d=ssa/check_bce/debug=1'` reports the
  luma cache lookup, chroma cache lookup, and trusted planner `filterMap` inputs
  as non-escaping, while the previous-record lookup helpers stay inlineable at
  cost `78` and the trusted planner wrapper remains inlineable at cost `78`.
  Same-session `BenchmarkLoopFilterPostFilterPlanTrusted` baseline at `100000x`
  had median `13936 ns/op`; the value-cache version measured median
  `13680 ns/op` over nine repeats, with `0 B/op` and `0 allocs/op`. Hot-size
  guards pin `FrameWorkLoopFilterBlockRecord` at `34 B`. The previous-record
  cache field order now keeps the `uint16` coordinates before the byte-sized
  validity/level fields, shrinking the luma previous cache from `76 B` to
  `74 B` and the chroma previous cache from `50 B` to `48 B` without changing
  the cached values. Focused `BenchmarkLoopFilterPostFilterPlanTrusted` rows
  stayed zero-alloc but did not improve in the same session (`13559 ns/op`
  baseline median versus `13838 ns/op` packed median), so this is kept as
  stack/cache-footprint hygiene rather than a planner speed claim.
- A segmented-luma first-cell seed was tested but not kept. The idea was to
  initialize the active segment from offset zero before scanning the remaining
  MI cells, avoiding the baseline's first empty `frameWorkStoreLoopFilterLumaEdgeSegment`
  call when the first resolved level/width is non-zero. It preserved tests and
  zero allocations, but same-session
  `BenchmarkLoopFilterPostFilterPlanTrusted` moved from median `13627 ns/op`
  to `13708 ns/op` at `100000x`, and compiler cost for
  `frameWorkAppendLoopFilterLumaEdgeSegmentsWithWidth` rose from `792` to
  `989` while the non-escape shape stayed unchanged. The current range-loop
  form remains better until a broader SVT `set_lpf_parameters`-shaped rewrite
  removes more than this one empty helper call.
- The 4x4 final luma coefficient writer now has a direct inter-tx-type entry
  point, matching SVT's `av1_write_coeffs_txb_1d` order: write `txb_skip`,
  return for all-zero blocks, then write luma `tx_type` before the EOB token.
  The p-frame caller guards the direct path with a 16-coefficient OR scan so
  zero blocks do not resolve a transform CDF that SVT/AV1 would not code.
  `BenchmarkWriteCoefficientsTXB4x4Y2DContextTrusted` measured steady rows of
  `139.8-145.1 ns/op` for the hook path and `138.6-140.8 ns/op` for the
  direct-tx path, both zero allocations. Full-frame throughput is neutral in
  same-machine repeats: clean `origin/main` measured median `74.35 ms/op`, and
  the guarded direct path measured median `74.53 ms/op`, zero steady
  allocations. This is kept as source-shape and hook-removal alignment, not as a
  claimed frame-level speedup.
- Two Go-only TXB prep rewrites for
  `CountCoefficientsTXB8x8Y2DTrustedArray` were rejected on 2026-06-14. The
  SVT equivalent keeps prep split across `svt_av1_txb_init_levels_neon`,
  `svt_av1_get_nz_map_contexts_neon`, and `svt_av1_compute_cul_level_neon`;
  goav1 already folds the nonzero-map contexts into `coeffScanHot8x8Y2D`, so
  the tested Go opportunities were memory-touch reductions. Skipping zero scan
  entries before padded-level stores preserved parity but moved
  `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` from median
  `424.5 ns/op` to `425.1 ns/op`. A one-forward-pass prep that combined EOB,
  padded levels, sign bits, and cumulative level was worse at median
  `436.7 ns/op`. Both stayed at zero allocations, and neither was kept.
- Trusted 8x8 count-only BCE proof hints were rejected on 2026-06-14. Adding a
  one-time `scanHot[eob-1]` / `absLevels[eob-1]` proof after the nonzero-EOB
  branch removed two reported bounds checks in the luma/UV count-function
  regions, but the extra proof did not pay for itself: paired `500000x` rows
  moved `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` from median
  `416.2 ns/op` to `420.1 ns/op`, and
  `BenchmarkCountCoefficientsTXB8x8UV2D/trusted-count` from `436.9 ns/op` to
  `443.1 ns/op`, all zero allocations. The loops stay on the current shape.
- The 8x8 UV count-only sign/Golomb pass now mirrors the luma sparse-bitset
  walk in SVT's `av1_write_coeffs_txb_1d` tail: while filling levels it records
  non-zero scan slots in a `uint64`, then emits signs/tails by set-bit order
  instead of scanning every entry up to EOB and branching on zero. Same-session
  `BenchmarkCountCoefficientsTXB8x8UV2D/trusted-count` moved from median
  `465.4 ns/op` to `452.7 ns/op` over seven `500000x` rows, with zero
  allocations; a `1000000x` patched repeat had median `448.9 ns/op`.
  Compiler reports keep the wrapper inlineable, `entropy.NewBitCounter`
  inlined, `cdfs`/coefficient pointers non-escaping, and the neighbor-context
  bounds checks elided.
- Applying that sparse-bitset sign/Golomb tail to the real 8x8 luma writer was
  rejected on 2026-06-14. It preserved 8x8 writer/count parity and zero
  allocations, but the luma-only patch regressed the guarded rows in a fresh
  paired run: `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted` moved from median
  `487.1 ns/op` to `491.6 ns/op`, and the untouched
  `trusted-count` control moved from `431.0 ns/op` to `434.3 ns/op`. The real
  writer keeps its scan-and-branch tail until an end-to-end row proves the extra
  bitset bookkeeping is worthwhile.
- A narrower branchless sign-bit extraction in the 8x8 luma prep loops was also
  rejected on 2026-06-14. Replacing the nested `cv < 0` branch with
  `uint16(cv)>>15` preserved writer/count parity and kept all hot inputs
  non-escaping, but compiler cost/BCE shape was unchanged and paired
  `1000000x`, `-count=9` rows moved
  `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted` from median `482.1 ns/op` to
  `492.3 ns/op`, while `trusted-count` was effectively neutral/noisy
  (`425.2 -> 426.2 ns/op`). The untouched UV count control measured
  `453.6 -> 452.5 ns/op`. The current explicit sign branch remains the better
  measured 8x8 luma prep shape.
- A SVT-shaped cumulative-level saturation guard was rejected on 2026-06-14 for
  the 8x8 trusted-count TXB prep loop. SVT's separate
  `svt_av1_compute_cul_level_c` can break once the sum reaches
  `COEFF_CONTEXT_MASK`, but goav1's combined prep loop must still fill level,
  sign, and tail state for coefficient coding. Guarding `culLevel += level` in
  both 8x8 luma and UV count paths preserved parity and zero allocations, but
  same-benchtime `1000000x` rows did not improve: luma median was effectively
  neutral (`415.6 ns/op` baseline versus `415.1 ns/op` patched), while UV
  regressed (`434.0 ns/op` baseline versus `441.3 ns/op` patched). The code was
  reverted; the useful source-shape lesson is that only a real split prep/kernel
  can benefit from the SVT early-exit form.
- The next tx-type-search experiment also stayed Go-only and was rejected on
  2026-06-14. SVT maps the comparable 8x8 forward-transform surface to
  `svt_av1_fwd_txfm2d_8x8_neon` plus the N2/N4 variants, with NEON ADST row and
  column helpers (`highbd_fadst8_*_neon`). Baseline
  `BenchmarkChooseInter8x8TXType` measured median `11261 ns/op`; an early-IDTX
  pruning bound preserved `TestChooseInter8x8TXTypeSeededMatchesReference` but
  moved the median to `11303 ns/op`, so it was reverted. A later source-shaped
  arm64 NEON IDTX kernel was kept because it directly covers SVT's identity
  branch for the active trusted 8x8 tx-type trial: same-shape
  `BenchmarkForwardBlock8x8HybridTrusted/IDTX` moved from median `25.83 ns/op`
  to `6.881 ns/op`, while direct `BenchmarkForwardBlock8x8IDTXImpl` measured
  median `7.365 ns/op` versus `25.22 ns/op` for the pure-Go body, all
  `0 allocs/op`. The follow-up arm64 NEON ADST kernels now cover all three
  active 8x8 ADST tx-type trials: ADST_DCT uses SVT's ADST-column/DCT-row
  shape, DCT_ADST uses DCT-column/ADST-row, and ADST_ADST uses
  ADST-column/ADST-row. They are guarded by impl-vs-pure tests with nontrivial
  strides. Same-session `3000000x` direct medians are:
  `BenchmarkForwardBlock8x8ADSTDCTImpl` `22.71 ns/op` versus
  `106.4 ns/op` pure, `BenchmarkForwardBlock8x8DCTADSTImpl` `22.85 ns/op`
  versus `107.4 ns/op` pure, and `BenchmarkForwardBlock8x8ADSTADSTImpl`
  `26.24 ns/op` versus `124.8 ns/op` pure. Trusted medians are
  `23.00 ns/op` for ADST_DCT, `23.02 ns/op` for DCT_ADST, and `26.09 ns/op`
  for ADST_ADST, all `0 allocs/op`.
- `BenchmarkForwardADST8` was added on 2026-06-14 as the tight 1-D gate for
  future 8x8 ADST SIMD work. Compiler reports show `fwdHalfBtf13` and
  `fwdRoundShift1x8` inline into the scalar hybrid path, while `fwdADST8` and
  the 8x8 hybrid block bodies are intentionally too large to inline; the
  trusted buffers do not escape. The rotating-input benchmark measures dense
  `fwdADST8` at `6.88-7.34 ns/op` and zero input at `6.87-7.08 ns/op`, zero
  allocations. A zero-vector shortcut was rejected: it improved the isolated
  zero-input row to `1.78-1.89 ns/op`, but dense rows moved to
  `7.15-7.41 ns/op` and full `ADST_ADST` 8x8 rows regressed to roughly
  `285.0-292.6 ns/op`.
- An 8x8 scalar column-rounding unroll was rejected on 2026-06-14. SVT's
  `svt_av1_fwd_txfm2d_8x8_sse4_1` keeps rounding inside the vectorized
  8x8 column pipeline before transpose; in Go, unrolling `fwdRoundShift1x8`
  globally helped `DCT_ADST` but regressed `ADST_ADST`. Narrowing the unroll to
  only `DCT_ADST` still was not safe: same-session `2000000x` baseline medians
  were `ADST_DCT=275.8 ns/op`, `DCT_ADST=305.0 ns/op`, and
  `ADST_ADST=278.7 ns/op`; the narrowed patch measured
  `ADST_DCT=276.4 ns/op`, `DCT_ADST=286.1 ns/op`, and
  `ADST_ADST=288.3 ns/op`, all zero allocations. The code was reverted; this
  reinforces that the remaining gap wants an actual 8x8 ADST SIMD kernel rather
  than scalar code-shape churn.
- The 8x8 hybrid ADST passes now use a value-returning `fwdADST8Values` core
  so the ADST row/column paths no longer fill an input array and drain an
  output array around every 1-D pass. The pointer-shaped `fwdADST8` remains the
  tight 1-D benchmark target. Compiler diagnostics keep the trusted wrappers
  inlineable, report coeff/residual/scratch as non-escaping, and inline
  `fwdRoundShift1Value`; the value core itself stays too large to inline. Paired
  `2000000x` rows moved `BenchmarkForwardBlock8x8HybridTrusted` medians from
  `ADST_DCT=272.0 ns/op`, `DCT_ADST=296.4 ns/op`, and
  `ADST_ADST=267.4 ns/op` to `201.2`, `219.4`, and `128.7 ns/op`, with zero
  allocations. The tx-type selector row moved from median `11049 ns/op` to
  `10730 ns/op`, and a small `GOMAXPROCS=4`, `-benchtime=8x`, `-count=5`
  P-frame row moved from median `79.21 ms/op` to `77.19 ms/op`, also at
  `0 allocs/op`. This is a scalar temp-array cleanup; SVT's batched ADST NEON
  surface remains an open assembly gap.
- The mixed 8x8 DCT halves now use a value-returning `fwdDCT8Values` core in
  the same style, scoped to `ADST_DCT` row DCT and `DCT_ADST` column DCT so the
  standalone DCT dispatch and tight `BenchmarkForwardDCT8x8` target stay
  comparable. A direct helper parity test checks `fwdDCT8Values` against the
  pointer-shaped `fwdDCT8`. Compiler diagnostics keep `fwdRoundShift1Value`
  inlined and report the touched coeff/residual/scratch buffers as
  non-escaping; the DCT value core is intentionally too large to inline, like
  the ADST value core. Paired `2000000x` transform rows moved medians from
  `ADST_DCT=204.3 ns/op` and `DCT_ADST=219.3 ns/op` to `112.6` and
  `113.3 ns/op`, with `ADST_ADST=131.6 -> 132.2 ns/op`,
  `IDTX=26.50 -> 26.07 ns/op`, and `BenchmarkForwardDCT8x8=21.19 -> 21.01 ns/op`;
  all rows stayed at `0 allocs/op`. The tx-type selector row moved from median
  `10903 ns/op` to `10607 ns/op`. A same-shape small `GOMAXPROCS=4`,
  `-benchtime=8x`, `-count=5` P-frame row was neutral/noisy
  (`78.28 ms/op -> 78.69 ms/op`), so no end-to-end win is claimed from that row.
  This closes the scalar temp-array gap in the mixed 8x8 hybrid DCT halves; the
  remaining SVT parity gap is still the batched assembly transform surface.

## Next Implementation Order

1. Run every SVT comparison in two rows: best SVT (`-svt-asm max` or omitted)
   and baseline NEON (`-svt-asm neon`). Report both `--lp` and `--asm`.
2. Prototype a tiny TXB prep kernel only if it replaces work already proven hot:
   eob/level-buffer/stat extraction for the 8x8 luma path. Keep it only if both
   `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` and the fair row improve.
3. Wire the new SSE/variance/SATD/Hadamard metric kernels into encoder search
   only with direct profile mapping, such as additional batched candidates or
   RD metrics that remain visible after the existing SAD and residual-stat NEON
   work.
4. Add forward ADST8/tx-type trial SIMD before broad transform-surface work.
5. Use the arm64 DOTPROD/I8MM feature metadata already detected by goav1 to
   decide whether convolve, SAD, or CDEF variants make sense relative to SVT
   `max` rows.

The rule for adding assembly is the same everywhere: pure-Go/reference parity
test first, zero allocations, focused benchmark, then fair `qualitybench` row.
