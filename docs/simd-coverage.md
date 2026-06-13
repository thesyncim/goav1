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
families. Therefore max-tier rows are not equivalent to goav1's current
baseline-NEON dispatch. Until DOTPROD/I8MM variants are wired and measured,
report max-tier SVT rows as best-SVT rows and baseline `-svt-asm neon` rows as
the closest assembly-tier control.

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
| TXB coefficient prep and contexts | `encodetxb_neon.c`: `svt_av1_txb_init_levels_neon`, `svt_av1_get_nz_map_contexts_neon`; `av1_quantize_neon.c`: `svt_av1_compute_cul_level_neon` | No assembly. Hot Go writer has trusted 4x4/8x8/16x16/32x32 count-only trial paths, stack level buffers, fixed CDF storage, and recorded sign bits. | High priority because profile points at coefficient/range coding. Prototype only narrow, measured kernels; previous nonzero-list, extra scan-table, and branchless sign rewrites regressed or tied after paired measurement. |
| Range coder and CDF update | SVT does not make this a comparable named SIMD surface; arithmetic coding is serial | `WriteBinaryCDFTrusted`, `WriteCDF4`, `normalize`, and `WriteBit` are top profile entries. `BenchmarkBitCounterCDF4Stream` now gates the exact count-only CDF4 path. | Keep source-shaped Go unless a benchmark proves assembly beats call/setup cost. This is a hot scalar issue, not an SVT SIMD parity item. |
| SAD/search metrics | Broad SAD loops, PME SAD, external all/eight SAD, highbd SAD in `compute_sad_neon.c` and `sad_neon.c`; DOTPROD variants exist | `sad8x8`, `sad16x16`, `sad32x32`, `sad8x8Dual`, emitted rect sizes `16x8`, `8x16`, `32x16`, `16x32`, the 8x8 compound-average precheck SAD, and the current 8x8/16x16/32x32/64x64 full-pel raster x4 candidate groups have arm64 NEON coverage through direct or composed kernels | Baseline NEON coverage for current SAD/search probes is now much closer. Add DOTPROD/I8MM only after runtime feature detection and profile proof. |
| Variance, SSE, block error, SATD, Hadamard | `variance_neon.c`, `sse_neon.c`, `block_error_neon.c`, `hadamard_path_neon.c`, plus DOTPROD SSE/variance | goav1 has residual/RD stats NEON, but not SVT's full metric surface | Medium-high. Implement only where the encoder actually uses the metric or where a mode-search change will use it. |
| Forward transforms | `highbd_fwd_txfm_neon.c` covers square, rectangular, N2/N4, and many tx types including ADST paths | Forward DCT 4/8/16/32 has NEON; forward ADST/other tx-type trial work is scalar, though the 8x8 trusted hybrid path now dispatches once per tx type instead of per 1-D row/column | High for the current profile: `transform.fwdADST8` is visible in P-frame TX-type trials. |
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
- `WriteCDF4` uses a single symbol switch for the quaternary coefficient-base
  path and is guarded by `TestWriteCDF4MatchesWriteCDF`; after byte-writer
  normalization moved to the 16-bit shift expression below, the specialized
  local writer stream benchmark is about `18.6-19.0 us` per 4096-symbol
  stream, zero allocations.
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
  fallback 64x64 search/zero probes with existing arm64 NEON coverage; focused
  benchmark rows remain pending while local runtime binaries are blocked.
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
- `BenchmarkSubpelRefine8x8` now gates the exact 8x8 subpel refine scorer. The
  initial rows are `151.5-159.8 ns/op`, zero allocations. A direct arm64 NEON
  helper experiment for the subpel probe was rejected: the same-checkout
  baseline probe was `38.0-39.3 ns/op` for 8x8, while the direct-helper variant
  measured `40.1-45.0 ns/op`.
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

## Next Implementation Order

1. Run every SVT comparison in two rows: best SVT (`-svt-asm max` or omitted)
   and baseline NEON (`-svt-asm neon`). Report both `--lp` and `--asm`.
2. Prototype a tiny TXB prep kernel only if it replaces work already proven hot:
   eob/level-buffer/stat extraction for the 8x8 luma path. Keep it only if both
   `BenchmarkWriteCoefficientsTXB8x8Y2D/trusted-count` and the fair row improve.
3. Add encoder search metric kernels only with direct profile mapping, such as
   additional batched candidates or RD metrics that remain visible after the
   existing SAD/residual-stat NEON work.
4. Add forward ADST8/tx-type trial SIMD before broad transform-surface work.
5. Use the arm64 DOTPROD/I8MM feature metadata already detected by goav1 to
   decide whether convolve, SAD, or CDEF variants make sense relative to SVT
   `max` rows.

The rule for adding assembly is the same everywhere: pure-Go/reference parity
test first, zero allocations, focused benchmark, then fair `qualitybench` row.
