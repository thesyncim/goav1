# SIMD Coverage Gaps — Audit Inventory

## STATUS 2026-07-04: SWEEP COMPLETE — all hot gaps closed, pushed, byte-exact.
- **arm64 NEON gaps A1-A3 DONE**: inverse ADST16 (5.2x kernel), highbd wide loop
  filter filter6/8/14 (~3% whole-decode 10-bit), highbd filter-intra (~8x). All
  merged with measured GOMAXPROCS=1 cpu wins; dryrun-extended 0-FAIL.
- **amd64 AVX2 buckets B1-B10 DONE** (parity-verified via GOARCH=amd64 Rosetta-
  executed differentials; perf pending NATIVE x86): B1 compound blend/convbuf,
  B2 wide loop filter, B3 frame widen, B4 cdef direction, B5 addRawTransform,
  B6 SGR (box+blend), B7 invglue, B8 encoder (pixelStats/satd/hadamard/residual/
  realtimeAvg/scaleNearest), B9 intra (CfL/dir/filter-intra), B10 coeff contexts
  (init-levels + nz-map).
- **Intentionally left pure-Go (documented non-wins, do NOT re-port):**
  coeffCulLevel (scan-order gather, scalar even in NEON ref, never wired to
  production on any arch); B8 tail rdStatsBlock (needs VPSRAQ emulation +
  per-lane bits.Len16, ~0.7% cold), forwardBlock8x8 hybrids (>16 live values,
  cold), blockError (cold). 16-bit AVX2 filter-intra (predictFilterIntra16 on
  amd64) — cold follow-up. See [[perf-deadends]] for CDEF-in-place + madvise-arena.
- Whole-tree `GOARCH=amd64 go vet ./...` clean; `go test ./...` 0-FAIL; arm64
  decode dav1d gap 3.94x after the loop-filter batcher (separate from this sweep).

Audit date: 2026-07-03. Host: darwin/arm64 (Apple M-series, NEON+i8mm+dotprod).
Method: enumerated every func-ptr `*Impl` dispatch slot + build-tag func-swap
kernel in `internal/av1/`, cross-checked arm64 (NEON/i8mm/dotprod) vs amd64
(AVX2/SSE2) bindings, then profiled `GOMAXPROCS=1` decode of representative
clips (8-bit inter p720_inter_q32, 8-bit intra p288_intra_q32, 10-bit inter
p360_inter_q32_10bit) to grade hotness. RESEARCH ONLY — nothing implemented.

Totals: **133 dispatch slots** (minus test/bench helpers ≈ **110 real kernels**).
- **arm64 NEON**: essentially COMPLETE on the 8-bit decode hot path. Genuine
  remaining NEON gaps are all **highbd (10/12-bit)** or **intra ADST/identity**
  scoped (3 families, below). Everything hot on 8-bit inter is already NEON.
- **amd64 AVX2**: large parity tail — **~40 kernels NEON-on-arm64 but pure-Go on
  amd64** (compound blend, wide loopfilter, frame widen, CDEF direction, SGR,
  addRawTransform, the whole encoder metric/rdstats/forward-hybrid set, intra
  directional/CfL helpers, tile coeff contexts, invglue).

---

## How to read this

For each family: `arm64` = is a NEON/i8mm kernel BOUND and REACHED on arm64;
`amd64` = is an AVX2 kernel bound; `hot%` = flat self-time on the clip where it
matters (from this audit's profiles). "unreached" = kernel exists in-tree but the
dispatch init does not bind it.

---

## Coverage matrix (by package)

### motion (inter prediction) — the dominant decode cost
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| convolveX8/Y8 (+scratch) | NEON+i8mm ✓ | AVX2 ✓ | parity |
| convolve2D8 (+scratch) | NEON+i8mm ✓ | AVX2 ✓ | parity |
| convolveX/Y/2D Clamped (+scratch) | NEON+i8mm ✓ | AVX2 ✓ (2D scratch only) | amd64 lacks X/Y `ClampedWithScratch` — minor, amd64 routes non-scratch clamped |
| convolveX/Y/2D HighBD (+Clamped) | NEON ✓ | AVX2 ✓ | parity; 8-bit warp still feeds `warpHorizontal8Resident` (OFF-LIMITS) |
| blendCompoundAvg8 | NEON ✓ | **PURE-GO** | **amd64 GAP (B1)** |
| predictInterCompoundRef8ToConvBuf {2D,X,Y,Copy} | NEON+i8mm+dotprod ✓ | **PURE-GO** | **amd64 GAP (B1)** |
| predictInterCompoundRefHighBDToConvBuf {2D,X,Y,Copy}Resident/Clamped (5) | NEON ✓ | **PURE-GO** | **amd64 GAP (B1)** |

### loopfilter
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| filter4Edge (8-bit) | NEON ✓ | AVX2 ✓ | parity |
| filter4Edge16 (10/12-bit narrow) | NEON ✓ | **PURE-GO** | **amd64 GAP (B2)** |
| filter6/8/14Edge (8-bit wide) | NEON ✓ | **PURE-GO** | **amd64 GAP (B2)**, hot |
| filter6/8/14 **highbd** (16-bit wide) — `filter{6,8,14}Samples` | **PURE-GO** | **PURE-GO** | **arm64 GAP (A2) + amd64 GAP** — no wide-highbd kernel on ANY arch |

### transform (inverse — decode; forward — encode)
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| inverseDCT8/16 Row2 & Col2 | NEON ✓ | AVX2 ✓ | parity |
| inverseDCT32 Row2/Col2 | NEON ✓ | **PURE-GO** | covered on amd64 by Row4/Col4; narrow-tail only |
| inverseDCT32/64 Row4 & Col4 | NEON ✓ | AVX2 ✓ | parity (the hot 32/64 path) |
| inverseDCT4 Row2 | NEON adapter **UNREACHED** | PURE-GO | deliberately unbound (measured slower); cold |
| inverseADST4/8 Row2 | NEON adapter **UNREACHED** | PURE-GO | row asm exists, unbound (measured slower); see A1 |
| inverseADST8/16 (column + ADST16 anywhere) | **PURE-GO** | **PURE-GO** | **arm64 GAP (A1)** — `inverseADST16To` 3.7% flat on intra; NO ADST-16 kernel exists |
| inverse identity (idtx) row/col | **PURE-GO** | **PURE-GO** | cold; skip (C) |
| inverseDCT64 Row2/Col2 (old 2-lane) | **UNBOUND (unsafe)** | — | ~1.7–2KB manual frame under NOSPLIT$0 = fuzz-seed corruption; **DELETE candidate** |
| clampRound / narrowStore (invglue) | NEON ✓ | **PURE-GO** | **amd64 GAP (B7)**, small glue |
| forwardDCT4/8/16/32 (square) | NEON ✓ | AVX2 ✓ | parity |
| forwardBlock8x8 {ADST-DCT, DCT-ADST, ADST-ADST, IDTX} hybrids | NEON ✓ | **PURE-GO** | **amd64 GAP (B8)** — encoder |

### cdef
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| filterBlock (pri/sec) | NEON ✓ | AVX2 ✓ | parity |
| findDirection / findDirectionDual | NEON ✓ | **PURE-GO** | **amd64 GAP (B4)** |

### restoration
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| wienerHorizontal/Trusted/Vertical | NEON ✓ | AVX2 ✓ | parity |
| boxsum / selfguided / selfguidedFast (SGR) | NEON ✓ | **PURE-GO** | **amd64 GAP (B6)** |

### dsp
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| blendA64Mask | NEON ✓ | AVX2 ✓ | parity (earlier "withheld" AVX2 now bound) |
| minMaxAbsDiff8x8 | NEON ✓ | AVX2 ✓ | parity |
| addResidualPlaneBlock | NEON ✓ | AVX2 ✓ | parity |
| addRawTransformPlaneBlock | NEON ✓ | **PURE-GO** | **amd64 GAP (B5)** |

### frame
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| loadSampleRows8 (u8→u16 CDEF-staging widen) | NEON ✓ | **PURE-GO (generic)** | **amd64 GAP (B3)**, hot (CDEF/restoration staging) |

### prediction (intra)
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| predictPaeth / Smooth{,V,H} | NEON ✓ | AVX2 ✓ | parity |
| applyCFL / sumSamples | NEON ✓ | AVX2 ✓ | parity |
| subsampleLuma8, dirRowInterp8, dirAboveRun8, dirLeftCol8 | NEON ✓ | **PURE-GO** | **amd64 GAP (B9)** |
| predictFilterIntra8 (8-bit) | NEON ✓ | **PURE-GO** | **amd64 GAP (B9)** |
| predictFilterIntraBlockDirect16 (highbd) | **PURE-GO** | **PURE-GO** | **arm64 GAP (A3)** — 1.7% on 10-bit intra |

### quantize / superres / filmgrain
| family | arm64 | amd64 | notes |
|---|---|---|---|
| dequantColumn | NEON ✓ | AVX2 ✓ | parity |
| quantizeFPBlock / quantizeBBlock | NEON ✓ | AVX2 ✓ | parity (encoder) |
| upscaleRow (superres) | NEON ✓ | AVX2 ✓ | parity |
| applySegment (film grain) | NEON ✓ | AVX2 ✓ | parity |

### encoder (SAD/metric/rdstats/pframe — the SVT-parity path)
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| sad8/16/32 {single,x4,x4Step4,Dual,CompoundAvg} | NEON+dotprod ✓ | AVX2 (+SSE2) ✓ | parity |
| pixelStats* (all 20 block shapes) | NEON+dotprod ✓ | **PURE-GO** | **amd64 GAP (B8)** |
| hadamard4/8/16/32, satdCoeffs | NEON ✓ | **PURE-GO** | **amd64 GAP (B8)** |
| residualBlock, rdStatsBlock | NEON ✓ | **PURE-GO** | **amd64 GAP (B8)** |
| realtimeAvg8x8 / Quad | NEON ✓ | **PURE-GO** | **amd64 GAP (B8)** |
| scalePlaneNearest / scalePlaneNearest16 | NEON ✓ | **PURE-GO** | **amd64 GAP (B8)** |

### tile (coeff context — entropy-adjacent)
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| coeffCulLevel, coeffInitLevels, coeffNzMapContexts | NEON ✓ | **PURE-GO (generic)** | **amd64 GAP** — moderate; entropy-adjacent |

### entropy
| kernel | arm64 | amd64 | notes |
|---|---|---|---|
| reader bit / readCDF4HighToken | NEON (partial) | PURE-GO | symbol-decode NEON is a **PINNED DEAD-END** (see below) — do not port |

---

## Ranked gap list

Ranking = (hotness × certainty) / effort.

### GROUP A — arm64 NEON missing / unreached (highest value: helps THIS host now)

| # | kernel | arch missing | hot% | upstream ref | approach | effort |
|---|---|---|---|---|---|---|
| **A1** | **Inverse ADST (4/8/16) row + column** | arm64 (+amd64) | **3.7% flat** intra (`inverseADST16To`); ~1% 10-bit | dav1d `src/arm/64/itx16.S` `inv_adst{4,8,16}` | ADST butterfly in int32 lanes; reuse the `tools/itxgen/` transliterate→clang→WORD-transcribe pipeline that produced the DCT32/64 kernels. Row asm for ADST4/8 already exists (unbound); ADST16 + all columns are new. Pre-clamp envelope trick from DCT Col4. | med-high |
| **A2** | **Highbd wide loop filter — filter6/8/14 (16-bit)** | arm64 (+amd64) | **~6% cum** `filter14Samples` + **3.3%** `filter4Samples` on 10-bit | dav1d `src/arm/64/loopfilter16.S` | Mirror the existing 8-bit `filter_wide_neon_arm64.s` (filter6/8/14) to two-byte samples; `filter4Edge16NEON` already proves the 16-bit narrow shape. | med |
| **A3** | **Highbd filter-intra — predictFilterIntraBlockDirect16** | arm64 (+amd64) | ~1.7% 10-bit intra | dav1d `ipred16` filter-intra; goav1 has 8-bit `predictFilterIntraBlockDirect8NEON` | Mirror the bound 8-bit filter-intra NEON to uint16. | low-med |

Notes on A: the 8-bit inter hot path is FULLY NEON on arm64 (profile top is
runtime madvise/memclr/memmove + already-NEON convolve/wiener/cdef/filter14/
compound-blend). The remaining arm64 wins are all highbd- or intra-scoped.

### GROUP B — amd64 AVX2 missing (HOT on amd64; arm64 already NEON)

These are all NEON-bound on arm64 AND appear hot in the arm64 profile, so their
scalar amd64 path is a hot gap. Port via VEX/avx2gen (int-lane).

| # | kernel(s) | hot proxy (arm64) | upstream ref | effort |
|---|---|---|---|---|
| **B1** | compound blend: blendCompoundAvg8 + predictInterCompoundRef8ToConvBuf{2D,X,Y,Copy} + 5 HighBD variants | 1.2%+ (`blendCompoundAvg8NEON`, `predictInterCompoundRefHighBD…2DClamped`) hot on inter/10-bit | dav1d `mc_avx2` `w_avg/mask/bidir` | med-high |
| **B2** | wide loop filter: filter6/8/14Edge (8-bit) + filter4Edge16 | `filter14VertNEON` 2.5% | dav1d `looprestoration`/`lf` avx2 | med |
| **B3** | frame samples widen: loadSampleRows8 | `loadSampleRows8NEONAsm` 2.1% | trivial `vpmovzxbw` u8→u16 | low |
| **B4** | cdef findDirection / findDirectionDual | `cdefFindDirectionNEONAsm` 1.2–2.4% | dav1d `cdef_dir_avx2` | med |
| **B5** | dsp addRawTransformPlaneBlock | `addRawTransform8NEONAsm` 1.2% | mirror existing addResidual AVX2 | low |
| **B6** | restoration SGR: boxsum/selfguided/selfguidedFast | cold on these clips; hot on SGR content | dav1d `looprestoration_avx2` sgr | med-high |
| **B7** | transform invglue: clampRound / narrowStore; inverseDCT32Col2/Row2 tails | small | mirror NEON invglue | low |
| **B8** | encoder tail (SVT-parity): forward hybrids 8x8 {ADST-DCT,DCT-ADST,ADST-ADST,IDTX}; pixelStats* (20 shapes); hadamard4/8/16/32; satdCoeffs; residualBlock; rdStatsBlock; realtimeAvg8x8/Quad; scalePlaneNearest{,16} | encoder hot path (BenchmarkVideoEncoderRealC1080p) | libaom/SVT `aom_dsp` avx2 (satd/hadamard/variance) + fwd_txfm avx2 | high (many kernels — fan out) |
| **B9** | intra amd64: subsampleLuma8, dirRowInterp8, dirAboveRun8, dirLeftCol8, predictFilterIntra8 | intra directional/CfL | dav1d `ipred_avx2` | med |
| **B10** | tile coeff contexts: coeffCulLevel, coeffInitLevels, coeffNzMapContexts | entropy-adjacent, moderate | goav1 NEON is the reference shape | med |

### GROUP C — cold / rare / intentional (skip or defer)

- **inverseADST4/8 Row2 rebind, inverseDCT4 Row2 rebind** — NEON adapters exist
  but init deliberately leaves them PURE-GO; documented "measured slower than
  pure-Go" (small/clamp-heavy kernels don't amortise the call). Do NOT flip
  without new measurement. (The ADST *value* is in A1's ADST16 + columns, which
  have NO kernel at all — that is the real win, not rebinding the row-4/8 asm.)
- **inverse identity (idtx) row/col NEON** — cold; identity residual is add-only.
- **inverseDCT64 Row2/Col2 (old 2-lane kernels)** — UNBOUND and unsafe (~2KB
  manual frame under NOSPLIT$0 overflows nosplit headroom = the historical
  "64x64 DCT fuzz seed" corruption). Superseded by Row4/Col4. **Recommend
  deleting the dead .s bodies** as a real cleanup fix.

---

## PORT THESE FIRST

**arm64 (group A — this host's decode/encode CPU immediately):**
1. **A2 highbd wide loop filter (filter6/8/14 16-bit)** — biggest single arm64
   win (~9% cum on 10-bit), and the cleanest (mirror the existing 8-bit
   `filter_wide_neon_arm64.s`; `filter4Edge16NEON` already proves the 16-bit
   sample layout).
2. **A1 inverse ADST NEON (ADST16 + ADST8/4 columns)** — 3.7% flat on intra;
   reuse `tools/itxgen/`. Highest ceiling, medium-high effort.
3. **A3 highbd filter-intra** — low-med effort, mirror the 8-bit NEON.

**amd64 (group B — fan out one agent per bucket):**
- Cheap/high-certainty first: **B3 frame widen**, **B5 addRawTransform**,
  **B7 invglue** (all low effort, direct NEON→AVX2 mirrors).
- Then the hot inter/postfilter set: **B1 compound blend**, **B2 wide loopfilter**,
  **B4 cdef direction**.
- Encoder SVT-parity: **B8** (large — split forward-hybrids / satd+hadamard /
  pixelStats / rdstats+residual / scale into separate agents).
- **B6 SGR**, **B9 intra directional**, **B10 tile coeff** as follow-ups.

Suggested fan-out (non-overlapping packages): `motion`(B1) · `loopfilter`(B2+A2)
· `frame`+`dsp`(B3+B5) · `cdef`(B4) · `transform`(A1+B7+B8-fwd-hybrids) ·
`prediction`(A3+B9) · `restoration`(B6) · `encoder`(B8-metric/rdstats) ·
`tile`(B10).

---

## Pinned DEAD-ENDS — do NOT re-attempt (from memory)

- **Entropy symbol-decode NEON** (`decode_symbol_adapt`): built + byte-exact, but
  goav1's trusted CDF path is ~89% 2–4-symbol alphabets; NEON only wins ≥~9
  symbols (never hit). Scalar `readSymbolTrusted` is optimal. The profile's
  entropy % is irreducible.
- **Blanket bounds-check elimination via reslicing**: measured REGRESSIONS on
  arm64 (add-residual +38%, Fill16 +10%). Hot inner loops already check-free.
- **In-place CDEF staging** (dav1d `cdef_apply_tmpl`): byte-exact + zero-alloc but
  +3.15% regression — goav1's CDEF kernel is uint16 so a byte→uint16 widen is
  mandatory; the one-shot NEON widen beats per-unit strided widens. Prereq to
  revisit = a full uint8 CDEF kernel rewrite (separate large effort).
- **madvise/arena port**: MIRAGE — the corpus-profile madvise % is test-harness
  setup churn; steady-state public decode is 0 B/op. No per-frame arena exists.
