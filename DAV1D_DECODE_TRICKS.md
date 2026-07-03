# dav1d → goav1 single-thread decode trick catalog

Standing program (user directive 2026-07-03): *"steal stuff from dav1d for decode
single-thread tricks and implement them no matter complexity, multi-session."*
This file is the ranked, source-grounded hit-list. **Research only — nothing here
is implemented yet; execution agents port each item.**

- Metric: **CPU time (core-seconds/frame)**, not ns/op. Measure `GOMAXPROCS=1`.
- Gap (2026-07-03, after the loop-filter NEON run-batcher landed): corpus decode
  **~3.94× dav1d single-thread** (was 4.43×; the LF batcher took −15% decode).
  Honest floor ~2–2.5× is irreducible pure-Go-vs-hand-asm; the winnable slice is
  distributed scalar/overhead.
- dav1d pinned at `third_party/upstream/dav1d/src/`; goav1 at `internal/av1/`.

## How this catalog was built
Fresh `GOMAXPROCS=1` CPU profiles of four corpus clips (`p720_inter_q32`,
`p288_inter_q55` sparse, `p360_inter_q32_10bit`, `p288_intra_q32` all-intra) via a
scratch `goav1_oracle` hook (`decodeCorpusClipDiscard` loop, 40 iters), plus
first-hand reads of dav1d `msac`, `itx_tmpl`, `recon_tmpl`, `cdef_apply_tmpl`,
`ipred_tmpl`, `env.h`, and the matching goav1 packages. **Key structural finding:
goav1 has ALREADY ported the overwhelming majority of dav1d's decode fast paths**
(DC-only itx, eob row-pruning, identity kernels, skip/all-zero shortcut, the
fused-ish raw add, a table-driven branchless coeff-context loop, and NEON for
PAETH/SMOOTH/DC/directional/CfL/convolve/CDEF/wiener). The genuinely missing items
are few and are ranked below.

---

## Profile snapshot (GOMAXPROCS=1, non-loop-filter reducible leaves)

Loop filter (`filter14EdgePureGo`, `Filter*Edge`, `flatMask4`, `needsFilter*`) is
now largely NEON and is **OWNED BY ANOTHER AGENT — excluded from this hit-list.**

| leaf | clip | flat% | nature |
|---|---|---|---|
| `runtime.madvise` | all | 8.6–14.9% | GC/alloc churn — **attribution disputed, see item 3** |
| `motion.convolve2DHighBDNEON` | 10-bit | 9.6% | SIMD floor (HBD 2D) |
| `prediction.predictFilterIntraBlockDirect8` | intra | 4.4% | **filter-intra, zero SIMD (item 1)** |
| `entropy.readCDF4UpdateKnown` | intra | 4.1% | entropy, near floor |
| `cdef.cdefFilterBlock8SecondaryNEON` | all | 2–5% | CDEF secondary-only pass (already fused when pri+sec both on) |
| `decoder.frameWorkStoreCDEFUnit` + `frame.storeSampleRowsTrusted` + `frameWorkFillCDEFInputSentinelRow` + `frame.loadSampleRows` | inter | ~4.3% combined | **CDEF full-plane staging copy (item 2)** |
| `transform.inverseSeparableBlockClampedRowsToScratch` | intra | 3.6% cum | recon column pass → scratch (item 4) |
| `tile.blockLoopLoadRootContext` | all | 1.3% | AoS ctx field-copy (item 5) |
| `frame.storeSampleRowsTrusted` / edge widen / `frameWorkClipVisiblePixelsToWindow` | intra | ~2–3% | edge-prep copy/fill/dedupe (item 6) |

**Correction on CDEF:** goav1 already fuses primary+secondary into one pass
(`cdef.cdefFilterBlock8NEON`) when both strengths are nonzero
(`cdef/filter_neon_arm64.go:89 dispatchFilterBlockNEON`). The profiled
`cdefFilterBlock8SecondaryNEON` is only the primary==0 case — so "fuse pri+sec"
is NOT an available lever. The CDEF win is the staging model (item 2), not fusion.

---

## Ranked hit-list — (yield × certainty) / effort

### 1. Filter-intra SIMD kernel  ★ top concrete win (SIMD-program-owned)
- **goav1 hotspot:** `prediction.predictFilterIntraBlockDirect8`
  (`prediction/filter_intra.go:157`) — **4.4% flat** on all-intra, present at lower
  weight on inter. It is the **only intra predictor with no dispatch slot** (grep:
  no `filterIntraImpl`, no `predictFilterIntraNEON`); every 4×2 chunk runs 8×7=56
  scalar MACs. The algorithm is already correctly tabled (`filterIntraTaps
  [5][8][8]int8`, filter_intra.go:25) and 4×2-chunked — only SIMD is missing.
- **dav1d technique:** `ipred_tmpl.c:618 ipred_filter_c` + `dav1d_filter_intra_taps`;
  NEON template `arm/64/ipred.S:3796 ipred_filter_8bpc_neon` (width jump table,
  `smlal` taps, `sqrshrun` for `(acc+8)>>4`+clamp).
- **Port approach:** add a `filterIntra8Func` dispatch slot in
  `prediction/intra_dispatch.go`, bind a NEON kernel in `intra_dispatch_arm64.go`
  modeled on `ipred.S:3796` — vectorize the 8 outputs × 7 taps within a chunk;
  keep the chunk→chunk left-column recursion sequential (matches dav1d). AVX2
  analog in `intra_avx2_amd64.s`.
- **Yield / certainty:** ~3–4% single-thread all-intra decode; **High** (self-
  contained kernel, existing bit-exact pure-Go golden to diff, dav1d asm template).
- **Byte-exact verification:** differential vs `predictFilterIntraBlockDirect8`
  over all shapes + zero-alloc; `make dryrun-extended` 226/226 with NEON active.
- **Effort:** Medium (hand-write + verify asm). NOTE: belongs to the SIMD
  sub-program ([[simd-plan]]); listed here because it is the #1 non-LF decode leaf.

### 2. CDEF staging → dav1d in-place 2-line + 2×8 backup
- **goav1 hotspot:** the full-plane bordered CDEF-input copy —
  `decoder.frameWorkStoreCDEFUnit` (1.71%) + `frame.storeSampleRowsTrusted`
  (1.55%) + `decoder.frameWorkFillCDEFInputSentinelRow` (1.06%) +
  `frame.loadSampleRows` (0.9%) = **~4.3% flat combined** on inter clips. goav1
  materializes a full padded plane per unit, then CDEF reads it.
- **dav1d technique:** `cdef_apply_tmpl.c` runs CDEF **in-place on the picture**,
  backing up only the ~2 lines + 2×8 pixel columns CDEF must read before they are
  overwritten: `backup2lines` (`cdef_apply_tmpl.c:41`), `backup2x8` (`:65`), guided
  by `CdefEdgeFlags` (`CDEF_HAVE_TOP/BOTTOM/LEFT/RIGHT`, `:106,124,146,181`). No
  full padded copy; borders handled by edge flags, not sentinel fills.
- **Port approach:** replace the per-unit full-plane staging with a small ring of
  backup line/column buffers + edge-flag-driven kernel entry, running the existing
  NEON CDEF kernels over the in-place plane. Restructures the postfilter CDEF I/O.
- **Yield / certainty:** ~3–4%; **Medium-High** (dav1d-grounded; requires careful
  border handling but output is deterministic).
- **Byte-exact verification:** `make dryrun-extended` 226/226; -race; 0-alloc.
- **Effort:** Medium-High (postfilter pipeline restructure; couples to the SB-row
  wavefront ordering).

### 3. Arena / scratch reuse to kill `runtime.madvise` (GC churn) — **VERIFY FIRST**
- **goav1 hotspot:** `runtime.madvise` **8.6–14.9% flat** across all clips
  (+`memclrNoHeapPointersChunked`, `kevent`). **Attribution is disputed and must be
  resolved before any port.** An `alloc_space` memprofile of the same run shows the
  churn is dominated by the **test harness**, not the decoder core:
  `testvector.corpusRunTileWork` 44%, `testvector.bindCorpusFramePool` 23%,
  `testvector.(*corpusPostFilterScratch).ensure` 19% — all in
  `internal/av1/testvector/`, which re-allocates scratch per frame. The decoder
  core allocates ~nothing (perf-baseline: steady-state ~2 allocs/op after the
  e2be4f0 alloc collapse). The coordinator/LF-agent flagged madvise ~9% as a real
  per-run cost; the memprofile says it is largely harness. **These must be
  reconciled on the PRODUCTION path.**
- **dav1d technique (if real):** one reused scratch union per task context —
  `Dav1dTaskContext.scratch` (`internal.h`) holds edges/prediction/coeff/cdef
  scratch in a single arena, allocated once, never per-block/per-frame. Plus flat
  per-frame buffers reused across frames.
- **Port approach:** *Phase-1 (bounded, ~1 session, zero risk):* re-profile via the
  **public streaming API** (`decoder.Stream.PushLowOverhead` with a reused decoder,
  the production integration shape) instead of `decodeCorpusClipDiscard`, and read
  the `alloc_space` profile. If madvise reproduces with allocations rooted in the
  `decoder`/`threading` packages → port the scratch-union arena (top item, ~6–9%).
  If it stays rooted in `testvector` → it is a **harness artifact; drop it** and the
  ~9% is not recoverable production cost.
- **Yield / certainty:** potentially **~6–9%** if real; **~0% if artifact.**
  Certainty of the yield is **LOW until Phase-1 resolves it** — which is exactly
  why Phase-1 is the cheapest, highest-information first action.
- **Byte-exact verification:** n/a for the measurement; arena port gated by
  dryrun-extended + -race + 0-alloc.
- **Effort:** Phase-1 = Small (measurement). Arena port = Medium.

### 4. Fused itx column-pass → destination store (recon)
- **goav1 hotspot:** the recon add path writes the int32 column-pass result into
  `transformScratch` (`transform/hybrid.go:243 inverseSeparableBlockClampedRowsToScratch`,
  3.6% cum intra) and then `dsp.AddRawTransformPlaneBlockTrusted`
  (`dsp/plane.go:156`) reads that scratch back, applies `(v+8)>>4`, adds to `dst`,
  clamps, stores. This already avoids the old int16-residual materialize+re-read,
  but still costs **one extra block-sized int32 write + read** vs dav1d.
- **dav1d technique:** the itx **column pass stores straight into the pixel plane**,
  folding round-by-4 + add + clip into the innermost store — no residual buffer at
  all: `itx_tmpl.c:116-118` (`dst[x] = iclip_pixel(dst[x] + ((*c++ + 8) >> 4))`);
  DC-only does the same (`:66-68`).
- **Port approach:** add fused column-kernel variants (NEON/AVX2 + pure-Go) that
  take dst plane + stride + bitdepth and do `dst = clip(dst + (col+8)>>4)` in the
  final store — push `AddRawTransform` *into* `inverse1DCol*`.
- **Yield / certainty:** low-single-digit % on transform-heavy clips; **Medium**
  (traffic saving is clear; benefit is diluted where predict/LF dominate).
- **Byte-exact verification:** the fold must reproduce `(v+8)>>4` + saturating add
  exactly per bytes-per-sample/bitdepth (`dsp/plane.go:205 rawTransformResidual`);
  full conformance matrix re-run.
- **Effort:** Medium (new fused SIMD kernel variants × arch × bitdepth).

### 5. Flat SoA block context (kill the per-root AoS field-copy)
- **goav1 hotspot:** `tile.blockLoopLoadRootContext` (`tile/block_loop.go:491`,
  1.3% flat) copies an **AoS struct of ~40 named scalar fields** one field at a
  time into `scratch` for every SB-row root (`:505-535` above, `:545+` left), plus
  per-plane `CoeffCtx.Above[plane]` copies.
- **dav1d technique:** `env.h:39 BlockContext` is pure SoA flat bytes
  (`uint8_t mode[32]; lcoef[32]; intra[32]; …`), updated by size-specialized
  contiguous splats `dav1d_memset_pow2[log2(w4)](…)` (`decode.c:735`, the
  `set_ctx`/`case_set`/`rep_macro` machinery `decode.c:712-780`). Neighbor context
  is read by index — **dav1d never copies a per-root scratch.**
- **Port approach:** convert above/left context to parallel flat arrays indexed by
  mi col/row; update via a memset-style splat sized by block-width-in-4s
  (mirroring `dav1d_memset_pow2`); read neighbors by index instead of materializing
  the per-root scratch copy.
- **Yield / certainty:** ~1–2%; **Medium** (pervasive but mechanical).
- **Byte-exact verification:** dryrun-extended + -race; the context values are
  unchanged, only their storage/copy pattern.
- **Effort:** Medium-High (touches the tile block-walk context plumbing broadly).

### 6. Edge-prep copy/fill/dedupe (byte-exact-safe, low risk)
- **goav1 hotspot:** `frameWorkLoadSampleRow`/`Col` (`threading/predict.go:4708`)
  **widen 8-bit samples to uint16 in a per-element loop** where dav1d memcpys;
  the tail extension (`predict.go:4133`) is a scalar store loop vs `pixel_set`
  memset; `frameWorkClipVisiblePixelsToWindow` (`predict.go:3679`) is called **2–3×
  per transform** (`:500,592,814`, and `blockPredictionPlaneGeometry :2232,2242`),
  each doing checked add/mul. Part of the ~9.5% "intra transform predict" cum.
- **dav1d technique:** `ipred_prepare_tmpl.c:76 dav1d_prepare_intra_edges` builds
  edges with `pixel_copy` (→memcpy, `:168`) + `pixel_set` (→memset, `:170,187`),
  native pixel width, gated by the `av1_intra_prediction_edges` bitfield (`:56`).
- **Port approach:** (i) replace the widen/tail loops with `copy()` / a memset-style
  fill helper; (ii) dedupe the repeated `frameWorkClipVisiblePixelsToWindow` per tx
  (memoize like the landed job-geometry cache). Optional bigger win: an 8-bit `[]byte`
  edge fast path (touches the `IntraEdges.Above []uint16` API — wider refactor).
- **Yield / certainty:** ~1–2% (more if the []byte edge path lands); **High** for
  the mechanical copy/dedupe part.
- **Byte-exact verification:** trivially byte-exact for copy/dedupe; dryrun-extended.
- **Effort:** Low (copy/dedupe); Medium (the []byte edge API change).

### 7. `activeRows` prefix-max table (free win)
- **goav1 hotspot:** `reconstruct/block.go:238 activeTransformRowsFromScan` computes
  the eob-pruned active row count with an **O(eob) loop + `%` per coefficient**
  (`row := int(scan[i]) % scanHeight; …`), where dav1d does a single table load.
- **dav1d technique:** `dav1d_last_nonzero_col_from_eob[tx][eob]` — a prefix-max of
  the scan's column index, built once (`scan.c:319-375`,
  `dav1d_init_last_nonzero_col_from_eob_tables:350`); `last_nonzero_col = table[eob]`.
- **Port approach:** precompute `prefixMaxRow[size][eob]` at init (the running max,
  identical formula); replace the loop with `activeRows = table[size][eob-1]+1`.
- **Yield / certainty:** tiny (<0.5%) but **certain**; it *is* the running max.
- **Byte-exact verification:** identical output by construction.
- **Effort:** Trivial. Good warm-up / bundle with another item.

### 8. CfL average-subtract dispatch/fuse (low priority)
- **goav1 hotspot:** `prediction.SubtractCFLAverage` (`prediction/cfl.go:160`) has
  **no dispatch slot** (subsample + apply already have NEON). goav1 splits CfL into
  3 passes (subsample→uint16, subtract→int16, apply) vs dav1d's fused `cfl_ac_c`.
- **dav1d technique:** `ipred_tmpl.c:657 cfl_ac_c` fuses subsample→sum→subtract-DC
  in one routine.
- **Port:** add `subtractCFLAverageImpl` NEON, or fuse subsample+subtract to drop a
  buffer pass.
- **Yield:** <1%; **Effort:** Low; **Risk:** Low. Bundle opportunistically.

### 9. Entropy 64-bit `ec_win` window (high risk, modest yield)
- **goav1 hotspot:** entropy is ~8–9% flat on intra (`readCDF4UpdateKnown` 4.1%,
  `ReadBitTrustedInline`, `readSymbolKnown`), refill folded inline. goav1 uses a
  **32-bit `dif` window** (`entropy/reader.go:19 ecWindow = 32`, `dif uint32`) — the
  libaom `od_ec` layout, refilling ~16–24 bits (2–3 bytes) per refill.
- **dav1d technique:** a **64-bit `ec_win` window** (`msac.h:36 typedef size_t
  ec_win`; refill `msac.c:41 ctx_refill` loads ~40 bits high-aligned; branchless
  renorm `ctx_norm:85` with `d = 15 ^ (31 ^ clz(rng))`, refills only when
  `cnt < d`). Roughly **halves refill frequency** vs the 32-bit window.
- **Port approach:** widen `dif` uint32→uint64 throughout the reader and adopt the
  high-aligned refill/renorm shape (byte-exact math). NOT the pinned entropy-NEON
  dead-end — this is a SHAPE/width change, not SIMD.
- **Yield / certainty:** ~1–2% on entropy-heavy clips, less elsewhere; **Medium.**
- **Byte-exact verification:** highest byte-exact risk in the catalog — gate with
  `go test ./...`, dryrun-fast, dryrun-profiles, extended 226/226, -race.
- **Effort:** High (pervasive width change in byte-exact-critical code). **Rank last.**

### (noted, LF-domain — not this agent's to port)
- Multi-tile decode still runs the **scalar edge-list LF apply**, not the mask
  path (single-tile already uses masks). Real lever, but loop-filter-owned by the
  other agent; cross-referenced here only.

---

## Already at parity (do NOT re-recommend — verified present in goav1)
- **itx DC-only** (`transform/block.go:189 InverseDCTDCOnlySampleBitDepth`, wired
  `reconstruct/block.go:143`) == dav1d `itx_tmpl.c:58-70`.
- **itx eob row-pruning** (`activeRows` threading, `reconstruct/block.go:193-235`)
  == dav1d `itx_tmpl.c:87-104` (derivation differs — see item 7).
- **itx identity/H_DCT/V_DCT** (`transform/hybrid.go:298`, `identity.go`).
- **recon skip/all-zero shortcut** (`threading/tile_residual.go:1989`) == dav1d
  `recon_tmpl.c:345,812`.
- **table-driven branchless coeff-context** (`tile/coeff.go:1706
  readCoefficientsTXBTracked2DWithGeo`, padded `levelsScratch`, `coeffPosTable`) ==
  dav1d `recon_tmpl.c:298 get_lo_ctx` + `dav1d_lo_ctx_offsets`.
- **tx-type-from-context** propagation (`block.Transform`).
- **NEON:** PAETH, SMOOTH/SMOOTH_V/SMOOTH_H, DC-sum, directional Z1/Z2/Z3,
  CfL subsample+apply, convolve X/Y/2D + clamped + HBD, CDEF (pri/sec + fused),
  wiener/SGR, dequant, inverse DCT8/16/32/64.
- **CDEF pri+sec fusion** when both strengths nonzero (`cdef.cdefFilterBlock8NEON`).

---

## Pinned dead-ends (built + measured + reverted — DO NOT re-attempt)
- **Entropy symbol-decode NEON** (dav1d `decode_symbol_adapt`): a full byte-exact
  NEON kernel was built and measured NON-winning. goav1's trusted CDF path is
  dominated by tiny alphabets (q32: ~66% 4-sym, ~23% 2-sym, **0% ≥9-sym**); NEON
  only beats the inlined scalar at ≥~9 symbols. `readSymbolTrusted` is already
  optimal for 1–2-iteration loops. See [[perf-deadends]]. (Item 9's window widening
  is a DIFFERENT, non-SIMD change — permitted.)
- **Blanket bounds-check elimination via reslicing**: `-gcflags=all=-B` overstates
  recoverable gain; on arm64 a bounds check is a single well-predicted CMP+branch,
  and the reslice idiom adds slice-header arithmetic + blocks `range` codegen
  (measured regressions: 16-bit add-residual +38%, Fill16 +10%). Targeted
  guard-disjunct BCE only (no reslice). See [[perf-deadends]].

---

## START NEXT — recommendation

**Start with item 3, Phase-1 (arena/madvise verification).** It is the cheapest,
highest-information action and it gates the single largest potential lever. The
profile shows `runtime.madvise` at 9–15% but an `alloc_space` memprofile attributes
the churn to the `testvector` corpus harness, not the decoder core — these must be
reconciled on the production path before spending any port effort.

**Bounded Phase-1 (≤1 session, zero decode-output risk):** re-profile
`GOMAXPROCS=1` decode via the **public streaming API** (`decoder.Stream.
PushLowOverhead` with a reused decoder, the production integration shape) instead
of `decodeCorpusClipDiscard`, capture both CPU and `alloc_space`. Decision gate:
- allocations root in `decoder`/`threading` → madvise is real; escalate to porting
  dav1d's `Dav1dTaskContext.scratch` arena (top item, ~6–9%).
- allocations stay rooted in `testvector` → harness artifact; **drop item 3** and
  start **item 1 (filter-intra SIMD)** as the #1 concrete win, with **item 2 (CDEF
  staging)** and **item 6 (edge-prep copy/dedupe, byte-exact-safe)** as the parallel
  medium track.

If a concrete kernel win is wanted immediately regardless: **item 1 (filter-intra
NEON)** is the highest-certainty non-LF decode leaf (4.4% flat, self-contained,
golden reference + dav1d asm template).
