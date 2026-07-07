# ARCH_BACKLOG — M2/M3 sliced program (agent-executable)

Companion to PLAN.md. Each slice is sized for ONE agent run: bounded files,
one coherent change, its own byte-exact gate. Dispatch in the wave/concurrency
order below (files are disjoint across programs → one slice per program runs
concurrently). Every slice cites file:func; a slice you can't execute from its
text alone isn't ready — re-read the code.

GATES (identical every slice):
- Decoder byte-exact: make dryrun-fast + dryrun-extended (226/226 strict MD5).
- Encoder byte-exact: four-clip SHA-256 identity (realA/B/C/screen) + the
  encode→decode→bytes.Equal round-trip (encoder/crosscheck_test.go) + spend
  gate unchanged + wall ≥101fps.
- Asm/representation changes: a lockstep differential harness (kernel-on vs
  kernel-off over adversarial streams + scrambled/warmed CDFs, comparing
  coder-state + post-op CDF image; a GOAV1_*_DIFF soak knob), modeled on
  tile/coeff_asm_diff_test.go and mv_asm_diff_test.go.
- Zero-alloc: scripts/check_allocs.sh; no closures/func-ptr dispatch escapes.

D3 TEMPLATE (reuse verbatim for B/C asm): reader_coeff_base_arm64.s (NOSPLIT
leaf, 64-bit window, shared refill block) + reader_coeff_base_arm64.go (shim +
compile-time offset asserts) + reader_coeff_base_generic.go (stub) +
tile/coeff_asm_dispatch.go (static-bool dispatch + GOAV1_DISABLE_COEFF_ASM) +
tile/coeff_asm_arm64_spec.go (spec doc) + tile/coeff_asm_diff_test.go (harness).

KEY FINDING: the fused walk ALREADY exists at block granularity
(frameWorkTileResidualLoopController is a BlockLoopCoeffController;
decodeBlockLoopVisitWithCoeffControllerPtr interleaves decode→predict→per-TXB
recon in one pass — block_loop.go:899,1016-1033). Program A is the RESIDUAL
gap only: per-TXB/plane geometry recompute + one buffer round-trip.

## PROGRAM A — decoder M2: collapse residual layer hops

A1 — Prime CHROMA job-geometry (skip per-TXB U/V lookup ladder).
  Files: threading/reconstruct.go, threading/tile_residual.go (bind ~1119-1163).
  Upstream: dav1d recon_tmpl.c:1927-1960 (uvdst resolved once, walked in place).
  Change: extend frameWorkPrimeLumaReconGeometry (reconstruct.go:161) to prime
    JobOutputPlane U/V into geomCache + a chromaReady flag + a
    blockCoeffGeometryChromaKnown fast path mirroring the luma one
    (reconstruct.go:569), so the chroma arm (tile_residual.go:1987) skips
    blockCoeffGeometryTrusted→cachedJobOutputPlaneTrusted (reconstruct.go:414).
    Guard on MonoChrome.
  Gate: decoder byte-exact. Yield ~0.5-1.5% (chroma≈1/3 of TXBs; peer to the
    landed luma prime a5f3c540). Risk L. Deps none. ∥ all of B, C0.

A2 — In-place LUMA TXB-grid walk (drop per-TXB position recompute). PIVOT.
  Files: threading/tile_residual.go (reconstructTXBWithNonZero, reconState),
    threading/reconstruct.go (new …WithRunningOrigin seam).
  Upstream: dav1d read_coef_tree/decode_b (recon_tmpl.c:731,1900-1922) — dst is
    the loop induction var (&dst[x*4], dst += stride*4*h); never recomputed.
  Change: thread a running per-plane (originX,originY,dstOffset) cursor through
    reconState, advanced by tx dims as TXBs are visited, replacing the
    blockCoeffGeometryLumaKnown recompute (reconstruct.go:593-608) with a
    running-origin add. Keep the edge-clip branch (reconstruct.go:611-631) only
    for the frame-edge tx; interior tx takes the branchless path.
    predictBlockBegin resets the cursor per block.
  Gate: decoder byte-exact + a table test over 19 tx sizes × edge positions
    asserting identical (x,y,visW,visH) vs blockCoeffGeometryLumaKnown.
  Yield ~1-3% (widest error bar — could be near-zero if the compiler already
    hoists the leaf; that's why the A1+A2 checkpoint exists). Risk M.
  Deps: A1 (shared cursor plumbing). NOT ∥ A1/A3. ∥ all of B, C0.

A3 — Hoist inter interp-filter + plane-geometry out of per-plane predict fan-out.
  Files: threading/predict.go (predictBlockInterPtr:197,
    predictBlockInterWithFiltersPtr:214, blockPredictionPlaneGeometry:2240).
  Upstream: dav1d resolves filter_2d once/block, dst offset once/plane.
  Change: compute filters + the 3 plane geometries once/block into a stack
    struct, pass by value into the existing …WithGeometry plane entry points
    (predict.go:1150,1436,1580); eliminate the per-plane
    blockPredictionPlaneGeometry→cachedJobOutputPlaneTrusted→
    frameWorkBlockPlanePosition (predict.go:2240-2282).
  Gate: decoder byte-exact. Yield ~0.5-1.5%. Risk M (broad inter surface:
    compound/OBMC/warp/inter-intra each must consume the hoisted geometry).
  Deps none. ∥ A1 (different file), all of B. NOT ∥ A2 only if A2 edited
    predict.go (it doesn't) → A3 ∥ A2 is valid.

A4 — Single dequant+transform scratch pass (fold the two int32 buffers).
  Files: reconstruct/block.go (reconstructPlaneBlockTrustedAtWithGeometry:163).
  Upstream: dav1d inv_txfm_add writes into dst; coeff buffer reused as scratch.
  Change: where the transform kernel is in-place-safe, alias dequant &
    transformScratch (block.go:175-176) so the inverse overwrites its input,
    removing one L1 buffer/TXB. Per-tx-type verification required (partial land
    OK). NOTE: buffer REUSE not a cache restructure — not covered by the
    Apple-cache pin, but verify no regression.
  Gate: decoder byte-exact + reconstruct unit tests all tx types/sizes.
  Yield ~0.3-0.8%. Risk M. Deps none (diff pkg). ∥ A1/A2/A3, all of B.

A GO/NO-GO: land A1+A2, measure. If A1+A2 < 3% e2e on p720_inter_q32 the
fused-walk plateau is real → stop A3/A4, reallocate to C.

## PROGRAM B — encoder M2: entropy-Writer spine ASM (mirror of decoder D3)

Serial-tail context: the wavefront parallelizes decision/predict/transform but
entropy WRITE stays a serial replay (adaptive CDFs are order-serial,
pframe_wavefront.go:9) — the ~6.5ms serial write bound. Writer hot funcs:
normalize (writer.go:141), encodeQ15 (:215), WriteBoolQ15 (:245), WriteSymbol
(:311), WriteCDF/4/4Zero/5/7 + WriteBinaryCDFTrusted (:351-1052). Per-TXB entry
points tile/coeff_write.go (16x16 body :1040/:1107-1171). Upstream: SVT
bitstream_unit.c svt_od_ec_enc_normalize:106 / encode_q15:212 / encode_bool_q15
:240 / encode_cdf_q15:270.

B0 — Writer-spine spec + differential harness (enabling; like D3-a).
  Files (new): entropy/writer_asm_arm64_spec.go (doc), entropy/
    writer_asm_diff_test.go (mirror coeff_asm_diff_test.go).
  Change: register/memory plan for a per-TXB write kernel (Writer offsets
    buf/offs/low/rng/cnt/err writer.go:56-63 pinned by compile-time asserts,
    reader_coeff_base_arm64.go:17-33 pattern); the normalize/writeOut/
    propagateCarry block as the shared textual sub-block (write-side analog of
    the reader refill); boundary = one kernel/TXB over base+BR+sign+golomb
    write loop (coeff_write.go:1107-1171). Harness: lockstep encode kernel-on
    vs off comparing Finish() bytes + Tell() + Writer state + post-write CDF
    image + the round-trip oracle. Plan GOAV1_DISABLE_WRITER_ASM kill-switch.
  Gate: harness green against pure-Go writer both switch positions (no kernel
    yet). Yield 0. Risk L. Deps none. ∥ all of A, C0.

B1 — normalize/carry write-out ASM primitive (shared sub-block bring-up).
  Files (new): entropy/writer_norm_arm64.{s,go}, writer_norm_generic.go.
  Upstream: svt_od_ec_enc_normalize:106 + goav1 writer.go:141-211.
  Change: lower normalize+writeOut+propagateCarry to one NOSPLIT leaf on
    *Writer in registers. De-RISK the shared sub-block; likely e2e-flat alone
    (primitive-only asm is flat — pin) → keep only as a building block for B2.
  Gate: encoder byte-exact both switch positions + harness. Risk M (carry into
    written bytes; round-trip oracle catches it). Deps B0. NOT ∥ B2.

B2 — Per-TXB coefficient-write loop kernel (load-bearing).
  Files (new): entropy/writer_coeff_arm64.{s,go} + stub; new
    tile/coeff_write_asm_dispatch.go; wire coeff_write.go:1107-1171 + 8x8/32/4x4.
  Upstream: svt_od_ec_encode_cdf_q15:270; goav1 coeff_write.go:1107-1171.
  Change: one NOSPLIT kernel/TXB over base+BR+sign+golomb write with low/rng/
    cnt/offs register-resident, CDF rows adapted in place (write-side mirror of
    D3-b + D3-c). Reuse B1 normalize sub-block textually. Keep P0/context in Go.
  Gate: encoder byte-exact both switch positions + soak. Yield ~2-4% wall
    (shortens the serial tail → raw-fps ~1:1 with serial fraction; D3 precedent
    took reads to ≤2%). Risk M-H. Deps B0,B1. ∥ A, C1 (coeff_write.go vs
    coeff.go different files). Serialize vs any slice editing coeff_write.go.

B3 — Tokenize/pack loop glue fusion (Go, no asm).
  Files: encoder/pframe_residual.go (:4264-4389), tile/coeff_write.go
    (prepTXB/level-scan :629,1076-1152).
  Change: collapse the double scan-order walk in …ContextTrustedArray writers
    into one pass (derive base-context inline like A2), removing absLevels/
    lowerContexts intermediates where B2 consumes levels directly.
  Gate: encoder byte-exact. Yield ~0.5-1.5%. Risk M. Deps: merge after B2
    (shared coeff_write.go). ∥ A, C.

B GO/NO-GO: after B2, if write-pass wall share < 2% the writer is call-bound
below the asm floor → stop B3.

## PROGRAM C — decoder M3: block-loop spine ASM (gated on A's post-landing profile)

C0 — Spine spec + fold decision (enabling; RE-GROUND on a fresh profile AFTER A).
  Files (new): tile/block_loop_asm_arm64_spec.go (doc).
  Change: plan for two folds: (a) D3-d — fold eob-class entry (P0) + base
    context into coeffBaseLevels2DARM64 so the whole TXB is one call (deferred
    at coeff_asm_arm64_spec.go:131); (b) ReadInterReferences run (~0.5-1.0%,
    context-coupled). Decide with A's operating point. Yield 0. Risk L.
    Deps: A's profile. ∥ all A, all B.

C1 — D3-d: fold P0 (eob entry + base context) into the TXB base-levels kernel.
  Files: entropy/reader_coeff_base_arm64.{s,go}; tile/coeff.go (:1147,:1877).
  Upstream: dav1d decode_coefs (recon_tmpl.c:340-525) — eob+base in one func,
    MsacContext register-resident throughout.
  Change: extend coeffBaseLevels2DARM64 to also do the eob-class read
    (EOBFlag16..1024), optional EOBExtra, and the offsetBits-1 equiprobable
    bits (P0) so dif/rng/cnt never round-trip between P0 and P1. Keep P3
    (coeffSignGolombARM64) separate.
  Gate: decoder byte-exact both switch positions + extend coeff_asm_diff_test.go
    (EOB-family streams + near-saturated CDF seeds :340-402). Yield ~0.5-1.5%.
    Risk M. Deps C0, prefer after A2. ∥ A (threading/reconstruct), B (writer).

C2 — ReadInterReferences run kernel — CONDITIONAL.
  Files (new): entropy/reader_inter_ref_arm64.{s,go} + stub; wire
    tile/block_loop.go decodeBlockPredictionModeInto:1072.
  Change: lower the ReadInterReferences contiguous run to a kernel taking
    neighbor contexts precomputed as args (like mvResidualARM64 takes row
    shapes). Yield ~0.3-0.8% (may be e2e-flat like primitive-only asm → carry
    kill-switch, be ready to pin-negative). Risk M-H. Deps C0,C1. ∥ A, B.

C GO/NO-GO: gated on A's profile. If post-A the spine < 6% cum, land C1 only,
re-plan M4 with data.

## WAVES (3-worker, files disjoint across programs)

Wave 1: A1 (threading/reconstruct) ∥ B0 (entropy writer spec/test) ∥ C0 (doc).
Wave 2: A2 (tile_residual+reconstruct) ∥ B1 (writer_norm) ∥ A3 (predict).
  → CHECKPOINT: measure A1+A2 e2e; apply A GO/NO-GO.
Wave 3: B2 (writer_coeff + coeff_write) ∥ A4 (reconstruct/block) ∥ C1
  (reader_coeff_base + coeff.go).  → CHECKPOINT: re-profile; apply C GO/NO-GO.
Wave 4: B3 (after B2, shared coeff_write) ∥ C2 (conditional).
Merge serial, worktrees, each slice runs its own full gate before land.

## CEILINGS (honest)
- A (fused-walk residual): ~2-6% e2e → decoder ~3.42x → ~3.2-3.3x. Bounded
  (kernels SIMD-saturated, cache restructures dead). A2 widest error bar.
- B (writer spine): ~2-5% wall → encoder ~3.3x → ~3.1-3.2x + raw-fps lift
  (shortens the serial tail). HIGHEST-CONFIDENCE mechanism (proven D3 mirror);
  magnitude capped by the 6.5ms serial-tail size.
- C (spine asm): ~0.5-2% e2e, widest error bars, conditional on A. Endgame tier.
- COMBINED: necessary early milestones, NOT sufficient for 1.4x (need ~2.4x
  more). The next-largest levers after these: F-1b full PREP_BIAS signed-prep
  migration (decoder, all-or-nothing) + E3 cross-package transform-kernel
  fusion (encoder). Re-plan there after this backlog.
