# Architecture-gap plan — round N+1 (audited 2026-07-06, source-verified)

Decoder bar: dav1d (byte-identity enforced; speed gap 3.64x). Encoder bar:
SVT p12 (ST cpu gap ~3.5x at better quality). Items already landed/in-flight/
pinned per PERF_PLAN.md + ENCODER_DEPTH_PLAN.md excluded. madvise ~9% in the
profile harness is a VERIFIED MIRAGE (corpus-harness churn) — not recoverable.

## DECODER (ranked)

D-1. **64-bit entropy window** (dav1d msac ec_win=size_t; goav1 ecWindow=32,
dif uint32, refills 2-3 bytes vs dav1d ~6 bytes at half the frequency).
The ONLY remaining structural divergence in the entropy decoder (CDF storage,
adapt specializations, cursor shape all at parity). ~1-2% e2e, more on
intra/entropy-heavy clips. Byte-exact by construction (buffering width only).
PLAN (each phase full-gated): P1 pure-Go widen (ecWindow 64, dif uint64,
8-byte BigEndian fast-path refill, mind tellOffs/trailing-bits accounting =
the subtle spot; route the 2 arm64 asm entries to pure-Go temporarily; if
net-red alone, hold for P2); P2 re-port readBitTrusted + readCDF4HighToken
arm64 asm for 64-bit dif (dav1d msac.S is the 64-bit template); P3 sweep
remaining inlined refills (reader.go:333,370,412,470,532,584). Files:
internal/av1/entropy/reader.go + 2 asm pairs. 2-3 sessions.

D-2. **sbrow-chained postfilter pipeline** (dav1d decode.c:3196-3240
interleaves recon and filter_sbrow LF→copy_lpf→CDEF(8-row lag, prev_mask)→
LR per sbrow, L1-hot; goav1 runs whole-frame stage-major passes —
postfilter.go:688-791 — restreaming 1.4-3.1MB frames 3-4x from L2/SLC;
"banded" variants band within a stage only and have no production callers).
~1.5-4% e2e, bigger at ≥1080p/10-bit; also the enabler for recon∥filter.
**PRECONDITION: wait for the CDEF u8 agent to land (its API is what the
chain drives).** PLAN: P0 resumable per-band step APIs (byte-identical
refactor); P1 chained driver gated to single-tile/8-bit/no-superres with
dav1d's exact lag discipline (CDEF 8-row lag + prev-mask seam = hardest
byte-exactness spot; differential chained-vs-stage-major over the full
vector suite first); P2 fold temporal-MV 16-row window (D-3); P3 interleave
with recon via the wavefront row-progress hooks; P4 multi-tile/HBD/superres.
Multi-session, HIGH risk, byte-gated per phase.

D-3. **Temporal-MV projection window** (dav1d rp_proj = 16-row ring ~19KB,
refmvs.c:701-742 load_tmvs per sbrow; goav1 whole-frame TemporalMotionField
~780KB + full Clear() per frame, motion_field.go:187-248). ~0.5-1%. Fold
into D-2 P2. Spatial ref-MV stack itself: PARITY (no work).

D-4. Coeff decode: PARITY confirmed (TX-class contexts, scan tables,
eob-class fast paths all present). Optional <1% micro-bundle: fuse dequant
into the sign loop (dav1d recon_tmpl.c:655-724) + activeRows prefix-max
table (scan.c:319-375). Low priority.

D-5. Misc: inverse-TX eob pruning present; ExtendBorders removal needs
audit-first (~0.3-0.5%); memmove 2.9% unattributed (peek first);
show-frame publishes by pointer (no copy). D-6: multi-tile bitmask LF
(wip branch) NOT worth finishing now (−8.2% on sparse multi-tile; no dav1d
shape to port for the run-merge dependency).

## ENCODER (ranked)

E-A. **TX path: goav1 does strictly MORE TX work than SVT p12** — SVT
P-frames: txs_level=0 (tx_depth 0 = largest square TX, up to 64x64),
DCT_DCT only, RDOQ off, + LPD1 TX shortcut detectors (set_lpd1_tx_ctrls:
zero_y_coeff_exit, skip_tx_th=30, use_mds3_shortcuts_th, neighbour info).
goav1: leaves capped at 8/16 (a 64x64 block = 16 TXBs of 16x16 vs SVT's 1),
+ trial-priced 16→8 splits + 8x8 TX-type trials. Hits BOTH cpu
(finishInterTXB 8.6% cum) AND the serial entropy write (~7.3ms/frame =
the wavefront Amdahl bound). Expected −3-8% cpu + serial-write shrink.
PLAN: P0 instrument (TXB counts by size, trial cpu share, write ms by
size); P1 LPD1 shortcut detectors (keeps machinery, stops paying on easy
blocks — SVT's actual mechanism, quality ≈ flat); P2 largest-TX experiment
behind a flag (audit TX32/64 inter writer support first; quality gate §3 —
goav1 has +0.4dB realC headroom to spend); P3 re-measure the serial-write
Amdahl bound (pays double for raw fps).

E-B. **Open-loop ME on retained SOURCE reference planes** (SVT HME+fullpel
ME run on PA source refs — motion_estimation.c:1132 me_ds_ref_array; only
MD/subpel/RD are closed-loop. goav1: only HME is source-based; everything
full-res runs on recon). Fresh evidence: depth-P2 measured recon noise
compressing dev_16x16_to_8x8 — the exact gate signal; ENCODER_DEPTH_PLAN
Risk 4 flagged it. Drift bounded (prediction/RD stay recon; source seed
only recenters unchanged subpel). Expected: quality +0.05-0.15dB and/or
unlocks depth-removal levels (each ≈ −5-12% P-frame cpu); direct cpu ≈
neutral (~2.1MB Y memcpy/frame ≈ 0.3%). PLAN: P1 retention plumbing
(lastSrcY/goldenSrcY + quarter planes, prealloc, pipelining double-buffer
discipline, byte-identity oracle); P2 switch signal producers one at a
time measured independently (depth sweep FIRST — dev distribution should
widen; then static-bypass grids, int-pro, leaf fallbacks); P3 var-tree
experiment (libaom feeds it recon — measure, keep winner, pin loser);
P4 golden + E5 gate. Coordinate with the depth P2/P3 agent.

E-C. **Cyclic-refresh AQ** (SVT rc_aq.c:599-660: ~20-25% of SBs refreshed
per base frame with segment delta-q, max_qdelta_perc=60, RC-integrated;
goav1 has full segmentation syntax but NO production caller — dead
plumbing). Attacks screen rate overshoot (E7) + post-key convergence.
Quality/rate-shaped, cpu ≈ neutral, MEDIUM-HIGH risk. Rank behind E-A/E-B.

E-D. MD staging: PARITY confirmed (mds0 = faithful LPD1 funnel; "full TX
on winner" IS md_stage_3 at LPD1). E-E. Encoder CDEF/LR: NO GAP AGAINST US
(SVT p12 runs a small CDEF search we skip; LR off both sides). Optional
future quality lever only.

## Execution order
Decoder: D-1 (independent, can start immediately) → D-2 after CDEF u8
lands (+D-3 folded) → D-4 micro-bundle opportunistically.
Encoder: E-A P0/P1 after depth P2+P3 lands → E-A P2 largest-TX experiment
→ E-B open-loop ME (coordinate with depth agent's sweep) → E-C.
