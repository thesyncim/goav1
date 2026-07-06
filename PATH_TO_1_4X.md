# Path to 1.4x — closing decode (vs dav1d) and encode (vs SVT p12) to within 40%

Written 2026-07-06 at decoder 3.52x / encoder ST ~3.4x. This is the honest
end-game roadmap. Companion to PERF_PLAN.md (execution rules + standings) and
ARCH_GAPS_PLAN.md (the audited architecture map). Everything here obeys §0 of
PERF_PLAN: port-don't-invent, byte-exact strict suite per decoder commit,
quality-gated matrix per encoder commit, cpu_total alongside wall.

## The honest shape of the remaining gap

After today's landings (restoration in-place u8, CDEF in-place u8, depth
architecture, LPD1 TX detectors, HBD completion, vertical LF transposes),
BOTH references' architectures are matched: there is no known remaining
"they do fundamentally less work" item. What remains is:
  (1) a mapped tail of conventional levers (~15-25% each side),
  (2) an encoder-only QUALITY-SPEND reframe (we are +0.4-0.5 dB above SVT
      — 1.4x is defined at MATCHED quality, so the surplus is spendable),
  (3) the CODEGEN RESIDUE: Go glue vs clang -O3 + hand-asm in the serial
      paths (entropy/coeff/mode walk; encoder TXB pipeline + entropy write).
      This is 30-40% of both CPUs and shrinks only by moving the spine into
      assembly. Nothing forbids it (asm is sanctioned; purego tag preserved)
      — it is grind + byte-exactness discipline, proven tractable today by
      the corruption-canary + differential + 226-vector method.

Verdict: 2.0-2.2x both sides = high confidence, mapped below as M1-M3.
1.4x = achievable ONLY through M4-M5 (the asm-spine program). Commit in
stages; each milestone has a go/no-go: if a phase yields < half its
estimate, stop and re-plan rather than grinding a dead curve.

## DECODER: 3.52x → 1.4x (corpus 2158ms → ~860ms; must remove ~60%)

Current CPU split (p720 single-worker, post-2026-07-06): serial symbol/
coeff/mode decode + block-loop orchestration ~30-35%; prediction ~15%;
inverse TX + recon ~8%; postfilters ~13% (LF 3, CDEF 7.7, LR 2);
copies/memmove ~4%; glue/other remainder.

M1 → ~3.2x (in flight / queued, days):
- D-1 64-bit entropy window (DONE on branch d1-ec64-window, gates green,
  readCDF4 share halved; merge after idle re-measure).
- D-2 sbrow-chained postfilter (agent in flight; 1.5-4%, more at 1080p).
- LR/CDEF arena + construction shrink (PERF_PLAN task; improves the
  measured corpus number legitimately — dav1d also constructs).
- memmove attribution hunt (~2.9% unattributed; peek then kill).

M2 → ~2.8x (toolchain + glue shape, 1-2 weeks):
- **Go PGO (NEVER TRIED)**: build with default.pgo from a corpus decode
  profile. Devirtualizes + inlines hot glue. Expect 3-8% on glue-heavy
  code; measure both codecs. Zero risk (output-identical by definition —
  still gate with dryrun-extended).
- Targeted BCE via the proven guard-disjunct pattern on the top remaining
  bounds-check sites (NOT blanket reslicing — that pin stands).
- Glue shape: dequant fused into coeff sign loop + activeRows prefix-max
  (ARCH_GAPS D-4, <1%); refmv/projection window (D-3, folds into D-2);
  intra edge-prep audit.

M3 → ~2.2x (serial-walk asm phase 1, 3-6 weeks):
- **readCoefficientsTXB as one NOSPLIT asm kernel** (the whole TXB loop:
  eob classes, base/BR levels, golomb, DC sign, CDF updates — self-
  contained state: reader window, CDF arrays, scan table, levels buffer).
  This is dav1d msac.S-parity for the heaviest serial subtree (~6% cum +
  the symbol reads inside). The pinned "entropy NEON dead-end" does NOT
  apply: that was SIMD inverse-CDF search; this is scalar asm to remove
  per-symbol CALL + codegen overhead, dav1d's actual shape.
- Then the mode/MV symbol subtrees (decodeBlockPredictionModeInto ~10%
  cum) by the same method, hottest first.
- Gate: per-kernel differential vs pure-Go over adversarial streams +
  full 226 strict per commit. purego tag keeps the Go reference path.

M4 → ~1.8x (recon orchestration fusion, 3-6 weeks):
- Per-SB fused walks: predict→add-residual→recon with shared setup,
  fewer per-block dispatch layers (dav1d recon_tmpl.c shape).
- D-2 P3: interleave postfilter chain with recon rows (cache handoff).
- Prediction dispatch flattening (one branch tree → table).

M5 → 1.4-1.6x (endgame, open-ended):
- Extend asm coverage until ~90% of hot cycles are in kernels (block-loop
  spine, remaining glue). At this point goav1's cycle budget = dav1d's
  kernels + small Go orchestration tax. The last 10-15% is scheduler/
  runtime/misc — accept or chase case by case.
- Re-evaluate here: if M3+M4 landed on-estimate we sit ~1.7-1.9x and the
  remaining curve is known; if they underdelivered, 1.6x may be the
  pure-Go+asm plateau. Decide with data, not hope.

## ENCODER: ~3.4x → 1.4x (realC ST 1.72s → ~0.69s; must remove ~60%)

Current split (realC ST): encodePBlock ~64% cum (prediction/convolve ~20%,
TXB pipeline prepare+finish ~20%, mds0 ~8%, intra ~6%), CDF inits ~6%,
LF ~8%, CDEF ~4%, entropy write ~5-7%, HME ~2%.

M1 → ~2.9x (mapped, days-weeks):
- Route the encoder's CDEF apply to the u8 in-place walk (decoder landed
  it; encoder banded path still stages u16) + LF mask apply on the
  wavefront path (PERF_PLAN follow-up (a)).
- CDF-init cut (~6%: per-frame frozen-table copies — audit what actually
  needs re-init vs memcpy of a prebuilt image).
- Go PGO (same as decoder M2).
- Entropy-write micro (Writer hot loop shape vs od_ec_enc).

M2 → ~2.4x (THE QUALITY-SPEND REFRAME, 2-4 weeks):
- We hold +0.4-0.5 dB over SVT on every clip. 1.4x is at MATCHED quality.
  Re-run every pruning ladder that measured "red" under the hold-our-
  quality gate against an SVT-quality-matched gate instead: depth-removal
  arms at levels >9/14 (the infra is landed + kill-switched), mds0
  candidate/dist-band cuts, leaf subpel schedule, golden probe breadth.
  Each was individually red at hold-quality; the spend budget changes the
  answer. Gate: quality >= SVT's own numbers per clip (same-run), never
  below.
- E-B open-loop ME (ARCH_GAPS) if it buys quality back to re-spend.

M3 → ~2.0x (TXB pipeline fusion, 3-5 weeks):
- Fuse prepare+finishInterTXB: residual→fdct→quant→dequant→invTX→recon
  as one per-block asm-backed pipeline without intermediate buffer
  round-trips (currently ~20% cum with each stage a separate NEON call +
  Go glue). Same differential discipline as decoder kernels.

M4 → ~1.7x (entropy-write asm + orchestration):
- Writer symbol+CDF-update spine in asm (mirrors decoder M3; also shrinks
  the serial wavefront bound → raw fps).
- MDS0 candidate prediction batching (shared halo/setup for the 6
  candidates).

M5 → 1.4-1.5x (endgame): same character as decoder M5 — asm the encode
block-loop spine, then measure what plateau remains. If M2's spend +
M3/M4 land on-estimate we reach ~1.6-1.7x before the spine work.

## Program rules
- One milestone at a time per side; 2 concurrent agents; every decoder
  commit passes strict 226 (dav1d parity IS the correctness bar); every
  encoder commit passes the matrix at its gate (hold-quality for M1/M3+,
  SVT-matched for M2 spend items); every claim reports wall + cpu_total.
- Go/no-go at each milestone boundary against the estimates above.
- Pins are law: dead-ends in PERF_PLAN/ENCODER_DEPTH_PLAN/memory are not
  re-attempted without new evidence of changed economics.
