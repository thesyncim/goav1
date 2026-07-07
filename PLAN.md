# PLAN — beat dav1d (decode) and SVT p12 (encode), single-thread, to 1.4x

STANDINGS (2026-07-07): decoder 3.42x behind dav1d (corpus MD5-verified vs
both refs) · encoder ~3.3x behind SVT ST cpu at +0.5dB quality (spend gate:
scripts/spend_gate.sh). History/details: git log + memory; this file holds
only rules, open work, and pins.

## Rules
1. Port, don't invent — cite upstream (dav1d / svt-av1 / libaom) file:func.
2. Decoder = byte-identical to dav1d: make dryrun-fast + dryrun-extended
   (226/226 strict) per commit. One byte = revert.
3. Encoder = full 30m suite + four-clip matrix. HOLD gate: realC ≥ −0.05dB
   rate-adj, A/B hold, screen ≥ −0.10 at not-worse rate, wall ≥ 101fps.
   SPEND gate (quality-surplus levers): per-clip ≥ SVT same-run, cpu win ≥1%.
4. Zero-alloc steady state (check_allocs.sh; BenchmarkVideoEncoderRealC1080p
   0 allocs). Traps: closures, variadic logging, func-ptr dispatch escapes.
5. Git: explicit paths only; lowercase pkg-prefixed messages; NO AI mentions
   or trailers; push after gates + idle numbers. Red levers revert + pin.
6. Env vars are scaffolding: kill-switch dies after soak; stats/debug behind
   build tags; sweeps become constants. (Only GOAV1_DISABLE_COEFF_ASM lives.)
7. FASTEST MATH WINS (user 2026-07-07): internal representations are chosen
   for speed alone — output byte-parity is the only constraint. If dav1d's
   encoding of an intermediate is faster, migrate to it (task F-1).
8. Report cpu_total + wall. Interleaved same-run A/B under load; land idle.
9. Worktree setup: symlink third_party/upstream, testdata/benchcorpus,
   internal/av1/testdata/libaom from the main repo (rm empty dirs first).
   Corpus at /tmp/corpus (rebuild recipe: memory / git history of this file).
10. Measure: corpus gap = TestCrossDecoderCorpus (§ Makefile bench-corpus,
    PGO wired); encoder = cmd/qualitybench four-clip commands + ST flags.

## Path to 1.4x — findings-calibrated ladder (updated 2026-07-07)

WHAT THE CAMPAIGN HAS PROVEN SO FAR (shapes the estimates below):
kernel-coverage and architecture parity are DONE and bought 4.34x→3.42x /
5.2x→3.3x; symbol-read asm is a SOLVED chapter (all mode/coeff symbol reads
now ≤~2% combined); cache-locality restructures are dead on this host; the
quality-spend surplus is saturated on this corpus. What remains, by fresh
profile, is GO GLUE (per-block orchestration, neighbor-context scans, grid
bookkeeping) and the irreducible-until-asm'd call/codegen tax around already-
optimal kernels. Therefore the road is: glue ports → walk fusion → spine-asm
expansion, each with a go/no-go.

### DECODER 3.42x → 1.4x
M1 glue ports — LANDED this session (byte-exact, each e2e-measured):
- [x] D3 spine kernels: TXB base-levels+BR (f8aa3904), sign/golomb (e9395e24),
      MV residual (69c7a787) — symbol-read chapter closed (all ≤~2% now).
- [x] compound-X 8-bit emu-edge 36d590f0 (-1.17% p720).
- [x] LF level-cache walk-by-block 2539e1e7 (-1.48% p720).
- [x] D3-g refmvs glue b9b70d71 (-0.21% p720 / -1.21% p288).
- [x] inverse-transform clamp-skip 3d289493 (DCT64 -24%, -3.74% p720).
M1 remaining / open:
- [ ] F-1b PREP_BIAS FULL migration (deferred): partial is e2e-neg; only an
      all-families signed-prep migration can pay — large all-or-nothing.
- [ ] micro-opt sweep (clear/min-max/struct-layout) — Opus subagent IN FLIGHT.
GO/NO-GO: M1 is thinning (each target yields less) → M2 is the pivot.

M2 → ~2.6-2.8x (walk fusion, multi-week):
- [~] D4-a phase 1 LANDED a5f3c540 (ref-plane view, -0.71%/-1.79% cpu); the
      full per-SB fused predict→add-residual→recon walk (dav1d recon_tmpl.c
      dispatch shape — collapse per-block layer hops into one resident walk,
      the biggest remaining Go-vs-C structure delta) is still OPEN.
- [ ] D2-b guard-disjunct BCE on whatever M1/M2 profiles show hot.
GO/NO-GO: M2 is the pivot — if fused-walk prototypes measure <5%, the
plateau is real and 1.4x needs the M3 spine bet re-scoped before more work.

M3 → ~2.0-2.2x (spine expansion): block-loop orchestration into asm regions
(the decodeBlockLoopVisit spine around the landed kernels), eob/context
derivation folded into the TXB kernels (D3-d revisit WITH the loop),
remaining Horiz/Vert base variants.
M4 → 1.4-1.7x (endgame): extend until ~90% of hot cycles are kernel/asm;
re-plan with data at each 0.2x step. Honest note: M3/M4 estimates carry the
widest error bars — every earlier milestone tightens them.

### ENCODER ~3.3x → 1.4x
M1 → ~2.9-3.0x (mapped, weeks):
- [~] E3 phase 1 LANDED e85ceaec (invTX+residual-add, -1.22% realC ST). Phase-2 forward fusion STOPPED: residual→fdct→quant already NEON-backed, further fusion needs new transform kernels (cross-pkg project, deferred). residual→fdct→quant→dequant→invTX
      →recon (~20% cum; buffer round-trips + per-stage call tax).
- [x] E1-b banded CDEF u8 LANDED df734aab (-1.46% wavefront cpu; per-band immutable backups; byte-identical, 16+ clean -race runs).
- [x] E4-b MDS0 batching — PINNED NEGATIVE: setup-hoist byte-identical but e2e flat/neg (per-cand cost is the mandatory NEON convolve, not dispatch). Lever closed.
GO/NO-GO: E3 is the load-bearing item; if fusion <5%, jump to M2 asm early.

M2 → ~2.4-2.6x (writer spine): E4-a entropy-Writer asm (symbol write + CDF
update loops, mirrors decoder D3; also cuts the 6.5ms serial wavefront bound
→ raw fps); tokenize/pack loop glue.
M3 → ~2.0x: encode block-loop glue fusion (selectIntraModeN ~6%, prepare/
finish orchestration), intra-path batching; re-run spend ladders at each new
operating point (cheaper compute may re-open pinned quality trades).
M4 → 1.4-1.7x (endgame): same spine-expansion character as decoder M4.

### Cross-cutting
- [ ] Periodic: re-measure both scoreboards idle at every milestone edge;
      update STANDINGS line; PGO profile regeneration after big shifts.
- [ ] D1-g 12-bit LF transposes (only if 12-bit enters the corpus).
- [ ] E2-f open-loop ME + E2-g cyclic AQ: QUALITY levers, park unless a
      milestone needs spend budget refilled.

## Pins (do NOT retry without new economics — details in git/memory)
Decoder: entropy SIMD symbol search (tiny alphabets) · primitive-only asm
(e2e-flat; asm the LOOPS) · blanket BCE reslicing · Apple-Silicon cache pin
(3x: in-place-u16-CDEF, sbrow postfilter chain, 16-row tmv ring — branches
preserved) · dequant-into-sign-loop fusion (+1.1%) · CDEF pri+sec re-fusion
(already fused; split single-strength kernels are deliberate) · Wiener
row-fusion · multi-tile bitmask LF finish (−8.2% sparse) · ExtendBorders
removal (public API exposes padding; internal readers don't need it) ·
madvise-in-harness mirage · tmv projection NEON (every NEON candidate SLOWER: cheap multiplies dwarfed by strided access + per-entry validity branches — gather/batch machinery costs more; scalar already optimal). Encoder: depth-removal beyond 9/14 (no-win even
at SPEND gate; ME already cheap — sweep tax unpayable; only ~4x cheaper NEON
sweep could reopen) · largest-TX (−0.66dB; libaom TX-size election is
load-bearing) · leaf skip-TX arm (−0.29dB realB) · use_neighbouring_mode cut
(fires ~46 blk/60f → +0.13%) · pruned-subpel (unrecoverable, was −0.20dB at
HOLD) · eighth-pel · SMOOTH@8x8 · 32-tier exact-rate w/o chroma · controller
variants (PI/step/tier) · golden qindex refresh · SSE-dist MDS0 · T0-only
MDS0 · wall-clock feedback (banned) · EC(N)∥MD(N+1) overlap (CDF chain,
proven both codecs). Lessons: grep `var im [` nil-scratch zero-fills ·
Rosetta differentials must EXECUTE (never skip on cpu.Detected) · fast
differential before conformance gate · memprofilerate=1 · strip AI trailers
from codex commits · cwd discipline: use git -C, never bare cd in chains.

## Protocol per landing
Review diff vs upstream → run every gate yourself → interleaved A/B (idle
for the land decision) → land w/ hash + one-line verdict here, or revert +
pin. Two Claude agents + codex-for-S/M concurrent; worktrees; merge serial.
