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

## Open work (decoder)
- [ ] D3-e mode/MV symbol-run asm (IN FLIGHT agent) — decodeBlockPrediction
      ModeInto ~10% cum; MV component chain first; same spine conventions.
- [ ] compound-X 8-bit NEON (IN REVIEW codex) — ~1.76% flat pure-Go path.
- [ ] F-1 fastest-math audit: prototype dav1d PREP_BIAS int16 CONV_BUF in one
      compound family, measure vs libaom-offset uint16; migrate all if red→
      green ≥1%; else pin with numbers. Output parity via existing diffs.
- [ ] D3-f next spine runs (after D3-e measures) → then D4-a per-SB fused
      predict→residual→recon walk (dav1d recon shape) → D5 endgame re-plan.
- [ ] tmv projection NEON (compute, ~2.4% flat — distinct from the HELD ring).
- [ ] D2-b guard-disjunct BCE pass on remaining hot sites (after D3 settles).
- [ ] D1-g 12-bit LF vert transposes (only if 12-bit enters corpus).

## Open work (encoder)
- [ ] E1-b CDEF u8 for the banded/wavefront path (needs per-band backups or
      stays u16 with analysis).
- [ ] E1-e entropy-Writer micro → grows into E4-a Writer spine asm (mirrors
      D3; also shrinks the 6.5ms serial wavefront bound → raw fps).
- [ ] E3 TXB pipeline fusion: spec residual→fdct→quant→dequant→invTX→recon
      dataflow (~20% cum), fuse hottest pair, iterate.
- [ ] E4-b MDS0 candidate prediction batching (shared setup for ≤4 cands).
- [ ] E2-f open-loop ME on retained source refs (quality lever; SVT PA-ref
      shape) · E2-g cyclic-refresh AQ (rc_aq.c; screen overshoot + post-key).
- [ ] E5 endgame re-plan after E3/E4 measure.

## Pins (do NOT retry without new economics — details in git/memory)
Decoder: entropy SIMD symbol search (tiny alphabets) · primitive-only asm
(e2e-flat; asm the LOOPS) · blanket BCE reslicing · Apple-Silicon cache pin
(3x: in-place-u16-CDEF, sbrow postfilter chain, 16-row tmv ring — branches
preserved) · dequant-into-sign-loop fusion (+1.1%) · CDEF pri+sec re-fusion
(already fused; split single-strength kernels are deliberate) · Wiener
row-fusion · multi-tile bitmask LF finish (−8.2% sparse) · ExtendBorders
removal (public API exposes padding; internal readers don't need it) ·
madvise-in-harness mirage. Encoder: depth-removal beyond 9/14 (no-win even
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
