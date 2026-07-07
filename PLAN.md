# PLAN.md — the single perf campaign doc (agent-facing, checkbox-tracked)

Goal: decoder within 1.4x of dav1d and encoder within 1.4x of SVT p12,
single-thread. Each unchecked box below is scoped to ONE short agent run.
Update this file (check the box + one-line result w/ commit hash) in the
same commit that lands the work. This file supersedes PERF_PLAN.md,
ARCH_GAPS_PLAN.md, ENCODER_DEPTH_PLAN.md, PATH_TO_1_4X.md (deleted).

## 0. Non-negotiable rules (read before ANY task)

1. PORT, DON'T INVENT. Every algorithmic change is a source-shaped port
   from third_party/upstream/ (decoder: dav1d; encoder: svt-av1 first,
   libaom second). Cite upstream file/function in a code comment.
2. DECODER BYTE-EXACTNESS IS ABSOLUTE (the bar is byte-identity with
   dav1d; AV1 decode is normative so the strict-MD5 suite IS that gate):
   make dryrun-fast (8/8) AND make dryrun-extended (must be 226/226,
   0 FAIL) per commit. One byte = revert.
3. ENCODER GATE: full `go test -timeout 30m ./... -count=1` (includes
   round-trips + aomdec/dav1d crosschecks) + the four-clip matrix (§2).
4. ZERO-ALLOC steady state: scripts/check_allocs.sh green;
   BenchmarkVideoEncoderRealC1080p 0 allocs/op. Traps: per-frame goroutine
   closures; variadic ...any logging (boxes args even when disabled —
   guard the CALL SITE); func-pointer SIMD dispatch heap-escapes stack
   scratch — use static build-tag dispatch.
5. GIT: never `git add -A`; stage explicit paths; lowercase imperative
   messages with package prefix; NEVER mention AI/assistants; push only
   after all gates + idle-machine measurement.
6. Every commit lands a real fix (instrumentation may ride with one).
7. NEVER touch internal/av1/motion/warp.go.
8. A red lever gets FULLY reverted and pinned in §6 with its numbers.
9. Report cpu_total (core-seconds) alongside wall for every before/after.
10. Machine load corrupts wall numbers (~1.5x seen): develop under load
    with interleaved same-run A/B; land/push on idle numbers. Quality/
    bytes are deterministic and load-immune.
11. ENV VARS ARE SCAFFOLDING, NOT SURFACE (user directive 2026-07-06):
    kill-switch env vars exist for the landing review + one soak cycle,
    then get REMOVED (the byte-identity proof at landing is the safety,
    not a permanent switch). Stats/debug dumps go behind build tags
    (compile-away pattern like trace_off.go) or get deleted when their
    program concludes. Sweep-tuning overrides get converted to code
    constants once the sweep picks a winner. See task H-1.

## 1. Agent environment setup (worktrees miss gitignored assets)

    ln -s /Users/thesyncim/GolandProjects/goav1/third_party/upstream third_party/upstream
    # decoder tasks also need:
    cp -r /Users/thesyncim/GolandProjects/goav1/testdata/benchcorpus testdata/ 2>/dev/null || \
      ln -s /Users/thesyncim/GolandProjects/goav1/testdata/benchcorpus testdata/benchcorpus
    # if TestAppendWebRTCScalabilityModesMatchesPinnedLibWebRTC fails:
    ln -s /Users/thesyncim/GolandProjects/goav1/internal/av1/testdata/libaom internal/av1/testdata/libaom

Encoder corpus: /tmp/corpus/{realA,realB,realC,screen}.yuv (1080p I420;
realA/B/C 60f 30fps, screen 120f 60fps). If missing, rebuild per §2 and
RE-BASELINE everything (content differs per rebuild).

    ffmpeg -ss {300|1800|3600} -i ~/Downloads/first.mp4 -frames:v 60 -pix_fmt yuv420p -f rawvideo /tmp/corpus/real{A|B|C}.yuv
    ffmpeg -ss 5 -i ~/Downloads/Grab*11.11*.mov -frames:v 120 -vf "scale=1920:1080,fps=60" -pix_fmt yuv420p -f rawvideo /tmp/corpus/screen.yuv

## 2. Measurement protocol (exact commands)

Decoder gap (idle, ~45s; the dav1d vs_goav1 ratio is THE number):

    GOAV1_BENCH_CORPUS_ALLOW_UNMANIFESTED=1 GOAV1_BENCH_CORPUS=1 \
      go test -tags goav1_oracle -run TestCrossDecoderCorpus \
      ./internal/av1/testvector -v -count=1 -timeout 1800s

Decoder profile (single worker): create throwaway hook
internal/av1/testvector/scratch_profile_test.go (build tag goav1_oracle,
reads GOAV1_PROFILE_CLIP, 30x decodeCorpusClipDiscard loop; DELETE before
committing), then:

    GOAV1_PROFILE_CLIP=$PWD/testdata/benchcorpus/p720_inter_q32.ivf \
      go test -tags goav1_oracle -run '^TestScratchProfileDecode$' \
      ./internal/av1/testvector -count=1 -cpuprofile /tmp/dec.out

Encoder quality matrix (deterministic; one line per clip):

    go run ./cmd/qualitybench -input /tmp/corpus/realC.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/realA.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/realB.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/screen.yuv -fps 60 -frames 120 -bitrates 1330000 -encoders goav1 -layers 2

Encoder single-thread cpu (add for ST tasks):
    -goav1-max-threads 1 -gomaxprocs 1  (and -encoders goav1,svt-av1
    -svt-preset 12 -svt-lp 1 -runs 3 for SVT compares — NEVER compare
    against a stored SVT number; same-run only, variance >25%).

HOLD-QUALITY acceptance (default): realC >= -0.05 dB rate-adjusted;
realA/B within -0.05; screen within -0.10 at not-worse rate; realC
default wall >= 101 fps idle; zero-alloc; round-trip green.
SVT-MATCHED acceptance (M-E2 spend tasks ONLY): per-clip PSNR@rate >=
SVT's own same-run numbers; everything else identical.

Gate checklist per commit, in order:

    go build ./... && GOARCH=amd64 go build ./... && go build -tags purego ./...
    go test ./internal/av1/<touched-pkg>/ -count=1   (+ -race, + -tags purego)
    go test -timeout 30m ./... -count=1
    make dryrun-fast && make dryrun-extended          # decoder-touching
    scripts/check_allocs.sh

## 3. State of the art (2026-07-06 evening, HEAD f94406b9)

| gap | 2026-07-02 | 2026-07-06 AM | NOW |
|---|---|---|---|
| decoder vs dav1d (corpus ST) | 4.34x | 3.91x | **3.52x** |
| encoder vs SVT p12 (realC ST cpu) | 5.2x | 4.2x | **~3.3x** (E2-c same-run 1.717 vs 0.516) |

Encoder quality vs SVT: goav1 +0.4-0.5 dB on every clip (this surplus is
the M-E2 spend budget). Decoder correctness: strict 226/226 + corpus
MD5 vs dav1d, continuously.

Landed 2026-07-06 (18 commits, all byte-exact/quality-gated):
4ab59fed encoder nil-scratch convolve (-11% ST) · 77addf06 HBD decode
completion (-20% 10-bit) · a6aae704 race-guard skip · 37538dac vertical
wd16 NEON transpose · 52367902 HBD compound emu-edge · 015fe524 LR
filtered-only round-trip · 99114ec1 LR u8 kernels · 783c1d10 LR pixel-
domain wiring · ddddce3e LR in-place stripes (SNAPSHOT GONE, 1.4MB→30KB/
frame) · 7c4bfac4 16-bit LF vert transposes · fdb6183c depth P1 (sweep +
disallow_below_16x16) · 116d8450 depth P2 (32/64 arms) · b2890ed7 CDEF u8
kernels · b25108a5 CDEF in-place (share 15.9→7.7%, -4.8% decode) ·
78ade855 TXB histograms · 25a2f876 LPD1 TX detectors (-2.3% realC ST) ·
f94406b9 docs.

Architecture parity status: restoration = dav1d shape DONE; CDEF = dav1d
shape DONE (8-bit); LF = bitmask apply DONE (single-tile); entropy = dav1d
msac shape DONE except window width (D1 on branch); depth signals = SVT
shape DONE; MD staging/coeff-decode shape/encoder CDEF-LR = verified
parity (audit 2026-07-06). Remaining gap = mapped tasks below + the
codegen residue (Go glue vs C+asm in serial paths) that M-D3/M-E3+ attack.

## 4. DECODER task board (milestones → ~2.8x → ~2.2x → 1.4x)

### M-D1: in-flight / small (target ~3.2x)
- [x] **D1-a 64-bit entropy window** — LANDED (rebased ba1b1996 +
  48e7ea53): idle A/B p720 dead-flat, p288_intra green all pairs
  (~0.5%); kernels −1.4..−3.3%, readCDF4 flat share halved. Honest e2e:
  flat-to-+0.5% (entropy share is only ~4-8% now). Kept because it is
  byte-exact, never slower, closes the last structural entropy
  divergence, and M-D3's TXB asm kernel builds on the 64-bit window.
  tellOffs pin: readerInitTellOffs stays −14 (width-invariant tell;
  mechanical rescaling would shift every BitsRead by −32). (done)
- [x] **D1-b sbrow-chained postfilter (D-2)** — BUILT, byte-exact
  (226/226 + new chained-vs-stage-major differentials), then MEASURED
  RED and HELD per rule 8: 720p q32 ~flat, q20 +0.4%, and a dedicated
  1080p A/B (aomenc stream, same-binary kill-switch) chain LOST all 3
  pairs, +1.0% cpu median. PIN: whole-frame stage-major postfilter is
  FINE on Apple Silicon — 1.4-3.1MB frames live in L2/SLC, dav1d's
  per-sbrow chaining pays only on smaller-cache hardware; the band-loop
  overhead exceeds the locality win here. Branch preserved with all
  gates green: worktree-agent-a47c0fa663e289d8c (P0 step APIs c1f1c649
  + P1 driver 42bcdb3c). This also WEAKENS the D4-b recon-interleave
  premise (same mechanism) — demand a measured prototype before
  investing there.
- [x] **D1-c temporal-MV 16-row window** — BUILT byte-exact (226/226,
  race-clean, single-worker gated) then MEASURED RED-TO-FLAT and HELD
  per rule 8 (branch codex-d1c): p720 cpu +1.8%, p360 +1.4%, projection
  share UP — per-SB-row re-projection overhead exceeds the 780KB→19KB
  footprint win. THIRD instance of the family pin: dav1d cache-locality
  restructures (in-place-with-u16, sbrow postfilter chain, tmv ring) do
  NOT pay on Apple Silicon's cache hierarchy — demand a measured
  prototype before any further one. (done)
- [x] **D1-d LR arena + construction shrink** — LANDED a08e2e52
  (codex, reviewed): walk scratch sizing extracted to one shared
  function; 8-bit non-optimized LR Data = band+backups only, Dst = 0;
  superres row-borrow sized explicitly via max(). 1080p LR arena
  14MB→24KB; construction 5.24→2.81 ms/op. CDEF staging deliberately
  NOT shrunk (encoder banded workers need the immutable snapshot —
  revisit only with E1-b). (done)
- [x] **D1-e memmove hunt** — CLOSED AS DOCUMENTED NEGATIVE (codex):
  full attribution table shows memmove already fell to ~1.0% (was 2.9%
  before today's landings) + memclr ~2.3%. Remaining callers: Frame.
  ExtendBorders (blocked by D2-d public-API verdict), LR band assembly
  (dav1d has equivalent staging), CopyPlaneBlock/intra fillBlock
  (fundamental pixel copies), harness large-alloc zeroing (not decoder),
  transform scratch clears (M-D3 spine territory). Nothing bounded left
  to kill. (done)
- [x] **D1-f CDEF native u8-store AVX2 epilogue** — LANDED d4aba79e
  (codex, reviewed): dedicated u8 AVX2 kernel .s with VPACKUSDW/
  VPACKUSWB byte stores replacing the stack-tmp+Go-narrow wrapper;
  differentials execute directly under Rosetta (PASS not SKIP);
  parity-only on this host — validate perf on native x86. (done)
- [ ] **D1-g 12-bit vertical LF transposes** — 12-bit still routes
  pure-Go in filter14Vert16/filter6/8Vert16 (10-bit done 7c4bfac4).
  Only if a 12-bit clip enters the corpus; else skip. (S, LOW PRIO)

### M-D2: toolchain + glue shape (target ~2.8x)
- [x] **D2-a Go PGO** — LANDED (codex, reviewed): merged corpus profile
  at cmd/qualitybench/default.pgo (54KB; main-package-scoped — module
  root does NOT activate, verified via go version -m), GOAV1_PGO_FLAGS
  wired into every test/bench/dryrun target, pgo-regenerate recipe in
  Makefile. Decoder p288 −1.5% cpu (4/4 pairs), p720 flat; encoder
  realC ST −0.84%. Byte-identical everywhere, extended 226/226 with
  PGO active. NOTE: campaign measurements through Makefile targets now
  include PGO; raw `go test` invocations need -pgo=<path> to match.
  (done)
- [ ] **D2-b targeted BCE pass** — top remaining bounds-check sites via
  `-gcflags=all=-d=ssa/check_bce` on hot files; ONLY guard-disjunct
  pattern (the additive-induction trick from the coeff-context commit);
  blanket reslicing is PINNED RED (§6). (M)
- [x] **D2-c activeRows prefix-max table** — LANDED 14df1c81 (codex,
  reviewed): dav1d scan.c init_tbl port, table lookup replaces the
  O(eob) %-loop; custom scans keep exact fallback. The dequant-fusion
  half was built and REVERTED ON REVIEW: e2e +1.1% cpu (nil-check
  branches in every sign-loop iteration ~+5% even unfused, per-coeff
  division, per-TXB clear) — PINNED in §6. (done)
- [x] **D2-d ExtendBorders removal audit** — CLOSED AS DOCUMENTED
  NEGATIVE (codex, 25-surface audit): every INTERNAL reader is
  border-independent (motion resident checks + clamped/emu-edge paths
  key on visible Width/Height, not padding; CDEF/LR/superres/film-grain
  build their own halos; encoder never reads decoder padding; the old
  "future inter reads need it" comments are stale). SOLE BLOCKER:
  public API exposes raw Plane.Pix backing slices and decoded frames
  currently publish edge-replicated padding — removal changes
  externally-observable bytes. If padding is ever declared undefined
  public state, the change is removing postfilter.go:778-780 + stale
  comments (~0.3-0.5%). USER DECISION required to reopen. (done)

### M-D3: asm-spine phase 1 (target ~2.2x) — the codegen-residue attack
Rationale: serial symbol/coeff/mode decode + orchestration = 30-35% of
decode CPU; per-symbol Go call overhead + codegen vs dav1d's msac.S +
clang-O3 C is the残 gap. The pinned "entropy NEON dead-end" does NOT
apply (that was SIMD inverse-CDF search; this is scalar asm).
- [x] **D3-a TXB-kernel spec** — LANDED: spec doc
  internal/av1/tile/coeff_asm_arm64_spec.go (register/memory plan;
  recommended cut = per-loop kernel for the base-levels walk with the
  BR chain inlined and the DC iteration folded in, one asm call per
  TXB; whole-TXB and per-symbol cuts rejected with reasoning) +
  differential harness coeff_asm_diff_test.go (three pure-Go TXB
  bodies in lockstep over randomized+adversarial streams/CDF states;
  full decode record compared incl. entropy.Reader.State() snapshot;
  GOAV1_TXB_DIFF_TXBS soak knob). (done)
- [x] **D3-b base-levels loop asm** — LANDED: coeffBaseLevels2DARM64
  (entropy/reader_coeff_base_arm64.s), one NOSPLIT leaf call per TXB
  covering the whole Class2D base-levels walk incl. the DC iteration
  and the BR high-token chain inlined; wired into tracked2D + WithGeo
  scanHot paths behind static bool dispatch with
  GOAV1_DISABLE_COEFF_ASM kill switch (switch byte-identity proven in
  the lockstep harness). Differential: ~5M randomized+adversarial TXBs
  green; dryrun-extended 226/226 0 FAIL with kernel active. Micro:
  tracked TXB decode -10..-12% (8x8/16x16/32x32, same-binary A/B).
  e2e idle same-run pairs (30x decode, cpu user): p720_inter_q32
  7.13s vs 7.22s (-1.2%), p288_inter_q20 3.10s vs 3.15s (-1.5%),
  kernel wins all 6 pairs. Boundary overhead is NOT eating the win;
  absolute e2e is bounded by this loop's share — D3-c next. (done)
- [x] **D3-c golomb/sign replay asm** (BR chain already landed inside
  D3-b): coeffSignGolombARM64 (entropy/reader_coeff_sign_arm64.s), one
  NOSPLIT leaf call per TXB replaying the whole P3 dirty-list pass —
  DC-sign binary CDF read + updateCDF2, one equiprobable sign bit per
  other nonzero, Exp-Golomb tails incl. the maxLength unwind (culLevel
  -1 error protocol, cursor at the exact failure point). KEPT SEPARATE
  from D3-b per the spec (fuse decision deferred to D3-d); the
  standalone cut wires into BOTH replay sites (tracked2D + WithGeo, all
  classes and update modes incl. Horiz/Vert that D3-b never sees). Same
  kill switch: GOAV1_DISABLE_COEFF_ASM=1 forces pure-Go for BOTH
  kernels; value "sign" disables only D3-c (incremental A/B hook,
  scaffolding per rule 11). Differential: golomb-focused lockstep test
  added (tail-heavy base/BR CDFs force a golomb tail per coefficient;
  0xff/0x00/random/short-tail streams pin length classes 1, >20-fail
  and past-end ecLotsBits; dirty lists spread 1..maxEOB); soak
  GOAV1_TXB_DIFF_TXBS=1000 = 114k streams x <=48 TXBs x 2 tests (~5M+
  TXBs, 5-variant lockstep) green; extended 226/226 0 FAIL kernel-on.
  Micro (same-binary "sign" switch A/B, n=8, ±1%): tracked TXB -1.6%
  (8x8) / -6.5% (16x16) / flat (32x32, its bench payload replays a
  near-empty dirty list). e2e same-binary interleaved pairs (30x
  decode, cpu user): p288_inter_q20 3.07 vs 3.10 (-1.0%, 7/8 pairs),
  p720_inter_q32 flat (7.05 vs 7.02 on the clean pairs; mixed under
  load) — the replay's share scales with coefficient density (one sign
  bit per nonzero), q32 has too few nonzeros for the boundary win to
  clear noise. (done)
- [ ] **D3-d integrate full-TXB kernel + measure** — e2e target: coeff
  subtree cum halves. Go/no-go for D3-e. (M)
- [ ] **D3-e mode/MV symbol subtrees** — same method, hottest first
  (decodeBlockPredictionModeInto ~10% cum). (L, phased)

### M-D4: recon orchestration fusion (target ~1.8x)
- [ ] **D4-a per-SB fused predict→add-residual→recon walk** (dav1d
  recon_tmpl.c shape; fewer per-block dispatch layers). (L, phased)
- [ ] **D4-b recon∥filter interleave** — D-2 P3: drive the chained
  postfilter per sbrow as recon rows complete (wavefront row-progress
  hooks exist: threading/tile_residual.go waitForRowProgress). (L)
- [ ] **D4-c prediction dispatch flattening** — branch trees → tables. (M)

### M-D5: endgame (1.4-1.6x) — define after M-D3/D4 measure
- [ ] D5-a re-plan with data: extend asm coverage to the block-loop spine
  vs accept plateau. Decision doc, not code. (S)

## 5. ENCODER task board (milestones → ~2.9x → ~2.4x → 1.4x)

### M-E1: mapped work (target ~2.9x)
- [x] **E1-a encoder CDEF u8 routing (serial)** — LANDED a5bed2cc
  (codex, reviewed): serial appliers route through the decoder's u8
  in-place walk via bindApplyContext(loadSnapshot); banded/wavefront +
  10/12-bit keep the u16 snapshot. SHA-256 byte-identity on encoded AND
  decoded output, all four clips; serial apply zero-alloc regression
  test; realC ST cpu −2..3% all pairs. (done)
- [ ] **E1-b CDEF u8 on the banded/wavefront path** — the banded apply
  (ApplyCDEFPostFilterUnitRows) needs an immutable snapshot today;
  design per-band backups or keep u16 there with analysis. (M)
- [ ] **E1-c LF mask apply on the wavefront path** — SetMaxThreads>1 is
  single-tile but still uses the parallel banded edge-list; route to a
  parallel mask apply (serial mask apply landed 8e7a93cb, -11.8% ST).
  (M)
- [x] **E1-d CDF-init cut** — LANDED 259736b1 (codex, reviewed):
  non-coefficient default CDF families now assign package-init images
  (entropy.CDF is a pure value struct — deep copy). BUT the ~6%
  attribution was STALE: coefficient defaults were already images, and
  the frozen rate tables depend on adapted frame-start CDFs (NOT
  cacheable by qIndex without changing decisions — profiled attribution
  in the task record). realC ST neutral; beneficiaries are keyframe
  multi-tile setup/trial-arm/prewarm paths. Byte-identical all four
  clips. (done)
- [ ] **E1-e entropy-write micro** — Writer hot loop vs od_ec_enc shape;
  small unless combined with M-E4. (S)
- [ ] **E1-f PGO** — shared with D2-a. (—)

### M-E2: quality-spend reframe (target ~2.4x) — SVT-MATCHED gate
We hold +0.4-0.5 dB over SVT; 1.4x is at MATCHED quality. Ladders that
were red at hold-quality get re-run against the SVT-matched gate (§2).
- [x] **E2-a spend harness** — LANDED e6d403a7: scripts/spend_gate.sh
  runs goav1+SVT same-run (§2 protocol, both ST) over the four clips and
  emits the verdict table (dPSNR at measured rates, rate delta flagged
  past +2%, cpu_total/wall both). Baseline at then-defaults: realA +1.98,
  realB +2.18, realC +0.52, screen +12.12 dB (rates +1.7/+3.8/+4.3/+7.4%
  vs SVT; cpu 1.29/1.29/1.75/1.19 vs 0.41/0.41/0.52/0.67). (done)
- [x] **E2-b depth-arms ladder re-run** — MEASURED, NO WIN (2026-07-06,
  no code change; sweep ran via throwaway env override, removed after,
  override-at-9,14 proven byte-identical first). Rungs 11,14 / 11,15 /
  13,15 / 15,15 all GREEN on the matched gate but realA+screen bytes are
  IDENTICAL across rungs and realB/C move <0.02 dB: the arms fire on
  ~235 swept SBs/60f (below16 216→231, below64 1→12 across the whole
  ladder). Max-removal rung 15,15 interleaved realC ST cpu = +0.5%
  (medians 1.749 vs 1.741) — nothing to harvest; §6 pin (removals only
  save leaf work that exists) confirmed under the MATCHED gate too. 9/14
  stays; SVT's own rtc 1080p ladder tops out at 9/14 anyway
  (enc_mode_config.c:9287-9295). (done)
- [x] **E2-c mds0 candidate/dist-band cuts ladder** — LANDED (see
  commit): mds0NearCount 3→1 = SVT parity fix + spend win. Preset-12 rtc
  operates at cand_reduction_level 5 (any lpd1 level > LPD1_LVL_3,
  enc_mode_config.c:7251-7259) whose near_count=1; the old 3 mis-cited
  the level<=2 rows. Interleaved realC ST cpu −2.2% (1.717 vs 1.756, B<A
  in 5/5 pairs) at realC −0.035 / realA −0.106 dB of surplus, all clips
  green same-run. Ladder rest: near_count 0 (level-6 stats-pass shape)
  = ~−2% more cpu but realC margin collapses to +0.14 dB at +3.5% rate —
  rejected; leaf band-gate 4→3→2 quality-flat AND cpu-flat (band3 stacked
  on near1: −1.8% vs −2.2% alone, noise) — band stays 4. (done)
- [x] **E2-d leaf subpel schedule ladder** — SKIPPED: the reverted
  SUBPEL_TREE_PRUNED port is not recoverable (searched all refs, stashes,
  156 dangling commits, sibling worktrees; only the pin record 17f4fdf8
  survives). Re-testing at MATCHED gate requires reimplementing the port
  — out of this slice per its own scope rule. If re-attempted: worth it
  only bundled with the E2-c surplus math, realC margin is now +0.49 dB.
  (skipped)
- [ ] **E2-e golden/probe breadth ladder** (S)
- [ ] **E2-f open-loop ME (audit E-B)** — retention plumbing (lastSrcY/
  goldenSrcY + quarter planes, pipelining double-buffer discipline), then
  switch signal producers one at a time. QUALITY lever (buys spend
  budget); the depth-sweep motivation is dead (§6) but static-bypass
  grids/int-pro/leaf fallbacks remain candidates. (L, phased)
- [ ] **E2-g cyclic-refresh AQ (audit E-C)** — SVT rc_aq.c:599-660
  segment delta-q, ~20-25% SBs/base frame; attacks screen +17% overshoot
  + post-key convergence. Segmentation syntax exists unused. (L, phased)

### M-E3: TXB pipeline fusion (target ~2.0x)
- [ ] **E3-a fusion spec** — prepare+finishInterTXB (~20% cum): map the
  residual→fdct→quant→dequant→invTX→recon dataflow, intermediate buffer
  round-trips, and which stage pairs can fuse (asm or Go). Spec first. (M)
- [ ] **E3-b implement hottest fusion pair** + measure. (L)
- [ ] **E3-c remaining pairs** per E3-a ranking. (L)

### M-E4: writer asm + orchestration (target ~1.7x)
- [ ] **E4-a entropy-Writer spine asm** (symbol write + CDF update;
  mirrors M-D3; also shrinks the serial wavefront bound → raw fps). (L)
- [ ] **E4-b MDS0 candidate prediction batching** — shared halo/setup
  for the up-to-6 candidate predictions. (M)

### M-E5: endgame — decision doc after M-E3/E4 measure. (S)

### H-1: env-var scaffolding sweep (after in-flight agents land)
- [x] **H-1 remove/build-tag experiment env vars** — LANDED 5d72ef52
  (codex, reviewed): 3 soaked switches removed, stats behind
  goav1_encstats tag, debug behind per-package tags (refmv boxing guard
  preserved), wavefront knob → for-test hook, DEFER_RECON frozen. ONE
  runtime env var remains: GOAV1_DISABLE_COEFF_ASM (asm-spine soak).
  Bytes identical all four clips; extended 226/226. — — production code
  currently reads: GOAV1_LPD1_TX, GOAV1_DEPTH_REMOVAL_DISABLE,
  GOAV1_DEPTH_REMOVAL_P2_DISABLE, GOAV1_DEPTH_REMOVAL_STATS,
  GOAV1_TX_WRITE_STATS (2026-07-06 experiment switches — remove after
  soak; stats to build tags), GOAV1_DEBUG_CDEF_UNIT, GOAV1_DEBUG_WIENER,
  GOAV1_DEBUG_REFMV, GOAV1_COEFF_TRACE (older debug — build-tag or
  delete; check the debugRefMV boxing pin before touching),
  GOAV1_WAVEFRONT_MIN_SBROWS_PER_WORKER, GOAV1_DEFER_RECON (tuning
  knobs — decide: constant or documented public config). WAIT for the
  M-E2 agent to land (it owns pframe_depth_removal.go and must itself
  convert its sweep override to constants per rule 11). (S, codex)

## 6. PINS — measured dead-ends. Do NOT re-attempt without new economics.

DECODER: entropy symbol-decode NEON/SIMD search (tiny alphabets; scalar
wins — scalar ASM is fine, that's M-D3) · blanket BCE reslicing (+38%
regressions; guard-disjunct only) · in-place CDEF with u16 kernels
(+3.15%; RESOLVED via u8 kernels b25108a5) · CDEF pri+sec re-fusion
(already fused; forcing fused on single-strength inputs -27..54%) ·
Wiener row-fusion (loop-order only, L1-resident intermediate, big asm
risk) · multi-tile bitmask LF finish (-8.2% on sparse multi-tile; branch
wip/lf-multitile holds the byte-exact port) · madvise in corpus-profile
harness is construction churn, not decode (streaming decode is 0-alloc).

ENCODER: depth-removal as a CPU lever CLOSED (P3 grid-reuse red in every
config: mesh budget ~3.5% of cpu vs sweep tax 7-10%; SVT sweep economics
don't transfer — our ME was already cheap; only a ~4x cheaper NEON sweep
kernel could reopen; RE-CONFIRMED under the SVT-MATCHED gate, E2-b
2026-07-06: level rungs up to 15,15 quality-green but +0.5% cpu — the
arms fire on ~235 SBs/60f, nothing to remove) · mds0 near_count 0
(stats-pass shape: −0.29..−0.38 dB, realC margin +0.14 dB at +3.5% rate)
· mds0 leaf band-gate 3/2 (quality-flat, cpu-flat) · largest-TX P-frames (-0.66 dB realA/B; libaom's
variance TX-size election is load-bearing; SVT tx_depth-0 needs its
SATD/RDOQ pipeline; diff preserved in scratchpad p2_largest_tx.diff) ·
leaf skip-TX arm (realB -0.29 dB; skip cascades through neighbour
feedback without SVT's per-SB detector stats) · wide sweep trigger at
9/14 (-0.28 dB) · pruned-subpel level 6/8 at HOLD-quality (-0.20;
E2-d re-tests at MATCHED) · eighth-pel MVs · SMOOTH family at 8x8 ·
32-tier exact-rate merges w/o chroma+CDF calibration · frame-wide
SHARP/SMOOTH · PI/step-clamp/tier-split controller variants · golden
qindex refresh rules · SSE distortion in MDS0 (-0.08) · T0-only MDS0
scope (screen +4.4% rate) · wall-clock feedback signals (banned) ·
entropy-write∥next-frame-decision overlap (PROVEN serialization both
goav1 and SVT: MD(N+1) needs N's frame-end CDFs) · compoundGoldenLikely
path (never arms on realC).

use_neighbouring_mode MDS0 cut (2026-07-07, codex): quality-green at the
matched gate but fires on only ~46 realC blocks/60f (0 on other clips) →
+0.13% cpu, below the ≥1% bar — implemented, measured, reverted; do not
re-attempt without content where intra-bordered inter blocks are common.
NEW PINS (2026-07-06 evening, codex/agent round): dequant-into-sign-loop
fusion (e2e +1.1% cpu; hot-loop nil branches tax the unfused path +5%;
per-coeff division; per-TXB clear — dav1d's fusion is unconditional C) ·
sbrow-chained postfilter on Apple Silicon (720p q20 +0.4%, 1080p +1.0%
cpu — frames live in L2/SLC, band-loop overhead beats locality; branch
worktree-agent-a47c0fa663e289d8c preserved, gates green) · E1-d's "~6%
CDF-init" was stale attribution (coeff defaults already images; rate
tables depend on adapted frame-start CDFs, not qIndex-cacheable).

LESSONS THAT KEEP PAYING: grep `var im [` for nil-scratch zero-fill
wrappers (found 3 today) · force amd64 differentials to EXECUTE under
Rosetta (cpu.Detected.AVX2=false there; gate-skips hid 2 real bugs) ·
build the FAST pixel/stream differential BEFORE the conformance gate ·
memprofilerate=1 for alloc attribution · agents killed by session limits
resume cleanly from their worktrees via SendMessage.

## 7. Reporting

After each landed task: check the box here with commit hash + one-line
result, update §3 standings if a scoreboard moved, append pins to §6,
push. Session memory mirrors: simd-plan (decoder), encoder-perf-target
(encoder), perf-deadends (pins).
