# goav1 performance campaign — handover plan

Standing mandate: **beat dav1d on the decoder** (single-thread wall clock on the
internal corpus) and **beat SVT-AV1 on the encoder** (wall-clock fps at equal or
better quality, matched rate). This document is the execution plan. Follow it
mechanically; do not improvise.

## 0. Non-negotiable rules

1. **Port, don't invent.** Every algorithmic change is a source-shaped port from
   the pinned upstreams in `third_party/upstream/` (decoder: `dav1d`; encoder:
   `svt-av1` preferred, `libaom` second). Cite the upstream file/function in a
   code comment. If you cannot ground a lever in upstream source, skip it and
   record why. Hand-tuned thresholds and invented gates have repeatedly measured
   red in this repo.
2. **Byte-exactness is absolute (decoder).** Any decoder change must pass
   `make dryrun-fast` AND `make dryrun-extended` (strict-MD5 conformance, runs
   the NEON paths on this arm64 host). One byte of divergence = revert.
3. **Round-trip is absolute (encoder).** `go test -timeout 30m ./... -count=1`
   includes decoder round-trips and aomdec/dav1d crosschecks. New syntax must be
   exercised by those tests, not just unit tests.
4. **Zero-alloc steady state.** `scripts/check_allocs.sh` must stay green;
   `BenchmarkVideoEncoderRealC1080p` must report 0 allocs/op. Known traps:
   per-frame goroutine closures; variadic `...any` logging in hot loops (boxes
   args even when disabled — guard the call site); **func-pointer SIMD dispatch
   can make stack scratch escape to the heap — use the static build-tag dispatch
   pattern** (see `internal/av1/cdef/filter_unit_generic.go` +
   `filter_dispatch_arm64.go`).
5. **Git hygiene.** Never `git add -A` (repo has untracked scratch and
   worktrees). Stage explicit paths. One commit per landed lever. Message style:
   lowercase imperative with package prefix ("cdef: ...", "encoder: ...").
   Never mention Claude/AI/assistants anywhere. Push only after ALL gates pass
   and a clean (idle-machine) measurement confirms the win.
6. **Every commit lands a real fix** — no docs-only or refactor-only commits.
7. **Do not touch** `internal/av1/motion/warp.go`.
8. A lever that measures red gets **fully reverted** and its numbers recorded in
   the report/memory. Reverted work is not wasted — it is a pin against retries.

## 1. Current standings (measured 2026-07-06 midday, HEAD 52367902)

Decoder, single-thread, internal corpus (18 clips, 48 frames each):

| decoder | total ms | vs goav1 |
|---|---|---|
| goav1 | 2243 | 1.00x |
| aomdec | 1001 | 2.24x faster |
| dav1d | 596 | **3.77x faster** ← the gap |

(Trajectory: 5.12x → 4.34x (07-02) → 3.91x (07-06 morning, bitmask LF +
coverage sweep) → 3.77x (07-06 wave: HBD completion 77addf06, vertical wd16
transpose 37538dac, HBD compound emu-edge 52367902) → **3.64x** (07-06
afternoon, restoration architecture P1-P3: 015fe524 filtered-only
round-trip, 99114ec1 u8 kernel family, 783c1d10 8-bit pixel-domain wiring —
no plane widen, no dst scratch, no store-back). 10-bit clip went
0.16x → 0.22x of dav1d. Per-clip 0.19–0.29x, tightest at p720.)

Encoder single-thread, same-run realC@5M (-goav1-max-threads 1 -gomaxprocs 1
-svt-lp 1, runs 3): goav1 cpu_total 1.743s (29.0 ms/frame) vs SVT 0.486s
(8.1 ms/frame) = **3.59x** (was 5.2x on 07-02, 4.2x this morning pre-fix),
quality +0.53 dB @ +4.6% rate. Nil-scratch fix 4ab59fed = the −11%; decoder
postfilter wins flow into the encoder loop too.

>>> 2026-07-06 DIRECTIVE (user): beat dav1d (decoder) + SVT (encoder) with
focus on SINGLE-THREAD; cpu_total (core-seconds) is reported alongside wall
for every before/after — a cpu drop at flat wall is a win.
Encoder single-thread (rebuilt corpus, same-run, max-threads 1, gomaxprocs 1,
svt-lp 1, realC@5M): goav1 35.0 ms/frame cpu vs SVT 8.3 = 4.2x (was 5.2x),
quality +0.53 dB @ +4.6% rate. After the nil-scratch fix (4ab59fed):
cpu_total 2.099→1.863 s (−11.2%), 29.6 ms/frame → gap ~3.7x, output
byte-identical. 10-bit decode clip: −20.5% wall / −20.3% cpu (77addf06).

Encoder vs SVT-AV1 v4.0.1 preset 12 LD-CBR, matched rates, goav1 `-layers 2`.
Two goav1 columns: the tile-column DEFAULT (no SetMaxThreads) and the
WAVEFRONT (SetMaxThreads>1 → single-tile SB-row wavefront, E1 landed). The
wavefront is the recommended threaded config — it eliminates the per-tile CDF
quality tax (+0.4–0.6 dB) and is what WebRTC Config.MaxThreads selects:

| clip | goav1 default (tiles) | goav1 wavefront | SVT | verdict |
|---|---|---|---|---|
| realC 30fps@5M | 44.714 @ 103.5 fps | **45.17 @ 77 fps** | 44.767 @ 128 fps | wavefront +0.4 dB > SVT, realtime; SVT faster raw |
| realA 30fps@5M | 46.673 @ 140 fps | **47.27 @ 84 fps** | 45.321 @ 139 fps | +1.95 dB > SVT |
| realB 30fps@5M | 46.985 @ 140 fps | **47.54 @ 84 fps** | ≈realA | big quality win |
| screen 60fps@1.33M | 50.767 @ 238 fps | **51.28 @ 139 fps** | 39.121 @ 147 fps | +12 dB, still >60 fps |

goav1 (wavefront) is now quality-superior to SVT on EVERY clip while staying
above each clip's frame-rate deadline. SVT keeps the raw-fps crown on realC
(122 vs 80). RAW-FPS ANALYSIS (2026-07-03, proven): a realC frame ≈ 13.6 ms =
decision-wavefront ~7.4 ms (parallel, row-capped: 1080p has only 17 SB-rows) +
entropy-write ~7.3 ms (SERIAL). The write is HALF the frame and CANNOT overlap
frame N+1's decision — N+1 prices RD against N's frame-END adapted CDFs
(e.frameCtx, written only by exportCDFs AFTER N's write; the mds0/interp rate
tables freeze from it). Traced through goav1 AND SVT source: SVT p12 has the
IDENTICAL EC(N)→MD(N+1) serialization (frame_context chaining, always-on at
p12; pic_manager blocks the next referencing frame until packetization writes
back the adapted CDF). So entropy-write∥decision overlap is a PROVEN DEAD-END,
not a missing optimization — pin it. SVT's 122 fps at only 1.82x parallelism =
~4.5 ms CPU/frame vs goav1 ~50 ms over 4.7 cores: the gap is per-frame WORK
(pure-Go vs SIMD-C + heavier search), not unexploited overlap. WITHIN the
low-delay contract the byte-safe wall levers are: (1) speed the serial entropy
write (7.3 ms, hard — entropy is tight), (2) reduce serial LF tail (loop-filter
bitmask, in progress), (3) less per-frame work (E-levers ~exhausted). The structural raw-fps lever is TEMPORAL-LAYER FRAME PIPELINING (L1T2 coding
order base0,leaf1,base2,leaf3…: leaf1 and base2 both depend only on base0 and
are mutually independent → encode them concurrently; bytes IDENTICAL) at the
cost of +1 FRAME LATENCY.
>>> LANDED opt-in (d215040f + 4a843b3d, user: "do whatever SVT does for
realtime webrtc, make it optional"): SetThroughputPipelining + EncodeThroughput
/Drain (default OFF = byte+latency identical to today). When ON, the buffered
L1T2 leaf runs CONCURRENTLY with the following base on a persistent parked
worker (own leafPC/leafHME/leafLF/leafWavefront), respecting the filter-join,
HME-pyramid-handoff, private-CDF-snapshot, recon-double-buffer couplings.
Byte-IDENTICAL to serial (TestVideoEncoderPipelineByteIdentical green under
-race across all layer/RC/thread/keyframe configs). Overlap fires ONLY where
provably safe (golden OFF, L1T2, single-tile inter, 8-bit 420, no scene-cut,
no render pad); everything else falls back to the serial FIFO. Also fixed a
latent loopfilter/cdef applier teardown race (present on the normal path too).
THROUGHPUT (same-process serial vs pipelined, 1080p pan): mt=1 20.9→33.3fps
(1.60x), mt=2 42.5→59.1 (1.39x), mt=4 57.6→74.5 (1.29x), mt=14 73.6→85.2
(1.16x, ABOVE the ~83fps wavefront Amdahl ceiling). Default fps UNCHANGED.
Does NOT beat SVT's raw ~131fps at 14-core saturation, but a decisive win at
low core counts and a real structural gain everywhere. NEXT increments (tree
green): (1) golden-enabled overlap needs the base compute/commit split
(defer golden decision past leaf's avgFrameLowMotion update); (2) pipelined
WebRTC (lag EncodePicture descriptor/metadata by one to match +1 latency);
(3) public-API forwarding + qualitybench +1-latency pairing for a same-run
SVT comparison. This is opt-in throughput; the default stays low-delay.

CPU PROFILE (realC, 60 frames, cpu-seconds ≈ energy; commit 99785179 routes
serial + multi-tile coders back to the FUSED single-pass walk — the split's
record/replay only pays off with ≥2 wavefront lanes): serial 3.02s@1 core /
tiles=32 default 3.23s@6.1 cores / wf-lanes=2 3.45s@2.6 cores / wf-lanes=8
3.66s@4.1 cores. TWO facts for CPU-conscious deployment: (1) total CPU is
~flat across configs — threading spreads work, it doesn't create it; the
wavefront is NOT a CPU-usage regression; (2) QUALITY comes from single-tile,
NOT lane count (wf-lanes=1..8 all ≈45.2 dB), so lanes only buy wall speed —
dial them to the core budget. Sweet spot for CPU-conscious realtime =
wf-lanes=2 (45 fps ≫ 30, only 2.6 cores, full +0.55 dB, FEWER cores than the
6.1-core tiles=32 default). The real absolute-CPU gap vs SVT (~55 ms/frame vs
~10) is work VOLUME (~5×), independent of threading — that is E8, not E1.

Speed numbers above are MEDIAN-OF-3 SAME-RUN side-by-sides (qualitybench
`-encoders goav1,svt-av1 -runs 3`, idle). RULE (learned the hard way twice):
never compare against a stored SVT fps — SVT run-to-run variance exceeds 25%
on this host; every speed claim needs both encoders in the SAME invocation
with `-runs 3`. An earlier "goav1 faster on realC" claim was a stale-baseline
artifact. CPU note: goav1 burns ~3.6x SVT's CPU on realC (3.17s vs 0.87s) at
~5.4x observed parallelism — the realC speed fix is likely CPU efficiency
(E7), not more threading.

**Definition of done:** decoder — goav1 ≥ dav1d single-thread on the corpus
aggregate (milestones: 4.0x → 3.0x → 2.0x → 1.0x). Encoder — realC PSNR ≥ SVT
at matched rate with wall fps ≥ SVT on all four clips. realC needs the last
~0.05 dB AND ~24% wall speed — the speed gap is now the bigger half.

## 2. Before you start: environment

- Host is Apple Silicon (darwin/arm64). NEON runs natively; amd64 differential
  tests run under Rosetta (`GOARCH=amd64 go test` — note CPUID reports AVX2
  false under Rosetta, so differential tests must CALL kernels directly, never
  skip on `cpu.Detected`).
- **Agent worktrees miss gitignored assets.** Symlink/copy from the main repo
  before anything else:
  `ln -s /Users/thesyncim/GolandProjects/goav1/third_party/upstream third_party/upstream`;
  copy `testdata/benchcorpus/`; symlink `internal/av1/testdata/libaom` if a test
  wants it. A failing `TestAppendWebRTCScalabilityModesMatchesPinnedLibWebRTC`
  without the symlink is environmental.
- Encoder corpus lives at `/tmp/corpus/{realA,realB,realC,screen}.yuv`
  (1080p I420; realA/B/C 60 frames 30fps, screen 120 frames 60fps). If missing,
  rebuild: realA/B/C = `ffmpeg -ss {300|1800|3600} -i ~/Downloads/first.mp4
  -frames:v 60 -pix_fmt yuv420p -f rawvideo /tmp/corpus/realX.yuv`; screen =
  first 120 frames of the 3024x1964 screen recording scaled to 1920x1080 (glob
  `~/Downloads/Grab*11.11*.mov`; the filename has unicode spaces — use a glob).
  **If you rebuild, re-measure ALL baselines first** — numbers above assume
  these exact files.
- **Two concurrent agents max**, non-overlapping packages, worktree isolation.
- Machine load corrupts wall-clock numbers (~1.5x inflation seen). Quality
  (PSNR/bytes) is deterministic and load-immune. Rule: develop under load if
  needed, but land/push decisions use idle-machine measurements only.

## 3. Measurement protocol (exact commands)

Decoder gap (idle machine, ~1 min):

    GOAV1_BENCH_CORPUS_ALLOW_UNMANIFESTED=1 GOAV1_BENCH_CORPUS=1 \
      go test -tags goav1_oracle -run TestCrossDecoderCorpus \
      ./internal/av1/testvector -v -count=1 -timeout 1800s

Read the AGGREGATE table; the dav1d `vs_goav1` ratio is the campaign number.

Decoder profile (single worker, representative clip): create the throwaway
hook `internal/av1/testvector/scratch_profile_test.go` (build tag
`goav1_oracle`; reads `GOAV1_PROFILE_CLIP`, calls `decodeCorpusClipDiscard` in
a 30x loop — an old copy may already sit untracked in the main repo), then:

    GOAV1_PROFILE_CLIP=$PWD/testdata/benchcorpus/p720_inter_q32.ivf \
      go test -tags goav1_oracle -run '^TestScratchProfileDecode$' \
      ./internal/av1/testvector -count=1 -cpuprofile /tmp/dec.out
    go tool pprof -top -nodecount=30 /tmp/dec.out

Delete the hook before committing anything (it breaks the untagged build if
written without the build tag).

Encoder quality (deterministic; one line per clip):

    go run ./cmd/qualitybench -input /tmp/corpus/realC.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/realA.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/realB.yuv  -fps 30 -frames 60  -bitrates 5000000 -encoders goav1 -layers 2
    go run ./cmd/qualitybench -input /tmp/corpus/screen.yuv -fps 60 -frames 120 -bitrates 1330000 -encoders goav1 -layers 2

Compare `psnr_avg` and `actual_bps` against §1. To re-baseline SVT, add
`-encoders goav1,svt-av1 -svt-preset 12`.

**Lever acceptance:** realC gains ≥ ~0.05 dB rate-adjusted; realA/realB hold
within −0.05 dB; screen within −0.10 dB at not-worse rate; realC wall ≥ 101 fps
measured idle; zero-alloc and round-trip gates green.

## 4. Gate checklist (run per commit, in this order)

    go build ./... && GOARCH=amd64 go build ./... && go build -tags purego ./...
    go test ./internal/av1/<touched-pkg>/ -count=1
    go test -race ./internal/av1/<touched-pkg>/
    go test -timeout 30m ./... -count=1          # root package alone takes ~10 min
    make dryrun-fast                              # 8/8
    make dryrun-extended                          # strict byte-exact suite, 0 FAIL
    scripts/check_allocs.sh

Then idle-machine measurement (§3), then commit (explicit paths), then push.

## 5. Decoder work queue (priority order)

Profile shares are % of single-worker decode CPU on p720_inter_q32 after wave
2. **Wave 3 landed since — RE-PROFILE FIRST (§3) before picking an item; the
distribution has shifted.**

**D1. ~~Inverse-transform columns~~ DONE (wave 3, commit 2ed2ad47).** Four-lane
NEON DCT32/DCT64 col+row kernels, dav1d itx16.S shape; DCT64 col went 23.7x,
e2e single-worker decode ~10-13% on the profile clip. Reusable generator +
constraints documented in `tools/itxgen/` — use it for the REMAINING transform
kernels: ADST/flip-ADST columns, identity scaling, and as the template for the
amd64 AVX2 wave. Also: the OLD DCT64 col2/row2 kernels
(NOSPLIT stack-headroom overflow, root cause of the May fuzz corruption) were
REMOVED in 539e1e2a — do not resurrect them; the pure-Go pair path is the
four-lane kernels' tail fallback.

**D2a. Loopfilter edge-builder run-length — LANDED (863e26a8).** Ported dav1d's
mask run-scanning idea to the scalar path: the luma previous-cache lookup now
returns a run length over which width+level are provably constant, so the
per-4x4-cell loops advance by whole runs. Byte-exact (227/227), ~1.3% decode.
A FULL dav1d bitmask port (Av1Filter.filter_y masks in src/lf_mask.c, bit-scan
consume in loopfilter_tmpl.c) is the bigger win but a high-risk rewrite of
byte-exact shared code — deferred. Encode LF cost is actually the transform-
tree replay / chroma path, not this luma loop.
SCOPING (2026-07-03, after a full dav1d-bitmask investigation — pinned): the
run-length win (863e26a8) ALREADY cut loopFilterPostFilterPlanTrustedSweep from
~36% to 10.7% cum decode — most of the tractable in-planner gain is captured.
The remaining cost is DISTRIBUTED (transform-tree resolution, neighbor-record
lookups frameWorkTryAppendLoopFilterFixedLumaEdge, edge storage), no single
hot line. dav1d is fast because it builds masks INCREMENTALLY during the decode
block-walk, carrying left/above tx-context arrays (a[]/l[]) so min(prev,cur)
width+level is resolved once and amortized; the deblock pass is then a pure
bit-scan with ZERO neighbor/tree lookups. goav1's threading.MarkBlockPtr stores
independent per-block records with NO cross-block edge context, so resolution
is deferred to deblock time (the 10.7%). The real port = a multi-commit project
rooted in threading.MarkBlockPtr + the tile block-loop building per-SB
Av1Filter-style bitmasks + level_cache[4] with a[]/l[] carry at DECODE time
(couples to the intra-tile wavefront recon), landed BEHIND the existing edge-
list API first (emit the identical segment list from a bit-scan to prove byte-
exactness on all 226 vectors), THEN swap the apply path. A partial in-file
"phase 1" is NOT separable — without the decode-time context it is pure added
indirection over the already-run-length-optimized path (no speedup, full
byte-divergence risk). Not a bounded single-session deliverable.
APPLY-SWAP LANDED (2026-07-03, dd5345fb + 66a77548 + 78945a50 + 9f2b5f62 +
dcfa3c1e): the dav1d bitmask loop filter is now the DEFAULT decode apply path.
Byte-exact (dryrun-extended + profiles + full suite PASS, race-clean, 0 alloc).
Controlled before/after (benchtime 5x, same machine): p288_inter_q20 −15.4%,
p360_inter_q20 −15.4%, p360_inter_q32 −15.1%, p720_inter_q32 −9.6% decode wall.
KEY FIXES the swap needed: (1) chroma has_chroma gate — masks emitted a chroma
edge+context for EVERY UV-tx block, but in 4:2:0 two sub-8px luma blocks share
one chroma sample and the 2nd read the 1st's just-written class as "previous"
(dav1d create_lf_mask_* only emit when (bw4>ss||bx&1)&&(bh4>ss||by&1)); (2)
run-merging contiguous same-(width,level) cells into one kernel call (per-cell
was SLOWER than edge-list — the merge is what makes it a win, byte-identical
since kernels filter each perpendicular line independently). GATED to SINGLE-
TILE (loopFilterMasksUsable); multi-tile falls back to the byte-exact edge-list.
FOLLOW-UPS status:
(a) ENCODER PATH — LANDED (8e7a93cb): FrameWorkLoopFilterMasks.BuildFromMap
walks the completed record map in MI-raster order (reproduces decode Z-order
masks for single-tile), encoder applySerialMasks routes the 3 single-tile
serial callers (P single-thread, L1T2 leaf, single-thread keyframe) to the mask
apply. Recon byte-identical, full suite green, 0-alloc. −11.8% single-thread
encode (37.97→33.49ms/op PFramePan1080pSingleThread), LF share 27%→20%. NOTE:
BenchmarkVideoEncoderRealC1080p (default multi-tile 32-col) does NOT move —
only the SetMaxThreads single-tile path. NEXT: the wavefront (SetMaxThreads>1)
path is single-tile but still uses the PARALLEL banded edge-list apply — route
it to a parallel mask apply → gives the realtime multithread encoder the LF win.
(b) MULTI-TILE decode — byte-exact port DONE but HELD (branch wip/lf-multitile,
f98b66bf): dav1d tx_lpf_right_edge tile-boundary fixup (store per-tile right/
bottom a[]/l[] context, apply imin(cur,neighbour) at boundaries; lf_apply_tmpl.c
313-430), dryrun-extended 0-FAIL with gate lifted. NOT merged: the per-cell mask
apply is −8.2% SLOWER than the edge-list on sparse multi-tile 720p (long uniform
edges favor the edge-list's run-merge; tile boundaries fragment the runs). To
land: make the mask apply run-length-batch better on sparse content so it beats
edge-list, THEN flip the single-tile gate (one-liner: return LoopFilterMasks
.Valid()). (c) public SetSideData path still edge-list fallback.
LATENT BUG FOUND (encoder agent): TestWebRTCStreamControlCombinationMatrixDecode
/L1T2 has a data race on e.singleThread — SetConfig writes it while the parked
background filter worker reads `if e.singleThread`. Reproduces on clean main
(pre-existing, unrelated to the mask work). Real fix needed (synchronize the
singleThread read/write or snapshot it per frame).
SECOND LATENT RACE (2026-07-06, full-suite -race, clean main): rare DATA RACE
in TestVideoEncoderCloseReleasesPersistentWorkers — write hme.go:84 (via
video.go:728) vs reads hme.go:67/156 (via video.go:544/818/790, overlap-test
teardown). Did not reproduce in 5 isolated runs; needs full-suite concurrency.
Same family as the singleThread race (worker teardown vs config write). Also
2026-07-06: TestKeyframeEncoder1080pMulticoreMemStatsAllocs' TotalAlloc==0
guard fails on clean main under -race (race runtime allocates); now skipped
under the race build tag — the zero-alloc contract is enforced by the plain
run + scripts/check_allocs.sh.
PRIOR STATUS (superseded) — LIBRARY + BUILD-WIRING DONE, APPLY-SWAP REMAINS:
LANDED ON MAIN: internal/av1/lfmask package (9fe8be8c build core + d50c2a09
scan core) + 1123cf53 differential test (7 cases). PRESERVED on branch
wip/lf-bitmask-build (a1a29475, NOT merged — regression alone): decode-time
mask BUILD wired into the tile-residual walk (FrameWorkLoopFilterMasks, a[]/l[]
carry in decode Z-order, 0-alloc, dryrun-fast byte-exact since masks unconsumed).
DO NOT merge a1a29475 alone: building masks nobody reads = pure added decode
work (~0.9ms/1080p frame) with no offset until the apply swap. THE REMAINING
WORK (the actual win) + 3 real gotchas the wiring agent found:
1. The differential test (1123cf53) compares UNORDERED per-4x4 cell maps — it
   does NOT prove the scan reproduces production's exact ORDERED + run-MERGED
   FrameWorkLoopFilterPostFilterEdge slice. So "emit the identical list" is
   real unproven work. BETTER PATH: skip the list entirely — apply the loop
   filter DIRECTLY from masks (dav1d loopfilter_tmpl.c bit-scan), verify the
   FILTERED PIXEL OUTPUT byte-exact via dryrun-extended. Output parity is the
   real requirement; the intermediate edge-list ordering is irrelevant if we
   stop materializing it.
2. TILE-BOUNDARY RACE: the build ORs into frame-wide per-128-region FilterMask
   shared across tiles; for 64px SBs a tile boundary splitting a 128-region
   lets two tile workers OR the same element. Benign while unconsumed; MUST fix
   (per-tile mask regions or serialized boundary fixup) before Part 2 consumes.
3. PUBLIC-PATH ASYMMETRY: BindDecoderFrameWorkSideData/SetSideData bypasses the
   internal runner → never builds masks. The apply consumer must build on the
   public path too, or keep the scalar builder as fallback there.
Each apply-swap step gated by the 20-min dryrun-extended. This is the deep grind
remaining; resume from wip/lf-bitmask-build, go straight to apply-from-masks
(skip list reproduction), fix the race first.

**D2. Loopfilter (~8% incl. plan sweep — was next after D1; re-profile to
confirm).** `filter14VertNEON`+`filter14Edge` (4.6% pre-wave-3) process one
edge per call; dav1d `src/arm/64/loopfilter.S` does 4 edges/16 pixels with
early-skip masks. Also `frameWorkAppendLoopFilterLumaEdgeSegments` (3.5% cum)
is scalar segment building — dav1d builds lf masks per superblock
(`src/lf_mask.c`). Only restructure along dav1d's actual shape. Wave-3 agent
did NOT start this — it is a clean target.

**D3. Restoration — AUDITED 2026-07-06; the win is ARCHITECTURAL (user
directive: "the real divergence is architectural so fix it").** Row-fusion of
the existing kernels was audited and DECLINED (loop-order only, intermediate
is L1-resident per unit, big asm rewrite risk for small win — see 52367902
commit message). The real divergence: goav1 restores via a WHOLE-PLANE
bordered uint16 round-trip (tile/restoration_frame_bridge.go: Load
BorderedSamplePlane full-plane u8→u16 widen → applyRestorationFrameToDst →
StoreBorderedSamplePlaneTrusted full-plane narrow store-back, with NONE units
hand-copied through), while dav1d (lr_apply_tmpl.c lr_stripe) filters IN
PLACE on native-depth pixels per 64-row stripe with small boundary backups
and never touches NONE units. PROGRAM STATUS (2026-07-06 afternoon — P1-P3 LANDED + PUSHED):
- **P1 LANDED (015fe524):** NONE units never read/copied/written; store-back
  narrows only filtered record rects; all-NONE planes skip the round-trip.
  copyRestorationUnit eliminated from the profile. p720 −1.4% cpu.
- **P2 LANDED (99114ec1):** u8-source Wiener+SGR kernel family (dav1d 8bpc
  shapes on goav1's pass structure; NEON+AVX2+pure-Go; differentials prove
  byte-parity vs u16 kernels). Per-kernel ≈parity — the win is structural.
- **P3 LANDED (783c1d10):** 8-bit LR runs in the pixel domain — bordered u8
  snapshot (byte view over the u16 arena, half footprint) replaces the plane
  widen; filters write DIRECTLY into the frame (VisibleRect-clipped — the
  FilterRect overhang polluted padding, caught by differential); dst scratch
  + store-back deleted for 8-bit. −1.2-1.3% on every LR-active 8-bit clip;
  10/12-bit keeps the u16 path.
- **P4 LANDED (ddddce3e) — PROGRAM COMPLETE.** In-place stripe walk (dav1d
  lr_stripe/backup4xU shape): vertical seam halo always comes from the saved
  boundary lines, left-neighbor halo from a 4-column pre-overwrite backup,
  per-stripe ~27KB bordered band assembly, kernels write the frame directly.
  No parallel gate needed — LR is a single serial whole-frame call with
  pinned row-major record order. LoadBorderedBytePlane gone; LR scratch
  traffic 1.4MB→30KB/frame; e2e p720_inter_q32 −1.15% cpu. FOLLOW-UPS: shrink
  the now-oversized u16 Data/Dst arena size contract (D5 memory, safe);
  remaining LR cost is the Wiener/SGR kernels themselves (row-fusion was
  audited + declined — only revisit with new evidence).

**D4. High-bit-depth coverage — MAJOR SLICE LANDED (77addf06, 2026-07-06):**
the HBD 2D convolve wrapper zero-filled a ~69KB int32 stack `im` on EVERY call
(11% flat of 10-bit decode) — scratch now threaded (ConvolveScratch gained
imHBD + edge16); blendCompoundAvgHighBD got a NEON kernel (dav1d mc16.S
bidir_fn avg shape, libaom-offset CONV_BUF arithmetic, ~17x); compound HBD
clamped 2D got emu-edge scratch. p360_inter_q32_10bit: wall −20.5%, cpu
−20.3%, dryrun-extended 226/0. REMAINING HBD: predictInterCompoundRefHighBD
ToConvBufXClamped (1.7% flat / 3.6% cum — emu_edge+NEON port is the next
lever); warpHorizontalHighBDResident ~2-3% is OFF-LIMITS (warp.go).
LESSON (recurring disease, grep for it): any wrapper with a `var im [big]T`
nil-scratch fallback zero-fills per call — the ENCODER had the same bug
(predictInto* passed nil scratch, 15.6% flat of single-thread encode, fixed
4ab59fed) and the amd64 AVX2 wrappers still do (convolve_avx2_amd64.go:119
34KB int16 + convolve_highbd_avx2_amd64.go 69KB int32 — now partially
scratch-capable via 77addf06's dispatch slots; verify native-x86).

**D5. Runtime overhead (~10%: madvise 6.4%, memmove 2.3%, memclr 1.7%).**
madvise = GC returning pages; find the per-decode-run allocations feeding it
(memprofile the corpus decode; suspects: per-run layer/surface allocation in
the corpus path). Reuse buffers across frames/runs where the public decoder
already does (its steady state is ~0 allocs). memmove/memclr: `pprof -peek` to
find sources; kill redundant copies/clears only where dav1d has none. Do NOT
suppress GC in benchmarks to fake the number.

**Pinned decoder dead-ends (do not retry):** entropy symbol-decode NEON (tiny
alphabets, scalar wins); blanket bounds-check-elimination reslicing (regresses
on arm64); single-worker pool inline fast path (ALREADY DONE — 71d5f36a);
remaining `pthread_cond_signal` is Go scheduler `wakep` at GOMAXPROCS>1, not
pool signaling — not actionable; **CDEF pri+sec fusion (2026-07-06 agent
audit): ALREADY the production shape** — cdefFilterBlock{8,4}NEON are fused
dav1d cdef_filter_pri_sec ports, and the profile's separate Primary/Secondary
kernels are the deliberate single-strength split (dav1d has the same 3-way
`_pri`/`_sec`/`_pri_sec` dispatch); forcing fused onto single-strength inputs
measured 27-54% SLOWER per kernel. The old "dav1d fuses = next CDEF lever"
note was stale. Next real CDEF lever would be in-place filtering to kill the
staging copies, which is ITSELF pinned dead until someone rewrites the 8-bit
kernels to read uint8 directly (see memory perf-deadends).

## 6. Encoder work queue (priority order)

**E0. RESOLVED (wave 3).** (a) Interpolation-filter search LANDED (commit
07f500a9): base frames signal SWITCHABLE and run the 3-combo IFS at MDS3 with
the Laplacian `model_rd_from_sse` model and frozen `switchable_interp_fac
_bitss` rates (SVT enc_inter_prediction.c), plus libaom's `fix_interp_filter`
collapse with one-frame lag; realC +0.21 dB, realA +0.13, realB +0.18, screen
flat. The IFS rate-table/syntax infrastructure (`tile.InterpFilterRateTables`,
`tile.WriteInterpFilters`, `pframe_ifs.go`) is reusable for other per-block
syntax decisions. (b) MD subpel level 6 measured RED and is PINNED DEAD — see
the dead-end list below.

**E1. THREADING MODEL: intra-tile wavefront — LANDED (stages a/b/c, commits
83a45933 / 31a4a0a2 / 21038fd6 / dbc455b7).** The realC quality endgame.
The old model parallelized across 32 tile columns at 1080p; per-tile CDFs
cost ~0.5 dB. Now SetMaxThreads(n>1) runs INTER frames on a single-tile
SB-row wavefront (n decision lanes, top-right dependency; SVT ENC-DEC
segment model in enc_dec_process.c + the decoder's own recon wavefront)
with a serial entropy replay (libaom encode_sb/pack_bs split); keyframes
keep n tile columns (no decision pass to wavefront — decoupled via
keyframeTileColsLog2, so no keyframe latency stall). RESULT (median-of-3,
idle, matched rate, -goav1-max-threads 14): realC 44.71→**45.17 dB @ 77 fps**
(vs SVT 44.77 @ 131 → goav1 +0.4 dB PSNR, +1.2 dB XPSNR, comfortably
realtime); realA 46.67→47.27; realB 46.99→47.54; screen 50.77→51.28 @ 139
fps. goav1 is now clearly quality-superior to SVT on every clip while staying
above each clip's frame-rate deadline. Byte-identical serial-vs-wavefront at
2..16 lanes (TestPFrameWavefrontMatchesSerial) and split-vs-fused at 1/4/32
tiles (TestPFrameSplitMatchesFused); zero-alloc (1.0/op amortized).
STILL OPEN — raw fps: the wavefront tops at ~83 fps single-tile (14 lanes)
because the SERIAL entropy write (~6.8 ms/frame) + loop filter is the Amdahl
bound; it does NOT beat SVT's raw 131 fps. To pass SVT on fps too, PIPELINE
the serial entropy write of frame N with the wavefront decision of frame N+1
(SVT overlaps EC of the finished segment rows with enc_dec of the next), and/
or overlap LF+CDEF of N with decision of N+1. That is the E1-followup and the
next threading lever. NOTE the wavefront is wired for the i420-8 VideoEncoder
only; mono/high-bd pixel formats still use tile columns (extend if needed).

**E2. NEAREST_NEAREST compound at MDS0.** SVT p12 light-PD1
DOES inject compound at MDS0: `inject_mvp_candidates_ii_light_pd1`
(Source/Lib/Codec — the rf[1] != NONE arm) prices NEAREST_NEARESTMV compound
(syntax-cheap, zero MV bits) against the single-ref set. Extend
`internal/av1/encoder/pframe_mds0.go` to inject and price it with the
already-frozen rate tables; prediction via the decoder-exact CONV_BUF path
(`PredictInterCompoundRefToConvBufWithScratch` + `BlendCompoundAvg` — now
NEON). NOTE this does NOT contradict the "compound dead on realC" pin: that
pin is about the OLD golden-gated compound path (`compoundGoldenLikely` never
arms); MDS0-priced NEAREST_NEAREST is the upstream-grounded replacement and is
untested. If it lands, consider deleting the dead golden-gated path.

**E3. NSQ (non-square) partition search.** SVT p12 runs `nsq_geom_level=4` /
`nsq_search_level=19` — cheap H/V rectangle candidates inside the MD loop with
real rate costing. goav1 has rect-32 partitions landed but SAD-gated
conservatively (measured neutral). Re-drive rect candidates through the MDS0
machinery (`internal/av1/encoder/pframe_mds0.go` — fully predict + RDCOST, same
frozen rate tables). Upstream: `Source/Lib/Codec/` nsq search config in
`signal_derivation_*` + the md_stage candidate injection for H/V shapes.

**E4. Partition depth — LANDED + EXHAUSTED (18b20370). RE-FRAME OF THE 5x.**
The disallow_below_16x16 absolute-cost gate removed 91% of CONTENT-driven 8x8
on realC (4576→420 splits/60f) at +0.106 dB / −5.2% P-frame work. Forensic
finding (decideRealtimePartition fate counts): the residual ~40% 8x8 by BLOCK
COUNT are STRUCTURAL forced-splits — frame edges (!haveRight/!haveBottom) +
the bottom 56px partial SB row (1080%64=56) — which bypass the var-partitioner
and MUST split for boundary correctness. They are only ~5% of frame AREA (the
40% count overstates them: a forced partial region emits many tiny blocks).
The SVT dev_16x16_to_8x8 DEVIATION arm was fully probed (independent per-16x16
ME, no double-ME, cacheable) and REJECTED: it moves 8x8 by 0.0–0.4pp / PSNR
flat — there is nothing left to remove. SEARCH-BREADTH REDUCTION IS EXHAUSTED
on realC. The remaining single-thread CPU (43 ms/frame) is: ~25% loop-filter
edge planning (SCALAR, shared decoder code — the one big reducible target, see
D2/D2a; full dav1d bitmask port is the win, byte-exact-risky) + ~40% MANDATORY
leaf compute already on NEON (residual/quarter-plane/convolve2D8/quant/rdStats)
— that part is the pure-Go-vs-SIMD-C structural floor. So "fix the 5x" further
= (a) the loop-filter bitmask port (dual-benefit with the 4.34x decode gap) or
(b) E1-followup pipelining for WALL time (search breadth is at its floor).

**E4-OLD. Depth removal — ARCHITECTURE-BLOCKED as SVT formulates it (pinned).**
SVT's `set_depth_removal_level_controls` (enc_mode_config.c:2935) needs
INDEPENDENT per-block-size ME distortions (dist_64/32/16/8, each size finding
its own MV) to compute `dev_16x16_to_8x8`. goav1 partitions with libaom's
var-based partitioner (realtimeFillVariance8x8AvgAt): all depths share ONE
SB-level int-pro MV, so the 8x8 residual SSEs sum EXACTLY to the 16x16 SSE →
SVT's deviation is identically 0. Producing the signal would need an eager
per-size ME pre-pass — the very work depth-removal saves. THE REAL WORK GAP
(measured realC): partition mix 44.5% 8x8 / 38.2% 16x16 / 12.3% 32 / 5.0% 64,
vs SVT near-zero 8x8 at p12/1080p — this 8x8 over-split IS the ~5x. The
portable lever (IN PROGRESS): SVT `disallow_below_16x16` absolute-cost gate
(disallow_below_16x16_cost_th) — give the var-partitioner a per-16x16 ME
distortion reused from the coarse int-pro projection (no new ME pass) and gate
16->8 splits on it. Decision-changing; quality gate realC >= 44.85 (we're
45.19, SVT 44.77 — ~0.35 dB to spend).

**E5. dist_based_ref_pruning — LANDED (0f48c679).** Ported SVT
`perform_md_reference_pruning` leaf gate (max_dev_to_best[PA_ME]=30): skip
GOLDEN's subpel+compound when its full-pel ME dist is >30% worse than LAST.
realC +0.02 dB, others flat. CPU-NEUTRAL though — GOLDEN is only 2.1% of realC
blocks (golden-off ceiling is just −4.4% CPU). Correct + quality-positive but
NOT a 5x lever; the 8x8 split (E4) is.

**E6. Full lpd1 detector for leaves.** The current leaf cap is the "moderate
distortion band" approximation of SVT's `lpd1_detector_post_pd0`. Port the
real detector (dist/rate/MV-length thresholds per SB) — closes the last
structural difference in leaf effort steering.

**E7. Screen rate overshoot (+17% over target).** SVT lands 1.363M on a 1.33M
target; we land 1.561M. Investigate the CBR loop's behavior at the q floor
(SVT undershoots when floored). Port SVT's rate-control clamping for floored
frames rather than tuning our controller (controller knob experiments are all
pinned dead: PI terms, step clamps, refinement schedules).

**E8. Wall-time work reduction (NOT a single-thread-parity program).**
USER DIRECTIVE (2026-07-02): single-thread parity with SVT is a NON-GOAL — a
realtime encoder deploys with cores, and threading here is CPU-free; wall fps
at the deployment thread budget is the only speed metric. Note the measured
"5.2x single-thread CPU vs SVT" is also confounded: goav1 at 1 thread encodes
single-tile at HIGHER quality (45.29 dB > SVT 44.77), i.e. more work for more
output. Reduce per-frame work only where it moves WALL time at normal thread
counts: the E4/E5/E6 SVT ports (depth pruning, ref pruning, lpd1 detector)
first, then NEON for scalar spots that survive. pprof NOTE: profile encoder
work at GOMAXPROCS=1 (pthread_cond percentages in MT profiles are
blocked-thread sampling artifacts), but JUDGE results by idle wall fps at
deployment threads. Work profile (realC): encodePBlock 41% cum
(convolve2D8I8MM ~13% subpel/recon, finishInterTXB 8.6%), LF sweep 8.4%. goav1 wall wins ride on ~5x
parallelism vs SVT's ~1.6x (3x more CPU burned). If wall targets are met,
reduce CPU: encoder-side NEON for the remaining scalar hot spots (profile
`BenchmarkVideoEncoderRealC1080p` with `-cpuprofile`), e.g. staging transposes,
SATD if E-levers introduce it.

**Pinned encoder dead-ends (do not retry):** GOLDEN-GATED compound path on
realC (`compoundGoldenLikely` never arms; libaom RT speed≥9 disables compound
too — but see E1: MDS0-priced NEAREST_NEAREST is a different, upstream-
grounded mechanism and is NOT pinned); SVT pruned-subpel level 6/8 on T0
(faithful port of `svt_av1_find_best_sub_pixel_tree_pruned` + level-6 controls
measured realC −0.20 dB rate-adj; even cost-model-only −0.03 — our coarser
full-pel ME needs the full first+second-level probe schedule; wave-3 numbers
in memory); eighth-pel MVs with current subpel; SMOOTH family at 8x8; 32-tier
exact-rate merges without chroma+CDF calibration; frame-wide SHARP/SMOOTH;
PI/step-clamp/tier-split controller variants; golden qindex refresh rules
(measured red — capture bad anchors during post-key convergence); SSE instead
of variance distortion in MDS0 (realC −0.08); T0-only MDS0 scoping (screen
+4.4% rate); wall-clock feedback signals (banned, irreproducible).

## 7. Reporting and memory

After each landed wave: update the scoreboards in this file (§1), append
results + new pins to the session memory (`encoder-perf-target`, `simd-plan`,
`encoder-quality-deadends` in the Claude memory dir), and push. Keep reports in
the shape: lever → upstream citation → all-four-clips (or corpus-aggregate)
before/after → verdict → commit hash.
