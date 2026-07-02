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

## 1. Current standings (measured clean 2026-07-02 evening, HEAD 2ed2ad47 + plan sync)

Decoder, single-thread, internal corpus (18 clips, 48 frames each):

| decoder | total ms | vs goav1 |
|---|---|---|
| goav1 | 2833 | 1.00x |
| aomdec | 1095 | 2.59x faster |
| dav1d | 652 | **4.34x faster** ← the gap |

(Session start was 5.12x; waves 1-3 recovered ~15.7% wall.)

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
(128 vs 77); closing THAT needs the E1-followup (pipeline entropy-write of
frame N with decision of N+1 — the ~6.8 ms serial write is the Amdahl bound).

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

**D2. Loopfilter (~8% incl. plan sweep — was next after D1; re-profile to
confirm).** `filter14VertNEON`+`filter14Edge` (4.6% pre-wave-3) process one
edge per call; dav1d `src/arm/64/loopfilter.S` does 4 edges/16 pixels with
early-skip masks. Also `frameWorkAppendLoopFilterLumaEdgeSegments` (3.5% cum)
is scalar segment building — dav1d builds lf masks per superblock
(`src/lf_mask.c`). Only restructure along dav1d's actual shape. Wave-3 agent
did NOT start this — it is a clean target.

**D3. Wiener restoration (~3.9%).** `wienerHorizontalNEONAsm` +
`wienerVerticalNEONAsm` exist but compare their structure against dav1d
`src/arm/64/looprestoration.S` (dav1d keeps 16-bit intermediates and fuses
rows). Port the delta if the shapes differ materially.

**D4. High-bit-depth coverage (worst clip: 10-bit at 0.12x of dav1d).** Wave-1
left high-BD 2D convolve variants pure-Go (`loadHighBDSample` coordinate helper
blocked reslicing). Sweep `internal/av1/motion`, `cdef`, `loopfilter`,
`transform` for `HighBD`/16-bit-pixel paths that still run pure-Go, and port
dav1d's 16bpc arm64 kernels for the biggest ones. Profile with a 10-bit clip:
`GOAV1_PROFILE_CLIP=$PWD/testdata/benchcorpus/p360_inter_q32_10bit.ivf`.

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
pool signaling — not actionable.

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

**E4. Depth removal / block-based depth refinement.** SVT p12: `depth_removal
_level=5`, `block_based_depth_refinement_level=10` — prunes partition depths
per SB from ME distortions before MD. Port to cut decide()-tree work; spend the
recovered time on E1 breadth. This is a speed lever that buys quality budget.

**E5. dist_based_ref_pruning (base 3 / leaf 6).** Prunes reference candidates
by ME distortion ratios. Cheap, upstream-exact, frees cycles on leaves.

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
