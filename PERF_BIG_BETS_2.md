# goav1 — strategic big bets, round 2 (cpu-time lens)

Author: strategic-architect pass, 2026-07-03. Companion to `PERF_BIG_BETS.md`
and `PERF_PLAN.md`. Bet B (amd64/AVX2 critical-path parity) is the prior file's
recommendation and has **landed for the 3 dominant gaps** (inverse DCT32/64,
encoder SAD, forward DCT16/32). This round plans the **next** structural bet,
judged strictly by **CPU TIME (core-seconds/frame)** — the multiplier that sits
under *both* single-stream latency *and* multicore throughput.

All numbers below were **measured this session on the arm64 M4 Max host**;
commands and raw pprof are cited inline. Two subagents audited dav1d's threading
model (`third_party/upstream/dav1d/src`) and goav1's frame-parallel readiness.

---

## 0. Executive summary — the one bet to start now

> **START: SINGLE-THREAD DECODE CPU REDUCTION, led by porting dav1d's
> DECODE-TIME loop-filter bitmask _build_.**

**The cpu-time argument (why this outranks frame-threading).** Throughput =
(work-per-frame) × (cores). Threading only multiplies the second factor; it does
**not** touch the first. dav1d *also* frame-threads, so at equal core counts the
**per-core cpu-time ratio (measured 4.39×) is the hard ceiling on aggregate
throughput parity** — no goav1 threading scheme can beat it, because both sides
scale across the same cores. The *only* lever that improves aggregate throughput
**and** single-stream latency simultaneously is **reducing cpu-time/frame.** That
makes single-thread cpu the top structural axis, exactly as the cpu-time rule
demands.

**The data says the reducible cpu is NOT where the last plan assumed.** A fresh
`GOMAXPROCS=1` profile on a representative post-filtered inter clip (14.78 s of
samples — see §1a) shows the **#1 reducible hotspot is the SCALAR loop-filter
edge planner**, not entropy:

| subtree | flat | cum | nature |
|---|---|---|---|
| `loopFilterPostFilterPlanTrustedSweep` | 3.18% | **20.57%** | scalar LF edge-list BUILD |
| `frameWorkAppendLoopFilter{FixedLumaTXBs,ChromaTXBs,LumaEdges,ChromaEdges,…}` | **~15% combined** | — | scalar per-4×4 edge emission |
| entropy (`readCDF4*`, `readSymbolKnown`, `ReadBit*`) | ~8% | — | serial wall, **near floor** |
| prediction geometry (`frameWorkClipVisiblePixelsToWindow`, `blockPredictionPlaneGeometry`, `frameWorkLoadSample`) | ~4.5% | — | recomputed per block |
| `filter4EdgePureGo` | 1.15% | — | LF apply **still pure-Go** (not NEON) |

The loop-filter **apply** swap already landed (dav1d bitmask, −15% decode). What
remains scalar is the **edge/mask BUILD** — dav1d builds masks *incrementally
during the block-walk* carrying left/above tx-context (`a[]`/`l[]`) so
`min(prev,cur)` width+level resolves once and amortizes, then the deblock is a
pure bit-scan with **zero** neighbour/tree lookups (`src/lf_mask.c`,
`src/loopfilter_tmpl.c`). goav1 still defers that resolution to plan-time (the
20.6% cum). Porting the decode-time build is the largest **non-floor**,
**dav1d-source-grounded**, **byte-exact-verifiable** cpu win left on the decoder,
and it is **shared code** so it pays the encoder too (the encoder's LF planning
is ~23% cum on its default path — `PERF_BIG_BETS.md` §1d).

**Expected yield.** Recover most of the ~15–20% scalar LF-build cost on decode
(→ ~10–13% e2e single-thread decode cpu), *plus* a matching cut on the encoder's
default multi-tile LF path — a **cpu-time reduction that lifts single-stream fps
and multicore throughput at once**, on both arches, byte-exact. It moves the
4.39× decode gap toward ~3.7–3.9× — real, not a leapfrog.

**Phase-1 (bounded, ~1 session, zero decode-output risk).** *Do not touch the
apply path yet.* First **re-confirm the target on the full corpus** (the 20.6%
was measured on one 160×128 4:4:4 clip; `PERF_PLAN` D2a claims a prior cut to
10.7% on 720p — reconcile before committing effort), then **build masks at
decode-time in the tile block-walk and emit the EXISTING edge-list from a
bit-scan**, gated behind the current scalar builder, proving byte-identical
`FrameWorkLoopFilterPostFilterEdge` output on all 226 strict vectors. Only after
that passes does a later phase delete the scalar builder. Acceptance: identical
strict-MD5 (`make dryrun-extended` 0-FAIL), 0-alloc, race-clean, and a measured
`GOMAXPROCS=1` decode-cpu delta on the corpus.

**Honesty up front.** Single-thread pure-Go decode will **not** reach dav1d
parity. Of the 4.39× gap, ~2–2.5× is the irreducible pure-Go-vs-hand-asm floor
(entropy is a serial scalar wall already near its floor; the NEON kernels are
~1.2–1.5× behind hand-tuned C-asm). The **recoverable** slice is the scalar
distributed overhead — LF build (~15%), geometry recompute (~4.5%), the residual
pure-Go LF apply (`filter4EdgePureGo`) — plausibly **~20–25% of decode cpu**
total across a focused program. That is the winnable part, and it is worth more
than frame-threading's honest yield.

### Ranked bets (this round)

| # | Bet | cpu/frame? | Aggregate throughput yield | Single-stream yield | Certainty | Effort | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **Single-thread decode cpu reduction (LF bitmask build + geometry + residual pure-Go apply)** | **↓ reduces** | **positive** (less work/core) | **positive** | **High** (dav1d-grounded, byte-exact gate) | Medium (multi-commit) | **START** |
| 2 | Frame-parallel decode threading | — spreads only | **~0** (dav1d also threads; recon already wavefronts) | +1.3–1.6× latency only | Medium (3 arch blockers) | Large/high-risk | Only if single-stream 4K latency is a product need |
| 3 | Single-thread ENCODE cpu reduction | ↓ reduces | positive | positive | Medium (near SIMD-C floor; LF-mask MT apply the one lever) | Small–Medium | Fold LF-mask MT apply into Bet 1 |
| 4 | amd64 native-x86 cpu-time VALIDATION | ↓ (measures the biggest deploy-arch win) | positive | positive | High | Small (infra) | Out-of-scope-flagged but the largest unrealized win |

---

## 1. Measurement snapshot (this session, arm64 M4 Max, idle)

### 1a. Fresh single-thread decode hotspot map (the campaign's LF/HBD/emu-edge work has shifted the distribution)

`GOMAXPROCS=1 BenchmarkDecodePostFilteredProfileClip -benchtime=1500x
-cpuprofile` (profile-1 4:4:4 8-bit cdef+restoration inter clip; 14.78 s total
samples — a solid, representative post-filtered inter profile, unlike the
lossless q00 clip which over-weights entropy):

- **Scalar loop-filter edge PLANNING dominates the reducible cost.**
  `loopFilterPostFilterPlanTrustedSweep` 20.57% cum; the `frameWorkAppendLoopFilter*`
  emission leaves sum to **~15% flat self-time**. This is scalar tree/neighbour
  resolution deferred to plan-time — exactly what dav1d amortizes at decode-time.
- **Entropy is ~8% flat here and near floor.** `readCDF4UpdateKnown` 3.86%,
  `readCDF4HighTokenUpdateArch` 2.71%, plus small `readSymbolKnown`/`ReadBit*`.
  (On the *lossless* q00 clip entropy self-time balloons to ~38% flat with
  `readCDF4UpdateKnown` alone 14.26% / `readCoefficientsTXBTracked2DWithGeo`
  46.4% cum — but lossless is the coeff-heavy worst case, not streaming content.)
- **Prediction geometry recompute ~4.5% flat** (`frameWorkClipVisiblePixelsToWindow`
  2.77%, `blockPredictionPlaneGeometry`, `frameWorkLoadSample`) — candidate for
  caching, same class as the landed job-geometry cache.
- **`filter4EdgePureGo` 1.15% flat — a loop-filter APPLY kernel still on the
  pure-Go path** (not NEON). A clean, low-risk NEON port target.
- Transform/CDEF/restoration leaves (`inverseSeparableBlockClampedRows`,
  `cdefFilterBlock8PrimaryNEON`, `boxsumInteriorBandNEON`) are already NEON —
  the SIMD-C floor.

### 1b. Wavefront reachability — CORRECTION to `PERF_BIG_BETS.md` §1b

`PERF_BIG_BETS.md` claimed the SB-row wavefront recon is "gated behind
`GOAV1_DEFER_RECON` … NOT active in production decode." **That is now
inaccurate.** Audit of `internal/av1/threading/pool.go:1137-1152` +
`tile_residual.go:1150-1256`:

- When a **single batch** is dispatched (single-tile frame), the pool sets
  `WavefrontWorkers = len(p.workers)` and `deferReconstruction` engages whenever
  `WavefrontWorkers > 1` — i.e. **`GOAV1_DEFER_RECON` is only a *force-on* hook;
  the real trigger is workers>1 on single-tile.**
- This path is driven by the **production stream-runner API**
  (`DecoderFrameWorkResidualLowOverheadStreamsPlan`, exercised by
  `wavefront_parallel_test.go` which explicitly says it "drives the same public
  stream-runner path that production callers use"). So **single-stream single-tile
  decode ALREADY parallelizes RECONSTRUCTION across cores** when the caller
  supplies workers>1.
- **What stays serial is ENTROPY (pass 1).** Entropy decode is strictly serial
  per tile (in-place CDF). So single-stream decode = *serial entropy → parallel
  recon wavefront*, Amdahl-bound by the serial entropy fraction (~8–38% of cpu,
  content-dependent).

Consequence: the "single-tile = 1 core" premise behind frame-threading is
**half-true** — recon is already scaled; only entropy is core-capped. This
materially lowers frame-threading's incremental yield (§2, Bet 2).

### 1c. Decode gap — 4.39× single-thread (unchanged, from `PERF_BIG_BETS.md` §1a)

goav1 827.9→3632.9 ms vs dav1d 827.9 ms on the 864-frame corpus (1 worker). The
gap is distributed pure-Go-vs-SIMD-C, not one hotspot.

### 1d. dav1d frame-threading machinery (subagent study of `src/`)

- **Frame contexts**: `Dav1dContext.fc[n_fc]`, round-robin assignment
  (`decode.c:3333+`). Two-pass mode only when `n_fc>1` (`thread_task.c:223`):
  Pass 1 = entropy + CDF adapt (`entropy_progress`); Pass 0 = reconstruction with
  row deps (`deblock_progress`/`frame_progress`).
- **Reference row-progress deps**: `Dav1dTileState.lowest_pixel[sbrow][7 refs][2]`
  computed during recon (`decode.c:1957-2032`); `check_tile()` (`thread_task.c:403-431`)
  re-queues a task until `refp[n].progress[!tp] >= lowest`. Pictures are atomic-
  refcounted (`picture.c`, `ref.c`); buffers stay alive until all dependents finish.
- **CDF chain**: `primary_ref_frame` selects the inherited adapted-CDF
  (`decode.c:3492-3501`); recon(N) can overlap entropy(N+1), but **entropy(N+1)
  waits on entropy(N)'s CDF progress** when the chain links — the same
  serialization class as the encoder's proven EC(N)→MD(N+1) dead-end.
- **Output reorder**: `out_delayed[n_fc]` + round-robin `first`/`next`
  (`decode.c:3333-3389`).
- **Scale**: ~5,300 lines across `thread_task.c` (936), `decode.c` (3,746),
  `internal.h`, `picture.c`, `cdf.h`; ~15–20 coordination points.

### 1e. goav1 frame-parallel readiness — 3 hard blockers (subagent audit)

| Blocker | current state (file:line) | needed |
|---|---|---|
| **Surface refcount** | LIFO pool, `used[bool]`, immediate serial release (`frame/pool.go:94-172`, `decoder/surface.go:5-12`) | per-surface refcount OR enlarged pool + delayed release |
| **CDF chain** | N+1 `initialTileResidualCDFs` reads N's END-of-frame saved ctx (`decoder/work.go:542-575, 710-741`) | per-in-flight-frame CDF buffers; N+1 entropy waits on N entropy |
| **Motion field** | published post-tile-work (`decoder/motion_field.go:8-31`) | early/eager publication |
| Frame state | single `DecoderFrameWorkState` (`decode_simple.go:40`) reused per frame | per-in-flight-frame state pool |

**Already present**: the byte-identical differential harness
`TestWavefrontByteIdenticalToSingleWorker` (`wavefront_parallel_test.go:164-173`)
hashes frame planes across workers=1/2/3/4 — the template for any thread-count
byte-exactness gate.

---

## 2. Per-bet analysis

### BET 1 — Single-thread decode cpu reduction  ★ RECOMMENDED

**Thesis.** cpu-time/frame is the multiplier under both latency and throughput;
it is the only axis on which goav1 can gain aggregate ground on dav1d (which also
threads). The fresh profile (§1a) locates the largest **non-floor** cpu in the
scalar loop-filter edge planner (~15% flat / 20.6% cum), with a second tier in
prediction-geometry recompute (~4.5%) and a stray pure-Go LF-apply kernel
(`filter4EdgePureGo`). All three are dav1d-grounded or mechanical.

**Data.** §1a profile; dav1d `src/lf_mask.c` + `src/loopfilter_tmpl.c` (decode-
time incremental `a[]`/`l[]` mask build → zero-lookup bit-scan deblock);
`PERF_PLAN` D2/D2a (the apply-swap landed −15% decode; the BUILD remains scalar).

**Estimated yield (honest, cpu-time).** LF build: recover most of ~15% flat →
~10–13% e2e single-thread decode cpu, **plus** a matching cut on the encoder's
default multi-tile LF planner (~23% cum, shared code). Geometry cache: ~2–4%.
`filter4EdgePureGo` NEON: ~1%. A focused program: **~20–25% single-thread decode
cpu**, i.e. 4.39× → ~3.5–3.7×. This lift applies to **every core** in a threaded
deployment (throughput) **and** to single-stream latency — the dual win the
cpu-time rule rewards.

**Risk: MEDIUM.** The LF-build port is byte-exact-critical shared code and
**couples to the intra-tile wavefront** (masks must build in decode Z-order).
Mitigated by the phased approach: build masks and **emit the existing edge-list
from a bit-scan first** (behind the scalar builder), proving byte-identical
`FrameWorkLoopFilterPostFilterEdge` on all 226 vectors before deleting the scalar
path. Geometry-cache and NEON-kernel work are low-risk. `PERF_PLAN` D2a warns the
decode-time build is "not a bounded single-session deliverable" — treat as
multi-commit, which fits a "big bet."

**Effort: MEDIUM, multi-commit** (parallelizable across the 3 sub-levers; the
2-agent default fits: one on LF-build, one on geometry+NEON-apply).

**Approach (phased).**
1. Re-profile the **full corpus** at `GOMAXPROCS=1` to confirm the LF-build share
   at 720p (reconcile the 20.6% here vs D2a's claimed 10.7% — clip-size effect).
   Also profile a 10-bit clip (HBD paths may still be scalar).
2. Decode-time mask build in the tile block-walk (`threading.MarkBlockPtr` +
   per-SB `Av1Filter`-style bitmasks + `level_cache[4]` with `a[]`/`l[]` carry),
   **emitting the identical edge-list from a bit-scan**, gated behind the scalar
   builder. Byte-exact via strict-MD5.
3. Swap the planner to the mask build; delete the scalar sweep. Extend to the
   encoder's default multi-tile path (dual benefit).
4. Geometry-recompute cache (extend the landed job-geometry cache to
   `frameWorkClipVisiblePixelsToWindow`/`blockPredictionPlaneGeometry`).
5. NEON-port `filter4EdgePureGo` (mechanical, dav1d `loopfilter.S` shape).

**Verification.** `make dryrun-extended` 0-FAIL (byte-identical) at each step;
`-race`; `scripts/check_allocs.sh` 0-alloc; measured `GOMAXPROCS=1` corpus
decode-cpu before/after; encoder round-trip + qualitybench unchanged.

**Phase-1 acceptance (bounded, ~1 session, ZERO decode-output risk).** Full-
corpus `GOMAXPROCS=1` profile committed as the target baseline; decode-time mask
build emitting the existing edge-list from a bit-scan, **gated behind the scalar
builder** (masks built but list still produced by the proven path OR by the scan
with a differential assert); byte-identical strict-MD5 on all 226 vectors;
0-alloc; race-clean. No apply/plan swap yet — that is a later phase. Success =
the scan reproduces the exact ordered+run-merged edge-list, unblocking the swap.

---

### BET 2 — Frame-parallel decode threading  (demoted: single-stream latency only)

**Thesis (the last plan's Bet A, rigorously re-evaluated).** dav1d's headline
throughput is frame-threaded; goav1 has none. The claim was ~6–9× single-stream
on 12 cores.

**Why the cpu-time lens demotes it.**
1. **Aggregate many-stream throughput yield ≈ 0.** Frame-threading spreads work,
   it does not reduce it. dav1d *also* frame-threads, so on N independent streams
   across N cores the 4.39× per-core cpu ratio is simply preserved — both sides
   scale, the ratio holds. There is **no aggregate-throughput parity to be won
   here.** (Confirmed by the cpu-time rule: threading is cpu-neutral.)
2. **Single-stream recon is ALREADY parallel** (§1b): the reachable wavefront
   scales reconstruction across cores today. Frame-threading's *only* incremental
   gain is overlapping the **serial entropy** of frame N+1 with recon(N) — bounded
   by the serial entropy fraction and by how much recon(N) already saturates the
   cores. Honest single-stream uplift **~1.3–1.6×**, not 6–9×.
3. **The CDF chain re-serializes entropy.** When `primary_ref_frame` links
   (the common low-delay/streaming case), entropy(N+1) must wait on entropy(N)'s
   adapted CDF (dav1d `in_cdf.progress`, `thread_task.c:585-591`) — the **same
   serialization class as the PROVEN encoder EC(N)→MD(N+1) dead-end**
   (`PERF_PLAN` §1). Only recon∥entropy overlap survives, not entropy∥entropy.

**Risk: HIGH.** 3 architectural blockers (§1e: surface refcount, per-frame CDF
buffers, early motion-field publish) + per-in-flight-frame state pool + output
reorder + ~5,000-line-class machinery, all under a byte-exactness-across-thread-
counts gate.

**Effort: LARGE.**

**Verdict.** Real but **niche**: it wins **single-stream latency** for one
high-bitrate/4K60 stream that must exceed a single core after the wavefront's
Amdahl bound — and **nothing** for aggregate throughput. Pursue only if that
single-stream latency case is an explicit product requirement. If pursued, the
minimal-viable shape is **not** full n_fc frame contexts but a **2-stage pipeline**
reusing the wavefront: entropy(N+1) ∥ recon-wavefront(N), with per-frame CDF
snapshot + surface refcount. Phase-1 would be the surface-refcount + frame-state
pool + the existing `TestWavefrontByteIdenticalToSingleWorker`-style differential
extended across frame-thread counts.

---

### BET 3 — Single-thread ENCODE cpu reduction

**Thesis / data.** Per `PERF_PLAN` E4/E8: search-breadth reduction is **exhausted**
on realC (disallow-16×16 gate landed); the residual ~5× SVT cpu is work-VOLUME
(pure-Go-vs-SIMD-C + heavier search) — the SIMD-C floor. The one identified
reducible structural chunk is the **default multi-tile LF planner (~23% cum,
scalar)** — the *same* shared code as Bet 1's LF build, and the wavefront path
still uses the parallel banded **edge-list** apply rather than a parallel **mask**
apply (`PERF_PLAN` D2a follow-up (a)).

**Yield / risk / effort.** Folding the encoder's default + wavefront LF paths
onto Bet 1's decode-time mask build gives the encoder the same LF cpu cut
(the single-tile serial mask apply already landed −11.8%). No standalone
big-bet content beyond that; the rest is the SIMD-C floor.

**Verdict.** **Fold into Bet 1** (shared LF code). Not a separate bet.

---

### BET 4 — amd64 native-x86 cpu-time VALIDATION (flagged out-of-scope)

The user scoped out "the amd64 tail," but the **largest unrealized cpu/frame win
on the arch that actually deploys** is still unmeasured: all Bet B AVX2 work is
byte-exact but **perf-unvalidated** (Rosetta emulates VEX). A native-x86 corpus
decode + encode run is small infra effort and could **flip the entire ranking**
(if x86 decode is 6–10× behind as suspected, closing it is worth more than any
arm64 single-thread grind). Recommend running it as a **measurement gate**
alongside Bet 1, even though implementing further amd64 kernels is out of scope.

---

## 3. Ranking — (yield × certainty) / effort, cpu-time weighted

| # | Bet | Yield | Certainty | Effort | Score |
|---|---|---|---|---|---|
| **1** | **Single-thread decode cpu reduction (LF bitmask build + geometry + pure-Go apply)** | Medium-high, **dual latency+throughput**, dual decode+encode | **High** (dav1d-grounded, byte-exact gate) | Medium | **Highest** |
| 2 | Frame-parallel decode threading | **~0 aggregate**; +1.3–1.6× single-stream latency only | Medium | Large/high-risk | Low |
| 3 | Single-thread encode cpu (LF-mask MT apply) | Medium (shared with #1) | Medium | Small | Fold into #1 |
| 4 | amd64 native-x86 validation | High signal (could reorder the board) | High | Small (infra) | Run as a gate |

**Rationale.** Only Bets 1/3/4 reduce cpu/frame — the metric that moves both
competitive axes. Bet 1 is the highest-certainty, dav1d-grounded, byte-exact,
dual-benefit cpu reducer, and the fresh profile proves its target is the current
#1 reducible hotspot. Bet 2 spends the most effort for the least cpu-time
movement (zero aggregate), and its single-stream slice is already partly captured
by the reachable wavefront; keep it in reserve for a specific product need.

---

## 4. Recommended sequence

1. **Bet 1, Phase 1 (START NOW).** Full-corpus `GOMAXPROCS=1` profile to lock the
   LF-build target (reconcile 20.6% vs D2a's 10.7%); decode-time mask build
   emitting the existing edge-list from a bit-scan, gated behind the scalar
   builder, byte-identical on all 226 vectors. No apply/plan swap. Zero decode-
   output risk. (Concurrently: run the **native-x86 validation gate**, Bet 4.)
2. **Bet 1, Phases 2–4.** Swap planner→mask build; delete scalar sweep; extend to
   encoder default+wavefront LF paths (Bet 3 folds in here); geometry-recompute
   cache; NEON-port `filter4EdgePureGo`. Each byte-exact-gated.
3. **Re-measure** single-thread decode+encode cpu on the corpus; record the
   recovered fraction against the ~2–2.5× irreducible floor.
4. **Bet 2 only if** a single-stream 4K/high-bitrate latency requirement is
   confirmed: build the 2-stage entropy(N+1)∥recon(N) pipeline (surface refcount
   + per-frame CDF snapshot + frame-state pool), byte-exact across frame-thread
   counts via the wavefront-differential harness extended to frame threads.

### Phase-1 acceptance (Bet 1)
- Full-corpus `GOMAXPROCS=1` profile committed as the target baseline.
- Decode-time mask build reproduces the exact ordered+run-merged edge-list
  (bit-scan), gated behind the scalar builder; differential assert green.
- `make dryrun-extended` strict-MD5 0-FAIL (byte-identical, all 226).
- `-race` clean; `scripts/check_allocs.sh` 0-alloc.
- Measured `GOMAXPROCS=1` corpus decode-cpu delta reported (expected neutral in
  phase-1 since the swap is deferred; the win lands in phase-2).
