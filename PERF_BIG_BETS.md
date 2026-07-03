# goav1 — strategic big bets (structural, needle-moving)

Author: strategic-architect pass, 2026-07-03. Companion to `PERF_PLAN.md`.
This file plans the **structural** levers that change the competitive position.
It deliberately does NOT rehash the incremental tail already tracked in
`PERF_PLAN.md` (remaining single-kernel NEON ports, loop-filter multi-tile /
wavefront follow-ups, HBD kernels, controller tuning). Those are the "easy"
grind. This is about the bets that move the whole board.

All numbers below were **measured this session on the arm64 M4 Max host**, not
copied from stale baselines. Commands and raw outputs are cited inline.

---

## 0. Executive summary — the one bet to start now

> STATUS 2026-07-03: BET B STARTED. Phase-1 LANDED (a54b1273): inverse DCT32/64
> four-lane AVX2 kernels (the biggest amd64 transform gap) + the avx2gen
> generator (tools/itxgen/avx2gen). Byte-exact (GOARCH=amd64 differential
> executes AVX2 under Rosetta, PASS; arm64 conformance unaffected). PERF NOT
> validated here — Rosetta emulates VEX as multiple NEON ops so amd64 ns/op is
> non-representative; the win is on NATIVE x86 (validate there).
> Phase-2 LANDED (a5733686 + 4947003e): 10 encoder SAD AVX2 kernels — the
> motion-search core (10 of 13 SAD slots were pure-Go on x86): sad{8,16,32}x4 +
> x4Step4 fan-out, sad32x32, sad16/32 dual-stride, sad8x8 compound-avg. Byte-
> exact (GOARCH=amd64 differential, PASS). NEXT amd64 slices: (i) the smaller
> encoder arm64-only kernels (rdstats, metric, hme_quarter, pframe avg/intpro,
> scale_nearest) → AVX2 [cleaner batch, do next]; (ii) FORWARD DCT16x16/DCT32x32
> AVX2 — big standalone kernels (NEON fdct32 ~9871 lines), high bug risk, needs
> a dedicated fast-differential slice, NOT crammed in; (iii) forward ADST/hybrid;
> (iv) avx2gen register-residency pass, all validated on NATIVE x86.
>
> MEASUREMENT RULE (user, 2026-07-03): judge by CPU TIME (cpu_total core-seconds),
> not ns/op wall time — wall hides parallelism (goav1 wins encoder fps by
> burning ~3.6x SVT's cpu) and Rosetta emulation. Example this session: the
> multithread encoder loop-filter mask apply looked wall-neutral/noisy (70-83
> fps under load) but is a clean −11% cpu_total (3.59→3.19s realC MT14). See
> memory cpu-time-not-nsop.

**START: BET B — amd64 (AVX2) critical-path kernel parity.**

**Thesis (data-grounded).** Every headline number in this campaign is measured
NEON-native on Apple Silicon. The dominant real deployment target — x86 servers,
WebRTC SFUs, cloud transcoders — runs **amd64**, and an audit of the kernel
surface shows amd64 is only **~45% at NEON parity overall and ~25% on the hot
decode/encode critical path.** The single worst gap is the transform package:
arm64 has 13 hand-written `.s` kernels (including the wave-3 DCT32/DCT64 col4 +
row4 inverse kernels that delivered ~10-13% e2e decode), while amd64 has **2**
(`dct_avx2`, `fdct_avx2`) covering only DCT8/16 — everything DCT32/64 and all
forward transforms fall back to pure-Go on x86 (verified in
`internal/av1/transform/colpass_dispatch_amd64.go`: only `DCT8Col2`/`DCT16Col2`
are bound; `DCT32Col4`/`DCT64Col4` are absent). Encoder SAD on amd64 is
SSE2-only for 8x8/16x16 — no 32/64/4-wide. Net: **goav1 on x86 is materially
slower than the ~4.4x-behind arm64 numbers suggest — the true x86 decode gap is
plausibly 6-10x dav1d, and the encoder loses its SAD/transform acceleration.**
This gap is invisible on the dev host and is the largest *unmeasured* real-world
regression in the project.

**Why this is the #1 bet.** It scores highest on (yield × certainty) / effort:
- **Certainty is near-total.** These are *mechanical ports of already-proven
  wins* — the arm64 kernels exist, are byte-exact, and their speedups are
  measured (DCT64 col went 23.7x on NEON). The generator/template
  (`tools/itxgen/`) was explicitly built for the AVX2 wave, and dav1d ships the
  reference asm (`third_party/upstream/dav1d/src/x86/itx_avx2.asm`,
  `itx16_avx2.asm`, `cdef_avx2.asm`, etc.).
- **No architecture risk.** Zero changes to control flow, threading, or
  bit-exact state — only per-block SIMD swaps behind the existing static
  build-tag dispatch. Byte-exactness is verified by the existing `GOARCH=amd64`
  differential tests (which call kernels directly, per the Rosetta CPUID caveat)
  plus a `GOARCH=amd64` corpus MD5 run.
- **Broad blast radius.** It fixes both the decoder AND the encoder on the arch
  that actually deploys, and it *multiplies* the value of every other bet
  (frame-threading on fast per-core kernels beats frame-threading on pure-Go).

**Phase-1 first slice (bounded, verifiable, ~1 session).** Port the **transform
AVX2 wave**: the inverse DCT32 and DCT64 col4 + row4 kernels — the single
biggest amd64 gap and a direct mirror of the arm64 wave-3 that moved e2e decode
~10-13%. Wire them into `colpass_dispatch_amd64.go` / `rowpass_dispatch_amd64.go`
(currently DCT8/16 only), ground the lane shapes in dav1d `itx_avx2.asm`, verify
byte-exact via the `GOARCH=amd64` differential tests + a Rosetta corpus MD5, and
report the Rosetta before/after delta on `p720_inter_q32.ivf` (relative delta is
valid even though Rosetta inflates absolute time). Success criterion: identical
MD5, measurable amd64 decode improvement on transform-heavy clips.

**Honesty up front (the irreducible part).** Single-thread pure-Go decode will
**not** reach dav1d parity — dav1d is fully hand-tuned asm across the entire
pipeline and the gap is distributed, not one hotspot. The winnable definitions of
"beat dav1d" are: (1) don't be catastrophically behind on the arch that deploys
(**Bet B**), and (2) unlock cores for single-stream throughput (**Bet A**,
frame-threading). Both are real; neither leapfrogs dav1d per-core. Plan the
program around that truth.

---

## 1. Measurement snapshot (this session, arm64 M4 Max, idle)

### 1a. Decode gap — CONFIRMED 4.39x single-thread
`GOAV1_BENCH_CORPUS=1 ... TestCrossDecoderCorpus` aggregate (864 frames, 1 worker):

| decoder | total ms | fps | vs goav1 |
|---|---|---|---|
| goav1 | 3632.9 | 237.8 | 1.00x |
| aomdec | 1320.4 | 654.3 | 2.75x |
| dav1d | **827.9** | **1043.6** | **4.39x faster** |

Matches the tracked ~4.34x. The gap is distributed pure-Go-vs-SIMD-C.

### 1b. Decode threading — TILE-ONLY, no frame parallelism
Audit of `internal/av1/decoder/work.go`, `internal/av1/threading/`:
- **Frame level: strictly serial.** One `FrameWorkState` per decoder; a frame
  must fully complete before the next begins. No multi-frame state pool, no
  reference-buffer refcount pool, no dependency graph.
- **Tile level: parallel** via `threading.Pool` (N worker goroutines,
  `PlanDecoderTileWork`/`ExecuteDecoderTileWork`). Ceiling = tile count. **Common
  streaming/WebRTC content is single-tile → zero decode parallelism today.**
- **Intra-tile SB-row wavefront recon exists but is gated behind the
  `GOAV1_DEFER_RECON` env var and documented as a "test harness"**
  (`internal/av1/threading/tile_residual.go:344`) — NOT active in production
  decode. (The ~3.2x wavefront win in memory is the ENCODER path.)
- Corpus bench uses `threading.NewPool(1)` deliberately for a fair 1-thread
  comparison (`cross_decoder_corpus_bench_test.go:497`).
- dav1d's headline throughput is **frame-threaded** (`src/thread_task.c`:
  `n_fc` frame contexts × `n_tc` tile threads, task pipeline per frame). goav1
  has no equivalent.

### 1c. amd64 vs arm64 SIMD parity — ~25% on the critical path
Audit of `internal/av1/*/{*_arm64.s,*_amd64.s,*_dispatch_*.go}`. `.s` counts:

| pkg | arm64.s | amd64.s | critical gap on amd64 |
|---|---|---|---|
| **transform** | **13** | **2** | DCT32/64 col4+row4, FDCT16/32, fhybrid ADST, invglue, txb → pure-Go |
| motion | 5 | 2 | dotprod/i8mm convolve arm64-only (feature-gated) |
| **encoder (SAD)** | (many) | 2 (SSE2) | 32x32, 64x64, all 4-wide, Dual, CompoundAvg → no AVX2 |
| loopfilter | 2 | 1 | filter_wide (14-tap) arm64-only |
| restoration | 2 | 1 | selfguided arm64-only |
| cdef | 2 | 1 | direction-find arm64-only |
| quantize | 3 | 2 | ~ok |
| superres/filmgrain/prediction | 1/1/2 | 1/1/1 | ~ok |

Overall ~45% parity; **~25% on transform+motion+SAD (the hot path).**

### 1d. Encoder per-frame work — LF edge planning still scalar on the default path
`GOMAXPROCS=1` CPU profile of `BenchmarkVideoEncoderRealC1080p` (default
multi-tile, 45.8 ms/op):
- `encodePBlock` 43% cum (mandatory leaf: `convolve2D8I8MM` ~10%, residual,
  quant, txb — already NEON, the pure-Go-vs-SIMD floor).
- **Scalar loop-filter edge planning ~23% cum**
  (`loopFilterPostFilterPlanTrustedSweep` 23.7%,
  `frameWorkAppendLoopFilterLumaEdges` 17%). The landed bitmask mask-apply only
  covers the **single-tile** path; the default 32-col path still runs the scalar
  edge-list. This is a real reducible chunk but is already tracked as the
  E1-followup / multi-tile-mask work (`PERF_PLAN.md` D2a) — noted, not re-planned.

---

## 2. Per-bet analysis

### BET A — Frame-parallel decode threading

**Thesis.** dav1d's real deployment advantage is not its single-thread asm; it's
**frame-threading** that scales decode near-linearly to N cores
(`src/thread_task.c`). goav1 decode has **no** frame parallelism and only
tile-parallelism, which is worthless on the common single-tile stream. For
single-stream high-fps decode (one 4K60 / high-bitrate stream that must exceed a
single core's throughput), goav1 is hard-capped at 1 core today while dav1d uses
all of them. Building frame-threading unlocks the 12 P-cores for that case.

**Data.** §1b: strictly serial frames, single-tile ⇒ 1-core-bound. §1a: 4.39x
per-core deficit. On 12 P-cores a near-linear frame-thread scaler yields ~6-9x
single-stream throughput.

**Estimated yield (honest).** For **single-stream** decode: up to ~6-9x
throughput on 12 cores → can *exceed* dav1d's single-thread 1043 fps and
approach/meet dav1d's own multi-thread number on reference-chained content
(where dav1d's frame-threading is itself limited by ref dependencies). For the
**many-independent-streams** server case: **near-zero marginal yield** — you
already scale by decoding each stream on its own goroutine/core, and there the
4.39x per-core gap is simply preserved across cores (both decoders scale, the
ratio holds). So Bet A wins *single-stream latency-throughput*, not aggregate
many-stream throughput. This is the crucial honesty: frame-threading does **not**
let goav1 "beat" dav1d on aggregate per-core work — it removes the single-stream
core cap.

**Risk: HIGH (architectural).** Requires (a) a pool of `FrameWorkState` in
flight; (b) reference-frame buffer lifetime/refcounting (dav1d `RefCntBuffer`
analog) so an in-flight frame's refs aren't recycled; (c) a frame dependency
graph (frame N's MC waits on its decoded+filtered refs; N inherits N-1's adapted
CDF via `primary_ref_frame`); (d) overlap discipline — dav1d runs recon(N)
concurrently with entropy(N+1) via task types. Determinism is *favorable*: decode
output is fully bitstream-determined, so threading only reorders *when* work runs,
never the result — byte-exactness is directly checkable by the existing
strict-MD5 conformance run at N frame-threads.

**Effort: LARGE, multi-phase.** Phase 1 (bounded): introduce a frame-state pool +
ref refcount + a scheduler that runs frames concurrently **only when dependency
analysis proves independence** (else serial), gated behind a decoder frame-thread
count knob, defaulting to 1. Prove byte-exact on the full conformance suite at
2/4/8 frame-threads. Even if v1 overlaps only a few frames, it establishes the
architecture. Phases 2-3: recon(N)∥entropy(N+1) task pipelining; postfilter as a
separate task stage (dav1d shape).

**Verification.** `make dryrun-extended` (strict MD5) at frame-threads = 2/4/8
must be 0-FAIL and identical to serial; a race-detector pass over the scheduler;
corpus MD5 unchanged.

---

### BET B — amd64 (AVX2) critical-path kernel parity  ★ RECOMMENDED FIRST

**Thesis.** The deployment arch (x86) runs ~25% of the hot kernel surface in
pure-Go because the AVX2 wave never followed the NEON wave. Closing it recovers
the *already-proven* speedups on the arch that actually ships.

**Data.** §1c: transform 13→2, SAD 32/64/4-wide missing, selfguided/filter_wide/
cdef-direction arm64-only. `colpass_dispatch_amd64.go` binds only DCT8/16. The
arm64 equivalents' wins are measured (DCT64 col 23.7x; wave-3 e2e ~10-13%).

**Estimated yield.** Decode on amd64: recovers most of the ~10-13% transform e2e
win plus the loopfilter-wide / selfguided / cdef-direction deltas — plausibly
**1.5-2.5x amd64 decode speedup**, i.e. moving x86 from ~6-10x-behind back toward
the arm64 ~4.4x line. Encoder on amd64: restores 32/64/4-wide SAD (ME is a large
share of encode) — recovers the SAD acceleration the arm64 encoder already enjoys.
This is the difference between goav1 being *usable* vs *uncompetitive* on x86.

**Risk: MEDIUM (volume, not novelty).** AVX2 is 256-bit vs NEON 128-bit, so
kernels are re-derived, not transliterated — but dav1d ships the exact AVX2
reference (`src/x86/itx_avx2.asm`, `cdef_avx2.asm`, `looprestoration_avx2.asm`,
`loopfilter_avx2.asm`) and `tools/itxgen/` is the generator. Byte-exactness is
mechanically checkable.

**Effort: MEDIUM, highly parallelizable** across independent kernels (2-agent
default fits: one on transform, one on SAD). Phases: (1) transform DCT32/64
col4+row4; (2) forward transforms FDCT16/32 + fhybrid (encoder); (3) SAD
32/64/4-wide/Dual; (4) filter_wide + selfguided + cdef-direction.

**Verification.** `GOARCH=amd64 go test` differential tests call kernels directly
(Rosetta reports AVX2 false — tests must not skip on `cpu.Detected`, per
PERF_PLAN §2); each kernel's output must match its pure-Go reference bit-for-bit,
then a `GOARCH=amd64` corpus MD5 confirms whole-pipeline byte-exactness. Report
Rosetta before/after relative deltas (absolute is inflated, delta is valid).

---

### BET C — Encoder raw-fps to SVT (structural remainder only)

**Status: mostly identified / in-flight; small structural remainder.** The two
big structural encoder facts are already resolved in `PERF_PLAN.md`: (1)
entropy-write ∥ decision overlap is a **proven dead-end** (EC(N)→MD(N+1)
serialization is identical in SVT p12); (2) temporal-layer L1T2 frame pipelining
is the real raw-fps lever and **landed opt-in** (up to 1.6x). The genuinely
structural remainder here is:
- **amd64 encoder SAD parity** — folds into Bet B; ME loses acceleration on x86.
- **Generalize the landed pipelining** (golden-enabled overlap via base
  compute/commit split; deeper pipeline depth) — tracked as the pipelining
  follow-up, incremental, not re-planned here.
- The default multi-tile LF scalar cost (§1d, 23% cum) — tracked as D2a
  multi-tile mask apply; the mask apply currently loses to the edge-list on
  sparse multi-tile content, so it's a real but bounded grind, not a new bet.

**Yield / risk / effort.** The only *new* structural encoder lever is the amd64
SAD parity, which is subsumed by Bet B. Everything else is the tracked tail.
**Conclusion: no standalone encoder big-bet beyond Bet B's amd64 work + the
already-planned pipelining generalization.** The per-frame work gap vs SVT (~5x)
is the pure-Go-vs-SIMD-C floor (E8) — irreducible without SIMD-everywhere, and
goav1 already wins quality on every clip and wall-fps at deployment thread counts.

---

### The irreducible gap (call it out)

- **Single-thread decode parity with dav1d: not achievable in pure Go.** dav1d is
  hand-asm end-to-end; the gap is distributed. Expect a pure-Go floor around
  ~2-2.5x even with full SIMD parity + zero-alloc.
- **Aggregate many-stream throughput: governed by per-core work**, which no
  amount of threading changes. Frame-threading (Bet A) helps *single-stream*
  only.
- **Encoder per-frame work ~5x SVT: SIMD-C floor.** goav1 already wins quality
  and hits deployment-thread wall-fps; closing raw single-thread CPU further is
  diminishing returns.

Being honest: the campaign's winnable goals are **amd64 parity** (don't lose the
deployment arch), **single-stream decode scaling** (frame-threading), and
**holding the encoder quality+wall-fps lead** — not a per-core decode leapfrog.

---

## 3. Ranking — (yield × certainty) / effort

| # | Bet | Yield | Certainty | Effort | Score | Note |
|---|---|---|---|---|---|---|
| **1** | **B: amd64 AVX2 critical-path parity** | High (fixes the deploy arch, decode+encode) | **Very high** (port proven wins, byte-exact verifiable) | Medium, parallelizable | **Highest** | Start here |
| 2 | A: frame-parallel decode threading | High for single-stream; ~0 many-stream | Medium (architectural; parity not leapfrog) | Large | High-but-risky | Do after per-core kernels are strong |
| 3 | C: encoder raw-fps remainder | Low-medium | Medium | Small (mostly tracked) | Low | Subsumed by B + tracked pipelining |

Rationale: Bet B dominates on certainty and blast radius, is independent, and
strengthens every core before Bet A parallelizes them. Bet A is the true
structural reframe but is higher-risk and its honest yield is single-stream only.
Bet C has no standalone big-bet content beyond B.

---

## 4. Recommended sequence

1. **Bet B, Phase 1 — transform AVX2 wave (START NOW).** DCT32/64 col4+row4
   inverse → `colpass_dispatch_amd64.go` / `rowpass_dispatch_amd64.go`. Ground in
   dav1d `itx_avx2.asm`; use `tools/itxgen/`. Verify: `GOARCH=amd64` differential
   tests (call kernels directly) + `GOARCH=amd64` corpus MD5 identical. Report
   Rosetta before/after on `p720_inter_q32.ivf`. **Bounded, ~1 session, zero
   arch risk, recovers the biggest x86 decode gap.**
2. **Bet B, Phases 2-4 (parallel, 2-agent default).** FDCT16/32 + fhybrid
   (encoder); SAD 32/64/4-wide/Dual (encoder ME); filter_wide + selfguided +
   cdef-direction (decode postfilter). Each byte-exact-gated.
3. **Re-measure amd64** with a Rosetta corpus run; establish the honest x86
   baseline (currently unknown) and quantify the recovered gap.
4. **Bet A, Phase 1 — frame-thread scaffolding** (frame-state pool + ref
   refcount + independence-gated scheduler, default 1 thread, byte-exact at 2/4/8
   frame-threads). Only after per-core kernels are strong on both arches, so the
   scaled cores each run fast.
5. **Bet A, Phases 2-3** — recon∥entropy pipelining and postfilter task stage,
   dav1d `thread_task.c` shape.

### Phase-1 acceptance (Bet B transform wave)
- `GOARCH=amd64 go build ./...` green; kernel differential tests bit-exact vs
  pure-Go reference.
- `GOARCH=amd64` corpus MD5 unchanged (byte-exact whole pipeline).
- arm64 paths untouched → `make dryrun-extended` still 0-FAIL.
- `scripts/check_allocs.sh` green (static build-tag dispatch, no func-pointer
  escape — PERF_PLAN §0.4).
- Measured Rosetta decode delta reported on a transform-heavy clip.
