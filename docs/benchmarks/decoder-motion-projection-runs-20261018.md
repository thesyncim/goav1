# Reuse equal motion-vector projections — 2026-10-18

This decoder-only change projects each horizontal run of identical reference/MV/valid entries once, then preserves the original destination writes. It follows `load_tmvs_c` in pinned dav1d `b546257f770768b2c88258c533da38b91a06f737`, `src/refmvs.c:725–758`. The Go change is 31 insertions /20 deletions in `internal/av1/tile/motion_field.go`.

`ReferenceMVFrame.MarkBlockPtr` replicates the chosen block entry across the half-MI grid. Previously, `TemporalMotionField.ProjectReferenceFrame` repeated reference lookup and vector scale/round/clamp/offset calculations for every entry. The run loop computes these once per equal run. Equality includes validity; runs end before row padding. It retains ascending row/column writes, per-position 8-cell clipping, frame edges, backward signs, zero-vector origin stores, validation/error order, and last-writer collisions. No API, layout, allocation contract, rounding, or quality setting changes.

The current PR13 warm inter profile attributes 2.93% cumulative CPU to the full projection function; this is overlapping diagnostic attribution, not a prediction of removable time.

| Warm public case | Paired geometric-mean latency saved | Pair savings | Faster pairs |
|---|---:|---|---:|
| p720_inter_q32 | 1.7463% | −0.1181%, +2.4586%, +2.8718% | 2/3 |
| p360_intra_q32 control | 0.7654% | +0.7974%, +0.8326%, +0.6661% | 3/3 |

Short screen: three sequential paired rounds, 15 complete 48-frame decodes per row, one worker, GOMAXPROCS=1, GOGC=100, Go1.27.0-X:simd, Darwin arm64/v8.0, Apple M4 Max. Every row is retained in the [raw evidence](decoder-motion-projection-runs-20261018.json), with role order and logs' hashes. All12 rows report zero allocations/op (0–3B/op). Baseline is merged PR13 `b2eff253323a3610beab87ef51c200d2586c8dbc`, using its already-frozen binary; source and binary hashes are recorded and stayed unchanged during timing.

Focused existing scalar-oracle tests plus new run lengths1–41, validity/reference boundaries, clipped windows, poisoned padding, source immutability, and last-writer collisions passed (0.417s). The oracle retains per-entry traversal and independently rounded numeric projection. The candidate's public one-worker pixel gate then passed all18 clips /864 frames against independent aomdec/dav1d MD5 sidecars. Existing error/setup/zero-allocation assertions remain unchanged. Encoder paths do not call this decoder motion-field API, so no encoder test was added or run.

These are short-screen estimates, not a confidence interval or an aggregate decoder claim. The intra case is a control, not evidence of a direct intra optimization. PR13's previously measured ~2.71× cold full-corpus dav1d ratio is historical and predates this change; no new native ratio or additive gap improvement is claimed.

Earlier temporal-candidate variants were rejected and are not included: V2's six pairs showed inter+1.3369% but intra−2.2196%; the single smaller V3 iteration showed inter−1.2522% and intra+1.7579%. Their worktrees, binaries, oracle logs, and all36 rows remain in the local research artifacts. This patch changes the separate frame-projection pass.
