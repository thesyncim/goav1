# Performance gap ledger

Last refreshed: 2026-07-16 on Apple M4 Max, Go 1.26, darwin/arm64.

## Policy

Single-thread decoder latency is the first acceptance gate. Multi-worker
throughput is measured only after a change wins or stays neutral with
`GOMAXPROCS=1` and `WithWorkers(1)`.

Production changes must build with the released Go toolchain. Experimental Go
SIMD is useful for code-generation research, but it is not a dependency of the
decoder. Architecture kernels use ordinary Go assembly with a pure-Go fallback
and differential tests.

Every retained decoder change must satisfy:

- byte-exact output against the strict frame MD5 oracles;
- zero steady-state allocations in the public decode benchmark;
- race, pure-Go, vet, and Linux amd64/arm64 compile gates;
- an end-to-end single-worker gain outside paired-run noise;
- a kill/revert decision when the public benchmark does not confirm a micro win.

Pinned comparison sources:

- dav1d `b546257f1dc1249c1c1e2ef58f9ba8ca2a48b28c`;
- libaom `047d8e271e0cbe586dcc553ad305a4ef1a428b95`;
- SVT-AV1 `c04f95dd154ef9c50b24b0c66b729a01ee6c0f12`.

## Retained safe points

The branch starts at `a28b7c8d`. Each row is independently committed and
pushed.

| Commit | Change | Single-thread relevance |
| --- | --- | --- |
| `d45f8d4a` | Route encoder reconstruction deblock through corrected masks | Quality/parity fix; no decoder claim |
| `2cbacca5` | Parallel CDEF postfilter bands | Neutral at one worker; multi-worker only |
| `601dd9fd` | Route public deblock through dav1d-style masks | Removes the old structural apply path |
| `3ad84745` | Parallel mask-deblock bands | Neutral at one worker; multi-worker only |
| `f91be83c` | Split clamped motion copies into fixed regions | Removes per-sample clamp work from the inter hot path |
| `1a1c45f9` | Add arm64 dot-product warp horizontal kernel | Production assembly; exact differential coverage |
| `43aaacf1` | Pool refcounted CDF frame contexts | Restores zero-allocation steady state |
| `7a4a2683` | Reuse tile workers for postfilter bands | Avoids worker lifecycle overhead; neutral at one worker |
| `e7780d32` | Fuse inter motion and interpolation-filter grid updates | Removes invalidate plus duplicate grid walks; about 0.3-0.7% p720 and 0.56% across the 18-clip exploratory corpus |
| `98263d63` | Collapse equal temporal-motion projection runs | 207.43 to 201.78 ms paired p720 median (2.72%); projection CPU 3.17% to 1.43% |

The controlled checkpoint before the last two changes measured approximately
11% lower p720 single-worker latency than `a28b7c8d` (256.59 to 228.50 ms).
The later changes have their own adjacent/pairwise gates above; absolute
cross-session timings are not combined because thermal bands moved materially.

## Current single-thread profile

Profile: public decoder, p720 inter q32, one worker, `GOMAXPROCS=1`, zero
steady-state allocations. Percentages are cumulative unless marked flat.

| Rank | Area | CPU | Status and next action |
| --- | --- | ---: | --- |
| 1 | Coefficient decode and reconstruction entry | 15.05% | The base-level and sign/Golomb spines already have arm64 kernels. Re-profile the EOB/P0-to-base handoff before attempting the planned fused entropy kernel; primitive-only assembly has repeatedly lost to inlined Go. |
| 2 | CDEF postfilter | 15.39% | Direction and all strength combinations already dispatch to production NEON. Secondary-only leaves are the best assembly experiment: 8-wide 3.95% flat plus 4-wide 2.02% flat. Specialize common strength/damping tuples and improve dual-row scheduling; do not add another dispatch layer without a kernel micro win. |
| 3 | Inter prediction | 18.42% | The largest leaves are already i8mm/dot-product assembly. Continue eliminating geometry and state recomputation around the kernels. |
| 4 | Loop-filter mask apply | 10.01% | Structural opportunity. Luma scan is 4.04%, chroma scan 3.87%, and the per-cell apply closure is 1.26% flat. Replace callback scanning plus repeated edge-fit validation with a trusted region/run scanner while preserving partial-frame padding rules. |
| 5 | Loop restoration | 7.23% | Wiener horizontal is 4.37% flat and vertical 1.85% flat, both already NEON. This is optimization of existing assembly, not a missing kernel. |
| 6 | Reference-MV stack | 4.88% | Temporal field setup is now 1.85%, down from 3.26%. `markGridInterMotionAndFilters` remains 1.60% flat. A packed splat seam modeled on dav1d `refmvs.S` is plausible, but the current split Go arrays make a direct port memory-layout hostile. |
| 7 | Block-loop root context load/store | 2.70% flat combined | Structural target: retain the hot root state in a compact working record and write back once. Validate exact state images, not only decoded pixels. |
| 8 | Warped-motion vertical pass | 1.09% flat | Genuine missing arm64 kernel, but its total ceiling is below the structural targets above. Implement only after the higher-ranked work or for architecture parity. |

## Kernel inventory conclusions

There is no single large production assembly hole in the current p720 profile.
The major pixel leaves already have arm64 implementations:

- CDEF direction, primary, secondary, and fused primary+secondary;
- Wiener horizontal and vertical restoration;
- 2-D inter convolution and compound blending using i8mm/dot-product paths;
- filter-14/filter-6 deblocking;
- coefficient base-level and sign/Golomb entropy loops;
- transform add and several inverse-transform leaves;
- warped-motion horizontal dot-product pass.

The most useful new assembly work is therefore either a specialization inside
an existing heavy family (CDEF secondary-only) or a layout-aware splat kernel
after the ref-MV state is packed. The missing warp vertical kernel is real but
cannot move total latency by more than its roughly 1.1% profile share.

## Structural divergences worth pursuing

### P1: trusted loop-filter region scanner

The mask geometry is already validated and bounded by the frame-wide mask
handle, yet each set cell passes through a function callback, length clamp,
coordinate scaling, width-fit loop, and run-builder callback. A direct scanner
can resolve levels and extend equal filter runs in the region loop itself.

Acceptance: exact mask-apply differential on odd crop sizes and subsampling
modes; strict 226-vector MD5; at least 0.7% public p720 improvement. Keep the
slow checked helpers for public/untrusted entry points.

### P2: CDEF secondary tuple specialization

Capture the real `(width, strength, damping)` distribution first. Add assembly
only for a tuple that dominates calls. Compare against the current dav1d-derived
secondary kernels at 4x4 and 8x8, including sentinel borders and every legal
direction. Require a material kernel win and at least 0.5% public p720 gain.

### P3: compact ref-MV splat state

dav1d's arm64 `splat_mv`, `save_tmvs`, and `load_tmvs` benefit from packed
records. The Go context currently writes motion, validity, block size, visited,
filter, and filter-valid arrays separately. First test a packed per-MI working
record or a write-combining helper; only then add a splat kernel. Preserve the
top/left public context arrays and compare complete context images.

### P4: coefficient EOB/base handoff

The coefficient walk remains large cumulatively, but its arm64 base loop is
already only 2.86% flat. The remaining credible assembly seam is to keep range
decoder state resident across EOB/P0 and base-level decoding. Use the existing
lockstep CDF/state differential harness and reject it if the extra kernel
surface loses to inlining.

## Rejected or held experiments

| Experiment | Decision |
| --- | --- |
| Exponential row replication in loop-filter map fill | Microbenchmark improved large fills, but public p720 regressed about 0.2-0.3%; rejected |
| Trailing-zero-only mask scan | Correct and slightly better in isolation, about 0.2% end-to-end and inside order noise; insufficient without direct scan/apply fusion |
| 16-row temporal-MV ring | Negative; full-frame run collapsing is the retained direction |
| Separate inter motion then filter marking | Replaced by the exact fused state update in `e7780d32` |
| Another CDEF primary/secondary fusion | Already present and dispatched; no missing fusion |
| Immediate warp-vertical assembly | Held: roughly 1.1% absolute ceiling in the current single-thread profile |

## Measurement notes

The ignored local 18-clip corpus has no committed `manifest.tsv`, so its
absolute throughput is exploratory rather than publishable. It is still useful
for byte-exact coverage and adjacent relative checks. Long runs on Apple silicon
showed large thermal/load bands; acceptance therefore uses alternating paired
public benchmarks and normalizes adjacent corpus passes against aomdec/dav1d
when their timings move in the same direction.
