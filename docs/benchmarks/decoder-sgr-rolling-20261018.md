# SGR rolling intermediate work — 2026-10-18

The shared 8/10/12-bit SGR coefficient producer now accumulates sample sums and squared sums together in two row-sized running buffers, then writes only coefficients consumed by the filter. This removes the two full box-sum image passes and their clear/readback work. The U8 entry also omits five redundant clears. Existing scratch allocation and final dgd/filter/coefficient storage sizes remain unchanged. All implementation is Go.

Base: `5e21bb5c62827a3f5890cb89be3a3d17c217d540` (PR16). Reference: pinned dav1d `b546257f770768b2c88258c533da38b91a06f737`, `src/looprestoration_tmpl.c:419–535` row-local sum/squared-sum construction. Go retains its libaom coefficient, filtering and projection formulas exactly.

## Short public screen

Go 1.27.0, `GOEXPERIMENT=simd`, Darwin arm64/M4 Max, ARM64v8.0, one worker, GOMAXPROCS=1 / GOGC=100. Three paired rounds per case, 200 warm Reset+complete-clip decodes per row. Baseline has only the identical new test-only HBD benchmark wrapper added to clean main. All 18 rows are in the companion JSON; no exclusions, retries or pooling.

| Existing public fixture | Frames / SGR units | Paired GM latency saved | Pair savings | Wins |
|---|---:|---:|---|---:|
| 8-bit 4:4:4 CDEF/restoration | 4 / 9 | +0.4564% | +4.7449%, −0.9101%, −2.6167% | 1/3 |
| 12-bit 4:2:2 inter/CDEF/restoration | 8 / 162 | **+2.9881%** | +5.7576%, +1.8935%, +1.2515% | 3/3 |
| 10-bit superres/restoration control | 4 / 0 | +19.6909% | +0.0759%, +0.3806%, +47.9669% | 3/3 |

The HBD SGR case supports the narrow performance result. U8 is mixed and does not establish a gain. The control aggregate is dominated by one 8.448ms baseline row versus the usual 4.4–4.6ms; it is retained, with no causal control-gain claim. Every row has 0 alloc/op. These are short-screen estimates, with no confidence-interval, aggregate-decoder or current dav1d-gap claim.

The preceding clear-only variant failed: U8 −0.9029% (0/3), control −0.2638% (1/3). Its separate 12 rows are retained in the JSON. Rolling calculation was the single source-justified iteration; no unchanged-binary resampling followed.

## Parity and review

- The frozen original intermediate oracle uses independent brute-force box sums. It checks every consumed A/B coefficient for all parameter/radius choices, row phases, tiny/odd/max shapes and 8/10/12-bit samples. Existing deterministic SGR numeric/error/allocation tests pass; focused run 0.477s.
- Poisoned reusable scratch and in-place U8 output match the widened reference; scratch tails stay untouched. Only scratch workspace cells no longer consumed may retain older values. Validated public sample/error checks remain, and the old general calculation remains for nonstandard radii/shapes.
- The three-pixel source border contains every required radius 1/2 window. Running int32 sums have the same terms modulo 2^32; uint32 variance/product wrapping, bit-depth shifts, lookup, raw unscaled B, filter rounding, int16 projection wrap and clipping remain exact. SIMD consumers’ skipped-row weights are exactly zero.
- Existing public profile clips match libaom golden frame MD5s, including active 8/10/12-bit SGR; one single-worker 18-clip/864-frame MD5 gate passes. The changed restoration apply functions have no encoder call site; no encoder algorithm/byte path changed.

A stale explanatory comment was corrected after timing (producer writes consumed A/B coefficients rather than clearing them); executable source and tests are unchanged. Measured and final source hashes, binary identities, all rows, gate hashes and fixture reachability counts are recorded in the JSON. Diagnostic counters were used only in a temporary coverage overlay and are absent from timed/published source.
