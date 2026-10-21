# Measure the bottleneck and the gain

## Establish comparable work

Record toolchain/build flags, CPU, backend/width, OS, GOMAXPROCS, input distribution, logical bytes/elements, and scratch/alias contracts. Measure the public operation including dispatch, packing, and reduction; separate kernel timing only to explain costs. Compare against a credible tuned library or assembly implementation when available, recording foreign-call overhead and semantic differences. Preserve a simple independent correctness oracle even when it is slower.

Keep mutation/reset, output reuse, and working sets equivalent. Use realistic warm-cache and larger streaming cases; do not label a case DRAM-bound solely from its size. Include small calls and vector-boundary tails. Use varying prepared inputs where constant folding or repeatedly unchanged state could remove work.

## Diagnose before selecting changes

- Dependency-limited: compare independent accumulator counts and inspect the dependency chain. Latency and throughput inform candidates; no fixed accumulator count is universally sufficient.
- Execution-limited: inspect instruction mix, load/store demand, and scheduling; use CPU-specific counters or a suitable static model when available.
- Memory/cache-limited: examine bytes moved, reuse, layout, fusion, packing amortization, and concurrency. Layout may precede instruction tuning.
- Front-end/branch-limited: investigate code footprint and unpredictable branches before further unrolling.
- Dispatch/setup-limited: inspect public wrappers and small calls before increasing vector width.

Counter availability varies by OS and CPU. State whether conclusions are measured, inferred, or untested. Static throughput models cannot establish runtime performance. Intel's [top-down methodology](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-0/general-exploration-analysis.html) is one platform-specific example.

## Repeated comparisons

Serialize timed runs; keep compilation, profilers, and other experiments out of their measurement windows. Record unavoidable contention, power/thermal state, core migration, and frequency variation. Use OS-supported controls when appropriate without silently changing persistent host settings. Stable conditions must still represent deployment. See [benchmark variance guidance](https://google.github.io/benchmark/reducing_variance.html).

Use interleaved or randomized A/B process order and multiple independent runs. The Go [benchstat guide](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) recommends at least ten samples; increase evidence for close results rather than repeatedly testing until significance appears. Report effect sizes and uncertainty, not only p-values. Preserve raw results and metadata outside the skill.

Screen a bounded set of hypotheses, then rerun finalists with fresh measurements. Tune thresholds on one set of sizes and check unseen sizes/distributions. Explain material regressions and code-size/maintenance costs. A few spills can be profitable if the whole operation improves; fewer spills or instructions alone is not success.
