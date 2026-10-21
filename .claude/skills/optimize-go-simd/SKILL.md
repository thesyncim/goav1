---
name: optimize-go-simd
description: Optimize Go simd/archsimd kernels using differential tests, native benchmarks, and generated-code evidence. Use for SIMD implementation, tuning, and performance regressions.
---

# Optimize Go SIMD

Optimize the representative public operation under its correctness and allocation contract. A faster microbenchmark is evidence only when the intended work and dispatch remain live.

## Workflow

1. **Pin the contract.** Record compiler revision, CPU/OS, experiment flags, selected backend, sizes, aliasing, tails, and exactness or numerical tolerance. Check APIs in that compiler. Choose portable `simd` when its operations fit; use `archsimd` for required instructions or measured benefit. Preserve build fallbacks and guard actual CPU features.
2. **Find the limiting cost.** Profile the real workload. Distinguish dependency latency, execution throughput, memory/cache traffic, branches, and dispatch/setup. Use hardware counters when available; otherwise label inference from disassembly and controlled size/variant experiments. Consider algorithm and data layout early when they explain the bottleneck.
3. **Compare plausible designs.** Keep a scalar oracle and a strong available baseline. Explore a small, justified set of widths, accumulator counts, tiles, layouts, or thresholds. Change one explanatory dimension at a time; evaluate interactions after narrowing. Counts of registers, spills, or instructions are diagnostics, never universal acceptance rules.
4. **Prove correctness and live work.** Differentially test the public path, short/vector/tail lengths, allowed overlaps and alignment, and numeric extremes. For floats, include cancellation, NaN/Inf, signed zero, and FMA/reduction-order effects as required. Test runtime emulation separately from experiment-off builds. Use observable outputs and `testing.B.Loop`; keep setup/reset comparable. A benchmark-only change to specialization or dispatch cannot establish a production win.
5. **Measure and explain.** Benchmark natively, serializing competing experiments. Compare repeated A/B runs statistically, including small inputs and representative working sets. Measure public-call allocations; inspect the selected hot body and callers for calls, spills, bounds checks, and dependency chains. Check whether the predicted bottleneck actually improved. Use safe indexing first; justify `unsafe` with measured benefit and bounds, alignment, lifetime, alias, and tail proofs.
6. **Keep supported gains.** Use the strongest compatible baseline for acceptance. Retest finalists on fresh runs to limit selection bias. Report latency/throughput, uncertainty, allocations, correctness, actual dispatch, tradeoffs, and tested hardware. Keep winners and thresholds scoped to the measured CPU, compiler, and workload; remeasure on other targets. Reject meaningless or workload-specific wins presented as general gains. Stop when remaining differences are noise or outside the task's scope; record unresolved limits.

## Read only the reference needed

Do not preload all references or upstream documentation.

- [API and dispatch](references/go-simd-apis.md): choosing/migrating APIs, portable widths, masks, tails, or runtime fallback tests.
- [Measurement](references/measurement.md): designing benchmark comparisons, diagnosing bottlenecks, or accepting small gains.
- [Code generation](references/diagnostic-patterns.md): investigating observed bounds checks, spills, broadcasts, or reductions.
- [amd64 tiling](references/amd64-tiling.md): tuning fixed-width x86 tiles and feature-specific variants.

Read the relevant section, then inspect the pinned source or targeted tool output. Do not load generated API catalogs, full disassemblies, or benchmark logs when a symbol or summary answers the question.
