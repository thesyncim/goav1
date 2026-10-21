# Fixed-width amd64 tuning

Use the selected ISA and compiler's register budget. ARM64 tile shapes do not transfer automatically to amd64, and amd64 widths/features do not share one budget.

- Count simultaneously live accumulators, sources, and temporaries. Inspect stack stores/reloads inside the loop, not just frame size. Compare smaller tiles, different schedules, and additional passes; include reread/cache costs.
- Compare supported x4/x8/x16 candidates at representative sizes. Guard each instruction's actual requirements: x8 floating arithmetic may require AVX while fused `MulAdd` requires FMA. A wider vector does not prove an optional operation is available. Check the pinned [archsimd API](https://pkg.go.dev/simd/archsimd).
- Choose accumulator counts and unrolling from dependency latency, throughput, and register pressure, then measure. Eight accumulators are not a general saturation threshold. Spills are a cost to investigate, not a veto on a faster candidate.
- Consider in-register reductions or combining halves before scalar stores when supported. Validate reduction order, NaN/Inf/zero behavior, scratch cost, and the complete public operation.
- Grouped shuffles operate within sublanes; test lane boundaries before changing width. Inspect full-register permutations separately.
- If traffic dominates, consider layout, loop fusion, and packing before further tile tuning. Charge layout conversion to the operation unless callers demonstrably amortize it.
- Prefer slice loads when they compile as well as pointer loads. Introduce unsafe only with the main skill's proof and measured benefit.

Record width/tile/unroll, correctness, spill evidence, ns/op or throughput, public allocations, and tested CPU for each finalist. Timings under translation (such as Rosetta) describe that environment, not native x86 performance. Do not carry numerical speedups or a worked example from an unrelated kernel into a new task as expected results.
