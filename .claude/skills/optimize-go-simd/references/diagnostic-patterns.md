# Inspect the selected hot body

Build with the actual flags. Use `go tool nm` to identify selected/specialized symbols, then `go tool objdump -s 'symbol'` on the test binary. Inspect callers and the complete loop; an opcode histogram alone cannot show dependencies, trip counts, or dispatch. Compiler escape/inlining reports explain hypotheses, not measured allocations.

| Observed cost | Candidate and verification |
|---|---|
| Rebuilt broadcasts or shift vectors | Hoist invariant setup or retain compile-time immediates; check signed, oversized, and negative shift semantics. |
| Calls, aggregate traffic, or vector spills | Compare inlining, separate vector values, scheduling, and tile sizes; include code footprint and register-pressure effects. |
| Bounds-check paths | Establish all slice lengths and use overflow-safe loop bounds; inspect whether checks remain in the executed body. A full-slice expression can expose length/capacity relationships and remove address masking as well as bounds checks. Prefer safe indexing before pointer code. |
| Table/broadcast overhead | Compare local vectors with precomputed tables, including loads and cache costs. |
| Long reduction chain | Sweep a few independent accumulator counts; validate reassociation/FMA tolerance and short-call overhead. |
| Compare/select or arithmetic sequences | Check compiler fusion before hand rewriting. Preserve overflow, saturation, mask, and exceptional-value behavior. |

On ARM64, `VDUP`, `VSSHL`/`VUSHL`, vector stack traffic, and calls can help orient inspection; emitted mnemonics vary by toolchain. Neither their presence nor their count proves a bottleneck. Sampled time attributed to a load does not by itself establish a memory-bandwidth limit. Inspect before and after, correlate with timing, and keep only changes whose benefit survives public-path measurement.
