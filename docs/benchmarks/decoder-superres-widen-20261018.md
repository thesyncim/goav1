# Exact ARM64 superres widening products

The final change removes four separate sign extensions per eight-tap superres dot product. Go `Int16x8.MulWidenLo` computes the same signed products directly in int32, then the existing addition, pairwise reduction, rounding and clipping remain unchanged. The high half uses the same existing lane extraction. AMD64 keeps its original helper; all scalar fallbacks, byte/uint16 plane staging, validation and ownership paths remain unchanged. This implementation is entirely Go/archsimd.

Compiler-output inspection on Darwin ARM64 shows the unrolled row changes from 36 sign extensions  + 18 int32 multiplies to 18 signed widening multiplies, with the same 144-byte frame and no vector stack spills. This is an instruction reduction, not a precision or quality change: every signed 16 × signed 16 product fits int32, and addition retains the original int32 semantics.

## Public decode screen

Base main `4e8cbc2ad0894099afb5990fe7a3e34e1782dcbb` (docs-only after PR17). Reuse the frozen PR17 binary because executable source is equivalent; its SGR comment correction is non-executable. Go 1.27.0 / GOEXPERIMENT=simd, Darwin arm64, Apple M4 Max, GOARM64=v8.0, GOMAXPROCS=1, GOGC=100. Three paired rounds, 200 operations per invocation, one Reset+full clip decode per operation; both orderings appear. Every row is retained. The rule was ≥0.5% paired geometric-mean latency saving with ≥2/3 wins in an affected case and no other case point regression >1%.

| Case | Paired latency saved | Pair savings | Warm allocations |
|---|---:|---|---:|
|8-bit 4:4:4 superres inter, 8 frames|2.5319%|+2.9198%,+2.9378%,+1.7331%|0|
|10-bit 4:4:4 superres inter, 8 frames|3.2925%|+2.0504%,+4.9547%,+2.8491%|0|
|No-superres 8-bit postfilter control, 4 frames|0.5980%|−0.8831%,+1.8706%,+0.7871%|0|

These are short-screen estimates, without a long confirmation or confidence intervals. The unchanged control is descriptive, not a causal gain claim. No fresh dav1d ratio was measured; the earlier PR13 ~2.71× cold-IVF snapshot is historical and does not describe this change.

Baseline SHA256: `b140cb9555e6a39fc7b687a8a74bbf9266cc03e153ee3042760ebc9e0133c31b`.
Candidate SHA256: `ee822414026a0fa7df468c41026a850c9d8268834d4b7c5e6fae0580a6d9f8e6`.

[Raw rows, orders, source/binary hashes and all variant summaries](decoder-superres-widen-20261018.json) include all 54 rows from the three distinct variants, without pooling. Complete logs and preserved experimental sources are under `/private/tmp/decoder-superres-direct-20261018`.

## Parity and review

The existing independent scalar superres tests cover 8/10/12-bit inputs, row/plane shapes, physical versus coded source spans, canaries, dispatch and zero allocations; the focused package passed in 0.336s. The final frozen public binary passed all 10 existing superres pixel-golden fixtures, including byte/HBD/inter/restoration cases. Compiler-output/source review confirms identical lane order, exact multiplication/addition, unchanged later round/clamp barriers and unchanged error/alias/output contracts. No encoder callsite uses this superres package. No extra codec test or broad validation matrix was introduced.

## Assessed staging variants, not included

Direct native HBD staging removal passed parity but showed −0.7429% inter and−0.1043% restoration estimates, with −3.4072% U8 control; unmerged. A bounded diagnosis found its original 16-bit bridge already used row memmove (~0.8% sampled CPU), no new hot helper, and unchanged row dispatch/phase/tail math. It did not establish a cause for the loss.

Direct byte I/O plus the widening-product fold passed scalar/state/golden checks, but U8 was +0.0665% and no-superres control −3.4526%; unmerged. HBD was 3/3 faster but its +11.8250% aggregate was dominated by one slow baseline, so that figure is not an intrinsic speed claim. Byte I/O introduces a zero-extension per overlapping output window; common caller inspection found identical call counts, without proving a cause for control drift. The final independently measured variant removes all byte routing/ownership changes and retains only the exact ARM64 fold. A preliminary unused-import test setup failure is preserved in the artifact directory, followed by passing corrected gates.
