# Go SIMD APIs and dispatch

Checked against Go 1.27.2 on 2026-10-16. Confirm signatures and features in the project's compiler; the experiment has no Go 1 compatibility guarantee. The [portable](https://go.dev/blog/simd-experiment) and [architecture-specific](https://go.dev/blog/archsimd) articles provide context, not a substitute for pinned source.

## API choice and migration

Go 1.27 adds portable `simd` and arm64/wasm `archsimd` to the earlier amd64 experiment. Enable with `GOEXPERIMENT=simd`; use `goexperiment.simd` build constraints and architecture constraints for direct intrinsics. Keep ordinary builds supported where required.

Check older examples: slice loads/stores use short names, array pointers use `Array`, `IfElse` replaces `Merge`, and bit reinterpretation uses `ToBits`, `ReshapeToUint…s`, and `BitsTo…`. Numeric conversion is different from reinterpreting bits. Planned reductions/SVE are not guaranteed in Go 1.27.

## Widths, tails, and masks

Use `Len()` for portable vectors. Full loads need enough elements; `Load…Part` zero-fills and reports elements loaded, while `StorePart` bounds the write. Zero padding is not neutral for every reduction, division, or comparison. Use a justified identity/mask or scalar tail. Check every slice's length; a subtraction-based bound `i < n-L+1` with positive `L` avoids the potential overflow of `i+L <= n`.

Masks are opaque. Check arithmetic/mask fusion before hand coding blends. Grouped x86 shuffles operate within 128-bit sublanes; `PermuteOrZero` and NEON/wasm `LookupOrZero` differ on invalid indices. Test boundaries when porting. FMA and regrouping need the numerical contract; measure reduction scratch allocation instead of assuming it disappears.

## Runtime and specialization

Run Go 1.27 portable code in fresh processes with default selection, `GODEBUG=simd=0`, and supported width settings. On Go 1.27.2 hardware backends, explicit `128`/`256`/`512` requests panic if the width or required features are unavailable; do not assume silent fallback. The `+` settings permit incomplete feature sets and can execute unsupported instructions; reserve them for explicit diagnostics. Preserve unrelated environment options. Log `simd.Emulated()` and `simd.VectorBitSize()`; a requested setting is not evidence of the backend selected.

Runtime emulation is distinct from experiment-off builds and from a custom direct-`archsimd` fallback. Exercise each through its real selection mechanism. Inspect portable specialized symbols such as `@simd128` and their wrappers. Mentioning a SIMD type can move dispatch outward; a benchmark-only mention can hide real production cost.

For missing operations, a small `ToArch`/`…FromArch` bridge can retain a portable algorithm. Provide architecture-specific cases and a default emulation path. Check compiler specialization before assuming interface/type-switch allocations. Direct intrinsics need the actual CPU feature predicates, which also help lowering; width alone does not imply FMA or optional extension support.
