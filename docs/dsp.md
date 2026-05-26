# DSP Dispatch Layer

The DSP layer (`internal/av1/dsp`) holds the small set of pixel-level
primitives that every higher-level package depends on: plane copy/fill,
clipped residual add, mask blending, and SAD-style block statistics. These
primitives are intentionally narrow: each one is a single hot inner loop
that today exists in pure Go and will eventually have per-architecture
SIMD variants.

This document describes the dispatch pattern those primitives use so that
SIMD variants can be added later without touching call sites and without
regressing the pure-Go reference path.

## Goals

The dispatch layer is constrained by three project-wide invariants:

- **Bit-exact behaviour across architectures.** Every SIMD variant MUST
  produce byte-for-byte identical output relative to the pure-Go reference
  on every supported (`bytesPerSample`, stride, shape) input. The existing
  libaom-conformance tests are the gate; the dispatch round-trip test
  (`TestMinMaxAbsDiff8x8DispatchMatchesPureGo`) catches regressions cheaply
  on every run.
- **Zero steady-state overhead.** Dispatch must resolve once, at package
  init, and the call site must become a single indirect call. There is no
  per-call feature check, no `runtime.GOARCH` switch, no `sync.Once`.
- **Zero allocations on the hot path.** The existing allocation guardrails
  (`TestMinMaxAbsDiff8x8Allocs`, the dispatch-specific
  `TestMinMaxAbsDiff8x8DispatchIsZeroAlloc`) protect this.

## Layout

```
internal/av1/dsp/
+-- minmax.go                      // exported entry, calls minMaxAbsDiff8x8Impl
+-- minmax_pure_go.go              // canonical reference implementation
+-- minmax_dispatch_amd64.go       // init() picks best variant on amd64
+-- minmax_dispatch_arm64.go       // init() picks best variant on arm64
+-- minmax_dispatch_generic.go     // init() locks in pure-Go on others
+-- cpu/
    +-- cpu.go                     // Features struct + Detected global
    +-- cpu_amd64.go               // populates Detected on amd64
    +-- cpu_arm64.go               // populates Detected on arm64
    +-- cpu_generic.go             // no-op on other architectures
```

The dispatch entry is a single package-scope function variable, e.g.
`minMaxAbsDiff8x8Impl`. The exported `MinMaxAbsDiff8x8` simply calls
through it. Each architecture file owns one `init()` whose only job is to
assign the variable to the best available variant given the flags in
`cpu.Detected`. Because Go runs all `init()` functions before `main`, by
the time any decoder goroutine starts work the function pointer has been
resolved.

## How to add a SIMD variant

The current scaffold uses the pure-Go reference on every architecture
because no SIMD assembly is shipped yet. Adding one is a four-step
change, none of which touch call sites:

1. **Write the variant** in a new build-tagged file, e.g.
   `minmax_neon_arm64.s` plus a Go stub that declares the assembly
   symbol. The variant has the same signature as `minMaxAbsDiff8x8PureGo`.
2. **Surface a feature flag** on `cpu.Features` if the existing set does
   not already cover what you target (most cases reuse `NEON`, `AVX2`,
   etc.). Populate it from the per-arch `cpu_*.go`.
3. **Wire it into dispatch** by editing the matching
   `minmax_dispatch_<arch>.go`: add a branch like
   `if cpu.Detected.NEON { minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8NEON }`.
4. **Keep the conformance gate** by ensuring the existing
   `Test*MatchesPureGo` / libaom-reference tests run with the new variant
   enabled. `cpu.OverrideForTest` lets a test force a specific branch on
   architectures that support multiple variants.

The current example kernel is `MinMaxAbsDiff8x8`. The same pattern applies
without modification to other hot primitives (`AddResidualPlaneBlock`,
`BlendA64Mask`, and the transform / CDEF / motion kernels in their
respective packages).

## Why this pattern, not build tags alone

A pure build-tag approach (separate file per arch, no dispatch variable)
forces a one-implementation-per-arch decision at build time. The
function-pointer dispatch lets a single binary carry several variants and
choose the best one for the actual CPU it lands on, which matters for
amd64 (SSE2 baseline vs. AVX2 vs. AVX512) and arm64 (NEON baseline vs.
SVE). The cost — one indirect call per kernel invocation — is below the
noise floor for the block sizes goav1 dispatches on.

## Testing

- `TestMinMaxAbsDiff8x8DispatchMatchesPureGo` exercises the public entry
  across every supported stride / sample-width combination and asserts
  byte-for-byte equality with the reference. Future SIMD variants get this
  coverage automatically.
- `TestMinMaxAbsDiff8x8DispatchIsZeroAlloc` keeps the zero-alloc
  invariant.
- `cpu.TestDetectedMatchesArch` is the unit test for the detection layer
  itself; it confirms each architecture's `init()` populates only the
  fields it should.
