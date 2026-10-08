# Go-native SIMD port

The arm64 Go-native SIMD kernels use `simd/archsimd` behind Go's experimental
`GOEXPERIMENT=simd` build experiment. This document describes how to build and
validate those paths; it does not make a performance claim. Earlier measurements
used a forked toolchain and are not evidence for the official Go release.

## Toolchain and commands

The modules require Go 1.27.0 or newer. Use the official Go distribution. The
`scripts/simdtip.sh` filename is retained for compatibility, but the script now
selects `go` from `PATH` by default or an executable supplied with `GO`:

~~~sh
scripts/simdtip.sh build ./...
scripts/simdtip.sh test ./internal/av1/dsp -run 'SIMD'
scripts/simdtip.sh conformance
GOEXPERIMENT=simd make dryrun-profiles
~~~

The script forces `simd` on, preserves other explicit experiment selections,
and inherits the caller's `GOROOT`, `GOTOOLCHAIN`, and `GOCACHE`. For normal
builds, use the standard Go command without the experiment. SIMD files are
selected by the `goexperiment.simd` build tag.

## Validation

For each kernel change, establish byte-for-byte agreement with an independent
scalar or assembly implementation across edge cases and relevant input ranges.
Then prove that the public dispatch selects the intended path and run the
applicable end-to-end parity gates:

~~~sh
scripts/simdtip.sh build ./...
scripts/simdtip.sh test ./...
scripts/simdtip.sh conformance
GOEXPERIMENT=simd make dryrun-profiles
~~~

The native arm64 CI lane uses the official Go 1.27.x toolchain with
`GOEXPERIMENT=simd` and runs build, tests, and the strict fast and profile
decoder parity gates. Those decoder gates validate exercised decoder inputs;
they do not measure encoder quality or prove universal AV1 conformance.

Measure performance separately on the same official toolchain and hardware for
the SIMD, scalar, and assembly implementations, using representative sizes and
inputs. Confirm the SIMD implementation is live and check allocations before
making performance claims. See [quality and parity](docs/quality-and-parity.md)
for project-level checks. The current arm64 kernel results, reproduction
commands, and claim limits are in the
[Go 1.27 SIMD measurement note](docs/go127-simd-performance.md).

## Remaining assembly (2026-10-08 snapshot)

The replacement is partial. This tree contains 109 assembly source files:
67 ARM64 and 42 AMD64, with 303 `TEXT` symbols across both architectures.
These are source counts, not runtime shares: a shared file can contain both
replaced kernels and live fallback kernels. Relative to `origin/main` at
`024942ea`, four assembly files have been removed entirely; additional replaced
bodies were removed from shared files. AMD64 remains primarily assembly-backed.

| Area | Assembly files | Main work still remaining |
| --- | ---: | --- |
| Transforms | 23 | Remaining forward transforms, fused transform/reconstruction glue, shapes and numeric-range fallbacks |
| Encoder | 14 | SAD shapes other than selected 8×8×4, motion-search helpers, RD statistics, residual preparation and resizing |
| Motion compensation | 10 | 8-bit I8MM/dot-product convolution, 8-bit warp, remaining compound and high-bit-depth fallback shapes |
| Intra prediction | 9 | Remaining directional, filter-intra and static predictor kernels |
| Restoration | 9 | Wiener and most self-guided filtering; the HBD final projection has a Go SIMD implementation |
| Loop filters | 8 | 8-bit filters and HBD 4/8-tap paths; HBD 6/14-tap Go SIMD is selected |
| CDEF | 7 | Remaining filtering variants; direction search and selected filtering paths use Go SIMD |
| Other DSP, entropy, quantization, frame/tile helpers, film grain and super-resolution | 29 | Includes serial entropy routines, so SIMD is not a direct substitute for every assembly function |

The current Go 1.27.2 ARM64 SIMD API lacks the I8MM matrix-multiply operations
used by the 8-bit motion kernels. Widening multiply/add implementations are
possible, but the tested versions have not matched that specialized assembly.
The six-tap I8MM improvement removes zero-tap work in retained assembly; it is
not an assembly replacement. The quantize-b encoder path also still selects
NEON assembly, as do measured-losing SAD and loop-filter shapes.

Do not delete an entire assembly file merely because one SIMD dispatch replaces
one of its symbols. Check default builds, experiment builds, other architectures,
assembly-to-assembly calls, and numeric/shape fallbacks. Remove replaced bodies
once those production references are gone, retaining an independent correctness
oracle without keeping otherwise-dead production code.

## Porting guidelines

- Keep architecture and experiment build constraints explicit. Standard builds
  use the Go fallback where SIMD replaces assembly, and retain assembly where
  SIMD has not shown a repeatable benefit.
- Test output bytes and state with an independent oracle. Do not derive expected
  results from the implementation under test or widen tolerances to mask drift.
- Include tails, unaligned boundaries, extreme values, and supported bit depths
  in differential coverage.
- Benchmark through public dispatch as well as the isolated kernel so the
  measured path matches production use.
- Keep kernels on the existing implementation when SIMD is not byte-exact or
  does not show a repeatable benefit under the official toolchain.
