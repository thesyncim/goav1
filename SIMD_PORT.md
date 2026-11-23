# Go-native SIMD

The arm64 (NEON) and amd64 (AVX2) kernels are written in Go with
`simd/archsimd`, behind Go's `GOEXPERIMENT=simd` build experiment. The
repository contains no assembly. Without the experiment, and under the `purego`
build tag, the scalar Go implementations run instead; they are also the
reference every kernel is tested against.

## Build and test

Use the official Go distribution, 1.27.4 or newer.

~~~sh
GOEXPERIMENT=simd go build ./...
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd make dryrun-fast
GOEXPERIMENT=simd make dryrun-profiles
~~~

SIMD files are selected by the `goexperiment.simd` build tag. amd64 kernels
require AVX2, detected through `archsimd.X86`; arm64 feature detection is in
`internal/av1/dsp/cpu`.

The native arm64 and amd64 CI lanes build, vet and test with the experiment,
run the kernel packages under `-race` (which enables checkptr), and run the
strict fast and profile decoder gates. Those gates validate the decoder inputs
they exercise; they do not prove universal AV1 conformance.

## Writing a kernel

- Match the scalar implementation byte for byte across every block shape,
  tail, unaligned boundary, extreme value and supported bit depth, with a
  differential test that does not derive its expectation from the kernel.
- Prove that the public dispatch selects the kernel, then run the decoder
  gates above (and the encoder byte-identity checks for encoder kernels).
- `unsafe` is allowed only where every access is provably in bounds and the
  kernel is measurably faster for it. Never form a pointer outside its
  allocation, not even one past the end: derive row pointers from an in-bounds
  base and an integer offset. `//go:nocheckptr` is not allowed.
  `scripts/check_unsafe.sh` lists the files that may import `unsafe`.
- State what a kernel may read and write around the block it is given, enforce
  it with a length check at the slice-taking entry point, and cover it with a
  guard-page test (`internal/guardpage`), which places the buffers against
  inaccessible pages.
- On amd64, avoid operations that lower to AVX-512 (`TruncTo*`,
  `SaturateTo*` on wide types, `Permute` on bytes) and call
  `archsimd.ClearAVXUpperBits()` before leaving a kernel that used 256-bit
  registers; Zen 3 otherwise stalls on the following SSE code.
- Measure through the public dispatch as well as the isolated kernel, on the
  same toolchain and hardware, and compare instructions as well as time. Check
  that the path allocates nothing.

Kernel measurements and their limits are in the
[Go 1.27 SIMD measurement note](docs/go127-simd-performance.md); project-level
gates are described in [quality and parity](docs/quality-and-parity.md).
