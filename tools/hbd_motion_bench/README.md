# Matched Go and dav1d HBD motion kernels

This harness compares the Go benchmark’s exact high-bit-depth motion fixtures with dav1d’s direct 16bpc NEON `put` and `prep` kernels. It covers 10-bit samples, regular8 phase 6 horizontally plus smooth8 phase 9 vertically, and square blocks from 8×8 through 128×128. The 144×144 source plane has eight pixels of padding on each side. Fixture generation calls the existing `makeHighBDRef` helper with the exact seeds from `BenchmarkMotionHBD6TapConvolve` (`0x6bdc`) and `BenchmarkMotionHBD6TapCompound` (`0x6bdd`).

The Go exporter template is named `export_fixtures_test.go.in` so normal `go test ./...` discovery cannot compile it as a standalone package. `prepare_fixtures.sh` maps it to a temporary `*_test.go` in `internal/av1/motion` through a Go overlay; no library or C code is linked into the Go package. Supply the optimized static archive built from the same pinned dav1d source, with the NEON symbols named in `hbd_motion_bench.c`. The script checks the source checkout, records the archive hash, and uses exact parity to validate behavior; the source revision alone does not prove which source produced the archive. It emits pure-Go expected output and the same raw reference bytes consumed by C. The C harness calls the retained dav1d NEON symbols directly through external function pointers. It does not use cgo or decoder initialization.

The ABI comes from the pinned `third_party/upstream/dav1d/src/mc.h`: `put` takes nine arguments including `bitdepth_max`; `prep` takes eight. Both source/destination strides are byte counts at the ABI boundary, as `PXSTRIDE` in `src/common/bitdepth.h` confirms. `prep` writes packed signed 16-bit samples. The Go compound buffer is compared against `(uint16_t)((int32_t)prep_sample + 32768)`. The preflight compares every sample and the build script refuses a dav1d source checkout other than commit `b546257f770768b2c88258c533da38b91a06f737`.

## Build and verify

Choose an output directory; the scripts preserve it for review. Pass the optimized static archive explicitly through `DAV1D_STATIC_LIB`:

```sh
repo_root=$(git rev-parse --show-toplevel)
outdir=$(mktemp -d "${TMPDIR:-/tmp}/goav1-hbd-motion.XXXXXX")
PROJECT_ENV=/path/to/project-env \
DAV1D_SOURCE=/path/to/pinned/dav1d/source \
DAV1D_STATIC_LIB=/path/to/pinned/libdav1d.a \
  "$repo_root/tools/hbd_motion_bench/build.sh" "$repo_root" "$outdir"
```

The build script invokes `project-env go test` to export fixtures, compiles the C harness with `-O3 -DNDEBUG -Wall -Wextra -Werror -std=c11`, writes host/compiler/Go/source/library-hash metadata, and runs the exact parity preflight. When `PROJECT_ENV` is set, Go commands run through that wrapper and use its project-scoped cache derived from `git rev-parse --git-common-dir`. Set it when repository policy requires the wrapper. `DAV1D_SOURCE` can point to the pinned checkout when the current worktree omits the vendored source directory. `CODEX_PROJECT_CACHE_ROOT`, `CODEX_AGENT_ID`, `GOTOOLCHAIN`, and `GOEXPERIMENT` can be overridden in the environment. The metadata records the official Go version returned by that wrapper, `GOTOOLCHAIN`/`GOEXPERIMENT`, the Go worktree revision and tracked motion-package status/diff hash, the exact dav1d source revision, static archive hash, compiler, and harness hash.

## C timing

The executable requires an explicit fixture directory. Timing mode first reruns full parity and sends verification diagnostics to stderr; stdout is raw CSV only.

```sh
"$outdir/hbd_motion_bench" --bench --fixtures "$outdir/fixtures" \
  > "$outdir/dav1d-kernels.csv" 2> "$outdir/dav1d-kernels.log"
```

For every operation and size, it runs a 200 ms warmup and then five 200 ms samples. The CSV records the actual monotonic elapsed time, actual call count, and ns/call. A volatile result sink keeps outputs observable. Timing is direct-kernel cost with resident input and reused output buffers; it excludes decoder setup and dispatch.

For a matched Go comparison, run the existing benchmarks through `project-env` with the same Go version, `GOEXPERIMENT=simd`, and serial CPU conditions. The `NEON` sub-benchmark measures the Go NEON reference; `GoSIMD-kernel` measures the resident six-tap candidate. Both use the same seeds and shapes as this harness:

```sh
PROJECT_ENV=/path/to/project-env
export PROJECT_ENV
CODEX_PROJECT_CACHE_ROOT=/tmp/codex-projects \
CODEX_AGENT_ID=hbd-motion-go \
GOTOOLCHAIN=auto GOEXPERIMENT=simd \
"$PROJECT_ENV" go test \
  ./internal/av1/motion -run '^$' \
  -bench 'BenchmarkMotionHBD6Tap(Convolve|Compound)/regular8/2D/(NEON|GoSIMD-kernel)/(W8H8|W16H16|W32H32|W64H64|W128H128)$' \
  -benchmem -benchtime=200ms -count=5
```

Keep the recorded Go and C runs on the same host and avoid running them concurrently. The C harness is a direct function comparison, not a claim about full decoder speed.
