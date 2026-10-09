# Matched 8-bit dav1d and Go I8MM motion kernels

This direct-kernel harness requires an ARM64 host with I8MM support.

This harness compares 8-bit AV1 2D motion filtering with the same input pixels, filter phases, and block shapes. It uses a deterministic 144×144 source plane with an 8-pixel border, regular8 phase 6 horizontally, smooth8 phase 9 vertically, and square blocks from 8×8 through 128×128. The Go PureGo implementation exports expected `put` and `prep` outputs; the C preflight compares every result sample before timing.

The timing boundary differs between the implementations. The Go benchmark calls the resident Go entry points `convolve2D8I8MMWithScratch` and `predictInterCompoundRef8ToConvBuf2DI8MM`, so its number includes their per-call feature/shape checks and coefficient preparation as well as the resident I8MM assembly kernel. The C harness calls dav1d's specialized `dav1d_put_8tap_regular_smooth_8bpc_neon_i8mm` and `dav1d_prep_8tap_regular_smooth_8bpc_neon_i8mm` symbols directly, bypassing dav1d's runtime dispatch. The results therefore compare a Go resident wrapper plus kernel against a direct specialized C kernel. They are not a pure assembly-to-assembly dispatch comparison or a full-decoder speed claim.

The source checkout must be pinned to dav1d commit `b546257f770768b2c88258c533da38b91a06f737`; the harness also records the static archive hash because the source revision alone cannot prove which source produced that archive. The harness links no C code into the Go package.

## Build and verify

Provide the source checkout and optimized static archive through environment variables. `PROJECT_ENV` is optional; set it to the project wrapper when repository policy requires Go commands to run through that wrapper. The scripts preserve caller-supplied `CODEX_AGENT_ID` and `CODEX_PROJECT_CACHE_ROOT`. `GOTOOLCHAIN` defaults to `local` (no version is pinned), and `GOEXPERIMENT` defaults to `simd`; callers can override either.

```sh
repo_root=$(git rev-parse --show-toplevel)
outdir=$(mktemp -d "${TMPDIR:-/tmp}/goav1-u8-motion.XXXXXX")
PROJECT_ENV=/path/to/project-env \
DAV1D_SOURCE=/path/to/pinned/dav1d/source \
DAV1D_STATIC_LIB=/path/to/pinned/libdav1d.a \
  "$repo_root/tools/u8_motion_bench/build.sh" "$repo_root" "$outdir"
```

`build.sh` exports fixtures through a temporary Go overlay, builds the C harness with `-O3 -DNDEBUG -Wall -Wextra -Werror -std=c11`, records host, compiler, local Go version, Go settings, source revision and Go/assembly source hashes from the supplied `REPO_ROOT`, and verifies all C outputs against the PureGo fixtures. The overlay maps only the temporary fixture-export test; it does not replace production Go or assembly files. The generated overlay and metadata are retained in `OUTDIR` for review.

## Direct C timing

The executable reruns parity before emitting raw CSV. Verification details go to stderr, and stdout contains only CSV.

```sh
"$outdir/u8_motion_bench" --bench --fixtures "$outdir/fixtures" \
  > "$outdir/dav1d-i8mm.csv" 2> "$outdir/dav1d-i8mm.log"
```

For every operation and size it uses one 200 ms warmup and five 200 ms samples. The C timing measures direct kernel cost with resident input and reused output buffers; it excludes decoder setup and dispatch.

## Matched Go wrapper timing

The same fixture-test overlay adds `BenchmarkU8MotionI8MMConvolve` and `BenchmarkU8MotionI8MMCompound`. They call the resident Go wrapper paths described above and use the same deterministic source plane, filters, and block sizes. Run serially on the same host as the C timing:

```sh
PROJECT_ENV=/path/to/project-env
GOTOOLCHAIN=${GOTOOLCHAIN:-local}
GOEXPERIMENT=${GOEXPERIMENT:-simd}
export GOTOOLCHAIN GOEXPERIMENT

"$PROJECT_ENV" go test \
  -overlay="$outdir/go-overlay.json" \
  ./internal/av1/motion -run '^$' \
  -bench '^BenchmarkU8MotionI8MM(Convolve|Compound)$' \
  -benchmem -benchtime=200ms -count=5
```

If `PROJECT_ENV` is unset and local repository policy permits direct Go commands, invoke `go test` with the same flags. Do not set a fixed toolchain version for this diagnostic. The Go benchmark number includes its resident wrapper work; compare it with the direct C kernel number as a diagnostic, not as an end-to-end decoder ratio.
