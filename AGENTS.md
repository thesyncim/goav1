# Repository instructions

## Implementation language & SIMD

- Write production codec implementations in Go, including optimized paths.
- Use Go `simd` or `simd/archsimd` APIs where appropriate (`GOEXPERIMENT=simd`).
- Do not add `.s` or `.S` files, handwritten assembly, or machine-code generation as an optimization workaround.
- Inspecting Go compiler output (`go tool objdump`, `-gcflags=-m`) for performance diagnostics is allowed; keep the implementation in Go.

## Memory safety

- Keep all `unsafe` usage strictly inside `internal/unsafeutil/`. Production codec packages must remain `unsafe`-free.
- Every change must pass `./scripts/check_unsafe.sh`.
- Never trade away bounds safety or add `-nobounds`-dependent assumptions in Go code.

## Package & file layout (anti-sprawl)

- **Do not create one-off files per PR, bug fix, or micro-kernel.** Add new functions to the existing cohesive file for that domain.
- **No empty or near-empty stub files.** If a file's contents are deleted or moved, delete the file (`git rm`) rather than leaving a bare `package foo` stub.
- **Keep SIMD variants grouped by domain, not per helper.** Prefer one `<domain>_gosimd_arm64.go` / `<domain>_gosimd_amd64.go` / `<domain>_generic.go` per subsystem rather than creating a 4-file cluster for each small kernel.
- **Keep the root package (`goav1`) small and focused on the public API.** Implementation details belong under `internal/`.

## Testing discipline

- **Consolidate tests into cohesive domain `*_test.go` files.** Do not create a new `*_test.go` file for a single regression test, benchmark, or bug fix—append it to the existing test file for that subsystem (e.g., `decode_frame_thread_test.go`, `encoder_test.go`, `bench_test.go`, `decode_fuzz_test.go`).
- **Prefer table-driven tests and invariant/parity tests** (e.g., SIMD vs scalar reference, bit-exact MD5 parity vs `dav1d`) over redundant micro-tests that mirror implementation details.
- **Avoid brittle meta-tests** (such as asserting internal dispatch function names or duplicating the same smoke test across multiple files) unless guarding a proven, recurring regression.
- **Delete obsolete tests** when an internal API or legacy path is removed; do not leave commented-out or skipped tests behind.

## Code quality & comments (anti-slop)

- **Comments must explain *why*, invariants, or AV1 spec formulas—not *what* the code obviously does.**
- **No archaeology in comments.** Do not reference PR numbers, issue numbers, "Phase 1 / Wave 2 / Milestone X", deleted planning docs (`*_PLAN.md`), or narrate what a previous version of the code used to do. That history belongs in git commits and `CHANGELOG.md`.
- **No dead code or speculative abstractions.** Do not add unused helpers, wrapper types for a single call site, or premature interfaces. Delete unused functions immediately.
- **Keep functions clear and bounded.** Avoid deeply nested copy-paste blocks; use small inlinable helpers or clean loops when unrolling causes register spills.
- **Do not leave scratch binaries, profiles, or temporary outputs** (`*.test`, `*.pprof`, `-`, etc.) in the repository root.

## Verification

Before finishing any code change:
1. `./scripts/check_unsafe.sh`
2. `GOEXPERIMENT=simd go vet ./...`
3. `GOEXPERIMENT=simd go test` on all affected packages
4. For decoder changes, verify bit-exact MD5 parity against `dav1d --filmgrain 1` (single-threaded and multi-threaded) when corpus streams are present.
