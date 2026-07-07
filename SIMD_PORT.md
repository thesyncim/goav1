# Go-native SIMD kernel port

Migrating goav1's arm64 kernels from hand-written Plan 9 assembly to Go-native
SIMD intrinsics (`simd/archsimd`, Go 1.27+ `GOEXPERIMENT=simd`). Motivation and
proof: the residual-add spike (PR #6) showed tuned Go-native SIMD **matches and
beats** the hand asm (64×64: 394 vs 500 ns) while staying maintainable Go.

## Toolchain

- Install/refresh tip: `go install golang.org/dl/gotip@latest && gotip download`
- All build/test/bench go through `scripts/simdtip.sh` (sets GOROOT=~/sdk/gotip,
  GOTOOLCHAIN=local, dedicated GOCACHE, GOEXPERIMENT=simd).
- Production (Go 1.26, no experiment) is unaffected: every SIMD file is behind
  `//go:build goexperiment.simd`, so `go build ./...` excludes it entirely.

## Wiring pattern (per package)

1. `foo_simd_arm64.go` (`//go:build goexperiment.simd && arm64 && !purego`) —
   the Go-native SIMD kernel(s), array-pointer (`*[N]T`) loads/stores only (no
   slice headers in the hot path).
2. `foo_dispatch_simd_arm64.go` (same tag) — `init()` binds the dispatch var to
   the SIMD kernel.
3. Existing `foo_dispatch_arm64.go` gets `&& !goexperiment.simd` added so the
   asm and SIMD inits never both run.
4. `foo_simd_arm64_test.go` (same tag) — differential test vs the scalar/asm
   reference (fuzz + edge sweep, byte-identical) + a scalar/SIMD/asm benchmark.

## Gates (every kernel)

- Byte-identical to the scalar reference (differential fuzz + edge cases).
- `scripts/simdtip.sh conformance` stays green with the SIMD kernel **active**
  (strict-MD5 dryrun; the whole-decoder proof).
- `go build ./...` on stock Go 1.26 unchanged (0 simd files in `go list`).
- Benchmark recorded (scalar / SIMD / asm) — SIMD must be ≥ asm or documented.

## Tiers (from the 60-kernel audit)

### Tier 1 — CANNOT be SIMD (inherently serial) — stays asm
- `entropy/*` MSAC arithmetic coder (reader + writer). Bit-serial; no data
  parallelism. dav1d's is scalar asm too. **Do not port.**
- Serial parts of `tile/*` token read and `encoder/*` rate/token paths.

### Tier 2 — CAN be SIMD at parity/better — PORT THESE (~45 kernels)
Ordered by decode hotness (port highest first):
- [x] `dsp` residual-add — PORTED + WIRED. Matches/beats asm (64×64 394 vs 500
      ns); byte-exact; `conformance` green under gotip+simd with it active.
- [x] `dsp` raw-transform-add — PORTED + WIRED. Dead parity with asm (64×64
      751 vs 750 ns, 7x over scalar); byte-exact (int32->int16 pack via uint64
      reshape bridge); conformance green with both dsp kernels active.
- [ ] `transform` (15) — inverse DCT/ADST/IDTX butterflies (add/sub/mul/shift/clamp).
- [ ] `loopfilter` (5) — deblock: abs-diff, thresholds, clamps, compare+select.
- [ ] `cdef` (4) — directional filter: min/max/abs/diff, constant taps.
- [ ] `restoration` (4) — SGR/Wiener box sums + weighted filters.
- [ ] `prediction` (4) — intra DC/paeth/smooth/directional.
- [ ] `motion` blends (part of 5) — compound avg / mask / OBMC.
- [ ] `quantize` (3) — mul/shift/round (encoder-side, still vectorizable).
- [x] `dsp` blend (blendA64Mask) — PORTED + WIRED. Parity with asm (64×64 776
      vs 775 ns, 11x over scalar); byte-exact; 8-bit non-subsampled fast path,
      scalar fallback for subsampled/HBD. conformance green (3 dsp kernels live).
- [ ] `dsp` minmax, `superres` (1), `frame` (1), `filmgrain`
      noise-add (the LFSR stays scalar).

### Tier 3 — CAN be SIMD but BLOCKED by missing package ops
- `motion` `convolve_dotprod`, `convolve_i8mm` — need **dot-product** (SDOT/UDOT)
  and **int8 matrix** (USMMLA); absent from the arm64 simd package. A MulWidenLo+
  add fallback works but likely loses to the current DOTPROD asm.
- Any kernel needing arbitrary byte-shuffle (**TBL**) — package has only
  structured permutes (InterleaveLo/Hi, ConcatShiftBytesRight, ConcatEven/Odd).
- Action: keep these on asm; upstream a request for SDOT/UDOT/TBL on arm64.

## Progress

- Foundation verified: whole module builds under gotip and gotip+simd;
  `conformance` green under gotip (asm path) AND with the Go-native SIMD
  residual-add active. Machine + wiring pattern + gates all proven.
- 3 / ~45 Tier-2 kernels ported (dsp add-family + blend). Remaining is a per-kernel grind: for each,
  mirror the residual-add pair (kernel + dispatch-simd), differential-test to
  byte-exact, gate with `conformance`. Some need API workarounds (int32→int16
  packing lacks a direct reshape; go via the uint bitcast path). Track here.
