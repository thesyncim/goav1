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
- [~] `transform` — DCT8Col2 PROVEN byte-exact (colpass_gosimd_arm64.go, 20k
      differential cases incl 12-bit range that overflows int32; Int64x2 2-column
      butterfly, MulWidenLo int64 products, SaturateToInt32+clamp = clipRange).
      NOT WIRED: even optimized (hoisted const vectors + Array loads) it is ~25%
      SLOWER than the hand asm (12.6 vs 10.1 ns) — the 2-column Col2 is too narrow
      + StorePart masking. FINDING: transform butterflies port CORRECTLY but only
      win at width; the perf case rests on the wider Col4 / row-pass kernels (TBD).
      Pattern + differential harness established for the whole family.
- [ ] `loopfilter` (5) — deblock: abs-diff, thresholds, clamps, compare+select.
- [ ] `cdef` (4) — directional filter: min/max/abs/diff, constant taps.
- [ ] `restoration` (4) — SGR/Wiener box sums + weighted filters.
- [ ] `prediction` (4) — intra DC/paeth/smooth/directional.
- [ ] `motion` blends (part of 5) — compound avg / mask / OBMC.
- [ ] `quantize` (3) — mul/shift/round (encoder-side, still vectorizable).
- [x] `dsp` blend (blendA64Mask) — PORTED + WIRED. Parity with asm (64×64 776
      vs 775 ns, 11x over scalar); byte-exact; 8-bit non-subsampled fast path,
      scalar fallback for subsampled/HBD. conformance green (3 dsp kernels live).
- [x] `dsp` minmax (minMaxAbsDiff8x8) — PORTED + WIRED. Parity with asm (26.2
      vs 26.0 ns); byte-exact reduction (running per-lane min/max + reduce).
      **dsp package COMPLETE.**
- [ ] `superres` (1), `frame` (1), `filmgrain`
      noise-add (the LFSR stays scalar).

### Tier 3 — CAN be SIMD but BLOCKED by missing package ops
- `motion` `convolve_dotprod`, `convolve_i8mm` — need **dot-product** (SDOT/UDOT)
  and **int8 matrix** (USMMLA); absent from the arm64 simd package. A MulWidenLo+
  add fallback works but likely loses to the current DOTPROD asm.
- Any kernel needing arbitrary byte-shuffle (**TBL**) — package has only
  structured permutes (InterleaveLo/Hi, ConcatShiftBytesRight, ConcatEven/Odd).
- Action: keep these on asm; upstream a request for SDOT/UDOT/TBL on arm64.

## Perf-at-width finding (2026-07-08) — decides which kernels are worth porting

The Go-native SIMD port is byte-exact for EVERY kernel shape tested (element-wise,
reduction, int32-pack, int64 butterfly). But the perf WIN is width-dependent:

- **WIDE / element-wise (dsp): Go-SIMD BEATS/MATCHES asm.** 16-wide uint8/uint16
  lanes amortize overhead. dsp package ported + wired (residual-add beats asm).
- **int64-precision transforms: Go-SIMD LOSES to asm ~25-33%.** All AV1 inverse
  transforms need int64 intermediates (12-bit coeff*const overflows int32), which
  caps SIMD at 2 lanes (Int64x2). Col4 = 2xCol2 batched; Row4 = 4 rows / 2 per
  vector — every tiling is inherently 2-wide. The asm schedules the 2-wide int64
  butterfly better than Go's compiler (register allocation on ~10 live constants),
  so wiring transform SIMD REGRESSES decode. DCT8Col2 proven byte-exact but NOT
  wired for this reason. Possible future win: a bit-depth-conditional int32 4-wide
  path for 8-bit content (products fit int32) — more complex, not yet attempted.

STRATEGY REFINEMENT: "port everything" -> "port what WINS". Port the wide/
element-wise kernels; keep asm for the int64-transform butterflies (byte-exact Go
SIMD exists as a maintainable fallback but costs perf). Next: assess loopfilter /
cdef / restoration / prediction — are they wide-element-wise (port) or int64-narrow
(skip for perf)?

## Progress

- Foundation verified: whole module builds under gotip and gotip+simd;
  `conformance` green under gotip (asm path) AND with the Go-native SIMD
  residual-add active. Machine + wiring pattern + gates all proven.
- 4 Tier-2 kernels WIRED (dsp package). Transform butterfly proven byte-exact
  (DCT8Col2) but not wired (loses to asm on narrow Col2). **dsp package fully ported** (residual-add,
  raw-transform, blend, minmax) — all byte-exact, conformance-green, at asm parity. Remaining is a per-kernel grind: for each,
  mirror the residual-add pair (kernel + dispatch-simd), differential-test to
  byte-exact, gate with `conformance`. Some need API workarounds (int32→int16
  packing lacks a direct reshape; go via the uint bitcast path). Track here.
