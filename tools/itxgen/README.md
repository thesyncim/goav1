# itxgen — inverse-transform generators

The active DCT32/DCT64 four-lane implementation is Go-native SIMD. AMD64
dispatch uses official `archsimd.X86.AVX2()` detection; ARM64 uses NEON.
There are no repository-owned transform assembly files or active assembly
generators. The old `avx2gen` emitter has been removed.

`dct32_from_lanes4.py` derives the DCT32 network from the even DCT32 half of
the existing DCT64 Go network, halves each even input index and stores its
32 outputs without the final DCT64 odd-half butterfly. It preserves both
narrow and wide coefficient decompositions and per-stage clamps. Regenerate
the portable Go kernel with:

```sh
python3 tools/itxgen/dct32_from_lanes4.py > internal/av1/transform/dct32_lanes4_gosimd.go
```

`TestNativeDCTLanes4Parity` checks both sizes against the independent scalar
Go implementation across stage bounds, strides, extremes and sparse inputs.
`TestNativeDCTLanes4ShortRows` checks mixed short rows and untouched tails;
`TestNativeDCTLanes4NoAlloc` checks zero steady-state allocations.
`TestAMD64DCTLanes4Dispatch` checks that live
AMD64 slots select Go SIMD rather than silently measuring scalar fallback.

## Historical NEON assembly tooling

The C harness and the older Python transcription pipeline below are retained
as historical reference/oracle tooling, not as the production build path.
Do not regenerate or reintroduce assembly into the transform package.

Pipeline:

1. `gen.py` — AST-transliterates the pure-Go butterflies in
   `internal/av1/transform/dct.go` into two C variants: a NEON intrinsics
   kernel (4 int32 lanes per `.4s` register, cosine weights |w| > 2048
   rewritten as w−4096 plus a compensating post-shift add so every multiply
   accumulator stays < 2^31 — exact integer identity at cos_bit=12) and an
   int64 scalar reference using the original weights. Emits `core_neon.h`,
   `core_ref.h`, `kernels.c`.
2. `harness.c` — cross-checks NEON vs reference over ~180k random + extremal
   vectors per kernel (inputs pre-clamped to the ±2^19 stage envelope the
   dispatch adapters guard).
3. `clang -O2 -ffixed-x18 -ffixed-x28 -c kernels.c -o kernels.o` compiles
   `kernels.c`; `transcribe.py` converts the disassembled bodies into the
   WORD-encoded Go assembly files. The fixed-register flags reserve Go's `g`
   register (`x28`) and Darwin's platform register (`x18`), and the
   transcriber rejects either register if one appears in a generated body.

Constraints learned the hard way (keep them):
- Go NOSPLIT frames have ~800 bytes of headroom; clang loves multi-KB frames.
  Keep stage buffers in Go-provided stack scratch and split stages into
  independent butterfly groups with memory barriers so clang cannot re-inflate
  the frame. Kernels with larger scratch areas must use a Go-managed split
  frame rather than a manual stack adjustment below a zero-sized NOSPLIT frame.
- The final gate is never the C harness: it is the repo's Go differential
  tests against the actual pure-Go code plus `make dryrun-extended`.
