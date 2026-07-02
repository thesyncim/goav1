# itxgen — inverse-transform NEON kernel generator

Tooling that produced the four-lane NEON DCT32/DCT64 kernels in
`internal/av1/transform/dct{32,64}_{col4,row4}_neon_arm64.s` (commit
"transform: add four-lane neon dct32/dct64 kernels"). Reuse it for the
remaining inverse-transform kernels (ADST, smaller DCT widening) and as the
template for an amd64 AVX2 wave.

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
3. `clang -O2` compiles `kernels.c`; `transcribe.py` converts the disassembled
   bodies into the WORD-encoded Go assembly files.

Constraints learned the hard way (keep them):
- Go NOSPLIT frames have ~800 bytes of headroom; clang loves multi-KB frames.
  Keep stage buffers in Go-provided stack scratch and split stages into
  independent butterfly groups with memory barriers so clang cannot re-inflate
  the frame. (The pre-existing DCT64 col2/row2 kernels violate this — that was
  the May fuzz-seed memory corruption.)
- The final gate is never the C harness: it is the repo's Go differential
  tests against the actual pure-Go code plus `make dryrun-extended`.
