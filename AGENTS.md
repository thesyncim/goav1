# Repository instructions

## Implementation language

- Write production codec implementations in Go, including optimized paths.
- Use Go `simd` or `simd/archsimd` APIs where appropriate.
- Do not add `.s` or `.S` files, handwritten assembly, or machine-code generation as an optimization workaround.
- Inspecting Go compiler output for performance diagnostics is allowed; keep the implementation in Go.
