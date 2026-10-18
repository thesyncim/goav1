# SIMD rolling SGR coefficient generation

On ARM64 with the Go SIMD experiment, calculate four adjacent SGR coefficients together while preserving the existing rolling sum/square producer and blend bodies. The scalar producer and AMD64 path remain unchanged. This targets coefficient variance, scaling and rounding; data-dependent lookups still use the live scalar tables. No handwritten assembly is added.

The baseline 12-bit SGR full-decoder CPU profile attributed 12.15% cumulative sampled CPU to the rolling producer, versus 1.87% to blending. These are Go sampled locations, not measured shares of the dav1d gap. The previous fixed-stencil candidate failed its whole-clip primary guard and was not included here.

A frozen screen compared three AB/BA/AB pairs on Apple M4 Max, Go 1.27.0, Darwin/ARM64, GOEXPERIMENT=simd, GOMAXPROCS=1 and GOGC=100. Each process decoded whole clips through the public API after warmup. HBD and U8 used 200 operations; the 48-frame ordinary-inter control used 10. Base 5746ed09 differs from the reused ccec baseline binary only in a compiler-diagnostic allowlist. Both binaries use the same build command and public benchmark harness, without PGO.

| Complete clip | Paired geometric-mean time saved | Faster pairs |
| --- | ---: | ---: |
| 12-bit 4:2:2 SGR inter, 8 frames (primary) | 9.119859% | 3/3 |
| 8-bit 4:4:4 postfiltered, 4 frames | 0.101812% | 1/3 |
| p720_inter_q32, 48 frames (control) | 1.561013% | 3/3 |

The predeclared primary gate required at least 0.5% savings and two faster pairs; each secondary/control allowed at most 1% geometric-mean regression. It passed. The HBD pair savings were 13.0122%, 7.9451% and 6.2650%. U8 was mixed and effectively flat. The ordinary-inter control had **zero hits** in the changed producer; its result does not establish a direct gain from this implementation. No generalized decoder speed or refreshed native ratio is inferred.

All 18 raw rows are retained in the [evidence JSON](decoder-sgr-coeff-simd-20261018.json). Root independently parsed every raw log and recomputed paired ratios. All 36 selected libaom golden validations passed. Every row reports 0 allocs/op; B/op is separately reported as 0 or 4. No row was excluded, no unchanged retry occurred, and no competing known Go/test/qualitybench process appeared in 34 host samples. PerfPowerServices and Chrome activity was recorded and remained uncontrolled; the host was not idle.

Focused checks cover 8/10/12-bit inputs, full-width int32 wrap cases, both passes, all parameters, complete A/B scratch including untouched padding/skipped rows, complete public backing and five overlap offsets, scratch reuse, invalid inputs and zero allocations. The unchanged scalar producer and preserved original driver are independent oracles. Focused checkptr, experiment-off tests, public golden clips and AMD64 cross-compilation passed; AMD64 was not executed on this ARM64 host. Integration copies the exact five frozen source/test files onto the same main base and reuses these checks.

Signed bias/rounding, uint32 multiplication wrap, saturated variance subtraction, raw unshifted B, lookup-table mutation and scalar tails retain the original behavior. The array-pointer windows have a documented bounds proof and focused checkptr coverage. Generated code removes per-vector bounds branches and shift broadcasts, with no hot helper calls or vector spills; the producer frame remains 608 bytes. These explain the implementation, not an independent speed claim.

The last full comparison, before recent optimizations, measured Go/dav1d decode **2.65255×** and Go/SVT encode **2.85944×** elapsed time. Decoded pixels matched exactly. Encoder settings were matched, not exact quality/search: PSNR was higher and SSIM lower. Native binary provenance was not attested to the audited source pins (dav1d b546257f770768b2c88258c533da38b91a06f737, SVT c04f951541ad600e0d9c10836f2ab7b9bc69816d). This specialized screen does not recompute or extrapolate those historical ratios.
