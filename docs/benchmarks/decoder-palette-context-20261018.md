# Palette context equality cases — 2026-10-18

The decoder now derives palette context and color order directly from top/left/diagonal equality, following dav1d `b546257f770768b2c88258c533da38b91a06f737`, `src/decode.c:order_palette`. The normal decoded-map path avoids weighted score construction, stable insertion sorting, and the context lookup. The original scorer remains for noncanonical neighbor values. Palette-size/origin validation, neighbor access order, full eight-color order, entropy symbol order, and CDF adaptation are preserved.

| Public warm-reset clip | Paired geometric-mean latency saved | Pair savings | Faster pairs |
| --- | ---: | --- | ---: |
| 64×64 8-bit palette, 3 frames | 7.0158% | 4.8317%, 6.6737%, 9.4828% | 3/3 |
| 64×64 10-bit palette, 3 frames | 20.8061% | 20.7094%, 20.4100%, 21.2964% | 3/3 |
| 64×64 8-bit control, 3 frames | 0.3966% | 0.3949%, −0.4892%, 1.2763% | 2/3 |

The equal-weight palette aggregate saved 14.1875%, with all three aggregate pairs faster. This passed the predeclared short-screen rule: aggregate saving ≥0.5% and ≥2/3 faster pairs; each palette case ≥2/3 faster pairs and at most 1% geometric-mean regression; control at most 1% regression; zero allocations and passing hashes throughout. The control difference is descriptive. This is not a significance or whole-corpus claim.

The screen ran three sequential A/B pairs, reversing order for pair 2, with 500 complete Reset+DecodeNext clip operations per row (1,500 frames). Each process verified every frame against independent libaom aomdec sidecars before timing. IVF reads, constructor, Close, and hashing were outside ns/op; process wall/CPU measurements remain in the raw evidence. All 18 rows reported zero B/op and allocs/op. No rows were discarded and no unchanged candidate was resampled.

The frozen baseline is `10f4a6ae956e881e8d9e85a4797c55a441cd27c3`, public test binary SHA `4646756895e8f0d8b2738c7bed1aeea279122ed71784bd4acc6224cb9cb38892`; candidate SHA `c7d348722c9bd5b64a39e977464a12d7f6f11f7daee1453bc900783f3ad68e24`. Both use the identical public harness, Go 1.27.0, GOEXPERIMENT=simd, Darwin arm64/v8.0, one decoder worker, GOMAXPROCS=1/GOGC=100, on Apple M4 Max. No root default.pgo was present.

The exhaustive frozen-old oracle matched 5,069 complete context/order/error/panic outcomes, including palette sizes 2..8, all legal neighbor combinations, missing edges, noncanonical values, and malformed geometry. SIMD and scalar/purego builds passed, with zero context allocations. Cold and reset public decoding matched every independent frame MD5. A diagnostic-only existing profile-test overlay confirmed live luma/chroma palette blocks: 13/11 at 8-bit and 65/65 at 10-bit; the control had 0/0. The committed sidecars copy pre-existing independent aomdec goldens from `conformance/publicpath_test.go` at the baseline commit; no Go-generated expectations were introduced.

Generated ARM64 code reduces the normal function's stack frame from 240 to 80 bytes and lowers unused-color enumeration to RBIT/CLZ without helper calls. The original inverse-array stores were already eliminated by the compiler; two associated bounds checks survived. This change does not claim removal of an inverse-array write pass.

These two three-frame palette fixtures are absent by both filename and SHA256 from every input in the current 18-clip native-gap manifest. The saved manifest identity and all 18 input hashes are in the [raw evidence](decoder-palette-context-20261018.json), together with the fixed protocol, all timing rows, source/codegen proof, and parity logs. No whole-corpus decoder saving or new dav1d-gap ratio is claimed. R2 intra gather/filter-copy work is separate and unmeasured here.

After one rebase onto current main `6c755647026d88b0111a48751b014e5d7f48f5d9`, the focused exhaustive-order/allocation and public cold/reset MD5 gate passed again. Production and harness source hashes remain identical to the measured candidate. No new timing was taken; all performance attribution remains the pinned `10f4a6ae` comparison.
