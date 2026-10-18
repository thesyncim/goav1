# Inspect owned postfilter context through pointers

Private postfilter active/remaining-stage queries and completion marking now inspect the owned local Context through pointers. This removes repeated large value copies around stage queries. Public value-receiver methods retain snapshot behavior. The boundary-save query is private. The supported-stage value receiver remains because a pointer draft enlarged the film-grain fallback frame.

A frozen three-pair whole-clip screen on main c72c0fb0 measured 48-frame warm Reset+DecodeNext operations, 20 operations per process, workers=1, GOMAXPROCS=1, GOGC=100, Go 1.27.0 GOEXPERIMENT=simd on Darwin/ARM64 Apple M4 Max. Pair order and primary/secondary cases were fixed before data. All 12 rows are retained in the [evidence JSON](decoder-postfilter-context-20261018.json).

| Case | Paired geometric-mean time saved | Faster pairs |
| --- | ---: | ---: |
| p720_inter_q32 | 2.543372% | 3/3 |
| p360_intra_q32 | 1.729843% | 3/3 |

The predeclared gate required inter savings of at least 0.5% and two faster pairs, at most 1% intra regression, and zero allocations in every row. It passed. Root independently checked every row and recalculated `100*(1-exp(mean(log(candidate/baseline))))`. Every row reports 0 allocs/op; measured B/op ranges from 0 to 2 and is reported separately. No row was removed and no unchanged retry occurred. All agents were idle and before/after host checks were quiet; unrelated host activity is uncontrolled.

Generated code contains 55→34 actual static Context copy sites, plus one unchanged zero-initialization loop. These are alternative argument/return sites, not 21 copies removed from every frame. Remaining copies include 21 argument and 13 return sites. The large postfilter frame stays 13072 bytes; the private boundary query drops 11648→32 bytes. The inspected call chain drops 2076→1752 instructions. These static facts explain the implementation, not a native-stage speed attribution.

Focused state, stage order, error and public-snapshot tests pass. Profile-1 goldens and the existing 18-clip/864-frame independent output-hash corpus pass with both constructors. The initial attempted narrow full-corpus selector actually ran all 18 clips; its complete passing result is retained instead of rerunning it. The measured two source/test files are byte-identical after integration onto 0034a13b, whose extra changes are an encoder optimization and a test-only layout repair. Existing completed validation is reused.

The last full native comparison, before recent optimizations, measured Go/dav1d decode 2.65255× and Go/SVT encode 2.85944× elapsed time. Decoder pixels matched exactly. Encoder settings were matched, not exact quality/search; PSNR was higher and SSIM lower. Those ratios remain historical; this two-clip screen does not refresh or extrapolate them. Installed native binary provenance was not attested to the audited source pins.
