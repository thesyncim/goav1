# Reset private decoder CDF counters without rescanning probabilities

After a successful decoder-owned tile job, initialized CDFs have been changed only by entropy adaptation. The private retention path now clears their adaptation counters directly. Generic `ResetCount`, `ResetCDFCounts`, and batch retention still validate writable `CDF.Values()` data. The trusted operation is a free function in the internal entropy package, preserving the public CDF method set.

Dispatch requires the existing opaque compact decoder capability, matching masks, no caller hooks, and no wavefront execution. It currently covers private 8-bit, single-tile, single-worker, non-superres/non-IBC frames with active deblocking. Every field is reset, including inter CDFs on intra frames. Storage layout, retained-state copies, and generic error/mutation/publication order are unchanged.

The pinned source is dav1d `b546257f770768b2c88258c533da38b91a06f737` (1.5.3), `src/cdf.c:3916 onward`: `dav1d_cdf_thread_update` copies the probability context and clears known count positions. The supplied warm intra profile sampled Go count reset/validation; this does not measure a native stage ratio. Linked arm64 compiler output confirms the trusted coefficient reset has no ordinary per-CDF calls, replacing the generic reset/validate call chain. Its frame shrinks from 192 to 16 bytes; the root caller stays 2272 bytes. The new ownership wrapper has a 64-byte frame and adds mask/hook checks once per retention.

## Paired public screen

Base `b9e260f470c33a5369b5357d74a54b302044ea74`, tree `3b988381227b930c4c2c2d5555c836a583580f25`. Frozen baseline SHA256 `2c7eb87ae4fcc3dfdb22dccb3e7ae7630e915214ffc042e43f7b59f9000cfd95`; candidate `c38c1df9c4a4eb2d0e25d1494b36447ff9440684461c933c555dbbf081cfcaef`. Darwin/arm64 Apple M4 Max, Go SIMD build, `GOARM64=v8.0`, `GOMAXPROCS=1`, `GOGC=100`, one decoder worker, warm `Reset` + `DecodeNext`. Each operation decodes 48 visible frames. Three alternating pairs ×20 operations per process produced exactly 12 intended rows. Every slash component was anchored; no rows were omitted or rerun. Coordinated agent CPU work was paused; other host activity was not controlled.

| Case | Paired latency saved | Faster pairs | Individual pair savings |
| --- | ---: | ---: | --- |
| p360 intra q32, target | 1.09199% | 3/3 | 0.06792%, 2.45108%, 0.74165% |
| p720 inter q32, affected comparison | 0.27228% | 3/3 | 0.01344%, 0.23609%, 0.56654% |

The predeclared gate required at least 0.5% target saving by `100 * (1 - exp(mean(log(candidate_ns / baseline_ns))))`, at least two faster target pairs, no more than 1% comparison regression, and zero allocs/op in every row. It passed. All 12 rows report zero allocs/op; B/op is 0 or 2 and was explicitly retained without a zero-byte gate. No memory reduction is claimed. Process-end one-minute host load ranged from 6.25 to 9.43. Three pairs support a narrow measured intra improvement, without a confidence interval, an updated whole-corpus dav1d gap, or an all-workload claim.

## Exactness checks

An independent reflection oracle covers all 1570 CDF slots (1463 active, 107 unused) in every coefficient q-context, comparing the complete probability storage after clearing only observed count positions. Tests cover 2–16-symbol CDFs, saturation, inactive-storage canaries, each hook fallback, malformed writable values, partial mutation before error, disabled retention, and non-update jobs. All pass.

An untimed MD5/count overlay passed both public clips: intra used 48 trusted/0 generic resets; inter used 49 trusted/3 generic resets. Four existing 8/10-bit inter/edge golden fixtures and the public warm allocation fixture passed. Those public checks used the operation-equivalent method-form draft; after review moved it to the internal free function, focused state tests passed again and compiler inlining/frame sizes were unchanged. Only the reviewed free-function binary was timed.

The exact frozen selected candidate passed the independent MD5 sidecars for all 18 clips / 864 frames through both IVF and pre-parsed-payload constructors at one worker. No encoder production call changed. No broad race/worker matrix or native timing campaign was run. The companion JSON contains every raw benchmark row, hashes, the original protocol, exact gates, source/compiler proof, and test logs.
