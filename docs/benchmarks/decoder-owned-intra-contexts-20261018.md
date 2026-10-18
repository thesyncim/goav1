# Private intra contexts: remove unused motion bookkeeping

On ordinary key and intra-only frames, the decoder previously cleared and copied motion grids and cross-superblock histories that intra prediction never reads. The existing decoder-owned compact capability now selects an intra context path that omits that work. It retains partition, skip/segment, intra/chroma, transform edge sizes and neighbor snapshots, palettes, coefficient entropy, and CDEF contexts. Current-MV frame writes and generic low-level observable state remain unchanged.

The change is based on main `f05383c85c62ba39d0cd0055e66379f6fa0f839c`, which already includes PR9–12. Selection requires the existing private compact route: 8-bit, one worker, one tile, active luma loop filtering, and no external ownership, super-resolution, or intrabc. The new selection additionally requires a key/intra-only frame, disabled skip mode, the owned carrier, the default controller, and no caller callbacks, custom neighborhood, or wavefront reconstruction. Selection lasts for one call; no request or scratch layout changes.

## Source and state proof

The source comparison uses dav1d commit `b546257f770768b2c88258c533da38b91a06f737`. In `src/decode.c`, ref-MV splatting at lines 1263–1264 and row setup at 2604–2609 are gated to inter/switch frames or intrabc. No extra full reconstruction pass or higher-quality decoder search was found. SVT-AV1 is an encoder comparator and does not explain a decoder ratio.

The new root reset clears 2,949 bytes of live context rather than the generic 17,026-byte prefix. It also omits dead reference/filter/grid stores and five-deep top/left/diagonal history maintenance. On an untimed exact-source diagnostic, p360 intra reached the private path for all 48 frames, 2,880 roots and 13,466 block marks. P720 inter reached it only for its initial key frame: 60 roots and 243 marks. The p360 census implies 40,541,760 fewer source-level reset-span bytes and 3,116,556 fewer logical grid/edge store bytes per 48 frames. These counts are not measured memory traffic or a speed prediction. History copy costs are not added to these counts or converted into cycles. Reusable backing capacity remains available.

Original full root load/store, `MarkIntra`, and generic reset bodies are unchanged and form the old-state oracle. Tests compare every retained field, poison omitted state, and exercise partial private jobs followed by generic inter/IBC work. Per-job binding resets Above/Left; first-row/column gating and subsequent captures prevent stale diagonal history reads. Public tests also force the generic hook path and compare frames across key/inter/switch transitions and Reset reuse. Unsafe clears stay within pointer-free context spans. No numeric or entropy approximation is introduced.

## Fresh paired confirmation

Both frozen normal binaries use Go 1.27.0-X:simd, Darwin/arm64, `GOARM64=v8.0`, `CGO_ENABLED=1`, `GOMAXPROCS=1`, and `GOGC=100` on an Apple M4 Max with 16 logical CPUs. Candidate binary SHA-256 is `051696d1f3c19db886085c6467576bea776145eec054aa5bc7607d125e42b0ce`; baseline is `d220fe8fd1386333e5a70f5be75eea714b10c892c17f9ae353d99b9efc6a0d82`.

A separate 24-row screen passed its predeclared selection rule. The fresh confirmation then retained all 80 rows: 10 pairs per case, 5AB/5BA, 15 operations per sample, 48 frames per operation. The point statistic is `100*(1-exp(mean(log(candidate_ns/baseline_ns))))`. Casewise 95% percentile bootstrap intervals use 20,000 resamples, `random.Random(23117)` with 10 `randrange(10)` draws per resample, one stream in table order, and linear percentile interpolation. Screen samples are not pooled into confirmation.

| Case | Latency saved | 95% interval | Faster pairs |
| --- | ---: | ---: | ---: |
| p360 intra, cold IVF | 4.9704% | [3.0189%,6.8161%] | 9/10 |
| p360 intra, warm Reset | 3.7865% | [0.9674%,6.3344%] | 8/10 |
| p720 inter, cold IVF | -0.6434% | [-2.6354%,0.7969%] | 5/10 |
| p720 inter, warm Reset | 1.1253% | [-0.1685%,2.7481%] | 6/10 |

The exact predeclared rule passed: at least one intra saving ≥0.5% with a positive interval lower bound, no case point regression >1%, and no inter interval entirely adverse. The intra gain is supported. Inter outcomes remain unresolved; the cold interval permits a 2.64% regression, so these data do not establish inter flatness or an inter win. All 82 subprocesses exited 0 and all 168 resource observations passed. Load1 ranged 6.31–8.95, with zero `uv` processes. Every warm row reported 0 allocations/op; B/op was 0 or 3 for both roles.

## Fresh installed-dav1d snapshot

A separate descriptive full-corpus run used the same frozen baseline/candidate and an unchanged dav1d 1.5.3 C helper. The six serial invocations were baseline, candidate, dav1d, then dav1d, candidate, baseline. Each verified 18 clips × 48 visible frames against the independent MD5 sidecars before collecting one warmup and nine samples per clip. The statistic below sums each clip's median across 18 samples. All 2,268 raw timing values across Go's three modes and the C mode are retained.

| Public operation | Baseline sum | Candidate sum | Baseline/dav1d | Candidate/dav1d |
| --- | ---: | ---: | ---: | ---: |
| Cold IVF | 1467.976877ms | 1474.329438ms | 2.701839× | 2.713531× |
| Cold pre-parsed payload | 1472.674252ms | 1481.831729ms | 2.710485× | 2.727339× |
| Warm Reset | 1429.547395ms | 1427.579376ms | 2.631109× | 2.627487× |

Dav1d's fresh-context sum was 543.325ms. This cohort does not establish an aggregate cold gain or a smaller overall dav1d gap. The paired intra confirmation above is the evidence for the change. The historical 2.7586× snapshot is not a paired control or additive progress estimate. Native samples are clustered by backend and clip; no significance claim or confidence interval is inferred from this cohort.

Round 2 versus round 1 drift was -2.204%/-2.061%/-2.857% for baseline cold/payload/warm, +1.426%/+2.147%/+2.526% for candidate, and -0.0654% for dav1d. All 16 invocation-boundary resource observations passed; load1 ranged 5.77–8.76, with zero `uv` processes. They do not provide per-clip CPU utilization. All identities remained stable before/after.

Go cold IVF includes parsing and packet-payload copying; C parses preloaded IVF and wraps packet views without copying. Go cold payload excludes parse/copies. Go warm Reset has no matched native warm-reset comparator, so its ratio is a lifecycle diagnostic. Both use one decoder thread, visible output only and grain enabled. Helper SHA-256 is `d94ec98ccaf7324529de560d7eee25ad849f7c147f6c08e3f8baa727e2e8661c`; linked library SHA-256 is `bdd18b4f70a9e8956a58de310d8acfdad2ecf6b1be00b813b397e14331d5d0bd`. The installed library's source-build provenance is unverified; the pinned source commit is an algorithm reference, not proof of that binary's revision. The local corpus has checksum identities but no source manifest or declared grain coverage.

## Validation and retained exceptions

The 18-clip, 864-frame MD5 matrix passed with workers 1, 2, 3, 4, 8. The fixed p720 encoder check retained 48 frames, 720,250 encoded bytes and exact stream/reconstruction hashes. Full tile and decoder suites, threading semantic tests, ownership/state tests under default race and strict checkptr2, purego checks, and hot-struct guards passed. Linux/amd64 compiler reports matched the existing 515-entry escape allowlist without edits and reported 4967 BCE sites. `go vet ./...` and `go mod tidy -diff` passed. The three gofmt-reported files are byte-identical to main; all changed files are formatted.

An unchanged replay-only `TestReconWavefrontSpeedup` assertion failed the candidate at GOMAXPROCS1 (0.98×); the single same-config baseline run passed 1.07×. A single intended-parallel candidate run at GOMAXPROCS4 passed 2.94×. This test bypasses the changed decode path. Its failure is retained, not relabeled as a baseline failure. The existing public warm-allocation assertion reports 6 allocations on both exact baseline and candidate under strict checkptr2; normal/default-race public warm allocations are zero. Assertions were not weakened. An initial over-anchored race selector ran no tile/threading tests; that log is retained and the corrected tests passed. Initial compile mistakes and all rejected optimization attempts remain recorded.

The [main evidence JSON](decoder-owned-intra-contexts-20261018.json) contains both full source manifests, source proof, coverage, self-review, all 24 screen rows and 80 confirmation rows, protocols, commands, resource observations, raw logs, gates and exceptions. The [native evidence JSON](decoder-owned-intra-dav1d-gap-20261018.json) contains every native comparison sample, input/hash identities, medians, drift, commands and raw logs. The [work ledger](decoder-extra-work-ledger-20261018.json) records the broader source comparison and rejected DCT64 variants; none of those variants is included in this patch.
