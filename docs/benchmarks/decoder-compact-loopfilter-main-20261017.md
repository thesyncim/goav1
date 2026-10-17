# Compact loop-filter state on the default decoder path

## Scope

This candidate is based on clean current main commit 3747195174c8b5f3d8491757ac0593dcb7ee19a2 (tree 170ec2914f66e4e13d5107f8c17ca6dbaec668df) and contains only the compact loop-filter change and its tests. It does not contain PR9 coefficient fusion or another experimental candidate.

The default public decoder selects the privately owned fast route automatically for 8-bit frames with one worker, one tile, nonzero luma loop filtering, and no external hooks, super-resolution, intrabc, or external ownership. Other public/generic and ineligible routes retain the legacy record-map behavior.

During block decode and mask construction, the eligible route writes the existing compact 4-byte-per-MI level-cache entries and the packed CDEF no-skip state directly. It avoids filling the replicated 34-byte-per-MI record map and scanning that map later. The map backing remains available for fallback, and the existing level cache remains. This change makes no memory-saving claim: allocation counts match, while measured B/op differences range from -4 to +4 bytes.

The exported mask and side-data layouts are unchanged. The private FrameWorkState grows by 8 bytes; FrameWorkBatch remains 1608 bytes and the existing residual scratch guard remains 89328 bytes.

## Paired decoder confirmation

Both frozen normal optimized binaries passed the 18-clip, 864-frame MD5 matrix at worker counts 1, 2, 3, 4, and 8. Both also passed the fixed p720_inter_q32 encoder bitstream and reconstruction hashes recorded in the evidence bundle.

The confirmation ran on Apple M4 Max, macOS Darwin/arm64, with Go 1.27.0-X:simd, GOEXPERIMENT=simd, GOMAXPROCS=1, and GOGC=100. It used 10 AB/BA pairs per case, 15 operations per sample, and 48 frames per operation. The point estimate is 100 × (1 - exp(mean(log(candidate_ns_per_op / baseline_ns_per_op)))) per case. The 95% casewise percentile intervals use 20,000 resamples with replacement via random.Random(23117).choices(paired_case_rows, k=10), one RNG stream carried across cases in fixed order: p360 intra cold, p360 intra warm, p720 inter cold, p720 inter warm. Percentile endpoints use linear interpolation. All 80 planned rows are retained. No cases are pooled, and the earlier exploratory screen is not combined with this confirmation.

| Case | Latency saved | 95% casewise interval | Candidate faster pairs | Result |
| --- | ---: | ---: | ---: | --- |
| p360 intra, warm reset | 1.0443% | [0.6699%, 1.4623%] | 10/10 | Supported |
| p720 inter, cold IVF | 3.5417% | [2.7548%, 4.4265%] | 10/10 | Supported |
| p720 inter, warm reset | 2.1274% | [1.4740%, 2.8020%] | 10/10 | Supported |
| p360 intra, cold fresh decoder | 0.4375% | [-0.1056%, 0.9862%] | 7/10 | Inconclusive |

The confirmation ran under high host load: 106.05–146.57 across 16 logical cores, with a reported mean of 128.72. These observations belong only to the confirmation run. The raw resource snapshots and all commands are embedded in the evidence bundle.

## Public comparison with installed dav1d

The separate two-round, 18-clip comparison measured aggregate Go cold-IVF medians of 1,491.840313 ms against 540.800 ms for dav1d, a 2.758580× ratio. Round ratios were 2.789858× and 2.723023×; run-order drift was -2.531431% for Go and -0.139121% for dav1d. The statistic sums the per-clip medians, each formed from the middle pair of 18 measurements. No paired-significance claim is made.

This is a descriptive Go-versus-dav1d gap for the frozen candidate, not a speedup caused by the loop-filter change. Go's timed NewDecoderFromIVF path parses IVF and copies packet payloads. The C harness parses a preloaded whole-file IVF buffer and wraps payload slices without copying. The installed runtime reports dav1d 1.5.3. The harness records reference commit b546257f770768b2c88258c533da38b91a06f737; the installed library's source checkout and build provenance are unverified, and the reference commit is not asserted as the installed binary revision.

The native-gap runner verifies all 18 clip MD5s for Go's IVF constructor, Go's payload constructor, the C helper, and the sidecars in both rounds. Helper and linked-library path/hash identities matched before and after. The native run records serialized backend wall intervals but no per-sample CPU/load snapshots.

## Local validation details

Focused native, purego, and race gates passed, including compact-route parity/lifecycle/fallback checks. The normal warm Reset+DecodeNext allocation test measured zero allocations. Under strict checkptr, semantic-filtered tests passed; the root warm-zero allocation assertion reports 6 allocations/run, and the internal CDEF allocation assertions report 1 and 24 allocations. Those two internal assertions fail identically on the clean baseline under the same checkptr setting. These failures are preserved; thresholds and production code were not changed to hide them.

The JSON evidence files beside this document contain both frozen identity records, the candidate source manifest, all 80 raw confirmation rows, confirmation manifest and environment, raw commands/logs/resource snapshots/checksums, gate commands and logs, the full native summary including per-clip raw sample arrays, plus native commands/environment/checksum texts.
