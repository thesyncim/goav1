# Decoder-owned MV grid reuse

The high-level decoder previously cleared and populated an invalid MV grid for ordinary intra frames, and copied every completed grid into each refreshed reference slot. This change shares immutable completed grids through a fixed nine-buffer pool. Ordinary key/intra-only frames without intrabc retain their reference metadata and omit the unused grid. The default low-level state keeps independent mutable copies, and shared/SVC routing stays unchanged.

Pinned dav1d `b546257f770768b2c88258c533da38b91a06f737` uses conditional MV allocation and reference-counted publication in `src/decode.c:3571–3610,3685–3692`. All prediction, projection, rounding and entropy math is unchanged. Intra descriptors remain non-nil when resolving references because Setup uses presence for LastOverlay/RefStamp before skipping intra projection.

A fixed three-pair warm Reset+Decode screen compared the candidate with PR19 `10f4a6ae956e881e8d9e85a4797c55a441cd27c3` on an Apple M4 Max, Go1.27.0-Xsimd, ARM64v8.0, one worker/GOMAXPROCS1, GOGC100. Each row decoded the full 48-frame clip 15 times. Both cases are affected.

| Case | Paired GM latency saved | Faster pairs | Individual pair savings |
|---|---:|---:|---|
| p720_inter_q32 | 1.2401% | 2/3 | −0.2615%, +0.0271%, +3.8995% |
| p360_intra_q32 | 1.1783% | 2/3 | −1.2627%, +2.8439%, +1.9071% |

The predeclared screen rule required at least one case >=0.5% with >=2/3 faster pairs, neither case >1% regression, and zero warm allocations. It passed; all 12 rows were retained, each invocation produced exactly one anchored benchmark row, and source/binary hashes were unchanged. The inter estimate is strongly influenced by its last slower baseline. These are noisy short-screen estimates, not a demonstrated whole-corpus or native-gap improvement. There was no retry, exclusion, variant pooling or unchanged resampling.

Focused tests passed in 0.370s. A frozen original prepare/publish oracle compares 48 mixed frames, refresh masks, resizing, temporal payloads and Setup counters. Tests also cover generic mutable isolation/private immutable sharing, nine-buffer capacity, error/abort/finish/Reset lifetimes, show-existing/shared/intrabc fallbacks and zero warm allocations. After integration with deblock PR20 `6605b664622f9ea290a6543662f51b19b923d873`, the same focused state selection passed in 0.357s and the single public 18-clip/864-frame MD5 gate passed in 5.20s, for both IVF and pre-parsed payload constructors. No R7 production/test bytes changed during integration and no timing was repeated.

Measured baseline binary SHA256: `ee822414026a0fa7df468c41026a850c9d8268834d4b7c5e6fae0580a6d9f8e6`. Measured candidate: `6f69413a0296672310718e022d19753bd4d1de6ea883737d152647cdba3cf6a8`. Final integrated binary: `c810e2f47b85398ef96b00c7a217a3d9fecdb6de20514ec8e0026ec35e9008ac`. Source hashes, exact ordering and all raw rows are retained in [the companion evidence](decoder-mv-grid-pool-20261018.json). No encoder-path change or new dav1d/SVT ratio is claimed.
