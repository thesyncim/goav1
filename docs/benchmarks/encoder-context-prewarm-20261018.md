# Remove repeated encoder context, mask, and warmup work

The candidate combines three byte-preserving Go changes. The final integrated screen saves **3.682% / 4.249% paired process wall time** on FourPeople / Johnny against main `6605b664`, passing every pre-data guard. All three pairs are faster in both cases and all encoded hashes match. The individual percentages below must not be added.

- Final inter commits use the existing decoder context fusion for six encoder call sites. Motion-first validation and partial-error state remain intact, and the independent setters remain available. Zeroing above/left fields in their existing update loops removes ten short `memclr` calls. The helper shrinks from 726 to 585 disassembly lines and from a 336-byte to 320-byte frame, with the same six grid copies. V1 inspection found call spills/reloads, rather than a new argument-ABI overflow. These static counts are diagnostics, not speed estimates.
- Serial, single-tile inter encoding builds loop-filter masks as it commits blocks in coding order. A reusable builder carries existing transform-edge geometry and eliminates the later full-map mask rebuild. The generic per-MI records, level resolution, CDEF consumers, keyframe path, and parallel/multitile paths remain available. This removes only the mask-construction fraction of filtering.
- Fresh single-layer prewarm uses the keyframe's existing golden allocation and visits both P reconstruction buffers, preserving the original buffer parity. Default construction runs one black key plus three black P frames instead of one plus seventeen. Temporal-layer, reused, and statistics-enabled encoders retain the legacy warmup because they retain state that can depend on its length.

No coding tools, search effort, quantizer decisions, or output expectations were reduced. The source is Go only. Decoder context fusion is shared, but this screen makes no decoder-performance claim.

## Isolated screens

Each screen used the original main `10f4a6ae`, default PGO, Go 1.27.0 with `GOEXPERIMENT=simd`, Darwin/arm64 on Apple M4 Max, `GOMAXPROCS=1`, `GOGC=100`, encoder MaxThreads=1 and effort=0. The two cases were 120 frames of 1280×720 60 fps FourPeople at 3 Mbps and Johnny at 9 Mbps. Statistics and frame statistics were disabled. Coordinated agent CPU work paused for each timing window; unrelated host activity was not controlled.

There were three alternating AB/BA/AB pairs per case after one warmup per binary/case. Each screen also retained one candidate/baseline pair on the first eight FourPeople frames with every frame forced key. All 18 rows per screen are preserved. The control still includes constructor inter warmup; it is neither a pure steady-state keyframe measurement nor an assertion that inter code was unreachable.

| Variant | FourPeople paired wall saved | Johnny paired wall saved | Key control wall saved | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Context V1 | +0.518% | -0.822% | -0.765% | Not a supported win |
| Direct masks V1 | -6.088% | +1.678% | -3.911% | CPU improves; retained wall stall prevents a combined wall claim |
| Masks + context V2 | +1.469% | +0.135% | -1.233% | Narrow FourPeople gain; Johnny inconclusive |
| Prewarm only | +2.226% | +2.926% | +7.894% | Narrow process-inclusive gain on both cases |

The statistic is `100 * (1 - exp(mean(log(candidate_wall / baseline_wall))))` within paired cases. These are small screens with no confidence-interval claim. Direct-mask V1's third FourPeople candidate had 262 ms of wall time beyond its child CPU time; that row remains included. Its CPU changes were +1.543% / +1.608%. Combined V2 CPU changes were +1.436% / +0.455%. Prewarm CPU changes were +2.416% / +2.659%; one FourPeople pair regressed in both wall and CPU time and remains included. No unchanged variant was resampled and no row was excluded.

The prewarm key control improves by +7.894% wall / +7.919% CPU in its single pair, consistent with the constructor mechanism. It does not quantify a steady-state encode improvement. Earlier sampled constructor scope was about 3–4% of the full operation; the removed throwaway frames do not explain the whole native gap.

## Final integrated screen

The frozen timing pair compared main `6605b664` with candidate source `84d91a157a5c0f531ece59800966b5aaa8138d12`. Before observing data, root declared acceptance as at least 0.5% paired geometric-mean wall savings and at least two of three faster pairs on **both** 120-frame cases, neither case exceeding 1% CPU regression, and every emitted hash matching its fixed fixture. Root then ran the exact script and independently recomputed all eighteen retained rows, hashes, wall statistics, and child user+system CPU statistics. Every guard passed.

| Case | Paired wall saved | Paired CPU saved | Faster wall pairs | Baseline / candidate wall median |
| --- | ---: | ---: | ---: | ---: |
| FourPeople 3 Mbps | +3.682% | +4.012% | 3/3 | 1.073765 / 1.014812 s |
| Johnny 9 Mbps | +4.249% | +4.666% | 3/3 | 1.428042 / 1.367891 s |
| Eight-key control | +8.690% | +9.190% | 1/1 | 0.328844 / 0.300266 s |

Per-pair wall savings were FourPeople +0.309%, +4.777%, +5.871% and Johnny +4.212%, +3.678%, +4.853%. The single key-control pair remains descriptive and includes the changed constructor warmup. Process medians and paired geometric means are reported separately; neither substitutes for the other. No outlier removal, unchanged retry, or additive attribution was used.

- Main binary SHA-256: `2402cf1e541fa3e94f5b54a845203d83b23e159341c51c09a011ac9db312cebe`.
- Integrated binary SHA-256: `6897d8c91eb43f45d53aa942f2d4210b8f5ba0dcf0f3d51e824f9c94eb6d8545`.

For publication, the two source commits were rebased onto decoder-only main `6c755647`, producing source commit `b8c20b66d2c5715960ed555db4d42daf607058ad`. Every changed source file has the same hash as the timed candidate. The final public L1T1/L1T2 encoded-byte and reconstruction fixture passed exactly after this integration. Timing remains attributed to the frozen `6605b664` pair; it was not repeated after the decoder-only rebase.

## Byte, state, and allocation gates

The integrated targeted tile and encoder checks passed. The existing frozen context oracle compares complete poisoned state, including stale fields, chroma conditions, invalid inputs, and motion-first partial errors. A new actual-encoder mask oracle compares every raw mask region against unchanged `BuildFromMap` output across frame reuse, partial superblock rows, and a keyframe transition.

The frozen original `Prewarm` body independently checks the candidate's returned reconstruction before the first real frame, retained rate/motion/buffer state, and twenty later encoded/reconstructed frames with a forced keyframe. Focused cases cover fresh CBR/fixed-Q, buffer parity, disabled golden, temporal state, reused state, and enabled statistics. The first externally visible keyframe plus both P reconstruction buffers allocate zero; the prewarmed ordered-mask encode check also reports zero allocations.

A separate public 192×128, twenty-frame moving/chroma fixture in both L1T1 and L1T2 matches original main and current main exactly for encoded bytes, reconstruction hashes, lengths, key positions, and temporal IDs. Every isolated and integrated real-case sample and warmup also matches baseline encoded hashes: FourPeople `37130ff37d12dee5c5279de42b51aa86ef25f17073a78bdd842c4402d88b2c67`, Johnny `b1140a4c298f975c1d6330e2d72e62276fc184c1926e36e8129e00499a0963c5`. The process helper itself does not hash encoder reconstruction; the separate fixture supplies that check.

The companion JSON retains exact commands, all process wall/user/system times, hashes and byte counts, isolated results, frozen source/binary identities, and the focused gate output. No broad per-prototype validation matrix or new native performance campaign was run. These results do not update the whole-corpus SVT gap or establish equal-quality speed.
