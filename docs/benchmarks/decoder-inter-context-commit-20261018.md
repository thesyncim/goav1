# Commit inter contexts once — 2026-10-18

This change removes the motion-validity clear and duplicate block-size grid population immediately before the same inter block fills those fields. The decoded path also commits reference, motion and filter edge fields together and combines the two motion/filter grid walks. This follows dav1d's final block-context/ref-MV splats (`src/decode.c:514–559,1902–1925`, pinned `b546257f770768b2c88258c533da38b91a06f737`). It removes bookkeeping only; motion vectors, filters and predictor math are unchanged.

Public `MarkInter`, `MarkInterMotion` and `MarkInterFilters` retain their standalone state/error contracts. Shared validation keeps the original motion-before-context and nil/size/coordinate/reference order. Invalid filters take the original two-step fallback, including the already-successful motion update and exact error prefix. Sub8x8 chroma collects previous grid cells before the combined commit. `ReadInterMotion` preserves the decoded references, and no callback or reader observes the old intermediate state. Every omitted clear/store is replaced by its final value; unrelated and validity-gated stale payloads stay unchanged.

| Warm public case | V2 paired geometric-mean latency saved | Pair savings | Faster pairs |
|---|---:|---|---:|
| p720_inter_q32 | 2.1153% | +4.7396%, −1.1825%, +2.6967% | 2/3 |
| p360_intra_q32 control | 2.9452% | +2.9973%, +3.6989%, +2.1332% | 3/3 |

Three sequential pairs, 15 complete 48-frame decodes per row, one worker, GOMAXPROCS=1/GOGC=100, Go 1.27.0+simd, Darwin arm64/v8.0, Apple M4 Max. Baseline is merged PR15 `45dc043b`, frozen binary `aab588cb0814de3d754d0d3b56a20b5c0b09641fdc6a2bb6bc7ef691a51a2a6b`; V2 is `6230c7007a86e6b6c2f8a03ac51fa9bf850fa6f6ecce9ddc8975f1f85941284f`. All 12 rows reported 0 allocations/op; a few rows reported 3 B/op on both roles. Exact role order, rows and source/binary/log hashes are in the [raw evidence](decoder-inter-context-commit-20261018.json). These are short-screen estimates. The unchanged intra path is a descriptive control, not a claimed causal gain, and no confidence interval or aggregate decoder gain is asserted.

V1 combined motion/filter commits but retained a separate reference-edge pass. Its disturbed screen estimated inter −22.2231% and intra +8.0455%; inter rows ranged 180–312 ms. An unrelated recursive process tree was subsequently identified and removed by the parent after user approval. All V1 rows remain separate in the evidence. V2's single source-justified iteration fuses the remaining reference-edge pass; its first and only screen ran after cleanup. No resampling, exclusions or pooling.

Focused parity tests passed for V1 (0.784 s) and V2 (0.715 s): frozen original MarkInter/Motion/Filters bodies compare full named context state over poisoned scratch, all block sizes, boundary positions, single/compound references, chroma ownership and filter combinations. Invalid inputs, nil receivers, exact motion-first errors, partial updates and sub8x8 neighbor capture are covered. The single opted-in public MD5 gate passed 18 clips / 864 frames against independent aomdec/dav1d sidecars through IVF and payload constructors (5.04 s). Self-review found no changed numeric operations, state layout, allocation contract or callback/error ordering. No encoder path changes or extra encoder validation were needed.

The larger private cross-superblock motion-history ring remains a separate R3/R7 follow-up. The prior PR13 ~2.71× cold-IVF dav1d ratio is historical and unremeasured here; no current native-gap claim is made.
