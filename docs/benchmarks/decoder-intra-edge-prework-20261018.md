# Remove repeated intra-edge prework — 2026-10-18

This change removes the zone-2 per-pixel reference-range prewalk and redundant value scans immediately after decoder edges are gathered from an 8-bit plane. It follows dav1d's direct `ipred_z2_c` edge selection (`src/ipred_tmpl.c`, pinned `b546257f770768b2c88258c533da38b91a06f737`), whose pixel-typed input is already bounded.

The range helper finds each row's above/left split and exact monotone endpoints. A bounded AV1 domain (dimensions 1–128, positive derivatives ≤1023, 0/1 upsampling) prevents overflow; other inputs retain the original walk. Full coded/visible extents and invalid-left errors remain unchanged. The scan omission is call-scoped: each used sample was just overwritten from byte pixels or 127/128/129 defaults; convex edge filters and clipped upsampling preserve that bound. General uint16 prediction APIs and the directly callable edge-filter helper still validate samples. Geometry, availability, slice bounds, modes, rounding and clipping are retained. No reusable trust flag, API layout or allocation growth.

| Warm public case | V2 paired geometric-mean latency saved | Pair savings | Faster pairs |
|---|---:|---|---:|
| p360_intra_q32 |1.6557%|+1.5620%,+1.1580%,+2.2440%|3/3|
| p720_inter_q32 control |0.8385%|−0.2792%,+2.7248%,+0.0422%|2/3|

Three sequential pairs, 15 full 48-frame decodes per row, one worker, GOMAXPROCS=1/GOGC=100, Go 1.27.0+simd, Darwin arm64/v8.0, Apple M4 Max. Baseline is merged PR14 `04a90b32`, frozen binary `b90eae34abebbb7744b85786ad589c82d15c810eb0f6b7054a28bdfa4a8bb9e9`; V2 is `aab588cb0814de3d754d0d3b56a20b5c0b09641fdc6a2bb6bc7ef691a51a2a6b`. All 12 rows report zero allocations; exact source/binary/log hashes, role orders and every row are in the [raw evidence](decoder-intra-edge-prework-20261018.json). These are short-screen estimates, not confidence intervals or an aggregate decoder claim.

V1 used additional non-inlined threading wrappers. Its separate three-pair screen estimated intra +0.8639%; an unusually slow 259 ms baseline inflated the inter control to +13.3635%, which is not claimed as a gain. V2's one source-justified iteration removes those wrappers; generated code directly calls the bounded cores. V1 rows remain in the evidence and are not pooled with V2.

Focused tests passed (prediction 0.280s, threading 0.354s): the frozen original pixel-range walk, existing libaom numeric fixtures at 8/10/12 bits, generic invalid-input checks, partial extents, byte-filter equivalence and poisoned reusable scratch. The opted-in public parity gate passed 18 clips /864 frames against independent aomdec/dav1d MD5 sidecars through both IVF and payload constructors (5.13 s). An earlier invocation omitted the opt-in and skipped; it is retained but not counted. A corrected edit-script assertion/partial-edit log is likewise retained outside the repository. No thresholds were relaxed.

No predictor math or encoder decisions changed. R1's 16-bit motion scratch experiment remains unmerged after inconclusive timings; it is not part of this patch. R2 gather/filter-copy fusion remains a separate follow-up. The prior PR13 ~2.71× cold-IVF dav1d ratio is historical and unremeasured here; this report makes no new native-gap claim.
