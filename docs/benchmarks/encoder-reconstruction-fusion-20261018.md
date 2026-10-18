# Fuse encoder reconstruction output and row-bound collection

The integrated candidate saves **0.646% / 1.436% paired process wall time** on FourPeople / Johnny against main `b9e260f4`, with **0.864% / 1.354% child CPU savings**. It passed the predeclared gate and every encoded hash matched. This is one small ARM64 screen of the combined implementation; no row-only or additive speedup is claimed.

The Go changes remove two traversals from the nonzero, non-DC, non-IDTX inter reconstruction path. A private kernel reads prediction and raw inverse-transform scratch and stores the rounded/clipped byte directly, avoiding the preliminary prediction copy. Dequantization collects the maximum coefficient row while visiting the same EOB prefix, avoiding the later scan walk. Zero-valued prefix entries remain included. All-zero, DC-only and IDTX behavior, transform/quantizer/search decisions, generic dequant APIs and entropy/CDF updates remain unchanged.

Pinned SVT `c04f951541ad600e0d9c10836f2ab7b9bc69816d`, `Source/Lib/ASM_NEON/av1_inv_txfm_neon.c:938`, supports separate prediction-read and reconstruction-write pointers. The independent oracle preserves the original Go arithmetic: widen before adding 8, shift 4, saturate to int16, add prediction and clip to byte. Partial buffer overlap keeps the exact original complete-copy-before-add behavior; exact in-place and disjoint buffers use fusion. AMD64 SIMD retains its original copy/add path pending native evidence. Production implementation is Go/archsimd only.

## Retained screens

Both variants used the same frozen main `b9e260f470c33a5369b5357d74a54b302044ea74` binary, default PGO, Go 1.27.0 `GOEXPERIMENT=simd`, Darwin/arm64 Apple M4 Max, GOMAXPROCS=1, GOGC=100, encoder MaxThreads=1 and effort=0. Inputs were 120 frames of 1280×720 60 fps I420 FourPeople 3 Mbps and Johnny 9 Mbps. Statistics/frame statistics were disabled. Root serialized timing while coordinated agent CPU work paused; unrelated host activity was uncontrolled.

Each screen retained 4 warmup rows, three alternating AB/BA/AB pairs per affected case, and one candidate/baseline pair on eight FourPeople frames with every frame forced key. The control includes constructor work and is descriptive. Before data, acceptance required at least 0.5% paired geometric-mean wall savings and at least 2/3 faster pairs in one affected case, no greater than 1% wall or CPU regression in the other affected case, and exact fixed hashes. No row was removed and no unchanged candidate was retried.

| Variant / case | Paired wall saved | Paired CPU saved | Faster wall pairs | Decision |
| --- | ---: | ---: | ---: | --- |
| Output V1 / FourPeople | +0.492910% | +0.321376% | 3/3 | Below 0.5% gate |
| Output V1 / Johnny | +0.345930% | -0.424024% | 1/3 | V1 rejected |
| Output + rows V2 / FourPeople | +0.645504% | +0.863558% | 3/3 | Pass |
| Output + rows V2 / Johnny | +1.436348% | +1.354261% | 2/3 | Pass |

V2 per-pair wall savings were FourPeople +0.578%, +0.477%, +0.880%; Johnny +2.777%, +2.204%, -0.707%. Baseline/candidate process wall medians were 0.993510/0.986973 s for FourPeople and 1.361738/1.331723 s for Johnny. The statistic is `100*(1-exp(mean(log(candidate/baseline))))`; paired geometric means and process medians are distinct. V1 was not rounded across its threshold. The single key controls were V1 -0.531% wall / +0.250% CPU and V2 -0.451% wall / -0.909% CPU. They are not steady-state keyframe claims.

All 36 actual rows, commands, child user/system times, fixed hashes, scripts and independently recomputed statistics are retained in [the evidence JSON](encoder-reconstruction-fusion-20261018.json). Timing identities:

- Baseline binary SHA256 `5233f6c1a32472dd786f2443306f0e025536eb48d3d4fb09d180fceb170b5579`.
- Output V1 source `f394b5beea55cfe95ba81269bebfc2a4df4656b9`; binary `6cf4756012d62a6740b0e38918ebe0a2b8f9994bdaa9ae748717ad68dd070387`.
- Integrated V2 source `53978627b94f636868d5bce57daa5cee3aa7eeff`, tree `23b58c199a449cc7084dab52b74a7948ff2e49a9`; binary `1fcfa9117f921af90fce7c05f75341e51140576e3f46982061d8e0e5c9d56a3b`.

## Correctness and cost evidence

The frozen original complete reconstruction oracle compares errors, output/guards, dequant scratch, inverse scratch and inverse-residual state across supported geometries, zero/DC/nonzero EOBs, IDTX and invalid transform types. A separate original two-pass numeric oracle tests every byte prediction against raw int32 rounding/clipping boundaries, exact windows, odd heights, padding, exact aliases and partial/cross-row/differing-stride overlaps. SIMD and scalar gates passed, as did existing public zero-allocation coverage.

The row-bound oracle independently spells out int64 dequantization and a separate scan traversal. It covers all small-block EOBs, larger boundary EOBs, custom scan orders, zero coefficients on the last row, int16/quantizer extrema, bit depths 8/10/12 and scales 0/1/2. Invalid scan positions and short prefixes preserve partial destination state at panic. The public 20-frame 192×128 moving/chroma fixture in both L1T1 and L1T2, including a forced keyframe, matches original encoded bytes, reconstruction hashes, lengths, keys and temporal IDs. Every screen row matches the fixed FourPeople/Johnny/key-control hashes. The process helper itself does not hash reconstruction; the public fixture supplies that check.

Generated ARM64 confirms no calls inside output vector loops. The normal raw route removes the preliminary copy dispatch and the public DSP wrapper's 192-byte plane-block materialization frame, while retaining one indirect dispatch, a 128-byte dispatcher frame, one width-leaf call and the original 32-byte 8-wide prediction-pair staging frame. The 4-wide/wide leaf loops need no frame. No extra per-vector call or spill was introduced.

The V2 dequant loop has inline scalar math, AND/ADD/CMP/CSINC row collection, no division, and no per-coefficient calls or stack accesses. Its helper frame is 48 bytes versus the generic 64; the reconstruction caller is 224 bytes versus V1's 240. The existing destination memclr remains once per block. Before timing, root recorded the dense-prefix tradeoff: the old separate scan stopped at full height, whereas V2 updates the row maximum for every EOB entry. These static differences explain the implementation, not a promised speedup.

Existing raw-profile inspection further bounds the affected work. Johnny's 30 ms copy samples included 20 ms on raw output and 10 ms on prediction scratch: only about 0.54% of 3720 ms sampled CPU was on the removed copy route. FourPeople's two copy samples were all-zero output and scratch copying; coarse sampling does not prove the raw route absent. Separate row scanning had 10/30 ms FourPeople/Johnny. Neither the entire 0.81% copy leaf nor the 8.33% enclosing reconstruction scope is removable. No new profile or native benchmark campaign was run.

## Integration

For publication, the two source commits were rebased onto decoder-only main `5484d71c3b5c1f5b7ee6bafc225425c9cf25ee2b`, producing source `4c7a6ff2eac3cd168296d3cf06452214899dd4c9`. All seven changed production/test file hashes match the timed V2 candidate exactly. The final focused encoder/quantize parity gates passed, and the existing public L1T1/L1T2 encoded-byte and reconstruction fixture matched exactly after integration. The final fixture binary SHA256 is `4dd48ff7e95dc9b9a8b2d64b4ef27bf716a0210192e7e467bbf8ccdfefd68377`. No timing was repeated: results remain attributed to the frozen b9e260f4 pair. This screen does not update the whole-corpus SVT gap or establish equal-quality speed.

Publication then incorporated entropy PR26, main `20763f4c3f4c6db29a9ba7b147a7de5f2afa05e8`, producing source `fb53b580711969b604f9596deefc1e29e0c91d0c`. All seven measured source/test hashes remain identical and the rebase had no conflicts. The separate entropy/tile changes expose the same immutable scan metadata to the writer; coefficient/EOB ordering and reconstruction interfaces are unchanged. Root independently verified PR26 against fixed encoder bytes. Completed parity gates and timing were not repeated for this source-independent integration.
