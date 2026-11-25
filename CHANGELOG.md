# Changelog

Release-level changes to goav1 are tracked here. Detailed progress logs,
per-vector archaeology, and work-in-progress notes belong in commits, issues,
or pull requests, not in long-lived markdown.

## [v0.1.2] - 2026-11-25

Pure-Go + `simd/archsimd` decoder performance release — brings 8-thread 5-stream totals to **1.040× wall / 1.030× cycles** of `dav1d --threads 8` (beating `dav1d` on `aomrt_johnny`, `svtrtc_crowdrun`, `aomrt_fourpeople` cycles and `svtra_mtbike` wall) and 1-thread totals to **1.101× cycles** on Apple M4 Max, with 100% memory safety (`scripts/check_unsafe.sh`) and 31/31 realcorpus bit-exact MD5 parity.

### Decoder Optimizations (`arm64` & generic)
- **Loop Filter (`internal/av1/loopfilter`)**:
  - Split `lpf8U8W` into `lpf8U8Cols` (true leaf function with zero `any(tv)` runtime type assertions and `BlockHalves` hoisted once) and `lpf8U8Rows`.
  - Gated `p6..p4`/`q4..q6` loads and `lpfSpread3` in `lpf16U8Wd16` on `m16 & mFlat != 0`, replaced `BroadcastUint16x8(128)` with L1 table load `lpfBias128`, sequenced low/high halves in flat8/flat6 to cut NEON register pressure, and hoisted `sbLowByteLanes` out of `filterSBVertical8`'s loop.
- **CDEF (`internal/av1/cdef`)**:
  - Hoisted destination bounds checks in `8×8` (`unsafeutil.Block8At`) and `4×4` (`unsafeutil.Grid`) byte-lane kernels and replaced the 4× unrolled strip with a 4-iteration loop, cutting `cdefByteFilter8Pri`'s stack frame from 160 B to 16 B (zero pointer spills).
  - Replaced `BroadcastInt16x8(128)` with L1 load `cdefDirBias128` and hoisted the 8 row loads in `findDirectionU8SIMD` via `unsafeutil.BlockAt`.
- **Motion Compensation (`internal/av1/motion`)**:
  - Added 4-row `Grid.Sub` unrolls for `mc8PutV4`, `mc8PrepH4S4`, `mc8PrepV4`, and `mc8PrepHV6x4`, and dispatched the 4-tap prep kernels in `mc8PrepGrid`.
- **Loop-Filter Masks (`internal/av1/lfmask`)**:
  - Unrolled `LevelCache.fill` 4 cells at a time with constant `comp` split; split `sidx` spans and fast-pathed `twl4c == 0` / `thl4c == 0` in `maskEdgesInter`.
- **Entropy (`internal/av1/entropy`)**:
  - Replaced per-TXB `% 3`, `% 5`, and `% 13` divisions in `DecodeCoefs` with range guards.

## [v0.1.1] - 2026-11-25

Pure-Go + `simd/archsimd` decoder performance release (-1.21B instructions / -3.2% on 1080p `aomrt_crowdrun` 1T, bringing 1T total wall time to 1.10× of `dav1d` and 8T cycles to 1.056× on Apple M4 Max, with 100% memory safety and bit-exact MD5 parity).

### Decoder Optimizations (`arm64` & generic)
- **Motion Compensation (`internal/av1/motion`)**:
  - Added fused in-register 4-tap and bilinear compound `Prep8` kernels (`mc8PrepHV6x4`, `mc8PrepHV2`, `mc8PrepHV4x4W4`, `mc8PrepHV4W4`, `mc8PrepH4S4`, `mc8PrepV4`) and 4-tap `Put8` kernels (`mc8PutHV6x4`, `mc8PutH4S4`, `mc8PutV4`).
  - Unrolled `mc8PutHV6`, `mc8PutHV6x4`, `mc8PutHV4x4W4`, `mc8PutHV4W4`, `mc8PrepHV4x4W4`, and `mc8PrepHV4W4` in 4-row `Grid.Sub` chunks, eliminating per-row bounds checks and `MADD` stride multiplies.
- **Loop Filter (`internal/av1/loopfilter`, `internal/unsafeutil`)**:
  - Added `unsafeutil.BlockHalves` and `unsafeutil.Block8`/`Block8At` to eliminate `MUL` row-address calculations in vertical loopfilter loads/stores and 16×16 transposes, and cut vertical level-pair bounds checks by 4×.
  - Added nibble-skipping (`bits.TrailingZeros32`) in `filterSBVertical8` and `sbRowsU8`, and dropped redundant `.HiToLo()` ops on broadcast constants.
- **CDEF (`internal/av1/cdef`, `internal/av1/decoder`)**:
  - Unrolled `8×8` CDEF strips (`cdefByteFilter8Pri`/`8PriSec`/`8Sec`) and added straight-line `4×4` chroma paths (`cdefByteFilter4Pri`/`4PriSec`/`4Sec`), replaced the 96B per-block direction offset copy with a `*cdefByteDirOffs` pointer, and hoisted loop-invariant chroma dispatch.
  - Computed sparse CDEF staging rects directly from the `uint64` unit mask (`frameWorkCDEFSparseStageRectFromMask`) with a static `frameWorkCDEFAllBlocks64` fast path.
- **Tile & Reconstruction (`internal/av1/tile`, `internal/av1/threading`)**:
  - Precomputed per-tile identity/warp global motion flags in `bindRefmvsForLoop`, fast-pathed non-switchable interpolation filters in `readInterpFiltersRefmvs`, guarded `ReadInterIntra`, and unrolled `reconCoefTree` non-recursively.

## [v0.1.0] - 2026-11-24

First tagged release of `goav1` — a pure-Go AV1 decoder and realtime WebRTC AV1 encoder using Go 1.27 `simd/archsimd` with zero assembly and audited memory safety (`scripts/check_unsafe.sh`).

### Highlights
- **Pure-Go SIMD + Memory Safety**: Zero `.s`/`.S` assembly files; all codec packages are `unsafe`-free, with footprint-checked, guard-page-tested views confined strictly to `internal/unsafeutil`.
- **Decoder Performance**: Fused in-register arm64 2D 6-tap, 4-tap (`mc8PutHV4W4` / `mc8PutHV4x4W4`), and 2-tap bilinear (`mc8PutHV2`) motion compensation; unsplit inter loop-filter mask fast path; precomputed CDEF direction and vector-splat tables; register-resident `DecodeVarTxTree`; tight EOB level clearing; and `GOMAXPROCS`-capped multi-worker scheduling (beating `dav1d` on 8-thread wall time on Apple M4 Max while matching `dav1d` MD5 bit-for-bit across all 42 real-content corpus streams at 1 and 8 workers).
- **Encoder Performance**: Realtime WebRTC AV1 encoder outperforming SVT-AV1 preset 12 RTC on both 1-thread and multi-threaded 720p workloads.

### Public API

- The package exports what an application uses and nothing else: `Decoder`
  and `LayeredDecoder` with their constructors and options, `Frame`, the IVF
  iterator, the encoders (`VideoEncoder`, `RTCEncoder`, `KeyframeEncoder`,
  their configs and frame types) and the WebRTC transport helpers (RTP, RTCP,
  RTX, sequencing, SDP). About 1,400 lower-level identifiers are gone: the
  frame-work decode pipeline (`DecoderFrameWork*`, `Bind*`, `Run*`), frame
  pools and layer pools, tile work and tile syntax readers, OBU and
  sequence-header parsing, prediction, transform, quantizer, loop-filter,
  CDEF, restoration, super-resolution, film-grain and DSP kernels, and the
  encoder's bitstream writers. In-module code uses the internal packages.

### Decoder

- Pure-Go AV1 decode pipeline covers IVF, OBU, Annex B, RTP, frame headers,
  tile decode, prediction, transforms, reconstruction, loop filter, CDEF,
  super-resolution, loop restoration, film grain, SVC routing, and raw YUV
  output.
- Frames split across several tile-group OBUs decode correctly: later tile
  groups keep the CDEF index map, loop-filter masks and restoration records
  the first tile group bound, instead of resetting them.
- Current committed decoder vector gates are strict-MD5 clean; the gates are
  described in [docs/quality-and-parity.md](docs/quality-and-parity.md).
- Streams coded with quantizer matrices decode correctly. The matrices were
  indexed row by row although they are stored in coefficient order, so every
  rectangular transform block was dequantized with the wrong weights; no
  committed vector used the tool. Two quantizer-matrix clips now gate it.
- Inter frames that follow a key frame coded with intra block copy decode
  correctly. Block-copy vectors adapted the motion-vector contexts instead of
  their own, so the following frames inherited the wrong entropy state (every
  SVT-AV1 screen-content stream was affected).
- `LayeredDecoder` projects temporal motion vectors. It kept no motion vectors
  for its reference buffers, so any frame coded with `use_ref_frame_mvs`
  (the default in libaom and SVT-AV1) decoded to wrong pixels without an
  error. Its output now matches `Decoder` and the libaom goldens on every
  vendored clip it accepts.
- Hot paths are designed around caller-owned buffers, reusable frame pools,
  explicit scratch sizing, and zero steady-state allocations.
- `WithFrameSizeLimit` makes the high-level decoders reject streams whose
  frames exceed a pixel count (dav1d's `frame_size_limit`), and RTP payloads
  are held to the same 64-OBU bound as low-overhead ones, so hostile input
  cannot make a decoder allocate thousands of times its size.
- Performance work remains active, with dav1d/libaom source-shaped scalar code,
  compiler/allocation guardrails, and initial SIMD dispatch/kernels.
- With more than one worker, low-overhead decoders (NewDecoder,
  NewDecoderFromIVF and the reader variants) run dav1d-style frame threading
  by default: up to ceil(sqrt(workers)) frames are in flight, so DecodeNext can
  return a frame a few calls after the payload that carried it, and the last
  calls drain the pipeline. Output order and pixels are unchanged.
  WithMaxFrameDelay(1) restores one-frame-per-call delivery (dav1d
  max_frame_delay). RTP decoders are realtime and never frame-thread.
  Frames with several tiles frame-thread too, as long as every tile arrives in
  one tile group: each tile's entropy pass runs on its own lane, and the
  frame's reconstruction and post-filter overlap the next frame's tiles.
- `aom-go-dec` uses one worker per CPU by default, as dav1d's CLI does;
  `-workers 1` selects the single-threaded path. The library default stays
  one worker.

### Encoder Scope

- The encoder target is a realtime WebRTC AV1 encoder, not a general-purpose
  offline AV1 encoder.
- The intended surface is WebRTC-style control and transport: bitrate,
  framerate, resolution and keyframe controls; temporal and spatial layer/SVC
  configuration; camera and screen-content tuning; realtime speed/quality
  controls; low-overhead OBU/RTP output; and dependency/scalability metadata
  needed by WebRTC integrations.
- Encoder implementation is in scope. The public realtime WebRTC API is usable
  for 8-bit I420 streams, including temporal layering, multi-spatial SVC and
  simulcast pictures, RTP payload packetization, dependency descriptors, and
  runtime bitrate/framerate/scalability reconfiguration. New behavior must
  continue to be ported from pinned libaom/libwebrtc source, with pinned
  SVT-AV1 used as the speed-architecture reference. New/touched code must keep
  relevant upstream C type widths and signedness, and should land with oracle
  tests and allocation evidence appropriate to the path.

### Encoder

- Spatial-layer and simulcast streams downscale each lower layer's picture
  with libyuv's `kFilterBox` (libwebrtc's simulcast scaler) instead of
  nearest-neighbour sampling, cascaded from the layer above. On the real
  corpus at S3T3 the lower layers gain 0.5-9.0 dB Y-PSNR against an area
  reference at equal or lower bitrate; the top layer is unchanged. An exact
  1/2 step's luma is the upper layer's HME pyramid plane, so the upper layer
  copies it instead of filtering again.
- In simulcast modes an upper layer's HME takes the layer below's motion
  as its level-0 result when that layer coded this picture's half-resolution
  pyramid plane against the same reference: one seed-centered level-0 region
  and a level-1 refinement of the seeds replace the four-region search.
  S3T3 single-thread cycles drop 2-5% on the real corpus at -0.4..+0.16%
  BD-rate per layer (fixed qindex).
- `VideoEncoder` no longer promotes delta frames to keyframes on detected
  scene cuts: the HME cut detector, `DisableSceneCutKeyframes`,
  `SetSceneCutKeyframes` and qualitybench `-goav1-scene-cut` are removed. This
  matches SVT-AV1 4.1.0 rtc, which forces scene-change detection off and codes
  cuts with intra blocks in an inter frame, and libwebrtc's libaom setup
  (`kf_mode = AOM_KF_DISABLED`).
- Screen content follows SVT-AV1 4.1.0 rtc. `VideoEncoderConfig.Content` and
  the WebRTC `Content` take `EncoderContentAuto` (the new zero value, SVT
  `--scm 2`), `EncoderContentCamera` or `EncoderContentScreen`. Auto runs
  SVT's `svt_aom_is_screen_content` on each keyframe (up to 1080p; inter
  frames inherit the class). Screen pictures use SVT's screen-content ME:
  pre-HME, HME level 2, the larger HME and ME search areas, and the early
  exits, plus the existing screen-content rate control and CDEF levels.
  Headers follow SVT: sequence headers always select screen-content tools
  and integer MV, `allow_screen_content_tools` is set on screen keyframes
  only, and inter frames keep fractional motion. A content change takes
  effect at the next keyframe. `SetScreenContentSelection` is removed.
  Screen keyframes run SVT's luma palette search (palette_level 3), and
  screen inter pictures add SVT's V/H/SMOOTH intra candidates
  (intra_level 6).
- Every coded frame is one OBU_FRAME (header, byte_alignment, tile group),
  as SVT-AV1 and libaom emit, instead of an OBU_FRAME_HEADER plus an
  OBU_TILE_GROUP: about 3.5 bytes less per frame, with the same
  reconstruction.
- Inter frames after a keyframe chain from the keyframe's CDFs
  (primary_ref_frame LAST), as SVT-AV1's do, instead of restarting
  error-resilient from default contexts.
- The 8-bit 4:2:0 encoder codes one tile by default, as SVT-AV1
  (tile_columns = tile_rows = 0), keyframes included; the goav1 2/4/32
  tile-column default is removed. Keyframes now run the decision/write
  split with the SB-row wavefront, as inter frames do, pricing decisions
  against frame-start trial contexts, so the output is identical at every
  thread count. MaxThreads 0 runs the wavefront with GOMAXPROCS lanes, and
  the CDEF strength search spreads its filter-block rows over the same
  lanes. TileColumns still selects tiles.

### Verification

- Local development gates: `go test ./...`, `make alloc`,
  `make compiler-reports`, and `make trace-zero`.
- Decoder vector gates: `make dryrun-fast`, `make dryrun-full`,
  `make dryrun-extended`, and `make dryrun-profiles`.
- Cross-decoder performance tracking: `make bench-cross` and the optional
  generated corpus lane documented in [README.md](README.md).
