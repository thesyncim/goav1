# goav1 AV1 Conformance Status

This document is the single-source-of-truth inventory of AV1 specification
features that the goav1 decoder implements, partially implements, or does
not yet implement. It is intended for integrators trying to predict whether
a given stream will decode bit-exactly, contributors picking up a missing
feature, and reviewers triaging a libaom conformance vector mismatch.

Pair it with [README.md](README.md) for project status and
[ARCHITECTURE.md](ARCHITECTURE.md) for the package map and per-frame
pipeline. The "Known Limitations and Roadmap" section of `ARCHITECTURE.md`
points here for the spec-level feature inventory.

Status legend:

- `Yes`     - feature implemented; expected to be bit-exact against libaom
              for the surface it covers.
- `Partial` - structure present and exercised by tests, but at least one
              sub-case is known to mismatch libaom or is not yet wired into
              the end-to-end frame-work pipeline. The "Notes" column lists
              what is missing.
- `No`      - not implemented; reads either fail with an explicit error or
              produce a no-op event.
- `N/A`     - the row does not apply to a Go decoder (encoder-only feature,
              encoder constraint, etc.).

All file references are repo-relative. Test vectors referenced from
"Vector coverage" below are part of the libaom AV1 conformance suite and
ship under `internal/av1/testdata/libaom/`.

---

## 1. Feature inventory

```
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| #  | Area                             | Status  | Implementing path(s)             | Notes                                          |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  1 | Profile 0 (Main)                 | Yes     | internal/av1/parser/sequence.go  | seq_profile=0 parsed; default decode path.     |
|    | Profile 1 (High)                 | Partial | internal/av1/parser/sequence.go  | Header parsed; 4:4:4 chroma decode not         |
|    |                                  |         |                                  | exercised by any committed vector.             |
|    | Profile 2 (Professional)         | Partial | internal/av1/parser/sequence.go  | Header (incl. 12-bit + 4:2:2 selectors)        |
|    |                                  |         |                                  | parsed; no committed vector reaches the        |
|    |                                  |         |                                  | end-to-end pipeline.                           |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  2 | Bit depth 8                      | Yes     | internal/av1/frame/              | Full pipeline; primary conformance target.     |
|    | Bit depth 10                     | Partial | internal/av1/frame/              | Frame storage, transforms, deblock, CDEF,      |
|    |                                  |         | internal/av1/loopfilter/         | superres, restoration, film grain all carry    |
|    |                                  |         | internal/av1/transform/          | 10-bit code paths; no committed 10-bit vector  |
|    |                                  |         |                                  | is in the dryrun-fast set yet.                 |
|    | Bit depth 12                     | Partial | internal/av1/frame/              | Same surface as 10-bit; no committed 12-bit    |
|    |                                  |         | internal/av1/loopfilter/         | vector; only profile-2 gate is exercised.      |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  3 | Subsampling 4:2:0                | Yes     | internal/av1/parser/sequence.go  | Default decode layout; covered by every fast   |
|    |                                  |         | internal/av1/frame/              | conformance vector.                            |
|    | Subsampling 4:2:2                | Partial | internal/av1/parser/sequence.go  | Header bits parsed (profile 2/12-bit toggle);  |
|    |                                  |         |                                  | end-to-end decode untested.                    |
|    | Subsampling 4:4:4                | Partial | internal/av1/parser/sequence.go  | Header bits parsed (profile 1); end-to-end     |
|    |                                  |         |                                  | decode untested.                               |
|    | Monochrome (4:0:0)               | Yes     | internal/av1/frame/              | Surfaces drop the UV planes; covered by        |
|    |                                  |         | internal/av1/parser/sequence.go  | av1-1-b8-24-monochrome.ivf (frame 0 PASS;      |
|    |                                  |         |                                  | later frames mismatch - see vector table).     |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  4 | Color range: limited (studio)    | Yes     | internal/av1/parser/sequence.go  | ColorRange=false parsed; no clipping path      |
|    |                                  |         |                                  | beyond sample range applies in decode.         |
|    | Color range: full                | Yes     | internal/av1/parser/sequence.go  | ColorRange=true parsed; output values are not  |
|    |                                  |         |                                  | rescaled, matching libaom.                     |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  5 | Superblock 64x64                 | Yes     | internal/av1/parser/sequence.go  | use_128x128_superblock=false; default for      |
|    |                                  |         | internal/av1/tile/partition.go   | every fast conformance vector.                 |
|    | Superblock 128x128               | Partial | internal/av1/parser/sequence.go  | Use128x128Superblock flag parsed and threaded  |
|    |                                  |         | internal/av1/parser/restoration  | into restoration unit sizing; no committed     |
|    |                                  |         |                                  | vector exercises the 128x128 partition tree.   |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  6 | Intra mode DC                    | Yes     | internal/av1/prediction/intra.go | Pure-Go reference; alloc-tested.               |
|    | Intra mode V (Vertical)          | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Intra mode H (Horizontal)        | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Intra mode D45                   | Yes     | internal/av1/prediction/directional.go | Directional table-driven; intra_only       |
|    |                                  |         |                                  | dryrun coverage.                               |
|    | Intra mode D67                   | Yes     | internal/av1/prediction/directional.go | Directional table-driven.                  |
|    | Intra mode D113                  | Yes     | internal/av1/prediction/directional.go | Directional table-driven.                  |
|    | Intra mode D135                  | Yes     | internal/av1/prediction/directional.go | Directional table-driven.                  |
|    | Intra mode D157                  | Yes     | internal/av1/prediction/directional.go | Directional table-driven.                  |
|    | Intra mode D203                  | Yes     | internal/av1/prediction/directional.go | Directional table-driven.                  |
|    | Intra mode SMOOTH                | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Intra mode SMOOTH_V              | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Intra mode SMOOTH_H              | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Intra mode PAETH                 | Yes     | internal/av1/prediction/intra.go | Pure-Go reference.                             |
|    | Chroma-from-Luma (CFL)           | Yes     | internal/av1/prediction/cfl.go   | Subsample + alpha-scaled predict; alloc-       |
|    |                                  |         |                                  | tested via prediction_public_test.go.          |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  7 | Filter intra modes               | Yes     | internal/av1/prediction/filter_intra.go | All five libaom FILTER_INTRA_MODE         |
|    |                                  |         |                                  | values (DC, V, H, D157, PAETH) implemented.    |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  8 | Palette Y                        | Partial | internal/av1/tile/palette.go     | Y palette size/color/CDF entry decoding        |
|    |                                  |         |                                  | implemented; not yet wired into block-loop     |
|    |                                  |         |                                  | predictor for end-to-end reconstruction.       |
|    | Palette UV                       | Partial | internal/av1/tile/palette.go     | UV palette size/color/CDF entry decoding       |
|    |                                  |         |                                  | implemented; same wiring gap as Y.             |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  9 | IntraBC                          | Partial | internal/av1/tile/block_loop.go  | DV decoding, DV validity check, predicted MV   |
|    |                                  |         | internal/av1/tile/intrabc_debug.go | stack and intra-mode entry implemented;       |
|    |                                  |         |                                  | end-to-end output diverges on the              |
|    |                                  |         |                                  | intrabc_extreme_dv fast vector (frame 0 MD5    |
|    |                                  |         |                                  | mismatch). See vector table row 7.             |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 10 | Inter mode NEARESTMV             | Yes     | internal/av1/tile/inter_mode.go  | InterModeNearestMV; ref_mv stack populated.    |
|    | Inter mode NEARMV                | Yes     | internal/av1/tile/inter_mode.go  | InterModeNearMV.                               |
|    | Inter mode NEWMV                 | Yes     | internal/av1/tile/inter_mode.go  | InterModeNewMV; signed diff decode in mv.go.   |
|    | Inter mode GLOBALMV              | Yes     | internal/av1/tile/inter_mode.go  | InterModeGlobalMV uses global motion params.   |
|    | NEAREST_NEARESTMV                | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNearestNearest.               |
|    | NEW_NEWMV                        | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNewNew.                       |
|    | NEW_NEARESTMV                    | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNewNearest.                   |
|    | NEAREST_NEWMV                    | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNearestNew.                   |
|    | NEAR_NEWMV                       | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNearNew.                      |
|    | NEW_NEARMV                       | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNewNear.                      |
|    | NEAR_NEARMV                      | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeNearNear.                     |
|    | GLOBAL_GLOBALMV                  | Yes     | internal/av1/tile/inter_mode.go  | CompoundInterModeGlobalGlobal.                 |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 11 | Difference-weighted compound     | Yes     | internal/av1/tile/inter_compound.go | CompoundTypeDiffWeight; DIFFWTD mask blends |
|    |                                  |         | internal/av1/dsp/                | use the dsp blend helpers.                     |
|    | Wedge compound                   | Yes     | internal/av1/tile/inter_compound.go | Wedge index CDFs; 16-mask wedge LUT;        |
|    |                                  |         | internal/av1/threading/wedge.go  | inter-intra wedge supported.                   |
|    | Masked compound (incl. avg/dist) | Yes     | internal/av1/tile/inter_compound.go | EnableMaskedCompound gating + average and   |
|    |                                  |         |                                  | distance-weighted compound types.              |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 12 | Sub-pel: regular (8-tap)         | Yes     | internal/av1/motion/filter.go    | InterpEightTap; bit-exact vs libaom convolve   |
|    |                                  |         |                                  | reference (convolve_libaom_test.go).           |
|    | Sub-pel: smooth                  | Yes     | internal/av1/motion/filter.go    | InterpEightTapSmooth.                          |
|    | Sub-pel: sharp                   | Yes     | internal/av1/motion/filter.go    | InterpEightTapSharp.                           |
|    | Sub-pel: bilinear                | Yes     | internal/av1/motion/filter.go    | InterpBilinear (size <= 4 fallback per spec).  |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 13 | Warped motion                    | Yes     | internal/av1/motion/warp.go      | 6-coefficient affine + warped 8-tap filter;    |
|    |                                  |         | internal/av1/motion/warp_filter.go | alpha/beta/gamma/delta derivation.            |
|    |                                  |         | internal/av1/tile/warped_motion.go | Frame-level warp params parsed.               |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 14 | Motion field MV projection       | Partial | internal/av1/tile/motion_field.go | Projection storage and Setup() implemented   |
|    |                                  |         | internal/av1/threading/ref_mv_frame.go | end-to-end output diverges at first      |
|    |                                  |         | internal/av1/decoder/motion_field.go | non-key frame on av1-1-b8-06-mfmv.ivf;    |
|    |                                  |         |                                  | see vector table row 6.                        |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 15 | OBMC                             | Yes     | internal/av1/tile/motion_mode.go | OBMC neighbor scan + overlap blending;         |
|    |                                  |         | internal/av1/threading/predict.go | per-block mask tables 1/2/4/8/16 wired.       |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 16 | Switchable interpolation filter  | Yes     | internal/av1/parser/tile.go      | InterpolationSwitchable parsed; per-block      |
|    |                                  |         | internal/av1/tile/inter_filter.go | filter decode + dispatch in inter_filter.go.  |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 17 | Inverse DCT (4..64)              | Yes     | internal/av1/transform/dct.go    | Pure-Go reference; per-size benchmarks.        |
|    | Inverse ADST                     | Yes     | internal/av1/transform/adst.go   | Pure-Go reference.                             |
|    | Inverse FlipADST                 | Yes     | internal/av1/transform/adst.go   | Reuses ADST + post-flip.                       |
|    | IDTX (identity transform)        | Yes     | internal/av1/transform/identity.go | Pure-Go reference; transform_test.go.        |
|    | WHT 4x4                          | Yes     | internal/av1/transform/wht.go    | Pure-Go reference.                             |
|    | Hybrid combinations              | Yes     | internal/av1/transform/hybrid.go | All 16 TX_TYPE values (DCT_DCT, ADST_DCT,      |
|    |                                  |         | internal/av1/transform/block.go  | DCT_ADST, ADST_ADST, FLIPADST_*, V_*, H_*,     |
|    |                                  |         | internal/av1/tile/tx_type.go     | IDTX) plus extended-tx-set gating.             |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 18 | Standard dequantization          | Yes     | internal/av1/quantize/dequant.go | Per-plane Q lookup + scan order dispatch.      |
|    | QMatrix (inverse qmatrix)        | Yes     | internal/av1/quantize/dequant.go | DequantizeBlockScaledQMatrix; libaom-formula   |
|    |                                  |         |                                  | matched by dequant_test.go.                    |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 19 | CDEF                             | Yes     | internal/av1/cdef/               | Direction search + primary/secondary filter;   |
|    |                                  |         | internal/av1/decoder/postfilter_cdef.go | per-block 8x8 strength index applied.   |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 20 | Deblocking loop filter           | Yes     | internal/av1/loopfilter/         | 4/6/8/14-sample edge filters with flat +       |
|    |                                  |         | internal/av1/decoder/postfilter_loopfilter.go | narrow fallbacks; 8/10/12-bit       |
|    |                                  |         |                                  | reconstruction planes covered.                 |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 21 | Loop restoration: Wiener         | Yes     | internal/av1/restoration/wiener.go | Frame-plan + per-unit apply; libaom-derived  |
|    |                                  |         | internal/av1/tile/restoration*.go | filter coefficients.                          |
|    | Loop restoration: SGR projection | Yes     | internal/av1/restoration/selfguided.go | Self-guided filter; libaom-derived         |
|    |                                  |         |                                  | r/s tables.                                    |
|    | Loop restoration: switchable     | Yes     | internal/av1/tile/restoration.go | Per-unit Wiener / SGR / None selector.         |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 22 | Super-resolution (disabled)      | Yes     | internal/av1/parser/frame_size.go | SuperResEnabled=false: no upscale path.       |
|    | Super-resolution (enabled)       | Yes     | internal/av1/superres/superres.go | Frame-level horizontal upscale between coded  |
|    |                                  |         | internal/av1/decoder/postfilter_superres.go | and display widths; denominator 9..16.  |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 23 | Film grain synthesis             | Yes     | internal/av1/filmgrain/          | Gaussian RNG, scaling LUTs, luma/chroma grain  |
|    |                                  |         | internal/av1/decoder/postfilter_filmgrain.go | blocks, per-row apply; covered by   |
|    |                                  |         |                                  | av1-1-b8-23-film_grain-50.ivf (not in fast).   |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 24 | Tile groups: single              | Yes     | internal/av1/parser/tile_group.go | Single-tile group is the default path.        |
|    | Tile groups: multiple            | Yes     | internal/av1/parser/tile_group.go | ParseTileGroupHeader supports start/end span;  |
|    |                                  |         | internal/av1/decoder/stream.go   | continuation tile groups re-use frameState.    |
|    | Tile lists (OBU_TILE_LIST)       | No      | internal/av1/decoder/stream.go   | OBU type recognised and emitted as             |
|    |                                  |         | internal/av1/obu/types.go        | EventTileList; payload not parsed.             |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 25 | Frame type: key                  | Yes     | internal/av1/parser/frame.go     | FrameTypeKey; full reference reset path.       |
|    | Frame type: intra-only           | Yes     | internal/av1/parser/frame.go     | FrameTypeIntraOnly; intra-only refresh.        |
|    | Frame type: inter                | Yes     | internal/av1/parser/frame.go     | FrameTypeInter; mostly bit-exact, mfmv/mv      |
|    |                                  |         |                                  | desync on later frames (see vector table).     |
|    | Frame type: switch               | Yes     | internal/av1/parser/frame.go     | FrameTypeSwitch parsed; ErrorResilientMode     |
|    |                                  |         |                                  | implied. No committed switch-frame vector.     |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 26 | Show-existing-frame              | Yes     | internal/av1/parser/frame.go     | parseShowExistingFrameHeader; decoder event    |
|    |                                  |         | internal/av1/decoder/stream.go   | EventExistingFrame; surface lookup + abort     |
|    |                                  |         | internal/av1/decoder/surface.go  | of in-flight frame work on apply.              |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 27 | 8-slot reference frame pool      | Yes     | internal/av1/decoder/surface.go  | NUM_REF_FRAMES=8 slot table.                   |
|    | refresh_frame_flags handling     | Yes     | internal/av1/parser/reference.go | Per-slot refresh bit applied at frame finish;  |
|    |                                  |         | internal/av1/decoder/surface_pool.go | atomic batch release of replaced surfaces. |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 28 | Temporal motion field projection | Partial | internal/av1/tile/motion_field.go | TemporalMotionField.Setup matches libaom      |
|    |                                  |         | internal/av1/threading/ref_mv_frame.go | ordering; storage and entry binding       |
|    |                                  |         | internal/av1/decoder/motion_field.go | exposed; bit-exactness still diverges on    |
|    |                                  |         |                                  | the mv / mfmv vectors at frame 1.              |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 29 | Segmentation                     | Yes     | internal/av1/parser/segmentation.go | SegmentationParams parsing + previous-state |
|    |                                  |         | internal/av1/tile/decode.go      | carryover into tile decode state.              |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 30 | Delta-Q                          | Yes     | internal/av1/parser/delta.go     | DeltaQPresent / DeltaQResLog2 parsed; per-     |
|    |                                  |         | internal/av1/tile/decode.go      | block delta tracked in tile decode state.      |
|    | Delta-LF                         | Yes     | internal/av1/parser/delta.go     | DeltaLFPresent / DeltaLFResLog2 / DeltaLFMulti |
|    |                                  |         | internal/av1/loopfilter/         | parsed and applied at the edge filter level.   |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 31 | OBU: TemporalDelimiter           | Yes     | internal/av1/obu/types.go        | EventTemporalDelimiter; clears pending frame.  |
|    | OBU: SequenceHeader              | Yes     | internal/av1/parser/sequence.go  | ParseSequenceHeader; full payload coverage.    |
|    | OBU: FrameHeader                 | Yes     | internal/av1/parser/frame.go     | ParseFrameHeaderPrefix + downstream parsers.   |
|    | OBU: Frame                       | Yes     | internal/av1/parser/frame.go     | Combined header + tile-group dispatch.         |
|    | OBU: TileGroup                   | Yes     | internal/av1/parser/tile_group.go | ParseTileGroupHeader + span splitting.        |
|    | OBU: Metadata                    | Partial | internal/av1/decoder/stream.go   | OBU type recognised and emitted as             |
|    |                                  |         |                                  | EventMetadata; payload not parsed.             |
|    | OBU: RedundantFrameHeader        | Yes     | internal/av1/decoder/stream.go   | Accepted; ignored if a frame header is already |
|    |                                  |         |                                  | active for the temporal unit.                  |
|    | OBU: Padding                     | Yes     | internal/av1/decoder/stream.go   | Recognised; emitted as EventPadding.           |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 32 | Annex B framing                  | Yes     | internal/av1/obu/annexb.go       | NewAnnexBIterator; length-prefixed temporal /  |
|    |                                  |         |                                  | frame / OBU unit iteration; fuzzed.            |
|    | Low-overhead framing (WebRTC)    | Yes     | internal/av1/obu/unit.go         | NewLowOverheadIterator; size-restoration       |
|    |                                  |         | internal/av1/obu/normalize.go    | helpers for the AV1 RTP payload format.        |
|    | Section 5 temporal-unit framing  | Yes     | internal/av1/obu/temporal_unit.go | NewTemporalUnitIterator for .obu conformance  |
|    |                                  |         |                                  | files.                                         |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 33 | IVF container (DKIF/AV01)        | Yes     | internal/av1/ivf/reader.go       | Zero-allocation NewIVFIterator; conformance    |
|    |                                  |         |                                  | harness and CLI both consume it.               |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 34 | AV1 RTP payload (RFC draft)      | Yes     | internal/av1/rtp/                | Aggregation header parse, payload iteration,   |
|    |                                  |         |                                  | single-OBU + fragmented OBU packetization,     |
|    |                                  |         |                                  | depacketizer state machine, frame assembler;   |
|    |                                  |         |                                  | round-trip alloc-tested and fuzzed.            |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
```

### Cross-cutting items not enumerated above

- **Encoder.** `internal/av1/encoder/` contains only `doc.go`. The realtime
  encoder is out of scope for the decoder MVP.
- **SIMD / assembly backends.** Every DSP entry point is pure Go today.
  The `internal/av1/dsp` and `internal/av1/transform` dispatch shapes are
  stable; an amd64/arm64 backend will land behind them.
- **Work-stealing scheduler.** `threading.Pool` does deterministic
  fan-out per batch; no dynamic stealing across workers mid-frame.
- **Allocation budget.** Every public hot-path helper is zero-allocation
  per `BenchmarkPublic*` and `make alloc`. Any regression on the rows
  marked "Yes" above must keep that guarantee.

---

## 2. Vector coverage (libaom AV1 8-bit fast suite)

This table lists the eight vectors returned by
`LibaomRemoteManifest().SelectRemote(SuiteLevelFast, 0, nil)` and tracks
their current pass/fail state under `make dryrun-fast` (lenient
first-frame MD5) and `GOAV1_STRICT_MD5=1` (every-frame MD5). All eight
ship under `internal/av1/testdata/libaom/`.

```
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| # | Vector                                            | Lenient | Strict | First mismatch (under strict)              |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 1 | av1-1-b8-00-quantizer-00.ivf                      | PASS    | FAIL   | Frame 1 (inter-frame residual MV path).    |
|   | (libaom av1 8-bit quantizer 00)                   |         |        | First-frame intra surface matches libaom.  |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 2 | av1-1-b8-01-size-16x16.ivf                        | PASS    | PASS   | No mismatch across the 2-frame clip; the   |
|   | (libaom av1 8-bit 16x16 size)                     |         |        | strict-mode baseline for the suite.        |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 3 | av1-1-b8-02-allintra.ivf                          | PASS    | FAIL   | Frame 3 (intra-only chain divergence       |
|   | (libaom av1 8-bit all-intra)                      |         |        | after frame 2). 36/39 frames match strict. |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 4 | av1-1-b8-04-cdfupdate.ivf                         | PASS    | FAIL   | Frame 1 (CDF update path on first inter    |
|   | (libaom av1 8-bit cdf update)                     |         |        | frame).                                    |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 5 | av1-1-b8-05-mv.ivf                                | PASS    | FAIL   | Frame 1 (motion-vector decode / inter      |
|   | (libaom av1 8-bit mv)                             |         |        | prediction divergence).                    |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 6 | av1-1-b8-06-mfmv.ivf                              | PASS    | FAIL   | Frame 1 (motion-field projection path      |
|   | (libaom av1 8-bit mfmv)                           |         |        | for the first ref-frame-MVS consumer).     |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 7 | av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf     | FAIL    | FAIL   | Frame 0 (intrabc extreme-DV intra-only     |
|   | (libaom av1 8-bit intra-only intrabc extreme dv)  |         |        | output diverges from libaom). Only vector  |
|   |                                                   |         |        | failing the lenient gate.                  |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 8 | av1-1-b8-24-monochrome.ivf                        | PASS    | FAIL   | Frame 1 (monochrome inter divergence;      |
|   | (libaom av1 8-bit monochrome)                     |         |        | overlaps with the mv/mfmv mismatch).       |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
```

### How to reproduce

Lenient gate (the default `make dryrun-fast`):

```sh
make dryrun-fast
# - or -
GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 go test -tags goav1_oracle \
    ./internal/av1/testvector -run TestLibaomFastFrameWorkDryRun \
    -count=1 -timeout 600s -v
```

Strict gate (every-frame MD5):

```sh
GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 \
    go test -tags goav1_oracle ./internal/av1/testvector \
    -run TestLibaomFastFrameWorkDryRun -count=1 -timeout 600s -v
```

Each subtest logs a trailing
`vector=NAME frames=N md5_matches=M first_mismatch=F` line.
`first_mismatch=-1` means every frame matched.

The committed `TestLibaomQuantizer00FrameWorkDryRun` test is the
single-vector strict guard rail referenced by CI; it currently exercises
`av1-1-b8-00-quantizer-00.ivf` for its frame-0 surface only.

---

## 3. Where each failure shows up in code

These pointers are the starting point for anyone picking up a vector
mismatch. Each entry links the vector to the code area whose surface is
likely to be at fault, based on the per-frame `txbs/residuals/cdef_units/
mfmv_refs/mfmv_projections` counters printed by the dry-run harness.

- **Vector 1, frame 1 mismatch (`quantizer-00`).** First inter frame.
  Reference-MV stack + inter prediction in
  `internal/av1/tile/ref_mv.go`, `internal/av1/tile/inter.go`, and the
  per-block predictor wiring in
  `internal/av1/threading/predict.go`.
- **Vector 3, frame 3 mismatch (`all-intra`).** Intra-only with frequent
  CDF updates. Suspects: per-block intra mode CDF carryover in
  `internal/av1/tile/intra.go`, `internal/av1/tile/coeff_context.go`,
  and entropy `Adapt` in `internal/av1/entropy/`.
- **Vector 4, frame 1 mismatch (`cdf-update`).** First inter frame with
  full CDF retention. Suspects: context-update tile selection in
  `internal/av1/tile/decode.go` and CDF clone/retain in
  `internal/av1/threading/cdf_reset.go`.
- **Vector 5, frame 1 mismatch (`mv`).** First inter frame with non-key
  MV. Suspects: NEWMV diff decode in `internal/av1/tile/mv.go`,
  reference-MV stack ordering in `internal/av1/tile/ref_mv.go`,
  sub-pel filter selection in `internal/av1/tile/inter_filter.go`,
  and the convolve filter in `internal/av1/motion/filter.go`.
- **Vector 6, frame 1 mismatch (`mfmv`).** First ref-frame-MVS consumer.
  Suspects: temporal motion field projection in
  `internal/av1/tile/motion_field.go` and
  `internal/av1/threading/ref_mv_frame.go`; libaom's
  `av1_setup_motion_field()` ordering parity.
- **Vector 7, frame 0 mismatch (`intrabc-extreme-dv`).** Intra-only.
  Suspects: DV decoding and DV validity in
  `internal/av1/tile/block_loop.go` (`intrabcPredictedMV*`,
  `intrabcDVValid`), edge-clip during intrabc reference fetch.
- **Vector 8, frame 1 mismatch (`monochrome`).** Same surface as
  vectors 5/6 (mv + mfmv): inter prediction on the Y plane only, no
  UV planes to mask the divergence.

When chasing one of these, the per-package `debug_*_dump_test.go`
files (gitignored, never committed) are the canonical place to add a
scoped dump. Pair the dump with `make sync-upstreams` and diff the
libaom dumper for the same vector.

---

## 4. Forward-looking gaps

The "Partial" and "No" rows above translate roughly into the following
work items, ordered by how much of the fast suite they would unblock:

1. **IntraBC end-to-end output parity.** The only fast vector failing the
   lenient gate is `intrabc-extreme-dv`. The implementation is wired all
   the way through `block_loop.go`; the open question is the
   intra-block-copy reference fetch when the DV crosses the previous
   superblock boundary.
2. **Motion field / motion-vector parity on frame 1 of inter vectors.**
   Strict mode fails immediately on frame 1 of `cdf-update`, `mv`,
   `mfmv`, and `monochrome`. The shared surface is the first non-key
   frame's reference-MV stack + temporal MV projection.
3. **CDF retention parity beyond the first context-update tile.** The
   `all-intra` strict failure shows up at frame 3, after the per-tile
   CDF retention has had a few rounds to drift.
4. **Profile 1 / Profile 2 end-to-end coverage.** 10-bit and 4:4:4 / 4:2:2
   conformance vectors are not in the fast suite. Adding them under
   `internal/av1/testdata/libaom/` and the dryrun harness will surface
   any remaining 10/12-bit path bugs.
5. **Palette wiring.** Palette mode decode is implemented in
   `internal/av1/tile/palette.go` but is not yet exercised by the
   block-loop predictor. Wire it in once a palette-using vector is in
   the dryrun-fast set.
6. **Tile list OBU and Metadata OBU payloads.** Currently emitted as
   opaque events; payloads are not parsed. No fast-suite vector
   requires either today.
7. **Switch frames.** Parsed but no committed vector exercises the
   switch-frame reset path end-to-end.

Once the fast suite is bit-exact under strict MD5, the next milestone is
`SuiteLevelRelevant`, which adds intra-bc beyond the extreme-DV case,
SVC, more film-grain vectors, monochrome variants, and 10-bit content.

---

## 5. Maintenance

When adding or removing a feature row above, update:

- This document (`CONFORMANCE.md`).
- The "Feature coverage status" sub-table in
  [ARCHITECTURE.md](ARCHITECTURE.md#known-limitations-and-roadmap) if
  the row crosses a major status boundary (`No` -> `Partial`,
  `Partial` -> `Yes`).
- The "Current Safe Point" bullets in [README.md](README.md) when the
  feature is part of the public API.

The vector coverage table in section 2 should be re-run after any change
to the decoder, residual reconstruction, or post-filter pipeline.
`make dryrun-fast` is the canonical command; record the
`vector=... first_mismatch=...` summary line of each subtest when the
suite shifts.
