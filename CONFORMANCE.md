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
|    |                                  |         | internal/av1/transform/          | 10-bit code paths; bit-depth clamps now flow   |
|    |                                  |         | internal/av1/quantize/           | through dequant (403e42b) and inverse          |
|    |                                  |         |                                  | transforms (42a8b88, 266e8cd); 10-bit q32 and  |
|    |                                  |         |                                  | q63 extended vectors still mismatch with root  |
|    |                                  |         |                                  | cause suspected elsewhere in reconstruction.   |
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
|  9 | IntraBC                          | Yes     | internal/av1/tile/block_loop.go  | DV decoding, DV validity check, predicted MV   |
|    |                                  |         | internal/av1/tile/ref_mv.go      | stack and intra-mode entry implemented;        |
|    |                                  |         |                                  | entropy stream bit-exact against libaom on    |
|    |                                  |         |                                  | intrabc frames (tx_size neighbor context       |
|    |                                  |         |                                  | snapshot, 2bae671); cross-SB DV diagonal-      |
|    |                                  |         |                                  | corner snapshot (5f88540) closes the           |
|    |                                  |         |                                  | reconstruction-layer gap, bringing the         |
|    |                                  |         |                                  | intrabc_extreme_dv fast-suite vector to PASS.  |
|    |                                  |         |                                  | See vector table row 7.                        |
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
| 24 | Tile groups: single              | Yes     | internal/av1/parser/tile_group.go | Single-tile group is the default path; every  |
|    |                                  |         |                                  | committed fast-suite vector uses Cols=1 Rows=1.|
|    | Tile groups: multiple            | Partial | internal/av1/parser/tile_group.go | ParseTileGroupHeader supports start/end span;  |
|    |                                  |         | internal/av1/decoder/stream.go   | continuation tile groups re-use frameState.    |
|    |                                  |         |                                  | Targeted vector coverage is limited: libaom    |
|    |                                  |         |                                  | v3.14.0's published AV1 test suite has no      |
|    |                                  |         |                                  | dedicated av1-1-b8-XX-tiles.ivf bitstream, so  |
|    |                                  |         |                                  | the only multi-tile vectors in the manifest    |
|    |                                  |         |                                  | are the SVC streams (L1T2: Cols=3 Rows=1;      |
|    |                                  |         |                                  | L2T1 / L2T2: Cols=3 base, Cols=4 enhancement). |
|    |                                  |         |                                  | All three carry VectorLabelMultiTile in        |
|    |                                  |         |                                  | testvector/libaom_manifest.go for filtering.   |
|    |                                  |         |                                  | L1T2 frame 0 PASS in the relevant-cohort       |
|    |                                  |         |                                  | dry-run; L2T1 / L2T2 enhancement-layer frame 0 |
|    |                                  |         |                                  | errors "threading: invalid batch" before tile  |
|    |                                  |         |                                  | reconstruction - the root cause is the multi-  |
|    |                                  |         |                                  | pool SVC surface routing path, not the tile    |
|    |                                  |         |                                  | boundary. See the SVC vector table in §2.      |
|    | Tile lists (OBU_TILE_LIST)       | Yes     | internal/av1/parser/tile_list.go | ParseTileListOBU parses the                    |
|    |                                  |         | internal/av1/decoder/stream.go   | tile_list_obu() header and per-tile entries    |
|    |                                  |         |                                  | (anchor_frame_idx, anchor_tile_row/col,        |
|    |                                  |         |                                  | tile_data_size, tile data slice); EventTileList|
|    |                                  |         |                                  | carries the parsed structure plus TileListErr  |
|    |                                  |         |                                  | for partial payloads. End-to-end tile-list     |
|    |                                  |         |                                  | decode (anchor-frame reuse + per-tile          |
|    |                                  |         |                                  | reconstruction blit) is not yet wired.         |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 25 | Frame type: key                  | Yes     | internal/av1/parser/frame.go     | FrameTypeKey; full reference reset path.       |
|    | Frame type: intra-only           | Yes     | internal/av1/parser/frame.go     | FrameTypeIntraOnly; intra-only refresh.        |
|    | Frame type: inter                | Yes     | internal/av1/parser/frame.go     | FrameTypeInter; mostly bit-exact, mfmv/mv      |
|    |                                  |         |                                  | desync on later frames (see vector table).     |
|    | Frame type: switch               | Yes     | internal/av1/parser/frame.go     | FrameTypeSwitch parsed; ErrorResilientMode     |
|    |                                  |         |                                  | + FrameSizeOverride + RefreshFrameFlags=0xff   |
|    |                                  |         |                                  | implied; reference slots reset on the          |
|    |                                  |         |                                  | switch surface. Parser + stream regression     |
|    |                                  |         |                                  | tests committed; no upstream libaom v3.14.0    |
|    |                                  |         |                                  | switch-frame conformance vector available.     |
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
|    | OBU: Metadata                    | Yes     | internal/av1/obu/metadata.go     | ParseMetadata; all spec variants (ITU-T T.35,  |
|    |                                  |         |                                  | HDR-CLL, HDR-MDCV, scalability, timecode).     |
|    | OBU: RedundantFrameHeader        | Yes     | internal/av1/decoder/stream.go   | Accepted; ignored if a frame header is already |
|    |                                  |         |                                  | active for the temporal unit.                  |
|    | OBU: Padding                     | Yes     | internal/av1/decoder/stream.go   | Recognised; emitted as EventPadding.           |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 32 | Annex B framing                  | Yes     | internal/av1/obu/annexb.go       | NewAnnexBIterator; length-prefixed temporal /  |
|    |                                  |         | internal/av1/obu/annexb_encode.go| frame / OBU unit iteration; fuzzed.            |
|    |                                  |         |                                  | LowOverheadToAnnexB / AnnexBToLowOverhead      |
|    |                                  |         |                                  | helpers transcode between framing variants     |
|    |                                  |         |                                  | for container repackaging.                     |
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
| 35 | OBU extension header (temporal_id| Yes     | internal/av1/obu/header.go       | obu_extension_header() parsed per spec section |
|    |  + spatial_id + reserved bits)   |         | internal/av1/obu/header_test.go  | 5.3.3: 3-bit temporal_id, 2-bit spatial_id,    |
|    |                                  |         |                                  | 3 reserved bits validated as zero              |
|    |                                  |         |                                  | (ErrReservedBit). PutHeader round-trips IDs    |
|    |                                  |         |                                  | for every (T<8, S<4) pair.                     |
|    | TemporalID / SpatialID on Event  | Yes     | internal/av1/decoder/stream.go   | Every Event PushUnit emits carries the OBU     |
|    |                                  |         | internal/av1/decoder/stream_svc_test.go | extension header's (TemporalID, SpatialID); |
|    |                                  |         |                                  | sequence, frame, frame-header, tile-group,     |
|    |                                  |         |                                  | metadata, tile-list, padding all tagged.       |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 36 | operating_points_cnt_minus_1 +   | Yes     | internal/av1/parser/sequence.go  | Sequence header parses up to 32 operating      |
|    | operating_point_idc[]            |         |                                  | points (12-bit IDC, level, tier, optional      |
|    |                                  |         |                                  | decoder-model + initial-display-delay).        |
|    | decoder_model_present_for_op[i]  | Yes     | internal/av1/parser/sequence.go  | Per-op DecoderModelPresent gates the           |
|    |                                  |         | internal/av1/parser/frame_size.go| operating_parameters_info(); frame-header      |
|    |                                  |         |                                  | buffer_removal_time decode honours the         |
|    |                                  |         |                                  | per-OP layer filter.                           |
|    | op_pt_idc layer routing          | Yes     | internal/av1/parser/operating_point.go | OperatingPointIDCMatches /                 |
|    |                                  |         | internal/av1/parser/operating_point_test.go | SelectOperatingPoint replicate libaom's |
|    |                                  |         |                                  | is_obu_in_current_operating_point() filter:    |
|    |                                  |         |                                  | idc==0 -> any layer; otherwise low-8 bits      |
|    |                                  |         |                                  | select temporal_id, high-4 bits select         |
|    |                                  |         |                                  | spatial_id. Re-exported as                     |
|    |                                  |         |                                  | OperatingPointIDCMatches and                   |
|    |                                  |         |                                  | SelectOperatingPoint at the root.              |
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

**The lenient gate is now 8/8 PASS** after `5f88540` closed the
last reconstruction-layer divergence on `intrabc_extreme_dv`. The
`testvectors` CI workflow asserts the full eight-vector pass set
on every push (`c55be7e`). The strict every-frame gate is still
informational; `16x16_size`, `all-intra`, and `intrabc_extreme_dv`
now clear it (3/8) — the entire intra path is byte-exact end-to-end.
The remaining five all fail in inter reconstruction: four
(`quantizer-00`, `cdf_update`, `mv`, `mfmv`) diverge at frame 1, the
first inter frame, and `monochrome` reaches frame 3 (frames 0-2,
including an inter frame with mfmv projections, are byte-exact).

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
| 3 | av1-1-b8-02-allintra.ivf                          | PASS    | PASS   | All 39 frames match under strict; the      |
|   | (libaom av1 8-bit all-intra)                      |         |        | intra chain is byte-exact end-to-end.      |
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
| 7 | av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf     | PASS    | PASS   | Both frames match under strict after the   |
|   | (libaom av1 8-bit intra-only intrabc extreme dv)  |         |        | diagonal-corner SB snapshot for the cross- |
|   |                                                   |         |        | SB intrabc DV scan (5f88540), paired with  |
|   |                                                   |         |        | the tx_size neighbor context snapshot      |
|   |                                                   |         |        | (2bae671).                                 |
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
| 8 | av1-1-b8-24-monochrome.ivf                        | PASS    | FAIL   | Frame 3 (frames 0-2 match, including a      |
|   | (libaom av1 8-bit monochrome)                     |         |        | frame-1 inter frame with mfmv projections).|
+---+---------------------------------------------------+---------+--------+--------------------------------------------+
```

### SVC vector coverage

The SVC vectors are not part of `SuiteLevelFast` and are gated by
`GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1` (`make dryrun-extended`)
or `GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1` against
`SuiteLevelRelevant`. Three SVC vectors ship under
`internal/av1/testdata/libaom/`; the integrator-facing usage is
documented in [docs/svc.md](docs/svc.md).

```
+---+---------------------------------------------------+--------+---------+--------+--------------------------------------------+
| # | Vector                                            | Layout | Lenient | Strict | Notes                                       |
+---+---------------------------------------------------+--------+---------+--------+--------------------------------------------+
| 1 | av1-1-b8-22-svc-L1T2.ivf                          | L1T2   | PARTIAL | FAIL   | Single spatial layer (640x360), two         |
|   | (libaom av1 8-bit svc L1T2)                       |        |         |        | temporal layers. Cols=3 Rows=1 (multi-      |
|   |                                                   |        |         |        | tile). Frame 0 (T0 base) PASS; frame 1      |
|   |                                                   |        |         |        | currently errors `threading: invalid batch` |
|   |                                                   |        |         |        | on the inter-layer reference path. Single-  |
|   |                                                   |        |         |        | pool friendly. Tagged VectorLabelMultiTile. |
+---+---------------------------------------------------+--------+---------+--------+--------------------------------------------+
| 2 | av1-1-b8-22-svc-L2T1.ivf                          | L2T1   | FAIL    | FAIL   | Two spatial layers (640x360 + 1280x720).    |
|   | (libaom av1 8-bit svc L2T1)                       |        |         |        | Cols=3 base, Cols=4 enhancement (multi-     |
|   |                                                   |        |         |        | tile). Requires the multi-pool surface      |
|   |                                                   |        |         |        | routing path (`FrameSurfaceProvider` /      |
|   |                                                   |        |         |        | `FrameSurfaceReleaser`); higher-layer       |
|   |                                                   |        |         |        | frame 0 errors `threading: invalid batch`   |
|   |                                                   |        |         |        | at spatial=1 before tile reconstruction.    |
|   |                                                   |        |         |        | Failure is multi-pool surface routing, not  |
|   |                                                   |        |         |        | the tile boundary. Tagged                   |
|   |                                                   |        |         |        | VectorLabelMultiTile.                       |
+---+---------------------------------------------------+--------+---------+--------+--------------------------------------------+
| 3 | av1-1-b8-22-svc-L2T2.ivf                          | L2T2   | FAIL    | FAIL   | Two spatial + two temporal layers           |
|   | (libaom av1 8-bit svc L2T2)                       |        |         |        | (640x360 + 1280x720). Cols=3 base, Cols=4   |
|   |                                                   |        |         |        | enhancement (multi-tile). Multi-pool path.  |
|   |                                                   |        |         |        | Higher-layer frame 0 shares the L2T1 multi- |
|   |                                                   |        |         |        | pool routing failure. Tagged                |
|   |                                                   |        |         |        | VectorLabelMultiTile.                       |
+---+---------------------------------------------------+--------+---------+--------+--------------------------------------------+
```

### Multi-tile vector coverage

The libaom v3.14.0 published AV1 test suite does not include a
dedicated "av1-1-b8-XX-tiles.ivf" stream, so the only committed
multi-tile bitstreams are the three SVC vectors above. All three
carry `VectorLabelMultiTile` in
`internal/av1/testvector/libaom_manifest.go`; the cohort can be
selected via
`LibaomRemoteManifest().SelectRemote(0, VectorLabelMultiTile, nil)`.

Frame-0 status on the multi-tile probe set:

- **L1T2 (Cols=3 Rows=1, single spatial layer).** Frame 0 PASS
  under the lenient gate via the `SuiteLevelRelevant` cohort
  (`GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1`). The 3-tile-
  column decode and tile-group boundary handling do *not* surface
  an entropy desync or tile-group continuation bug on this vector.
  This is the only multi-tile vector that exercises the tile path
  without SVC multi-pool complexity.
- **L2T1 / L2T2.** Frame 0 spatial=0 (the 3-tile base layer)
  decodes; frame 0 spatial=1 (the 4-tile enhancement layer) errors
  `threading: invalid batch` before tile reconstruction begins.
  The root cause is the multi-pool SVC surface routing path and
  *not* the multi-tile boundary; this is the same gap that blocks
  every enhancement-layer frame.

Net: targeted multi-tile decode parity is unproblematic at
Cols<=3 (L1T2 frame 0). Cols=4 with SVC spatial scalability stays
gated behind the multi-pool surface-routing gap that already
blocks L2T1 / L2T2 single-frame parity. A dedicated non-SVC
multi-tile vector remains absent from the libaom suite; if one
becomes available upstream it should be added to
`SuiteLevelExtended` so the tile path can be probed independently
of SVC.

Run the SVC dry-run with:

```sh
GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 \
GOAV1_SCALED_PRED=1 \
    go test -tags goav1_oracle ./internal/av1/testvector \
    -run TestLibaomExtendedFrameWorkDryRun -count=1 -timeout 1800s -v
```

Each subtest emits the standard `vector=NAME frames=N md5_matches=M
first_mismatch=F` summary line. The `GOAV1_SCALED_PRED=1` env var
opts the scaled-inter-prediction path in for L2T1 / L2T2; without it
mismatched-size references fail with `threading: invalid batch`. See
[docs/svc.md](../docs/svc.md) for the integrator-facing guide and the
list of supported / unsupported SVC features.

The SVC vectors are *not* in the CI gate today. The reference
multi-pool harness is `libaomSpatialLayers` in
`internal/av1/testvector/libaom_oracle_test.go`; it is the canonical
SVC integration shape to mirror when wiring SVC into production
callers.

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
- **Vector 7 (`intrabc-extreme-dv`).** Intra-only. **Lenient
  gate now PASS** after `5f88540` added the diagonal SB-corner
  snapshot for the cross-superblock intrabc DV scan, building on
  the `tx_size` neighbor context snapshot (`2bae671`) that brought
  the entropy stream to bit-exact parity. Strict every-frame mode
  still mismatches on later frames; the surface to inspect there
  is `internal/av1/tile/block_loop.go` (intrabc reference-fetch /
  edge-clip across the cross-SB copy region).
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

1. **IntraBC strict every-frame parity.** The fast-suite intrabc
   vector now PASSES the lenient first-frame gate after the
   diagonal SB-corner snapshot for the cross-superblock intrabc DV
   scan (`5f88540`) and the `tx_size` neighbor context snapshot
   (`2bae671`). Strict every-frame mode still mismatches on later
   frames; that is now a diagnostic gap rather than a lenient-gate
   failure.
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
   any remaining 10/12-bit path bugs. Bit-depth conformance fixes have
   landed on the dequant and inverse-transform stages (`42a8b88`,
   `403e42b`, `266e8cd`), but the extended-cohort 10-bit `quantizer_32`
   and `quantizer_63` vectors still mismatch. The root cause is
   suspected to live elsewhere in reconstruction rather than in the
   transform/dequant inputs.
5. **Scalable video coding (SVC).** Per-spatial-layer dry-run frame
   state (`ea6ad77`), inter-layer reference resolution
   (`51de381`), and scaled inter prediction (`34375a5`) plus the
   warp+scaled fallback (`cba7b3e`) are wired end-to-end. The
   scaled-prediction dispatcher is gated behind `GOAV1_SCALED_PRED=1`
   (runtime) and the `goav1_scaled_pred` build tag. L1T2 still
   mismatches at frame 1 and L2T1 / L2T2 still mismatch on the
   higher-spatial-layer frame 0; the multi-pool surface routing
   (`FrameSurfaceProvider` / `FrameSurfaceReleaser`) is implemented
   in `internal/av1/decoder/svc.go` but not yet re-exported at the
   root package. See [docs/svc.md](docs/svc.md) and the SVC vector
   coverage table in section 2.
6. **Palette wiring.** Palette mode decode is implemented in
   `internal/av1/tile/palette.go` but is not yet exercised by the
   block-loop predictor. Wire it in once a palette-using vector is in
   the dryrun-fast set.
7. **Tile list OBU end-to-end decode.** Tile list OBU parsing now
   landed via `parser.ParseTileListOBU` and the `TileList*` public
   types; `EventTileList` carries the parsed structure. The remaining
   work is end-to-end tile-list playback (anchor-frame reuse,
   per-tile reconstruction, and the output-frame blit step from
   libaom's `read_and_decode_one_tile_list`). No fast-suite vector
   requires it today. (Metadata OBU payload parsing landed via
   `obu.ParseMetadata` and the `Metadata*` public types in the root
   package.)
8. **Switch frames.** Parser routes `FrameTypeSwitch` with
   `error_resilient_mode = 1`, `frame_size_override_flag = 1`, and
   `refresh_frame_flags = 0xff` per AV1 spec §5.9.1 / §5.9.5;
   `ReferenceState` and `SurfaceReferences` both refresh every slot
   onto the decoded switch surface so the decoder can resume
   bitstream switching without any prior context. Parser, frame-
   size, and stream-level regression tests are committed
   (`TestParseFrameHeaderPrefixSwitchFrame`,
   `TestParseFrameSizeSwitchFrame*`,
   `TestStreamSwitchFrame*`). The upstream libaom v3.14.0 test-data
   set does not ship a dedicated `S_FRAME` IVF, so no end-to-end
   MD5 oracle vector is committed to the extended cohort.

With the fast suite now 8/8 PASS under the lenient first-frame
gate, the next milestone is bit-exact strict every-frame parity on
the same eight vectors, followed by `SuiteLevelRelevant`, which
adds intra-bc beyond the extreme-DV case, SVC, more film-grain
vectors, monochrome variants, and 10-bit content.

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
