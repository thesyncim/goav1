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
|    | Profile 1 (High)                 | Yes     | internal/av1/parser/sequence.go  | 4:4:4 8/10-bit all-intra, inter, palette,      |
|    |                                  |         | internal/av1/testvector/profiles | CDEF/restoration, film grain, and super-res    |
|    |                                  |         |                                  | clips pass.                                    |
|    | Profile 2 (Professional)         | Partial | internal/av1/parser/sequence.go  | 4:2:2 8/10-bit and 4:2:0 12-bit edge/super-res |
|    |                                  |         | internal/av1/testvector/profiles | clips pass; broader 10/12-bit vector sweep     |
|    |                                  |         |                                  | covered by opt-in extended gates.              |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  2 | Bit depth 8                      | Yes     | internal/av1/frame/              | Full pipeline; primary conformance target.     |
|    | Bit depth 10                     | Yes     | internal/av1/frame/              | 10-bit libaom relevant/extended/film-grain     |
|    |                                  |         | internal/av1/loopfilter/         | vectors pass strict per-frame MD5.             |
|    |                                  |         | internal/av1/transform/          |                                                |
|    |                                  |         | internal/av1/quantize/           |                                                |
|    | Bit depth 12                     | Partial | internal/av1/frame/              | Vendored 12-bit profile-2 clips pass           |
|    |                                  |         | internal/av1/loopfilter/         | byte-exactly; wider 12-bit vector coverage is  |
|    |                                  |         |                                  | still limited.                                 |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  3 | Subsampling 4:2:0                | Yes     | internal/av1/parser/sequence.go  | Default decode layout; covered by every fast   |
|    |                                  |         | internal/av1/frame/              | conformance vector.                            |
|    | Subsampling 4:2:2                | Yes     | internal/av1/parser/sequence.go  | 4:2:2 8/10-bit all-intra and 8-bit inter clips |
|    |                                  |         | internal/av1/testvector/profiles | pass byte-exactly.                             |
|    | Subsampling 4:4:4                | Yes     | internal/av1/parser/sequence.go  | 4:4:4 8/10-bit all-intra and inter profile    |
|    |                                  |         | internal/av1/testvector/profiles | clips pass byte-exactly.                       |
|    | Monochrome (4:0:0)               | Yes     | internal/av1/frame/              | Surfaces drop the UV planes; 8-bit and 10-bit  |
|    |                                  |         | internal/av1/parser/sequence.go  | libaom monochrome vectors pass strict MD5.     |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  4 | Color range: limited (studio)    | Yes     | internal/av1/parser/sequence.go  | ColorRange=false parsed; no clipping path      |
|    |                                  |         |                                  | beyond sample range applies in decode.         |
|    | Color range: full                | Yes     | internal/av1/parser/sequence.go  | ColorRange=true parsed; output values are not  |
|    |                                  |         |                                  | rescaled, matching libaom.                     |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
|  5 | Superblock 64x64                 | Yes     | internal/av1/parser/sequence.go  | use_128x128_superblock=false; default for      |
|    |                                  |         | internal/av1/tile/partition.go   | every fast conformance vector.                 |
|    | Superblock 128x128               | Yes     | internal/av1/parser/sequence.go  | Use128x128Superblock flag parsed and threaded  |
|    |                                  |         | internal/av1/parser/restoration  | into restoration unit sizing; forced-root      |
|    |                                  |         | internal/av1/testvector/profiles | 128x128 vendored clip passes byte-exactly.     |
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
|  8 | Palette Y                        | Yes     | internal/av1/tile/palette.go     | Y palette size/color/CDF entry decoding, map   |
|    |                                  |         | internal/av1/threading/predict.go | prediction, and 4:4:4 profile vector pass.     |
|    | Palette UV                       | Yes     | internal/av1/tile/palette.go     | UV palette size/color/CDF entry decoding, map  |
|    |                                  |         | internal/av1/threading/predict.go | prediction, and 4:4:4 profile vector pass.     |
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
|    | Super-resolution (enabled)       | Yes     | internal/av1/superres/superres.go | Caller-owned full path publishes upscaled     |
|    |                                  |         | internal/av1/decoder/postfilter_superres.go | references through external surface |
|    |                                  |         | internal/av1/decoder/svc.go       | IDs; all-key, restoration, and Profile 1      |
|    |                                  |         |                                  | inter superres clips pass.                    |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 23 | Film grain synthesis             | Yes     | internal/av1/filmgrain/          | Gaussian RNG, scaling LUTs, luma/chroma grain  |
|    |                                  |         | internal/av1/decoder/postfilter_filmgrain.go | blocks, per-row apply; covered by   |
|    |                                  |         |                                  | 8-bit, 10-bit, and Profile 1 4:4:4 strict     |
|    |                                  |         |                                  | film-grain vectors.                            |
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 24 | Tile groups: single              | Yes     | internal/av1/parser/tile_group.go | Single-tile group is the default path; every  |
|    |                                  |         |                                  | committed fast-suite vector uses Cols=1 Rows=1.|
|    | Tile groups: multiple            | Yes     | internal/av1/parser/tile_group.go | SVC multi-tile vectors and a vendored non-SVC  |
|    |                                  |         | internal/av1/decoder/stream.go   | multi-tile clip pass strict MD5.               |
|    | Tile lists (OBU_TILE_LIST)       | Partial | internal/av1/parser/tile_list.go | ParseTileListOBU parses the                    |
|    |                                  |         | internal/av1/decoder/stream.go   | tile_list_obu() header and per-tile entries    |
|    |                                  |         |                                  | (anchor_frame_idx, anchor_tile_row/col,        |
|    |                                  |         |                                  | tile_data_size, tile data slice); EventTileList|
|    |                                  |         |                                  | carries the parsed structure plus TileListErr  |
|    |                                  |         |                                  | for partial payloads. The residual decode      |
|    |                                  |         |                                  | runner fails loudly with                       |
|    |                                  |         |                                  | ErrDecoderUnsupportedTileList until            |
|    |                                  |         |                                  | end-to-end tile-list decode (anchor-frame      |
|    |                                  |         |                                  | reuse + per-tile reconstruction blit) is wired.|
+----+----------------------------------+---------+----------------------------------+------------------------------------------------+
| 25 | Frame type: key                  | Yes     | internal/av1/parser/frame.go     | FrameTypeKey; full reference reset path.       |
|    | Frame type: intra-only           | Yes     | internal/av1/parser/frame.go     | FrameTypeIntraOnly; intra-only refresh.        |
|    | Frame type: inter                | Yes     | internal/av1/parser/frame.go     | FrameTypeInter; covered by fast, relevant,     |
|    |                                  |         |                                  | extended, and profile multi-frame gates.       |
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
| 28 | Temporal motion field projection | Yes     | internal/av1/tile/motion_field.go | TemporalMotionField.Setup matches libaom      |
|    |                                  |         | internal/av1/threading/ref_mv_frame.go | ordering; storage and entry binding       |
|    |                                  |         | internal/av1/decoder/motion_field.go | exposed; mv, mfmv, and SVC vectors pass    |
|    |                                  |         |                                  | strict per-frame MD5.                          |
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

## 2. Vector coverage

The framework dry-run gates use strict per-frame MD5. Current coverage is:

- `make dryrun-fast`: 14/14 libaom fast vectors pass.
- `make dryrun-relevant-supported`: 14/14 relevant vectors pass, including
  8-bit and 10-bit film grain and monochrome.
- `make dryrun-full`: 240/240 committed remote libaom vectors pass.
- `make dryrun-extended`: 226/226 opt-in diagnostic vectors pass,
  including the 8-bit and 10-bit quantizer sweeps, odd-size clips, larger
  sizes, SVC L1T2/L2T1/L2T2, and multi-tile coverage carried by the SVC
  streams.
- `make dryrun-profiles`: 27/27 vendored profile clips pass, covering
  profile 1 4:4:4 8/10-bit all-intra, inter, screen-content palette,
  CDEF/restoration, 8/10-bit film grain, all-key superres, and inter superres,
  profile 2 4:2:2 8-bit all-intra/inter, profile 2 4:2:2 10-bit edge-size,
  profile 2 4:2:0 12-bit
  including odd edge sizes, 12-bit superres, 8-bit superres,
  superres plus loop restoration, a forced-root 128x128 superblock clip,
  edge-MV, and a non-SVC multi-tile clip.
- `make dryrun-corpus`: optional generated real-content corpus stream-MD5
  conformance. The current generated matrix is 18 clips / 864 visible frames
  across 144p, 288p, 360p, 720p, 8/10-bit, inter/intra, and tiled cases. This
  target requires the local ignored corpus generated by
  `scripts/gen_bench_corpus.sh` and skips cleanly when it is absent.

### SVC vector coverage

The remote libaom SVC vectors are part of the strict relevant/extended gates
and currently pass:

- `av1-1-b8-22-svc-L1T2.ivf` in `make dryrun-relevant-supported`.
- `av1-1-b8-22-svc-L2T1.ivf` and `av1-1-b8-22-svc-L2T2.ivf` in
  `make dryrun-extended`.

The framework path handles per-spatial-layer state, cross-layer reference
resolution, and multi-pool surface routing in the oracle harness. Public
ergonomics for multi-pool SVC are documented separately in
[docs/svc.md](docs/svc.md).

### Multi-tile vector coverage

The remote libaom suite carries multi-tile coverage through the SVC vectors,
and the vendored profile corpus adds a non-SVC `multitile-2x1-rows-256x256`
clip. Both paths pass strict MD5. A broader non-SVC libaom tile corpus should
still be added if upstream publishes one.

### Full and extended coverage

`SuiteLevelFull` carries all 240 checksum-pinned remote libaom vectors.
`SuiteLevelExtended` carries the 226 opt-in size, quantizer, and SVC diagnostic
vectors that used to expose edge and inter-prediction gaps. The current
`make dryrun-full` and `make dryrun-extended` gates run them under strict MD5
and pass every vector.

### How to reproduce

Fast strict gate:

```sh
make dryrun-fast
```

Relevant strict gate:

```sh
make dryrun-relevant-supported
```

Full strict gate:

```sh
make dryrun-full
```

Extended strict gate:

```sh
make dryrun-extended
```

Vendored profile clips:

```sh
make dryrun-profiles
```

Generated real-content corpus:

```sh
make dryrun-corpus
```

Each remote-vector subtest logs a trailing
`vector=NAME temporal_units=N md5_matches=M first_mismatch=F` line.
`first_mismatch=-1` means every emitted frame matched.

---

## 3. Current diagnostic surfaces

There are no known MD5 mismatches in the committed remote-vector framework
manifest or vendored profile corpus. New failures should be triaged from the
strict dry-run summary line and then narrowed to one of these surfaces:

- Parser/sequence/profile gating: `internal/av1/parser/sequence.go`.
- Prediction, residual, and transform decode:
  `internal/av1/tile/`, `internal/av1/threading/`, and
  `internal/av1/transform/`.
- Reference-MV and temporal-motion storage:
  `internal/av1/tile/ref_mv.go`, `internal/av1/tile/motion_field.go`, and
  `internal/av1/threading/ref_mv_frame.go`.
- Postfilter publication and display output:
  `internal/av1/decoder/postfilter*.go`.

When chasing a new vector mismatch, use a local untracked `debug_*_test.go`
probe and keep committed changes focused on the resulting fix or guard.

---

## 4. Forward-looking gaps

The remaining work is no longer driven by known failures in the current
manifest. The next production-readiness items are:

1. **Broader real-world/profile corpus.** Keep expanding the local generated
   real-content corpus and add small committed representative clips when the
   licensing/size tradeoff is acceptable. Keep adding profile-2 10/12-bit 4:2:2 and
   12-bit 4:4:4 streams as upstream or locally generated goldens become
   available; the current vendored corpus includes 4:2:0 12-bit odd-size CDEF,
   edge-motion, and super-res coverage.
2. **Tile list OBU playback.** `EventTileList` parsing is present and the
   residual decode runner now returns `ErrDecoderUnsupportedTileList` instead
   of silently ignoring playback, but end-to-end tile-list reconstruction and
   output-frame blitting remain future work.
3. **Switch-frame oracle coverage.** Parser and stream-level switch-frame
   regressions exist, but the upstream libaom v3.14.0 test-data set does not
   ship a dedicated `S_FRAME` IVF with MD5 goldens.

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
