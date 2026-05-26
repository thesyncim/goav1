package goav1

// This file re-exports the per-tile decode types from the internal tile,
// loopfilter, and restoration packages.
//
// The exported identifiers fall into the following groups:
//
//   - LoopFilter* identifies loop-filter planes and edges used by the
//     loop-filter helpers in loopfilter.go.
//   - TileBlockLevel*, TilePartition*, TileBlockSize* describe the AV1
//     superblock partitioning tree and the block sizes that result from it.
//   - TileBlockLoop* drives a deterministic block-by-block walk over a tile
//     with caller-owned context carriers.
//   - TileBlockCoeff* / TileLuma*/Chroma*CoeffTree* parameterize the
//     coefficient decode tree the tile entropy reader walks.
//   - TileTXBContext, TileCoeffContextRequest, TileTXBDecodeRequest /
//     TileTXBDecodeResult drive the per-transform-block entropy context.
//   - TileIntraMode*, TileFilterIntraMode*, TileChromaIntraMode*,
//     TileCFLAlphaResult, TileInterMode*, TileMVJoint, TileMVSubpelPrecision
//     describe the per-block mode and motion decode outputs.
//   - TileTransformSize*, TileTransformPartition*, TileTransformTree*,
//     TileExtTXSetType, TileExtTXSets* describe the AV1 transform partition
//     tree and ext-tx set selection.
//   - TileReferenceMV* / TileTemporalMotion* hold the per-tile reference and
//     temporal motion-vector buffers used by inter prediction.
//   - TileRestoration* / RestorationWiener*, RestorationSGRParams hold the
//     loop-restoration plan, records, scratches, and apply results consumed
//     by tile_restoration.go and the post-filter helpers in postfilter.go.
//
// Concrete entry points for the operations these types parameterize live in
// the corresponding *.go files (postfilter.go, tile_restoration.go,
// tile_coeff_tree.go, tile_coeff_context.go, tile_transform_context.go,
// tile_geometry.go, decoder_prediction.go, decoder_coeff_reconstruct.go).

import (
	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// LoopFilterPlane identifies the luma or chroma plane targeted by a
// loop-filter operation.
type LoopFilterPlane = internalloopfilter.Plane

// LoopFilterEdge identifies the orientation (vertical or horizontal) of a
// loop-filter edge.
type LoopFilterEdge = internalloopfilter.Edge

// TileBlockLevel selects one of the AV1 superblock partition levels
// (128x128 down to 8x8). See the TileBlockLevel* constants for the
// enumeration.
type TileBlockLevel = internaltile.BlockLevel

// TilePartition is one of the AV1 partition codes (none, H/V split,
// 4-way split, T-split variants, H4/V4). See the TilePartition*
// constants for the enumeration.
type TilePartition = internaltile.Partition

// TileBlockSize selects one of the AV1 block-size codes (4x4 through
// 128x128 with rectangular variants). See the TileBlockSize* constants.
//
// See also: TileBlockDimensions.
type TileBlockSize = internaltile.BlockSize

// TileBlockDimensions reports width and height in mi units for a
// TileBlockSize.
type TileBlockDimensions = internaltile.BlockDimensions

// TileBlockVisit identifies one block within a tile's block tree (mi
// origin, mi size, partition path). It is the unit of work the
// TileBlockLoopVisitor callback receives.
type TileBlockVisit = internaltile.BlockVisit

// TileReferenceFrame identifies one of the seven AV1 inter reference
// slots. See the TileReferenceFrame* constants.
type TileReferenceFrame = internaltile.ReferenceFrame

// TileReferenceMVFrame is the per-reference MV frame data the
// inter-prediction pipeline consults during block decode.
type TileReferenceMVFrame = internaltile.ReferenceMVFrame

// TileReferenceMVEntry is one entry of a TileReferenceMVFrame.
type TileReferenceMVEntry = internaltile.ReferenceMVEntry

// TileTemporalMotionField holds the per-tile temporal motion-vector
// field used by AV1 inter prediction.
type TileTemporalMotionField = internaltile.TemporalMotionField

// TileTemporalMotionEntry is one entry of a TileTemporalMotionField.
type TileTemporalMotionEntry = internaltile.TemporalMotionEntry

// TileDecodeState is the per-job decode state: entropy reader,
// delta-q/loopfilter state, segmentation/loop-filter caches, and the
// frame-context retention flag.
//
// See also: TileDecodeOptions.
type TileDecodeState = internaltile.DecodeState

// TileDecodeOptions configures a TileDecodeState (CDF update behavior,
// retention flag overrides).
type TileDecodeOptions = internaltile.DecodeOptions

// TileBlockLoopRequest configures a deterministic block-by-block walk
// over a tile, including the segmentation maps and caller-supplied
// callbacks invoked at superblock/coefficient boundaries.
//
// See also: TileBlockLoopVisitor, TileBlockLoopContextCarrier.
type TileBlockLoopRequest = internaltile.BlockLoopRequest

// TileBlockLoopContextCarrier holds the caller-owned root-column
// context state the block loop reads and writes as it walks the tile.
type TileBlockLoopContextCarrier = internaltile.BlockLoopContextCarrier

// TileBlockLoopRootAboveContext is one entry of the above-context array
// inside a TileBlockLoopContextCarrier.
type TileBlockLoopRootAboveContext = internaltile.BlockLoopRootAboveContext

// TileBlockLoopRootLeftContext is the per-superblock left-context state
// the block loop tracks while walking a tile.
type TileBlockLoopRootLeftContext = internaltile.BlockLoopRootLeftContext

// TileBlockLoopVisit is the per-block visit value passed to the
// TileBlockLoopVisitor callback.
type TileBlockLoopVisit = internaltile.BlockLoopVisit

// TileBlockLoopStats reports per-tile block-loop counters (partition
// reads, blocks decoded, modes, motion vectors, transform blocks, etc.).
type TileBlockLoopStats = internaltile.BlockLoopStats

// TileBlockLoopVisitor is the per-block callback the tile block loop
// invokes for each decoded block.
type TileBlockLoopVisitor = internaltile.BlockLoopVisitor

// TileBlockLoopCoeffVisitor is the per-block coefficient callback the
// tile block loop invokes alongside the visitor when coefficient data
// is required.
type TileBlockLoopCoeffVisitor = internaltile.BlockLoopCoeffVisitor

// TileTransformCDFs holds the per-tile transform CDF tables used during
// transform-tree decode.
//
// See also: InitTileTransformCDFsDefault.
type TileTransformCDFs = internaltile.TransformCDFs

// TileBlockCoeffCDFs holds the per-tile block-level coefficient CDF
// tables consumed by the block coefficient decoder.
type TileBlockCoeffCDFs = internaltile.BlockCoeffCDFs

// TileBlockCoeffScratch is the caller-owned per-block scratch consumed
// by the block coefficient decoder.
type TileBlockCoeffScratch = internaltile.BlockCoeffScratch

// TileBlockCoeffRequest bundles caller-owned inputs (CDFs, scratch,
// transform tree, visitors) for the block coefficient decoder.
//
// See also: TileBlockCoeffResult, TileBlockCoeffVisitor.
type TileBlockCoeffRequest = internaltile.BlockCoeffRequest

// TileBlockCoeffBlock is one decoded coefficient block produced by the
// block coefficient decoder.
type TileBlockCoeffBlock = internaltile.BlockCoeffBlock

// TileBlockCoeffResult reports the per-block decoded coefficient
// statistics.
//
// See also: TileBlockCoeffRequest.
type TileBlockCoeffResult = internaltile.BlockCoeffResult

// TileBlockCoeffVisitor is the per-decoded-block callback the
// coefficient decoder invokes once a block's coefficients are
// available.
type TileBlockCoeffVisitor = internaltile.BlockCoeffVisitor

// TileCoeffPlaneType identifies a luma or chroma plane within the
// coefficient decode helpers.
type TileCoeffPlaneType = internaltile.CoeffPlaneType

// TileTXBContext is the per-transform-block entropy context (skip/dc
// sign) shared across the coefficient decode pipeline.
type TileTXBContext = internaltile.TXBContext

// TileCoeffContextRequest bundles caller-owned inputs (above/left
// neighbors, plane type, block size, transform type) for marking
// TileTXBContext slots.
type TileCoeffContextRequest = internaltile.CoeffContextRequest

// TileTXBDecodeRequest bundles caller-owned inputs for one
// transform-block coefficient decode call.
//
// See also: TileTXBDecodeResult.
type TileTXBDecodeRequest = internaltile.TXBDecodeRequest

// TileTXBDecodeResult reports the per-transform-block coefficient
// decode statistics.
//
// See also: TileTXBDecodeRequest.
type TileTXBDecodeResult = internaltile.TXBDecodeResult

// TileCoeffCDFs holds the per-tile coefficient CDF tables shared
// across all transform-block decode calls in a tile.
//
// See also: InitTileCoeffCDFsDefault.
type TileCoeffCDFs = internaltile.CoeffCDFs

// TileCoeffTreeScratch is the caller-owned scratch shared by the luma
// and chroma coefficient-tree decode helpers.
type TileCoeffTreeScratch = internaltile.LumaCoeffTreeScratch

// TileLumaCoeffTreeRequest bundles caller-owned inputs (CDFs, scratch,
// visitor) for the luma coefficient-tree decoder.
//
// See also: TileLumaCoeffBlock, TileLumaCoeffStats.
type TileLumaCoeffTreeRequest = internaltile.LumaCoeffTreeRequest

// TileLumaCoeffBlock is one decoded luma coefficient block produced by
// the luma coefficient-tree decoder.
type TileLumaCoeffBlock = internaltile.LumaCoeffBlock

// TileLumaCoeffStats reports the per-tile luma coefficient decode
// counters.
type TileLumaCoeffStats = internaltile.LumaCoeffStats

// TileLumaCoeffVisitor is the per-block callback the luma
// coefficient-tree decoder invokes once a block's coefficients are
// available.
type TileLumaCoeffVisitor = internaltile.LumaCoeffVisitor

// TileChromaCoeffTreeRequest bundles caller-owned inputs (CDFs,
// scratch, visitor) for the chroma coefficient-tree decoder.
//
// See also: TileChromaCoeffBlock.
type TileChromaCoeffTreeRequest = internaltile.ChromaCoeffTreeRequest

// TileChromaCoeffBlock is one decoded chroma coefficient block produced
// by the chroma coefficient-tree decoder.
type TileChromaCoeffBlock = internaltile.ChromaCoeffBlock

// TileChromaCoeffVisitor is the per-block callback the chroma
// coefficient-tree decoder invokes once a block's coefficients are
// available.
type TileChromaCoeffVisitor = internaltile.ChromaCoeffVisitor

// TileCoeffTransformRequest carries the caller-owned inputs for
// resolving one transform-type selection at coefficient decode time.
type TileCoeffTransformRequest = internaltile.CoeffTransformRequest

// TileCoeffTransformSelector is the per-block transform-type selector
// hook the coefficient decoder consults when the caller wants to
// override the default ext-tx set choice.
type TileCoeffTransformSelector = internaltile.CoeffTransformSelector

// TileCoeffTransformSelectorFunc is the function form of
// TileCoeffTransformSelector.
type TileCoeffTransformSelectorFunc = internaltile.CoeffTransformSelectorFunc

// TileBlockModeResult reports the decoded per-block mode metadata
// (intra/inter, segment id, skip flag) for one block.
type TileBlockModeResult = internaltile.BlockModeResult

// TileBlockPredictionModeResult reports the per-block prediction mode
// (intra mode, filter-intra mode, chroma mode) selected for one block.
type TileBlockPredictionModeResult = internaltile.BlockPredictionModeResult

// TileIntraMode is the AV1 luma intra mode for one block. See the
// TileIntraMode* constants for the enumeration.
type TileIntraMode = internaltile.IntraMode

// TileFilterIntraMode is the AV1 filter-intra mode for one block. See
// the TileFilterIntraMode* constants.
type TileFilterIntraMode = internaltile.FilterIntraMode

// TileChromaIntraMode is the AV1 chroma intra mode for one block. See
// the TileChromaIntraMode* constants.
type TileChromaIntraMode = internaltile.ChromaIntraMode

// TileCFLAlphaResult reports the decoded CFL alpha indices and joint
// sign for one chroma-from-luma block.
type TileCFLAlphaResult = internaltile.CFLAlphaResult

// TileInterReferencesResult reports the resolved inter reference
// frame(s) for one block.
type TileInterReferencesResult = internaltile.InterReferencesResult

// TileInterMode is the AV1 inter mode for one block (NEARESTMV, NEARMV,
// GLOBALMV, NEWMV). See the TileInterMode* constants.
type TileInterMode = internaltile.InterMode

// TileCompoundInterMode is the AV1 compound inter mode for one block.
// See the TileCompoundInterMode* constants.
type TileCompoundInterMode = internaltile.CompoundInterMode

// TileCompoundInterModeComponents reports the per-component decode
// inputs derived from a TileCompoundInterMode.
type TileCompoundInterModeComponents = internaltile.CompoundInterModeComponents

// TileInterModeResult reports the decoded inter mode for one block,
// including the optional compound and DRL selections.
type TileInterModeResult = internaltile.InterModeResult

// TileReferenceMVCandidate is one entry of a TileReferenceMVStack.
type TileReferenceMVCandidate = internaltile.ReferenceMVCandidate

// TileReferenceMVStack is the caller-owned MV candidate stack the inter
// mode decoder fills before selecting a motion vector.
//
// See also: TileReferenceMVStackResult.
type TileReferenceMVStack = internaltile.ReferenceMVStack

// TileReferenceMVStackResult reports the populated TileReferenceMVStack
// size and candidate metadata for one block.
//
// See also: TileReferenceMVStack.
type TileReferenceMVStackResult = internaltile.ReferenceMVStackResult

// TileInterMVReferenceSet holds the per-reference MV inputs the
// motion-vector decoder consults for one block.
type TileInterMVReferenceSet = internaltile.InterMVReferenceSet

// TileMVJoint selects one of the four AV1 MV joints (zero, horizontal,
// vertical, both). See the TileMVJoint* constants.
type TileMVJoint = internaltile.MVJoint

// TileMVSubpelPrecision selects the AV1 MV sub-pixel precision (none,
// low, high). See the TileMVSubpelPrecision* constants.
type TileMVSubpelPrecision = internaltile.MVSubpelPrecision

// TileMVComponentResult reports the decoded per-component MV value and
// the entropy state at the end of the decode.
type TileMVComponentResult = internaltile.MVComponentResult

// TileMVResidualResult reports the decoded MV residual together with
// the joint and component results.
type TileMVResidualResult = internaltile.MVResidualResult

// TileInterMotionResult reports the resolved inter motion vector
// (precision, joint, components) for one block.
type TileInterMotionResult = internaltile.InterMotionResult

// TileMotionMode is the AV1 motion mode for one block (translation,
// OBMC, warp). See the TileMotionMode* constants.
type TileMotionMode = internaltile.MotionMode

// TileOverlappableNeighbor describes one overlappable neighbor block
// used by OBMC.
type TileOverlappableNeighbor = internaltile.OverlappableNeighbor

// TileOverlappableNeighborSet collects the overlappable neighbors for
// one block during OBMC mode decision.
type TileOverlappableNeighborSet = internaltile.OverlappableNeighborSet

// TileWarpedMotionModel is the per-block warped-motion model used when
// motion mode is warp.
type TileWarpedMotionModel = internaltile.WarpedMotionModel

// TileInterIntraMode is the AV1 inter-intra mode for one block. See
// the TileInterIntraMode* constants.
type TileInterIntraMode = internaltile.InterIntraMode

// TileInterIntraResult reports the decoded inter-intra parameters
// (mode, wedge index, blend mask) for one block.
type TileInterIntraResult = internaltile.InterIntraResult

// TileCompoundType identifies the AV1 compound type for one block
// (average, dist-wtd, wedge, diff-wtd). See the TileCompoundType*
// constants.
type TileCompoundType = internaltile.CompoundType

// TileDiffWtdMaskType selects one of the AV1 diff-wtd compound mask
// variants. See the TileDiffWtdMaskType* constants.
type TileDiffWtdMaskType = internaltile.DiffWtdMaskType

// TileCompoundBlendResult reports the decoded compound blend
// parameters (type, mask index) for one block.
type TileCompoundBlendResult = internaltile.CompoundBlendResult

// TileTransformSize is one of the AV1 transform sizes (4x4 through
// 64x64 with rectangular variants). See the TileTransformSize*
// constants.
//
// See also: TileTransformDimensions.
type TileTransformSize = internaltile.TransformSize

// TileTransformDimensions reports width and height in samples for one
// TileTransformSize.
type TileTransformDimensions = internaltile.TransformDimensions

// TileTransformPartitionRequest bundles caller-owned inputs (block
// size, transform CDFs, scratch) for resolving one transform partition
// decision.
type TileTransformPartitionRequest = internaltile.TransformPartitionRequest

// TileTransformTreeRequest bundles caller-owned inputs (block visit,
// transform CDFs, scratch) for walking the per-block transform
// partition tree.
//
// See also: TileTransformTreeResult, ForEachTileLumaTransformBlock.
type TileTransformTreeRequest = internaltile.TransformTreeRequest

// TileTransformTreeResult reports the decoded transform tree (per-block
// transform sizes and locations) for one block.
//
// See also: TileTransformTreeRequest.
type TileTransformTreeResult = internaltile.TransformTreeResult

// TileTransformBlock is one leaf of a TileTransformTreeResult: its
// location, size, and orientation within the parent block.
type TileTransformBlock = internaltile.TransformBlock

// TileTransformBlockVisitor is the per-transform-block callback invoked
// by ForEachTileLumaTransformBlock.
type TileTransformBlockVisitor = internaltile.TransformBlockVisitor

// TileExtTXSetType selects one of the AV1 ext-tx sets used to choose
// transform types during coefficient decode. See the TileExtTXSet*
// constants.
type TileExtTXSetType = internaltile.ExtTXSetType

// TileRestorationSGRProjInfo holds the per-unit SGR-proj restoration
// parameters (xqd values).
type TileRestorationSGRProjInfo = internaltile.SGRProjInfo

// TileRestorationUnit is one loop-restoration unit description (rect,
// stripe, plane geometry).
type TileRestorationUnit = internaltile.RestorationUnit

// TileRestorationUnitRect describes the pixel rectangle of one
// loop-restoration unit.
type TileRestorationUnitRect = internaltile.RestorationUnitRect

// TileRestorationProcessingStripe describes one processing stripe used
// when applying loop restoration.
type TileRestorationProcessingStripe = internaltile.RestorationProcessingStripe

// TileRestorationProcessingUnit is one processing unit within a
// TileRestorationProcessingStripe.
type TileRestorationProcessingUnit = internaltile.RestorationProcessingUnit

// TileRestorationPlaneGrid is the per-plane loop-restoration unit grid
// derived from the active TileRestorationFramePlan.
type TileRestorationPlaneGrid = internaltile.RestorationPlaneGrid

// TileRestorationUnitRange describes the inclusive unit index range one
// stripe spans within a TileRestorationPlaneGrid.
type TileRestorationUnitRange = internaltile.RestorationUnitRange

// TileRestorationUnitRecord is one caller-owned record describing the
// decoded parameters of one loop-restoration unit.
//
// See also: BindTileRestorationFrameRecordBuffers.
type TileRestorationUnitRecord = internaltile.RestorationUnitRecord

// TileRestorationFramePlan describes the per-plane loop-restoration
// geometry for one frame (unit grid, stripe layout, boundary buffer
// sizing).
//
// See also: DecoderFrameWorkRestorationFramePlan,
// BindTileRestorationFrameRecordBuffers.
type TileRestorationFramePlan = internaltile.RestorationFramePlan

// TileRestorationStripeBoundaries holds the per-plane stripe-boundary
// scratch (above and below) used by loop restoration.
type TileRestorationStripeBoundaries = internaltile.RestorationStripeBoundaries

// TileRestorationFrameBoundaryPlane describes the per-plane boundary
// geometry one TileRestorationStripeBoundaries entry covers.
type TileRestorationFrameBoundaryPlane = internaltile.RestorationFrameBoundaryPlane

// TileRestorationStripeBoundaryScratchSize reports the caller-owned
// per-stripe boundary scratch lengths required by loop restoration.
type TileRestorationStripeBoundaryScratchSize = internaltile.RestorationStripeBoundaryScratchSize

// TileRestorationStripeBoundaryScratch carries caller-owned per-stripe
// boundary scratch arenas (above and below).
type TileRestorationStripeBoundaryScratch = internaltile.RestorationStripeBoundaryScratch

// TileRestorationStripeBoundaryBufferSize reports the caller-owned
// per-frame boundary buffer length required by loop restoration.
type TileRestorationStripeBoundaryBufferSize = internaltile.RestorationStripeBoundaryBufferSize

// TileRestorationUnitScratchSize reports the per-unit Wiener and
// SGR-proj scratch lengths required during loop-restoration apply.
type TileRestorationUnitScratchSize = internaltile.RestorationUnitScratchSize

// TileRestorationUnitScratch carries caller-owned per-unit scratch
// (Wiener buffer and SGR-proj int32 buffer).
type TileRestorationUnitScratch = internaltile.RestorationUnitScratch

// TileRestorationUnitApplyResult reports the per-unit outcome of one
// loop-restoration apply call (success, samples-restored counters).
type TileRestorationUnitApplyResult = internaltile.RestorationUnitApplyResult

// TileRestorationUnitRecordApplyResult reports the per-unit apply
// result alongside the source record for callers that fan out replay.
type TileRestorationUnitRecordApplyResult = internaltile.RestorationUnitRecordApplyResult

// TileRestorationPlaneApplyResult aggregates per-unit apply results
// across one plane.
type TileRestorationPlaneApplyResult = internaltile.RestorationPlaneApplyResult

// TileRestorationFramePlane describes one plane in the frame-level
// loop-restoration apply request.
type TileRestorationFramePlane = internaltile.RestorationFramePlane

// TileRestorationFrameApplyResult aggregates per-plane apply results
// across one frame.
type TileRestorationFrameApplyResult = internaltile.RestorationFrameApplyResult

// TileRestorationUnitRecordBoundaryScratchSize reports per-unit
// boundary scratch lengths when records are paired with stripe
// boundary buffers.
type TileRestorationUnitRecordBoundaryScratchSize = internaltile.RestorationUnitRecordBoundaryScratchSize

// TileRestorationUnitRecordBoundaryScratch carries caller-owned
// per-unit/per-boundary scratch arenas.
type TileRestorationUnitRecordBoundaryScratch = internaltile.RestorationUnitRecordBoundaryScratch

// TileRestorationFrameSampleScratchSize reports the per-frame sample
// scratch lengths needed when loop restoration is applied.
type TileRestorationFrameSampleScratchSize = internaltile.RestorationFrameSampleScratchSize

// RestorationWienerFilter is the 7-tap Wiener filter described by an
// AV1 restoration unit.
type RestorationWienerFilter = internalrestoration.WienerFilter

// RestorationWienerInfo carries the Wiener filter info (taps and clipping)
// for one restoration unit.
type RestorationWienerInfo = internalrestoration.WienerInfo

// RestorationSGRParams is one entry of the AV1 self-guided restoration
// parameter table (radii r0/r1 and per-radius eps0/eps1).
type RestorationSGRParams = internalrestoration.SGRParams

// AV1 tile decode constants.
//
// The TilePartition*, TileBlockModeContexts, TileBlockSize*, TileBlockLevel*,
// and TileTransform*/TileExtTXSet* values enumerate the per-block partition,
// mode, block-size, level, and ext-tx set categories defined by sections 5-9
// of the AV1 specification.
//
// TileReferenceFrame* identifies the seven AV1 inter reference slots, the
// LoopFilterPlane* / LoopFilterEdge* values identify loop-filter targets,
// and TileIntraMode*, TileFilterIntraMode*, TileChromaIntraMode*,
// TileInterMode*, TileCompoundInterMode*, TileMotionMode*,
// TileInterIntraMode*, TileCompoundType*, and TileDiffWtdMaskType* values
// enumerate the per-block prediction modes available to AV1.
const (
	TilePartitionContexts = internaltile.PartitionContexts
	TileMaxPartitionSlots = internaltile.MaxPartitionSlots

	TileBlockModeContexts    = internaltile.BlockModeContexts
	TileMaxBlockModeSlots    = internaltile.MaxBlockModeSlots
	TileFrameLoopFilterCount = internaltile.FrameLoopFilterCount

	TileReferenceFrameNone    TileReferenceFrame = internaltile.ReferenceFrameNone
	TileReferenceFrameLast    TileReferenceFrame = internaltile.ReferenceFrameLast
	TileReferenceFrameLast2   TileReferenceFrame = internaltile.ReferenceFrameLast2
	TileReferenceFrameLast3   TileReferenceFrame = internaltile.ReferenceFrameLast3
	TileReferenceFrameGolden  TileReferenceFrame = internaltile.ReferenceFrameGolden
	TileReferenceFrameBWD     TileReferenceFrame = internaltile.ReferenceFrameBWD
	TileReferenceFrameAltref2 TileReferenceFrame = internaltile.ReferenceFrameAltref2
	TileReferenceFrameAltref  TileReferenceFrame = internaltile.ReferenceFrameAltref

	TileTransformSizeContexts      = internaltile.TransformSizeContexts
	TileTransformSizeCategories    = internaltile.TransformSizeCategories
	TileTransformPartitionContexts = internaltile.TransformPartitionContexts
	TileTransformPartitionCats     = internaltile.TransformPartitionCats

	TileExtTXSetTypes  = internaltile.ExtTXSetTypes
	TileExtTXSetsIntra = internaltile.ExtTXSetsIntra
	TileExtTXSetsInter = internaltile.ExtTXSetsInter
	TileExtTXSizes     = internaltile.ExtTXSizes

	LoopFilterPlaneY LoopFilterPlane = internalloopfilter.PlaneY
	LoopFilterPlaneU LoopFilterPlane = internalloopfilter.PlaneU
	LoopFilterPlaneV LoopFilterPlane = internalloopfilter.PlaneV

	LoopFilterEdgeVertical   LoopFilterEdge = internalloopfilter.EdgeVertical
	LoopFilterEdgeHorizontal LoopFilterEdge = internalloopfilter.EdgeHorizontal

	TileBlockLevel128x128 TileBlockLevel = internaltile.BlockLevel128x128
	TileBlockLevel64x64   TileBlockLevel = internaltile.BlockLevel64x64
	TileBlockLevel32x32   TileBlockLevel = internaltile.BlockLevel32x32
	TileBlockLevel16x16   TileBlockLevel = internaltile.BlockLevel16x16
	TileBlockLevel8x8     TileBlockLevel = internaltile.BlockLevel8x8

	TilePartitionNone         TilePartition = internaltile.PartitionNone
	TilePartitionH            TilePartition = internaltile.PartitionH
	TilePartitionV            TilePartition = internaltile.PartitionV
	TilePartitionSplit        TilePartition = internaltile.PartitionSplit
	TilePartitionTTopSplit    TilePartition = internaltile.PartitionTTopSplit
	TilePartitionTBottomSplit TilePartition = internaltile.PartitionTBottomSplit
	TilePartitionTLeftSplit   TilePartition = internaltile.PartitionTLeftSplit
	TilePartitionTRightSplit  TilePartition = internaltile.PartitionTRightSplit
	TilePartitionH4           TilePartition = internaltile.PartitionH4
	TilePartitionV4           TilePartition = internaltile.PartitionV4

	TileBlockSize128x128 TileBlockSize = internaltile.BlockSize128x128
	TileBlockSize128x64  TileBlockSize = internaltile.BlockSize128x64
	TileBlockSize64x128  TileBlockSize = internaltile.BlockSize64x128
	TileBlockSize64x64   TileBlockSize = internaltile.BlockSize64x64
	TileBlockSize64x32   TileBlockSize = internaltile.BlockSize64x32
	TileBlockSize64x16   TileBlockSize = internaltile.BlockSize64x16
	TileBlockSize32x64   TileBlockSize = internaltile.BlockSize32x64
	TileBlockSize32x32   TileBlockSize = internaltile.BlockSize32x32
	TileBlockSize32x16   TileBlockSize = internaltile.BlockSize32x16
	TileBlockSize32x8    TileBlockSize = internaltile.BlockSize32x8
	TileBlockSize16x64   TileBlockSize = internaltile.BlockSize16x64
	TileBlockSize16x32   TileBlockSize = internaltile.BlockSize16x32
	TileBlockSize16x16   TileBlockSize = internaltile.BlockSize16x16
	TileBlockSize16x8    TileBlockSize = internaltile.BlockSize16x8
	TileBlockSize16x4    TileBlockSize = internaltile.BlockSize16x4
	TileBlockSize8x32    TileBlockSize = internaltile.BlockSize8x32
	TileBlockSize8x16    TileBlockSize = internaltile.BlockSize8x16
	TileBlockSize8x8     TileBlockSize = internaltile.BlockSize8x8
	TileBlockSize8x4     TileBlockSize = internaltile.BlockSize8x4
	TileBlockSize4x16    TileBlockSize = internaltile.BlockSize4x16
	TileBlockSize4x8     TileBlockSize = internaltile.BlockSize4x8
	TileBlockSize4x4     TileBlockSize = internaltile.BlockSize4x4

	TileIntraModeDC               TileIntraMode = internaltile.IntraModeDC
	TileIntraModeVertical         TileIntraMode = internaltile.IntraModeVertical
	TileIntraModeHorizontal       TileIntraMode = internaltile.IntraModeHorizontal
	TileIntraModeD45              TileIntraMode = internaltile.IntraModeD45
	TileIntraModeD135             TileIntraMode = internaltile.IntraModeD135
	TileIntraModeD113             TileIntraMode = internaltile.IntraModeD113
	TileIntraModeD157             TileIntraMode = internaltile.IntraModeD157
	TileIntraModeD203             TileIntraMode = internaltile.IntraModeD203
	TileIntraModeD67              TileIntraMode = internaltile.IntraModeD67
	TileIntraModeSmooth           TileIntraMode = internaltile.IntraModeSmooth
	TileIntraModeSmoothVertical   TileIntraMode = internaltile.IntraModeSmoothVertical
	TileIntraModeSmoothHorizontal TileIntraMode = internaltile.IntraModeSmoothHorizontal
	TileIntraModePaeth            TileIntraMode = internaltile.IntraModePaeth

	TileFilterIntraModeDC         TileFilterIntraMode = internaltile.FilterIntraModeDC
	TileFilterIntraModeVertical   TileFilterIntraMode = internaltile.FilterIntraModeVertical
	TileFilterIntraModeHorizontal TileFilterIntraMode = internaltile.FilterIntraModeHorizontal
	TileFilterIntraModeD157       TileFilterIntraMode = internaltile.FilterIntraModeD157
	TileFilterIntraModePaeth      TileFilterIntraMode = internaltile.FilterIntraModePaeth

	TileChromaIntraModeDC               TileChromaIntraMode = internaltile.ChromaIntraModeDC
	TileChromaIntraModeVertical         TileChromaIntraMode = internaltile.ChromaIntraModeVertical
	TileChromaIntraModeHorizontal       TileChromaIntraMode = internaltile.ChromaIntraModeHorizontal
	TileChromaIntraModeD45              TileChromaIntraMode = internaltile.ChromaIntraModeD45
	TileChromaIntraModeD135             TileChromaIntraMode = internaltile.ChromaIntraModeD135
	TileChromaIntraModeD113             TileChromaIntraMode = internaltile.ChromaIntraModeD113
	TileChromaIntraModeD157             TileChromaIntraMode = internaltile.ChromaIntraModeD157
	TileChromaIntraModeD203             TileChromaIntraMode = internaltile.ChromaIntraModeD203
	TileChromaIntraModeD67              TileChromaIntraMode = internaltile.ChromaIntraModeD67
	TileChromaIntraModeSmooth           TileChromaIntraMode = internaltile.ChromaIntraModeSmooth
	TileChromaIntraModeSmoothVertical   TileChromaIntraMode = internaltile.ChromaIntraModeSmoothVertical
	TileChromaIntraModeSmoothHorizontal TileChromaIntraMode = internaltile.ChromaIntraModeSmoothHorizontal
	TileChromaIntraModePaeth            TileChromaIntraMode = internaltile.ChromaIntraModePaeth
	TileChromaIntraModeCFL              TileChromaIntraMode = internaltile.ChromaIntraModeCFL

	TileInterModeNearestMV TileInterMode = internaltile.InterModeNearestMV
	TileInterModeNearMV    TileInterMode = internaltile.InterModeNearMV
	TileInterModeGlobalMV  TileInterMode = internaltile.InterModeGlobalMV
	TileInterModeNewMV     TileInterMode = internaltile.InterModeNewMV

	TileCompoundInterModeNearestNearest TileCompoundInterMode = internaltile.CompoundInterModeNearestNearest
	TileCompoundInterModeNearNear       TileCompoundInterMode = internaltile.CompoundInterModeNearNear
	TileCompoundInterModeNearestNew     TileCompoundInterMode = internaltile.CompoundInterModeNearestNew
	TileCompoundInterModeNewNearest     TileCompoundInterMode = internaltile.CompoundInterModeNewNearest
	TileCompoundInterModeNearNew        TileCompoundInterMode = internaltile.CompoundInterModeNearNew
	TileCompoundInterModeNewNear        TileCompoundInterMode = internaltile.CompoundInterModeNewNear
	TileCompoundInterModeGlobalGlobal   TileCompoundInterMode = internaltile.CompoundInterModeGlobalGlobal
	TileCompoundInterModeNewNew         TileCompoundInterMode = internaltile.CompoundInterModeNewNew

	TileMVJointZero       TileMVJoint = internaltile.MVJointZero
	TileMVJointHorizontal TileMVJoint = internaltile.MVJointHorizontal
	TileMVJointVertical   TileMVJoint = internaltile.MVJointVertical
	TileMVJointBoth       TileMVJoint = internaltile.MVJointBoth

	TileMVSubpelNone TileMVSubpelPrecision = internaltile.MVSubpelNone
	TileMVSubpelLow  TileMVSubpelPrecision = internaltile.MVSubpelLow
	TileMVSubpelHigh TileMVSubpelPrecision = internaltile.MVSubpelHigh

	TileMotionModeTranslation TileMotionMode = internaltile.MotionModeTranslation
	TileMotionModeOBMC        TileMotionMode = internaltile.MotionModeOBMC
	TileMotionModeWarp        TileMotionMode = internaltile.MotionModeWarp

	TileInterIntraModeDC         TileInterIntraMode = internaltile.InterIntraModeDC
	TileInterIntraModeVertical   TileInterIntraMode = internaltile.InterIntraModeVertical
	TileInterIntraModeHorizontal TileInterIntraMode = internaltile.InterIntraModeHorizontal
	TileInterIntraModeSmooth     TileInterIntraMode = internaltile.InterIntraModeSmooth

	TileCompoundTypeAverage TileCompoundType = internaltile.CompoundTypeAverage
	TileCompoundTypeDistWtd TileCompoundType = internaltile.CompoundTypeDistWtd
	TileCompoundTypeWedge   TileCompoundType = internaltile.CompoundTypeWedge
	TileCompoundTypeDiffWtd TileCompoundType = internaltile.CompoundTypeDiffWtd

	TileDiffWtdMaskType38    TileDiffWtdMaskType = internaltile.DiffWtdMaskType38
	TileDiffWtdMaskType38Inv TileDiffWtdMaskType = internaltile.DiffWtdMaskType38Inv

	TileCoeffPlaneY  TileCoeffPlaneType = internaltile.CoeffPlaneY
	TileCoeffPlaneUV TileCoeffPlaneType = internaltile.CoeffPlaneUV

	TileTransformSize4x4   TileTransformSize = internaltile.TransformSize4x4
	TileTransformSize8x8   TileTransformSize = internaltile.TransformSize8x8
	TileTransformSize16x16 TileTransformSize = internaltile.TransformSize16x16
	TileTransformSize32x32 TileTransformSize = internaltile.TransformSize32x32
	TileTransformSize64x64 TileTransformSize = internaltile.TransformSize64x64
	TileTransformSize4x8   TileTransformSize = internaltile.TransformSize4x8
	TileTransformSize8x4   TileTransformSize = internaltile.TransformSize8x4
	TileTransformSize8x16  TileTransformSize = internaltile.TransformSize8x16
	TileTransformSize16x8  TileTransformSize = internaltile.TransformSize16x8
	TileTransformSize16x32 TileTransformSize = internaltile.TransformSize16x32
	TileTransformSize32x16 TileTransformSize = internaltile.TransformSize32x16
	TileTransformSize32x64 TileTransformSize = internaltile.TransformSize32x64
	TileTransformSize64x32 TileTransformSize = internaltile.TransformSize64x32
	TileTransformSize4x16  TileTransformSize = internaltile.TransformSize4x16
	TileTransformSize16x4  TileTransformSize = internaltile.TransformSize16x4
	TileTransformSize8x32  TileTransformSize = internaltile.TransformSize8x32
	TileTransformSize32x8  TileTransformSize = internaltile.TransformSize32x8
	TileTransformSize16x64 TileTransformSize = internaltile.TransformSize16x64
	TileTransformSize64x16 TileTransformSize = internaltile.TransformSize64x16

	TileExtTXSetDCTOnly       TileExtTXSetType = internaltile.ExtTXSetDCTOnly
	TileExtTXSetDCTIDTX       TileExtTXSetType = internaltile.ExtTXSetDCTIDTX
	TileExtTXSetDTT4IDTX      TileExtTXSetType = internaltile.ExtTXSetDTT4IDTX
	TileExtTXSetDTT4IDTX1DDCT TileExtTXSetType = internaltile.ExtTXSetDTT4IDTX1DDCT
	TileExtTXSetDTT9IDTX1DDCT TileExtTXSetType = internaltile.ExtTXSetDTT9IDTX1DDCT
	TileExtTXSetAll16         TileExtTXSetType = internaltile.ExtTXSetAll16
)

// ErrTileInvalidDecodeState is returned by the tile-decode helpers when the
// supplied state is nil or otherwise inconsistent with the requested operation.
var ErrTileInvalidDecodeState = internaltile.ErrInvalidDecodeState
