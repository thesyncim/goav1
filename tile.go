package goav1

import (
	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

type LoopFilterPlane = internalloopfilter.Plane
type LoopFilterEdge = internalloopfilter.Edge

type TileBlockLevel = internaltile.BlockLevel
type TilePartition = internaltile.Partition
type TileBlockSize = internaltile.BlockSize
type TileBlockDimensions = internaltile.BlockDimensions
type TileBlockVisit = internaltile.BlockVisit
type TileReferenceFrame = internaltile.ReferenceFrame
type TileReferenceMVFrame = internaltile.ReferenceMVFrame
type TileReferenceMVEntry = internaltile.ReferenceMVEntry
type TileTemporalMotionField = internaltile.TemporalMotionField
type TileTemporalMotionEntry = internaltile.TemporalMotionEntry
type TileDecodeState = internaltile.DecodeState
type TileTransformSize = internaltile.TransformSize
type TileTransformDimensions = internaltile.TransformDimensions
type TileTransformPartitionRequest = internaltile.TransformPartitionRequest
type TileTransformTreeRequest = internaltile.TransformTreeRequest
type TileTransformTreeResult = internaltile.TransformTreeResult
type TileTransformBlock = internaltile.TransformBlock
type TileTransformBlockVisitor = internaltile.TransformBlockVisitor
type TileExtTXSetType = internaltile.ExtTXSetType

type TileRestorationSGRProjInfo = internaltile.SGRProjInfo
type TileRestorationUnit = internaltile.RestorationUnit
type TileRestorationUnitRect = internaltile.RestorationUnitRect
type TileRestorationProcessingStripe = internaltile.RestorationProcessingStripe
type TileRestorationProcessingUnit = internaltile.RestorationProcessingUnit
type TileRestorationPlaneGrid = internaltile.RestorationPlaneGrid
type TileRestorationUnitRange = internaltile.RestorationUnitRange
type TileRestorationUnitRecord = internaltile.RestorationUnitRecord
type TileRestorationFramePlan = internaltile.RestorationFramePlan
type TileRestorationStripeBoundaries = internaltile.RestorationStripeBoundaries
type TileRestorationFrameBoundaryPlane = internaltile.RestorationFrameBoundaryPlane
type TileRestorationStripeBoundaryScratchSize = internaltile.RestorationStripeBoundaryScratchSize
type TileRestorationStripeBoundaryScratch = internaltile.RestorationStripeBoundaryScratch
type TileRestorationStripeBoundaryBufferSize = internaltile.RestorationStripeBoundaryBufferSize
type TileRestorationUnitScratchSize = internaltile.RestorationUnitScratchSize
type TileRestorationUnitScratch = internaltile.RestorationUnitScratch
type TileRestorationUnitApplyResult = internaltile.RestorationUnitApplyResult
type TileRestorationUnitRecordApplyResult = internaltile.RestorationUnitRecordApplyResult
type TileRestorationPlaneApplyResult = internaltile.RestorationPlaneApplyResult
type TileRestorationFramePlane = internaltile.RestorationFramePlane
type TileRestorationFrameApplyResult = internaltile.RestorationFrameApplyResult
type TileRestorationUnitRecordBoundaryScratchSize = internaltile.RestorationUnitRecordBoundaryScratchSize
type TileRestorationUnitRecordBoundaryScratch = internaltile.RestorationUnitRecordBoundaryScratch
type TileRestorationFrameSampleScratchSize = internaltile.RestorationFrameSampleScratchSize

type RestorationWienerFilter = internalrestoration.WienerFilter
type RestorationWienerInfo = internalrestoration.WienerInfo
type RestorationSGRParams = internalrestoration.SGRParams

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

var ErrTileInvalidDecodeState = internaltile.ErrInvalidDecodeState
