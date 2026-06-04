package threading

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	decodework "github.com/thesyncim/goav1/internal/av1/work"
)

// BatchFunc processes one deterministic batch. The jobs slice is the exact
// contiguous range described by the Batch.
type BatchFunc func(batch Batch, jobs []tile.Job) error

// FrameWorkSequenceContext is the compact sequence-level context needed by
// frame-work callbacks. It avoids copying the full sequence header, including
// operating-point arrays, into every worker task.
type FrameWorkSequenceContext struct {
	Profile                    uint8
	Use128x128Superblock       bool
	SBSizeLog2                 uint8
	SBSizeMIB                  uint8
	EnableFilterIntra          bool
	EnableIntraEdgeFilter      bool
	EnableInterIntraCompound   bool
	EnableMaskedCompound       bool
	EnableWarpedMotion         bool
	EnableDualFilter           bool
	EnableOrderHint            bool
	EnableJNTComp              bool
	EnableRefFrameMVS          bool
	SeqForceScreenContentTools uint8
	SeqForceIntegerMV          uint8
	OrderHintBits              uint8
	EnableSuperRes             bool
	EnableCDEF                 bool
	EnableRestoration          bool
	ColorConfig                parser.ColorConfig
	FilmGrainParamsPresent     bool
}

// FrameWorkSequenceContextFromHeader derives worker-facing sequence context
// from a parsed sequence header without allocating.
func FrameWorkSequenceContextFromHeader(seq parser.SequenceHeader) FrameWorkSequenceContext {
	sbSizeLog2 := uint8(6)
	sbSizeMIB := uint8(16)
	if seq.Use128x128Superblock {
		sbSizeLog2 = 7
		sbSizeMIB = 32
	}
	return FrameWorkSequenceContext{
		Profile:                    seq.SeqProfile,
		Use128x128Superblock:       seq.Use128x128Superblock,
		SBSizeLog2:                 sbSizeLog2,
		SBSizeMIB:                  sbSizeMIB,
		EnableFilterIntra:          seq.EnableFilterIntra,
		EnableIntraEdgeFilter:      seq.EnableIntraEdgeFilter,
		EnableInterIntraCompound:   seq.EnableInterIntraCompound,
		EnableMaskedCompound:       seq.EnableMaskedCompound,
		EnableWarpedMotion:         seq.EnableWarpedMotion,
		EnableDualFilter:           seq.EnableDualFilter,
		EnableOrderHint:            seq.EnableOrderHint,
		EnableJNTComp:              seq.EnableJNTComp,
		EnableRefFrameMVS:          seq.EnableRefFrameMVS,
		SeqForceScreenContentTools: seq.SeqForceScreenContentTools,
		SeqForceIntegerMV:          seq.SeqForceIntegerMV,
		OrderHintBits:              seq.OrderHintBits,
		EnableSuperRes:             seq.EnableSuperRes,
		EnableCDEF:                 seq.EnableCDEF,
		EnableRestoration:          seq.EnableRestoration,
		ColorConfig:                seq.ColorConfig,
		FilmGrainParamsPresent:     seq.FilmGrainParamsPresent,
	}
}

// Valid reports whether the context was derived from a parsed sequence header.
func (c FrameWorkSequenceContext) Valid() bool {
	return c.ColorConfig.BitDepth != 0 &&
		((c.SBSizeLog2 == 6 && c.SBSizeMIB == 16) ||
			(c.SBSizeLog2 == 7 && c.SBSizeMIB == 32))
}

// FrameWorkJobRegion is the clipped reconstruction region for one scheduled
// tile job. Pixel bounds are clipped to the coded frame, while MI bounds follow
// AV1's 8-aligned MI grid.
type FrameWorkJobRegion struct {
	Tile uint16
	Row  uint8
	Col  uint8

	SBX    uint16
	SBY    uint16
	SBCols uint16
	SBRows uint16

	PixelX      uint32
	PixelY      uint32
	PixelWidth  uint32
	PixelHeight uint32

	MIColStart uint16
	MIRowStart uint16
	MIColEnd   uint16
	MIRowEnd   uint16
}

// FrameWorkPlane identifies one frame plane.
type FrameWorkPlane uint8

const (
	FrameWorkPlaneY FrameWorkPlane = iota
	FrameWorkPlaneU
	FrameWorkPlaneV
)

// FrameWorkReference identifies one AV1 inter reference frame in the resolved
// References slice. Values follow AV1's LAST_FRAME through ALTREF_FRAME order.
type FrameWorkReference uint8

const (
	FrameWorkReferenceLast FrameWorkReference = iota
	FrameWorkReferenceLast2
	FrameWorkReferenceLast3
	FrameWorkReferenceGolden
	FrameWorkReferenceBwd
	FrameWorkReferenceAltRef2
	FrameWorkReferenceAltRef
)

// FrameWorkPlaneRegion is a checked output or reference plane window. Pix
// starts at (X, Y) and spans through the final row; row starts are separated by
// Stride bytes and each row has RowBytes valid bytes.
//
// Width/Height bound the visible (coded-frame) extent that downstream MD5 /
// output consumers care about. ClipWidth/ClipHeight bound the writable extent
// (the MI-aligned region of the underlying plane buffer); they are always
// >= Width/Height. AV1 prediction and residual writes may extend past the
// visible edge into the MI-aligned region because libaom writes whole
// transform blocks regardless of where the visible boundary lands; later
// blocks read those past-visible samples as predictor neighbors. Callers
// that want to clip to the writable extent use ClipWidth/ClipHeight. Callers
// that want to know which blocks are entirely outside the coded frame use
// Width/Height (or frameWorkPlaneBlockStartsBeyondOutput). When ClipWidth
// (resp. ClipHeight) is zero, callers fall back to Width (resp. Height).
type FrameWorkPlaneRegion struct {
	Pix []byte

	Stride   uint32
	X        uint32
	Y        uint32
	Width    uint32
	Height   uint32
	RowBytes uint32
	// ClipWidth / ClipHeight are the superblock-aligned writable extent in plane
	// samples (full-transform write extent for partial bottom/right superblock
	// blocks, clamped to the allocation). Zero defaults to Width / Height (i.e.,
	// visible == aligned). ClipRowBytes is ClipWidth * BytesPerSample.
	ClipWidth    uint32
	ClipHeight   uint32
	ClipRowBytes uint32
	// ReadWidth / ReadHeight are the MI-aligned neighbor-read extent in plane
	// samples (libaom's n_top_px/n_left_px cap from mi_{cols,rows} * MI_SIZE and
	// the TX-origin reject bound). They satisfy Width <= ReadWidth <= ClipWidth.
	// Zero defaults to ClipWidth / ClipHeight (then to Width / Height).
	ReadWidth  uint32
	ReadHeight uint32

	Plane          FrameWorkPlane
	BytesPerSample uint8
}

// FrameWorkLoopRestorationPlan carries the frame-level post-filter decisions
// used around CDEF, superres, and loop restoration.
type FrameWorkLoopRestorationPlan struct {
	Restoration tile.RestorationFramePlan

	DoCDEF            bool
	DoSuperRes        bool
	DoLoopRestoration bool

	OptimizedLoopRestoration bool
	SaveBoundariesBeforeCDEF bool
	SaveBoundariesAfterCDEF  bool
}

// FrameWorkFrameContext is the parsed frame context supplied to frame-work
// tile batches. It is copied from the current decoder event so reconstruction
// callbacks can map tile jobs to frame geometry and frame-level syntax without
// reparsing payload headers.
type FrameWorkFrameContext struct {
	Sequence            FrameWorkSequenceContext
	FrameHeader         parser.FrameHeaderPrefix
	FrameSize           parser.FrameSize
	TileInfo            parser.TileInfo
	Quantization        parser.QuantizationParams
	Segmentation        parser.SegmentationParams
	Delta               parser.DeltaParams
	LoopFilter          parser.LoopFilterParams
	CDEF                parser.CDEFParams
	Restoration         parser.RestorationParams
	TransformRef        parser.TransformReferenceParams
	SkipMode            parser.SkipModeParams
	FrameMode           parser.FrameModeParams
	GlobalMotion        parser.GlobalMotionParams
	FilmGrain           parser.FilmGrainParams
	ReferenceOrderHints [parser.InterRefsPerFrame]uint8
}

// FrameWorkBatch is the decoder context supplied to one frame-work tile batch.
// Payload, References, CurrentMVFrame, TemporalMVs, Jobs, and CDF pointers
// alias caller-owned storage and are valid for the callback invocation.
type FrameWorkBatch struct {
	Step       decodework.FrameStep
	Output     *frame.Frame
	Payload    []byte
	References []*frame.Frame
	// CurrentMVFrame receives libaom-style MV_REF side data while the block
	// loop decodes prediction modes. It is optional until temporal projection
	// consumers are fully wired.
	CurrentMVFrame *tile.ReferenceMVFrame
	// TemporalMVs carries projected ref-frame-MVS samples consumed by
	// BuildReferenceMVStack when use_ref_frame_mvs is enabled.
	TemporalMVs *tile.TemporalMotionField
	// ReferenceMVs carries resolved per-reference MV_REF metadata used to set up
	// TemporalMVs before block prediction.
	ReferenceMVs [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	// CDEFIndexMap receives decoded cdef_idx values while tile residuals run.
	// It is optional; callers can still pass a per-request map.
	CDEFIndexMap *FrameWorkCDEFIndexMap
	// LoopFilterMap receives block-local loop-filter metadata while tile
	// residuals run. It is optional; callers can still pass a per-request map.
	LoopFilterMap *FrameWorkLoopFilterMap
	// RestorationFrameBuffers receives decoded loop-restoration unit records
	// while tile residuals run. It is optional; callers can still pass a
	// per-request restoration request.
	RestorationFrameBuffers *FrameWorkRestorationFrameBuffers
	FrameWorkFrameContext
	// InitialTileResidualCDFs is the frame context selected by
	// primary_ref_frame for this tile group. RetainedTileResidualCDFs receives
	// the context_update_tile_id tile's adapted state when callbacks opt in via
	// RetainTileResidualCDFStorage.
	InitialTileResidualCDFs       *FrameWorkTileResidualCDFStorage
	RetainedTileResidualCDFs      *FrameWorkTileResidualCDFStorage
	RetainedTileResidualCDFsValid *bool
	Batch                         Batch
	DisableCDFUpdate              bool

	// WavefrontWorkers is the number of idle pool goroutines that the per-tile
	// reconstruction may employ for an SB-row wavefront when this batch is the
	// only batch dispatched for the frame (a single-tile frame leaves the other
	// W-1 pool lanes idle). It is zero on the single-worker fast path and on
	// multi-batch (multi-tile) frames, where every lane is already busy, so the
	// deferred wavefront stays off and the fused single-thread path runs. The
	// pool sets it from WorkerCount when it dispatches exactly one batch.
	WavefrontWorkers uint16
	Jobs             []tile.Job

	// geomCache optionally memoizes the job-constant JobRegion and per-plane
	// JobOutputPlane windows for a single job index. It is caller-owned scratch
	// installed by the per-job residual loop (which decodes thousands of
	// transform blocks that all share one index) so the geometry is computed
	// once per job/plane instead of once per coefficient block. When nil, the
	// geometry methods recompute on every call, preserving the stateless
	// value-receiver contract for all other callers. The cache holds geometry
	// for exactly one index at a time and is keyed/validated by index, so a
	// worker that reuses one cache across successive jobs stays byte-exact.
	geomCache *frameWorkJobGeometryCache
}

// frameWorkJobGeometryCache memoizes the job-constant reconstruction geometry
// for one job index. JobRegion and JobOutputPlane depend only on the job index
// (and, for the plane window, the plane); within a single tile job they are
// invariant across every decoded transform block. Caching them removes the
// per-block recompute observed in profiling while producing bit-identical
// results (the cache returns exactly what the original computation would).
type frameWorkJobGeometryCache struct {
	validMask   uint8
	regionIndex uint16
	region      FrameWorkJobRegion

	planeIndex [3]uint16
	plane      [3]FrameWorkPlaneRegion
}

const (
	frameWorkJobGeometryRegionValid = uint8(1 << iota)
	frameWorkJobGeometryPlaneYValid
	frameWorkJobGeometryPlaneUValid
	frameWorkJobGeometryPlaneVValid

	frameWorkJobGeometryPlanesValid = frameWorkJobGeometryPlaneYValid |
		frameWorkJobGeometryPlaneUValid |
		frameWorkJobGeometryPlaneVValid

	frameWorkMaxCacheIndex = 1<<16 - 1
)

// reset invalidates every cached entry. The per-job loop calls this before a
// job runs so a reused cache never serves stale geometry for a new index.
func (c *frameWorkJobGeometryCache) reset() {
	c.validMask = 0
}

// Surface returns the frame-pool surface that this batch reconstructs into.
func (b *FrameWorkBatch) Surface() (int, error) {
	switch b.Step.Kind {
	case decodework.FrameStepBegin:
		if b.Step.Begin.Surface < 0 {
			return -1, ErrInvalidBatch
		}
		return b.Step.Begin.Surface, nil
	case decodework.FrameStepTile:
		if b.Step.Tile.Surface < 0 {
			return -1, ErrInvalidBatch
		}
		return b.Step.Tile.Surface, nil
	default:
		return -1, ErrInvalidBatch
	}
}

// JobPayload returns the exact tile payload bytes for Jobs[index]. The
// returned slice aliases Payload and is valid only for the callback invocation.
func (b *FrameWorkBatch) JobPayload(index int) ([]byte, error) {
	if index < 0 || index >= len(b.Jobs) {
		return nil, ErrInvalidBatch
	}
	return b.Jobs[index].Payload(b.Payload)
}

// JobEntropyReader returns an entropy reader over Jobs[index]'s exact tile
// payload. The reader inherits the frame-level CDF update mode carried by b.
func (b *FrameWorkBatch) JobEntropyReader(index int) (entropy.Reader, error) {
	if index < 0 || index >= len(b.Jobs) {
		return entropy.Reader{}, ErrInvalidBatch
	}
	return tile.NewEntropyReader(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
	})
}

// JobDecodeState initializes state for Jobs[index]'s exact tile payload. The
// state records whether this job's adapted entropy context should be retained.
func (b *FrameWorkBatch) JobDecodeState(index int, state *tile.DecodeState) error {
	if index < 0 || index >= len(b.Jobs) || state == nil {
		return ErrInvalidBatch
	}
	return state.Reset(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
		BaseQIdx:         b.Quantization.BaseQIdx,
	})
}

// JobRegion returns the clipped reconstruction region for Jobs[index]. When a
// job-geometry cache is installed (by the residual loop), the result is
// memoized per index so the pure integer recompute is amortized across the
// many transform blocks of one job.
func (b *FrameWorkBatch) JobRegion(index int) (FrameWorkJobRegion, error) {
	cacheIndex, cacheIndexOK := frameWorkJobCacheIndex(index)
	if c := b.geomCache; c != nil && cacheIndexOK &&
		c.validMask&frameWorkJobGeometryRegionValid != 0 && c.regionIndex == cacheIndex {
		return c.region, nil
	}
	region, err := b.computeJobRegion(index)
	if err != nil {
		return region, err
	}
	if c := b.geomCache; c != nil && cacheIndexOK {
		// A new index invalidates any plane windows cached for the old one.
		if c.validMask&frameWorkJobGeometryRegionValid == 0 || c.regionIndex != cacheIndex {
			c.validMask &^= frameWorkJobGeometryPlanesValid
		}
		c.region = region
		c.regionIndex = cacheIndex
		c.validMask |= frameWorkJobGeometryRegionValid
	}
	return region, nil
}

func (b *FrameWorkBatch) computeJobRegion(index int) (FrameWorkJobRegion, error) {
	if index < 0 || index >= len(b.Jobs) || !b.Sequence.Valid() ||
		b.FrameSize.CodedWidth == 0 || b.FrameSize.Height == 0 {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}

	job := b.Jobs[index]
	if job.SBCols == 0 || job.SBRows == 0 {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}

	sbXEnd := uint32(job.SBX) + uint32(job.SBCols)
	sbYEnd := uint32(job.SBY) + uint32(job.SBRows)
	pixelX := uint32(job.SBX) << b.Sequence.SBSizeLog2
	pixelY := uint32(job.SBY) << b.Sequence.SBSizeLog2
	pixelXEnd := sbXEnd << b.Sequence.SBSizeLog2
	pixelYEnd := sbYEnd << b.Sequence.SBSizeLog2
	if pixelX >= b.FrameSize.CodedWidth || pixelY >= b.FrameSize.Height {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}
	if pixelXEnd > b.FrameSize.CodedWidth {
		pixelXEnd = b.FrameSize.CodedWidth
	}
	if pixelYEnd > b.FrameSize.Height {
		pixelYEnd = b.FrameSize.Height
	}
	if pixelXEnd <= pixelX || pixelYEnd <= pixelY {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}

	miCols, ok := frameWorkMIExtent(b.FrameSize.CodedWidth)
	if !ok {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}
	miRows, ok := frameWorkMIExtent(b.FrameSize.Height)
	if !ok {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}
	miColStart := uint32(job.SBX) * uint32(b.Sequence.SBSizeMIB)
	miRowStart := uint32(job.SBY) * uint32(b.Sequence.SBSizeMIB)
	miColEnd := sbXEnd * uint32(b.Sequence.SBSizeMIB)
	miRowEnd := sbYEnd * uint32(b.Sequence.SBSizeMIB)
	if miColStart >= miCols || miRowStart >= miRows {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}
	if miColEnd > miCols {
		miColEnd = miCols
	}
	if miRowEnd > miRows {
		miRowEnd = miRows
	}
	if miColEnd <= miColStart || miRowEnd <= miRowStart {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}
	if miColEnd > uint32(^uint16(0)) || miRowEnd > uint32(^uint16(0)) {
		return FrameWorkJobRegion{}, ErrInvalidBatch
	}

	return FrameWorkJobRegion{
		Tile:        job.Tile,
		Row:         job.Row,
		Col:         job.Col,
		SBX:         job.SBX,
		SBY:         job.SBY,
		SBCols:      job.SBCols,
		SBRows:      job.SBRows,
		PixelX:      pixelX,
		PixelY:      pixelY,
		PixelWidth:  pixelXEnd - pixelX,
		PixelHeight: pixelYEnd - pixelY,
		MIColStart:  uint16(miColStart),
		MIRowStart:  uint16(miRowStart),
		MIColEnd:    uint16(miColEnd),
		MIRowEnd:    uint16(miRowEnd),
	}, nil
}

// JobBlockDeltaContext returns the AV1 block-delta context for an absolute MI
// coordinate inside Jobs[index]'s region.
func (b *FrameWorkBatch) JobBlockDeltaContext(index int, miCol uint32, miRow uint32, fullSuperblock bool, skipTransform bool) (tile.BlockDeltaContext, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.BlockDeltaContext{}, err
	}
	if miCol < uint32(region.MIColStart) || miCol >= uint32(region.MIColEnd) ||
		miRow < uint32(region.MIRowStart) || miRow >= uint32(region.MIRowEnd) {
		return tile.BlockDeltaContext{}, ErrInvalidBatch
	}
	return tile.BlockDeltaContext{
		MICol:          uint16(miCol),
		MIRow:          uint16(miRow),
		SBSizeMIB:      b.Sequence.SBSizeMIB,
		FullSuperblock: fullSuperblock,
		SkipTransform:  skipTransform,
		Monochrome:     b.Sequence.ColorConfig.MonoChrome,
	}, nil
}

// RestorationPlaneGrid returns the loop-restoration unit grid for one plane of
// this batch's frame context.
func (b *FrameWorkBatch) RestorationPlaneGrid(plane FrameWorkPlane) (tile.RestorationPlaneGrid, error) {
	if plane > FrameWorkPlaneV || !b.Sequence.Valid() {
		return tile.RestorationPlaneGrid{}, ErrInvalidBatch
	}
	return tile.BuildRestorationPlaneGrid(b.Restoration, b.FrameSize, b.Sequence.ColorConfig, int(plane))
}

// RestorationFramePlan returns the frame-level loop-restoration allocation plan
// for this batch's frame context.
func (b *FrameWorkBatch) RestorationFramePlan() (tile.RestorationFramePlan, error) {
	if !b.Sequence.Valid() {
		return tile.RestorationFramePlan{}, ErrInvalidBatch
	}
	return tile.BuildRestorationFramePlan(b.Restoration, b.FrameSize, b.Sequence.ColorConfig)
}

// LoopRestorationPlan ports libaom's CDEF/superres/restoration scheduling
// decision for the frame. skipLoopFilter is the decoder-level skip flag from
// libaom's AV1Decoder.
func (b *FrameWorkBatch) LoopRestorationPlan(skipLoopFilter bool) (FrameWorkLoopRestorationPlan, error) {
	restoration, err := b.RestorationFramePlan()
	if err != nil {
		return FrameWorkLoopRestorationPlan{}, err
	}
	doCDEF := !skipLoopFilter && frameWorkCDEFActive(b.CDEF)
	doSuperRes := b.FrameSize.SuperResEnabled
	optimized := !doCDEF && !doSuperRes
	doLoopRestoration := restoration.Active
	return FrameWorkLoopRestorationPlan{
		Restoration:              restoration,
		DoCDEF:                   doCDEF,
		DoSuperRes:               doSuperRes,
		DoLoopRestoration:        doLoopRestoration,
		OptimizedLoopRestoration: optimized,
		SaveBoundariesBeforeCDEF: doLoopRestoration && !optimized,
		SaveBoundariesAfterCDEF:  doLoopRestoration && !optimized,
	}, nil
}

// JobRestorationUnitRange returns the restoration units whose top-left corners
// are decoded while processing the superblock at sbX,sbY inside Jobs[index].
func (b *FrameWorkBatch) JobRestorationUnitRange(index int, plane FrameWorkPlane, sbX uint16, sbY uint16) (tile.RestorationUnitRange, bool, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.RestorationUnitRange{}, false, err
	}
	if uint32(sbX) < uint32(region.SBX) || uint32(sbY) < uint32(region.SBY) ||
		uint32(sbX) >= uint32(region.SBX)+uint32(region.SBCols) ||
		uint32(sbY) >= uint32(region.SBY)+uint32(region.SBRows) {
		return tile.RestorationUnitRange{}, false, ErrInvalidBatch
	}
	grid, err := b.RestorationPlaneGrid(plane)
	if err != nil {
		return tile.RestorationUnitRange{}, false, err
	}
	miCol := uint32(sbX) * uint32(b.Sequence.SBSizeMIB)
	miRow := uint32(sbY) * uint32(b.Sequence.SBSizeMIB)
	return grid.UnitsInSuperblock(miCol, miRow, b.Sequence.SBSizeMIB)
}

// JobOutputPlane returns the clipped output-plane window for Jobs[index]. The
// visible Width/Height are clamped to the coded-frame extent; the writable
// ClipWidth/ClipHeight extend to the MI-aligned region so AV1 prediction and
// residual writes can land past the visible edge (libaom emits whole
// transform blocks regardless of the visible boundary, and later blocks read
// those past-visible samples as predictor neighbors).
func (b *FrameWorkBatch) JobOutputPlane(index int, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
	cacheIndex, cacheIndexOK := frameWorkJobCacheIndex(index)
	if c := b.geomCache; c != nil && plane <= FrameWorkPlaneV && cacheIndexOK {
		planeMask := frameWorkJobGeometryPlaneMask(plane)
		if c.validMask&planeMask != 0 && c.planeIndex[plane] == cacheIndex {
			return c.plane[plane], nil
		}
	}
	window, err := b.computeJobOutputPlane(index, plane)
	if err != nil {
		return window, err
	}
	if c := b.geomCache; c != nil && plane <= FrameWorkPlaneV && cacheIndexOK {
		planeMask := frameWorkJobGeometryPlaneMask(plane)
		c.plane[plane] = window
		c.planeIndex[plane] = cacheIndex
		c.validMask |= planeMask
	}
	return window, nil
}

func frameWorkJobGeometryPlaneMask(plane FrameWorkPlane) uint8 {
	return uint8(1) << (uint8(plane) + 1)
}

func frameWorkJobCacheIndex(index int) (uint16, bool) {
	if index < 0 || index > frameWorkMaxCacheIndex {
		return 0, false
	}
	return uint16(index), true
}

func (b *FrameWorkBatch) computeJobOutputPlane(index int, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return FrameWorkPlaneRegion{}, err
	}
	if b.Output == nil {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	outputPlane, subsamplingX, subsamplingY, ok := frameWorkFramePlane(b.Output, plane)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	bytesPerSample := b.Output.Layout.BytesPerSample
	if bytesPerSample <= 0 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}

	x0, x1 := frameWorkPlaneRange(region.PixelX, region.PixelX+region.PixelWidth, subsamplingX)
	y0, y1 := frameWorkPlaneRange(region.PixelY, region.PixelY+region.PixelHeight, subsamplingY)

	// Compute two trailing extents.
	//
	// ReadWidth/ReadHeight is the MI-aligned extent (region.MIColEnd/MIRowEnd in
	// 4x4 luma MI units, capped to mi_cols/mi_rows). It bounds intra-predictor
	// neighbor reads (libaom's n_top_px/n_left_px cap, derived from
	// xd->mi_params.mi_{cols,rows} * MI_SIZE) and the TX-origin reject (a
	// transform is skipped only when its origin reaches the MI grid edge).
	//
	// ClipWidth/ClipHeight is the writable extent. libaom's
	// av1_inverse_transform_block writes the WHOLE tx_size and cfl_store_tx
	// subsamples the full reconstructed transform with no boundary clamp; a
	// transform whose origin is inside the MI grid but whose extent crosses the
	// cropped edge spills into the YV12 border. Since a transform never crosses
	// a superblock boundary, that spill stays inside this job's superblock
	// extent, so we expand the writable trailing edge to the superblock pixel
	// extent. frameWorkPlaneWindow then clamps to the underlying allocation
	// (which RequiredSize sizes to the superblock-aligned extent).
	readLumaX := uint32(region.MIColEnd) * 4
	readLumaY := uint32(region.MIRowEnd) * 4
	clipLumaX := uint32(region.SBX+region.SBCols) << b.Sequence.SBSizeLog2
	clipLumaY := uint32(region.SBY+region.SBRows) << b.Sequence.SBSizeLog2
	xRead0, xRead1 := frameWorkPlaneRange(uint32(region.MIColStart)*4, readLumaX, subsamplingX)
	yRead0, yRead1 := frameWorkPlaneRange(uint32(region.MIRowStart)*4, readLumaY, subsamplingY)
	xAligned0, xAligned1 := frameWorkPlaneRange(uint32(region.MIColStart)*4, clipLumaX, subsamplingX)
	yAligned0, yAligned1 := frameWorkPlaneRange(uint32(region.MIRowStart)*4, clipLumaY, subsamplingY)
	// The window's pixel-space origin must remain aligned with the visible
	// origin; only the trailing edge expands.
	if xAligned0 != x0 || yAligned0 != y0 || xRead0 != x0 || yRead0 != y0 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	if xRead1 < x1 {
		xRead1 = x1
	}
	if yRead1 < y1 {
		yRead1 = y1
	}
	if xAligned1 < xRead1 {
		xAligned1 = xRead1
	}
	if yAligned1 < yRead1 {
		yAligned1 = yRead1
	}
	return frameWorkPlaneWindow(plane, outputPlane, bytesPerSample, x0, y0, x1, y1, xRead1, yRead1, xAligned1, yAligned1)
}

// ReferenceFrame returns the resolved frame for one AV1 inter-reference slot.
func (b *FrameWorkBatch) ReferenceFrame(reference FrameWorkReference) (*frame.Frame, error) {
	index := int(reference)
	if index >= parser.InterRefsPerFrame || index >= len(b.References) || b.References[index] == nil {
		return nil, ErrInvalidBatch
	}
	return b.References[index], nil
}

// ReferencePlane returns the full plane from one resolved reference frame.
func (b *FrameWorkBatch) ReferencePlane(reference FrameWorkReference, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
	refFrame, err := b.ReferenceFrame(reference)
	if err != nil {
		return FrameWorkPlaneRegion{}, err
	}
	refPlane, _, _, ok := frameWorkFramePlane(refFrame, plane)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	bytesPerSample := refFrame.Layout.BytesPerSample
	if refPlane.Width <= 0 ||
		refPlane.Height <= 0 ||
		uint64(refPlane.Width) > uint64(^uint32(0)) ||
		uint64(refPlane.Height) > uint64(^uint32(0)) {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	w := uint32(refPlane.Width)
	h := uint32(refPlane.Height)
	return frameWorkPlaneWindow(plane, refPlane, bytesPerSample, 0, 0, w, h, w, h, w, h)
}

// JobReferencePlane returns the reference-plane window matching Jobs[index]'s
// clipped output region. It is valid for same-size reference sampling; scaled
// references require motion-compensation code to map coordinates explicitly.
func (b *FrameWorkBatch) JobReferencePlane(index int, reference FrameWorkReference, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
	return b.JobReferencePlaneWindow(index, reference, plane, 0, 0)
}

// JobReferencePlaneWindow returns the reference-plane window matching
// Jobs[index]'s clipped output region and expanded by marginX/marginY plane
// samples. The window is clipped to the resolved reference frame plane.
func (b *FrameWorkBatch) JobReferencePlaneWindow(index int, reference FrameWorkReference, plane FrameWorkPlane, marginX uint32, marginY uint32) (FrameWorkPlaneRegion, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return FrameWorkPlaneRegion{}, err
	}
	refFrame, err := b.ReferenceFrame(reference)
	if err != nil {
		return FrameWorkPlaneRegion{}, err
	}
	refPlane, subsamplingX, subsamplingY, ok := frameWorkFramePlane(refFrame, plane)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	bytesPerSample := refFrame.Layout.BytesPerSample
	if bytesPerSample <= 0 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}

	x0, x1 := frameWorkPlaneRange(region.PixelX, region.PixelX+region.PixelWidth, subsamplingX)
	y0, y1 := frameWorkPlaneRange(region.PixelY, region.PixelY+region.PixelHeight, subsamplingY)
	x0, x1, ok = frameWorkExpandPlaneRange(x0, x1, refPlane.Width, marginX)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	y0, y1, ok = frameWorkExpandPlaneRange(y0, y1, refPlane.Height, marginY)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	return frameWorkPlaneWindow(plane, refPlane, bytesPerSample, x0, y0, x1, y1, x1, y1, x1, y1)
}

// JobUpdatesFrameContext reports whether Jobs[index] is the designated tile
// whose adapted entropy state should refresh the frame context.
func (b *FrameWorkBatch) JobUpdatesFrameContext(index int) (bool, error) {
	if index < 0 || index >= len(b.Jobs) {
		return false, ErrInvalidBatch
	}
	return b.Jobs[index].UpdatesFrameContext, nil
}

// ValidatePayloads checks that every job in the batch names a byte range inside
// Payload.
func (b *FrameWorkBatch) ValidatePayloads() error {
	return tile.ValidatePayloads(b.Payload, b.Jobs)
}

func frameWorkMIExtent(pixels uint32) (uint32, bool) {
	if pixels > ^uint32(0)-7 {
		return 0, false
	}
	return ((pixels + 7) >> 3) << 1, true
}

func frameWorkCDEFActive(cdef parser.CDEFParams) bool {
	return cdef.Bits != 0 || cdef.YStrength[0] != 0 || cdef.UVStrength[0] != 0
}

func frameWorkFramePlane(output *frame.Frame, plane FrameWorkPlane) (frame.Plane, bool, bool, bool) {
	switch plane {
	case FrameWorkPlaneY:
		return output.Y, false, false, true
	case FrameWorkPlaneU:
		if output.Format.MonoChrome {
			return frame.Plane{}, false, false, false
		}
		return output.U, output.Format.SubsamplingX, output.Format.SubsamplingY, true
	case FrameWorkPlaneV:
		if output.Format.MonoChrome {
			return frame.Plane{}, false, false, false
		}
		return output.V, output.Format.SubsamplingX, output.Format.SubsamplingY, true
	default:
		return frame.Plane{}, false, false, false
	}
}

// frameWorkPlaneRange projects a luma [start, end) pixel range onto its
// chroma extent when the axis is sub-sampled (ss=1). It implements libaom's
// chroma-extent convention: floor on the start, ceil on the end. The end
// formula (end >> 1) + (end & 1) is the expanded form of libaom's
// ROUND_POWER_OF_TWO(end, ss) for ss=1, i.e. (end + 1) >> 1 — they produce
// identical results for every uint32 input. Ceil rounding on the end is
// REQUIRED at frame edges where CodedWidth/Height can be odd (the partially
// covered last luma column contributes a chroma sample to the visible/MD5
// extent). Tile/SB boundaries are always MI-aligned (multiples of MI_SIZE=4
// luma => even), so the odd-end branch only fires at the right/bottom frame
// edge; floor and ceil agree for every interior boundary, which keeps two
// abutting MI-aligned jobs from overlapping on the chroma column. See
// TestFrameWorkPlaneRangeChromaSubsampleMatchesLibaomCeil and
// TestFrameWorkPlaneRangeAdjacentJobsNoChromaOverlap for the pinned
// behaviour.
func frameWorkPlaneRange(start uint32, end uint32, subsampled bool) (uint32, uint32) {
	if !subsampled {
		return start, end
	}
	return start >> 1, (end >> 1) + (end & 1)
}

func frameWorkExpandPlaneRange(start uint32, end uint32, limit int, margin uint32) (uint32, uint32, bool) {
	if limit <= 0 || uint64(limit) > uint64(^uint32(0)) || end <= start {
		return 0, 0, false
	}
	max := uint32(limit)
	if start >= max {
		return 0, 0, false
	}
	if end > max {
		end = max
	}
	if margin > start {
		start = 0
	} else {
		start -= margin
	}
	if margin > max-end {
		end = max
	} else {
		end += margin
	}
	return start, end, end > start
}

// frameWorkPlaneWindow builds a window over a plane. x0/x1/y0/y1 bound the
// visible (coded-frame) extent that consumers MD5 and emit. xRead1/yRead1 bound
// the MI-aligned neighbor-read extent (>= x1/y1); xAligned1/yAligned1 bound the
// superblock-aligned writable extent (>= xRead1/yRead1). The window's Pix slice
// covers samples through (yAligned1-1, xAligned1-1) so prediction and residual
// writes can land in the past-visible padding without running off the
// underlying plane buffer.
func frameWorkPlaneWindow(which FrameWorkPlane, plane frame.Plane, bytesPerSample int, x0 uint32, y0 uint32, x1 uint32, y1 uint32, xRead1 uint32, yRead1 uint32, xAligned1 uint32, yAligned1 uint32) (FrameWorkPlaneRegion, error) {
	if bytesPerSample <= 0 ||
		bytesPerSample > int(^uint8(0)) ||
		plane.Stride <= 0 ||
		uint64(plane.Stride) > uint64(^uint32(0)) ||
		plane.Width <= 0 ||
		plane.Height <= 0 ||
		uint64(plane.Width) > uint64(^uint32(0)) ||
		uint64(plane.Height) > uint64(^uint32(0)) ||
		x1 <= x0 ||
		y1 <= y0 ||
		x1 > uint32(plane.Width) ||
		y1 > uint32(plane.Height) ||
		xRead1 < x1 ||
		yRead1 < y1 ||
		xAligned1 < xRead1 ||
		yAligned1 < yRead1 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}

	// Clamp the aligned trailing edge to the underlying plane buffer extent.
	// libaom allocates the YV12 buffer at the MI-aligned dimensions, so the
	// byte buffer spans more than the cropped plane.Width/plane.Height: the
	// row count is len(Pix)/Stride and each row holds Stride/BytesPerSample
	// samples. The MI-aligned write extent (xAligned1/yAligned1) lands inside
	// that allocation; clamp to it (not the cropped plane.Width/plane.Height)
	// so partial bottom/right superblocks reconstruct into the padding the
	// way libaom does.
	strideSamples := plane.Stride / bytesPerSample
	if strideSamples <= 0 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	allocRows := max(len(plane.Pix)/plane.Stride, plane.Height)
	if xAligned1 > uint32(strideSamples) {
		xAligned1 = uint32(strideSamples)
	}
	if yAligned1 > uint32(allocRows) {
		yAligned1 = uint32(allocRows)
	}
	// The neighbor-read extent never exceeds the writable extent; clamp it down
	// if the allocation clamp pulled the writable edge in (the read extent is
	// MI-aligned and always lands inside the SB-aligned allocation, but keep the
	// invariant Width <= ReadWidth <= ClipWidth defensively).
	if xRead1 > xAligned1 {
		xRead1 = xAligned1
	}
	if yRead1 > yAligned1 {
		yRead1 = yAligned1
	}
	if xAligned1 < x1 || yAligned1 < y1 || xRead1 < x1 || yRead1 < y1 {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}

	x := int(x0)
	y := int(y0)
	width := int(x1 - x0)
	clipWidth := int(xAligned1 - x0)
	clipHeight := int(yAligned1 - y0)
	readWidth := int(xRead1 - x0)
	readHeight := int(yRead1 - y0)
	rowBytes, ok := frameWorkCheckedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	clipRowBytes, ok := frameWorkCheckedMul(clipWidth, bytesPerSample)
	if !ok || clipRowBytes <= 0 || clipRowBytes > plane.Stride {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	rowOffset, ok := frameWorkCheckedMul(y, plane.Stride)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	colOffset, ok := frameWorkCheckedMul(x, bytesPerSample)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	offset, ok := frameWorkCheckedAdd(rowOffset, colOffset)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	// Slice Pix to cover (clipHeight-1) rows of stride plus the final row's
	// clipRowBytes: this is the smallest slice that lets prediction write the
	// aligned trailing samples while still allowing reads of the same range
	// from past-visible neighbors.
	lastRowOffset, ok := frameWorkCheckedMul(clipHeight-1, plane.Stride)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	windowLen, ok := frameWorkCheckedAdd(lastRowOffset, clipRowBytes)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	end, ok := frameWorkCheckedAdd(offset, windowLen)
	if !ok || offset < 0 || end > len(plane.Pix) {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	return FrameWorkPlaneRegion{
		Pix:            plane.Pix[offset:end],
		Stride:         uint32(plane.Stride),
		X:              x0,
		Y:              y0,
		Width:          x1 - x0,
		Height:         y1 - y0,
		RowBytes:       uint32(rowBytes),
		ClipWidth:      uint32(clipWidth),
		ClipHeight:     uint32(clipHeight),
		ClipRowBytes:   uint32(clipRowBytes),
		ReadWidth:      uint32(readWidth),
		ReadHeight:     uint32(readHeight),
		Plane:          which,
		BytesPerSample: uint8(bytesPerSample),
	}, nil
}

func frameWorkCheckedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func frameWorkCheckedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

// FrameWorkBatchFunc processes one deterministic frame-work tile batch.
type FrameWorkBatchFunc func(FrameWorkBatch) error

// FrameWorkBatchRunner processes one deterministic frame-work tile batch
// without requiring callers to construct a callback adapter.
type FrameWorkBatchRunner interface {
	Run(FrameWorkBatch) error
}

// Pool is a reusable bounded worker pool for frame/tile work.
type Pool struct {
	mu      sync.Mutex
	workers []poolWorker
	done    chan workerResult
	closed  bool
}

type poolWorker struct {
	tasks chan poolTask
}

type poolTask struct {
	fn          BatchFunc
	frameFn     FrameWorkBatchFunc
	frameRunner FrameWorkBatchRunner
	frameBatch  FrameWorkBatch
	batch       Batch
	jobs        []tile.Job
}

type workerResult struct {
	err error
}

// NewPool starts a reusable bounded worker pool. Pool creation is a cold-path
// operation; Execute is the reusable hot path.
func NewPool(workers int) (*Pool, error) {
	if workers <= 0 || workers > int(^uint16(0)) {
		return nil, ErrInvalidWorkerCount
	}
	p := &Pool{
		workers: make([]poolWorker, workers),
		done:    make(chan workerResult, workers),
	}
	for i := range workers {
		ch := make(chan poolTask)
		p.workers[i].tasks = ch
		go poolWorkerLoop(ch, p.done)
	}
	return p, nil
}

func (p *Pool) WorkerCount() int {
	if p == nil {
		return 0
	}
	return len(p.workers)
}

// Execute dispatches batches to their assigned worker lanes and waits for all
// lanes to finish. It does not allocate after the pool has been created.
func (p *Pool) Execute(batches []Batch, jobs []tile.Job, fn BatchFunc) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if fn == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	// Single-worker fast path: run batches synchronously on the calling
	// goroutine. This avoids the channel/condvar handoff (task send + result
	// recv = 2 wakeups per batch) that dominates the profile when workers==1.
	// Results are identical to dispatching to the lone worker goroutine, which
	// would itself process the batches sequentially.
	if len(p.workers) == 1 {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return ErrPoolClosed
		}
		p.mu.Unlock()
		var firstErr error
		for i := range batches {
			batch := batches[i]
			batchJobs, err := batch.jobSlice(jobs)
			if err == nil {
				err = fn(batch, batchJobs)
			}
			if firstErr == nil && err != nil {
				firstErr = err
			}
		}
		return firstErr
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.jobSlice(jobs)
		if err != nil {
			p.mu.Unlock()
			return err
		}
		p.workers[batch.Worker].tasks <- poolTask{
			fn:    fn,
			batch: batch,
			jobs:  batchJobs,
		}
	}

	var firstErr error
	for range batches {
		result := <-p.done
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	p.mu.Unlock()
	return firstErr
}

// ExecuteFrameWork dispatches frame-work batches with decoder context. It
// follows Execute's validation and synchronization rules while copying base
// context into each worker task.
func (p *Pool) ExecuteFrameWork(batches []Batch, jobs []tile.Job, base FrameWorkBatch, fn FrameWorkBatchFunc) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if fn == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	// Single-worker fast path: run batches inline (see Execute).
	if len(p.workers) == 1 {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return ErrPoolClosed
		}
		p.mu.Unlock()
		var firstErr error
		for i := range batches {
			batch := batches[i]
			batchJobs, err := batch.jobSlice(jobs)
			if err != nil {
				return err
			}
			ctx := base
			ctx.Batch = batch
			ctx.Jobs = batchJobs
			if err := fn(ctx); firstErr == nil && err != nil {
				firstErr = err
			}
		}
		return firstErr
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	wavefrontWorkers := uint16(0)
	if len(batches) == 1 {
		wavefrontWorkers = uint16(len(p.workers))
	}

	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.jobSlice(jobs)
		if err != nil {
			p.mu.Unlock()
			return err
		}
		ctx := base
		ctx.WavefrontWorkers = wavefrontWorkers
		p.workers[batch.Worker].tasks <- poolTask{
			frameFn:    fn,
			frameBatch: ctx,
			batch:      batch,
			jobs:       batchJobs,
		}
	}

	var firstErr error
	for range batches {
		result := <-p.done
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	p.mu.Unlock()
	return firstErr
}

// ExecuteFrameWorkRunner dispatches frame-work batches with decoder context
// through runner. It follows ExecuteFrameWork's validation and synchronization
// rules without constructing a callback adapter.
func (p *Pool) ExecuteFrameWorkRunner(batches []Batch, jobs []tile.Job, base FrameWorkBatch, runner FrameWorkBatchRunner) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if runner == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	// Single-worker fast path: run batches inline (see Execute).
	if len(p.workers) == 1 {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return ErrPoolClosed
		}
		p.mu.Unlock()
		var firstErr error
		for i := range batches {
			batch := batches[i]
			batchJobs, err := batch.jobSlice(jobs)
			if err != nil {
				return err
			}
			ctx := base
			ctx.Batch = batch
			ctx.Jobs = batchJobs
			if err := runner.Run(ctx); firstErr == nil && err != nil {
				firstErr = err
			}
		}
		return firstErr
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	// A single batch on a multi-worker pool means a single-tile frame: only one
	// lane runs the per-tile decode while the other W-1 lanes sit idle. Offer
	// those idle lanes to the per-tile reconstruction so it can run an SB-row
	// wavefront. With more than one batch every lane is busy, so leave the count
	// zero and let each batch run the fused single-thread path.
	wavefrontWorkers := uint16(0)
	if len(batches) == 1 {
		wavefrontWorkers = uint16(len(p.workers))
	}

	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.jobSlice(jobs)
		if err != nil {
			p.mu.Unlock()
			return err
		}
		ctx := base
		ctx.WavefrontWorkers = wavefrontWorkers
		p.workers[batch.Worker].tasks <- poolTask{
			frameRunner: runner,
			frameBatch:  ctx,
			batch:       batch,
			jobs:        batchJobs,
		}
	}

	var firstErr error
	for range batches {
		result := <-p.done
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	p.mu.Unlock()
	return firstErr
}

func (p *Pool) Close() {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	for i := 0; i < len(p.workers); i++ {
		close(p.workers[i].tasks)
	}
	p.mu.Unlock()
}

func validateBatches(batches []Batch, jobs []tile.Job, workers int) error {
	for i := range batches {
		batch := batches[i]
		start, end, err := batch.jobRange(len(jobs))
		if err != nil || int(batch.Worker) >= workers {
			return ErrInvalidBatch
		}
		if batch.FirstTile != jobs[start].Tile ||
			batch.LastTile != jobs[end-1].Tile {
			return ErrInvalidBatch
		}
	}
	return nil
}

func poolWorkerLoop(tasks <-chan poolTask, done chan<- workerResult) {
	for task := range tasks {
		if task.frameFn != nil {
			ctx := task.frameBatch
			ctx.Batch = task.batch
			ctx.Jobs = task.jobs
			done <- workerResult{err: task.frameFn(ctx)}
			continue
		}
		if task.frameRunner != nil {
			ctx := task.frameBatch
			ctx.Batch = task.batch
			ctx.Jobs = task.jobs
			done <- workerResult{err: task.frameRunner.Run(ctx)}
			continue
		}
		done <- workerResult{err: task.fn(task.batch, task.jobs)}
	}
}
