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

	MIColStart uint32
	MIRowStart uint32
	MIColEnd   uint32
	MIRowEnd   uint32
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
type FrameWorkPlaneRegion struct {
	Plane FrameWorkPlane
	Pix   []byte

	Stride         int
	X              int
	Y              int
	Width          int
	Height         int
	BytesPerSample int
	RowBytes       int
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
	ReferenceOrderHints [parser.InterRefsPerFrame]uint32
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
	FrameWorkFrameContext
	DisableCDFUpdate bool
	// InitialTileResidualCDFs is the frame context selected by
	// primary_ref_frame for this tile group. RetainedTileResidualCDFs receives
	// the context_update_tile_id tile's adapted state when callbacks opt in via
	// RetainTileResidualCDFStorage.
	InitialTileResidualCDFs       *FrameWorkTileResidualCDFStorage
	RetainedTileResidualCDFs      *FrameWorkTileResidualCDFStorage
	RetainedTileResidualCDFsValid *bool
	Batch                         Batch
	Jobs                          []tile.Job
}

// Surface returns the frame-pool surface that this batch reconstructs into.
func (b FrameWorkBatch) Surface() (int, error) {
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
func (b FrameWorkBatch) JobPayload(index int) ([]byte, error) {
	if index < 0 || index >= len(b.Jobs) {
		return nil, ErrInvalidBatch
	}
	return b.Jobs[index].Payload(b.Payload)
}

// JobEntropyReader returns an entropy reader over Jobs[index]'s exact tile
// payload. The reader inherits the frame-level CDF update mode carried by b.
func (b FrameWorkBatch) JobEntropyReader(index int) (entropy.Reader, error) {
	if index < 0 || index >= len(b.Jobs) {
		return entropy.Reader{}, ErrInvalidBatch
	}
	return tile.NewEntropyReader(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
	})
}

// JobDecodeState initializes state for Jobs[index]'s exact tile payload. The
// state records whether this job's adapted entropy context should be retained.
func (b FrameWorkBatch) JobDecodeState(index int, state *tile.DecodeState) error {
	if index < 0 || index >= len(b.Jobs) || state == nil {
		return ErrInvalidBatch
	}
	return state.Reset(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
		BaseQIdx:         b.Quantization.BaseQIdx,
	})
}

// JobRegion returns the clipped reconstruction region for Jobs[index].
func (b FrameWorkBatch) JobRegion(index int) (FrameWorkJobRegion, error) {
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
		MIColStart:  miColStart,
		MIRowStart:  miRowStart,
		MIColEnd:    miColEnd,
		MIRowEnd:    miRowEnd,
	}, nil
}

// JobBlockDeltaContext returns the AV1 block-delta context for an absolute MI
// coordinate inside Jobs[index]'s region.
func (b FrameWorkBatch) JobBlockDeltaContext(index int, miCol uint32, miRow uint32, fullSuperblock bool, skipTransform bool) (tile.BlockDeltaContext, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.BlockDeltaContext{}, err
	}
	if miCol < region.MIColStart || miCol >= region.MIColEnd ||
		miRow < region.MIRowStart || miRow >= region.MIRowEnd {
		return tile.BlockDeltaContext{}, ErrInvalidBatch
	}
	return tile.BlockDeltaContext{
		MICol:          miCol,
		MIRow:          miRow,
		SBSizeMIB:      b.Sequence.SBSizeMIB,
		FullSuperblock: fullSuperblock,
		SkipTransform:  skipTransform,
		Monochrome:     b.Sequence.ColorConfig.MonoChrome,
	}, nil
}

// RestorationPlaneGrid returns the loop-restoration unit grid for one plane of
// this batch's frame context.
func (b FrameWorkBatch) RestorationPlaneGrid(plane FrameWorkPlane) (tile.RestorationPlaneGrid, error) {
	if plane > FrameWorkPlaneV || !b.Sequence.Valid() {
		return tile.RestorationPlaneGrid{}, ErrInvalidBatch
	}
	return tile.BuildRestorationPlaneGrid(b.Restoration, b.FrameSize, b.Sequence.ColorConfig, int(plane))
}

// RestorationFramePlan returns the frame-level loop-restoration allocation plan
// for this batch's frame context.
func (b FrameWorkBatch) RestorationFramePlan() (tile.RestorationFramePlan, error) {
	if !b.Sequence.Valid() {
		return tile.RestorationFramePlan{}, ErrInvalidBatch
	}
	return tile.BuildRestorationFramePlan(b.Restoration, b.FrameSize, b.Sequence.ColorConfig)
}

// LoopRestorationPlan ports libaom's CDEF/superres/restoration scheduling
// decision for the frame. skipLoopFilter is the decoder-level skip flag from
// libaom's AV1Decoder.
func (b FrameWorkBatch) LoopRestorationPlan(skipLoopFilter bool) (FrameWorkLoopRestorationPlan, error) {
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
func (b FrameWorkBatch) JobRestorationUnitRange(index int, plane FrameWorkPlane, sbX uint16, sbY uint16) (tile.RestorationUnitRange, bool, error) {
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

// JobOutputPlane returns the clipped output-plane window for Jobs[index].
func (b FrameWorkBatch) JobOutputPlane(index int, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
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
	return frameWorkPlaneWindow(plane, outputPlane, bytesPerSample, x0, y0, x1, y1)
}

// ReferenceFrame returns the resolved frame for one AV1 inter-reference slot.
func (b FrameWorkBatch) ReferenceFrame(reference FrameWorkReference) (*frame.Frame, error) {
	index := int(reference)
	if index >= parser.InterRefsPerFrame || index >= len(b.References) || b.References[index] == nil {
		return nil, ErrInvalidBatch
	}
	return b.References[index], nil
}

// ReferencePlane returns the full plane from one resolved reference frame.
func (b FrameWorkBatch) ReferencePlane(reference FrameWorkReference, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
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
	return frameWorkPlaneWindow(plane, refPlane, bytesPerSample, 0, 0, uint32(refPlane.Width), uint32(refPlane.Height))
}

// JobReferencePlane returns the reference-plane window matching Jobs[index]'s
// clipped output region. It is valid for same-size reference sampling; scaled
// references require motion-compensation code to map coordinates explicitly.
func (b FrameWorkBatch) JobReferencePlane(index int, reference FrameWorkReference, plane FrameWorkPlane) (FrameWorkPlaneRegion, error) {
	return b.JobReferencePlaneWindow(index, reference, plane, 0, 0)
}

// JobReferencePlaneWindow returns the reference-plane window matching
// Jobs[index]'s clipped output region and expanded by marginX/marginY plane
// samples. The window is clipped to the resolved reference frame plane.
func (b FrameWorkBatch) JobReferencePlaneWindow(index int, reference FrameWorkReference, plane FrameWorkPlane, marginX uint32, marginY uint32) (FrameWorkPlaneRegion, error) {
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
	return frameWorkPlaneWindow(plane, refPlane, bytesPerSample, x0, y0, x1, y1)
}

// JobUpdatesFrameContext reports whether Jobs[index] is the designated tile
// whose adapted entropy state should refresh the frame context.
func (b FrameWorkBatch) JobUpdatesFrameContext(index int) (bool, error) {
	if index < 0 || index >= len(b.Jobs) {
		return false, ErrInvalidBatch
	}
	return b.Jobs[index].UpdatesFrameContext, nil
}

// ValidatePayloads checks that every job in the batch names a byte range inside
// Payload.
func (b FrameWorkBatch) ValidatePayloads() error {
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

func frameWorkPlaneWindow(which FrameWorkPlane, plane frame.Plane, bytesPerSample int, x0 uint32, y0 uint32, x1 uint32, y1 uint32) (FrameWorkPlaneRegion, error) {
	if bytesPerSample <= 0 ||
		plane.Stride <= 0 ||
		plane.Width <= 0 ||
		plane.Height <= 0 ||
		uint64(plane.Width) > uint64(^uint32(0)) ||
		uint64(plane.Height) > uint64(^uint32(0)) ||
		x1 <= x0 ||
		y1 <= y0 ||
		x1 > uint32(plane.Width) ||
		y1 > uint32(plane.Height) {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}

	x := int(x0)
	y := int(y0)
	width := int(x1 - x0)
	height := int(y1 - y0)
	rowBytes, ok := frameWorkCheckedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
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
	lastRowOffset, ok := frameWorkCheckedMul(height-1, plane.Stride)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	windowLen, ok := frameWorkCheckedAdd(lastRowOffset, rowBytes)
	if !ok {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	end, ok := frameWorkCheckedAdd(offset, windowLen)
	if !ok || offset < 0 || end > len(plane.Pix) {
		return FrameWorkPlaneRegion{}, ErrInvalidBatch
	}
	return FrameWorkPlaneRegion{
		Plane:          which,
		Pix:            plane.Pix[offset:end],
		Stride:         plane.Stride,
		X:              x,
		Y:              y,
		Width:          width,
		Height:         height,
		BytesPerSample: bytesPerSample,
		RowBytes:       rowBytes,
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
	fn         BatchFunc
	frameFn    FrameWorkBatchFunc
	frameBatch FrameWorkBatch
	batch      Batch
	jobs       []tile.Job
}

type workerResult struct {
	err error
}

// NewPool starts a reusable bounded worker pool. Pool creation is a cold-path
// operation; Execute is the reusable hot path.
func NewPool(workers int) (*Pool, error) {
	if workers <= 0 {
		return nil, ErrInvalidWorkerCount
	}
	p := &Pool{
		workers: make([]poolWorker, workers),
		done:    make(chan workerResult, workers),
	}
	for i := 0; i < workers; i++ {
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

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		p.workers[batch.Worker].tasks <- poolTask{
			fn:    fn,
			batch: batch,
			jobs:  jobs[batch.FirstJob : batch.FirstJob+batch.Count],
		}
	}

	var firstErr error
	for i := 0; i < len(batches); i++ {
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

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		p.workers[batch.Worker].tasks <- poolTask{
			frameFn:    fn,
			frameBatch: base,
			batch:      batch,
			jobs:       jobs[batch.FirstJob : batch.FirstJob+batch.Count],
		}
	}

	var firstErr error
	for i := 0; i < len(batches); i++ {
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
	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		if int(batch.Worker) >= workers ||
			batch.FirstJob < 0 ||
			batch.Count <= 0 ||
			batch.FirstJob+batch.Count > len(jobs) {
			return ErrInvalidBatch
		}
		if batch.FirstTile != jobs[batch.FirstJob].Tile ||
			batch.LastTile != jobs[batch.FirstJob+batch.Count-1].Tile {
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
		done <- workerResult{err: task.fn(task.batch, task.jobs)}
	}
}
