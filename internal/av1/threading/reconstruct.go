package threading

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// FrameWorkBlockCoeffReconstruction carries one decoded TXB plus the block-loop
// position needed to place it in the worker's output frame.
type FrameWorkBlockCoeffReconstruction struct {
	Visit tile.BlockVisit
	Block tile.BlockCoeffBlock

	Transform     transform.Type
	CurrentQIndex uint8
	SegmentID     uint8

	Int32Scratch    []int32
	ResidualScratch []int16
}

// BlockQuantizer derives the dequantizer and lossless flag for one decoded
// block using the current delta-q state and, when segmentation is enabled, the
// block's segment delta.
func (b *FrameWorkBatch) BlockQuantizer(currentQIndex uint8, segmentID uint8, plane FrameWorkPlane) (quantize.Quantizer, bool, error) {
	qPlane, ok := frameWorkQuantizePlane(plane)
	if !ok {
		return quantize.Quantizer{}, false, ErrInvalidBatch
	}
	return b.blockQuantizerForPlane(currentQIndex, segmentID, qPlane)
}

// blockQuantizerForPlane is BlockQuantizer once the quantize.Plane is already
// resolved. ReconstructBlockCoeff needs the qPlane for the inverse-quant matrix
// too, so it resolves frameWorkQuantizePlane once and shares it here instead of
// having BlockQuantizer derive it a second time per transform block.
func (b *FrameWorkBatch) blockQuantizerForPlane(currentQIndex uint8, segmentID uint8, qPlane quantize.Plane) (quantize.Quantizer, bool, error) {
	if !b.Sequence.Valid() {
		return quantize.Quantizer{}, false, ErrInvalidBatch
	}
	qIndex, lossless, err := b.BlockQIndex(currentQIndex, segmentID)
	if err != nil {
		return quantize.Quantizer{}, false, err
	}
	q, err := quantize.PlaneQuantizer(b.Quantization, qIndex, b.Sequence.ColorConfig.BitDepth, qPlane)
	if err != nil {
		return quantize.Quantizer{}, false, ErrInvalidBatch
	}
	return q, lossless, nil
}

// BlockQIndex derives the segment-adjusted qindex for the current block. The
// current qindex is the tile decode state's CurrentBaseQIdx after delta_qindex
// has been applied for the block.
func (b *FrameWorkBatch) BlockQIndex(currentQIndex uint8, segmentID uint8) (uint8, bool, error) {
	qIndex := currentQIndex
	if b.Segmentation.Enabled {
		if segmentID >= parser.MaxSegments {
			return 0, false, ErrInvalidBatch
		}
		qIndex = quantize.ClampQIndex(currentQIndex, b.Segmentation.Data.Segments[segmentID].DeltaQ)
	} else if segmentID != 0 {
		return 0, false, ErrInvalidBatch
	}
	return qIndex, qIndex == 0 && frameWorkQuantDeltasZero(b.Quantization), nil
}

// BlockCoeffPlanePosition maps a decoded coefficient block to absolute output
// plane samples. The returned coordinates are in the plane's own sample grid.
func (b *FrameWorkBatch) BlockCoeffPlanePosition(index int, visit tile.BlockVisit, block tile.BlockCoeffBlock) (FrameWorkPlane, int, int, error) {
	geom, err := b.blockCoeffGeometry(index, visit, &block)
	if err != nil {
		return 0, 0, 0, err
	}
	return geom.plane, geom.x, geom.y, nil
}

// ReconstructBlockCoeff dequantizes and adds one decoded transform block into
// Jobs[index]'s clipped output-plane window.
func (b *FrameWorkBatch) ReconstructBlockCoeff(index int, req FrameWorkBlockCoeffReconstruction) error {
	return b.reconstructBlockCoeffCore(index, req.Visit, &req.Block, req.Transform, req.CurrentQIndex, req.SegmentID, req.Int32Scratch, req.ResidualScratch)
}

// reconstructBlockCoeffCore is the by-pointer reconstruction seam shared by the
// public ReconstructBlockCoeff and the hot per-TXB reconstructTXB path. Taking
// the decoded block by pointer lets reconstructTXB hand the live decode block
// straight through instead of materializing a 216-byte
// FrameWorkBlockCoeffReconstruction (with an embedded 120-byte BlockCoeffBlock
// copy) for every transform block.
func (b *FrameWorkBatch) reconstructBlockCoeffCore(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock, txType transform.Type, currentQIndex uint8, segmentID uint8, int32Scratch []int32, residualScratch []int16) error {
	geom, err := b.blockCoeffGeometry(index, visit, block)
	if err != nil {
		return err
	}
	if geom.visibleWidth == 0 || geom.visibleHeight == 0 {
		return nil
	}
	// Resolve the quantize.Plane once and share it between the dequantizer and
	// the inverse-quant matrix lookup (BlockQuantizer would otherwise re-derive
	// it per transform block).
	qPlane, ok := frameWorkQuantizePlane(geom.plane)
	if !ok {
		return ErrInvalidBatch
	}
	q, lossless, err := b.blockQuantizerForPlane(currentQIndex, segmentID, qPlane)
	if err != nil {
		return err
	}
	iqMatrix, err := quantize.InverseQMatrix(b.Quantization, qPlane, geom.size, txType, lossless)
	if err != nil {
		return ErrInvalidBatch
	}

	dst := frameWorkPlaneFromWindow(geom.window)
	cfg := reconstruct.Block{
		Size:           geom.size,
		Transform:      txType,
		Quantizer:      q,
		InverseQMatrix: iqMatrix,
		Lossless:       lossless,
		EOB:            block.Result.EOB,
	}
	if err := reconstruct.ReconstructPlaneBlockVisibleWithGeometryAndScan(dst, geom.window.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
		geom.x-geom.window.X, geom.y-geom.window.Y, geom.visibleWidth, geom.visibleHeight,
		block.Coeffs, geom.scanSize.Height, block.Scan, geom.scanSize, geom.txScale, int32Scratch, residualScratch, cfg); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

type frameWorkBlockCoeffGeometry struct {
	plane         FrameWorkPlane
	window        FrameWorkPlaneRegion
	x             int
	y             int
	size          transform.Size
	scanSize      transform.Size
	txScale       uint8
	visibleWidth  int
	visibleHeight int
}

func (b *FrameWorkBatch) blockCoeffGeometry(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock) (frameWorkBlockCoeffGeometry, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	plane, ssX, ssY, err := b.blockCoeffPlane(block.Plane)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	window, err := b.JobOutputPlane(index, plane)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	size, err := block.Block.Size.TransformSize()
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	x, y, err := frameWorkBlockCoeffPosition(region, visit, block.Block, ssX, ssY)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	// libaom's av1_inverse_transform_block adds the residual of the WHOLE
	// tx_size to the predicted block; it never trims to the cropped visible
	// rectangle (trailing rows/cols of a transform crossing the cropped edge
	// spill into the YV12 border). A transform is skipped only when its
	// blk_row/blk_col origin reaches the MI grid edge (the
	// frameWorkPlaneBlockStartsBeyondOutput case below). Mirror that by writing
	// the full transform size clamped to the window's superblock-aligned
	// writable extent rather than block.VisibleW4/H4. (The visible-size check is
	// retained for input validation.)
	if _, _, err := frameWorkBlockCoeffVisibleSize(block.Block, size); err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	txScale, err := quantize.TransformScale(size.Width, size.Height)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	visibleWidth, visibleHeight, ok := frameWorkClipVisiblePixelsToWindow(window, x, y, size.Width, size.Height)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, x, y) {
			if plane == FrameWorkPlaneY {
				if _, _, err := frameWorkBlockLumaTransformPosition(visit, block.Block); err != nil {
					return frameWorkBlockCoeffGeometry{}, err
				}
			}
			return frameWorkBlockCoeffGeometry{
				plane:    plane,
				window:   window,
				x:        x,
				y:        y,
				size:     size,
				scanSize: scanSize,
				txScale:  txScale,
			}, nil
		}
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	return frameWorkBlockCoeffGeometry{
		plane:         plane,
		window:        window,
		x:             x,
		y:             y,
		size:          size,
		scanSize:      scanSize,
		txScale:       txScale,
		visibleWidth:  visibleWidth,
		visibleHeight: visibleHeight,
	}, nil
}

func frameWorkBlockCoeffVisibleSize(block tile.TransformBlock, size transform.Size) (int, int, error) {
	if block.VisibleW4 == 0 || block.VisibleH4 == 0 {
		return 0, 0, ErrInvalidBatch
	}
	visibleWidth, ok := frameWorkInt64Mul4(int64(block.VisibleW4))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	visibleHeight, ok := frameWorkInt64Mul4(int64(block.VisibleH4))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	if visibleWidth > size.Width || visibleHeight > size.Height {
		return 0, 0, ErrInvalidBatch
	}
	return visibleWidth, visibleHeight, nil
}

func (b *FrameWorkBatch) blockCoeffPlane(tilePlane int) (FrameWorkPlane, uint, uint, error) {
	switch tilePlane {
	case 0:
		return FrameWorkPlaneY, 0, 0, nil
	case 1:
		if b.Sequence.ColorConfig.MonoChrome {
			return 0, 0, 0, ErrInvalidBatch
		}
		return FrameWorkPlaneU, frameWorkSubsampleShift(b.Sequence.ColorConfig.SubsamplingX), frameWorkSubsampleShift(b.Sequence.ColorConfig.SubsamplingY), nil
	case 2:
		if b.Sequence.ColorConfig.MonoChrome {
			return 0, 0, 0, ErrInvalidBatch
		}
		return FrameWorkPlaneV, frameWorkSubsampleShift(b.Sequence.ColorConfig.SubsamplingX), frameWorkSubsampleShift(b.Sequence.ColorConfig.SubsamplingY), nil
	default:
		return 0, 0, 0, ErrInvalidBatch
	}
}

func frameWorkBlockCoeffPosition(region FrameWorkJobRegion, visit tile.BlockVisit, block tile.TransformBlock, ssX uint, ssY uint) (int, int, error) {
	if visit.MICol < region.MIColStart || visit.MIRow < region.MIRowStart ||
		visit.MIColEnd > region.MIColEnd || visit.MIRowEnd > region.MIRowEnd ||
		visit.MIColEnd <= visit.MICol || visit.MIRowEnd <= visit.MIRow ||
		visit.X4 < 0 || visit.Y4 < 0 || block.X4 < 0 || block.Y4 < 0 {
		return 0, 0, ErrInvalidBatch
	}
	rootCol := int64(visit.MICol) - int64(visit.X4)
	rootRow := int64(visit.MIRow) - int64(visit.Y4)
	if rootCol < 0 || rootRow < 0 {
		return 0, 0, ErrInvalidBatch
	}
	x4 := (rootCol >> ssX) + int64(block.X4)
	y4 := (rootRow >> ssY) + int64(block.Y4)
	x, ok := frameWorkInt64Mul4(x4)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	y, ok := frameWorkInt64Mul4(y4)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return x, y, nil
}

func frameWorkQuantizePlane(plane FrameWorkPlane) (quantize.Plane, bool) {
	switch plane {
	case FrameWorkPlaneY:
		return quantize.PlaneY, true
	case FrameWorkPlaneU:
		return quantize.PlaneU, true
	case FrameWorkPlaneV:
		return quantize.PlaneV, true
	default:
		return 0, false
	}
}

func frameWorkQuantDeltasZero(params parser.QuantizationParams) bool {
	return params.YDCDelta == 0 &&
		params.UDCDelta == 0 &&
		params.UACDelta == 0 &&
		params.VDCDelta == 0 &&
		params.VACDelta == 0
}

func frameWorkSubsampleShift(subsampled bool) uint {
	if subsampled {
		return 1
	}
	return 0
}

func frameWorkInt64Mul4(v int64) (int, bool) {
	if v < 0 || v > int64(^uint(0)>>3) {
		return 0, false
	}
	return int(v << 2), true
}
