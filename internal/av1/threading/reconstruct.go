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
	return geom.plane, int(geom.x), int(geom.y), nil
}

// ReconstructBlockCoeff dequantizes and adds one decoded transform block into
// Jobs[index]'s clipped output-plane window.
func (b *FrameWorkBatch) ReconstructBlockCoeff(index int, req FrameWorkBlockCoeffReconstruction) error {
	return b.reconstructBlockCoeffCore(index, req.Visit, &req.Block, req.Transform, req.CurrentQIndex, req.SegmentID, req.Int32Scratch, req.ResidualScratch)
}

type frameWorkReconQuantCache struct {
	matrix        []uint16
	quantizer     [3]quantize.Quantizer
	quantKey      [3]uint16
	matrixPlane   quantize.Plane
	matrixSize    transform.Size
	matrixType    transform.Type
	losslessPlane uint8
	matrixValid   bool
	matrixLoss    bool
}

// reconstructBlockCoeffCore is the checked by-pointer reconstruction seam used
// by public callers. It preserves ReconstructBlockCoeff's invalid-input
// behavior while avoiding the large request struct copy.
func (b *FrameWorkBatch) reconstructBlockCoeffCore(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock, txType transform.Type, currentQIndex uint8, segmentID uint8, int32Scratch []int32, residualScratch []int16) error {
	geom, err := b.blockCoeffGeometry(index, visit, block)
	if err != nil {
		return err
	}
	if geom.visibleWidth == 0 || geom.visibleHeight == 0 {
		return nil
	}
	relX := int(geom.x) - int(geom.window.X)
	relY := int(geom.y) - int(geom.window.Y)
	visibleWidth := int(geom.visibleWidth)
	visibleHeight := int(geom.visibleHeight)
	scanStride := int(geom.scanSize.Height)
	qPlane := quantize.Plane(geom.plane)
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
		EOB:            int16(block.Result.EOB),
	}
	if err := reconstruct.ReconstructPlaneBlockVisibleWithGeometryAndScan(dst, int(geom.window.BytesPerSample), b.Sequence.ColorConfig.BitDepth,
		relX, relY, visibleWidth, visibleHeight,
		block.Coeffs, scanStride, block.Scan, geom.scanSize, geom.txScale, int32Scratch, residualScratch, cfg); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// reconstructBlockCoeffCoreTrusted is the hot per-TXB variant. Tile residual
// decoding has already validated the batch, block geometry, and worker scratch
// ownership, so this path can hand exact-cap scratch/window slices to the
// reconstruction core without re-running the public API guard rails.
func (b *FrameWorkBatch) reconstructBlockCoeffCoreTrusted(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock, txType transform.Type, currentQIndex uint8, segmentID uint8, int32Scratch []int32, residualScratch []int16, cache *frameWorkReconQuantCache) error {
	geom, err := b.blockCoeffGeometryTrusted(index, visit, block)
	if err != nil {
		return err
	}
	return b.reconstructBlockCoeffCoreWithGeometry(geom, block, txType, currentQIndex, segmentID, int32Scratch, residualScratch, cache)
}

func frameWorkPrimeLumaReconGeometry(b *FrameWorkBatch, index int) error {
	if _, err := b.JobRegion(index); err != nil {
		return err
	}
	if _, err := b.JobOutputPlane(index, FrameWorkPlaneY); err != nil {
		return err
	}
	return nil
}

// reconstructBlockCoeffLumaPrimed reconstructs the residual for a luma TXB
// using the job geometry entries that DecodeAndReconstructJobResidualsPtr
// primes once per tile job. The public and chroma paths keep the fully checked
// geometry route; this hot path only skips the per-TXB cache lookup ladder when
// the already-keyed region and Y plane are present for the current job.
func (b *FrameWorkBatch) reconstructBlockCoeffLumaPrimed(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock, txType transform.Type, currentQIndex uint8, segmentID uint8, int32Scratch []int32, residualScratch []int16, cache *frameWorkReconQuantCache) (bool, error) {
	if block.Plane != 0 {
		return false, nil
	}
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	c := b.geomCache
	if c == nil || !ok ||
		c.validMask&(frameWorkJobGeometryRegionValid|frameWorkJobGeometryPlaneYValid) != frameWorkJobGeometryRegionValid|frameWorkJobGeometryPlaneYValid ||
		c.regionIndex != cacheIndex || c.planeIndex[FrameWorkPlaneY] != cacheIndex {
		return false, nil
	}
	geom, err := b.blockCoeffGeometryLumaKnown(c.region, c.plane[FrameWorkPlaneY], visit, block)
	if err != nil {
		return true, err
	}
	return true, b.reconstructBlockCoeffCoreWithGeometry(geom, block, txType, currentQIndex, segmentID, int32Scratch, residualScratch, cache)
}

func (b *FrameWorkBatch) reconstructBlockCoeffCoreWithGeometry(geom frameWorkBlockCoeffGeometry, block *tile.BlockCoeffBlock, txType transform.Type, currentQIndex uint8, segmentID uint8, int32Scratch []int32, residualScratch []int16, cache *frameWorkReconQuantCache) error {
	if geom.visibleWidth == 0 || geom.visibleHeight == 0 {
		return nil
	}
	relX := int(geom.x) - int(geom.window.X)
	relY := int(geom.y) - int(geom.window.Y)
	visibleWidth := int(geom.visibleWidth)
	visibleHeight := int(geom.visibleHeight)
	scanStride := int(geom.scanSize.Height)
	qPlane := quantize.Plane(geom.plane)
	q, lossless, err := b.cachedBlockQuantizerTrusted(cache, currentQIndex, segmentID, qPlane)
	if err != nil {
		return err
	}
	iqMatrix, err := b.cachedInverseQMatrixTrusted(cache, qPlane, geom.size, txType, lossless)
	if err != nil {
		return ErrInvalidBatch
	}

	bytesPerSample := int(geom.window.BytesPerSample)
	dstStride := int(geom.window.Stride)
	dstOffset := relY*dstStride + relX*bytesPerSample
	rowBytes := visibleWidth * bytesPerSample
	dstLen := (visibleHeight-1)*dstStride + rowBytes
	dst := geom.window.Pix[dstOffset : dstOffset+dstLen : dstOffset+dstLen]
	cfg := reconstruct.Block{
		Size:           geom.size,
		Transform:      txType,
		Quantizer:      q,
		InverseQMatrix: iqMatrix,
		Lossless:       lossless,
		EOB:            int16(block.Result.EOB),
	}
	if err := reconstruct.ReconstructPlaneBlockVisibleTrustedAtWithGeometryAndScan(dst, dstStride, bytesPerSample, b.Sequence.ColorConfig.BitDepth,
		visibleWidth, visibleHeight,
		block.Coeffs, scanStride, block.Scan, geom.scanSize, geom.txScale, int32Scratch, residualScratch, cfg); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b *FrameWorkBatch) cachedBlockQuantizer(cache *frameWorkReconQuantCache, currentQIndex uint8, segmentID uint8, qPlane quantize.Plane) (quantize.Quantizer, bool, error) {
	idx := int(qPlane)
	key := frameWorkReconQuantKey(currentQIndex, segmentID)
	if cache != nil && uint(idx) < uint(len(cache.quantKey)) &&
		cache.quantKey[idx] == key {
		return cache.quantizer[idx], cache.losslessPlane&(1<<uint(idx)) != 0, nil
	}
	q, lossless, err := b.blockQuantizerForPlane(currentQIndex, segmentID, qPlane)
	if err != nil {
		return quantize.Quantizer{}, false, err
	}
	if cache != nil && uint(idx) < uint(len(cache.quantKey)) {
		cache.quantKey[idx] = key
		cache.quantizer[idx] = q
		if lossless {
			cache.losslessPlane |= 1 << uint(idx)
		} else {
			cache.losslessPlane &^= 1 << uint(idx)
		}
	}
	return q, lossless, nil
}

func (b *FrameWorkBatch) cachedBlockQuantizerTrusted(cache *frameWorkReconQuantCache, currentQIndex uint8, segmentID uint8, qPlane quantize.Plane) (quantize.Quantizer, bool, error) {
	idx := int(qPlane)
	key := frameWorkReconQuantKey(currentQIndex, segmentID)
	if uint(idx) < uint(len(cache.quantKey)) &&
		cache.quantKey[idx] == key {
		return cache.quantizer[idx], cache.losslessPlane&(1<<uint(idx)) != 0, nil
	}
	q, lossless, err := b.blockQuantizerForPlane(currentQIndex, segmentID, qPlane)
	if err != nil {
		return quantize.Quantizer{}, false, err
	}
	if uint(idx) < uint(len(cache.quantKey)) {
		cache.quantKey[idx] = key
		cache.quantizer[idx] = q
		if lossless {
			cache.losslessPlane |= 1 << uint(idx)
		} else {
			cache.losslessPlane &^= 1 << uint(idx)
		}
	}
	return q, lossless, nil
}

func frameWorkReconQuantKey(currentQIndex uint8, segmentID uint8) uint16 {
	return 0x8000 | uint16(segmentID)<<8 | uint16(currentQIndex)
}

func (b *FrameWorkBatch) cachedInverseQMatrix(cache *frameWorkReconQuantCache, qPlane quantize.Plane, size transform.Size, txType transform.Type, lossless bool) ([]uint16, error) {
	if !b.Quantization.UsingQMatrix || lossless || txType >= transform.TypeIDTX {
		return nil, nil
	}
	if cache != nil &&
		cache.matrixValid &&
		cache.matrixPlane == qPlane &&
		cache.matrixSize == size &&
		cache.matrixType == txType &&
		cache.matrixLoss == lossless {
		return cache.matrix, nil
	}
	matrix, err := quantize.InverseQMatrix(b.Quantization, qPlane, size, txType, lossless)
	if err != nil {
		return nil, err
	}
	if cache != nil {
		cache.matrixValid = true
		cache.matrixPlane = qPlane
		cache.matrixSize = size
		cache.matrixType = txType
		cache.matrixLoss = lossless
		cache.matrix = matrix
	}
	return matrix, nil
}

func (b *FrameWorkBatch) cachedInverseQMatrixTrusted(cache *frameWorkReconQuantCache, qPlane quantize.Plane, size transform.Size, txType transform.Type, lossless bool) ([]uint16, error) {
	if !b.Quantization.UsingQMatrix || lossless || txType >= transform.TypeIDTX {
		return nil, nil
	}
	if cache.matrixValid &&
		cache.matrixPlane == qPlane &&
		cache.matrixSize == size &&
		cache.matrixType == txType &&
		cache.matrixLoss == lossless {
		return cache.matrix, nil
	}
	matrix, err := quantize.InverseQMatrix(b.Quantization, qPlane, size, txType, lossless)
	if err != nil {
		return nil, err
	}
	cache.matrixValid = true
	cache.matrixPlane = qPlane
	cache.matrixSize = size
	cache.matrixType = txType
	cache.matrixLoss = lossless
	cache.matrix = matrix
	return matrix, nil
}

type frameWorkBlockCoeffGeometry struct {
	window        FrameWorkPlaneRegion
	size          transform.Size
	scanSize      transform.Size
	x             uint32
	y             uint32
	visibleWidth  uint8
	visibleHeight uint8
	txScale       uint8
	plane         FrameWorkPlane
}

type frameWorkTransformMeta struct {
	size     transform.Size
	scanSize transform.Size
	txScale  uint8
}

var frameWorkTransformMetaBySize = [...]frameWorkTransformMeta{
	tile.TransformSize4x4:   {size: transform.Size{Width: 4, Height: 4}, scanSize: transform.Size{Width: 4, Height: 4}, txScale: 0},
	tile.TransformSize8x8:   {size: transform.Size{Width: 8, Height: 8}, scanSize: transform.Size{Width: 8, Height: 8}, txScale: 0},
	tile.TransformSize16x16: {size: transform.Size{Width: 16, Height: 16}, scanSize: transform.Size{Width: 16, Height: 16}, txScale: 0},
	tile.TransformSize32x32: {size: transform.Size{Width: 32, Height: 32}, scanSize: transform.Size{Width: 32, Height: 32}, txScale: 1},
	tile.TransformSize64x64: {size: transform.Size{Width: 64, Height: 64}, scanSize: transform.Size{Width: 32, Height: 32}, txScale: 2},
	tile.TransformSize4x8:   {size: transform.Size{Width: 4, Height: 8}, scanSize: transform.Size{Width: 4, Height: 8}, txScale: 0},
	tile.TransformSize8x4:   {size: transform.Size{Width: 8, Height: 4}, scanSize: transform.Size{Width: 8, Height: 4}, txScale: 0},
	tile.TransformSize8x16:  {size: transform.Size{Width: 8, Height: 16}, scanSize: transform.Size{Width: 8, Height: 16}, txScale: 0},
	tile.TransformSize16x8:  {size: transform.Size{Width: 16, Height: 8}, scanSize: transform.Size{Width: 16, Height: 8}, txScale: 0},
	tile.TransformSize16x32: {size: transform.Size{Width: 16, Height: 32}, scanSize: transform.Size{Width: 16, Height: 32}, txScale: 1},
	tile.TransformSize32x16: {size: transform.Size{Width: 32, Height: 16}, scanSize: transform.Size{Width: 32, Height: 16}, txScale: 1},
	tile.TransformSize32x64: {size: transform.Size{Width: 32, Height: 64}, scanSize: transform.Size{Width: 32, Height: 32}, txScale: 2},
	tile.TransformSize64x32: {size: transform.Size{Width: 64, Height: 32}, scanSize: transform.Size{Width: 32, Height: 32}, txScale: 2},
	tile.TransformSize4x16:  {size: transform.Size{Width: 4, Height: 16}, scanSize: transform.Size{Width: 4, Height: 16}, txScale: 0},
	tile.TransformSize16x4:  {size: transform.Size{Width: 16, Height: 4}, scanSize: transform.Size{Width: 16, Height: 4}, txScale: 0},
	tile.TransformSize8x32:  {size: transform.Size{Width: 8, Height: 32}, scanSize: transform.Size{Width: 8, Height: 32}, txScale: 0},
	tile.TransformSize32x8:  {size: transform.Size{Width: 32, Height: 8}, scanSize: transform.Size{Width: 32, Height: 8}, txScale: 0},
	tile.TransformSize16x64: {size: transform.Size{Width: 16, Height: 64}, scanSize: transform.Size{Width: 16, Height: 32}, txScale: 1},
	tile.TransformSize64x16: {size: transform.Size{Width: 64, Height: 16}, scanSize: transform.Size{Width: 32, Height: 16}, txScale: 1},
}

func frameWorkBlockCoeffTransformMeta(size tile.TransformSize) (frameWorkTransformMeta, bool) {
	if !size.Valid() {
		return frameWorkTransformMeta{}, false
	}
	return frameWorkTransformMetaBySize[size], true
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
	return b.blockCoeffGeometryKnown(region, window, plane, ssX, ssY, visit, block)
}

func (b *FrameWorkBatch) blockCoeffGeometryTrusted(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock) (frameWorkBlockCoeffGeometry, error) {
	if block.Plane == 0 {
		return b.blockCoeffGeometryLumaTrusted(index, visit, block)
	}
	color := b.Sequence.ColorConfig
	if color.MonoChrome {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	ssX := uint(frameWorkSubsampleShift(color.SubsamplingX))
	ssY := uint(frameWorkSubsampleShift(color.SubsamplingY))
	plane := FrameWorkPlaneU
	if block.Plane == 2 {
		plane = FrameWorkPlaneV
	} else if block.Plane != 1 {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	var err error
	region, regionOK := b.cachedJobRegionTrusted(index)
	if !regionOK {
		region, err = b.JobRegion(index)
		if err != nil {
			return frameWorkBlockCoeffGeometry{}, err
		}
	}
	window, windowOK := b.cachedJobOutputPlaneTrusted(index, plane)
	if !windowOK {
		window, err = b.JobOutputPlane(index, plane)
		if err != nil {
			return frameWorkBlockCoeffGeometry{}, err
		}
	}
	return b.blockCoeffGeometryKnown(region, window, plane, ssX, ssY, visit, block)
}

func (b *FrameWorkBatch) blockCoeffGeometryLumaTrusted(index int, visit tile.BlockVisit, block *tile.BlockCoeffBlock) (frameWorkBlockCoeffGeometry, error) {
	cacheIndex, cacheOK := frameWorkJobCacheIndex(index)
	cache := b.geomCache
	var region FrameWorkJobRegion
	if cache == nil || !cacheOK ||
		cache.validMask&frameWorkJobGeometryRegionValid == 0 ||
		cache.regionIndex != cacheIndex {
		var err error
		region, err = b.JobRegion(index)
		if err != nil {
			return frameWorkBlockCoeffGeometry{}, err
		}
	} else {
		region = cache.region
	}
	var window FrameWorkPlaneRegion
	if cache == nil || !cacheOK ||
		cache.validMask&frameWorkJobGeometryPlaneYValid == 0 ||
		cache.planeIndex[FrameWorkPlaneY] != cacheIndex {
		var err error
		window, err = b.JobOutputPlane(index, FrameWorkPlaneY)
		if err != nil {
			return frameWorkBlockCoeffGeometry{}, err
		}
	} else {
		window = cache.plane[FrameWorkPlaneY]
	}
	return b.blockCoeffGeometryLumaKnown(region, window, visit, block)
}

func (b *FrameWorkBatch) cachedJobRegionTrusted(index int) (FrameWorkJobRegion, bool) {
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	c := b.geomCache
	if c == nil || !ok ||
		c.validMask&frameWorkJobGeometryRegionValid == 0 ||
		c.regionIndex != cacheIndex {
		return FrameWorkJobRegion{}, false
	}
	return c.region, true
}

func (b *FrameWorkBatch) cachedJobOutputPlaneTrusted(index int, plane FrameWorkPlane) (FrameWorkPlaneRegion, bool) {
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	c := b.geomCache
	if c == nil || !ok || plane > FrameWorkPlaneV {
		return FrameWorkPlaneRegion{}, false
	}
	planeMask := frameWorkJobGeometryPlaneMask(plane)
	if c.validMask&planeMask == 0 || c.planeIndex[plane] != cacheIndex {
		return FrameWorkPlaneRegion{}, false
	}
	return c.plane[plane], true
}

func (b *FrameWorkBatch) blockCoeffGeometryKnown(region FrameWorkJobRegion, window FrameWorkPlaneRegion, plane FrameWorkPlane, ssX uint, ssY uint, visit tile.BlockVisit, block *tile.BlockCoeffBlock) (frameWorkBlockCoeffGeometry, error) {
	meta, ok := frameWorkBlockCoeffTransformMeta(block.Block.Size)
	if !ok {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	size := meta.size
	x, y, err := frameWorkBlockCoeffPosition(region, visit, block.Block, ssX, ssY)
	if err != nil {
		return frameWorkBlockCoeffGeometry{}, err
	}
	geomX, geomY, ok := frameWorkBlockCoeffCoords(x, y)
	if !ok {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
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
	scanSize := meta.scanSize
	width := int(size.Width)
	height := int(size.Height)
	txScale := meta.txScale
	visibleWidth, visibleHeight, ok := frameWorkClipVisiblePixelsToWindow(window, x, y, width, height)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, x, y) {
			if plane == FrameWorkPlaneY {
				if _, _, err := frameWorkBlockLumaTransformPosition(visit, block.Block); err != nil {
					return frameWorkBlockCoeffGeometry{}, err
				}
			}
			return frameWorkBlockCoeffGeometry{
				window:   window,
				size:     size,
				scanSize: scanSize,
				x:        geomX,
				y:        geomY,
				txScale:  txScale,
				plane:    plane,
			}, nil
		}
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	geomVisibleWidth, geomVisibleHeight, ok := frameWorkBlockCoeffVisiblePixels(visibleWidth, visibleHeight)
	if !ok {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	return frameWorkBlockCoeffGeometry{
		window:        window,
		size:          size,
		scanSize:      scanSize,
		x:             geomX,
		y:             geomY,
		visibleWidth:  geomVisibleWidth,
		visibleHeight: geomVisibleHeight,
		txScale:       txScale,
		plane:         plane,
	}, nil
}

func (b *FrameWorkBatch) blockCoeffGeometryLumaKnown(region FrameWorkJobRegion, window FrameWorkPlaneRegion, visit tile.BlockVisit, block *tile.BlockCoeffBlock) (frameWorkBlockCoeffGeometry, error) {
	meta, ok := frameWorkBlockCoeffTransformMeta(block.Block.Size)
	if !ok {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	if visit.MICol < region.MIColStart || visit.MIRow < region.MIRowStart ||
		visit.MIColEnd > region.MIColEnd || visit.MIRowEnd > region.MIRowEnd ||
		visit.MIColEnd <= visit.MICol || visit.MIRowEnd <= visit.MIRow {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	txX4 := uint16(block.Block.X4)
	txY4 := uint16(block.Block.Y4)
	blockX4 := uint16(visit.X4)
	blockY4 := uint16(visit.Y4)
	visibleW4 := uint16(block.Block.VisibleW4)
	visibleH4 := uint16(block.Block.VisibleH4)
	if visibleW4 == 0 || visibleH4 == 0 ||
		txX4 < blockX4 || txY4 < blockY4 ||
		txX4+visibleW4 > blockX4+uint16(visit.VisibleW4) ||
		txY4+visibleH4 > blockY4+uint16(visit.VisibleH4) ||
		visibleW4<<2 > uint16(meta.size.Width) ||
		visibleH4<<2 > uint16(meta.size.Height) {
		return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
	}
	x := (uint32(visit.MICol) + uint32(txX4-blockX4)) << 2
	y := (uint32(visit.MIRow) + uint32(txY4-blockY4)) << 2
	width := int(meta.size.Width)
	height := int(meta.size.Height)
	var geomVisibleWidth, geomVisibleHeight uint8
	clipWidth := window.ClipWidth
	if clipWidth == 0 {
		clipWidth = window.Width
	}
	clipHeight := window.ClipHeight
	if clipHeight == 0 {
		clipHeight = window.Height
	}
	if x >= window.X && y >= window.Y &&
		uint64(x-window.X)+uint64(uint32(width)) <= uint64(clipWidth) &&
		uint64(y-window.Y)+uint64(uint32(height)) <= uint64(clipHeight) {
		geomVisibleWidth = uint8(width)
		geomVisibleHeight = uint8(height)
	} else {
		visibleWidth, visibleHeight, ok := frameWorkClipVisiblePixelsToWindow(window, int(x), int(y), width, height)
		if !ok {
			if frameWorkPlaneBlockStartsBeyondOutput(b.Output, FrameWorkPlaneY, int(x), int(y)) {
				return frameWorkBlockCoeffGeometry{
					window:   window,
					size:     meta.size,
					scanSize: meta.scanSize,
					x:        x,
					y:        y,
					txScale:  meta.txScale,
					plane:    FrameWorkPlaneY,
				}, nil
			}
			return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
		}
		geomVisibleWidth, geomVisibleHeight, ok = frameWorkBlockCoeffVisiblePixels(visibleWidth, visibleHeight)
		if !ok {
			return frameWorkBlockCoeffGeometry{}, ErrInvalidBatch
		}
	}
	return frameWorkBlockCoeffGeometry{
		window:        window,
		size:          meta.size,
		scanSize:      meta.scanSize,
		x:             x,
		y:             y,
		visibleWidth:  geomVisibleWidth,
		visibleHeight: geomVisibleHeight,
		txScale:       meta.txScale,
		plane:         FrameWorkPlaneY,
	}, nil
}

func frameWorkBlockCoeffCoords(x int, y int) (uint32, uint32, bool) {
	if x < 0 || y < 0 || uint64(x) > uint64(^uint32(0)) || uint64(y) > uint64(^uint32(0)) {
		return 0, 0, false
	}
	return uint32(x), uint32(y), true
}

func frameWorkBlockCoeffVisiblePixels(width int, height int) (uint8, uint8, bool) {
	if width <= 0 || height <= 0 || width > int(^uint8(0)) || height > int(^uint8(0)) {
		return 0, 0, false
	}
	return uint8(width), uint8(height), true
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
	if visibleWidth > int(size.Width) || visibleHeight > int(size.Height) {
		return 0, 0, ErrInvalidBatch
	}
	return visibleWidth, visibleHeight, nil
}

func (b *FrameWorkBatch) blockCoeffPlane(tilePlane uint8) (FrameWorkPlane, uint, uint, error) {
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
		visit.MIColEnd <= visit.MICol || visit.MIRowEnd <= visit.MIRow {
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
