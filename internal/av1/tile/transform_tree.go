package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

// TransformTreeRequest describes the AV1 transform-size decisions for one
// decoded block. X4/Y4 are superblock-local 4x4 coordinates.
type TransformTreeRequest struct {
	Size BlockSize

	X4 int
	Y4 int

	VisibleW4 uint8
	VisibleH4 uint8

	Color         parser.ColorConfig
	TransformMode parser.TransformMode

	Inter         bool
	SkipTransform bool
	Lossless      bool
}

// TransformTreeResult stores the luma transform tree and chroma max-transform
// decision for a block. Split masks follow dav1d's tx_split0/tx_split1 layout:
// bit y_off*4+x_off records a split decision at that recursive depth.
type TransformTreeResult struct {
	Y        TransformSize
	UV       TransformSize
	HasUV    bool
	Variable bool

	Split [2]uint16
}

// TransformBlock identifies one transform block emitted by a transform tree.
type TransformBlock struct {
	X4 int
	Y4 int

	Size TransformSize

	VisibleW4 uint8
	VisibleH4 uint8
}

// TransformBlockVisitor receives transform blocks in coefficient decode order.
type TransformBlockVisitor func(TransformBlock) error

// DecodeTransformTree decodes AV1 selected transform-size or inter var-tx
// syntax for one block and updates the caller-owned transform contexts.
func (s *DecodeState) DecodeTransformTree(cdfs *TransformCDFs, ctx *BlockModeContext, req TransformTreeRequest) (TransformTreeResult, error) {
	if s == nil || cdfs == nil || ctx == nil {
		return TransformTreeResult{}, ErrInvalidDecodeState
	}
	blockDims, err := validateTransformTreeRequest(req)
	if err != nil {
		return TransformTreeResult{}, err
	}

	maxY, err := MaxTransformSize(req.Size, parser.ColorConfig{}, 0)
	if err != nil {
		return TransformTreeResult{}, err
	}
	result := TransformTreeResult{Y: maxY}
	if !req.Color.MonoChrome {
		uv, err := MaxTransformSize(req.Size, req.Color, 1)
		if err != nil {
			return TransformTreeResult{}, err
		}
		result.UV = uv
		result.HasUV = true
	}

	if req.Inter && shouldReadVariableTransformTree(req, maxY) {
		result.Variable = true
		walker := transformTreeDecoder{
			state:  s,
			cdfs:   cdfs,
			ctx:    ctx,
			req:    req,
			result: &result,
		}
		maxDims, _ := maxY.Dimensions()
		for y := 0; y < int(req.VisibleH4); y += int(maxDims.H4) {
			for x := 0; x < int(req.VisibleW4); x += int(maxDims.W4) {
				if err := walker.read(maxY, 0, req.X4+x, req.Y4+y, x/int(maxDims.W4), y/int(maxDims.H4)); err != nil {
					return TransformTreeResult{}, err
				}
			}
		}
		return result, nil
	}

	tx, err := s.readFixedTransformSize(cdfs, ctx, req, maxY)
	if err != nil {
		return TransformTreeResult{}, err
	}
	result.Y = tx
	markLog2W, markLog2H, err := fixedTransformMarkContext(req, blockDims, tx)
	if err != nil {
		return TransformTreeResult{}, err
	}
	if err := ctx.MarkTransformArea(req.X4, req.Y4, int(blockDims.W4), int(blockDims.H4),
		markLog2W, markLog2H, !req.Inter); err != nil {
		return TransformTreeResult{}, err
	}
	return result, nil
}

// ForEachLumaTXB replays the decoded luma transform tree in coefficient decode
// order and calls visit for each visible luma transform block.
func (r TransformTreeResult) ForEachLumaTXB(req TransformTreeRequest, visit TransformBlockVisitor) error {
	if visit == nil {
		return ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(req); err != nil {
		return err
	}
	if !r.Y.Valid() {
		return ErrInvalidDecodeState
	}
	if req.SkipTransform {
		return nil
	}

	if r.Variable {
		walker := transformTreeReplay{result: r, req: req, visit: visit}
		dims, _ := r.Y.Dimensions()
		for y := 0; y < int(req.VisibleH4); y += int(dims.H4) {
			for x := 0; x < int(req.VisibleW4); x += int(dims.W4) {
				if err := walker.replay(r.Y, 0, req.X4+x, req.Y4+y, x/int(dims.W4), y/int(dims.H4)); err != nil {
					return err
				}
			}
		}
		return nil
	}

	dims, ok := r.Y.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	for y := 0; y < int(req.VisibleH4); y += int(dims.H4) {
		for x := 0; x < int(req.VisibleW4); x += int(dims.W4) {
			if err := emitTransformBlock(req, req.X4+x, req.Y4+y, r.Y, visit); err != nil {
				return err
			}
		}
	}
	return nil
}

type transformTreeDecoder struct {
	state  *DecodeState
	cdfs   *TransformCDFs
	ctx    *BlockModeContext
	req    TransformTreeRequest
	result *TransformTreeResult
}

func (d *transformTreeDecoder) read(from TransformSize, depth int, x4 int, y4 int, xOff int, yOff int) error {
	dims, ok := from.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	split, err := d.state.ReadTransformPartitionSplit(d.cdfs, d.ctx, TransformPartitionRequest{
		Size:  d.req.Size,
		From:  from,
		Depth: depth,
		X4:    x4,
		Y4:    y4,
	})
	if err != nil {
		return err
	}
	if split {
		d.result.Split[depth] |= 1 << (yOff*4 + xOff)
	}
	if split && dims.Max > 1 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := d.read(sub, depth+1, x4, y4, xOff*2, yOff*2); err != nil {
			return err
		}
		if dims.Log2W >= dims.Log2H && transformChildVisible(d.req, x4+int(subDims.W4), y4) {
			if err := d.read(sub, depth+1, x4+int(subDims.W4), y4, xOff*2+1, yOff*2); err != nil {
				return err
			}
		}
		if dims.Log2H >= dims.Log2W && transformChildVisible(d.req, x4, y4+int(subDims.H4)) {
			if err := d.read(sub, depth+1, x4, y4+int(subDims.H4), xOff*2, yOff*2+1); err != nil {
				return err
			}
			if dims.Log2W >= dims.Log2H && transformChildVisible(d.req, x4+int(subDims.W4), y4+int(subDims.H4)) {
				if err := d.read(sub, depth+1, x4+int(subDims.W4), y4+int(subDims.H4), xOff*2+1, yOff*2+1); err != nil {
					return err
				}
			}
		}
		return nil
	}

	markW, markH := dims.Log2W, dims.Log2H
	if split {
		markW, markH = 0, 0
	}
	return d.ctx.MarkTransformArea(x4, y4, int(dims.W4), int(dims.H4), markW, markH, false)
}

type transformTreeReplay struct {
	result TransformTreeResult
	req    TransformTreeRequest
	visit  TransformBlockVisitor
}

func (r *transformTreeReplay) replay(from TransformSize, depth int, x4 int, y4 int, xOff int, yOff int) error {
	dims, ok := from.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	split := false
	if depth < len(r.result.Split) && yOff < 4 {
		split = r.result.Split[depth]&(1<<(yOff*4+xOff)) != 0
	}
	if split && from > TransformSize4x4 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := r.replay(sub, depth+1, x4, y4, xOff*2, yOff*2); err != nil {
			return err
		}
		if dims.Log2W >= dims.Log2H && transformChildVisible(r.req, x4+int(subDims.W4), y4) {
			if err := r.replay(sub, depth+1, x4+int(subDims.W4), y4, xOff*2+1, yOff*2); err != nil {
				return err
			}
		}
		if dims.Log2H >= dims.Log2W && transformChildVisible(r.req, x4, y4+int(subDims.H4)) {
			if err := r.replay(sub, depth+1, x4, y4+int(subDims.H4), xOff*2, yOff*2+1); err != nil {
				return err
			}
			if dims.Log2W >= dims.Log2H && transformChildVisible(r.req, x4+int(subDims.W4), y4+int(subDims.H4)) {
				if err := r.replay(sub, depth+1, x4+int(subDims.W4), y4+int(subDims.H4), xOff*2+1, yOff*2+1); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return emitTransformBlock(r.req, x4, y4, from, r.visit)
}

func (s *DecodeState) readFixedTransformSize(cdfs *TransformCDFs, ctx *BlockModeContext, req TransformTreeRequest, maxY TransformSize) (TransformSize, error) {
	if req.Lossless || req.TransformMode == parser.TransformMode4x4Only || maxY == TransformSize4x4 {
		return TransformSize4x4, nil
	}
	if req.Inter {
		if req.TransformMode == parser.TransformModeLargest || req.TransformMode == parser.TransformModeSwitchable {
			return maxY, nil
		}
		return 0, ErrInvalidDecodeState
	}
	return s.ReadSelectedTransformSize(cdfs, ctx, SelectedTransformRequest{
		Size:          req.Size,
		TransformMode: req.TransformMode,
		Lossless:      req.Lossless,
		X4:            req.X4,
		Y4:            req.Y4,
	})
}

func fixedTransformMarkContext(req TransformTreeRequest, blockDims BlockDimensions, tx TransformSize) (uint8, uint8, error) {
	if req.Inter && req.SkipTransform {
		return blockDims.Log2W, blockDims.Log2H, nil
	}
	txDims, ok := tx.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidDecodeState
	}
	return txDims.Log2W, txDims.Log2H, nil
}

func shouldReadVariableTransformTree(req TransformTreeRequest, maxY TransformSize) bool {
	if !req.Inter || req.SkipTransform || req.Lossless ||
		req.TransformMode != parser.TransformModeSwitchable ||
		!blockSignalsTransformSize(req.Size) ||
		maxY <= TransformSize4x4 {
		return false
	}
	return true
}

func blockSignalsTransformSize(size BlockSize) bool {
	dims, ok := size.Dimensions()
	return ok && (dims.W4 > 1 || dims.H4 > 1)
}

func transformChildVisible(req TransformTreeRequest, x4 int, y4 int) bool {
	return x4 < req.X4+int(req.VisibleW4) && y4 < req.Y4+int(req.VisibleH4)
}

func emitTransformBlock(req TransformTreeRequest, x4 int, y4 int, size TransformSize, visit TransformBlockVisitor) error {
	dims, ok := size.Dimensions()
	if !ok || !transformChildVisible(req, x4, y4) {
		return ErrInvalidDecodeState
	}
	visibleW := minInt(int(dims.W4), req.X4+int(req.VisibleW4)-x4)
	visibleH := minInt(int(dims.H4), req.Y4+int(req.VisibleH4)-y4)
	if visibleW <= 0 || visibleH <= 0 {
		return ErrInvalidDecodeState
	}
	return visit(TransformBlock{
		X4:        x4,
		Y4:        y4,
		Size:      size,
		VisibleW4: uint8(visibleW),
		VisibleH4: uint8(visibleH),
	})
}

func validateTransformTreeRequest(req TransformTreeRequest) (BlockDimensions, error) {
	dims, ok := req.Size.Dimensions()
	if !ok || req.VisibleW4 == 0 || req.VisibleH4 == 0 ||
		req.VisibleW4 > dims.W4 || req.VisibleH4 > dims.H4 ||
		req.X4 < 0 || req.Y4 < 0 ||
		req.X4+int(dims.W4) > MaxBlockModeSlots ||
		req.Y4+int(dims.H4) > MaxBlockModeSlots {
		return BlockDimensions{}, ErrInvalidDecodeState
	}
	return dims, nil
}
