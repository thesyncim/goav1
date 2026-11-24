package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TransformTreeRequest describes the AV1 transform-size decisions for one
// decoded block. X4/Y4 are superblock-local 4x4 coordinates.
type TransformTreeRequest struct {
	Size BlockSize

	X4 uint8
	Y4 uint8

	VisibleW4 uint8
	VisibleH4 uint8
	HaveTop   bool
	HaveLeft  bool

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
	X4        uint8
	Y4        uint8
	Size      TransformSize
	VisibleW4 uint8
	VisibleH4 uint8
}

// TransformBlockVisitor receives transform blocks in coefficient decode order.
type TransformBlockVisitor func(TransformBlock) error

// DecodeTransformTree decodes AV1 selected transform-size or inter var-tx
// syntax for one block described by req and updates the caller-owned
// transform contexts. The block decode reads the sizes through readTxSizes
// directly from its block visit.
func (s *DecodeState) DecodeTransformTree(cdfs *TransformCDFs, ctx *BlockModeContext, req *TransformTreeRequest) (TransformTreeResult, error) {
	if s == nil || cdfs == nil || ctx == nil {
		return TransformTreeResult{}, ErrInvalidDecodeState
	}
	block := BlockVisit{
		X4: req.X4, Y4: req.Y4, Size: req.Size,
		VisibleW4: req.VisibleW4, VisibleH4: req.VisibleH4,
		HaveTop: req.HaveTop, HaveLeft: req.HaveLeft,
	}
	var result TransformTreeResult
	if err := s.readTxSizes(cdfs, ctx, &block, &req.Color, req.TransformMode, req.Inter, req.SkipTransform, req.Lossless, &result); err != nil {
		return TransformTreeResult{}, err
	}
	return result, nil
}

// uvTxLayoutColumn is dav1d's f->cur.p.layout column of
// dav1d_max_txfm_size_for_bs: 1 for 4:2:0, 2 for 4:2:2, 3 for 4:4:4.
func uvTxLayoutColumn(color *parser.ColorConfig) int {
	if !color.SubsamplingX {
		return 3
	}
	if color.SubsamplingY {
		return 1
	}
	return 2
}

// readTxSizes is the transform-size step of dav1d's decode_b: read_vartx_tree
// (src/decode.c) for inter blocks and the read_tx_size depth read for intra
// blocks. max_ytx and uvtx come straight from dav1d_max_txfm_size_for_bs, the
// var-tx tree is read per max_ytx unit by entropy's DecodeTxTree, and the fixed-size paths
// set the transform contexts with dav1d_memset_pow2 stores. The result is
// written in place into out.
func (s *DecodeState) readTxSizes(cdfs *TransformCDFs, ctx *BlockModeContext, block *BlockVisit, color *parser.ColorConfig, mode parser.TransformMode, inter, skip, lossless bool, out *TransformTreeResult) error {
	bs := block.Size
	if bs >= blockSizeCount {
		return ErrInvalidDecodeState
	}
	bDim := &blockDimensions[bs]
	bx4, by4 := int(block.X4), int(block.Y4)
	vw4, vh4 := int(block.VisibleW4), int(block.VisibleH4)
	if vw4 == 0 || vh4 == 0 || vw4 > int(bDim.W4) || vh4 > int(bDim.H4) ||
		bx4+int(bDim.W4) > MaxBlockModeSlots || by4+int(bDim.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	maxTx := &maxTransformSizeForBlock[bs]
	maxYtx := maxTx[0]
	*out = TransformTreeResult{Y: maxYtx}
	if !color.MonoChrome {
		out.HasUV = true
		if !lossless {
			out.UV = maxTx[uvTxLayoutColumn(color)]
		}
	}

	if inter && !skip && !lossless && mode == parser.TransformModeSwitchable && maxYtx != TransformSize4x4 {
		// read_vartx_tree: one read_tx_tree per max_ytx unit. A unit past the
		// first row/column has its prior sibling as top/left neighbour.
		out.Variable = true
		t := entropy.TxTree{CDF: &cdfs.TxPartition, A: &ctx.A.Tx, L: &ctx.L.Tx, Right: bx4 + vw4, Bottom: by4 + vh4}
		// The symbols are read straight from the tile's range decoder, as
		// dav1d reads through &ts->msac.
		c := (*entropy.Cursor)(&s.Reader)
		c.DecodeVarTxTree(&t, uint(maxYtx), bx4, by4, vw4, vh4, block.HaveTop, block.HaveLeft)
		out.Split = t.Split
		return nil
	}

	tx := maxYtx
	switch {
	case lossless || mode == parser.TransformMode4x4Only || maxYtx == TransformSize4x4:
		tx = TransformSize4x4
	case inter:
		if mode != parser.TransformModeLargest && mode != parser.TransformModeSwitchable {
			return ErrInvalidDecodeState
		}
	default:
		var err error
		tx, err = s.ReadSelectedTransformSize(cdfs, ctx, SelectedTransformRequest{
			Size:          bs,
			TransformMode: mode,
			Lossless:      lossless,
			X4:            block.X4,
			Y4:            block.Y4,
			HaveTop:       block.HaveTop,
			HaveLeft:      block.HaveLeft,
		})
		if err != nil {
			return err
		}
	}
	out.Y = tx
	// An inter skip block marks its whole extent; otherwise the context holds
	// the transform size over the block's span.
	ctxW, ctxH := bDim.Log2W, bDim.Log2H
	if !inter || !skip {
		td := tx.Dims()
		ctxW, ctxH = td.Log2W, td.Log2H
	}
	setCtxPow2(ctx.A.Tx[bx4:], bDim.Log2W, ctxW)
	setCtxPow2(ctx.L.Tx[by4:], bDim.Log2H, ctxH)
	if !inter {
		setCtxPow2(ctx.A.TxIntra[bx4:], bDim.Log2W, ctxW)
		setCtxPow2(ctx.L.TxIntra[by4:], bDim.Log2H, ctxH)
	}
	return nil
}

// ForEachLumaTXB replays the decoded luma transform tree in coefficient decode
// order and calls visit for each visible luma transform block.
func (r TransformTreeResult) ForEachLumaTXB(req TransformTreeRequest, visit TransformBlockVisitor) error {
	return r.forEachLumaTXBInWindow(req, fullCoeffUnitWindow(req), visit)
}

// coeffUnitWindow restricts a transform-tree replay to one 64x64 max-unit of a
// prediction block, expressed in superblock-local luma 4x4 coordinates. libaom
// (av1/decoder/decodeframe.c decode_token_recon_block) and the spec 5.11.34
// residual() split prediction blocks larger than 64px into BLOCK_64X64 units and
// interleave luma+chroma coefficient decode per unit; the window selects one
// such unit. For blocks <= 64px the window spans the whole block.
type coeffUnitWindow struct {
	X4Start uint8
	Y4Start uint8
	X4End   uint8
	Y4End   uint8
}

// fullCoeffUnitWindow returns the window spanning the whole prediction block,
// reproducing the unwindowed transform-tree replay.
func fullCoeffUnitWindow(req TransformTreeRequest) coeffUnitWindow {
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	return coeffUnitWindow{
		X4Start: uint8(reqX4),
		Y4Start: uint8(reqY4),
		X4End:   uint8(reqX4 + int(req.VisibleW4)),
		Y4End:   uint8(reqY4 + int(req.VisibleH4)),
	}
}

// forEachLumaTXBInWindow replays the luma transform tree but only emits
// transform blocks whose top-left lies inside the given 64x64 unit window. The
// step and per-block emission are identical to the unwindowed walk; only the
// loop bounds are clamped to the window, so for a block that is itself a single
// unit the call sequence matches ForEachLumaTXB exactly.
func (r TransformTreeResult) forEachLumaTXBInWindow(req TransformTreeRequest, window coeffUnitWindow, visit TransformBlockVisitor) error {
	if visit == nil {
		return ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(&req); err != nil {
		return err
	}
	if !r.Y.Valid() {
		return ErrInvalidDecodeState
	}
	if req.SkipTransform {
		return nil
	}

	dims, ok := r.Y.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	yStart := maxInt(int(window.Y4Start), reqY4)
	yEnd := minInt(int(window.Y4End), reqY4+int(req.VisibleH4))
	xStart := maxInt(int(window.X4Start), reqX4)
	xEnd := minInt(int(window.X4End), reqX4+int(req.VisibleW4))

	if r.Variable {
		var blocks [lumaTXBWindowCap]TransformBlock
		n, err := r.collectLumaTXBsInWindow(req, window, &blocks)
		if err == nil {
			for _, block := range blocks[:n] {
				if err := visit(block); err != nil {
					return err
				}
			}
			return nil
		}
		if err != errTXBWindowFull {
			return err
		}
		walker := transformTreeReplay{result: r, req: req, visit: visit}
		for y := yStart; y < yEnd; y += int(dims.H4) {
			for x := xStart; x < xEnd; x += int(dims.W4) {
				xOff := (x - reqX4) / int(dims.W4)
				yOff := (y - reqY4) / int(dims.H4)
				if err := walker.replay(r.Y, 0, x, y, xOff, yOff); err != nil {
					return err
				}
			}
		}
		return nil
	}

	for y := yStart; y < yEnd; y += int(dims.H4) {
		for x := xStart; x < xEnd; x += int(dims.W4) {
			if err := emitTransformBlock(req, x, y, r.Y, visit); err != nil {
				return err
			}
		}
	}
	return nil
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

func fixedTransformMarkContext(req *TransformTreeRequest, blockDims BlockDimensions, tx TransformSize) (uint8, uint8, error) {
	if req.Inter && req.SkipTransform {
		return blockDims.Log2W, blockDims.Log2H, nil
	}
	txDims, ok := tx.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidDecodeState
	}
	return txDims.Log2W, txDims.Log2H, nil
}

func shouldReadVariableTransformTree(req *TransformTreeRequest, maxY TransformSize) bool {
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
	return x4 < int(req.X4)+int(req.VisibleW4) && y4 < int(req.Y4)+int(req.VisibleH4)
}

func emitTransformBlock(req TransformTreeRequest, x4 int, y4 int, size TransformSize, visit TransformBlockVisitor) error {
	block, err := transformBlockAt(req, x4, y4, size)
	if err != nil {
		return err
	}
	return visit(block)
}

// transformBlockAt builds the transform block of the given size at x4/y4,
// clipped to the visible part of the request.
func transformBlockAt(req TransformTreeRequest, x4 int, y4 int, size TransformSize) (TransformBlock, error) {
	dims, ok := size.Dimensions()
	if !ok || !transformChildVisible(req, x4, y4) ||
		x4 < 0 || y4 < 0 || x4 > int(^uint8(0)) || y4 > int(^uint8(0)) {
		return TransformBlock{}, ErrInvalidDecodeState
	}
	visibleW := minInt(int(dims.W4), int(req.X4)+int(req.VisibleW4)-x4)
	visibleH := minInt(int(dims.H4), int(req.Y4)+int(req.VisibleH4)-y4)
	if visibleW <= 0 || visibleH <= 0 {
		return TransformBlock{}, ErrInvalidDecodeState
	}
	return TransformBlock{
		X4:        uint8(x4),
		Y4:        uint8(y4),
		Size:      size,
		VisibleW4: uint8(visibleW),
		VisibleH4: uint8(visibleH),
	}, nil
}

func validateTransformTreeRequest(req *TransformTreeRequest) (BlockDimensions, error) {
	dims, ok := req.Size.Dimensions()
	if !ok || req.VisibleW4 == 0 || req.VisibleH4 == 0 ||
		req.VisibleW4 > dims.W4 || req.VisibleH4 > dims.H4 ||
		int(req.X4)+int(dims.W4) > MaxBlockModeSlots ||
		int(req.Y4)+int(dims.H4) > MaxBlockModeSlots {
		return BlockDimensions{}, ErrInvalidDecodeState
	}
	return dims, nil
}
