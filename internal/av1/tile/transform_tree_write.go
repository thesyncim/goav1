package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// WriteTransformTree codes the AV1 transform-size syntax for one block, the
// forward of DecodeTransformTree. For switchable non-skip inter blocks it
// walks the var-tx tree selected by tree.Split and writes one txfm_partition
// symbol per visited node; for switchable intra blocks it writes the selected
// transform-size depth symbol for tree.Y; all other cases are symbol-free.
// Every path applies the same MarkTransformArea context updates as the
// decoder so later tx-size contexts stay in sync.
func WriteTransformTree(w *entropy.Writer, cdfs *TransformCDFs, ctx *BlockModeContext, req TransformTreeRequest, tree TransformTreeResult) error {
	if w == nil || cdfs == nil || ctx == nil {
		return ErrInvalidDecodeState
	}
	blockDims, err := validateTransformTreeRequest(req)
	if err != nil {
		return err
	}
	maxY, err := MaxTransformSize(req.Size, parser.ColorConfig{}, 0)
	if err != nil {
		return err
	}

	if req.Inter && shouldReadVariableTransformTree(req, maxY) {
		walker := transformTreeWriter{
			w:    w,
			cdfs: cdfs,
			ctx:  ctx,
			req:  req,
			tree: tree,
		}
		maxDims, _ := maxY.Dimensions()
		reqX4 := int(req.X4)
		reqY4 := int(req.Y4)
		for y := 0; y < int(req.VisibleH4); y += int(maxDims.H4) {
			for x := 0; x < int(req.VisibleW4); x += int(maxDims.W4) {
				// Same availability rule as the decode walk: past the first
				// row/column of maxY units the prior sibling has marked the
				// above/left contexts.
				haveTop := req.HaveTop || y > 0
				haveLeft := req.HaveLeft || x > 0
				if err := walker.write(maxY, 0, reqX4+x, reqY4+y, x/int(maxDims.W4), y/int(maxDims.H4), haveTop, haveLeft); err != nil {
					return err
				}
			}
		}
		return nil
	}

	tx, err := writeFixedTransformSize(w, cdfs, ctx, req, maxY, tree)
	if err != nil {
		return err
	}
	markLog2W, markLog2H, err := fixedTransformMarkContext(req, blockDims, tx)
	if err != nil {
		return err
	}
	return ctx.MarkTransformArea(int(req.X4), int(req.Y4), int(blockDims.W4), int(blockDims.H4),
		markLog2W, markLog2H, !req.Inter)
}

type transformTreeWriter struct {
	w    *entropy.Writer
	cdfs *TransformCDFs
	ctx  *BlockModeContext
	req  TransformTreeRequest
	tree TransformTreeResult
}

// write mirrors transformTreeDecoder.read: same symbol gating, recursion
// order, availability propagation, and context marks, with the split decision
// taken from the caller's tree.Split masks instead of the bitstream.
func (d *transformTreeWriter) write(from TransformSize, depth int, x4 int, y4 int, xOff int, yOff int, haveTop bool, haveLeft bool) error {
	dims, ok := from.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	split := false
	if depth < len(d.tree.Split) && yOff < 4 {
		split = d.tree.Split[depth]&(1<<(yOff*4+xOff)) != 0
	}
	if depth >= 2 || from <= TransformSize4x4 {
		// The decoder reads no symbol here and derives no-split.
		if split {
			return ErrInvalidDecodeState
		}
	} else {
		category, context, err := d.ctx.TransformPartitionContext(TransformPartitionRequest{
			Size:     d.req.Size,
			From:     from,
			Depth:    uint8(depth),
			X4:       uint8(x4),
			Y4:       uint8(y4),
			HaveTop:  haveTop,
			HaveLeft: haveLeft,
		})
		if err != nil {
			return err
		}
		cdf, err := d.cdfs.TxPartitionCDF(category, context)
		if err != nil {
			return err
		}
		var symbol int
		if split {
			symbol = 1
		}
		d.w.WriteCDF(cdf, symbol)
	}
	if split && dims.Max > 1 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := d.write(sub, depth+1, x4, y4, xOff*2, yOff*2, haveTop, haveLeft); err != nil {
			return err
		}
		if dims.Log2W >= dims.Log2H && transformChildVisible(d.req, x4+int(subDims.W4), y4) {
			if err := d.write(sub, depth+1, x4+int(subDims.W4), y4, xOff*2+1, yOff*2, haveTop, true); err != nil {
				return err
			}
		}
		if dims.Log2H >= dims.Log2W && transformChildVisible(d.req, x4, y4+int(subDims.H4)) {
			if err := d.write(sub, depth+1, x4, y4+int(subDims.H4), xOff*2, yOff*2+1, true, haveLeft); err != nil {
				return err
			}
			if dims.Log2W >= dims.Log2H && transformChildVisible(d.req, x4+int(subDims.W4), y4+int(subDims.H4)) {
				if err := d.write(sub, depth+1, x4+int(subDims.W4), y4+int(subDims.H4), xOff*2+1, yOff*2+1, true, true); err != nil {
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

// writeFixedTransformSize mirrors readFixedTransformSize: only the switchable
// intra case codes a symbol (the depth of tree.Y below the block's max
// transform size).
func writeFixedTransformSize(w *entropy.Writer, cdfs *TransformCDFs, ctx *BlockModeContext, req TransformTreeRequest, maxY TransformSize, tree TransformTreeResult) (TransformSize, error) {
	if req.Lossless || req.TransformMode == parser.TransformMode4x4Only || maxY == TransformSize4x4 {
		return TransformSize4x4, nil
	}
	if req.Inter {
		if req.TransformMode == parser.TransformModeLargest || req.TransformMode == parser.TransformModeSwitchable {
			return maxY, nil
		}
		return 0, ErrInvalidDecodeState
	}
	if req.TransformMode == parser.TransformModeLargest {
		return maxY, nil
	}
	if req.TransformMode != parser.TransformModeSwitchable {
		return 0, ErrInvalidDecodeState
	}

	dims, ok := maxY.Dimensions()
	if !ok || dims.Max == 0 {
		return 0, ErrInvalidDecodeState
	}
	depth := 0
	size := maxY
	for size != tree.Y {
		sd, ok := size.Dimensions()
		if !ok || sd.Sub == size {
			return 0, ErrInvalidDecodeState
		}
		size = sd.Sub
		depth++
	}
	if limit := min(int(dims.Max), 2); depth > limit {
		return 0, ErrInvalidDecodeState
	}
	context, err := ctx.SelectedTransformContextWithAvailability(maxY, int(req.X4), int(req.Y4), req.HaveTop, req.HaveLeft)
	if err != nil {
		return 0, err
	}
	cdf, err := cdfs.TxSizeCDF(int(dims.Max)-1, context)
	if err != nil {
		return 0, err
	}
	w.WriteCDF(cdf, depth)
	return tree.Y, nil
}
