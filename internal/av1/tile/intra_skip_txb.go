package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

// skipIntraTXBWalk enumerates one all-zero transform-block visit per luma and
// chroma transform block of an intra block coded with skip=1.
//
// An intra block reads its transform size even when skip is set (spec
// read_block_tx_size: allowSelect = !skip || !is_inter), and dav1d's
// recon_b_intra (src/recon_tmpl.c) predicts every intra block transform block
// by transform block, gating only the residual add on !b->skip. Predicting a
// skip intra block in one whole-block call is therefore wrong whenever the
// selected luma transform is smaller than the block (or the block is larger
// than 64x64): DC/smooth/directional/filter-intra edges come from the
// neighbouring, already reconstructed transform blocks of the same block.
//
// The walk mirrors the non-skip residual order: the block is split into 64x64
// luma max-units, and within each unit the luma transform blocks are visited
// in raster order, followed by the U and V transform blocks of that unit. Every
// visit carries an AllZero result and no coefficients, so the reconstruction
// side predicts the transform block and adds nothing, exactly like a coded
// block whose transform blocks all have zero coded-block flags.
//
// It is a plain iterator (no callbacks) so the hot coefficient walk does not
// heap-allocate a closure.
type skipIntraTXBWalk struct {
	y, uv      TransformSize
	yW4, yH4   int
	uvW4, uvH4 int

	reqX4, reqY4 int
	visW, visH   int
	muW, muH     int

	hasChroma      bool
	ssX, ssY       int
	cx4, cy4       int
	cVisW4, cVisH4 int

	// cursor
	row, col   int
	plane      uint8
	x, y4      int
	xEnd, yEnd int
	xStart     int
	started    bool
	done       bool
}

// init prepares the walk; it reports false (with ok=true) when the block is
// not a skip intra block and nothing is to be visited.
func (w *skipIntraTXBWalk) init(req *TransformTreeRequest, tree *TransformTreeResult) (bool, error) {
	*w = skipIntraTXBWalk{}
	if req.Inter || !req.SkipTransform {
		return false, nil
	}
	yDims, ok := tree.Y.Dimensions()
	if !ok || tree.Variable {
		return false, ErrInvalidDecodeState
	}
	w.y = tree.Y
	w.yW4, w.yH4 = int(yDims.W4), int(yDims.H4)
	w.reqX4, w.reqY4 = int(req.X4), int(req.Y4)
	w.visW, w.visH = int(req.VisibleW4), int(req.VisibleH4)
	if w.visW <= 0 || w.visH <= 0 {
		return false, ErrInvalidDecodeState
	}
	const maxUnit4 = 16
	w.muW = minInt(maxUnit4, w.visW)
	w.muH = minInt(maxUnit4, w.visH)
	w.hasChroma = tree.HasUV && !req.Color.MonoChrome && HasChromaBlock(*req, req.Color)
	if w.hasChroma {
		uvDims, ok := tree.UV.Dimensions()
		if !ok {
			return false, ErrInvalidDecodeState
		}
		w.uv = tree.UV
		w.uvW4, w.uvH4 = int(uvDims.W4), int(uvDims.H4)
		w.ssX = int(boolToShift(req.Color.SubsamplingX))
		w.ssY = int(boolToShift(req.Color.SubsamplingY))
		w.cx4 = w.reqX4 >> w.ssX
		w.cy4 = w.reqY4 >> w.ssY
		w.cVisW4 = ((w.reqX4 + w.visW + w.ssX) >> w.ssX) - w.cx4
		w.cVisH4 = ((w.reqY4 + w.visH + w.ssY) >> w.ssY) - w.cy4
		if w.cVisW4 <= 0 || w.cVisH4 <= 0 {
			return false, ErrInvalidDecodeState
		}
	}
	return true, nil
}

// setPlane positions the cursor at the first transform block of plane within
// the current max-unit and reports whether that plane has any.
func (w *skipIntraTXBWalk) setPlane(plane uint8) bool {
	w.plane = plane
	rowEnd := minInt(w.row+w.muH, w.visH)
	colEnd := minInt(w.col+w.muW, w.visW)
	if plane == 0 {
		w.xStart, w.y4 = w.col, w.row
		w.xEnd, w.yEnd = colEnd, rowEnd
	} else {
		w.xStart, w.y4 = w.col>>w.ssX, w.row>>w.ssY
		w.xEnd = minInt((colEnd+w.ssX)>>w.ssX, w.cVisW4)
		w.yEnd = minInt((rowEnd+w.ssY)>>w.ssY, w.cVisH4)
	}
	w.x = w.xStart
	return w.x < w.xEnd && w.y4 < w.yEnd
}

// advanceUnit moves to the next plane or max-unit; it reports false at the end.
func (w *skipIntraTXBWalk) advanceUnit() bool {
	for {
		if w.hasChroma && w.plane < 2 {
			if w.setPlane(w.plane + 1) {
				return true
			}
			continue
		}
		w.col += w.muW
		if w.col >= w.visW {
			w.col = 0
			w.row += w.muH
			if w.row >= w.visH {
				return false
			}
		}
		if w.setPlane(0) {
			return true
		}
	}
}

// next fills out with the next all-zero transform-block visit.
func (w *skipIntraTXBWalk) next(out *BlockCoeffBlock) bool {
	if w.done {
		return false
	}
	if !w.started {
		w.started = true
		if !w.setPlane(0) && !w.advanceUnit() {
			w.done = true
			return false
		}
	} else {
		stepW, stepH := w.yW4, w.yH4
		if w.plane != 0 {
			stepW, stepH = w.uvW4, w.uvH4
		}
		w.x += stepW
		if w.x >= w.xEnd {
			w.x = w.xStart
			w.y4 += stepH
		}
		if w.y4 >= w.yEnd && !w.advanceUnit() {
			w.done = true
			return false
		}
	}
	if w.plane == 0 {
		*out = BlockCoeffBlock{
			Plane: 0,
			Block: TransformBlock{
				X4:        uint8(w.reqX4 + w.x),
				Y4:        uint8(w.reqY4 + w.y4),
				Size:      w.y,
				VisibleW4: uint8(minInt(w.yW4, w.visW-w.x)),
				VisibleH4: uint8(minInt(w.yH4, w.visH-w.y4)),
			},
			Transform: transform.TypeDCTDCT,
			Result:    TXBDecodeResult{AllZero: true},
		}
		return true
	}
	*out = BlockCoeffBlock{
		Plane: w.plane,
		Block: TransformBlock{
			X4:        uint8(w.cx4 + w.x),
			Y4:        uint8(w.cy4 + w.y4),
			Size:      w.uv,
			VisibleW4: uint8(minInt(w.uvW4, w.cVisW4-w.x)),
			VisibleH4: uint8(minInt(w.uvH4, w.cVisH4-w.y4)),
		},
		Transform: transform.TypeDCTDCT,
		Result:    TXBDecodeResult{AllZero: true},
	}
	return true
}
