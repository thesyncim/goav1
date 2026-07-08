// Ported from libaom: av1/common/av1_inv_txfm2d.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

type tx1DType uint8

const (
	tx1DDCT tx1DType = iota
	tx1DADST
	tx1DFlipADST
	tx1DIdentity
)

func (t Type) tx1DTypes() (vertical tx1DType, horizontal tx1DType, ok bool) {
	switch t {
	case TypeDCTDCT:
		return tx1DDCT, tx1DDCT, true
	case TypeADSTDCT:
		return tx1DADST, tx1DDCT, true
	case TypeDCTADST:
		return tx1DDCT, tx1DADST, true
	case TypeADSTADST:
		return tx1DADST, tx1DADST, true
	case TypeFlipADSTDCT:
		return tx1DFlipADST, tx1DDCT, true
	case TypeDCTFlipADST:
		return tx1DDCT, tx1DFlipADST, true
	case TypeFlipADSTFlipADST:
		return tx1DFlipADST, tx1DFlipADST, true
	case TypeADSTFlipADST:
		return tx1DADST, tx1DFlipADST, true
	case TypeFlipADSTADST:
		return tx1DFlipADST, tx1DADST, true
	case TypeIDTX:
		return tx1DIdentity, tx1DIdentity, true
	case TypeVDCT:
		return tx1DDCT, tx1DIdentity, true
	case TypeHDCT:
		return tx1DIdentity, tx1DDCT, true
	case TypeVADST:
		return tx1DADST, tx1DIdentity, true
	case TypeHADST:
		return tx1DIdentity, tx1DADST, true
	case TypeVFlipADST:
		return tx1DFlipADST, tx1DIdentity, true
	case TypeHFlipADST:
		return tx1DIdentity, tx1DFlipADST, true
	default:
		return 0, 0, false
	}
}

func tx1DSupported(t tx1DType, length int) bool {
	switch t {
	case tx1DDCT:
		return dctLengthSupported(length)
	case tx1DADST, tx1DFlipADST:
		return adstLengthSupported(length)
	case tx1DIdentity:
		_, ok := identity1DValue(0, length)
		return ok
	default:
		return false
	}
}

func inverseSeparableBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, typ Type) error {
	return inverseSeparableBlockClamped(dst, dstStride, coeff, coeffStride, scratch, size, typ, minInt16, maxInt16, minInt16, maxInt16)
}

// inverseSeparableBlockClamped is the bit-depth-aware variant of
// inverseSeparableBlock. (rowMin, rowMax) clamp the row-transform input and
// stage outputs; (colMin, colMax) clamp the column path. libaom uses bd+8
// bits for the row clamp and max(bd+6, 16) bits for the column clamp — see
// clamp_buf() and av1_gen_inv_stage_range() in av1_inv_txfm2d.c.
func inverseSeparableBlockClamped(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, typ Type, rowMin int32, rowMax int32, colMin int32, colMax int32) error {
	return inverseSeparableBlockClampedRows(dst, dstStride, coeff, coeffStride, scratch, size, typ, rowMin, rowMax, colMin, colMax, 0)
}

func inverseSeparableBlockClampedRows(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, typ Type, rowMin int32, rowMax int32, colMin int32, colMax int32, activeRows int) error {
	width := int(size.Width)
	height := int(size.Height)
	if dstStride < width || !blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidTransform
	}
	if err := inverseSeparableBlockClampedRowsToScratch(coeff, coeffStride, scratch, size, typ, rowMin, rowMax, colMin, colMax, activeRows, nil); err != nil {
		return err
	}
	narrowStoreImpl(dst, dstStride, scratch, width, height)
	return nil
}

func inverseSeparableBlockClampedRowsToScratch(coeff []int32, coeffStride int, scratch []int32, size Size, typ Type, rowMin int32, rowMax int32, colMin int32, colMax int32, activeRows int, col16 []int16) error {
	// Resolve every per-size datum from a single compact index instead of
	// re-deriving it through size.shift(), adjustedScanSize() and IsRect2(),
	// each of which would recompute sizeIndex on this hot path.
	idx := sizeIndex(size)
	ok := idx >= 0 && sizeValidTable[idx]
	shift := 0
	var coeffSize Size
	if ok {
		shift = int(sizeShiftTable[idx])
		coeffSize = adjustedScanSizeTable[idx]
	}
	vertical, horizontal, okType := typ.tx1DTypes()
	width := int(size.Width)
	height := int(size.Height)
	scratchLen := width * height
	coeffW := int(coeffSize.Width)
	coeffH := int(coeffSize.Height)
	// Validate the transform shape without re-running typ.Supported(size),
	// which would recompute tx1DTypes and the per-axis 1D support checks
	// already implied by okType and the tx1DSupported() calls below.
	if !ok ||
		!okType ||
		typ == TypeIDTX ||
		!tx1DSupported(horizontal, width) ||
		!tx1DSupported(vertical, height) ||
		coeffStride < coeffH ||
		len(scratch) < scratchLen ||
		!coeffBlockFits(len(coeff), coeffStride, coeffW, coeffH) {
		return ErrInvalidTransform
	}
	if activeRows < 0 || activeRows > height {
		return ErrInvalidTransform
	}

	// Reslice scratch to its exact span so the row/column copy loops below
	// index a provably-bounded buffer.
	scratch = scratch[:scratchLen]
	rect2 := size.IsRect2()
	rowLimit := height
	if activeRows > 0 && activeRows < rowLimit {
		rowLimit = activeRows
	}

	// Stage the row inputs into scratch, then run the row pass. The input
	// staging is kept separate from the transform so the transform can batch
	// two already-staged rows through the SIMD-accelerated inverse1DRow2.
	if rowLimit < height {
		rowsToStage := rowLimit
		if rowsToStage > coeffH {
			rowsToStage = coeffH
		}
		if rect2 {
			for row := 0; row < rowsToStage; row++ {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := 0; col < coeffW; col++ {
					tmpLine[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), rowMin, rowMax)
				}
				clear(tmpLine[coeffW:])
			}
		} else {
			for row := 0; row < rowsToStage; row++ {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := 0; col < coeffW; col++ {
					tmpLine[col] = clipRange(int64(coeff[col*coeffStride+row]), rowMin, rowMax)
				}
				clear(tmpLine[coeffW:])
			}
		}
		clear(scratch[rowsToStage*width:])
		rowLimit = rowsToStage
	} else if coeffW == width && coeffH == height {
		if rect2 {
			for row := range height {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := range tmpLine {
					tmpLine[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), rowMin, rowMax)
				}
			}
		} else {
			for row := range height {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := range tmpLine {
					tmpLine[col] = clipRange(int64(coeff[col*coeffStride+row]), rowMin, rowMax)
				}
			}
		}
	} else {
		if rect2 {
			for row := 0; row < coeffH; row++ {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := 0; col < coeffW; col++ {
					tmpLine[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), rowMin, rowMax)
				}
				clear(tmpLine[coeffW:])
			}
		} else {
			for row := 0; row < coeffH; row++ {
				tmpLine := scratch[row*width : row*width+width : row*width+width]
				for col := 0; col < coeffW; col++ {
					tmpLine[col] = clipRange(int64(coeff[col*coeffStride+row]), rowMin, rowMax)
				}
				clear(tmpLine[coeffW:])
			}
		}
		clear(scratch[coeffH*width:])
		rowLimit = coeffH
	}

	// Row pass: transform four staged rows per iteration when a batched
	// kernel exists, then two, falling back to one scalar row at the tail or
	// for lengths without a batched kernel. inverse1DRow4 / inverse1DRow2
	// guarantee the result equals independent inverse1DRow calls. The staging
	// loops above established the four-row kernels' precondition: every input
	// is clamped to [rowMin, rowMax].
	row := 0
	for ; row+3 < rowLimit; row += 4 {
		r0 := scratch[row*width : row*width+width : row*width+width]
		r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
		r2 := scratch[(row+2)*width : (row+2)*width+width : (row+2)*width+width]
		r3 := scratch[(row+3)*width : (row+3)*width+width : (row+3)*width+width]
		inverse1DRow4(r0, r1, r2, r3, width, horizontal, rowMin, rowMax)
	}
	for ; row+1 < rowLimit; row += 2 {
		r0 := scratch[row*width : row*width+width : row*width+width]
		r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
		inverse1DRow2(r0, r1, width, horizontal, rowMin, rowMax)
	}
	if row < rowLimit {
		tmpLine := scratch[row*width : row*width+width : row*width+width]
		inverse1DRow(tmpLine, width, horizontal, rowMin, rowMax)
	}

	// int16 column pipeline (dav1d 8bpc): narrow the int32 row output into the
	// int16 scratch during the round/clamp (free), then run the int16 DCT column
	// pass with no boundary conversion. col16 is only supplied for bitDepth==8
	// blocks whose vertical transform is a DCT.
	if col16 != nil {
		total := width * height
		if rowLimit == height {
			clampRoundNarrowInt16(scratch, col16, shift, colMin, colMax)
		} else {
			clampRoundNarrowInt16(scratch[:rowLimit*width], col16[:rowLimit*width], shift, colMin, colMax)
			for i := rowLimit * width; i < total; i++ {
				col16[i] = 0
			}
		}
		inverseDCTColumnPassInt16(col16, width, height, colMin, colMax)
		return nil
	}

	// Rows beyond rowLimit have already been zeroed; zero is unchanged by the
	// round/shift and lies inside every supported column clamp range.
	if rowLimit == height {
		clampRoundImpl(scratch, shift, colMin, colMax)
	} else {
		clampRoundImpl(scratch[:rowLimit*width], shift, colMin, colMax)
	}

	// Column pass: transform four adjacent columns per iteration when a
	// batched kernel exists (inverse1DCol4 guarantees the result equals four
	// independent inverse1D calls), then two, falling back to one scalar
	// column at the tail or for lengths/types without a batched kernel.
	// The clamp/round pass above plus the zeroed tail rows establish the
	// four-column kernels' precondition: every input is in [colMin, colMax].
	col := 0
	for ; col+3 < width; col += 4 {
		inverse1DCol4(scratch[col:], width, height, vertical, colMin, colMax)
	}
	for ; col+1 < width; col += 2 {
		inverse1DCol2(scratch[col:], width, height, vertical, colMin, colMax)
	}
	if col < width {
		inverse1D(scratch[col:], width, height, vertical, colMin, colMax)
	}

	return nil
}

// clampRoundPureGo is the portable mid-pass round and clamp.
func clampRoundPureGo(scratch []int32, shift int, min, max int32) {
	if shift > 0 {
		for i := range scratch {
			scratch[i] = clipRange(roundShift(int64(scratch[i]), shift), min, max)
		}
	} else {
		for i := range scratch {
			scratch[i] = clipRange(int64(scratch[i]), min, max)
		}
	}
}

// narrowStorePureGo is the portable final store: every value is
// rounding-shifted right by four and saturated to int16.
func narrowStorePureGo(dst []int16, dstStride int, scratch []int32, width, height int) {
	for row := range height {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		tmpLine := scratch[row*width : row*width+width : row*width+width]
		for col, v := range tmpLine {
			// roundShift(int64(v), 4) is in [-2^27, 2^27), so it always fits an
			// int32 and the surrounding clipInt32 is a no-op; cast directly and
			// clamp once to int16, matching libaom's clip_pixel exactly.
			dstLine[col] = clipInt16(int32(roundShift(int64(v), 4)))
		}
	}
}

func inverse1D(c []int32, stride int, length int, typ tx1DType, min int32, max int32) {
	switch typ {
	case tx1DDCT:
		inverseDCT1D(c, stride, length, min, max)
	case tx1DADST:
		inverseADST1D(c, stride, length, min, max)
	case tx1DFlipADST:
		inverseFlipADST1D(c, stride, length, min, max)
	case tx1DIdentity:
		inverseIdentity1D(c, stride, length)
	}
}

func inverseIdentity1D(c []int32, stride int, length int) {
	for i := range length {
		c[i*stride], _ = identity1DValue(c[i*stride], length)
	}
}
