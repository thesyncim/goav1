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
	if _, err := inverseSeparableBlockClampedRowsToScratch(coeff, coeffStride, scratch, size, typ, rowMin, rowMax, colMin, colMax, activeRows, 0, nil, false); err != nil {
		return err
	}
	narrowStoreImpl(dst, dstStride, scratch, width, height)
	return nil
}

// stageTransposeClampScalar stages the rows x cols rectangle of the
// column-major coefficient block into row-major scratch lines of length
// width, clamping every value to [lo, hi]; rect2 applies the rectangular-
// transform sqrt2 scale before the clamp. Only the rectangle is written —
// line tails and trailing lines are the caller's to clear.
func stageTransposeClampScalar(scratch []int32, width int, coeff []int32, coeffStride int, rows int, cols int, rect2 bool, lo int32, hi int32) {
	if rect2 {
		for row := 0; row < rows; row++ {
			tmpLine := scratch[row*width : row*width+cols : row*width+cols]
			for col := range tmpLine {
				tmpLine[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), lo, hi)
			}
		}
	} else {
		for row := 0; row < rows; row++ {
			tmpLine := scratch[row*width : row*width+cols : row*width+cols]
			for col := range tmpLine {
				tmpLine[col] = clipRange(int64(coeff[col*coeffStride+row]), lo, hi)
			}
		}
	}
}

func inverseSeparableBlockClampedRowsToScratch(coeff []int32, coeffStride int, scratch []int32, size Size, typ Type, rowMin int32, rowMax int32, colMin int32, colMax int32, activeRows int, activeCols int, col16 []int16, col16AliasesScratch bool) (bool, error) {
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
		return false, ErrInvalidTransform
	}
	if activeRows < 0 || activeRows > height {
		return false, ErrInvalidTransform
	}
	if col16 != nil && len(col16) < scratchLen {
		return false, ErrInvalidTransform
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
	// stageTransposeClamp writes only the rows x cols rectangle, so each
	// branch clears its line tails and trailing lines exactly as before.
	if rowLimit < height {
		rowsToStage := rowLimit
		if rowsToStage > coeffH {
			rowsToStage = coeffH
		}
		stageTransposeClamp(scratch, width, coeff, coeffStride, rowsToStage, coeffW, rect2, rowMin, rowMax)
		if coeffW < width {
			for row := 0; row < rowsToStage; row++ {
				clear(scratch[row*width+coeffW : (row+1)*width])
			}
		}
		clear(scratch[rowsToStage*width:])
		rowLimit = rowsToStage
	} else if coeffW == width && coeffH == height {
		stageTransposeClamp(scratch, width, coeff, coeffStride, height, width, rect2, rowMin, rowMax)
	} else {
		stageTransposeClamp(scratch, width, coeff, coeffStride, coeffH, coeffW, rect2, rowMin, rowMax)
		if coeffW < width {
			for row := 0; row < coeffH; row++ {
				clear(scratch[row*width+coeffW : (row+1)*width])
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
	// dav1d's eob-bounded sub-sizes: when every staged row's nonzero inputs
	// lie in its first 8 or 16 entries, the DCT32 row pass runs the kernels
	// specialized for that many inputs.
	lowN := 0
	if width == dct32Size && horizontal == tx1DDCT && activeCols > 0 && activeCols <= 16 && inverseDCT32Row4LowImpl != nil {
		lowN = 16
		if activeCols <= 8 {
			lowN = 8
		}
	}
	row := 0
	if lowN != 0 {
		for ; row+3 < rowLimit; row += 4 {
			r0 := scratch[row*width : row*width+width : row*width+width]
			r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
			r2 := scratch[(row+2)*width : (row+2)*width+width : (row+2)*width+width]
			r3 := scratch[(row+3)*width : (row+3)*width+width : (row+3)*width+width]
			inverseDCT32Row4LowImpl(r0, r1, r2, r3, lowN, rowMin, rowMax)
		}
		for ; row < rowLimit; row += 2 {
			// A lone last row pairs with the zero row after it (height >= 8).
			r0 := scratch[row*width : row*width+width : row*width+width]
			r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
			inverseDCT32Row2LowImpl(r0, r1, lowN, rowMin, rowMax)
		}
		row = rowLimit
	} else if width == dct64Size && horizontal == tx1DDCT && inverseDCT64Row4LowImpl != nil {
		// AV1 codes at most 32 DCT64 inputs; activeCols narrows that further.
		low64 := 32
		if activeCols > 0 && activeCols < 32 {
			low64 = activeCols
		}
		for ; row < rowLimit; row += 4 {
			// Rows past rowLimit are zero; heights of 64-wide blocks are >= 16.
			r0 := scratch[row*width : row*width+width : row*width+width]
			r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
			r2 := scratch[(row+2)*width : (row+2)*width+width : (row+2)*width+width]
			r3 := scratch[(row+3)*width : (row+3)*width+width : (row+3)*width+width]
			inverseDCT64Row4LowImpl(r0, r1, r2, r3, low64, rowMin, rowMax)
		}
		row = rowLimit
	} else if width == dct16Size && horizontal == tx1DDCT && activeCols > 0 && activeCols <= 8 && inverseDCT16Row2LowImpl != nil {
		low16 := 8
		if activeCols <= 4 {
			low16 = 4
		}
		for ; row < rowLimit; row += 2 {
			r0 := scratch[row*width : row*width+width : row*width+width]
			r1 := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
			inverseDCT16Row2LowImpl(r0, r1, low16, rowMin, rowMax)
		}
		row = rowLimit
	}
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
		if row+1 < height {
			// Rows past rowLimit are zero and transform to zero, so pairing the
			// last active row with the next one keeps it on the batched kernel
			// (dav1d never drops to a scalar row pass).
			next := scratch[(row+1)*width : (row+1)*width+width : (row+1)*width+width]
			inverse1DRow2(tmpLine, next, width, horizontal, rowMin, rowMax)
		} else {
			inverse1DRow(tmpLine, width, horizontal, rowMin, rowMax)
		}
	}

	// int16 column pipeline (dav1d 8bpc): narrow the int32 row output into the
	// int16 scratch during the round/clamp (free), then run the int16 DCT column
	// pass with no boundary conversion. col16 is only supplied for bitDepth==8
	// blocks whose vertical transform is a DCT.
	midPassClamped := false
	if col16 != nil {
		total := width * height
		if rowLimit == height {
			clampRoundNarrowInt16Impl(scratch, col16, shift, colMin, colMax)
		} else {
			clampRoundNarrowInt16Impl(scratch[:rowLimit*width], col16[:rowLimit*width], shift, colMin, colMax)
			for i := rowLimit * width; i < total; i++ {
				col16[i] = 0
			}
		}
		// Scalar int16 kernels use exact wide intermediates and support the full
		// int16 range. SIMD dispatch also checks the certified bounds for wider
		// transforms and the measured DCT8 profitability cutoff.
		if !int16ColumnFast || int16ColumnSIMDInputEligible(col16, width, height, colMin, colMax) {
			inverseDCTColumnPassInt16(col16, width, height, colMin, colMax)
			return true, nil
		}
		if col16AliasesScratch {
			// The in-place narrow already applied the mid-pass round/clamp. Restore
			// those values to int32 before continuing through the exact int32 column
			// path. Descending order preserves unread packed int16 lanes.
			for i := total - 1; i >= 0; i-- {
				scratch[i] = int32(col16[i])
			}
			midPassClamped = true
		}
	}

	// Rows beyond rowLimit have already been zeroed; zero is unchanged by the
	// round/shift and lies inside every supported column clamp range.
	if !midPassClamped {
		if rowLimit == height {
			clampRoundImpl(scratch, shift, colMin, colMax)
		} else {
			clampRoundImpl(scratch[:rowLimit*width], shift, colMin, colMax)
		}
	}

	// int16 DCT16 column pass (dav1d 8bpc form) over the raw scratch: exact
	// under its per-column guard, else the int32 kernels below run.
	if itx16RawColumnPass != nil && height == dct16Size && vertical == tx1DDCT &&
		colMin == minInt16 && colMax == maxInt16 && width >= 8 && width&7 == 0 {
		itx16RawColumnPass(scratch, width, rowLimit, colMin, colMax)
		return false, nil
	}

	// Low-row DCT32 SIMD specialization. Staging and the horizontal pass have
	// left rows rowLimit..31 at exact zero; the kernel derives every output row
	// from rows 0..7 and preserves the full transform's clamp graph.
	if inverseDCT32Low8Col4Impl != nil &&
		rowLimit > 0 && rowLimit <= 8 && height == dct32Size && vertical == tx1DDCT &&
		colMin == minInt16 && colMax == maxInt16 && width >= 4 && width%4 == 0 {
		for col := 0; col < width; col += 4 {
			inverseDCT32Low8Col4Impl(scratch[col:], width, colMin, colMax)
		}
		return false, nil
	}
	if inverseDCT32Low16Col4Impl != nil &&
		rowLimit > 8 && rowLimit <= 16 && height == dct32Size && vertical == tx1DDCT &&
		colMin == minInt16 && colMax == maxInt16 && width >= 4 && width%4 == 0 {
		for col := 0; col < width; col += 4 {
			inverseDCT32Low16Col4Impl(scratch[col:], width, colMin, colMax)
		}
		return false, nil
	}

	if inverseDCT64LowCol4Impl != nil && rowLimit > 0 && rowLimit <= 32 && height == dct64Size && vertical == tx1DDCT &&
		width >= 4 && width%4 == 0 {
		for col := 0; col < width; col += 4 {
			inverseDCT64LowCol4Impl(scratch[col:], width, rowLimit, colMin, colMax)
		}
		return false, nil
	}
	if inverseDCT16LowCol4Impl != nil && rowLimit > 0 && rowLimit <= 8 && height == dct16Size && vertical == tx1DDCT &&
		width >= 4 && width%4 == 0 {
		lowN := 8
		if rowLimit <= 4 {
			lowN = 4
		}
		for col := 0; col < width; col += 4 {
			inverseDCT16LowCol4Impl(scratch[col:], width, lowN, colMin, colMax)
		}
		return false, nil
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

	return false, nil
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
