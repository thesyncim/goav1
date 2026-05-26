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
	shift, ok := size.shift()
	vertical, horizontal, okType := typ.tx1DTypes()
	coeffSize := adjustedScanSize(size)
	scratchLen := size.Width * size.Height
	if !ok ||
		!okType ||
		!typ.Supported(size) ||
		dstStride < size.Width ||
		coeffStride < coeffSize.Height ||
		len(scratch) < scratchLen ||
		!blockFits(len(dst), dstStride, size.Width, size.Height) ||
		!coeffBlockFits(len(coeff), coeffStride, coeffSize.Width, coeffSize.Height) {
		return ErrInvalidTransform
	}

	for row := 0; row < size.Height; row++ {
		tmpLine := scratch[row*size.Width : row*size.Width+size.Width]
		for col := 0; col < size.Width; col++ {
			v := int32(0)
			if col < coeffSize.Width && row < coeffSize.Height {
				v = coeff[col*coeffStride+row]
			}
			if size.IsRect2() {
				v = rect2Scale(v)
			}
			tmpLine[col] = clipRange(int64(v), rowMin, rowMax)
		}
		inverse1D(tmpLine, 1, size.Width, horizontal, rowMin, rowMax)
	}

	if shift > 0 {
		for i := range scratchLen {
			scratch[i] = clipRange(roundShift(int64(scratch[i]), shift), colMin, colMax)
		}
	} else {
		for i := range scratchLen {
			scratch[i] = clipRange(int64(scratch[i]), colMin, colMax)
		}
	}

	for col := 0; col < size.Width; col++ {
		inverse1D(scratch[col:], size.Width, size.Height, vertical, colMin, colMax)
	}

	for row := 0; row < size.Height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+size.Width]
		tmpLine := scratch[row*size.Width : row*size.Width+size.Width]
		for col := 0; col < size.Width; col++ {
			dstLine[col] = clipInt16(clipInt32(roundShift(int64(tmpLine[col]), 4)))
		}
	}
	return nil
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
