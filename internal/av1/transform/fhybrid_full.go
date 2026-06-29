// Ported from libaom:
//   av1/encoder/av1_fwd_txfm1d.c
//   av1/encoder/av1_fwd_txfm2d.c
//   av1/common/av1_txfm.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

var fwdSinpi12 = [5]int32{0, 1321, 2482, 3344, 3803}

var fwdCosBitCol = [numDimSlots][numDimSlots]uint8{
	{13, 13, 13, 0, 0},
	{13, 13, 13, 12, 0},
	{13, 13, 13, 12, 13},
	{0, 13, 13, 12, 13},
	{0, 0, 13, 12, 13},
}

var fwdCosBitRow = [numDimSlots][numDimSlots]uint8{
	{13, 13, 12, 0, 0},
	{13, 13, 13, 12, 0},
	{13, 13, 12, 13, 12},
	{0, 12, 13, 12, 11},
	{0, 0, 12, 11, 10},
}

func forwardGenericBlock(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, size Size, typ Type) error {
	params, ok := fwdTxfm2DParams(size)
	if !ok || !forwardGenericSupported(size, typ) {
		return ErrInvalidTransform
	}
	width := int(size.Width)
	height := int(size.Height)
	coeffSize := adjustedScanSize(size)
	coeffWidth := int(coeffSize.Width)
	coeffHeight := int(coeffSize.Height)
	if coeffStride < coeffHeight ||
		residualStride < width ||
		len(scratch) < width*height ||
		!blockFits(len(residual), residualStride, width, height) ||
		!coeffBlockFits(len(coeff), coeffStride, coeffWidth, coeffHeight) {
		return ErrInvalidTransform
	}

	vertical, horizontal, _ := typ.tx1DTypes()
	udFlip, lrFlip := fwdFlipFlags(typ)
	scratch = scratch[:width*height]

	var tempIn, tempOut [64]int32
	for c := range width {
		for r := range height {
			srcRow := r
			if udFlip {
				srcRow = height - r - 1
			}
			tempIn[r] = int32(residual[srcRow*residualStride+c])
		}
		fwdRoundShiftArray(tempIn[:height], -int(params.shift[0]))
		fwd1DTyped(tempIn[:height], tempOut[:height], vertical, uint(params.cosBitCol))
		fwdRoundShiftArray(tempOut[:height], -int(params.shift[1]))
		dstCol := c
		if lrFlip {
			dstCol = width - c - 1
		}
		for r := range height {
			scratch[r*width+dstCol] = tempOut[r]
		}
	}

	for r := range height {
		row := scratch[r*width : r*width+width : r*width+width]
		fwd1DTyped(row, tempOut[:width], horizontal, uint(params.cosBitRow))
		fwdRoundShiftArray(tempOut[:width], -int(params.shift[2]))
		if size.IsRect2() {
			fwdRoundShiftSqrt2(tempOut[:width])
		}
		if r >= coeffHeight {
			continue
		}
		for c := range coeffWidth {
			coeff[c*coeffStride+r] = tempOut[c]
		}
	}
	return nil
}

func forwardGenericSupported(size Size, typ Type) bool {
	if typ == TypeDCTDCT {
		return false
	}
	switch size {
	case Size{Width: 16, Height: 64}, Size{Width: 64, Height: 16}:
		return false
	}
	if _, ok := fwdTxfm2DParams(size); !ok {
		return false
	}
	vertical, horizontal, ok := typ.tx1DTypes()
	return ok &&
		forward1DSupported(vertical, int(size.Height)) &&
		forward1DSupported(horizontal, int(size.Width))
}

type fwdTxfm2DConfig struct {
	shift     [3]int8
	cosBitCol uint8
	cosBitRow uint8
}

func fwdTxfm2DParams(size Size) (fwdTxfm2DConfig, bool) {
	idx := sizeIndex(size)
	if idx < 0 || !sizeValidTable[idx] {
		return fwdTxfm2DConfig{}, false
	}
	w := dimSlot(int(size.Width))
	h := dimSlot(int(size.Height))
	if w < 0 || h < 0 {
		return fwdTxfm2DConfig{}, false
	}
	cfg := fwdTxfm2DConfig{
		shift:     fwdShiftForSize(size),
		cosBitCol: fwdCosBitCol[w][h],
		cosBitRow: fwdCosBitRow[w][h],
	}
	if !fwdCosBitSupported(cfg.cosBitCol) || !fwdCosBitSupported(cfg.cosBitRow) {
		return fwdTxfm2DConfig{}, false
	}
	return cfg, true
}

func fwdCosBitSupported(cosBit uint8) bool {
	return cosBit >= 10 && cosBit <= 13
}

func fwdShiftForSize(size Size) [3]int8 {
	switch size {
	case Size{Width: 4, Height: 4}:
		return [3]int8{2, 0, 0}
	case Size{Width: 8, Height: 8}:
		return [3]int8{2, -1, 0}
	case Size{Width: 16, Height: 16}:
		return [3]int8{2, -2, 0}
	case Size{Width: 32, Height: 32}:
		return [3]int8{2, -4, 0}
	case Size{Width: 64, Height: 64}:
		return [3]int8{0, -2, -2}
	case Size{Width: 4, Height: 8}, Size{Width: 8, Height: 4}:
		return [3]int8{2, -1, 0}
	case Size{Width: 8, Height: 16}, Size{Width: 16, Height: 8}:
		return [3]int8{2, -2, 0}
	case Size{Width: 16, Height: 32}, Size{Width: 32, Height: 16}:
		return [3]int8{2, -4, 0}
	case Size{Width: 32, Height: 64}:
		return [3]int8{0, -2, -2}
	case Size{Width: 64, Height: 32}:
		return [3]int8{2, -4, -2}
	case Size{Width: 4, Height: 16}, Size{Width: 16, Height: 4}:
		return [3]int8{2, -1, 0}
	case Size{Width: 8, Height: 32}, Size{Width: 32, Height: 8}:
		return [3]int8{2, -2, 0}
	case Size{Width: 16, Height: 64}:
		return [3]int8{0, -2, 0}
	case Size{Width: 64, Height: 16}:
		return [3]int8{2, -4, 0}
	default:
		return [3]int8{}
	}
}

func fwdFlipFlags(typ Type) (udFlip bool, lrFlip bool) {
	switch typ {
	case TypeFlipADSTDCT, TypeFlipADSTADST, TypeVFlipADST:
		return true, false
	case TypeDCTFlipADST, TypeADSTFlipADST, TypeHFlipADST:
		return false, true
	case TypeFlipADSTFlipADST:
		return true, true
	default:
		return false, false
	}
}

func forward1DSupported(typ tx1DType, length int) bool {
	switch typ {
	case tx1DDCT:
		switch length {
		case 4, 8, 16, 32, 64:
			return true
		}
	case tx1DADST, tx1DFlipADST:
		switch length {
		case 4, 8, 16:
			return true
		}
	case tx1DIdentity:
		switch length {
		case 4, 8, 16, 32:
			return true
		}
	}
	return false
}

func fwd1DTyped(input []int32, output []int32, typ tx1DType, cosBit uint) {
	switch typ {
	case tx1DDCT:
		fwdDCTByLength(input, output, cosBit)
	case tx1DADST, tx1DFlipADST:
		fwdADSTByLength(input, output, cosBit)
	case tx1DIdentity:
		fwdIdentity1D(input, output)
	}
}

func fwdDCTByLength(input []int32, output []int32, cosBit uint) {
	cospi := fwdCospiForBit(cosBit)
	switch len(input) {
	case 4:
		var in, out [4]int32
		copy(in[:], input)
		fwdDCT4ByCosBit(&in, &out, cospi, cosBit)
		copy(output, out[:])
	case 8:
		var in, out [8]int32
		copy(in[:], input)
		fwdDCT8ByCosBit(&in, &out, cospi, cosBit)
		copy(output, out[:])
	case 16:
		var in, out [16]int32
		copy(in[:], input)
		fwdDCT16(&in, &out, cospi, cosBit)
		copy(output, out[:])
	case 32:
		var in, out [32]int32
		copy(in[:], input)
		fwdDCT32(&in, &out, cospi, cosBit)
		copy(output, out[:])
	case 64:
		var in, out [64]int32
		copy(in[:], input)
		fwdDCT64(&in, &out, cospi, cosBit)
		copy(output, out[:])
	}
}

func fwdADSTByLength(input []int32, output []int32, cosBit uint) {
	cospi := fwdCospiForBit(cosBit)
	switch len(input) {
	case 4:
		var in, out [4]int32
		copy(in[:], input)
		fwdADST4ByCosBit(&in, &out, fwdSinpiForBit(cosBit), cosBit)
		copy(output, out[:])
	case 8:
		var in, out [8]int32
		copy(in[:], input)
		fwdADST8ByCosBit(&in, &out, cospi, cosBit)
		copy(output, out[:])
	case 16:
		var in, out [16]int32
		copy(in[:], input)
		fwdADST16(&in, &out, cospi, cosBit)
		copy(output, out[:])
	}
}

func fwdCospiForBit(cosBit uint) *[64]int32 {
	switch cosBit {
	case 10:
		return &fwdCospi10
	case 11:
		return &fwdCospi11
	case 12:
		return &fwdCospi12
	default:
		return &fwdCospi13
	}
}

func fwdSinpiForBit(cosBit uint) *[5]int32 {
	if cosBit == 12 {
		return &fwdSinpi12
	}
	return &fwdSinpi13
}

func fwdRoundShiftArray(values []int32, bit int) {
	if bit == 0 {
		return
	}
	if bit > 0 {
		for i, v := range values {
			values[i] = int32(roundShift(int64(v), bit))
		}
		return
	}
	shift := -bit
	for i, v := range values {
		values[i] = saturatingLeftShiftInt32(v, shift)
	}
}

func fwdDCT4ByCosBit(input, output *[4]int32, cospi *[64]int32, cosBit uint) {
	s0 := input[0] + input[3]
	s1 := input[1] + input[2]
	s2 := -input[2] + input[1]
	s3 := -input[3] + input[0]
	t0 := fwdHalfBtf(cospi[32], s0, cospi[32], s1, cosBit)
	t1 := fwdHalfBtf(-cospi[32], s1, cospi[32], s0, cosBit)
	t2 := fwdHalfBtf(cospi[48], s2, cospi[16], s3, cosBit)
	t3 := fwdHalfBtf(cospi[48], s3, -cospi[16], s2, cosBit)
	output[0] = t0
	output[1] = t2
	output[2] = t1
	output[3] = t3
}

func fwdDCT8ByCosBit(input, output *[8]int32, cospi *[64]int32, cosBit uint) {
	var b, s [8]int32
	b[0] = input[0] + input[7]
	b[1] = input[1] + input[6]
	b[2] = input[2] + input[5]
	b[3] = input[3] + input[4]
	b[4] = -input[4] + input[3]
	b[5] = -input[5] + input[2]
	b[6] = -input[6] + input[1]
	b[7] = -input[7] + input[0]

	s[0] = b[0] + b[3]
	s[1] = b[1] + b[2]
	s[2] = -b[2] + b[1]
	s[3] = -b[3] + b[0]
	s[4] = b[4]
	s[5] = fwdHalfBtf(-cospi[32], b[5], cospi[32], b[6], cosBit)
	s[6] = fwdHalfBtf(cospi[32], b[6], cospi[32], b[5], cosBit)
	s[7] = b[7]

	b[0] = fwdHalfBtf(cospi[32], s[0], cospi[32], s[1], cosBit)
	b[1] = fwdHalfBtf(-cospi[32], s[1], cospi[32], s[0], cosBit)
	b[2] = fwdHalfBtf(cospi[48], s[2], cospi[16], s[3], cosBit)
	b[3] = fwdHalfBtf(cospi[48], s[3], -cospi[16], s[2], cosBit)
	b[4] = s[4] + s[5]
	b[5] = -s[5] + s[4]
	b[6] = -s[6] + s[7]
	b[7] = s[7] + s[6]

	s[0] = b[0]
	s[1] = b[1]
	s[2] = b[2]
	s[3] = b[3]
	s[4] = fwdHalfBtf(cospi[56], b[4], cospi[8], b[7], cosBit)
	s[5] = fwdHalfBtf(cospi[24], b[5], cospi[40], b[6], cosBit)
	s[6] = fwdHalfBtf(cospi[24], b[6], -cospi[40], b[5], cosBit)
	s[7] = fwdHalfBtf(cospi[56], b[7], -cospi[8], b[4], cosBit)

	output[0] = s[0]
	output[1] = s[4]
	output[2] = s[2]
	output[3] = s[6]
	output[4] = s[1]
	output[5] = s[5]
	output[6] = s[3]
	output[7] = s[7]
}

func fwdADST4ByCosBit(input, output *[4]int32, sinpi *[5]int32, cosBit uint) {
	x0, x1, x2, x3 := input[0], input[1], input[2], input[3]
	if x0|x1|x2|x3 == 0 {
		clear(output[:])
		return
	}
	s0 := sinpi[1] * x0
	s1 := sinpi[4] * x0
	s2 := sinpi[2] * x1
	s3 := sinpi[1] * x1
	s4 := sinpi[3] * x2
	s5 := sinpi[4] * x3
	s6 := sinpi[2] * x3
	s7 := x0 + x1

	s7 -= x3

	x0 = s0 + s2
	x1 = sinpi[3] * s7
	x2 = s1 - s3
	x3 = s4

	x0 += s5
	x2 += s6

	s0 = x0 + x3
	s1 = x1
	s2 = x2 - x3
	s3 = x2 - x0

	s3 += x3

	output[0] = int32(roundShift(int64(s0), int(cosBit)))
	output[1] = int32(roundShift(int64(s1), int(cosBit)))
	output[2] = int32(roundShift(int64(s2), int(cosBit)))
	output[3] = int32(roundShift(int64(s3), int(cosBit)))
}

func fwdADST8ByCosBit(input, output *[8]int32, cospi *[64]int32, cosBit uint) {
	var step [8]int32
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = -input[7]
	bf1[2] = -input[3]
	bf1[3] = input[4]
	bf1[4] = -input[1]
	bf1[5] = input[6]
	bf1[6] = input[2]
	bf1[7] = -input[5]

	bf0 := output
	bf1 = &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = fwdHalfBtf(cospi[32], bf0[2], cospi[32], bf0[3], cosBit)
	bf1[3] = fwdHalfBtf(cospi[32], bf0[2], -cospi[32], bf0[3], cosBit)
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = fwdHalfBtf(cospi[32], bf0[6], cospi[32], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[32], bf0[6], -cospi[32], bf0[7], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[2]
	bf1[1] = bf0[1] + bf0[3]
	bf1[2] = bf0[0] - bf0[2]
	bf1[3] = bf0[1] - bf0[3]
	bf1[4] = bf0[4] + bf0[6]
	bf1[5] = bf0[5] + bf0[7]
	bf1[6] = bf0[4] - bf0[6]
	bf1[7] = bf0[5] - bf0[7]

	bf0, bf1 = output, &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = fwdHalfBtf(cospi[16], bf0[4], cospi[48], bf0[5], cosBit)
	bf1[5] = fwdHalfBtf(cospi[48], bf0[4], -cospi[16], bf0[5], cosBit)
	bf1[6] = fwdHalfBtf(-cospi[48], bf0[6], cospi[16], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[16], bf0[6], cospi[48], bf0[7], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[4]
	bf1[1] = bf0[1] + bf0[5]
	bf1[2] = bf0[2] + bf0[6]
	bf1[3] = bf0[3] + bf0[7]
	bf1[4] = bf0[0] - bf0[4]
	bf1[5] = bf0[1] - bf0[5]
	bf1[6] = bf0[2] - bf0[6]
	bf1[7] = bf0[3] - bf0[7]

	bf0, bf1 = output, &step
	bf1[0] = fwdHalfBtf(cospi[4], bf0[0], cospi[60], bf0[1], cosBit)
	bf1[1] = fwdHalfBtf(cospi[60], bf0[0], -cospi[4], bf0[1], cosBit)
	bf1[2] = fwdHalfBtf(cospi[20], bf0[2], cospi[44], bf0[3], cosBit)
	bf1[3] = fwdHalfBtf(cospi[44], bf0[2], -cospi[20], bf0[3], cosBit)
	bf1[4] = fwdHalfBtf(cospi[36], bf0[4], cospi[28], bf0[5], cosBit)
	bf1[5] = fwdHalfBtf(cospi[28], bf0[4], -cospi[36], bf0[5], cosBit)
	bf1[6] = fwdHalfBtf(cospi[52], bf0[6], cospi[12], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[12], bf0[6], -cospi[52], bf0[7], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[1]
	bf1[1] = bf0[6]
	bf1[2] = bf0[3]
	bf1[3] = bf0[4]
	bf1[4] = bf0[5]
	bf1[5] = bf0[2]
	bf1[6] = bf0[7]
	bf1[7] = bf0[0]
}

func fwdADST16(input, output *[16]int32, cospi *[64]int32, cosBit uint) {
	var step [16]int32
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = -input[15]
	bf1[2] = -input[7]
	bf1[3] = input[8]
	bf1[4] = -input[3]
	bf1[5] = input[12]
	bf1[6] = input[4]
	bf1[7] = -input[11]
	bf1[8] = -input[1]
	bf1[9] = input[14]
	bf1[10] = input[6]
	bf1[11] = -input[9]
	bf1[12] = input[2]
	bf1[13] = -input[13]
	bf1[14] = -input[5]
	bf1[15] = input[10]

	bf0 := output
	bf1 = &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = fwdHalfBtf(cospi[32], bf0[2], cospi[32], bf0[3], cosBit)
	bf1[3] = fwdHalfBtf(cospi[32], bf0[2], -cospi[32], bf0[3], cosBit)
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = fwdHalfBtf(cospi[32], bf0[6], cospi[32], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[32], bf0[6], -cospi[32], bf0[7], cosBit)
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = fwdHalfBtf(cospi[32], bf0[10], cospi[32], bf0[11], cosBit)
	bf1[11] = fwdHalfBtf(cospi[32], bf0[10], -cospi[32], bf0[11], cosBit)
	bf1[12] = bf0[12]
	bf1[13] = bf0[13]
	bf1[14] = fwdHalfBtf(cospi[32], bf0[14], cospi[32], bf0[15], cosBit)
	bf1[15] = fwdHalfBtf(cospi[32], bf0[14], -cospi[32], bf0[15], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[2]
	bf1[1] = bf0[1] + bf0[3]
	bf1[2] = bf0[0] - bf0[2]
	bf1[3] = bf0[1] - bf0[3]
	bf1[4] = bf0[4] + bf0[6]
	bf1[5] = bf0[5] + bf0[7]
	bf1[6] = bf0[4] - bf0[6]
	bf1[7] = bf0[5] - bf0[7]
	bf1[8] = bf0[8] + bf0[10]
	bf1[9] = bf0[9] + bf0[11]
	bf1[10] = bf0[8] - bf0[10]
	bf1[11] = bf0[9] - bf0[11]
	bf1[12] = bf0[12] + bf0[14]
	bf1[13] = bf0[13] + bf0[15]
	bf1[14] = bf0[12] - bf0[14]
	bf1[15] = bf0[13] - bf0[15]

	bf0, bf1 = output, &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = fwdHalfBtf(cospi[16], bf0[4], cospi[48], bf0[5], cosBit)
	bf1[5] = fwdHalfBtf(cospi[48], bf0[4], -cospi[16], bf0[5], cosBit)
	bf1[6] = fwdHalfBtf(-cospi[48], bf0[6], cospi[16], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[16], bf0[6], cospi[48], bf0[7], cosBit)
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = bf0[10]
	bf1[11] = bf0[11]
	bf1[12] = fwdHalfBtf(cospi[16], bf0[12], cospi[48], bf0[13], cosBit)
	bf1[13] = fwdHalfBtf(cospi[48], bf0[12], -cospi[16], bf0[13], cosBit)
	bf1[14] = fwdHalfBtf(-cospi[48], bf0[14], cospi[16], bf0[15], cosBit)
	bf1[15] = fwdHalfBtf(cospi[16], bf0[14], cospi[48], bf0[15], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[4]
	bf1[1] = bf0[1] + bf0[5]
	bf1[2] = bf0[2] + bf0[6]
	bf1[3] = bf0[3] + bf0[7]
	bf1[4] = bf0[0] - bf0[4]
	bf1[5] = bf0[1] - bf0[5]
	bf1[6] = bf0[2] - bf0[6]
	bf1[7] = bf0[3] - bf0[7]
	bf1[8] = bf0[8] + bf0[12]
	bf1[9] = bf0[9] + bf0[13]
	bf1[10] = bf0[10] + bf0[14]
	bf1[11] = bf0[11] + bf0[15]
	bf1[12] = bf0[8] - bf0[12]
	bf1[13] = bf0[9] - bf0[13]
	bf1[14] = bf0[10] - bf0[14]
	bf1[15] = bf0[11] - bf0[15]

	bf0, bf1 = output, &step
	for i := range 8 {
		bf1[i] = bf0[i]
	}
	bf1[8] = fwdHalfBtf(cospi[8], bf0[8], cospi[56], bf0[9], cosBit)
	bf1[9] = fwdHalfBtf(cospi[56], bf0[8], -cospi[8], bf0[9], cosBit)
	bf1[10] = fwdHalfBtf(cospi[40], bf0[10], cospi[24], bf0[11], cosBit)
	bf1[11] = fwdHalfBtf(cospi[24], bf0[10], -cospi[40], bf0[11], cosBit)
	bf1[12] = fwdHalfBtf(-cospi[56], bf0[12], cospi[8], bf0[13], cosBit)
	bf1[13] = fwdHalfBtf(cospi[8], bf0[12], cospi[56], bf0[13], cosBit)
	bf1[14] = fwdHalfBtf(-cospi[24], bf0[14], cospi[40], bf0[15], cosBit)
	bf1[15] = fwdHalfBtf(cospi[40], bf0[14], cospi[24], bf0[15], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[8]
	bf1[1] = bf0[1] + bf0[9]
	bf1[2] = bf0[2] + bf0[10]
	bf1[3] = bf0[3] + bf0[11]
	bf1[4] = bf0[4] + bf0[12]
	bf1[5] = bf0[5] + bf0[13]
	bf1[6] = bf0[6] + bf0[14]
	bf1[7] = bf0[7] + bf0[15]
	bf1[8] = bf0[0] - bf0[8]
	bf1[9] = bf0[1] - bf0[9]
	bf1[10] = bf0[2] - bf0[10]
	bf1[11] = bf0[3] - bf0[11]
	bf1[12] = bf0[4] - bf0[12]
	bf1[13] = bf0[5] - bf0[13]
	bf1[14] = bf0[6] - bf0[14]
	bf1[15] = bf0[7] - bf0[15]

	bf0, bf1 = output, &step
	bf1[0] = fwdHalfBtf(cospi[2], bf0[0], cospi[62], bf0[1], cosBit)
	bf1[1] = fwdHalfBtf(cospi[62], bf0[0], -cospi[2], bf0[1], cosBit)
	bf1[2] = fwdHalfBtf(cospi[10], bf0[2], cospi[54], bf0[3], cosBit)
	bf1[3] = fwdHalfBtf(cospi[54], bf0[2], -cospi[10], bf0[3], cosBit)
	bf1[4] = fwdHalfBtf(cospi[18], bf0[4], cospi[46], bf0[5], cosBit)
	bf1[5] = fwdHalfBtf(cospi[46], bf0[4], -cospi[18], bf0[5], cosBit)
	bf1[6] = fwdHalfBtf(cospi[26], bf0[6], cospi[38], bf0[7], cosBit)
	bf1[7] = fwdHalfBtf(cospi[38], bf0[6], -cospi[26], bf0[7], cosBit)
	bf1[8] = fwdHalfBtf(cospi[34], bf0[8], cospi[30], bf0[9], cosBit)
	bf1[9] = fwdHalfBtf(cospi[30], bf0[8], -cospi[34], bf0[9], cosBit)
	bf1[10] = fwdHalfBtf(cospi[42], bf0[10], cospi[22], bf0[11], cosBit)
	bf1[11] = fwdHalfBtf(cospi[22], bf0[10], -cospi[42], bf0[11], cosBit)
	bf1[12] = fwdHalfBtf(cospi[50], bf0[12], cospi[14], bf0[13], cosBit)
	bf1[13] = fwdHalfBtf(cospi[14], bf0[12], -cospi[50], bf0[13], cosBit)
	bf1[14] = fwdHalfBtf(cospi[58], bf0[14], cospi[6], bf0[15], cosBit)
	bf1[15] = fwdHalfBtf(cospi[6], bf0[14], -cospi[58], bf0[15], cosBit)

	bf0, bf1 = &step, output
	bf1[0] = bf0[1]
	bf1[1] = bf0[14]
	bf1[2] = bf0[3]
	bf1[3] = bf0[12]
	bf1[4] = bf0[5]
	bf1[5] = bf0[10]
	bf1[6] = bf0[7]
	bf1[7] = bf0[8]
	bf1[8] = bf0[9]
	bf1[9] = bf0[6]
	bf1[10] = bf0[11]
	bf1[11] = bf0[4]
	bf1[12] = bf0[13]
	bf1[13] = bf0[2]
	bf1[14] = bf0[15]
	bf1[15] = bf0[0]
}
