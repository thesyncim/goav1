// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

func init() {
	itx16RawColumnPass = itx16RawColumnPassSIMD
	itx16RawColumnPass32Low8 = itx16RawColumnPass32Low8SIMD
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16v2
}

// itx16ColumnL1Bound is the per-column guard of the int16 DCT16 column kernel:
// sum_j |x_j| <= bound certifies that every rotation output fits int16, since
// the largest input coefficient of any rotation output is 0.9951 and the
// accumulated rounding slack is at most 4.03 (itx16_gain_test.go), with
// margin.
const itx16ColumnL1Bound = 32767 - 64

// itx16RawColumnPassSIMD runs the DCT16 column pass over the int32 raw
// scratch (width x 16, already rounded and clamped to the int16 range) eight
// columns at a time with the int16 kernel; a column group that fails the
// kernel's guard is untouched and takes the int32 kernels (the low-row ones
// when only rowLimit input rows are nonzero).
func itx16RawColumnPassSIMD(scratch []int32, width, rowLimit int, min, max int32) {
	for c := 0; c < width; c += 8 {
		var ok bool
		switch {
		case rowLimit > 0 && rowLimit <= 4:
			ok = inverseDCT16Col8RawLow4SIMD16(scratch[c:], width)
		case rowLimit > 0 && rowLimit <= 8:
			ok = inverseDCT16Col8RawLow8SIMD16(scratch[c:], width)
		default:
			ok = inverseDCT16Col8RawSIMD16(scratch[c:], width)
		}
		if ok {
			continue
		}
		if rowLimit > 0 && rowLimit <= 8 && inverseDCT16LowCol4Impl != nil {
			lowN := 8
			if rowLimit <= 4 {
				lowN = 4
			}
			inverseDCT16LowCol4Impl(scratch[c:], width, lowN, min, max)
			inverseDCT16LowCol4Impl(scratch[c+4:], width, lowN, min, max)
			continue
		}
		inverse1DCol4(scratch[c:], width, dct16Size, tx1DDCT, min, max)
		inverse1DCol4(scratch[c+4:], width, dct16Size, tx1DDCT, min, max)
	}
}
