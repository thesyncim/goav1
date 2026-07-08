// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

// int16 column pipeline (dav1d 8bpc form). For bitDepth==8 blocks whose column
// (vertical) transform is a DCT, the column pass runs entirely in int16: the
// clamp/round between the row and column passes narrows int32->int16 for free,
// then the int16 column kernels run with no boundary conversion. Intermediate
// math stays int64 in the scalar kernels (reference-equivalent, byte-exact).
//
// 10/12-bit stay on the int32 pipeline, matching dav1d's 8bpc/16bpc split.

// inverseDCT8Col8Impl16 is the batched 8-column int16 DCT8 kernel. The default
// is a scalar loop; GOEXPERIMENT=simd binds the int16 8-wide SIMD kernel.
var inverseDCT8Col8Impl16 = inverseDCT8Col8Scalar16

func inverseDCT8Col8Scalar16(buf []int16, stride int, min int32, max int32) {
	for col := 0; col < 8; col++ {
		inverseDCT8(buf[col:], stride, min, max)
	}
}

// clampRoundNarrowInt16 is the mid-pass round+clamp that also narrows the int32
// row-pass output into the int16 column scratch. Equivalent to clampRoundImpl
// followed by an int16 narrow, but done in a single sweep.
func clampRoundNarrowInt16(src []int32, dst []int16, shift int, lo int32, hi int32) {
	if shift > 0 {
		for i := range src {
			dst[i] = clipRangeT[int16](roundShift(int64(src[i]), shift), lo, hi)
		}
	} else {
		for i := range src {
			dst[i] = clipRangeT[int16](int64(src[i]), lo, hi)
		}
	}
}

// inverseDCTColumnPassInt16 runs the DCT column pass over an int16 scratch.
func inverseDCTColumnPassInt16(scratch []int16, width int, height int, min int32, max int32) {
	if height == dct8Size {
		col := 0
		for ; col+8 <= width; col += 8 {
			inverseDCT8Col8Impl16(scratch[col:], width, min, max)
		}
		for ; col < width; col++ {
			inverseDCT8(scratch[col:], width, min, max)
		}
		return
	}
	for col := 0; col < width; col++ {
		inverseDCT1D(scratch[col:], width, height, min, max)
	}
}

// narrowStoreFromInt16 applies the final round/shift and writes the residual
// from the int16 column scratch, matching narrowStoreImpl bit-for-bit.
func narrowStoreFromInt16(dst []int16, dstStride int, scratch []int16, width int, height int) {
	for row := 0; row < height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		tmpLine := scratch[row*width : row*width+width : row*width+width]
		for col, v := range tmpLine {
			dstLine[col] = clipInt16(int32(roundShift(int64(v), 4)))
		}
	}
}
