// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// AVX2 Go-native SIMD self-guided kernels: the box sums, the nine-tap blend, and
// the 8-bit final projection. Eight int32 lanes per vector (one Int32x8) throughout.
// Every operation is the scalar reference's int32 operation, so the results are
// byte-identical; the 8-bit projection narrows with AVX2 packs only (no AVX-512).

// sgrBlendLanes is the number of int32 columns the blend processes per vector.
const sgrBlendLanes = 8

// boxsumSIMD is the Go-native SIMD box sum. The interior columns, whose horizontal
// window [col-r, col+r] is fully in bounds, are summed eight lanes at a time: each
// lane accumulates the (squared) source values over all rows y0..y1 and all 2r+1
// taps. The r clamped edge columns keep the scalar reference. int32 addition wraps
// the same way regardless of grouping, so the sums are bit-identical to boxsum.
func boxsumSIMD(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	interiorStart := radius
	interiorEnd := width - 1 - radius // inclusive
	interiorCount := interiorEnd - interiorStart + 1
	if width <= 0 || height <= 0 || radius < 0 || interiorCount < sgrBlendLanes {
		boxsum(src, srcOrigin, width, height, srcStride, radius, squared, dst, dstStride)
		return
	}
	for row := 0; row < height; row++ {
		y0 := maxInt(0, row-radius)
		y1 := minInt(height-1, row+radius)
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := 0; col < interiorStart; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
		simdCols := interiorCount &^ (sgrBlendLanes - 1)
		for c := 0; c < simdCols; c += sgrBlendLanes {
			acc := archsimd.BroadcastInt32x8(0)
			for y := y0; y <= y1; y++ {
				// Lane L of this group is output column interiorStart+c+L; its tap k
				// reads source column (interiorStart+c+L-radius+k).
				rowBase := srcOrigin + y*srcStride + (interiorStart - radius) + c
				for k := 0; k <= 2*radius; k++ {
					v := archsimd.LoadInt32x8(src[rowBase+k:])
					if squared {
						v = v.Mul(v)
					}
					acc = acc.Add(v)
				}
			}
			acc.Store(dstRow[interiorStart+c:])
		}
		for col := interiorStart + simdCols; col < width; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
	}
}

// sgrBlendRowSIMD computes one blended self-guided output row for cols columns (a
// multiple of sgrBlendLanes). It evaluates dst[c] = roundPowerOfTwo(a[c]*dgd[c] +
// b[c], shift), where a and b are the nine-tap weighted 3x3 stencils over the A/B
// rows (see the arm64 kernel for the tap layout). Every slice is indexed from the
// column one to the left of the first output column.
func sgrBlendRowSIMD(dst []int32, dgd []int32, aPrev []int32, aCur []int32, aNext []int32, bPrev []int32, bCur []int32, bNext []int32, w *[9]int32, shift int, cols int) {
	var wv [9]archsimd.Int32x8
	for i := range wv {
		wv[i] = archsimd.BroadcastInt32x8(w[i])
	}
	biasV := archsimd.BroadcastInt32x8(int32(1) << (shift - 1))
	for c := 0; c < cols; c += sgrBlendLanes {
		a := sgrStencil8(aPrev, aCur, aNext, c, &wv)
		b := sgrStencil8(bPrev, bCur, bNext, c, &wv)
		r := a.Mul(loadI32x8(dgd, c)).Add(b).Add(biasV).ShiftAllRight(uint64(shift))
		r.Store(dst[c:])
	}
}

// sgrStencil8 evaluates the nine-tap weighted 3x3 stencil for eight consecutive
// columns starting at c, using the weight vectors in wv.
func sgrStencil8(prev []int32, cur []int32, next []int32, c int, wv *[9]archsimd.Int32x8) archsimd.Int32x8 {
	return loadI32x8(prev, c).Mul(wv[0]).
		Add(loadI32x8(prev, c+1).Mul(wv[1])).
		Add(loadI32x8(prev, c+2).Mul(wv[2])).
		Add(loadI32x8(cur, c).Mul(wv[3])).
		Add(loadI32x8(cur, c+1).Mul(wv[4])).
		Add(loadI32x8(cur, c+2).Mul(wv[5])).
		Add(loadI32x8(next, c).Mul(wv[6])).
		Add(loadI32x8(next, c+1).Mul(wv[7])).
		Add(loadI32x8(next, c+2).Mul(wv[8]))
}

// loadI32x8 loads eight int32 values starting at p[i].
func loadI32x8(p []int32, i int) archsimd.Int32x8 {
	return archsimd.LoadInt32x8(p[i:])
}

// sgrConsts8 holds the loop-invariant vectors of the 8-bit projection.
type sgrConsts8 struct {
	xq0V, xq1V, cuV, bias archsimd.Int32x8
}

// newSGRConsts8 builds the projection constants. cuV = (1<<PrjBits) - xq0 - xq1
// turns u<<PrjBits + xq0*(f0-u) + xq1*(f1-u) into u*cuV + xq0*f0 + xq1*f1 exactly
// in wrapping int32 arithmetic.
func newSGRConsts8(xq0, xq1 int32) sgrConsts8 {
	return sgrConsts8{
		xq0V: archsimd.BroadcastInt32x8(xq0),
		xq1V: archsimd.BroadcastInt32x8(xq1),
		cuV:  archsimd.BroadcastInt32x8((1 << SGRProjPrjBits) - xq0 - xq1),
		bias: archsimd.BroadcastInt32x8(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
	}
}

// sgrWeightedRowU8SIMD is the Go-native SIMD form of sgrWeightedRowU8. Sixteen
// columns per iteration: one 16-byte load, two 8-lane projections, one 16-byte
// store. The trailing width%16 columns use the scalar reference.
func sgrWeightedRowU8SIMD(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := newSGRConsts8(xq0, xq1)
	lowBytes := archsimd.LoadInt8x16Array(&u8LowByteIdx)
	zeroBytes := archsimd.BroadcastUint8x16(0)
	col := 0
	for ; col+16 <= width; col += 16 {
		sv := archsimd.LoadUint8x16(src[col:])
		// zeroBytes.ConcatShiftBytesRight(sv, 8) yields sv[8..15] in the low eight bytes.
		upper := zeroBytes.ConcatShiftBytesRight(sv, 8)
		outA := sgrWeightedU8Project8(sv.ExtendLo8ToUint32(), f0, f1, col, c, lowBytes)
		outB := sgrWeightedU8Project8(upper.ExtendLo8ToUint32(), f0, f1, col+8, c, lowBytes)
		joined := outA.ReshapeToUint64s().InterleaveLo(outB.ReshapeToUint64s()).ReshapeToUint8s()
		joined.StoreArray((*[16]uint8)(unsafe.Pointer(&dst[col])))
	}
	if col < width {
		sgrWeightedRowU8(dst[col:], src[col:], f0[col:], f1[col:], xq0, xq1)
	}
}

// sgrWeightedU8Project8 projects eight samples (zero-extended to 32-bit lanes in s)
// against the flt rows at index i and returns the eight clamped bytes in the low
// 64 bits. The int16 wrap is a sign-extension of the low 16 bits, and the [0,255]
// clamp is a Max/Min pair before the AVX2 narrow.
func sgrWeightedU8Project8(s archsimd.Uint32x8, f0 []int32, f1 []int32, i int, c sgrConsts8, lowBytes archsimd.Int8x16) archsimd.Uint8x16 {
	u := s.ConvertToInt32().ShiftAllLeft(SGRProjRstBits)
	v := c.xq0V.Mul(loadI32x8(f0, i)).Add(c.xq1V.Mul(loadI32x8(f1, i))).
		Add(c.cuV.Mul(u)).Add(c.bias)
	r := v.ShiftAllRight(SGRProjPrjBits + SGRProjRstBits)
	w := r.ShiftAllLeft(16).ShiftAllRight(16)
	w = w.Max(archsimd.BroadcastInt32x8(0)).Min(archsimd.BroadcastInt32x8(255))
	return packClampedInt32x8ToUint16(w).ReshapeToUint8s().PermuteOrZero(lowBytes)
}
