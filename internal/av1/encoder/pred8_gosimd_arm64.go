//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	pred8Impl = pred8SIMD
	pred8IMImpl = pred8IM
	pred8VIMImpl = pred8VIM
}

// The AV1 8-tap kernels are all even, so the 8-bit passes accumulate halved
// taps in int16 lanes without overflow (the positive taps of any kernel sum
// to at most 92 after halving, so a sum stays below 92*255 + 34). With
// a = sum(k/2 * x):
//
//	convolve_x_sr: clip(round(round(2a, 3), 4)) = clip((a + 34) >> 6)
//	convolve_y_sr: clip(round(2a, 7))           = clip((a + 32) >> 6)
//	2-D first pass: round(16384 + 2a, 3)        = (a + 8194) >> 2
//
// and the 2-D second pass over the intermediate im (int32 lanes, full taps):
//
//	clip(round(2^19 + sum(k*im), 11) - 384) = clip((sum(k*im) - 261120) >> 11)
//
// (2^19 and 384<<11 are multiples of 2^11, so they fold into the bias).

// pred8Out stores the 16 output bytes of a row group or folds their absolute
// difference with the source into the SAD accumulator.
func pred8Out16(out archsimd.Uint8x16, dst unsafe.Pointer, dstOff int, src unsafe.Pointer, srcOff int, acc archsimd.Uint16x8) archsimd.Uint16x8 {
	if src != nil {
		return acc.Add(widen16(absDiffU8x16(out, load16(unsafe.Add(src, srcOff)))))
	}
	out.StoreArray((*[16]uint8)(unsafe.Add(dst, dstOff)))
	return acc
}

func pred8Pack16(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	return lo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
}

func pred8Store8(dst unsafe.Pointer, v archsimd.Int16x8) {
	*(*uint64)(dst) = v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
}

func pred8SIMD(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int,
	ref unsafe.Pointer, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	switch {
	case subX == 0 && subY == 0:
		return pred8Copy(dst, dstStride, src, srcStride, ref, refStride, w, h)
	case subY == 0:
		return pred8H(dst, dstStride, src, srcStride, ref, refStride, w, h, xk)
	case subX == 0:
		return pred8V(dst, dstStride, src, srcStride, ref, refStride, w, h, yk)
	}
	imp := unsafe.Pointer(&im[0])
	pred8IM(imp, 64*2, unsafe.Add(ref, -3*refStride), refStride, w, h+7, xk)
	return pred8VIM(dst, dstStride, src, srcStride, imp, 64*2, w, h, yk)
}

func pred8Copy(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int) int {
	var acc archsimd.Uint16x8
	sad := 0
	for y := 0; y < h; y++ {
		r := unsafe.Add(ref, y*refStride)
		if src == nil && w == 8 {
			*(*uint64)(unsafe.Add(dst, y*dstStride)) = *(*uint64)(r)
			continue
		}
		for x := 0; x < w; x += 16 {
			acc = pred8Out16(load16(unsafe.Add(r, x)), dst, y*dstStride+x, src, y*srcStride+x, acc)
		}
		if src != nil && y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}

// pred8H is convolve_x_sr: per output row, each tap's window is one 16-byte
// load whose low and high halves feed the two 8-lane accumulators.
func pred8H(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int, k *[8]int16) int {
	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	bias := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	var acc archsimd.Uint16x8
	sad := 0
	p := unsafe.Add(ref, -3)
	for y := 0; y < h; y++ {
		x := 0
		for ; x+16 <= w; x += 16 {
			q := unsafe.Add(p, x)
			v := load16(q)
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = load16(unsafe.Add(q, 1))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = load16(unsafe.Add(q, 2))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = load16(unsafe.Add(q, 3))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = load16(unsafe.Add(q, 4))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = load16(unsafe.Add(q, 5))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = load16(unsafe.Add(q, 6))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = load16(unsafe.Add(q, 7))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
		}
		if x < w {
			q := unsafe.Add(p, x)
			a0 := load16(q).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = load16(unsafe.Add(q, 1)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = load16(unsafe.Add(q, 2)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = load16(unsafe.Add(q, 3)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = load16(unsafe.Add(q, 4)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = load16(unsafe.Add(q, 5)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = load16(unsafe.Add(q, 6)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = load16(unsafe.Add(q, 7)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(unsafe.Add(dst, y*dstStride+x), a0.Shift(sh))
		}
		if src != nil && y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
		p = unsafe.Add(p, refStride)
	}
	return sad + sumU16(acc)
}

// pred8V is convolve_y_sr over 16-column (or a final 8-column) strips. The
// pinned libaom av1/common/arm/convolve_neon.c six-tap vertical kernel omits
// coefficients 0 and 7. Regular/smooth kernels have zero outer taps; fractional
// sharp kernels retain the full eight-tap path.
func pred8V(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int, k *[8]int16) int {
	if k[0] == 0 && k[7] == 0 {
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(32)
		sh := archsimd.BroadcastInt16x8(-6)
		var acc archsimd.Uint16x8
		sad := 0
		s := refStride
		for y := 0; y < h; y++ {
			p := unsafe.Add(ref, (y-2)*s)
			x := 0
			for ; x+16 <= w; x += 16 {
				q := unsafe.Add(p, x)
				v := load16(q)
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
				v = load16(unsafe.Add(q, s))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = load16(unsafe.Add(q, 2*s))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = load16(unsafe.Add(q, 3*s))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = load16(unsafe.Add(q, 4*s))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = load16(unsafe.Add(q, 5*s))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
			}
			if x < w {
				q := unsafe.Add(p, x)
				a0 := load16(q).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = load16(unsafe.Add(q, s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = load16(unsafe.Add(q, 2*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = load16(unsafe.Add(q, 3*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = load16(unsafe.Add(q, 4*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = load16(unsafe.Add(q, 5*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				pred8Store8(unsafe.Add(dst, y*dstStride+x), a0.Shift(sh))
			}
			if src != nil && y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
		return sad + sumU16(acc)
	}

	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	bias := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	var acc archsimd.Uint16x8
	sad := 0
	s := refStride
	for y := 0; y < h; y++ {
		p := unsafe.Add(ref, (y-3)*s)
		x := 0
		for ; x+16 <= w; x += 16 {
			q := unsafe.Add(p, x)
			v := load16(q)
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = load16(unsafe.Add(q, s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = load16(unsafe.Add(q, 2*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = load16(unsafe.Add(q, 3*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = load16(unsafe.Add(q, 4*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = load16(unsafe.Add(q, 5*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = load16(unsafe.Add(q, 6*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = load16(unsafe.Add(q, 7*s))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
		}
		if x < w {
			q := unsafe.Add(p, x)
			a0 := load16(q).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = load16(unsafe.Add(q, s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = load16(unsafe.Add(q, 2*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = load16(unsafe.Add(q, 3*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = load16(unsafe.Add(q, 4*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = load16(unsafe.Add(q, 5*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = load16(unsafe.Add(q, 6*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = load16(unsafe.Add(q, 7*s)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(unsafe.Add(dst, y*dstStride+x), a0.Shift(sh))
		}
		if src != nil && y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}

// pred8IM is the 2-D first pass: rows -3..h+3 of the block into im with the
// supplied intermediate stride in bytes, as int16 (a + 8194) >> 2.
func pred8IM(im unsafe.Pointer, imStride int, ref unsafe.Pointer, refStride, w, rows int, k *[8]int16) {
	if k[0] == 0 && k[7] == 0 {
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(8194)
		sh := archsimd.BroadcastInt16x8(-2)
		p := unsafe.Add(ref, -2)
		for y := 0; y < rows; y++ {
			row := unsafe.Add(im, y*imStride)
			x := 0
			for ; x+16 <= w; x += 16 {
				q := unsafe.Add(p, x)
				v := load16(q)
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
				v = load16(unsafe.Add(q, 1))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = load16(unsafe.Add(q, 2))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = load16(unsafe.Add(q, 3))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = load16(unsafe.Add(q, 4))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = load16(unsafe.Add(q, 5))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				a0.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x)))
				a1.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x+16)))
			}
			if x < w {
				q := unsafe.Add(p, x)
				a0 := load16(q).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = load16(unsafe.Add(q, 1)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = load16(unsafe.Add(q, 2)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = load16(unsafe.Add(q, 3)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = load16(unsafe.Add(q, 4)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = load16(unsafe.Add(q, 5)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x)))
			}
			p = unsafe.Add(p, refStride)
		}
		return
	}
	c0 := archsimd.BroadcastInt16x8(k[0] >> 1)
	c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
	c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
	c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
	c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
	c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
	c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
	c7 := archsimd.BroadcastInt16x8(k[7] >> 1)
	bias := archsimd.BroadcastInt16x8(8194)
	sh := archsimd.BroadcastInt16x8(-2)
	p := unsafe.Add(ref, -3)
	for y := 0; y < rows; y++ {
		row := unsafe.Add(im, y*imStride)
		x := 0
		for ; x+16 <= w; x += 16 {
			q := unsafe.Add(p, x)
			v := load16(q)
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = load16(unsafe.Add(q, 1))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = load16(unsafe.Add(q, 2))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = load16(unsafe.Add(q, 3))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = load16(unsafe.Add(q, 4))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = load16(unsafe.Add(q, 5))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = load16(unsafe.Add(q, 6))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = load16(unsafe.Add(q, 7))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			a0.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x)))
			a1.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x+16)))
		}
		if x < w {
			q := unsafe.Add(p, x)
			a0 := load16(q).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = load16(unsafe.Add(q, 1)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = load16(unsafe.Add(q, 2)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = load16(unsafe.Add(q, 3)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = load16(unsafe.Add(q, 4)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = load16(unsafe.Add(q, 5)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = load16(unsafe.Add(q, 6)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = load16(unsafe.Add(q, 7)).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a0.Shift(sh).StoreArray((*[8]int16)(unsafe.Add(row, 2*x)))
		}
		p = unsafe.Add(p, refStride)
	}
}

// pred8VIM is the 2-D second pass over im in 8-column groups with int32
// accumulation. The pinned libaom av1/common/arm/convolve_neon.c
// (av1_convolve_2d_sr_neon) uses its six-tap kernel when the first and last
// coefficients are zero; the regular and smooth tables in av1/common/filter.h
// have that shape, while fractional sharp filters retain the eight-tap path.
func pred8VIM(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, im unsafe.Pointer, imStride, w, h int, k *[8]int16) int {
	if k[0] != 0 || k[7] != 0 {
		return pred8VIM8(dst, dstStride, src, srcStride, im, imStride, w, h, k)
	}
	c1 := archsimd.BroadcastInt32x4(int32(k[1]))
	c2 := archsimd.BroadcastInt32x4(int32(k[2]))
	c3 := archsimd.BroadcastInt32x4(int32(k[3]))
	c4 := archsimd.BroadcastInt32x4(int32(k[4]))
	c5 := archsimd.BroadcastInt32x4(int32(k[5]))
	c6 := archsimd.BroadcastInt32x4(int32(k[6]))
	bias := archsimd.BroadcastInt32x4(-261120)
	sh := archsimd.BroadcastInt32x4(-11)
	var acc archsimd.Uint16x8
	sad := 0
	for y := 0; y < h; y++ {
		p := unsafe.Add(im, (y+1)*imStride)
		x := 0
		for ; x+8 <= w; x += 8 {
			q := unsafe.Add(p, 2*x)
			v := archsimd.LoadInt16x8Array((*[8]int16)(q))
			lo := v.ExtendLo4ToInt32().MulAdd(c1, bias)
			hi := v.HiToLo().ExtendLo4ToInt32().MulAdd(c1, bias)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 1*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 2*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 3*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 4*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 5*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				s := *(*uint64)(unsafe.Add(src, y*srcStride+x))
				var pair [16]uint8
				*(*uint64)(unsafe.Pointer(&pair[0])) = s
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				*(*uint64)(unsafe.Add(dst, y*dstStride+x)) = out.ReshapeToUint64s().GetElem(0)
			}
		}
		if src != nil && y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}

// pred8VIM8 retains the full eight-tap path for kernels with nonzero outer
// coefficients.
func pred8VIM8(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, im unsafe.Pointer, imStride, w, h int, k *[8]int16) int {
	c0 := archsimd.BroadcastInt32x4(int32(k[0]))
	c1 := archsimd.BroadcastInt32x4(int32(k[1]))
	c2 := archsimd.BroadcastInt32x4(int32(k[2]))
	c3 := archsimd.BroadcastInt32x4(int32(k[3]))
	c4 := archsimd.BroadcastInt32x4(int32(k[4]))
	c5 := archsimd.BroadcastInt32x4(int32(k[5]))
	c6 := archsimd.BroadcastInt32x4(int32(k[6]))
	c7 := archsimd.BroadcastInt32x4(int32(k[7]))
	bias := archsimd.BroadcastInt32x4(-261120)
	sh := archsimd.BroadcastInt32x4(-11)
	var acc archsimd.Uint16x8
	sad := 0
	for y := 0; y < h; y++ {
		p := unsafe.Add(im, y*imStride)
		x := 0
		for ; x+8 <= w; x += 8 {
			q := unsafe.Add(p, 2*x)
			v := archsimd.LoadInt16x8Array((*[8]int16)(q))
			lo := v.ExtendLo4ToInt32().MulAdd(c0, bias)
			hi := v.HiToLo().ExtendLo4ToInt32().MulAdd(c0, bias)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 1*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c1, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c1, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 2*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 3*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 4*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 5*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 6*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			v = archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 7*imStride)))
			lo = v.ExtendLo4ToInt32().MulAdd(c7, lo)
			hi = v.HiToLo().ExtendLo4ToInt32().MulAdd(c7, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				// Eight output bytes in the low half; compare them with the
				// matching eight source bytes.
				s := *(*uint64)(unsafe.Add(src, y*srcStride+x))
				var pair [16]uint8
				*(*uint64)(unsafe.Pointer(&pair[0])) = s
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				*(*uint64)(unsafe.Add(dst, y*dstStride+x)) = out.ReshapeToUint64s().GetElem(0)
			}
		}
		if src != nil && y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}
