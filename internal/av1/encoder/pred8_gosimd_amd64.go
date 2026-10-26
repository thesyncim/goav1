//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	if gosimdKernelsSupported() {
		pred8Impl = pred8AVX2
	}
}

// pred8AVX2 is pred8SIMD (the NEON kernel's arithmetic, see
// pred8_gosimd_arm64.go) on AVX2, in the shape of SVT-AV1's
// Source/Lib/ASM_AVX2/convolve_avx2.c / dav1d's put_8tap_8bpc:
//
//   - Horizontal: VPSHUFB gathers the (x+t, x+t+1) byte pairs of each tap pair
//     and VPMADDUBSW multiplies them by the halved taps packed as int8 pairs,
//     four multiplies per sixteen outputs. Each 128-bit lane holds eight
//     outputs: lane 0 the columns at x, lane 1 those at x+8 (or, for 8-wide
//     blocks, the next row).
//   - Vertical: rows zero-extend to words (VPMOVZXBW) and the next row is
//     shifted into the high byte, so one VPMADDUBSW applies a vertical tap
//     pair (amd64 archsimd has no VPUNPCKLBW).
//   - 2-D: the first pass stores int16 intermediate rows; the second
//     word-interleaves rows t and t+1 (VPUNPCKLWD/HWD) and VPMADDWD applies a
//     full-tap pair in int32, VPACKSSDW restoring column order per lane.
//
// The halved-tap int16 sums are exact (no AV1 kernel's positive halved taps
// sum above 92, so no pair product or total leaves int16), and the roundings
// are the NEON kernel's single biased shifts. Every entry clears the upper
// YMM state before returning.
func pred8AVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int,
	ref unsafe.Pointer, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	var sad int
	switch {
	case subX == 0 && subY == 0:
		sad = pred8CopyAVX2(dst, dstStride, src, srcStride, ref, refStride, w, h)
	case subY == 0:
		sad = pred8HAVX2(dst, dstStride, src, srcStride, ref, refStride, w, h, xk)
	case subX == 0:
		sad = pred8VAVX2(dst, dstStride, src, srcStride, ref, refStride, w, h, yk)
	default:
		imp := unsafe.Pointer(&im[0])
		if yk[0] == 0 && yk[7] == 0 {
			// Six vertical taps read block rows -2..h+2 only.
			pred8IMAVX2(imp, unsafe.Add(ref, -2*refStride), refStride, w, h+5, xk)
		} else {
			pred8IMAVX2(imp, unsafe.Add(ref, -3*refStride), refStride, w, h+7, xk)
		}
		sad = pred8VIMAVX2(dst, dstStride, src, srcStride, imp, w, h, yk)
	}
	archsimd.ClearAVXUpperBits()
	return sad
}

// pred8HShuf are the VPSHUFB pair gathers of tap pairs (0,1), (2,3), (4,5)
// and (6,7) over a lane whose first byte is column x-3: output i takes bytes
// 2p+i and 2p+i+1.
var pred8HShuf = func() (s [4][32]int8) {
	for p := range 4 {
		for i := range 8 {
			for lane := range 2 {
				s[p][16*lane+2*i] = int8(2*p + i)
				s[p][16*lane+2*i+1] = int8(2*p + i + 1)
			}
		}
	}
	return s
}()

// pred8HShuf6 are the gathers of tap pairs (1,2), (3,4) and (5,6): the
// six-tap form (dav1d PUT_6TAP_H) for kernels whose outer taps are zero
// (regular and smooth).
var pred8HShuf6 = func() (s [3][32]int8) {
	for p := range 3 {
		for i := range 8 {
			for lane := range 2 {
				s[p][16*lane+2*i] = int8(2*p + 1 + i)
				s[p][16*lane+2*i+1] = int8(2*p + 2 + i)
			}
		}
	}
	return s
}()

// pred8HTaps holds a horizontal pass's gathers and tap pairs. A six-tap
// kernel uses three of the four (s3 and t3 unused).
type pred8HTaps struct {
	s0, s1, s2, s3 archsimd.Int8x32
	t0, t1, t2, t3 archsimd.Int8x32
	six            bool
}

func pred8LoadHTaps(k *[8]int16) pred8HTaps {
	if k[0] == 0 && k[7] == 0 {
		return pred8HTaps{
			s0: archsimd.LoadInt8x32Array(&pred8HShuf6[0]), s1: archsimd.LoadInt8x32Array(&pred8HShuf6[1]),
			s2: archsimd.LoadInt8x32Array(&pred8HShuf6[2]),
			t0: pred8TapPairB(k[1], k[2]), t1: pred8TapPairB(k[3], k[4]), t2: pred8TapPairB(k[5], k[6]),
			six: true,
		}
	}
	return pred8HTaps{
		s0: archsimd.LoadInt8x32Array(&pred8HShuf[0]), s1: archsimd.LoadInt8x32Array(&pred8HShuf[1]),
		s2: archsimd.LoadInt8x32Array(&pred8HShuf[2]), s3: archsimd.LoadInt8x32Array(&pred8HShuf[3]),
		t0: pred8TapPairB(k[0], k[1]), t1: pred8TapPairB(k[2], k[3]),
		t2: pred8TapPairB(k[4], k[5]), t3: pred8TapPairB(k[6], k[7]),
	}
}

// pred8PackIdx and pred8PackLanes narrow sixteen int16 lanes holding 0..255
// to bytes in lane order: VPSHUFB keeps each lane's low bytes in its low
// eight bytes, VPERMD joins the two halves.
var (
	pred8PackIdx   = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	pred8PackLanes = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}
	pred8Pack8Idx  = [16]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
)

// pred8TapPairB is the VPMADDUBSW operand of the halved taps (a, b).
func pred8TapPairB(a, b int16) archsimd.Int8x32 {
	return archsimd.BroadcastInt16x16(int16(uint8(int8(a>>1))) | (b>>1)<<8).AsInt8x32()
}

// pred8TapPairD is the VPMADDWD operand of the full taps (a, b).
func pred8TapPairD(a, b int16) archsimd.Int16x16 {
	return archsimd.BroadcastInt32x8(int32(uint16(a)) | int32(b)<<16).AsInt16x16()
}

// pred8Pack16 clamps v to 0..255 and narrows it to sixteen bytes.
func pred8Pack16(v archsimd.Int16x16, zero, c255 archsimd.Int16x16) archsimd.Uint8x16 {
	b := v.Max(zero).Min(c255).AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&pred8PackIdx))
	return b.AsUint32x8().Permute(archsimd.LoadUint32x8Array(&pred8PackLanes)).AsUint8x32().GetLo()
}

// pred8Ld2 loads sixteen bytes at a into the low lane and sixteen at b into
// the high lane.
func pred8Ld2(a, b unsafe.Pointer) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	return z.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(a))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(b)))
}

// pred8H8 is the halved-tap horizontal sum of the eight outputs of each lane
// of v (lane bytes start at column x-3).
func pred8H8(v archsimd.Uint8x32, t *pred8HTaps) archsimd.Int16x16 {
	a := v.PermuteOrZeroGrouped(t.s0).DotProductPairsSaturated(t.t0)
	b := v.PermuteOrZeroGrouped(t.s1).DotProductPairsSaturated(t.t1)
	c := v.PermuteOrZeroGrouped(t.s2).DotProductPairsSaturated(t.t2)
	if t.six {
		return a.Add(b).Add(c)
	}
	d := v.PermuteOrZeroGrouped(t.s3).DotProductPairsSaturated(t.t3)
	return a.Add(b).Add(c.Add(d))
}

// pred8Out stores sixteen output bytes or adds their SAD against src to acc.
func pred8Out(out archsimd.Uint8x16, dst unsafe.Pointer, dstOff int, src unsafe.Pointer, srcOff int, acc archsimd.Uint64x2) archsimd.Uint64x2 {
	if src != nil {
		return acc.Add(out.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, srcOff)))))
	}
	out.StoreArray((*[16]uint8)(unsafe.Add(dst, dstOff)))
	return acc
}

func pred8CopyAVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int) int {
	if src == nil {
		pred8CopyRowsAVX2(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		r := unsafe.Add(ref, y*refStride)
		for x := 0; x < w; x += 16 {
			acc = pred8Out(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, x))), nil, 0, src, y*srcStride+x, acc)
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8CopyRowsAVX2 is svt_av1_convolve_2d_copy_sr_avx2: one loop per block
// width (the row pointers are formed by index: one past the last row would
// trip checkptr).
func pred8CopyRowsAVX2(dst unsafe.Pointer, dstStride int, ref unsafe.Pointer, refStride, w, h int) {
	switch w {
	case 8:
		for y := range h {
			*(*uint64)(unsafe.Add(dst, y*dstStride)) = *(*uint64)(unsafe.Add(ref, y*refStride))
		}
	case 16:
		for y := range h {
			archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ref, y*refStride))).StoreArray((*[16]uint8)(unsafe.Add(dst, y*dstStride)))
		}
	case 32:
		for y := range h {
			archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(ref, y*refStride))).StoreArray((*[32]uint8)(unsafe.Add(dst, y*dstStride)))
		}
	case 64:
		for y := range h {
			r, d := unsafe.Add(ref, y*refStride), unsafe.Add(dst, y*dstStride)
			archsimd.LoadUint8x32Array((*[32]uint8)(r)).StoreArray((*[32]uint8)(d))
			archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(r, 32))).StoreArray((*[32]uint8)(unsafe.Add(d, 32)))
		}
	default:
		for y := range h {
			d, r := unsafe.Add(dst, y*dstStride), unsafe.Add(ref, y*refStride)
			x := 0
			for ; x+16 <= w; x += 16 {
				archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, x))).StoreArray((*[16]uint8)(unsafe.Add(d, x)))
			}
			if x < w {
				*(*uint64)(unsafe.Add(d, x)) = *(*uint64)(unsafe.Add(r, x))
			}
		}
	}
}

// pred8HAVX2 is convolve_x_sr: clip((a + 34) >> 6). Sixteen-column strips
// put columns x and x+8 in the two lanes; an 8-wide block puts rows y and
// y+1 there instead.
func pred8HAVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int, k *[8]int16) int {
	taps := pred8LoadHTaps(k)
	bias := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	p := unsafe.Add(ref, -3)
	if w == 8 {
		y := 0
		for ; y+2 <= h; y += 2 {
			q := unsafe.Add(p, y*refStride)
			a := pred8H8(pred8Ld2(q, unsafe.Add(q, refStride)), &taps)
			out := pred8Pack16(a.Add(bias).ShiftAllRight(6), zero, c255).AsUint64x2()
			*(*uint64)(unsafe.Add(dst, y*dstStride)) = out.GetElem(0)
			*(*uint64)(unsafe.Add(dst, (y+1)*dstStride)) = out.GetElem(1)
		}
		if y < h {
			q := unsafe.Add(p, y*refStride)
			a := pred8H8(pred8Ld2(q, q), &taps)
			*(*uint64)(unsafe.Add(dst, y*dstStride)) = pred8Pack16(a.Add(bias).ShiftAllRight(6), zero, c255).AsUint64x2().GetElem(0)
		}
		return 0
	}
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		q := unsafe.Add(p, y*refStride)
		x := 0
		for ; x+16 <= w; x += 16 {
			a := pred8H8(pred8Ld2(unsafe.Add(q, x), unsafe.Add(q, x+8)), &taps)
			acc = pred8Out(pred8Pack16(a.Add(bias).ShiftAllRight(6), zero, c255), dst, y*dstStride+x, src, y*srcStride+x, acc)
		}
		if x < w {
			a := pred8H8(pred8Ld2(unsafe.Add(q, x), unsafe.Add(q, x)), &taps)
			*(*uint64)(unsafe.Add(dst, y*dstStride+x)) = pred8Pack16(a.Add(bias).ShiftAllRight(6), zero, c255).AsUint64x2().GetElem(0)
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8VExt is the sixteen samples at p zero-extended to words.
func pred8VExt(p unsafe.Pointer) archsimd.Uint16x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendToUint16()
}

// pred8VPair interleaves two zero-extended rows into VPMADDUBSW byte pairs
// (a in the low byte, b in the high byte of each word).
func pred8VPair(a, b archsimd.Uint16x16) archsimd.Uint8x32 {
	return a.Or(b.ShiftAllLeft(8)).AsUint8x32()
}

// pred8VAVX2 is convolve_y_sr: clip((a + 32) >> 6), sixteen columns per
// step (an 8-wide strip keeps the low eight). As dav1d's put_6tap_v /
// put_8tap_v, each strip walks two output rows at a time with the row pairs
// held in registers: output y uses pairs (y+i, y+i+1) for the even taps i,
// output y+2 the same pairs shifted down by one, so a row step loads two
// rows and builds two new pairs.
func pred8VAVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, ref unsafe.Pointer, refStride, w, h int, k *[8]int16) int {
	bias := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	s := refStride
	six := k[0] == 0 && k[7] == 0
	var acc archsimd.Uint64x2
	for x := 0; x < w; x += 16 {
		wide := x+16 <= w
		if six {
			// Taps 1..6 over rows y-2..y+3.
			t0, t1, t2 := pred8TapPairB(k[1], k[2]), pred8TapPairB(k[3], k[4]), pred8TapPairB(k[5], k[6])
			p := unsafe.Add(ref, x-2*s)
			r0, r1, r2 := pred8VExt(p), pred8VExt(unsafe.Add(p, s)), pred8VExt(unsafe.Add(p, 2*s))
			r3, r4 := pred8VExt(unsafe.Add(p, 3*s)), pred8VExt(unsafe.Add(p, 4*s))
			e0, e1 := pred8VPair(r0, r1), pred8VPair(r2, r3)
			o0, o1 := pred8VPair(r1, r2), pred8VPair(r3, r4)
			prev := r4
			for y := 0; y < h; y += 2 {
				r5 := pred8VExt(unsafe.Add(p, (y+5)*s))
				r6 := pred8VExt(unsafe.Add(p, (y+6)*s))
				e2, o2 := pred8VPair(prev, r5), pred8VPair(r5, r6)
				a := e0.DotProductPairsSaturated(t0).Add(e1.DotProductPairsSaturated(t1)).Add(e2.DotProductPairsSaturated(t2))
				b := o0.DotProductPairsSaturated(t0).Add(o1.DotProductPairsSaturated(t1)).Add(o2.DotProductPairsSaturated(t2))
				acc = pred8VStore(a, b, bias, zero, c255, dst, dstStride, src, srcStride, x, y, y+1 < h, wide, acc)
				e0, e1, o0, o1, prev = e1, e2, o1, o2, r6
			}
			continue
		}
		t0, t1 := pred8TapPairB(k[0], k[1]), pred8TapPairB(k[2], k[3])
		t2, t3 := pred8TapPairB(k[4], k[5]), pred8TapPairB(k[6], k[7])
		p := unsafe.Add(ref, x-3*s)
		r0, r1, r2 := pred8VExt(p), pred8VExt(unsafe.Add(p, s)), pred8VExt(unsafe.Add(p, 2*s))
		r3, r4 := pred8VExt(unsafe.Add(p, 3*s)), pred8VExt(unsafe.Add(p, 4*s))
		r5, r6 := pred8VExt(unsafe.Add(p, 5*s)), pred8VExt(unsafe.Add(p, 6*s))
		e0, e1, e2 := pred8VPair(r0, r1), pred8VPair(r2, r3), pred8VPair(r4, r5)
		o0, o1, o2 := pred8VPair(r1, r2), pred8VPair(r3, r4), pred8VPair(r5, r6)
		prev := r6
		for y := 0; y < h; y += 2 {
			r7 := pred8VExt(unsafe.Add(p, (y+7)*s))
			r8 := pred8VExt(unsafe.Add(p, (y+8)*s))
			e3, o3 := pred8VPair(prev, r7), pred8VPair(r7, r8)
			a := e0.DotProductPairsSaturated(t0).Add(e1.DotProductPairsSaturated(t1)).
				Add(e2.DotProductPairsSaturated(t2).Add(e3.DotProductPairsSaturated(t3)))
			b := o0.DotProductPairsSaturated(t0).Add(o1.DotProductPairsSaturated(t1)).
				Add(o2.DotProductPairsSaturated(t2).Add(o3.DotProductPairsSaturated(t3)))
			acc = pred8VStore(a, b, bias, zero, c255, dst, dstStride, src, srcStride, x, y, y+1 < h, wide, acc)
			e0, e1, e2, o0, o1, o2, prev = e1, e2, e3, o1, o2, o3, r8
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8VStore rounds and stores (or SADs) output rows y (sum a) and y+1
// (sum b) of the strip at x; a narrow strip stores eight columns.
// Row y+1 exists unless the block height is odd and y is its last row.
func pred8VStore(a, b, bias, zero, c255 archsimd.Int16x16, dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride, x, y int, two, wide bool, acc archsimd.Uint64x2) archsimd.Uint64x2 {
	oa := pred8Pack16(a.Add(bias).ShiftAllRight(6), zero, c255)
	if !wide {
		*(*uint64)(unsafe.Add(dst, y*dstStride+x)) = oa.AsUint64x2().GetElem(0)
		if two {
			ob := pred8Pack16(b.Add(bias).ShiftAllRight(6), zero, c255)
			*(*uint64)(unsafe.Add(dst, (y+1)*dstStride+x)) = ob.AsUint64x2().GetElem(0)
		}
		return acc
	}
	acc = pred8Out(oa, dst, y*dstStride+x, src, y*srcStride+x, acc)
	if two {
		acc = pred8Out(pred8Pack16(b.Add(bias).ShiftAllRight(6), zero, c255), dst, (y+1)*dstStride+x, src, (y+1)*srcStride+x, acc)
	}
	return acc
}

// pred8IMAVX2 is the 2-D first pass: rows -3..h+3 of the block into im (row
// stride 64) as int16 (a + 8194) >> 2.
func pred8IMAVX2(im unsafe.Pointer, ref unsafe.Pointer, refStride, w, rows int, k *[8]int16) {
	const imStride = 64 * 2
	taps := pred8LoadHTaps(k)
	bias := archsimd.BroadcastInt16x16(8194)
	p := unsafe.Add(ref, -3)
	if w == 8 {
		y := 0
		for ; y+2 <= rows; y += 2 {
			q := unsafe.Add(p, y*refStride)
			a := pred8H8(pred8Ld2(q, unsafe.Add(q, refStride)), &taps).Add(bias).ShiftAllRight(2)
			a.GetLo().StoreArray((*[8]int16)(unsafe.Add(im, y*imStride)))
			a.GetHi().StoreArray((*[8]int16)(unsafe.Add(im, (y+1)*imStride)))
		}
		if y < rows {
			q := unsafe.Add(p, y*refStride)
			a := pred8H8(pred8Ld2(q, q), &taps).Add(bias).ShiftAllRight(2)
			a.GetLo().StoreArray((*[8]int16)(unsafe.Add(im, y*imStride)))
		}
		return
	}
	for y := 0; y < rows; y++ {
		q := unsafe.Add(p, y*refStride)
		row := unsafe.Add(im, y*imStride)
		x := 0
		for ; x+16 <= w; x += 16 {
			a := pred8H8(pred8Ld2(unsafe.Add(q, x), unsafe.Add(q, x+8)), &taps)
			a.Add(bias).ShiftAllRight(2).StoreArray((*[16]int16)(unsafe.Add(row, 2*x)))
		}
		if x < w {
			a := pred8H8(pred8Ld2(unsafe.Add(q, x), unsafe.Add(q, x)), &taps)
			a.Add(bias).ShiftAllRight(2).GetLo().StoreArray((*[8]int16)(unsafe.Add(row, 2*x)))
		}
	}
}

// pred8VIMAVX2 is the 2-D second pass over im (row stride 64):
// clip((sum(k*im) - 261120) >> 11) in int32 lanes, sixteen columns per step
// (an 8-wide strip runs on 128-bit vectors).
func pred8VIMAVX2(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, im unsafe.Pointer, w, h int, k *[8]int16) int {
	const imStride = 64 * 2
	// A six-tap kernel's intermediate starts at block row -2 and its pairs
	// are taps (1,2), (3,4), (5,6); the fourth pair is zero.
	six := k[0] == 0 && k[7] == 0
	d0, d1 := pred8TapPairD(k[0], k[1]), pred8TapPairD(k[2], k[3])
	d2, d3 := pred8TapPairD(k[4], k[5]), pred8TapPairD(k[6], k[7])
	if six {
		d0, d1, d2 = pred8TapPairD(k[1], k[2]), pred8TapPairD(k[3], k[4]), pred8TapPairD(k[5], k[6])
	}
	bias := archsimd.BroadcastInt32x8(-261120)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	var acc archsimd.Uint64x2
	for y := 0; y < h; y++ {
		p := unsafe.Add(im, y*imStride)
		x := 0
		for ; x+16 <= w; x += 16 {
			q := unsafe.Add(p, 2*x)
			m0 := archsimd.LoadInt16x16Array((*[16]int16)(q))
			m1 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, imStride)))
			m2 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 2*imStride)))
			m3 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 3*imStride)))
			m4 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 4*imStride)))
			m5 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 5*imStride)))
			lo := m0.InterleaveLoGrouped(m1).DotProductPairs(d0).Add(m2.InterleaveLoGrouped(m3).DotProductPairs(d1)).
				Add(m4.InterleaveLoGrouped(m5).DotProductPairs(d2))
			hi := m0.InterleaveHiGrouped(m1).DotProductPairs(d0).Add(m2.InterleaveHiGrouped(m3).DotProductPairs(d1)).
				Add(m4.InterleaveHiGrouped(m5).DotProductPairs(d2))
			if !six {
				m6 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 6*imStride)))
				m7 := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(q, 7*imStride)))
				lo = lo.Add(m6.InterleaveLoGrouped(m7).DotProductPairs(d3))
				hi = hi.Add(m6.InterleaveHiGrouped(m7).DotProductPairs(d3))
			}
			v := lo.Add(bias).ShiftAllRight(11).SaturateToInt16ConcatGrouped(hi.Add(bias).ShiftAllRight(11))
			acc = pred8Out(pred8Pack16(v, zero, c255), dst, y*dstStride+x, src, y*srcStride+x, acc)
		}
		if x < w {
			pred8VIM8Cols(unsafe.Add(dst, y*dstStride+x), unsafe.Add(p, 2*x), d0.GetLo(), d1.GetLo(), d2.GetLo(), d3.GetLo(), six)
		}
	}
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// pred8VIM8Cols is pred8VIMAVX2 for one row of eight columns on 128-bit
// vectors.
func pred8VIM8Cols(dst, q unsafe.Pointer, d0, d1, d2, d3 archsimd.Int16x8, six bool) {
	const imStride = 64 * 2
	m0 := archsimd.LoadInt16x8Array((*[8]int16)(q))
	m1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, imStride)))
	m2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 2*imStride)))
	m3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 3*imStride)))
	m4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 4*imStride)))
	m5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 5*imStride)))
	lo := m0.InterleaveLo(m1).DotProductPairs(d0).Add(m2.InterleaveLo(m3).DotProductPairs(d1)).
		Add(m4.InterleaveLo(m5).DotProductPairs(d2))
	hi := m0.InterleaveHi(m1).DotProductPairs(d0).Add(m2.InterleaveHi(m3).DotProductPairs(d1)).
		Add(m4.InterleaveHi(m5).DotProductPairs(d2))
	if !six {
		m6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 6*imStride)))
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(q, 7*imStride)))
		lo = lo.Add(m6.InterleaveLo(m7).DotProductPairs(d3))
		hi = hi.Add(m6.InterleaveHi(m7).DotProductPairs(d3))
	}
	bias := archsimd.BroadcastInt32x4(-261120)
	v := lo.Add(bias).ShiftAllRight(11).SaturateToInt16Concat(hi.Add(bias).ShiftAllRight(11))
	v = v.Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	*(*uint64)(dst) = v.AsUint8x16().PermuteOrZero(archsimd.LoadInt8x16Array(&pred8Pack8Idx)).AsUint64x2().GetElem(0)
}
