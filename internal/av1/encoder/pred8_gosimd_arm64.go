//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

func init() {
	pred8Impl = pred8SIMD
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
//
// Every kernel reads and writes through bounds-checked windows: ref is the
// reference plane with the block origin at ref[refOff], dst and src are the
// destination and source blocks from their first sample.

// pred8Out16 stores the 16 output bytes of a row group at dst[do:] or folds
// their absolute difference with src[so:] into the SAD accumulator.
func pred8Out16(out archsimd.Uint8x16, dst []byte, do int, src []byte, so int, acc archsimd.Uint16x8) archsimd.Uint16x8 {
	if src != nil {
		return acc.Add(widen16(absDiffU8x16(out, archsimd.LoadUint8x16Array((*[16]uint8)(src[so:so+16])))))
	}
	out.StoreArray((*[16]uint8)(dst[do : do+16]))
	return acc
}

// pred8Pack16 saturates lo and hi into one vector, the high half as SQXTUN2
// (SetElem of the low 64 bits) rather than a second SQXTUN joined by ZIP1.
func pred8Pack16(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	l := lo.SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	return l.SetElem(1, hi.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)).ToBits().ReshapeToUint8s()
}

// put64 stores v little-endian into b (one 8-byte store).
func put64(b *[8]byte, v uint64) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	b[4], b[5], b[6], b[7] = byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

// pred8Store8 stores the eight saturated bytes of v at dst[o:].
func pred8Store8(dst []byte, o int, v archsimd.Int16x8) {
	put64((*[8]byte)(dst[o:o+8]), v.SaturateToUint8().ReshapeToUint64s().GetElem(0))
}

func pred8SIMD(dst []byte, dstStride int, src []byte, srcStride int,
	ref []byte, refOff, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int {
	switch {
	case subX == 0 && subY == 0:
		return pred8Copy(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h)
	case subY == 0:
		return pred8H(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h, xk)
	case subX == 0:
		return pred8V(dst, dstStride, src, srcStride, ref, refOff, refStride, w, h, yk)
	}
	pred8IM(im[:], 64, ref, refOff-3*refStride, refStride, w, h+7, xk)
	return pred8VIM(dst, dstStride, src, srcStride, im[:], 64, w, h, yk)
}

func pred8Copy(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int) int {
	ref = ref[refOff:]
	if src == nil {
		pred8CopyRows(dst, dstStride, ref, refStride, w, h)
		return 0
	}
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint16x8
	sad := 0
	for y := 0; y < h; y++ {
		r, s := ref[y*refStride:y*refStride+w], src[y*srcStride:y*srcStride+w]
		s = s[:len(r)]
		for x := 0; x+16 <= len(r); x += 16 {
			acc = acc.Add(widen16(absDiffU8x16(archsimd.LoadUint8x16Array((*[16]uint8)(r[x:x+16])), archsimd.LoadUint8x16Array((*[16]uint8)(s[x:x+16])))))
		}
		if y&15 == 15 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	return sad + sumU16(acc)
}

// pred8CopyRows is svt_av1_convolve_2d_copy_sr_neon: copyBlockRows, one
// loop per block width with a bounds-checked window per row pair.
func pred8CopyRows(dst []byte, dstStride int, ref []byte, refStride, w, h int) {
	copyBlockRows(dst, dstStride, ref, refStride, w, h)
}

// pred8H is convolve_x_sr: per output row, each tap's window is one 16-byte
// load whose low and high halves feed the two 8-lane accumulators. Rows run
// in pairs: each 16-column group of a pair reads one bounds-checked window
// holding both rows' 23 bytes (the eight taps at constant offsets), and
// stores through one window per row pair; the 8-column tail reads the same
// windows.
func pred8H(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	ref = ref[refOff-3:]
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
	rn := refStride + 23
	for i := 0; i < h/2; i++ {
		ro, do, so := 2*i*refStride, 2*i*dstStride, 2*i*srcStride
		x := 0
		for ; x+16 <= w; x += 16 {
			rw := ref[ro+x : ro+x+rn : ro+x+rn]
			var o0, o1 archsimd.Uint8x16
			{
				q := (*[23]uint8)(rw)
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
				o0 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
			}
			{
				q := (*[23]uint8)(rw[refStride:])
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
				o1 = pred8Pack16(a0.Shift(sh), a1.Shift(sh))
			}
			acc = pred8Out16x2(o0, o1, dst, do+x, dstStride, src, so+x, srcStride, acc)
		}
		if x < w {
			rw := ref[ro+x : ro+x+rn : ro+x+rn]
			{
				q := (*[23]uint8)(rw)
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				pred8Store8(dst, do+x, a0.Shift(sh))
			}
			{
				q := (*[23]uint8)(rw[refStride:])
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				pred8Store8(dst, do+dstStride+x, a0.Shift(sh))
			}
		}
		if src != nil && i&7 == 7 {
			sad += sumU16(acc)
			acc = archsimd.Uint16x8{}
		}
	}
	if h&1 != 0 {
		y := h - 1
		ro, do, so := y*refStride, y*dstStride, y*srcStride
		x := 0
		for ; x+16 <= w; x += 16 {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, do+x, src, so+x, acc)
		}
		if x < w {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(dst, do+x, a0.Shift(sh))
		}
	}
	return sad + sumU16(acc)
}

// pred8Out16x2 is pred8Out16 for two rows (stride ds or ss apart), through
// one window per row pair.
func pred8Out16x2(o0, o1 archsimd.Uint8x16, dst []byte, do, ds int, src []byte, so, ss int, acc archsimd.Uint16x8) archsimd.Uint16x8 {
	if src != nil {
		n := ss + 16
		sw := src[so : so+n : so+n]
		acc = acc.Add(widen16(absDiffU8x16(o0, archsimd.LoadUint8x16Array((*[16]uint8)(sw)))))
		return acc.Add(widen16(absDiffU8x16(o1, archsimd.LoadUint8x16Array((*[16]uint8)(sw[ss:])))))
	}
	n := ds + 16
	dw := dst[do : do+n : do+n]
	o0.StoreArray((*[16]uint8)(dw))
	o1.StoreArray((*[16]uint8)(dw[ds:]))
	return acc
}

// pred8V is convolve_y_sr over 16-column (or a final 8-column) strips. The
// pinned libaom av1/common/arm/convolve_neon.c six-tap vertical kernel omits
// coefficients 0 and 7. Regular/smooth kernels have zero outer taps; fractional
// sharp kernels retain the full eight-tap path. Like libaom's
// convolve_y_sr_6tap/8tap_neon, a strip keeps its tap rows in registers and
// loads one new row per output row, each through one checked window.
func pred8V(dst []byte, dstStride int, src []byte, srcStride int, ref []byte, refOff, refStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	var acc archsimd.Uint16x8
	sad := 0
	if k[0] == 0 && k[7] == 0 {
		ref := ref[refOff-2*refStride:]
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(32)
		sh := archsimd.BroadcastInt16x8(-6)
		for x := 0; x < w; x += 16 {
			// Strip x: the first 5 rows load up front, then each output row
			// loads its bottom tap row and the rows slide down one register.
			b := x
			r0 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b : b+16]))
			r1 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+1*refStride : b+1*refStride+16]))
			r2 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+2*refStride : b+2*refStride+16]))
			r3 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+3*refStride : b+3*refStride+16]))
			r4 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+4*refStride : b+4*refStride+16]))
			if x+16 <= w {
				for y := 0; y < h; y++ {
					o := (y+5)*refStride + x
					r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
					a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
					a1 := widenHi8(r0).BitsToInt16().MulAdd(c1, bias)
					a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
					a1 = widenHi8(r1).BitsToInt16().MulAdd(c2, a1)
					a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
					a1 = widenHi8(r2).BitsToInt16().MulAdd(c3, a1)
					a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
					a1 = widenHi8(r3).BitsToInt16().MulAdd(c4, a1)
					a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
					a1 = widenHi8(r4).BitsToInt16().MulAdd(c5, a1)
					a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
					a1 = widenHi8(r5).BitsToInt16().MulAdd(c6, a1)
					acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
					r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
					if src != nil && y&15 == 15 {
						sad += sumU16(acc)
						acc = archsimd.Uint16x8{}
					}
				}
				continue
			}
			for y := 0; y < h; y++ {
				o := (y+5)*refStride + x
				r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
				a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				pred8Store8(dst, y*dstStride+x, a0.Shift(sh))
				r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			}
		}
		return sad + sumU16(acc)
	}
	ref = ref[refOff-3*refStride:]
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
	for x := 0; x < w; x += 16 {
		// Strip x: the first 7 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		b := x
		r0 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b : b+16]))
		r1 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+1*refStride : b+1*refStride+16]))
		r2 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+2*refStride : b+2*refStride+16]))
		r3 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+3*refStride : b+3*refStride+16]))
		r4 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+4*refStride : b+4*refStride+16]))
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+5*refStride : b+5*refStride+16]))
		r6 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[b+6*refStride : b+6*refStride+16]))
		if x+16 <= w {
			for y := 0; y < h; y++ {
				o := (y+7)*refStride + x
				r7 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
				a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
				a1 := widenHi8(r0).BitsToInt16().MulAdd(c0, bias)
				a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
				a1 = widenHi8(r1).BitsToInt16().MulAdd(c1, a1)
				a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(r2).BitsToInt16().MulAdd(c2, a1)
				a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(r3).BitsToInt16().MulAdd(c3, a1)
				a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(r4).BitsToInt16().MulAdd(c4, a1)
				a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(r5).BitsToInt16().MulAdd(c5, a1)
				a0 = r6.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(r6).BitsToInt16().MulAdd(c6, a1)
				a0 = r7.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
				a1 = widenHi8(r7).BitsToInt16().MulAdd(c7, a1)
				acc = pred8Out16(pred8Pack16(a0.Shift(sh), a1.Shift(sh)), dst, y*dstStride+x, src, y*srcStride+x, acc)
				r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
				if src != nil && y&15 == 15 {
					sad += sumU16(acc)
					acc = archsimd.Uint16x8{}
				}
			}
			continue
		}
		for y := 0; y < h; y++ {
			o := (y+7)*refStride + x
			r7 := archsimd.LoadUint8x16Array((*[16]uint8)(ref[o : o+16]))
			a0 := r0.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = r1.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = r2.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = r3.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = r4.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = r5.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = r6.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = r7.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			pred8Store8(dst, y*dstStride+x, a0.Shift(sh))
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
	return sad + sumU16(acc)
}

// pred8IM is the 2-D first pass: rows -3..h+3 of the block into im (row y
// at y*imStride), as int16 (a + 8194) >> 2. ref[refOff] is the first row's
// origin sample; each 16-column group reads one window of the row.
func pred8IM(im []int16, imStride int, ref []byte, refOff, refStride, w, rows int, k *[8]int16) {
	if uint(imStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(w) > rowsMax || uint(rows) > rowsMax {
		panic(errKernelArgs)
	}
	if k[0] == 0 && k[7] == 0 {
		ref := ref[refOff-2:]
		c1 := archsimd.BroadcastInt16x8(k[1] >> 1)
		c2 := archsimd.BroadcastInt16x8(k[2] >> 1)
		c3 := archsimd.BroadcastInt16x8(k[3] >> 1)
		c4 := archsimd.BroadcastInt16x8(k[4] >> 1)
		c5 := archsimd.BroadcastInt16x8(k[5] >> 1)
		c6 := archsimd.BroadcastInt16x8(k[6] >> 1)
		bias := archsimd.BroadcastInt16x8(8194)
		sh := archsimd.BroadcastInt16x8(-2)
		for y := 0; y < rows; y++ {
			ro, io := y*refStride, y*imStride
			x := 0
			for ; x+16 <= w; x += 16 {
				q := (*[21]uint8)(ref[ro+x : ro+x+21])
				v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
				a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a1 := widenHi8(v).BitsToInt16().MulAdd(c1, bias)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
				v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
				a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
				o := (*[16]int16)(im[io+x : io+x+16])
				a0.Shift(sh).StoreArray((*[8]int16)(o[0:8]))
				a1.Shift(sh).StoreArray((*[8]int16)(o[8:16]))
			}
			if x < w {
				q := (*[21]uint8)(ref[ro+x : ro+x+21])
				a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, bias)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
				a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
				a0.Shift(sh).StoreArray((*[8]int16)(im[io+x : io+x+8]))
			}
		}
		return
	}
	ref = ref[refOff-3:]
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
	for y := 0; y < rows; y++ {
		ro, io := y*refStride, y*imStride
		x := 0
		for ; x+16 <= w; x += 16 {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			v := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16]))
			a0 := v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a1 := widenHi8(v).BitsToInt16().MulAdd(c0, bias)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c1, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c2, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c3, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c4, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c5, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c6, a1)
			v = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23]))
			a0 = v.ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a1 = widenHi8(v).BitsToInt16().MulAdd(c7, a1)
			o := (*[16]int16)(im[io+x : io+x+16])
			a0.Shift(sh).StoreArray((*[8]int16)(o[0:8]))
			a1.Shift(sh).StoreArray((*[8]int16)(o[8:16]))
		}
		if x < w {
			q := (*[23]uint8)(ref[ro+x : ro+x+23])
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(q[0:16])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c0, bias)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[1:17])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c1, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[2:18])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c2, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[3:19])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c3, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[4:20])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c4, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[5:21])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c5, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[6:22])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c6, a0)
			a0 = archsimd.LoadUint8x16Array((*[16]uint8)(q[7:23])).ExtendLo8ToUint16().BitsToInt16().MulAdd(c7, a0)
			a0.Shift(sh).StoreArray((*[8]int16)(im[io+x : io+x+8]))
		}
	}
}

// pred8VIM is the 2-D second pass over im in 8-column groups with int32
// accumulation. The pinned libaom av1/common/arm/convolve_neon.c
// (av1_convolve_2d_sr_neon) uses its six-tap kernel when the first and last
// coefficients are zero; the regular and smooth tables in av1/common/filter.h
// have that shape, while fractional sharp filters retain the eight-tap path.
// A strip keeps its tap rows in registers and loads one new im row per
// output row.
func pred8VIM(dst []byte, dstStride int, src []byte, srcStride int, im []int16, imStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(imStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	if k[0] != 0 || k[7] != 0 {
		return pred8VIM8(dst, dstStride, src, srcStride, im, imStride, w, h, k)
	}
	im = im[1*imStride:]
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
	for x := 0; x+8 <= w; x += 8 {
		// Strip x: the first 5 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		r0 := archsimd.LoadInt16x8Array((*[8]int16)(im[x : x+8]))
		r1 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+1*imStride : x+1*imStride+8]))
		r2 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+2*imStride : x+2*imStride+8]))
		r3 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+3*imStride : x+3*imStride+8]))
		r4 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+4*imStride : x+4*imStride+8]))
		for y := 0; y < h; y++ {
			o := (y+5)*imStride + x
			r5 := archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
			lo := r0.ExtendLo4ToInt32().MulAdd(c1, bias)
			hi := r0.HiToLo().ExtendLo4ToInt32().MulAdd(c1, bias)
			lo = r1.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = r1.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			lo = r2.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = r2.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			lo = r3.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = r3.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			lo = r4.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = r4.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			lo = r5.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = r5.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				// Eight output bytes in the low half; compare them with the
				// matching eight source bytes.
				so := y*srcStride + x
				var pair [16]uint8
				*(*[8]uint8)(pair[:8]) = *(*[8]uint8)(src[so : so+8])
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				do := y*dstStride + x
				put64((*[8]byte)(dst[do:do+8]), out.ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			if src != nil && y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
	}
	return sad + sumU16(acc)
}

// pred8VIM8 retains the full eight-tap path for kernels with nonzero outer
// coefficients.
func pred8VIM8(dst []byte, dstStride int, src []byte, srcStride int, im []int16, imStride, w, h int, k *[8]int16) int {
	if uint(dstStride) > rowStrideMax || uint(srcStride) > rowStrideMax || uint(imStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
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
	for x := 0; x+8 <= w; x += 8 {
		// Strip x: the first 7 rows load up front, then each output row
		// loads its bottom tap row and the rows slide down one register.
		r0 := archsimd.LoadInt16x8Array((*[8]int16)(im[x : x+8]))
		r1 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+1*imStride : x+1*imStride+8]))
		r2 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+2*imStride : x+2*imStride+8]))
		r3 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+3*imStride : x+3*imStride+8]))
		r4 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+4*imStride : x+4*imStride+8]))
		r5 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+5*imStride : x+5*imStride+8]))
		r6 := archsimd.LoadInt16x8Array((*[8]int16)(im[x+6*imStride : x+6*imStride+8]))
		for y := 0; y < h; y++ {
			o := (y+7)*imStride + x
			r7 := archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
			lo := r0.ExtendLo4ToInt32().MulAdd(c0, bias)
			hi := r0.HiToLo().ExtendLo4ToInt32().MulAdd(c0, bias)
			lo = r1.ExtendLo4ToInt32().MulAdd(c1, lo)
			hi = r1.HiToLo().ExtendLo4ToInt32().MulAdd(c1, hi)
			lo = r2.ExtendLo4ToInt32().MulAdd(c2, lo)
			hi = r2.HiToLo().ExtendLo4ToInt32().MulAdd(c2, hi)
			lo = r3.ExtendLo4ToInt32().MulAdd(c3, lo)
			hi = r3.HiToLo().ExtendLo4ToInt32().MulAdd(c3, hi)
			lo = r4.ExtendLo4ToInt32().MulAdd(c4, lo)
			hi = r4.HiToLo().ExtendLo4ToInt32().MulAdd(c4, hi)
			lo = r5.ExtendLo4ToInt32().MulAdd(c5, lo)
			hi = r5.HiToLo().ExtendLo4ToInt32().MulAdd(c5, hi)
			lo = r6.ExtendLo4ToInt32().MulAdd(c6, lo)
			hi = r6.HiToLo().ExtendLo4ToInt32().MulAdd(c6, hi)
			lo = r7.ExtendLo4ToInt32().MulAdd(c7, lo)
			hi = r7.HiToLo().ExtendLo4ToInt32().MulAdd(c7, hi)
			r16 := lo.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s().
				InterleaveLo(hi.Shift(sh).SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
			out := r16.SaturateToUint8()
			if src != nil {
				// Eight output bytes in the low half; compare them with the
				// matching eight source bytes.
				so := y*srcStride + x
				var pair [16]uint8
				*(*[8]uint8)(pair[:8]) = *(*[8]uint8)(src[so : so+8])
				acc = acc.Add(absDiffU8x16(out, archsimd.LoadUint8x16Array(&pair)).ExtendLo8ToUint16())
			} else {
				do := y*dstStride + x
				put64((*[8]byte)(dst[do:do+8]), out.ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
			if src != nil && y&15 == 15 {
				sad += sumU16(acc)
				acc = archsimd.Uint16x8{}
			}
		}
	}
	return sad + sumU16(acc)
}
