//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtDownsample2DArch is svtDownsample2D on NEON: the 2x2 sums of 16 output
// samples are two row-widened adds folded by pairwise adds, then
// (sum + 2) >> 2 exactly as svt_aom_downsample_2d_c.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	two := archsimd.BroadcastUint16x8(2)
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		x := 0
		if len(r0) >= 2*w && len(r1) >= 2*w {
			p0 := unsafe.Pointer(&r0[0])
			p1 := unsafe.Pointer(&r1[0])
			for ; x+16 <= w; x += 16 {
				a := load16(unsafe.Add(p0, 2*x))
				b := load16(unsafe.Add(p1, 2*x))
				c := load16(unsafe.Add(p0, 2*x+16))
				e := load16(unsafe.Add(p1, 2*x+16))
				lo := a.ExtendLo8ToUint16().Add(b.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(a).Add(widenHi8(b)))
				hi := c.ExtendLo8ToUint16().Add(e.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(c).Add(widenHi8(e)))
				lo = lo.Add(two).ShiftAllRight(2)
				hi = hi.Add(two).ShiftAllRight(2)
				out := lo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
				out.StoreArray((*[16]uint8)(unsafe.Pointer(&d[x])))
			}
		}
		for ; x < w; x++ {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
}

// sad16xNx4 returns the SADs of the 16-wide, rows-tall source block against
// reference origins ref+0..ref+3 (svt_sad_loop_kernel's per-row position
// batch).
func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+18]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var c0, c1, c2, c3 archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		s := load16(unsafe.Add(sp, r*srcStride))
		p := unsafe.Add(rp, r*refStride)
		c0 = c0.Add(widen16(absDiffU8x16(s, load16(p))))
		c1 = c1.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 1)))))
		c2 = c2.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 2)))))
		c3 = c3.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 3)))))
	}
	return sumU16(c0), sumU16(c1), sumU16(c2), sumU16(c3)
}

// svtSAD16xhx8 is the eight-position step of SVT-AV1's
// svt_sad_loop_kernel16xh_neon_dotprod / svt_sad_loop_kernel32xh_neon_dotprod
// (Source/Lib/ASM_NEON_DOTPROD/compute_sad_neon_dotprod.c, the
// search_area_width % 8 == 0 branch): the SADs of the 16-wide, rows-tall
// source block against ref+0..ref+7, as two sad16xhx4d batches that share the
// source row. rows <= 16.
//
// archsimd has no UABD/UDOT/UADALP, so each row's absolute differences are
// MAX-MIN and accumulate as uint16 lanes two ways: S sums the byte pairs as
// lo + 256*hi (wrapping) and H sums the high bytes. The pair sum lo + hi is
// S - 255*H modulo 2^16, exact because a position's total is at most
// 16*16*255 = 65280. The pairwise reduction is linear, so it runs on S and H
// before the single correction.
func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+22]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	// USHL by a hoisted -8: stock archsimd lowers ShiftAllRight to a
	// per-use DUP + USHL rather than USHR #8.
	hiShift := archsimd.BroadcastInt16x8(-8)
	for r := 0; r < rows; r++ {
		// Written out rather than through load16 / absDiffU8x16: each inlined
		// helper call leaves an inline-mark NOP in the loop.
		s := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, r*srcStride)))
		p := unsafe.Add(rp, r*refStride)
		q := archsimd.LoadUint8x16Array((*[16]uint8)(p))
		d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 1)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 4)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 5)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 6)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 7)))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
	}
	// horizontal_add_4d_u32x4 for both batches at once: lane k of the
	// reduced vectors is position k.
	sv := s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3)).
		ConcatAddPairs(s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7)))
	hv := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3)).
		ConcatAddPairs(h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7)))
	t := sv.Sub(hv.Shift(archsimd.BroadcastInt16x8(8))).Add(hv)
	lo := (*[4]uint32)(unsafe.Pointer(&acc[0]))
	hi := (*[4]uint32)(unsafe.Pointer(&acc[4]))
	archsimd.LoadUint32x4Array(lo).Add(t.ExtendLo4ToUint32()).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(t.HiToLo().ExtendLo4ToUint32()).StoreArray(hi)
}

// svtSAD32xhx8 is the eight-position step of
// svt_sad_loop_kernel32xh_neon_dotprod (sad32xhx4d_neon_dotprod twice): the
// SADs of the 32-wide, rows-tall source block against ref+0..ref+7, both
// 16-column halves accumulating into the same byte-pair accumulators. The
// two in-register folds keep every uint16 lane at most 2*16*4*255 = 65280,
// so the S - 255*H correction is exact there and the last fold runs in
// uint32. rows <= 16.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	_ = src[(rows-1)*srcStride+31]
	_ = ref[(rows-1)*refStride+38]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	hiShift := archsimd.BroadcastInt16x8(-8)
	for r := 0; r < rows; r++ {
		for c := 0; c < 32; c += 16 {
			s := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, r*srcStride+c)))
			p := unsafe.Add(rp, r*refStride+c)
			q := archsimd.LoadUint8x16Array((*[16]uint8)(p))
			d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 1)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 4)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 5)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 6)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 7)))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
		}
	}
	// Two folds in uint16 (lane pairs of positions 2j, 2j+1), the exact
	// correction, then the last fold in uint32.
	shl8 := archsimd.BroadcastInt16x8(8)
	sa := s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3))
	ha := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3))
	sb := s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7))
	hb := h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7))
	ta := sa.Sub(ha.Shift(shl8)).Add(ha)
	tb := sb.Sub(hb.Shift(shl8)).Add(hb)
	lo := (*[4]uint32)(unsafe.Pointer(&acc[0]))
	hi := (*[4]uint32)(unsafe.Pointer(&acc[4]))
	archsimd.LoadUint32x4Array(lo).Add(ta.ExtendLo4ToUint32().ConcatAddPairs(ta.HiToLo().ExtendLo4ToUint32())).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(tb.ExtendLo4ToUint32().ConcatAddPairs(tb.HiToLo().ExtendLo4ToUint32())).StoreArray(hi)
}
