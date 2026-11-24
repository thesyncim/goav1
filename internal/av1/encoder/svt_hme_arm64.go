//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// svtDownsample2DArch is svtDownsample2D on NEON: the 2x2 sums of 16 output
// samples are two row-widened adds folded by pairwise adds, then
// (sum + 2) >> 2 exactly as svt_aom_downsample_2d_c.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	two := archsimd.BroadcastUint16x8(2)
	// The count is built once: ShiftAllRight(2) rebuilds it (MOVD + INS + DUP)
	// at every use. The high half saturates as UQXTN2 rather than a second
	// UQXTN joined by ZIP1.
	sh2 := archsimd.BroadcastInt16x8(-2)
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		x := 0
		if len(r0) >= 2*w && len(r1) >= 2*w {
			// One 32-byte window per input row and 16 outputs: j = 2x
			// steps by 32 so each window is one compare.
			r0, r1 := r0[:2*w], r1[:2*w]
			r1 = r1[:len(r0)]
			for j := 0; j+32 <= len(r0); j += 32 {
				q0, q1 := (*[32]uint8)(r0[j:j+32]), (*[32]uint8)(r1[j:j+32])
				a := archsimd.LoadUint8x16Array((*[16]uint8)(q0[0:16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(q1[0:16]))
				c := archsimd.LoadUint8x16Array((*[16]uint8)(q0[16:32]))
				e := archsimd.LoadUint8x16Array((*[16]uint8)(q1[16:32]))
				lo := a.ExtendLo8ToUint16().Add(b.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(a).Add(widenHi8(b)))
				hi := c.ExtendLo8ToUint16().Add(e.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(c).Add(widenHi8(e)))
				lf := lo.Add(two).Shift(sh2).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				hf := hi.Add(two).Shift(sh2).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				out := lf.SetElem(1, hf.GetElem(0)).ToBits().ReshapeToUint8s()
				x = j >> 1
				out.StoreArray((*[16]uint8)(d[x : x+16]))
			}
			x = w &^ 15
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
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var c0, c1, c2, c3 archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		so, ro := r*srcStride, r*refStride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
		p := (*[19]uint8)(ref[ro : ro+19])
		c0 = c0.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16])))))
		c1 = c1.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17])))))
		c2 = c2.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18])))))
		c3 = c3.Add(widen16(absDiffU8x16(s, archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19])))))
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
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	// USHL by a hoisted -8: stock archsimd lowers ShiftAllRight to a
	// per-use DUP + USHL rather than USHR #8.
	hiShift := archsimd.BroadcastInt16x8(-8)
	for r := 0; r < rows; r++ {
		// Written out rather than through load16 / absDiffU8x16: each inlined
		// helper call leaves an inline-mark NOP in the loop. One window per
		// row: the eight reference loads sit at constant offsets inside it.
		so, ro := r*srcStride, r*refStride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[so : so+16]))
		p := (*[23]uint8)(ref[ro : ro+23])
		q := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
		d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))
		d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
		s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))
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
	lo := (*[4]uint32)(acc[0:4])
	hi := (*[4]uint32)(acc[4:8])
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
	if uint(rows) > rowsMax || uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	hiShift := archsimd.BroadcastInt16x8(-8)
	for r := 0; r < rows; r++ {
		so, ro := r*srcStride, r*refStride
		sw := (*[32]uint8)(src[so : so+32])
		rw := (*[39]uint8)(ref[ro : ro+39])
		for c := 0; c < 32; c += 16 {
			s := archsimd.LoadUint8x16Array((*[16]uint8)(sw[c : c+16]))
			p := (*[23]uint8)(rw[c : c+23])
			q := archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))
			d := s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))
			d = s.Max(q).Sub(s.Min(q)).ReshapeToUint16s()
			s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))
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
	lo := (*[4]uint32)(acc[0:4])
	hi := (*[4]uint32)(acc[4:8])
	archsimd.LoadUint32x4Array(lo).Add(ta.ExtendLo4ToUint32().ConcatAddPairs(ta.HiToLo().ExtendLo4ToUint32())).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(tb.ExtendLo4ToUint32().ConcatAddPairs(tb.HiToLo().ExtendLo4ToUint32())).StoreArray(hi)
}

// svtHMEColSums enables svtHMESearchColSum (the MIN-based eight-position
// SAD) on NEON, where the absolute difference is MAX, MIN, SUB.
const svtHMEColSums = true

// svtColTotalsArch is svtColTotals sixteen columns at a time. Each row
// accumulates as wrapping uint16 byte pairs S (even column + 256 * odd
// column) plus the odd columns H, so a row costs one add and one
// shift-add; the even columns' totals are S - 256*H (each total is at most
// 16*255) and the two are zipped back into column order.
func svtColTotalsArch(cw *[svtHMEMaxCols]uint16, ref []byte, s2, rows, span int) {
	if uint(rows) > rowsMax || uint(s2) > rowStrideMax {
		panic(errKernelArgs)
	}
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	for c := 0; c < span; c += 16 {
		var sv, hv archsimd.Uint16x8
		col := ref[c:]
		for r := 0; r < rows; r++ {
			o := r * s2
			v := archsimd.LoadUint8x16Array((*[16]uint8)(col[o : o+16])).ReshapeToUint16s()
			sv, hv = sv.Add(v), hv.Add(v.Shift(hiShift))
		}
		ev := sv.Sub(hv.Shift(loShift))
		ev.InterleaveLo(hv).StoreArray((*[8]uint16)(cw[c:]))
		ev.InterleaveHi(hv).StoreArray((*[8]uint16)(cw[c+8:]))
	}
}

// svtColTotalsSlideArch is svtColTotalsSlide sixteen columns at a time.
func svtColTotalsSlideArch(cw *[svtHMEMaxCols]uint16, out, in []byte, span int) {
	out = out[:span]
	in = in[:len(out)]
	for c := 0; c <= len(out)-16; c += 16 {
		o := archsimd.LoadUint8x16Array((*[16]uint8)(out[c:]))
		n := archsimd.LoadUint8x16Array((*[16]uint8)(in[c:]))
		lo := archsimd.LoadUint16x8Array((*[8]uint16)(cw[c:]))
		hi := archsimd.LoadUint16x8Array((*[8]uint16)(cw[c+8:]))
		lo = lo.Add(n.ExtendLo8ToUint16()).Sub(o.ExtendLo8ToUint16())
		hi = hi.Add(n.HiToLo().ExtendLo8ToUint16()).Sub(o.HiToLo().ExtendLo8ToUint16())
		lo.StoreArray((*[8]uint16)(cw[c:]))
		hi.StoreArray((*[8]uint16)(cw[c+8:]))
	}
}

// svtHMEColSumRow scores one HME search row: the positions x = 0..w-1 (w a
// multiple of 8, w + bw <= svtHMEMaxCols) of the bw-wide (16 or 32),
// rows-tall block s (stride s2) against ref[x:], eight at a time, from the
// source total srcSum and the row's reference column totals cw. It
// returns the smallest key t<<7 | x (t the SAD; t <= 16*32*255, x < 128): the
// row's minimum SAD and its first position. When t is not nil, t[x/8][k] is
// the SAD of position x+k (svtHMEEightColSumScalar).
//
// Per eight positions the sums of minima accumulate as svtSAD16xhx8's byte
// pairs (one MIN per row and position), and the reference window totals are
// the column totals' sliding sums: with C0 the first eight column totals and
// Cb the eight starting bw columns later, position k's window is
// T + sum(Cb[j] - C0[j], j < k), T the first window, as an exclusive prefix
// sum over int16 lanes (each term is at most 7*16*255 in magnitude). The
// positions loop inside the kernel so a search row is one call.
func svtHMEColSumRow(t *[svtHMEMaxCols / 8][8]uint32, s []byte, s2 int, ref []byte, cw *[svtHMEMaxCols]uint16, bw, rows, w int, srcSum uint32) uint32 {
	if uint(s2) > rowStrideMax || uint(rows) > rowsMax || uint(w) > svtHMEMaxCols || (bw != 16 && bw != 32) {
		panic(errKernelArgs)
	}
	// Each step reads one 16-byte source window and one 23-byte reference
	// window (the eight positions at constant offsets inside it).
	// Only the shift count stays in a register across the row loop; the
	// reduction's constants load from one table each step (kept live, the
	// register allocator reloaded them inside the row loop).
	hiShift := archsimd.BroadcastInt16x8(-8)
	kc := &svtHMEKeyConsts
	key := ^uint32(0)
	cShift := bw >> 5 // 16-column steps per row: 1 << cShift
	cMask := cShift
	steps := rows << cShift
	for x := 0; x < w; x += 8 {
		rx := ref[x:]
		// The rows x 16-column steps run as one loop; the first seeds the
		// accumulators, so no zero vector stays live across it.
		sv := archsimd.LoadUint8x16Array((*[16]uint8)(s[0:16]))
		p := (*[23]uint8)(rx[0:23])
		var d archsimd.Uint16x8
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
		s0, h0 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
		s1, h1 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
		s2v, h2 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
		s3, h3 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
		s4, h4 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
		s5, h5 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
		s6, h6 := d, d.Shift(hiShift)
		d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
		s7, h7 := d, d.Shift(hiShift)
		for i := 1; i < steps; i++ {
			off := (i>>cShift)*s2 + (i&cMask)<<4
			sw := (*[16]uint8)(s[off : off+16])
			p = (*[23]uint8)(rx[off : off+23])
			sv = archsimd.LoadUint8x16Array(sw)
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[0:16]))).ReshapeToUint16s()
			s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[1:17]))).ReshapeToUint16s()
			s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[2:18]))).ReshapeToUint16s()
			s2v, h2 = s2v.Add(d), h2.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[3:19]))).ReshapeToUint16s()
			s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[4:20]))).ReshapeToUint16s()
			s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[5:21]))).ReshapeToUint16s()
			s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[6:22]))).ReshapeToUint16s()
			s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
			d = sv.Min(archsimd.LoadUint8x16Array((*[16]uint8)(p[7:23]))).ReshapeToUint16s()
			s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
		}
		// Minima: two folds in uint16 (each lane at most 2*16*4*255), the
		// exact byte-pair correction S - 255*H as one MLA by 0xFF01, the
		// last fold in uint32.
		m255 := archsimd.LoadUint32x4Array((*[4]uint32)(kc[0:4])).ReshapeToUint16s()
		sa := s0.ConcatAddPairs(s1).ConcatAddPairs(s2v.ConcatAddPairs(s3))
		ha := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3))
		sb := s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7))
		hb := h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7))
		ta := ha.MulAdd(m255, sa)
		tb := hb.MulAdd(m255, sb)
		mlo := ta.ExtendLo4ToUint32().ConcatAddPairs(ta.HiToLo().ExtendLo4ToUint32())
		mhi := tb.ExtendLo4ToUint32().ConcatAddPairs(tb.HiToLo().ExtendLo4ToUint32())

		// Reference window totals.
		var c0, win, cb archsimd.Uint16x8
		if bw == 32 {
			c := (*[40]uint16)(cw[x : x+40])
			c0 = archsimd.LoadUint16x8Array((*[8]uint16)(c[0:8]))
			win = c0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[8:16])))
			win = win.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[16:24]))).Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[24:32])))
			cb = archsimd.LoadUint16x8Array((*[8]uint16)(c[32:40]))
		} else {
			c := (*[24]uint16)(cw[x : x+24])
			c0 = archsimd.LoadUint16x8Array((*[8]uint16)(c[0:8]))
			win = c0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(c[8:16])))
			cb = archsimd.LoadUint16x8Array((*[8]uint16)(c[16:24]))
		}
		// T in every uint32 lane (column-total lanes at most 4*16*255).
		tv := win.ExtendLo4ToUint32().Add(win.HiToLo().ExtendLo4ToUint32())
		tv = tv.ConcatAddPairs(tv)
		tv = tv.ConcatAddPairs(tv)
		// Exclusive prefix of Cb - C0: lane k gets the sum of lanes j < k.
		zero := archsimd.LoadUint32x4Array((*[4]uint32)(kc[16:20])).ReshapeToUint8s()
		e := cb.Sub(c0).ReshapeToUint8s()
		e = e.ConcatShiftBytesRight(zero, 14)
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 14).ReshapeToUint16s()).ReshapeToUint8s()
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 12).ReshapeToUint16s()).ReshapeToUint8s()
		e = e.ReshapeToUint16s().Add(e.ConcatShiftBytesRight(zero, 8).ReshapeToUint16s()).ReshapeToUint8s()
		ei := e.ReshapeToUint16s().BitsToInt16()
		// t - srcSum (signed: the reference windows minus twice the minima).
		tlo := tv.Add(ei.ExtendLo4ToInt32().ToBits()).Sub(mlo.Add(mlo)).BitsToInt32()
		thi := tv.Add(ei.HiToLo().ExtendLo4ToInt32().ToBits()).Sub(mhi.Add(mhi)).BitsToInt32()
		if t != nil {
			tx := &t[x/8]
			tlo.ToBits().StoreArray((*[4]uint32)(tx[0:4]))
			thi.ToBits().StoreArray((*[4]uint32)(tx[4:8]))
			for k := range tx {
				tx[k] += srcSum
			}
		}
		// The smallest (t - srcSum)<<7 + k, in int32 (|t - srcSum| <=
		// 16*32*255), then srcSum<<7 + x: adding the constants keeps the
		// order, and no srcSum or x vector is live across the row loop.
		k128 := archsimd.LoadUint32x4Array((*[4]uint32)(kc[4:8])).BitsToInt32()
		klo := archsimd.LoadUint32x4Array((*[4]uint32)(kc[8:12])).BitsToInt32()
		khi := archsimd.LoadUint32x4Array((*[4]uint32)(kc[12:16])).BitsToInt32()
		kmin := tlo.MulAdd(k128, klo).Min(thi.MulAdd(k128, khi)).ReduceMin()
		key = min(key, uint32(int(kmin)+int(srcSum)<<7+x))
	}
	return key
}

// svtHMEKeyConsts are svtHMEColSumRow's reduction constants: 0xFF01 in
// every uint16 lane, 128 in every uint32 lane, positions 0..7, zero.
var svtHMEKeyConsts = [20]uint32{
	0xFF01FF01, 0xFF01FF01, 0xFF01FF01, 0xFF01FF01,
	128, 128, 128, 128,
	0, 1, 2, 3, 4, 5, 6, 7,
	0, 0, 0, 0,
}

// svtBlockSumSubArch is svtBlockSumSub for bw a multiple of 16 (rows <= 16):
// byte pairs accumulate as wrapping uint16 S plus high bytes H, corrected
// as S - 255*H per lane (each lane's pair total is at most 2*2*16*255).
func svtBlockSumSubArch(s []byte, s2, bw, rows int) uint32 {
	if uint(rows) > rowsMax || uint(s2) > rowStrideMax || uint(bw) > rowsMax {
		panic(errKernelArgs)
	}
	hiShift := archsimd.BroadcastInt16x8(-8)
	var sa, ha archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		row := s[r*s2 : r*s2+bw]
		for c := 0; c+16 <= len(row); c += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(row[c : c+16])).ReshapeToUint16s()
			sa, ha = sa.Add(v), ha.Add(v.Shift(hiShift))
		}
	}
	t := sa.Sub(ha.Shift(archsimd.BroadcastInt16x8(8))).Add(ha)
	return t.ExtendLo4ToUint32().Add(t.HiToLo().ExtendLo4ToUint32()).ReduceSum()
}
