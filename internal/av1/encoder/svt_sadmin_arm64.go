//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtSAD16xhx8Min is svtSAD16xhx8 on the identity
//
//	sum |a - b| = sum a + sum b - 2 * sum min(a, b),
//
// which replaces the MAX, MIN, SUB absolute difference of every row and
// position by one MIN: the per-position work is MIN plus the byte-pair
// accumulate (ADD, USHR, ADD), four ops instead of six. The source total and
// the reference column sums are shared by the eight positions: the column
// sums over the 23 reference columns come from the position-0 and
// position-7 loads (bytes 0..15 and 7..22), and the eight 16-wide window
// sums are a sliding sum over them in the epilogue.
//
// Every accumulator is a wrapping uint16 byte-pair sum S plus a high-byte
// sum H, corrected as S - 255*H after the folds; the final SAD, source and
// window totals are all at most 16*16*255 = 65280, so uint16 arithmetic is
// exact (the intermediate A + W may wrap but the ring arithmetic recovers
// the final value). rows <= 16.
func svtSAD16xhx8Min(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+22]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Uint16x8
	var h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Uint16x8
	hiShift := archsimd.BroadcastInt16x8(-8)
	// The shared sums run in their own pass over the (L1-resident) rows so
	// the position loop keeps the register footprint of the MAX-MIN kernel.
	var sa, ha, sq, hq, sr, hr archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		s16 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, r*srcStride))).ReshapeToUint16s()
		sa, ha = sa.Add(s16), ha.Add(s16.Shift(hiShift))
		p := unsafe.Add(rp, r*refStride)
		q16 := archsimd.LoadUint8x16Array((*[16]uint8)(p)).ReshapeToUint16s()
		sq, hq = sq.Add(q16), hq.Add(q16.Shift(hiShift))
		q16 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 7))).ReshapeToUint16s()
		sr, hr = sr.Add(q16), hr.Add(q16.Shift(hiShift))
	}
	for r := 0; r < rows; r++ {
		s := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, r*srcStride)))
		p := unsafe.Add(rp, r*refStride)
		q := archsimd.LoadUint8x16Array((*[16]uint8)(p))
		d := s.Min(q).ReshapeToUint16s()
		s0, h0 = s0.Add(d), h0.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 1)))
		d = s.Min(q).ReshapeToUint16s()
		s1, h1 = s1.Add(d), h1.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2)))
		d = s.Min(q).ReshapeToUint16s()
		s2, h2 = s2.Add(d), h2.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3)))
		d = s.Min(q).ReshapeToUint16s()
		s3, h3 = s3.Add(d), h3.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 4)))
		d = s.Min(q).ReshapeToUint16s()
		s4, h4 = s4.Add(d), h4.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 5)))
		d = s.Min(q).ReshapeToUint16s()
		s5, h5 = s5.Add(d), h5.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 6)))
		d = s.Min(q).ReshapeToUint16s()
		s6, h6 = s6.Add(d), h6.Add(d.Shift(hiShift))
		q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 7)))
		d = s.Min(q).ReshapeToUint16s()
		s7, h7 = s7.Add(d), h7.Add(d.Shift(hiShift))
	}
	shl8 := archsimd.BroadcastInt16x8(8)
	// Lane k of the folded vectors is position k: the min sums M_k.
	sv := s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3)).
		ConcatAddPairs(s4.ConcatAddPairs(s5).ConcatAddPairs(s6.ConcatAddPairs(s7)))
	hv := h0.ConcatAddPairs(h1).ConcatAddPairs(h2.ConcatAddPairs(h3)).
		ConcatAddPairs(h4.ConcatAddPairs(h5).ConcatAddPairs(h6.ConcatAddPairs(h7)))
	m := sv.Sub(hv.Shift(shl8)).Add(hv)
	// The source total in every lane.
	av := sa.Sub(ha.Shift(shl8)).Add(ha)
	av = av.ConcatAddPairs(av)
	av = av.ConcatAddPairs(av)
	av = av.ConcatAddPairs(av)
	// Reference column sums V[0..22]: the even columns from S - 256*H, the
	// odd ones from H, zipped back into column order.
	qlo := sq.Sub(hq.Shift(shl8))
	va := qlo.InterleaveLo(hq) // V[0..7]
	vb := qlo.InterleaveHi(hq) // V[8..15]
	rlo := sr.Sub(hr.Shift(shl8))
	vc := rlo.InterleaveHi(hr) // V[15..22]
	// X[i] = V[i] + V[i+8] for i = 0..15 (X[15] is never read).
	var zero archsimd.Uint8x16
	u0 := va.Add(vb)
	u1 := vb.Add(zero.ConcatShiftBytesRight(vc.ReshapeToUint8s(), 2).ReshapeToUint16s())
	// W_k = sum X[k..k+7]: the 16-wide window at position k, as EXT slides
	// over the X[0..15] pair (receiver = high half).
	u0b, u1b := u0.ReshapeToUint8s(), u1.ReshapeToUint8s()
	w := u0.Add(u1b.ConcatShiftBytesRight(u0b, 2).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 4).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 6).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 8).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 10).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 12).ReshapeToUint16s())
	w = w.Add(u1b.ConcatShiftBytesRight(u0b, 14).ReshapeToUint16s())
	t := av.Add(w).Sub(m).Sub(m)
	lo := (*[4]uint32)(unsafe.Pointer(&acc[0]))
	hi := (*[4]uint32)(unsafe.Pointer(&acc[4]))
	archsimd.LoadUint32x4Array(lo).Add(t.ExtendLo4ToUint32()).StoreArray(lo)
	archsimd.LoadUint32x4Array(hi).Add(t.HiToLo().ExtendLo4ToUint32()).StoreArray(hi)
}
