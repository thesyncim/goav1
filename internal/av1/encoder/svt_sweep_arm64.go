//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod
// with sub_sad (Source/Lib/ASM_NEON_DOTPROD/sad_neon_dotprod.c) over the
// superblock at src[0] / ref[0]: for each 16x16 block (offsets[] order),
// per position k = 0..7 the four 8x8 SUB_SADs land in one vector
// [TL TR BL BR], doubled (the dotprod kernel's mult = 2), the 8x8 bests are
// a vector compare-and-select (vcltq + vbslq), and the sum (vaddvq) is
// p_eight_sad16x16[n][k] and updates the 16x16 best.
//
// archsimd has no UABD/UDOT, so a row's MAX-MIN absolute differences
// accumulate as wrapping uint16 byte pairs S and high bytes H, corrected to
// the pair sums as S - 256*H + H after the pairwise folds (exact: an 8x8
// SUB_SAD before doubling is at most 32*255).
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	_ = src[63*stride+63]
	_ = ref[63*stride+63+7]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	s2 := 2 * stride
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	var mvs [8]archsimd.Uint32x4
	for k := range 8 {
		mvs[k] = archsimd.BroadcastUint32x4(svtPackMV(mvX+k, mvY))
	}
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		sbp := unsafe.Add(sp, y16*stride+x16)
		a0 := archsimd.LoadUint8x16Array((*[16]uint8)(sbp))
		a1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, s2)))
		a2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 2*s2)))
		a3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 3*s2)))
		a4 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 4*s2)))
		a5 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 5*s2)))
		a6 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 6*s2)))
		a7 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sbp, 7*s2)))
		pu8 := 21 + n*4
		best := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		bestMV := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		rb := unsafe.Add(rp, y16*stride+x16)
		for k := range 8 {
			r := unsafe.Add(rb, k)
			q := archsimd.LoadUint8x16Array((*[16]uint8)(r))
			d := a0.Max(q).Sub(a0.Min(q)).ReshapeToUint16s()
			st, ht := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, s2)))
			d = a1.Max(q).Sub(a1.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 2*s2)))
			d = a2.Max(q).Sub(a2.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 3*s2)))
			d = a3.Max(q).Sub(a3.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 4*s2)))
			d = a4.Max(q).Sub(a4.Min(q)).ReshapeToUint16s()
			sb, hb := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 5*s2)))
			d = a5.Max(q).Sub(a5.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 6*s2)))
			d = a6.Max(q).Sub(a6.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, 7*s2)))
			d = a7.Max(q).Sub(a7.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			sv := st.ConcatAddPairs(sb)
			hv := ht.ConcatAddPairs(hb)
			sv = sv.ConcatAddPairs(sv)
			hv = hv.ConcatAddPairs(hv)
			sad := sv.Sub(hv.Shift(loShift)).Add(hv).ExtendLo4ToUint32()
			sad = sad.Add(sad)
			lt := sad.Less(best)
			best = sad.IfElse(lt, best)
			bestMV = mvs[k].IfElse(lt, bestMV)
			sad16[n][k] = sad.ReduceSum()
		}
		best.StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		bestMV.StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		me.update8(5+n, &sad16[n], mvX, mvY)
	}
}

// svtSweepPoint1 is svtAllSAD8x8_16x16Sub for one search position, the
// svt_ext_sad_calculation_8x8_16x16 (sub_sad) remainder of
// open_loop_me_get_search_point_results_block: the four 8x8 SUB_SADs of each
// 16x16 block in one vector with a compare-and-select best update, then the
// 16x16, 32x32 and 64x64 sums. ref needs no slack past the 64x64 block.
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	_ = src[63*stride+63]
	_ = ref[63*stride+63]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	s2 := 2 * stride
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		a := unsafe.Add(sp, off)
		r := unsafe.Add(rp, off)
		var st, ht, sb, hb archsimd.Uint16x8
		for row := range 4 {
			p := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(a, row*s2)))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, row*s2)))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
		}
		for row := 4; row < 8; row++ {
			p := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(a, row*s2)))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(r, row*s2)))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
		}
		sv := st.ConcatAddPairs(sb)
		hv := ht.ConcatAddPairs(hb)
		sv = sv.ConcatAddPairs(sv)
		hv = hv.ConcatAddPairs(hv)
		sad := sv.Sub(hv.Shift(loShift)).Add(hv).ExtendLo4ToUint32()
		sad = sad.Add(sad)
		pu8 := 21 + n*4
		best := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		bestMV := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		mv.IfElse(lt, bestMV).StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		s := sad.ReduceSum()
		sad16[n] = s
		me.update(5+n, s, mvX, mvY)
	}
	var sad64 uint32
	for q := range 4 {
		s32 := sad16[q*4] + sad16[q*4+1] + sad16[q*4+2] + sad16[q*4+3]
		me.update(1+q, s32, mvX, mvY)
		sad64 += s32
	}
	me.update(0, sad64, mvX, mvY)
}

// sweepPoint1Arch searches one position: the vector kernel when the 64x64
// windows fit, else the generic path.
func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	if srcBase >= 0 && refBase >= 0 && srcBase+63*stride+64 <= len(src) && refBase+63*stride+64 <= len(ref) {
		svtSweepPoint1(me, src[srcBase:], ref[refBase:], stride, mvX, mvY)
		return
	}
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
