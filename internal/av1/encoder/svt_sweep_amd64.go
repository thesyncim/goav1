//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// svtEightSAD8x8Sub is svt_ext_all_sad_calculation_8x8_16x16_avx2's SUB_SAD
// pass (Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c) for one 16x16 block
// at eight positions. archsimd has no VMPSADBW or VPHMINPOSUW, so each
// position is one 256-bit VPSADBW per source row pair, in the
// svt_ext_sad_calculation_8x8_16x16_avx2_intrin shape: rows r and r+8 share
// a vector, the 64-bit lanes accumulate [TL TR BL BR] and VPACKSSDW folds
// them to four 32-bit SADs. The per-8x8 bests are a vector compare and blend
// per position in position order, which keeps the first strict minimum as
// _mm_minpos_epu16 does; sad16[k] is the four quadrants' sum. An 8x8 SUB_SAD
// is at most 2*32*255, so the signed pack is exact. The rows are
// bounds-checked windows taken once: the source's 16 bytes and the
// reference's 23 (the eight positions at constant offsets).
func svtEightSAD8x8Sub(src, ref []byte, stride int, best *[4]uint32, bestMV *[4]uint32, mvX, mvY int, sad16 *[8]uint32) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	s8 := 8 * stride
	a0 := sweepRows16(src, 0, s8)
	a1 := sweepRows16(src, s2, s2+s8)
	a2 := sweepRows16(src, 2*s2, 2*s2+s8)
	a3 := sweepRows16(src, 3*s2, 3*s2+s8)
	w0, w1 := (*[23]uint8)(ref[0:23]), (*[23]uint8)(ref[s8:s8+23])
	w2, w3 := (*[23]uint8)(ref[s2:s2+23]), (*[23]uint8)(ref[s2+s8:s2+s8+23])
	w4, w5 := (*[23]uint8)(ref[2*s2:2*s2+23]), (*[23]uint8)(ref[2*s2+s8:2*s2+s8+23])
	w6, w7 := (*[23]uint8)(ref[3*s2:3*s2+23]), (*[23]uint8)(ref[3*s2+s8:3*s2+s8+23])
	bestV := archsimd.LoadUint32x4Array(best)
	bestMVV := archsimd.LoadUint32x4Array(bestMV)
	for k := range 8 {
		acc := a0.SumOf8AbsDiff(sweepPos16(w0, w1, k)).
			Add(a1.SumOf8AbsDiff(sweepPos16(w2, w3, k))).
			Add(a2.SumOf8AbsDiff(sweepPos16(w4, w5, k))).
			Add(a3.SumOf8AbsDiff(sweepPos16(w6, w7, k)))
		acc = acc.ShiftAllLeft(1)
		v := acc.AsInt32x8()
		sad := v.GetLo().SaturateToInt16Concat(v.GetHi()).AsUint32x4()
		lt := sad.Less(bestV)
		bestV = sad.IfElse(lt, bestV)
		bestMVV = archsimd.BroadcastUint32x4(svtPackMV(mvX+k, mvY)).IfElse(lt, bestMVV)
		h := acc.GetLo().Add(acc.GetHi())
		sad16[k] = uint32(h.GetElem(0) + h.GetElem(1))
	}
	bestV.StoreArray(best)
	bestMVV.StoreArray(bestMV)
	archsimd.ClearAVXUpperBits()
}

// sweepRows16 joins the 16-byte rows of s at a and b into one vector.
func sweepRows16(s []byte, a, b int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(s[a : a+16]))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(s[b : b+16])))
}

// sweepPos16 joins position k's 16 bytes of the reference row windows w and
// v into one vector.
func sweepPos16(w, v *[23]uint8, k int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(w[k : k+16]))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(v[k : k+16])))
}

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16 (SUB_SAD)
// over the superblock at src[0] / ref[0]: the 16x16 blocks in
// p_eight_sad16x16 order, each scored at eight positions, then the 16x16
// bests.
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		pu8 := 21 + n*4
		o := y16*stride + x16
		svtEightSAD8x8Sub(src[o:], ref[o:], stride, (*[4]uint32)(me.bestSAD[pu8:pu8+4]),
			(*[4]uint32)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
		for k, s := range sad16[n] {
			me.update(5+n, s, mvX+k, mvY)
		}
	}
}

// svtSweepPoint1 is open_loop_me_get_search_point_results_block for one
// search position with sub_sad: per 16x16 block,
// svt_ext_sad_calculation_8x8_16x16_avx2_intrin (compute_sad_intrin_avx2.c)
// pairs source rows r and r+8 in one 256-bit VPSADBW, so the 64-bit lanes
// accumulate [TL TR BL BR]; VPACKSSDW folds them to four 32-bit SADs for the
// compare-and-blend 8x8 best update, and their sum updates the 16x16 best.
// The 32x32 and 64x64 SADs are the sums of the 16x16s
// (svt_ext_sad_calculation_32x32_64x64). ref needs no slack past the 64x64
// block. An 8x8 SUB_SAD is at most 2*32*255, so the signed pack is exact.
// Rows r and r+8 are one bounds-checked window per array.
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	s8 := 8 * stride
	n8 := s8 + 16
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		var acc archsimd.Uint64x4
		for row := range 4 {
			o := off + row*s2
			aw, rw := src[o:o+n8:o+n8], ref[o:o+n8:o+n8]
			p := archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(aw))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(aw[s8:])))
			q := archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(rw))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(rw[s8:])))
			acc = acc.Add(p.SumOf8AbsDiff(q))
		}
		acc = acc.ShiftAllLeft(1)
		v := acc.AsInt32x8()
		sad := v.GetLo().SaturateToInt16Concat(v.GetHi()).AsUint32x4()
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv.IfElse(lt, archsimd.LoadUint32x4Array(mp)).StoreArray(mp)
		h := acc.GetLo().Add(acc.GetHi())
		s := uint32(h.GetElem(0) + h.GetElem(1))
		sad16[n] = s
		me.update(5+n, s, mvX, mvY)
	}
	archsimd.ClearAVXUpperBits()
	var sad64 uint32
	for q := range 4 {
		s32 := sad16[q*4] + sad16[q*4+1] + sad16[q*4+2] + sad16[q*4+3]
		me.update(1+q, s32, mvX, mvY)
		sad64 += s32
	}
	me.update(0, sad64, mvX, mvY)
}

// sweepPoint1Arch searches one position: the AVX2 kernel when the 64x64
// windows fit, else the generic path.
func (me *realtimeSBMultiSizeME) sweepPoint1Arch(src []byte, stride int, ref []byte, srcBase, refBase, mvX, mvY int) {
	if encAVX2 && srcBase >= 0 && refBase >= 0 && srcBase+63*stride+64 <= len(src) && refBase+63*stride+64 <= len(ref) {
		svtSweepPoint1(me, src[srcBase:], ref[refBase:], stride, mvX, mvY)
		return
	}
	me.sweepSearchPoint1(src, stride, ref, stride, srcBase, refBase, mvX, mvY)
}
