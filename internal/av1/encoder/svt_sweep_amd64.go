//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtEightSAD8x8Sub is svt_ext_eight_sad_calculation_8x8_16x16 (SUB_SAD,
// Source/Lib/ASM_AVX2/compute_sad_intrin_avx2.c shape) for one 16x16 block:
// VPSADBW of a 16-byte row yields the left and right 8x8 row sums, so the
// four 8x8 SUB_SADs of a position are two accumulators; the bests update
// with strict '<' per 8x8 and sad16[k] is the four quadrants' sum.
func svtEightSAD8x8Sub(src, ref []byte, stride int, best *[4]uint32, bestMV *[4][2]int16, mvX, mvY int, sad16 *[8]uint32) {
	_ = src[14*stride+15]
	_ = ref[14*stride+22]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	s2 := 2 * stride
	a0 := load16(sp)
	a1 := load16(unsafe.Add(sp, s2))
	a2 := load16(unsafe.Add(sp, 2*s2))
	a3 := load16(unsafe.Add(sp, 3*s2))
	a4 := load16(unsafe.Add(sp, 4*s2))
	a5 := load16(unsafe.Add(sp, 5*s2))
	a6 := load16(unsafe.Add(sp, 6*s2))
	a7 := load16(unsafe.Add(sp, 7*s2))
	for k := range 8 {
		r := unsafe.Add(rp, k)
		top := a0.SumOf8AbsDiff(load16(r)).
			Add(a1.SumOf8AbsDiff(load16(unsafe.Add(r, s2)))).
			Add(a2.SumOf8AbsDiff(load16(unsafe.Add(r, 2*s2)))).
			Add(a3.SumOf8AbsDiff(load16(unsafe.Add(r, 3*s2))))
		bot := a4.SumOf8AbsDiff(load16(unsafe.Add(r, 4*s2))).
			Add(a5.SumOf8AbsDiff(load16(unsafe.Add(r, 5*s2)))).
			Add(a6.SumOf8AbsDiff(load16(unsafe.Add(r, 6*s2)))).
			Add(a7.SumOf8AbsDiff(load16(unsafe.Add(r, 7*s2))))
		q := [4]uint32{
			uint32(top.GetElem(0)) << 1, uint32(top.GetElem(1)) << 1,
			uint32(bot.GetElem(0)) << 1, uint32(bot.GetElem(1)) << 1,
		}
		mv := [2]int16{int16(mvX + k), int16(mvY)}
		for c, s := range q {
			if s < best[c] {
				best[c] = s
				bestMV[c] = mv
			}
		}
		sad16[k] = q[0] + q[1] + q[2] + q[3]
	}
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
			(*[4][2]int16)(me.bestMV[pu8:pu8+4]), mvX, mvY, &sad16[n])
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
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	_ = src[63*stride+63]
	_ = ref[63*stride+63]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	s2 := 2 * stride
	s8 := 8 * stride
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		a := unsafe.Add(sp, off)
		r := unsafe.Add(rp, off)
		var acc archsimd.Uint64x4
		for row := range 4 {
			o := row * s2
			p := archsimd.Uint8x32{}.SetLo(load16(unsafe.Add(a, o))).SetHi(load16(unsafe.Add(a, o+s8)))
			q := archsimd.Uint8x32{}.SetLo(load16(unsafe.Add(r, o))).SetHi(load16(unsafe.Add(r, o+s8)))
			acc = acc.Add(p.SumOf8AbsDiff(q))
		}
		acc = acc.ShiftAllLeft(1)
		v := acc.AsInt32x8()
		sad := v.GetLo().SaturateToInt16Concat(v.GetHi()).AsUint32x4()
		pu8 := 21 + n*4
		bp := (*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8]))
		mp := (*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8]))
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
