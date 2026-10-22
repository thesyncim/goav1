//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod
// with sub_sad (Source/Lib/ASM_NEON_DOTPROD/sad_neon_dotprod.c) over the
// superblock at src[0] / ref[0]: for each 16x16 block, per position
// k = 0..7 the four 8x8 SUB_SADs land in one vector [TL TR BL BR], doubled
// (the dotprod kernel's mult = 2), the 8x8 bests are a vector
// compare-and-select (vcltq + vbslq), the eight 16x16 sums (vaddvq) go to
// sad16[n] and update the 16x16 best.
//
// archsimd has no UABD/UDOT, so a row's MAX-MIN absolute differences
// accumulate as wrapping uint16 byte pairs S and high bytes H, corrected to
// the pair sums as S - 255*H after the pairwise reduction (exact: an 8x8
// SUB_SAD before doubling is at most 32*255).
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	_ = src[63*stride+63]
	_ = ref[63*stride+63+7]
	s2 := 2 * stride
	hiShift := archsimd.BroadcastInt16x8(-8)
	c255 := archsimd.BroadcastUint16x8(255)
	// tmp_mv = (y << 16) | ((x + k) & 0xFFFF): the x half steps by one in
	// 16-bit lanes so it wraps like the mask.
	mv0 := archsimd.BroadcastUint32x4(uint32(uint16(int16(mvY)))<<16 | uint32(uint16(int16(mvX))))
	inc := archsimd.BroadcastUint32x4(1).ReshapeToUint16s()
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		sp := unsafe.Pointer(&src[y16*stride+x16])
		rp := unsafe.Pointer(&ref[y16*stride+x16])
		// The source block's even rows, loaded once for all eight positions.
		a0 := load16(sp)
		a1 := load16(unsafe.Add(sp, s2))
		a2 := load16(unsafe.Add(sp, 2*s2))
		a3 := load16(unsafe.Add(sp, 3*s2))
		a4 := load16(unsafe.Add(sp, 4*s2))
		a5 := load16(unsafe.Add(sp, 5*s2))
		a6 := load16(unsafe.Add(sp, 6*s2))
		a7 := load16(unsafe.Add(sp, 7*s2))
		pu8 := 21 + n*4
		best := (*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8]))
		bmv := (*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8]))
		bestV := archsimd.LoadUint32x4Array(best)
		bestMVV := archsimd.LoadUint32x4Array(bmv)
		mv := mv0
		var sads [8]archsimd.Uint32x4
		for k := range 8 {
			r := unsafe.Add(rp, k)
			d := absDiffU8x16(a0, load16(r)).ReshapeToUint16s()
			st, ht := d, d.Shift(hiShift)
			d = absDiffU8x16(a1, load16(unsafe.Add(r, s2))).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			d = absDiffU8x16(a2, load16(unsafe.Add(r, 2*s2))).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			d = absDiffU8x16(a3, load16(unsafe.Add(r, 3*s2))).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			d = absDiffU8x16(a4, load16(unsafe.Add(r, 4*s2))).ReshapeToUint16s()
			sb, hb := d, d.Shift(hiShift)
			d = absDiffU8x16(a5, load16(unsafe.Add(r, 5*s2))).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			d = absDiffU8x16(a6, load16(unsafe.Add(r, 6*s2))).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			d = absDiffU8x16(a7, load16(unsafe.Add(r, 7*s2))).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			// Lanes 0-3 of the top accumulators are TL byte pairs, 4-7 TR;
			// the bottom ones BL/BR. Two pairwise folds give [TL TR BL BR].
			sv := st.ConcatAddPairs(sb)
			hv := ht.ConcatAddPairs(hb)
			sv = sv.ConcatAddPairs(sv)
			hv = hv.ConcatAddPairs(hv)
			sad := sv.Sub(hv.Mul(c255)).ExtendLo4ToUint32()
			sad = sad.Add(sad)
			lt := sad.Less(bestV)
			bestV = sad.IfElse(lt, bestV)
			bestMVV = mv.IfElse(lt, bestMVV)
			sads[k] = sad
			mv = mv.ReshapeToUint16s().Add(inc).ReshapeToUint32s()
		}
		bestV.StoreArray(best)
		bestMVV.StoreArray(bmv)
		// vaddvq per position, as one pairwise fold: lane k is position k.
		row := &sad16[n]
		lo := sads[0].ConcatAddPairs(sads[1]).ConcatAddPairs(sads[2].ConcatAddPairs(sads[3]))
		hi := sads[4].ConcatAddPairs(sads[5]).ConcatAddPairs(sads[6].ConcatAddPairs(sads[7]))
		lo.StoreArray((*[4]uint32)(row[0:4]))
		hi.StoreArray((*[4]uint32)(row[4:8]))
		for k, s := range row {
			me.update(5+n, s, mvX+k, mvY)
		}
	}
}
