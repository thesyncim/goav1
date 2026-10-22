//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// sweepSearchPoint8 is the open-loop ME sweep's eight-position batch in the
// shape of SVT-AV1's Source/Lib/ASM_NEON/sad_neon.c
// svt_ext_all_sad_calculation_8x8_16x16_neon (SUB_SAD_SEARCH path) and
// svt_ext_eight_sad_calculation_32x32_64x64: per 16x16 block the four 8x8
// sub-sampled SADs of each of the eight positions (mvX..mvX+7, mvY) land in
// one vector, the 8x8 argmins are a vector compare-and-select against the
// running best (strict '<', earliest position wins ties, as the scalar
// scan), and the 16x16 sums feed the 32x32/64x64 argmins. It reports false
// (nothing done) when the batch's loads would leave the slices.
func (me *realtimeSBMultiSizeME) sweepSearchPoint8(src, ref []byte, stride, srcBase, refBase, mvX, mvY int) bool {
	if srcBase < 0 || refBase < 0 || srcBase+63*stride+64 > len(src) || refBase+63*stride+64+8 > len(ref) {
		return false
	}
	sp := unsafe.Pointer(&src[srcBase])
	rp := unsafe.Pointer(&ref[refBase])
	s2 := 2 * stride
	var sad16 [16][8]uint32
	var mvs [8]archsimd.Uint32x4
	for k := range 8 {
		mvs[k] = archsimd.BroadcastUint32x4(uint32(uint16(int16(mvY)))<<16 | uint32(uint16(int16(mvX+k))))
	}
	for n := range 16 {
		q, w := n>>2, n&3
		x16 := (q&1)*32 + (w&1)*16
		y16 := (q>>1)*32 + (w>>1)*16
		sb := unsafe.Add(sp, y16*stride+x16)
		// The source block's even rows (SUB_SAD), loaded once for all
		// eight positions.
		a0 := load16(sb)
		a1 := load16(unsafe.Add(sb, s2))
		a2 := load16(unsafe.Add(sb, 2*s2))
		a3 := load16(unsafe.Add(sb, 3*s2))
		a4 := load16(unsafe.Add(sb, 4*s2))
		a5 := load16(unsafe.Add(sb, 5*s2))
		a6 := load16(unsafe.Add(sb, 6*s2))
		a7 := load16(unsafe.Add(sb, 7*s2))
		pu8 := 21 + n*4
		best := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		bestMV := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		rb := unsafe.Add(rp, y16*stride+x16)
		for k := range 8 {
			r := unsafe.Add(rb, k)
			d := absDiffU8x16(a0, load16(r))
			tl, tr := d.ExtendLo8ToUint16(), widenHi8(d)
			d = absDiffU8x16(a1, load16(unsafe.Add(r, s2)))
			tl, tr = tl.Add(d.ExtendLo8ToUint16()), tr.Add(widenHi8(d))
			d = absDiffU8x16(a2, load16(unsafe.Add(r, 2*s2)))
			tl, tr = tl.Add(d.ExtendLo8ToUint16()), tr.Add(widenHi8(d))
			d = absDiffU8x16(a3, load16(unsafe.Add(r, 3*s2)))
			tl, tr = tl.Add(d.ExtendLo8ToUint16()), tr.Add(widenHi8(d))
			d = absDiffU8x16(a4, load16(unsafe.Add(r, 4*s2)))
			bl, br := d.ExtendLo8ToUint16(), widenHi8(d)
			d = absDiffU8x16(a5, load16(unsafe.Add(r, 5*s2)))
			bl, br = bl.Add(d.ExtendLo8ToUint16()), br.Add(widenHi8(d))
			d = absDiffU8x16(a6, load16(unsafe.Add(r, 6*s2)))
			bl, br = bl.Add(d.ExtendLo8ToUint16()), br.Add(widenHi8(d))
			d = absDiffU8x16(a7, load16(unsafe.Add(r, 7*s2)))
			bl, br = bl.Add(d.ExtendLo8ToUint16()), br.Add(widenHi8(d))
			u := tl.ConcatAddPairs(tr).ConcatAddPairs(bl.ConcatAddPairs(br))
			sad := u.ConcatAddPairs(u).ExtendLo4ToUint32().ShiftAllLeft(1) // [TL TR BL BR], doubled
			lt := sad.Less(best)
			best = sad.IfElse(lt, best)
			bestMV = mvs[k].IfElse(lt, bestMV)
			sad16[n][k] = sad.ReduceSum()
		}
		best.StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestSAD[pu8])))
		bestMV.StoreArray((*[4]uint32)(unsafe.Pointer(&me.bestMV[pu8])))
		for k := range 8 {
			me.update(5+n, sad16[n][k], mvX+k, mvY)
		}
	}
	for k := range 8 {
		var sad64 uint32
		for q := range 4 {
			s32 := sad16[q*4][k] + sad16[q*4+1][k] + sad16[q*4+2][k] + sad16[q*4+3][k]
			me.update(1+q, s32, mvX+k, mvY)
			sad64 += s32
		}
		me.update(0, sad64, mvX+k, mvY)
	}
	return true
}
