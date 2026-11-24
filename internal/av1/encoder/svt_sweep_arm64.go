//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// svtAllSAD8x8_16x16Sub is svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod
// with sub_sad (Source/Lib/ASM_NEON_DOTPROD/sad_neon_dotprod.c) over the
// superblock at src[0] / ref[0]: for each 16x16 block (offsets[] order) and
// position k = 0..7 the four 8x8 SUB_SADs [TL TR BL BR], doubled (the
// dotprod kernel's mult = 2), keep each 8x8's first strict minimum (best SAD,
// best MV) and give p_eight_sad16x16[n][k], whose first strict minimum
// updates the 16x16 best.
//
// archsimd has no UABD/UDOT, so a row's MAX-MIN absolute differences
// accumulate as wrapping uint16 byte pairs S and high bytes H. The
// reduction runs on two positions per vector: pairwise adds fold each
// position's S and H to [TL TR BL BR] and pair positions 2j, 2j+1 in one
// uint16x8, and one MLA by 0xFF01 (-255) corrects S - 255*H to the
// undoubled SUB_SADs t (exact: t <= 32*255 = 8160). The 8x8 bests are the
// minimum over the eight positions of the key 8*t + k (<= 65287), which
// orders by SAD then position, so it is the first strict minimum the
// sequential compare-and-select keeps; it replaces the stored best only when
// 2*t is below it. The 16x16 sums (<= 2*4*8160) fold from the t vectors in
// uint16 and widen once.
//
// The rows are grid rows: see the footprint check below.
func svtAllSAD8x8_16x16Sub(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int, sad16 *[16][8]uint32) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	// The superblock's even rows as grids, checked once: 32 rows of 64
	// source bytes and of 72 reference bytes (the eight positions' slack)
	// at stride s2. Each 16x16 block's eight rows are constant rows of a
	// sub-grid, read at their load sites from one base address, so neither
	// a row nor a position window costs a check.
	if !unsafeutil.GridFit(len(src), 0, s2, 64, 32) || !unsafeutil.GridFit(len(ref), 0, s2, 72, 32) {
		panic(errKernelArgs)
	}
	gs := unsafeutil.NewGrid(src, 0, s2, 64, 32)
	gr := unsafeutil.NewGrid(ref, 0, s2, 72, 32)
	hiShift := archsimd.BroadcastInt16x8(-8)
	m255 := archsimd.BroadcastUint16x8(0xFF01)
	eight := archsimd.BroadcastUint16x8(8)
	two := archsimd.BroadcastUint16x8(2)
	// Lanes 0-3 position 0, lanes 4-7 position 1.
	k01 := archsimd.LoadUint16x8Array(&[8]uint16{0, 0, 0, 0, 1, 1, 1, 1})
	keyShift := archsimd.BroadcastInt32x4(-2)
	keyMask := archsimd.BroadcastUint32x4(^uint32(1))
	posMask := archsimd.BroadcastUint32x4(7)
	mv0 := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY)).ReshapeToUint16s()
	eight32 := archsimd.BroadcastUint32x4(8)
	k03 := archsimd.LoadUint32x4Array(&[4]uint32{0, 1, 2, 3})
	k47 := archsimd.LoadUint32x4Array(&[4]uint32{4, 5, 6, 7})
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		bs := gs.Sub(y16/2, x16, 16, 8)
		br := gr.Sub(y16/2, x16, 24, 8)
		a0, a1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 0, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 1, 0))
		a2, a3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 2, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 3, 0))
		a4, a5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 4, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 5, 0))
		a6, a7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 6, 0)), archsimd.LoadUint8x16Array(unsafeutil.Row[[16]uint8](bs, 7, 0))
		var tv [4]archsimd.Uint16x8
		var keyMin archsimd.Uint16x8
		kv := k01
		for j := range 4 {
			c := 2 * j
			q := archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 0, 0)[c : c+16]))
			d := a0.Max(q).Sub(a0.Min(q)).ReshapeToUint16s()
			st0, ht0 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 1, 0)[c : c+16]))
			d = a1.Max(q).Sub(a1.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 2, 0)[c : c+16]))
			d = a2.Max(q).Sub(a2.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 3, 0)[c : c+16]))
			d = a3.Max(q).Sub(a3.Min(q)).ReshapeToUint16s()
			st0, ht0 = st0.Add(d), ht0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 4, 0)[c : c+16]))
			d = a4.Max(q).Sub(a4.Min(q)).ReshapeToUint16s()
			sb0, hb0 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 5, 0)[c : c+16]))
			d = a5.Max(q).Sub(a5.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 6, 0)[c : c+16]))
			d = a6.Max(q).Sub(a6.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 7, 0)[c : c+16]))
			d = a7.Max(q).Sub(a7.Min(q)).ReshapeToUint16s()
			sb0, hb0 = sb0.Add(d), hb0.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 0, 0)[c+1 : c+17]))
			d = a0.Max(q).Sub(a0.Min(q)).ReshapeToUint16s()
			st1, ht1 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 1, 0)[c+1 : c+17]))
			d = a1.Max(q).Sub(a1.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 2, 0)[c+1 : c+17]))
			d = a2.Max(q).Sub(a2.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 3, 0)[c+1 : c+17]))
			d = a3.Max(q).Sub(a3.Min(q)).ReshapeToUint16s()
			st1, ht1 = st1.Add(d), ht1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 4, 0)[c+1 : c+17]))
			d = a4.Max(q).Sub(a4.Min(q)).ReshapeToUint16s()
			sb1, hb1 := d, d.Shift(hiShift)
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 5, 0)[c+1 : c+17]))
			d = a5.Max(q).Sub(a5.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 6, 0)[c+1 : c+17]))
			d = a6.Max(q).Sub(a6.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(unsafeutil.Row[[24]uint8](br, 7, 0)[c+1 : c+17]))
			d = a7.Max(q).Sub(a7.Min(q)).ReshapeToUint16s()
			sb1, hb1 = sb1.Add(d), hb1.Add(d.Shift(hiShift))
			sv := st0.ConcatAddPairs(sb0).ConcatAddPairs(st1.ConcatAddPairs(sb1))
			hv := ht0.ConcatAddPairs(hb0).ConcatAddPairs(ht1.ConcatAddPairs(hb1))
			t := hv.MulAdd(m255, sv)
			tv[j] = t
			key := t.MulAdd(eight, kv)
			if j == 0 {
				keyMin = key
			} else {
				keyMin = keyMin.Min(key)
			}
			kv = kv.Add(two)
		}
		// The 8x8 bests: lanes 0-3 hold the even positions' minimum keys,
		// lanes 4-7 the odd ones'.
		km := keyMin.Min(keyMin.HiToLo()).ExtendLo4ToUint32()
		sad := km.Shift(keyShift).And(keyMask)
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv := mv0.Add(km.And(posMask).ReshapeToUint16s()).ReshapeToUint32s()
		mv.IfElse(lt, archsimd.LoadUint32x4Array(mp)).StoreArray(mp)
		// The 16x16 sums: [p0 .. p7], undoubled, then doubled in uint16.
		u := tv[0].ConcatAddPairs(tv[1]).ConcatAddPairs(tv[2].ConcatAddPairs(tv[3]))
		u = u.Add(u)
		lo, hi := u.ExtendLo4ToUint32(), u.HiToLo().ExtendLo4ToUint32()
		out := &sad16[n]
		lo.StoreArray((*[4]uint32)(out[0:4]))
		hi.StoreArray((*[4]uint32)(out[4:8]))
		// update_best for the 16x16 PU, its first strict minimum: the
		// minimum key 8*sad + k (<= 8*65280 + 7), so no lane is read back
		// from the stores.
		key := lo.MulAdd(eight32, k03).Min(hi.MulAdd(eight32, k47)).ReduceMin()
		if m := key >> 3; m < me.bestSAD[5+n] {
			me.bestSAD[5+n] = m
			me.bestMV[5+n] = svtPackMV(mvX+int(key&7), mvY)
		}
	}
}

// svtSweepPoint1 is svtAllSAD8x8_16x16Sub for one search position, the
// svt_ext_sad_calculation_8x8_16x16 (sub_sad) remainder of
// open_loop_me_get_search_point_results_block: the four 8x8 SUB_SADs of each
// 16x16 block in one vector with a compare-and-select best update, then the
// 16x16, 32x32 and 64x64 sums. ref needs no slack past the 64x64 block.
// Rows are bounds-checked windows, one per row pair and array.
func svtSweepPoint1(me *realtimeSBMultiSizeME, src, ref []byte, stride, mvX, mvY int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	s2 := 2 * stride
	n2 := s2 + 16
	hiShift := archsimd.BroadcastInt16x8(-8)
	loShift := archsimd.BroadcastInt16x8(8)
	mv := archsimd.BroadcastUint32x4(svtPackMV(mvX, mvY))
	var sad16 [16]uint32
	for n := range 16 {
		x16, y16 := sweepBlock16(n)
		off := y16*stride + x16
		var st, ht, sb, hb archsimd.Uint16x8
		for i := range 2 {
			o := off + 2*i*s2
			aw, rw := src[o:o+n2:o+n2], ref[o:o+n2:o+n2]
			p := archsimd.LoadUint8x16Array((*[16]uint8)(aw))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
			p = archsimd.LoadUint8x16Array((*[16]uint8)(aw[s2:]))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(rw[s2:]))
			d = p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			st, ht = st.Add(d), ht.Add(d.Shift(hiShift))
		}
		for i := 2; i < 4; i++ {
			o := off + 2*i*s2
			aw, rw := src[o:o+n2:o+n2], ref[o:o+n2:o+n2]
			p := archsimd.LoadUint8x16Array((*[16]uint8)(aw))
			q := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
			d := p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
			p = archsimd.LoadUint8x16Array((*[16]uint8)(aw[s2:]))
			q = archsimd.LoadUint8x16Array((*[16]uint8)(rw[s2:]))
			d = p.Max(q).Sub(p.Min(q)).ReshapeToUint16s()
			sb, hb = sb.Add(d), hb.Add(d.Shift(hiShift))
		}
		sv := st.ConcatAddPairs(sb)
		hv := ht.ConcatAddPairs(hb)
		sv = sv.ConcatAddPairs(sv)
		hv = hv.ConcatAddPairs(hv)
		sad := sv.Sub(hv.Shift(loShift)).Add(hv).ExtendLo4ToUint32()
		sad = sad.Add(sad)
		pu8 := 21 + n*4
		bp := (*[4]uint32)(me.bestSAD[pu8 : pu8+4])
		mp := (*[4]uint32)(me.bestMV[pu8 : pu8+4])
		best := archsimd.LoadUint32x4Array(bp)
		bestMV := archsimd.LoadUint32x4Array(mp)
		lt := sad.Less(best)
		sad.IfElse(lt, best).StoreArray(bp)
		mv.IfElse(lt, bestMV).StoreArray(mp)
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
