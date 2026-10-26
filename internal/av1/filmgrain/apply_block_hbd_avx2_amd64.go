// Ported from dav1d: src/x86/filmgrain16_avx2.asm (fgy_32x32xn_16bpc,
// fguv_32x32xn_i420/i422/i444_16bpc)
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

//go:build goexperiment.simd && amd64 && !purego

package filmgrain

import "simd/archsimd"

// fgConstsHBD holds the per-row-band constants of the 10/12-bit AVX2 block
// kernels.
type fgConstsHBD struct {
	scalingShift uint64
	// shiftV is the vpsravd count of the noise round (a variable vpsrad
	// count is materialized with a legacy-SSE movq, which stalls on Zen 3).
	shiftV [8]uint32
	// ix and sc are the gather spills, kept here so the row loops do not
	// re-zero local arrays with legacy-SSE stores.
	ix               [16]int16
	sc               [2]uint64
	grainMin         int16
	grainMax         int16
	oxOld, oxCur     [16]int16
	oyTop, oyCur     [2]int16
	minV, maxV       int16
	uvLumaMult       int16
	uvMult           int16
	uvOffset, idxMax int32
}

func (k *fgConstsHBD) init(bitDepth uint8, scalingShift, minValue, maxValue, sx, sy int) {
	k.scalingShift = uint64(scalingShift)
	for i := range k.shiftV {
		k.shiftV[i] = uint32(scalingShift)
	}
	k.grainMin, k.grainMax = int16(-(1 << (bitDepth - 1))), int16((1<<(bitDepth-1))-1)
	for i := range k.oxCur {
		k.oxOld[i], k.oxCur[i] = 0, 32
	}
	if sx != 0 {
		k.oxOld[0], k.oxCur[0] = 23, 22
	} else {
		k.oxOld[0], k.oxCur[0] = 27, 17
		k.oxOld[1], k.oxCur[1] = 17, 27
	}
	if sy != 0 {
		k.oyTop[0], k.oyCur[0] = 23, 22
	} else {
		k.oyTop = [2]int16{27, 17}
		k.oyCur = [2]int16{17, 27}
	}
	k.minV, k.maxV = int16(minValue), int16(maxValue)
	k.idxMax = int32(1)<<bitDepth - 1
}

func (k *fgConstsHBD) initChroma(params *ChromaRowParams) {
	k.uvLumaMult = int16(int(params.ChromaLumaMult) - 128)
	k.uvMult = int16(int(params.ChromaMult) - 128)
	k.uvOffset = int32((int(params.ChromaOffset) << (params.BitDepth - 8)) - (1 << params.BitDepth))
}

// fgMAdd16 is a*wa + b*wb per lane in int32 (vpmaddwd on the interleaved
// pairs), returned as the in-lane low and high halves.
func fgMAdd16(a, b, wa, wb archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	return a.InterleaveLoGrouped(b).DotProductPairs(wa.InterleaveLoGrouped(wb)),
		a.InterleaveHiGrouped(b).DotProductPairs(wa.InterleaveHiGrouped(wb))
}

// fgBlendHBD16 is round2(old*wOld + cur*wCur, 5) clipped to the grain range,
// in int32 (high-bit-depth grain times the weights overflows int16).
func fgBlendHBD16(old, cur, wOld, wCur archsimd.Int16x16, k *fgConstsHBD) archsimd.Int16x16 {
	lo, hi := fgMAdd16(old, cur, wOld, wCur)
	r := archsimd.BroadcastInt32x8(16)
	v := lo.Add(r).ShiftAllRight(5).SaturateToInt16ConcatGrouped(hi.Add(r).ShiftAllRight(5))
	return v.Max(archsimd.BroadcastInt16x16(k.grainMin)).Min(archsimd.BroadcastInt16x16(k.grainMax))
}

// fgNoiseHBD16 is clip(src + round2(scaling*grain, scaling_shift)) for
// sixteen samples, the product formed in int32.
func fgNoiseHBD16(scale, grain, src archsimd.Int16x16, k *fgConstsHBD) archsimd.Int16x16 {
	var zero archsimd.Int16x16
	lo, hi := fgMAdd16(scale, zero, grain, zero)
	r := archsimd.BroadcastInt32x8(1 << (k.scalingShift - 1))
	sh := archsimd.LoadUint32x8Array(&k.shiftV)
	n := lo.Add(r).ShiftRight(sh).SaturateToInt16ConcatGrouped(hi.Add(r).ShiftRight(sh))
	return n.Add(src).Max(archsimd.BroadcastInt16x16(k.minV)).Min(archsimd.BroadcastInt16x16(k.maxV))
}

// fgLeftHBD loads the eight left-block grain samples at off into the low
// lanes.
func fgLeftHBD(grain []int16, off int) archsimd.Int16x16 {
	return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8(grain[off : off+8]))
}

// fgGrainRowHBD loads the sixteen-sample grain run at cur with the left and
// top overlap blends.
func fgGrainRowHBD(grain []int16, cur, old, top, topOld int, ox, oy bool, k *fgConstsHBD, wt, wc archsimd.Int16x16) archsimd.Int16x16 {
	g := archsimd.LoadInt16x16(grain[cur : cur+16])
	oxOld := archsimd.LoadInt16x16Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x16Array(&k.oxCur)
	if ox {
		g = fgBlendHBD16(fgLeftHBD(grain, old), g, oxOld, oxCur, k)
	}
	if oy {
		t := archsimd.LoadInt16x16(grain[top : top+16])
		if ox {
			t = fgBlendHBD16(fgLeftHBD(grain, topOld), t, oxOld, oxCur, k)
		}
		g = fgBlendHBD16(t, g, wt, wc, k)
	}
	return g
}

// fgGatherHBD16 looks up the scaling bytes of sixteen sample-value indices
// (dav1d gathers from a per-value table; archsimd has no gather) and
// widens them to int16.
func fgGatherHBD16(table *[ScalingTableSize]uint8, idx archsimd.Int16x16, k *fgConstsHBD) archsimd.Int16x16 {
	ix, sc := &k.ix, &k.sc
	idx.StoreArray(ix)
	for j := range sc {
		var v uint64
		for i := 7; i >= 0; i-- {
			v = v<<8 | uint64(table[uint16(ix[8*j+i])&(ScalingTableSize-1)])
		}
		sc[j] = v
	}
	return archsimd.LoadUint64x2Array(sc).AsUint8x16().ExtendToUint16().AsInt16x16()
}

// fgGatherSrcHBD16 is fgGatherHBD16 for indices read straight from sixteen
// samples in memory.
func fgGatherSrcHBD16(table *[ScalingTableSize]uint8, s []uint16, k *fgConstsHBD) archsimd.Int16x16 {
	s = s[:16]
	sc := &k.sc
	for j := range sc {
		q := s[8*j : 8*j+8]
		sc[j] = uint64(table[q[0]&(ScalingTableSize-1)]) | uint64(table[q[1]&(ScalingTableSize-1)])<<8 |
			uint64(table[q[2]&(ScalingTableSize-1)])<<16 | uint64(table[q[3]&(ScalingTableSize-1)])<<24 |
			uint64(table[q[4]&(ScalingTableSize-1)])<<32 | uint64(table[q[5]&(ScalingTableSize-1)])<<40 |
			uint64(table[q[6]&(ScalingTableSize-1)])<<48 | uint64(table[q[7]&(ScalingTableSize-1)])<<56
	}
	return archsimd.LoadUint64x2Array(sc).AsUint8x16().ExtendToUint16().AsInt16x16()
}

// fguvIndexHBD16 is the !csfl scaling index for sixteen chroma samples:
// clip(((avg*uv_luma_mult + src*uv_mult) >> 6) + uv_offset, 0, 2^bd-1), in
// int32.
func fguvIndexHBD16(avg, src archsimd.Int16x16, k *fgConstsHBD) archsimd.Int16x16 {
	lo, hi := fgMAdd16(avg, src, archsimd.BroadcastInt16x16(k.uvLumaMult), archsimd.BroadcastInt16x16(k.uvMult))
	off := archsimd.BroadcastInt32x8(k.uvOffset)
	var zero archsimd.Int32x8
	mx := archsimd.BroadcastInt32x8(k.idxMax)
	lo = lo.ShiftAllRight(6).Add(off).Max(zero).Min(mx)
	hi = hi.ShiftAllRight(6).Add(off).Max(zero).Min(mx)
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// fgLoadU16 loads sixteen 16-bit samples as int16 lanes (<= 4095).
func fgLoadU16(s []uint16) archsimd.Int16x16 {
	return archsimd.LoadUint16x16(s[:16]).AsInt16x16()
}

// fgStoreU16 stores sixteen clipped samples.
func fgStoreU16(d []uint16, v archsimd.Int16x16) {
	v.AsUint16x16().Store(d[:16])
}

// fgyBlockHBDAVX2 is fgy_32x32xn_16bpc for one full-width 32xbh luma block.
func fgyBlockHBDAVX2(dst, src []uint16, stride int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox bool) {
	for y := 0; y < bh; y++ {
		off := y * stride
		gy := y * LumaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g0 := fgGrainRowHBD(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		g1 := fgGrainRowHBD(grain, cur+gy+16, 0, top+gy+16, 0, false, oy, k, wt, wc)
		x0, x1 := fgLoadU16(src[off:]), fgLoadU16(src[off+16:])
		fgStoreU16(dst[off:], fgNoiseHBD16(fgGatherSrcHBD16(table, src[off:], k), g0, x0, k))
		fgStoreU16(dst[off+16:], fgNoiseHBD16(fgGatherSrcHBD16(table, src[off+16:], k), g1, x1, k))
	}
	archsimd.ClearAVXUpperBits()
}

// fgPairOrder is the vpermd order that undoes vpackssdw's in-lane qword
// order (0, 2, 1, 3).
var fgPairOrder = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// fguvBlockHBDSX1AVX2 is fguv_32x32xn_i420/i422_16bpc for one full-width
// 16xbh chroma block. lumaStep is the luma stride (in samples) times
// 1<<ss_y.
func fguvBlockHBDSX1AVX2(dst, src []uint16, stride int, luma []uint16, lumaStep int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	ones := archsimd.BroadcastInt16x16(1)
	one32 := archsimd.BroadcastInt32x8(1)
	perm := archsimd.LoadUint32x8Array(&fgPairOrder)
	for y := 0; y < bh; y++ {
		off := y * stride
		gy := y * ChromaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g := fgGrainRowHBD(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		x := fgLoadU16(src[off:])
		li := y * lumaStep
		// (l[2i] + l[2i+1] + 1) >> 1: vpmaddwd against ones sums the pairs,
		// and vpermd undoes the pack's in-lane order.
		s0 := fgLoadU16(luma[li:]).DotProductPairs(ones).Add(one32).ShiftAllRight(1)
		s1 := fgLoadU16(luma[li+16:]).DotProductPairs(ones).Add(one32).ShiftAllRight(1)
		avg := s0.SaturateToInt16ConcatGrouped(s1).AsInt32x8().Permute(perm).AsInt16x16()
		if !csfl {
			avg = fguvIndexHBD16(avg, x, k)
		}
		fgStoreU16(dst[off:], fgNoiseHBD16(fgGatherHBD16(table, avg, k), g, x, k))
	}
	archsimd.ClearAVXUpperBits()
}

// fguvBlockHBDSX0AVX2 is fguv_32x32xn_i444_16bpc for one full-width 32xbh
// chroma block.
func fguvBlockHBDSX0AVX2(dst, src []uint16, stride int, luma []uint16, lumaStep int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	for y := 0; y < bh; y++ {
		off := y * stride
		gy := y * ChromaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g0 := fgGrainRowHBD(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		g1 := fgGrainRowHBD(grain, cur+gy+16, 0, top+gy+16, 0, false, oy, k, wt, wc)
		li := y * lumaStep
		x0, x1 := fgLoadU16(src[off:]), fgLoadU16(src[off+16:])
		a0, a1 := fgLoadU16(luma[li:]), fgLoadU16(luma[li+16:])
		var s0, s1 archsimd.Int16x16
		if csfl {
			s0, s1 = fgGatherSrcHBD16(table, luma[li:], k), fgGatherSrcHBD16(table, luma[li+16:], k)
		} else {
			s0, s1 = fgGatherHBD16(table, fguvIndexHBD16(a0, x0, k), k), fgGatherHBD16(table, fguvIndexHBD16(a1, x1, k), k)
		}
		fgStoreU16(dst[off:], fgNoiseHBD16(s0, g0, x0, k))
		fgStoreU16(dst[off+16:], fgNoiseHBD16(s1, g1, x1, k))
	}
	archsimd.ClearAVXUpperBits()
}

// fgLumaBlockHBD places grain on one 10/12-bit luma block: full-width
// blocks take the AVX2 kernel, a partial right-edge block (or a machine
// without AVX2) the scalar path.
func fgLumaBlockHBD(dst, src []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	if !fgAVX2 || blockWidth != LumaBlockSize {
		applyLumaBlock(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := lumaGrainSampleIndex(offsets[0][0], 0, 0, 0, 0)
	old := lumaGrainSampleIndex(offsets[1][0], 1, 0, 0, 0)
	top := lumaGrainSampleIndex(offsets[0][1], 0, 1, 0, 0)
	topOld := lumaGrainSampleIndex(offsets[1][1], 1, 1, 0, 0)
	fgyBlockHBDAVX2(dst[blockX:], src[blockX:], params.Stride, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0)
}

// fgChromaBlockHBD is fgLumaBlockHBD for a chroma block.
func fgChromaBlockHBD(dst, src, luma []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	if !fgAVX2 || blockWidth != LumaBlockSize>>shiftX {
		applyChromaBlock(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := chromaGrainSampleIndex(offsets[0][0], shiftX, shiftY, 0, 0, 0, 0)
	old := chromaGrainSampleIndex(offsets[1][0], shiftX, shiftY, 1, 0, 0, 0)
	top := chromaGrainSampleIndex(offsets[0][1], shiftX, shiftY, 0, 1, 0, 0)
	topOld := chromaGrainSampleIndex(offsets[1][1], shiftX, shiftY, 1, 1, 0, 0)
	lumaRow := luma[blockX<<shiftX:]
	if shiftX != 0 {
		fguvBlockHBDSX1AVX2(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
		return
	}
	fguvBlockHBDSX0AVX2(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
}
