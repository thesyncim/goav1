// Ported from dav1d: src/arm/64/filmgrain16.S (fgy_32x32_16bpc_neon,
// fguv_32x32_*_16bpc_neon)
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import "simd/archsimd"

// fgConstsHBD holds the per-row-band vector constants of the 10/12-bit
// block kernels, loaded once per block (see fgConstsU8).
type fgConstsHBD struct {
	noiseRound [4]int32
	noiseShift [4]int32
	blendRound [4]int32
	blendShift [4]int32
	grainMin   [8]int16
	grainMax   [8]int16
	oxOld      [8]int16
	oxCur      [8]int16
	oyTop      [2][8]int16
	oyCur      [2][8]int16
	uvLumaMult [8]int16
	uvMult     [8]int16
	uvOffset   [4]int32
	uvShift    [4]int32
	idxMax     [4]int32
	minV       [8]int16
	maxV       [8]int16
}

func fill4(dst *[4]int32, v int) {
	for i := range dst {
		dst[i] = int32(v)
	}
}

func (k *fgConstsHBD) init(bitDepth uint8, scalingShift, minValue, maxValue, sx, sy int) {
	fill4(&k.noiseRound, 1<<(scalingShift-1))
	fill4(&k.noiseShift, -scalingShift)
	fill4(&k.blendRound, 16)
	fill4(&k.blendShift, -5)
	fill8(&k.grainMin, -(1 << (bitDepth - 1)))
	fill8(&k.grainMax, (1<<(bitDepth-1))-1)
	if sx != 0 {
		k.oxOld = [8]int16{23}
		k.oxCur = [8]int16{22, 32, 32, 32, 32, 32, 32, 32}
	} else {
		k.oxOld = [8]int16{27, 17}
		k.oxCur = [8]int16{17, 27, 32, 32, 32, 32, 32, 32}
	}
	if sy != 0 {
		fill8(&k.oyTop[0], 23)
		fill8(&k.oyCur[0], 22)
	} else {
		fill8(&k.oyTop[0], 27)
		fill8(&k.oyCur[0], 17)
		fill8(&k.oyTop[1], 17)
		fill8(&k.oyCur[1], 27)
	}
	fill8(&k.minV, minValue)
	fill8(&k.maxV, maxValue)
	fill4(&k.idxMax, (1<<bitDepth)-1)
}

func (k *fgConstsHBD) initChroma(params *ChromaRowParams) {
	fill8(&k.uvLumaMult, int(params.ChromaLumaMult)-128)
	fill8(&k.uvMult, int(params.ChromaMult)-128)
	fill4(&k.uvOffset, (int(params.ChromaOffset)<<(params.BitDepth-8))-(1<<params.BitDepth))
	fill4(&k.uvShift, -6)
}

// fgNarrow32 packs two int32x4 halves into int16x8 (sqxtn + sqxtn2).
func fgNarrow32(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	return lo.SaturateToInt16().ToBits().ReshapeToUint64s().InterleaveLo(hi.SaturateToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

// fgBlendHBD is round2(old*wOld + cur*wCur, 5) clipped to the grain range,
// in int32 (high-bit-depth grain times the weights overflows int16).
func fgBlendHBD(old, cur, wOld, wCur archsimd.Int16x8, round, shift archsimd.Int32x4, lo, hi archsimd.Int16x8) archsimd.Int16x8 {
	l := old.MulWidenLo(wOld).Add(cur.MulWidenLo(wCur)).Add(round).Shift(shift)
	h := old.HiToLo().MulWidenLo(wOld.HiToLo()).Add(cur.HiToLo().MulWidenLo(wCur.HiToLo())).Add(round).Shift(shift)
	return fgNarrow32(l, h).Max(lo).Min(hi)
}

// fgNoiseHBD is clip(src + round2(scaling*grain, scaling_shift)) for eight
// samples; the product is formed in int32 (smull/smull2).
func fgNoiseHBD(scale archsimd.Uint8x16, grain archsimd.Int16x8, src archsimd.Uint16x8, round, shift archsimd.Int32x4, minV, maxV archsimd.Int16x8) archsimd.Uint16x8 {
	s := scale.ExtendLo8ToUint16().BitsToInt16()
	l := s.MulWidenLo(grain).Add(round).Shift(shift)
	h := s.HiToLo().MulWidenLo(grain.HiToLo()).Add(round).Shift(shift)
	return fgNarrow32(l, h).Add(src.BitsToInt16()).Max(minV).Min(maxV).ToBits()
}

// The scaling gather of the high-bit-depth kernels (dav1d's gather with a
// per-sample-value table): eight samples from two 64-bit words of four
// 16-bit indices, into one word of eight scaling bytes. It is written inline
// in each row loop; a call would clobber every vector register.

// fgyBlockHBD is fgy_32x32_16bpc_neon for one full-width 32xbh luma block.
// table maps every sample value to its scaling byte.
func fgyBlockHBD(dst, src []uint16, stride int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox bool) {
	nRound := archsimd.LoadInt32x4Array(&k.noiseRound)
	nShift := archsimd.LoadInt32x4Array(&k.noiseShift)
	bRound := archsimd.LoadInt32x4Array(&k.blendRound)
	bShift := archsimd.LoadInt32x4Array(&k.blendShift)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	minV := archsimd.LoadInt16x8Array(&k.minV)
	maxV := archsimd.LoadInt16x8Array(&k.maxV)
	t := table
	var sc [4]uint64
	var ix [8]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[32]uint16)(src[off : off+32])
		d := (*[32]uint16)(dst[off : off+32])
		gy := y * LumaGrainWidth
		g := (*[32]int16)(grain[cur+gy : cur+gy+32])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		g2 := archsimd.LoadInt16x8Array((*[8]int16)(g[16:24]))
		g3 := archsimd.LoadInt16x8Array((*[8]int16)(g[24:32]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlendHBD(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			tr := (*[32]int16)(grain[top+gy : top+gy+32])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(tr[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlendHBD(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlendHBD(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
			g2 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[16:24])), g2, wt, wc, bRound, bShift, gMin, gMax)
			g3 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[24:32])), g3, wt, wc, bRound, bShift, gMin, gMax)
		}
		x0 := archsimd.LoadUint16x8Array((*[8]uint16)(s[0:8]))
		x1 := archsimd.LoadUint16x8Array((*[8]uint16)(s[8:16]))
		x2 := archsimd.LoadUint16x8Array((*[8]uint16)(s[16:24]))
		x3 := archsimd.LoadUint16x8Array((*[8]uint16)(s[24:32]))
		w0, w1, w2, w3 := x0.ReshapeToUint64s(), x1.ReshapeToUint64s(), x2.ReshapeToUint64s(), x3.ReshapeToUint64s()
		ix = [8]uint64{w0.GetElem(0), w0.GetElem(1), w1.GetElem(0), w1.GetElem(1), w2.GetElem(0), w2.GetElem(1), w3.GetElem(0), w3.GetElem(1)}
		for j := range sc {
			a, b := ix[2*j], ix[2*j+1]
			sc[j] = uint64(t[a&0xfff]) | uint64(t[(a>>16)&0xfff])<<8 | uint64(t[(a>>32)&0xfff])<<16 | uint64(t[(a>>48)&0xfff])<<24 |
				uint64(t[b&0xfff])<<32 | uint64(t[(b>>16)&0xfff])<<40 | uint64(t[(b>>32)&0xfff])<<48 | uint64(t[(b>>48)&0xfff])<<56
		}
		s0 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[0:2])).ReshapeToUint8s()
		s1 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[2:4])).ReshapeToUint8s()
		fgNoiseHBD(s0, g0, x0, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[0:8]))
		fgNoiseHBD(s0.HiToLo(), g1, x1, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[8:16]))
		fgNoiseHBD(s1, g2, x2, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[16:24]))
		fgNoiseHBD(s1.HiToLo(), g3, x3, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[24:32]))
	}
}

// fguvIndexHBD is the !csfl scaling index for eight chroma samples:
// clip(((avg*uv_luma_mult + src*uv_mult) >> 6) + uv_offset, 0, 2^bd-1), in
// int32.
func fguvIndexHBD(avg, src archsimd.Uint16x8, lm, m archsimd.Int16x8, off, sh, idxMax archsimd.Int32x4) archsimd.Uint16x8 {
	a, x := avg.BitsToInt16(), src.BitsToInt16()
	var zero archsimd.Int32x4
	l := a.MulWidenLo(lm).Add(x.MulWidenLo(m)).Shift(sh).Add(off).Max(zero).Min(idxMax)
	h := a.HiToLo().MulWidenLo(lm.HiToLo()).Add(x.HiToLo().MulWidenLo(m.HiToLo())).Shift(sh).Add(off).Max(zero).Min(idxMax)
	return fgNarrow32(l, h).ToBits()
}

// fguvBlockHBDSX1 is fguv_loop_sx1 (4:2:0 / 4:2:2) of the 16bpc kernels for
// one full-width 16xbh chroma block. lumaStep is the luma stride (in
// samples) times 1<<ss_y.
func fguvBlockHBDSX1(dst, src []uint16, stride int, luma []uint16, lumaStep int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	nRound := archsimd.LoadInt32x4Array(&k.noiseRound)
	nShift := archsimd.LoadInt32x4Array(&k.noiseShift)
	bRound := archsimd.LoadInt32x4Array(&k.blendRound)
	bShift := archsimd.LoadInt32x4Array(&k.blendShift)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	lm := archsimd.LoadInt16x8Array(&k.uvLumaMult)
	m := archsimd.LoadInt16x8Array(&k.uvMult)
	uvOff := archsimd.LoadInt32x4Array(&k.uvOffset)
	uvSh := archsimd.LoadInt32x4Array(&k.uvShift)
	idxMax := archsimd.LoadInt32x4Array(&k.idxMax)
	minV := archsimd.LoadInt16x8Array(&k.minV)
	maxV := archsimd.LoadInt16x8Array(&k.maxV)
	t := table
	var sc [2]uint64
	var ix [4]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[16]uint16)(src[off : off+16])
		d := (*[16]uint16)(dst[off : off+16])
		li := y * lumaStep
		l := (*[32]uint16)(luma[li : li+32])
		gy := y * ChromaGrainWidth
		g := (*[16]int16)(grain[cur+gy : cur+gy+16])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlendHBD(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			tr := (*[16]int16)(grain[top+gy : top+gy+16])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(tr[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlendHBD(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlendHBD(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
		}
		x0 := archsimd.LoadUint16x8Array((*[8]uint16)(s[0:8]))
		x1 := archsimd.LoadUint16x8Array((*[8]uint16)(s[8:16]))
		l0 := archsimd.LoadUint16x8Array((*[8]uint16)(l[0:8]))
		l1 := archsimd.LoadUint16x8Array((*[8]uint16)(l[8:16]))
		l2 := archsimd.LoadUint16x8Array((*[8]uint16)(l[16:24]))
		l3 := archsimd.LoadUint16x8Array((*[8]uint16)(l[24:32]))
		// (l[2i] + l[2i+1] + 1) >> 1: uzp1/uzp2 then urhadd.
		a0 := l0.ConcatEven(l1).Average(l0.ConcatOdd(l1))
		a1 := l2.ConcatEven(l3).Average(l2.ConcatOdd(l3))
		if !csfl {
			a0 = fguvIndexHBD(a0, x0, lm, m, uvOff, uvSh, idxMax)
			a1 = fguvIndexHBD(a1, x1, lm, m, uvOff, uvSh, idxMax)
		}
		w0, w1 := a0.ReshapeToUint64s(), a1.ReshapeToUint64s()
		ix = [4]uint64{w0.GetElem(0), w0.GetElem(1), w1.GetElem(0), w1.GetElem(1)}
		for j := range sc {
			a, b := ix[2*j], ix[2*j+1]
			sc[j] = uint64(t[a&0xfff]) | uint64(t[(a>>16)&0xfff])<<8 | uint64(t[(a>>32)&0xfff])<<16 | uint64(t[(a>>48)&0xfff])<<24 |
				uint64(t[b&0xfff])<<32 | uint64(t[(b>>16)&0xfff])<<40 | uint64(t[(b>>32)&0xfff])<<48 | uint64(t[(b>>48)&0xfff])<<56
		}
		sv := archsimd.LoadUint64x2Array(&sc).ReshapeToUint8s()
		fgNoiseHBD(sv, g0, x0, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[0:8]))
		fgNoiseHBD(sv.HiToLo(), g1, x1, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[8:16]))
	}
}

// fguvBlockHBDSX0 is fguv_loop_sx0 (4:4:4) of the 16bpc kernels for one
// full-width 32xbh chroma block.
func fguvBlockHBDSX0(dst, src []uint16, stride int, luma []uint16, lumaStep int, grain []int16, table *[ScalingTableSize]uint8, k *fgConstsHBD, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	nRound := archsimd.LoadInt32x4Array(&k.noiseRound)
	nShift := archsimd.LoadInt32x4Array(&k.noiseShift)
	bRound := archsimd.LoadInt32x4Array(&k.blendRound)
	bShift := archsimd.LoadInt32x4Array(&k.blendShift)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	lm := archsimd.LoadInt16x8Array(&k.uvLumaMult)
	m := archsimd.LoadInt16x8Array(&k.uvMult)
	uvOff := archsimd.LoadInt32x4Array(&k.uvOffset)
	uvSh := archsimd.LoadInt32x4Array(&k.uvShift)
	idxMax := archsimd.LoadInt32x4Array(&k.idxMax)
	minV := archsimd.LoadInt16x8Array(&k.minV)
	maxV := archsimd.LoadInt16x8Array(&k.maxV)
	t := table
	var sc [4]uint64
	var ix [8]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[32]uint16)(src[off : off+32])
		d := (*[32]uint16)(dst[off : off+32])
		li := y * lumaStep
		l := (*[32]uint16)(luma[li : li+32])
		gy := y * ChromaGrainWidth
		g := (*[32]int16)(grain[cur+gy : cur+gy+32])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		g2 := archsimd.LoadInt16x8Array((*[8]int16)(g[16:24]))
		g3 := archsimd.LoadInt16x8Array((*[8]int16)(g[24:32]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlendHBD(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			tr := (*[32]int16)(grain[top+gy : top+gy+32])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(tr[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlendHBD(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlendHBD(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
			g2 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[16:24])), g2, wt, wc, bRound, bShift, gMin, gMax)
			g3 = fgBlendHBD(archsimd.LoadInt16x8Array((*[8]int16)(tr[24:32])), g3, wt, wc, bRound, bShift, gMin, gMax)
		}
		x0 := archsimd.LoadUint16x8Array((*[8]uint16)(s[0:8]))
		x1 := archsimd.LoadUint16x8Array((*[8]uint16)(s[8:16]))
		x2 := archsimd.LoadUint16x8Array((*[8]uint16)(s[16:24]))
		x3 := archsimd.LoadUint16x8Array((*[8]uint16)(s[24:32]))
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(l[0:8]))
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(l[8:16]))
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(l[16:24]))
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(l[24:32]))
		if !csfl {
			a0 = fguvIndexHBD(a0, x0, lm, m, uvOff, uvSh, idxMax)
			a1 = fguvIndexHBD(a1, x1, lm, m, uvOff, uvSh, idxMax)
			a2 = fguvIndexHBD(a2, x2, lm, m, uvOff, uvSh, idxMax)
			a3 = fguvIndexHBD(a3, x3, lm, m, uvOff, uvSh, idxMax)
		}
		w0, w1, w2, w3 := a0.ReshapeToUint64s(), a1.ReshapeToUint64s(), a2.ReshapeToUint64s(), a3.ReshapeToUint64s()
		ix = [8]uint64{w0.GetElem(0), w0.GetElem(1), w1.GetElem(0), w1.GetElem(1), w2.GetElem(0), w2.GetElem(1), w3.GetElem(0), w3.GetElem(1)}
		for j := range sc {
			a, b := ix[2*j], ix[2*j+1]
			sc[j] = uint64(t[a&0xfff]) | uint64(t[(a>>16)&0xfff])<<8 | uint64(t[(a>>32)&0xfff])<<16 | uint64(t[(a>>48)&0xfff])<<24 |
				uint64(t[b&0xfff])<<32 | uint64(t[(b>>16)&0xfff])<<40 | uint64(t[(b>>32)&0xfff])<<48 | uint64(t[(b>>48)&0xfff])<<56
		}
		s0 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[0:2])).ReshapeToUint8s()
		s1 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[2:4])).ReshapeToUint8s()
		fgNoiseHBD(s0, g0, x0, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[0:8]))
		fgNoiseHBD(s0.HiToLo(), g1, x1, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[8:16]))
		fgNoiseHBD(s1, g2, x2, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[16:24]))
		fgNoiseHBD(s1.HiToLo(), g3, x3, nRound, nShift, minV, maxV).StoreArray((*[8]uint16)(d[24:32]))
	}
}

// fgLumaBlockHBD places grain on one 10/12-bit luma block: full-width
// blocks take fgyBlockHBD, a partial right-edge block the scalar path.
func fgLumaBlockHBD(dst, src []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	if blockWidth != LumaBlockSize {
		applyLumaBlock(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := lumaGrainSampleIndex(offsets[0][0], 0, 0, 0, 0)
	old := lumaGrainSampleIndex(offsets[1][0], 1, 0, 0, 0)
	top := lumaGrainSampleIndex(offsets[0][1], 0, 1, 0, 0)
	topOld := lumaGrainSampleIndex(offsets[1][1], 1, 1, 0, 0)
	fgyBlockHBD(dst[blockX:], src[blockX:], params.Stride, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0)
}

// fgChromaBlockHBD is fgLumaBlockHBD for a chroma block.
func fgChromaBlockHBD(dst, src, luma []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	if blockWidth != LumaBlockSize>>shiftX {
		applyChromaBlock(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := chromaGrainSampleIndex(offsets[0][0], shiftX, shiftY, 0, 0, 0, 0)
	old := chromaGrainSampleIndex(offsets[1][0], shiftX, shiftY, 1, 0, 0, 0)
	top := chromaGrainSampleIndex(offsets[0][1], shiftX, shiftY, 0, 1, 0, 0)
	topOld := chromaGrainSampleIndex(offsets[1][1], shiftX, shiftY, 1, 1, 0, 0)
	lumaRow := luma[blockX<<shiftX:]
	if shiftX != 0 {
		fguvBlockHBDSX1(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
		return
	}
	fguvBlockHBDSX0(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, table, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
}
