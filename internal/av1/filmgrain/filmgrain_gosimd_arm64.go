// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import (
	"encoding/binary"
	"simd/archsimd"
)

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
	// The kernel averages full luma pairs, so an odd-width edge block (whose
	// last column has a single luma sample) takes the scalar path.
	if blockWidth != LumaBlockSize>>shiftX || oddLumaEdge(params, blockX, blockWidth) {
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

// fgConstsU8 holds the per-row-band vector constants, the registers
// fgy_32x32_8bpc_neon / fguv_32x32_*_8bpc_neon set up before their row loop.
// They live in memory and are loaded into vectors once per block: archsimd
// lowers broadcasts and constant shifts to GPR->lane inserts.
type fgConstsU8 struct {
	// shift is -(scaling_shift-1). dav1d rounds the scaled noise with
	// srshl by -scaling_shift; archsimd has no SRSHL, so the kernels use
	// round2(p, s) = (p >> (s-1)) + 1) >> 1, the second step being SRHADD
	// (Average) with zero.
	shift [8]int16
	// The overlap blend round2(old*w0 + cur*w1, 5) clipped to the 8-bit
	// grain range, as smull/smlal + sqrshrn #5.
	blendShift [8]int16
	blendRound [8]int16
	grainMin   [8]int16
	grainMax   [8]int16
	// Left-overlap weights in dav1d's overlap_coeffs lane layout: the
	// overlapped columns blend, every other lane computes cur*32, which the
	// rounding shift returns unchanged.
	oxOld [8]int16
	oxCur [8]int16
	// Top-overlap weights per overlapped row.
	oyTop [2][8]int16
	oyCur [2][8]int16
	// !chroma_scaling_from_luma index: avg*uv_luma_mult + src*uv_mult (sqadd),
	// >> 6, + uv_offset.
	uvLumaMult [8]int16
	uvMult     [8]int16
	uvOffset   [8]int16
	uvShift    [8]int16
	minB       [16]uint8
	maxB       [16]uint8
}

func fill8(dst *[8]int16, v int) {
	for i := range dst {
		dst[i] = int16(v)
	}
}

func (k *fgConstsU8) init(scalingShift int, minValue, maxValue int, sx int, sy int) {
	fill8(&k.shift, -(scalingShift - 1))
	fill8(&k.blendShift, -5)
	fill8(&k.blendRound, 16)
	fill8(&k.grainMin, -128)
	fill8(&k.grainMax, 127)
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
	for i := range k.minB {
		k.minB[i] = uint8(minValue)
		k.maxB[i] = uint8(maxValue)
	}
}

func (k *fgConstsU8) initChroma(params *ChromaRowParams) {
	fill8(&k.uvLumaMult, int(params.ChromaLumaMult)-128)
	fill8(&k.uvMult, int(params.ChromaMult)-128)
	fill8(&k.uvOffset, int(params.ChromaOffset)-256)
	fill8(&k.uvShift, -6)
}

// fgBlend is round2(old*wOld + cur*wCur, 5) clipped to [-128, 127]. The sum
// is at most 128*45 in magnitude, so int16 lanes cannot wrap.
func fgBlend(old, cur, wOld, wCur, round, shift, lo, hi archsimd.Int16x8) archsimd.Int16x8 {
	return cur.MulAdd(wCur, old.Mul(wOld)).Add(round).Shift(shift).Max(lo).Min(hi)
}

// fgNoise is src + round2(scaling*grain, scaling_shift). scaling <= 255 and
// |grain| <= 128, so the product fits int16 as in dav1d's 8bpc mul.8h.
func fgNoise(scale archsimd.Uint8x16, grain archsimd.Int16x8, src archsimd.Uint8x16, shift, zero archsimd.Int16x8) archsimd.Int16x8 {
	p := scale.ExtendLo8ToUint16().BitsToInt16().Mul(grain)
	return p.Shift(shift).Average(zero).Add(src.ExtendLo8ToUint16().BitsToInt16())
}

// fgPack is sqxtun + sqxtun2.
func fgPack(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	return lo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
}

// fgyBlockU8 is fgy_32x32_8bpc_neon for one full-width 32xbh luma block.
// cur, old, top and topOld index grain at the block's (0,0) sample for the
// current, left, top and top-left offsets; ystart rows blend with the top
// block, and ox blends the left two columns.
func fgyBlockU8(dst, src []uint8, stride int, grain []int16, scaling *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox bool) {
	lut := scaling
	shift := archsimd.LoadInt16x8Array(&k.shift)
	bShift := archsimd.LoadInt16x8Array(&k.blendShift)
	bRound := archsimd.LoadInt16x8Array(&k.blendRound)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	minB := archsimd.LoadUint8x16Array(&k.minB)
	maxB := archsimd.LoadUint8x16Array(&k.maxB)
	var zero archsimd.Int16x8
	var sc [4]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[32]uint8)(src[off : off+32])
		d := (*[32]uint8)(dst[off : off+32])
		gy := y * LumaGrainWidth
		g := (*[32]int16)(grain[cur+gy : cur+gy+32])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		g2 := archsimd.LoadInt16x8Array((*[8]int16)(g[16:24]))
		g3 := archsimd.LoadInt16x8Array((*[8]int16)(g[24:32]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlend(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			t := (*[32]int16)(grain[top+gy : top+gy+32])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(t[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlend(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlend(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
			g2 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[16:24])), g2, wt, wc, bRound, bShift, gMin, gMax)
			g3 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[24:32])), g3, wt, wc, bRound, bShift, gMin, gMax)
		}
		// dav1d's gather (umov + ld1 lane per sample) reads the scaling LUT
		// once per sample. Here eight indices come out of one 64-bit load and
		// the eight results pack into one register, so there are no per-sample
		// index loads or byte stores; the gather stays inline because a call
		// would clobber every vector register in the row loop.
		for j := range sc {
			v := binary.LittleEndian.Uint64(s[j*8:])
			sc[j] = uint64(lut[v&0xff]) | uint64(lut[(v>>8)&0xff])<<8 | uint64(lut[(v>>16)&0xff])<<16 | uint64(lut[(v>>24)&0xff])<<24 |
				uint64(lut[(v>>32)&0xff])<<32 | uint64(lut[(v>>40)&0xff])<<40 | uint64(lut[(v>>48)&0xff])<<48 | uint64(lut[v>>56])<<56
		}
		x0 := archsimd.LoadUint8x16Array((*[16]uint8)(s[0:16]))
		x1 := archsimd.LoadUint8x16Array((*[16]uint8)(s[16:32]))
		s0 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[0:2])).ReshapeToUint8s()
		s1 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[2:4])).ReshapeToUint8s()
		n0 := fgNoise(s0, g0, x0, shift, zero)
		n1 := fgNoise(s0.HiToLo(), g1, x0.HiToLo(), shift, zero)
		n2 := fgNoise(s1, g2, x1, shift, zero)
		n3 := fgNoise(s1.HiToLo(), g3, x1.HiToLo(), shift, zero)
		fgPack(n0, n1).Max(minB).Min(maxB).StoreArray((*[16]uint8)(d[0:16]))
		fgPack(n2, n3).Max(minB).Min(maxB).StoreArray((*[16]uint8)(d[16:32]))
	}
}

// fguvIndex16 is fguv's !csfl scaling index for 16 chroma samples:
// clip_pixel(((avg*uv_luma_mult + src*uv_mult) >> 6) + uv_offset). The sum
// saturates in int16 (sqadd) like dav1d: a saturated sum >> 6 is +-512, and
// any uv_offset in [-256, 255] then clips to the same 0 or 255 as the exact
// sum.
func fguvIndex16(avg, src archsimd.Uint8x16, lm, m, off, sh archsimd.Int16x8) archsimd.Uint8x16 {
	aLo := avg.ExtendLo8ToUint16().BitsToInt16().Mul(lm)
	aHi := avg.HiToLo().ExtendLo8ToUint16().BitsToInt16().Mul(lm)
	sLo := src.ExtendLo8ToUint16().BitsToInt16().Mul(m)
	sHi := src.HiToLo().ExtendLo8ToUint16().BitsToInt16().Mul(m)
	lo := aLo.AddSaturated(sLo).Shift(sh).Add(off)
	hi := aHi.AddSaturated(sHi).Shift(sh).Add(off)
	return fgPack(lo, hi)
}

// fguvBlockU8SX1 is fguv_loop_sx1 (4:2:0 / 4:2:2) for one full-width 16xbh
// chroma block. lumaStep is the luma stride times 1<<ss_y.
func fguvBlockU8SX1(dst, src []uint8, stride int, luma []uint8, lumaStep int, grain []int16, scaling *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	lut := scaling
	shift := archsimd.LoadInt16x8Array(&k.shift)
	bShift := archsimd.LoadInt16x8Array(&k.blendShift)
	bRound := archsimd.LoadInt16x8Array(&k.blendRound)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	lm := archsimd.LoadInt16x8Array(&k.uvLumaMult)
	m := archsimd.LoadInt16x8Array(&k.uvMult)
	uvOff := archsimd.LoadInt16x8Array(&k.uvOffset)
	uvSh := archsimd.LoadInt16x8Array(&k.uvShift)
	minB := archsimd.LoadUint8x16Array(&k.minB)
	maxB := archsimd.LoadUint8x16Array(&k.maxB)
	var zero archsimd.Int16x8
	var sc [2]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[16]uint8)(src[off : off+16])
		d := (*[16]uint8)(dst[off : off+16])
		li := y * lumaStep
		l := (*[32]uint8)(luma[li : li+32])
		gy := y * ChromaGrainWidth
		g := (*[16]int16)(grain[cur+gy : cur+gy+16])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlend(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			t := (*[16]int16)(grain[top+gy : top+gy+16])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(t[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlend(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlend(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
		}
		x := archsimd.LoadUint8x16Array(s)
		l0 := archsimd.LoadUint8x16Array((*[16]uint8)(l[0:16]))
		l1 := archsimd.LoadUint8x16Array((*[16]uint8)(l[16:32]))
		// (l[2i] + l[2i+1] + 1) >> 1: uzp1/uzp2 then urhadd.
		avg := l0.ConcatEven(l1).Average(l0.ConcatOdd(l1))
		if !csfl {
			avg = fguvIndex16(avg, x, lm, m, uvOff, uvSh)
		}
		iv := avg.ReshapeToUint64s()
		ix := [2]uint64{iv.GetElem(0), iv.GetElem(1)}
		for j := range sc {
			v := ix[j]
			sc[j] = uint64(lut[v&0xff]) | uint64(lut[(v>>8)&0xff])<<8 | uint64(lut[(v>>16)&0xff])<<16 | uint64(lut[(v>>24)&0xff])<<24 |
				uint64(lut[(v>>32)&0xff])<<32 | uint64(lut[(v>>40)&0xff])<<40 | uint64(lut[(v>>48)&0xff])<<48 | uint64(lut[v>>56])<<56
		}
		sv := archsimd.LoadUint64x2Array(&sc).ReshapeToUint8s()
		n0 := fgNoise(sv, g0, x, shift, zero)
		n1 := fgNoise(sv.HiToLo(), g1, x.HiToLo(), shift, zero)
		fgPack(n0, n1).Max(minB).Min(maxB).StoreArray(d)
	}
}

// fguvBlockU8SX0 is fguv_loop_sx0 (4:4:4) for one full-width 32xbh chroma
// block. lumaStep is the luma stride.
func fguvBlockU8SX0(dst, src []uint8, stride int, luma []uint8, lumaStep int, grain []int16, scaling *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	lut := scaling
	shift := archsimd.LoadInt16x8Array(&k.shift)
	bShift := archsimd.LoadInt16x8Array(&k.blendShift)
	bRound := archsimd.LoadInt16x8Array(&k.blendRound)
	gMin := archsimd.LoadInt16x8Array(&k.grainMin)
	gMax := archsimd.LoadInt16x8Array(&k.grainMax)
	oxOld := archsimd.LoadInt16x8Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x8Array(&k.oxCur)
	lm := archsimd.LoadInt16x8Array(&k.uvLumaMult)
	m := archsimd.LoadInt16x8Array(&k.uvMult)
	uvOff := archsimd.LoadInt16x8Array(&k.uvOffset)
	uvSh := archsimd.LoadInt16x8Array(&k.uvShift)
	minB := archsimd.LoadUint8x16Array(&k.minB)
	maxB := archsimd.LoadUint8x16Array(&k.maxB)
	var zero archsimd.Int16x8
	var sc [4]uint64
	for y := 0; y < bh; y++ {
		off := y * stride
		s := (*[32]uint8)(src[off : off+32])
		d := (*[32]uint8)(dst[off : off+32])
		li := y * lumaStep
		l := (*[32]uint8)(luma[li : li+32])
		gy := y * ChromaGrainWidth
		g := (*[32]int16)(grain[cur+gy : cur+gy+32])
		g0 := archsimd.LoadInt16x8Array((*[8]int16)(g[0:8]))
		g1 := archsimd.LoadInt16x8Array((*[8]int16)(g[8:16]))
		g2 := archsimd.LoadInt16x8Array((*[8]int16)(g[16:24]))
		g3 := archsimd.LoadInt16x8Array((*[8]int16)(g[24:32]))
		if ox {
			o := archsimd.LoadInt16x8Array((*[8]int16)(grain[old+gy : old+gy+8]))
			g0 = fgBlend(o, g0, oxOld, oxCur, bRound, bShift, gMin, gMax)
		}
		if y < ystart {
			t := (*[32]int16)(grain[top+gy : top+gy+32])
			t0 := archsimd.LoadInt16x8Array((*[8]int16)(t[0:8]))
			if ox {
				o := archsimd.LoadInt16x8Array((*[8]int16)(grain[topOld+gy : topOld+gy+8]))
				t0 = fgBlend(o, t0, oxOld, oxCur, bRound, bShift, gMin, gMax)
			}
			wt := archsimd.LoadInt16x8Array(&k.oyTop[y])
			wc := archsimd.LoadInt16x8Array(&k.oyCur[y])
			g0 = fgBlend(t0, g0, wt, wc, bRound, bShift, gMin, gMax)
			g1 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[8:16])), g1, wt, wc, bRound, bShift, gMin, gMax)
			g2 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[16:24])), g2, wt, wc, bRound, bShift, gMin, gMax)
			g3 = fgBlend(archsimd.LoadInt16x8Array((*[8]int16)(t[24:32])), g3, wt, wc, bRound, bShift, gMin, gMax)
		}
		x0 := archsimd.LoadUint8x16Array((*[16]uint8)(s[0:16]))
		x1 := archsimd.LoadUint8x16Array((*[16]uint8)(s[16:32]))
		a0 := archsimd.LoadUint8x16Array((*[16]uint8)(l[0:16]))
		a1 := archsimd.LoadUint8x16Array((*[16]uint8)(l[16:32]))
		if !csfl {
			a0 = fguvIndex16(a0, x0, lm, m, uvOff, uvSh)
			a1 = fguvIndex16(a1, x1, lm, m, uvOff, uvSh)
		}
		i0, i1 := a0.ReshapeToUint64s(), a1.ReshapeToUint64s()
		ix := [4]uint64{i0.GetElem(0), i0.GetElem(1), i1.GetElem(0), i1.GetElem(1)}
		for j := range sc {
			v := ix[j]
			sc[j] = uint64(lut[v&0xff]) | uint64(lut[(v>>8)&0xff])<<8 | uint64(lut[(v>>16)&0xff])<<16 | uint64(lut[(v>>24)&0xff])<<24 |
				uint64(lut[(v>>32)&0xff])<<32 | uint64(lut[(v>>40)&0xff])<<40 | uint64(lut[(v>>48)&0xff])<<48 | uint64(lut[v>>56])<<56
		}
		s0 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[0:2])).ReshapeToUint8s()
		s1 := archsimd.LoadUint64x2Array((*[2]uint64)(sc[2:4])).ReshapeToUint8s()
		n0 := fgNoise(s0, g0, x0, shift, zero)
		n1 := fgNoise(s0.HiToLo(), g1, x0.HiToLo(), shift, zero)
		n2 := fgNoise(s1, g2, x1, shift, zero)
		n3 := fgNoise(s1.HiToLo(), g3, x1.HiToLo(), shift, zero)
		fgPack(n0, n1).Max(minB).Min(maxB).StoreArray((*[16]uint8)(d[0:16]))
		fgPack(n2, n3).Max(minB).Min(maxB).StoreArray((*[16]uint8)(d[16:32]))
	}
}

// fgLumaBlockU8 places grain on one luma block of a row band: full-width
// blocks take fgyBlockU8, a partial right-edge block the scalar path.
func fgLumaBlockU8(dst, src []uint8, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	if blockWidth != LumaBlockSize {
		applyLumaBlockU8(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := lumaGrainSampleIndex(offsets[0][0], 0, 0, 0, 0)
	old := lumaGrainSampleIndex(offsets[1][0], 1, 0, 0, 0)
	top := lumaGrainSampleIndex(offsets[0][1], 0, 1, 0, 0)
	topOld := lumaGrainSampleIndex(offsets[1][1], 1, 1, 0, 0)
	fgyBlockU8(dst[blockX:], src[blockX:], params.Stride, grain, (*[ScalingLUTSize]uint8)(scaling), k, cur, old, top, topOld, params.Height, yStart, xStart != 0)
}

// fgChromaBlockU8 is fgLumaBlockU8 for a chroma block.
func fgChromaBlockU8(dst, src, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	// The kernel averages full luma pairs, so an odd-width edge block (whose
	// last column has a single luma sample) takes the scalar path.
	if blockWidth != LumaBlockSize>>shiftX || oddLumaEdge(params, blockX, blockWidth) {
		applyChromaBlockU8(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := chromaGrainSampleIndex(offsets[0][0], shiftX, shiftY, 0, 0, 0, 0)
	old := chromaGrainSampleIndex(offsets[1][0], shiftX, shiftY, 1, 0, 0, 0)
	top := chromaGrainSampleIndex(offsets[0][1], shiftX, shiftY, 0, 1, 0, 0)
	topOld := chromaGrainSampleIndex(offsets[1][1], shiftX, shiftY, 1, 1, 0, 0)
	lumaRow := luma[blockX<<shiftX:]
	lut := (*[ScalingLUTSize]uint8)(scaling)
	if shiftX != 0 {
		fguvBlockU8SX1(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, lut, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
		return
	}
	fguvBlockU8SX0(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, lut, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
}

// blendGrainRow is the overlap two-tap blend entry point under GOEXPERIMENT=simd
// on arm64; blendGrainRowPureGo remains the bit-exact reference. The call is
// concrete (not a func variable) so the caller's blend scratch buffer stays on
// the stack and the apply path stays zero-alloc.
func blendGrainRow(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	blendGrainRowSIMD(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
}

// blendGrainRowSIMD is the Go SIMD twin of blendGrainRowPureGo. Eight grain
// samples per iteration: prev and cur widen to int32 lanes, form
// prev*prevWeight+cur*curWeight exactly (grain fits int16 and the weights sum
// to 44 or 45, so the sum fits int32), apply the rounding shift that reproduces
// roundPowerOfTwo(v, 5) with an arithmetic right shift, and clamp to
// [grainMin, grainMax]. The clamped values fit int16, so the narrow is exact.
// The final (<8) samples take the scalar reference.
func blendGrainRowSIMD(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	n := len(dst)
	// Slicing every operand to n checks each one once. The group loop then
	// consumes the heads of these slices; its condition proves each holds a
	// full group, so the array-pointer conversions need no bounds checks.
	pr, cr, dl := prev[:n], cur[:n], dst[:n]
	pw := archsimd.BroadcastInt32x4(int32(prevWeight))
	cw := archsimd.BroadcastInt32x4(int32(curWeight))
	round := archsimd.BroadcastInt32x4(1 << 4)
	minV := archsimd.BroadcastInt32x4(int32(grainMin))
	maxV := archsimd.BroadcastInt32x4(int32(grainMax))
	for len(dl) >= 8 && len(pr) >= 8 && len(cr) >= 8 {
		pLo, pHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(pr)))
		cLo, cHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(cr)))
		lo := pLo.Mul(pw).Add(cLo.Mul(cw)).Add(round).ShiftAllRight(5).Max(minV).Min(maxV)
		hi := pHi.Mul(pw).Add(cHi.Mul(cw)).Add(round).ShiftAllRight(5).Max(minV).Min(maxV)
		grainPack8(lo, hi).BitsToInt16().StoreArray((*[8]int16)(dl))
		pr, cr, dl = pr[8:], cr[8:], dl[8:]
	}
	for k := range dl {
		v := roundPowerOfTwo(int(pr[k])*prevWeight+int(cr[k])*curWeight, 5)
		dl[k] = int16(clipInt(v, grainMin, grainMax))
	}
}

// arLag3Taps holds the 21 above-row lag-3 coefficients (rows y-3..y-1,
// columns x-3..x+3) as int16 lanes, loaded per use: archsimd lowers
// broadcasts to GPR->lane inserts.
type arLag3Taps [21][8]int16

func (t *arLag3Taps) init(coeffs []int8) {
	for i := range t {
		for j := range t[i] {
			t[i][j] = int16(coeffs[i])
		}
	}
}

// arLag3Above is dav1d's sum_lag3_above_neon: the above-row part of the
// lag-3 auto-regression for eight grain samples per step, x = 3..3+n
// (rounded up to whole steps), into above[x-3]. Each tap is a widening
// smull/smull2 into int32, exact for every bit depth. It reads grain rows
// row-3..row-1 only, so the serial left-neighbour pass can follow.
func arLag3Above(above *[80]int32, grain []int16, row int, n int, taps *arLag3Taps) {
	for i := 0; i < n; i += 8 {
		var lo, hi archsimd.Int32x4
		tap := 0
		for r := 3; r >= 1; r-- {
			base := row - r*LumaGrainWidth + i
			g := grain[base : base+14]
			for dc := 0; dc < 7; dc++ {
				v := archsimd.LoadInt16x8Array((*[8]int16)(g[dc : dc+8]))
				c := archsimd.LoadInt16x8Array(&taps[tap])
				lo = lo.Add(v.MulWidenLo(c))
				hi = hi.Add(v.HiToLo().MulWidenLo(c.HiToLo()))
				tap++
			}
		}
		lo.StoreArray((*[4]int32)(above[i : i+4]))
		hi.StoreArray((*[4]int32)(above[i+4 : i+8]))
	}
}

// arLag3 runs the lag-3 auto-regression on rows 3..height-1, columns
// 3..width-4: the above rows through arLag3Above, then the serial
// left-neighbour taps. For chroma (luma != nil) coeffs[24] multiplies the
// co-located luma grain average, as chromaLumaAverage.
func arLag3(grain []int16, luma []int16, coeffs []int8, sx, sy, width, height, roundingOffset, coeffShift, grainMin, grainMax int) {
	var taps arLag3Taps
	taps.init(coeffs)
	c21, c22, c23 := int(coeffs[21]), int(coeffs[22]), int(coeffs[23])
	cl := 0
	if luma != nil {
		cl = int(coeffs[24])
	}
	var above [80]int32
	for y := 3; y < height; y++ {
		row := y * LumaGrainWidth
		arLag3Above(&above, grain, row, width-6, &taps)
		if luma != nil {
			lumaRow := (((y - 3) << sy) + 3) * LumaGrainWidth
			for x := 3; x < width-3; x++ {
				li := lumaRow + ((x - 3) << sx) + 3
				avg := int(luma[li])
				if sx != 0 {
					avg += int(luma[li+1])
				}
				if sy != 0 {
					avg += int(luma[li+LumaGrainWidth])
					if sx != 0 {
						avg += int(luma[li+LumaGrainWidth+1])
					}
				}
				above[x-3] += int32(roundPowerOfTwo(avg, sx+sy) * cl)
			}
		}
		g := grain[row : row+width]
		for x := 3; x < width-3; x++ {
			sum := int(above[x-3]) + c21*int(g[x-3]) + c22*int(g[x-2]) + c23*int(g[x-1])
			g[x] = int16(clipInt(int(g[x])+((sum+roundingOffset)>>coeffShift), grainMin, grainMax))
		}
	}
}

func lumaGrainARLag3(grain []int16, params *LumaGrainParams, roundingOffset, coeffShift, grainMin, grainMax int) {
	arLag3(grain, nil, params.ARCoeffs[:], 0, 0, LumaGrainWidth, LumaGrainHeight, roundingOffset, coeffShift, grainMin, grainMax)
}

func chromaGrainARLag3(grain []int16, luma []int16, params *ChromaGrainParams, width, height, roundingOffset, coeffShift, grainMin, grainMax int) {
	if params.NumYPoints == 0 {
		luma = nil
	}
	arLag3(grain, luma, params.ARCoeffs[:], chromaSubsamplingShift(params.SubsamplingX), chromaSubsamplingShift(params.SubsamplingY), width, height, roundingOffset, coeffShift, grainMin, grainMax)
}

// grainSIMDAvailable reports whether the Go SIMD grain kernels may run. NEON
// (AdvSIMD) is mandatory on every arm64 target Go runs on.
func grainSIMDAvailable() bool {
	return true
}

// grainPack8 narrows eight int32 values (lo holds lanes 0..3, hi holds lanes
// 4..7; every value is already clamped into the int16 range) to uint16 lanes in
// order. SaturateToInt16 leaves each four-lane result in the low half of its
// vector, so a 64-bit interleave packs the two halves into one row.
func grainPack8(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	l := lo.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	h := hi.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint16s()
}
