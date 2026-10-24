// Ported from dav1d: src/arm/64/filmgrain.S (fgy_32x32_8bpc_neon,
// fguv_32x32_*_8bpc_neon)
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import (
	"encoding/binary"
	"simd/archsimd"
)

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
	if blockWidth != LumaBlockSize>>shiftX {
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
