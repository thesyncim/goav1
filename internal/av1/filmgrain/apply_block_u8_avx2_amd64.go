// Ported from dav1d: src/x86/filmgrain_avx2.asm (fgy_32x32xn_8bpc,
// fguv_32x32xn_i420/i422/i444_8bpc)
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

//go:build goexperiment.simd && amd64 && !purego

package filmgrain

import (
	"encoding/binary"
	"simd/archsimd"
)

// fgAVX2 gates the AVX2 block kernels; without it every block takes the
// scalar path.
var fgAVX2 = archsimd.X86.AVX2()

// fgConstsU8 holds the per-row-band constants of the AVX2 block kernels, the
// values dav1d's fgy/fguv_32x32xn_8bpc broadcast before their row loop.
type fgConstsU8 struct {
	scalingShift uint64
	// noiseMul is 2^(17 - scaling_shift): vpmulhw by it is an exact
	// arithmetic >> (scaling_shift - 1) of the int16 product, with no
	// variable shift count (whose legacy-SSE movq stalls on Zen 3).
	noiseMul [16]int16
	// ix is the index spill of fgGatherIdx16, kept here so the row loops do
	// not re-zero a local array with legacy-SSE stores.
	ix [16]int16
	// The left-overlap weights in dav1d's lane layout: the overlapped
	// columns blend, every other lane computes cur*32, which the rounding
	// shift returns unchanged.
	oxOld, oxCur [16]int16
	// Top-overlap weights per overlapped row.
	oyTop, oyCur [2]int16
	minV, maxV   int16
	// !chroma_scaling_from_luma index: avg*uv_luma_mult + src*uv_mult
	// (saturating), >> 6, + uv_offset.
	uvLumaMult, uvMult, uvOffset int16
}

func (k *fgConstsU8) init(scalingShift int, minValue, maxValue int, sx int, sy int) {
	k.scalingShift = uint64(scalingShift)
	for i := range k.noiseMul {
		k.noiseMul[i] = int16(1) << (17 - scalingShift)
	}
	k.minV, k.maxV = int16(minValue), int16(maxValue)
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
}

func (k *fgConstsU8) initChroma(params *ChromaRowParams) {
	k.uvLumaMult = int16(int(params.ChromaLumaMult) - 128)
	k.uvMult = int16(int(params.ChromaMult) - 128)
	k.uvOffset = int16(int(params.ChromaOffset) - 256)
}

// fgBlend16 is round2(old*wOld + cur*wCur, 5) clipped to [-128, 127]. The
// sum is at most 128*45 in magnitude, so int16 lanes cannot wrap.
func fgBlend16(old, cur, wOld, wCur archsimd.Int16x16) archsimd.Int16x16 {
	v := old.Mul(wOld).Add(cur.Mul(wCur)).Add(archsimd.BroadcastInt16x16(16)).ShiftAllRight(5)
	return v.Max(archsimd.BroadcastInt16x16(-128)).Min(archsimd.BroadcastInt16x16(127))
}

// fgNoise16 is clip(src + round2(scale*grain, shift), minV, maxV) on sixteen
// samples. scale <= 255 and |grain| <= 128, so the product fits int16 as in
// dav1d's pmaddubsw; round2(p, s) is ((p >> (s-1)) + 1) >> 1 (no pmulhrsw),
// the first shift as vpmulhw by mul = 2^(17-s).
func fgNoise16(scale, grain, src, mul archsimd.Int16x16, minV, maxV archsimd.Int16x16) archsimd.Int16x16 {
	p := scale.Mul(grain).MulHigh(mul).Add(archsimd.BroadcastInt16x16(1)).ShiftAllRight(1)
	return src.Add(p).Max(minV).Min(maxV)
}

// fgWiden16 zero-extends sixteen bytes to int16 lanes.
func fgWiden16(b []uint8) archsimd.Int16x16 {
	return archsimd.LoadUint8x16(b[:16]).ExtendToUint16().AsInt16x16()
}

// fgPackLo picks the low byte of each int16 lane into the low half of its
// 128-bit lane (vpshufb; -1 zeroes).
var fgPackLo = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

// fgPackEvenOdd splits each 128-bit lane's bytes into its even then odd
// bytes.
var fgPackEvenOdd = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
	0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
}

// fgQwordOrder is the vpermd order that puts the qwords (0, 2, 1, 3) back in
// sample order.
var fgQwordOrder = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// fgStore32 stores two sixteen-sample int16 rows already clipped to
// [0, 255] as 32 bytes: the second row goes into the high byte of each
// lane, vpshufb separates the bytes within 128-bit lanes and vpermd restores
// the qword order (vpackuswb is not in archsimd).
func fgStore32(d []uint8, n0, n1 archsimd.Int16x16) {
	w := n0.Or(n1.ShiftAllLeft(8)).AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&fgPackEvenOdd))
	w.AsUint32x8().Permute(archsimd.LoadUint32x8Array(&fgQwordOrder)).AsUint8x32().Store(d[:32])
}

// fgStore16 stores one sixteen-sample int16 row clipped to [0, 255].
func fgStore16(d []uint8, n archsimd.Int16x16) {
	b := n.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&fgPackLo))
	b.GetLo().AsUint64x2().InterleaveLo(b.GetHi().AsUint64x2()).AsUint8x16().Store(d[:16])
}

// fgGather8 looks up eight scaling values for the eight index bytes of v.
func fgGather8(lut *[ScalingLUTSize]uint8, v uint64) uint64 {
	return uint64(lut[v&0xff]) | uint64(lut[(v>>8)&0xff])<<8 | uint64(lut[(v>>16)&0xff])<<16 | uint64(lut[(v>>24)&0xff])<<24 |
		uint64(lut[(v>>32)&0xff])<<32 | uint64(lut[(v>>40)&0xff])<<40 | uint64(lut[(v>>48)&0xff])<<48 | uint64(lut[v>>56])<<56
}

// fgScale16 widens the sixteen gathered scaling bytes of sc to int16.
func fgScale16(sc *[2]uint64) archsimd.Int16x16 {
	return archsimd.LoadUint64x2Array(sc).AsUint8x16().ExtendToUint16().AsInt16x16()
}

// fgLeft8 loads the eight left-block grain samples at off into the low
// lanes (only the first one or two are weighted, and the run may end at the
// end of the grain template).
func fgLeft8(grain []int16, off int) archsimd.Int16x16 {
	return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8(grain[off : off+8]))
}

// fgGrainRow loads the sixteen-sample grain run at off, blending the left
// overlap (old) and the top overlap row (top, topOld) as dav1d does.
func fgGrainRow(grain []int16, cur, old, top, topOld int, ox, oy bool, k *fgConstsU8, wt, wc archsimd.Int16x16) archsimd.Int16x16 {
	g := archsimd.LoadInt16x16(grain[cur : cur+16])
	oxOld := archsimd.LoadInt16x16Array(&k.oxOld)
	oxCur := archsimd.LoadInt16x16Array(&k.oxCur)
	if ox {
		g = fgBlend16(fgLeft8(grain, old), g, oxOld, oxCur)
	}
	if oy {
		t := archsimd.LoadInt16x16(grain[top : top+16])
		if ox {
			t = fgBlend16(fgLeft8(grain, topOld), t, oxOld, oxCur)
		}
		g = fgBlend16(t, g, wt, wc)
	}
	return g
}

// fgyBlockU8AVX2 is fgy_32x32xn_8bpc for one full-width 32xbh luma block.
// cur, old, top and topOld index grain at the block's (0,0) sample for the
// current, left, top and top-left offsets; rows below ystart blend with the
// top block, and ox blends the left two columns.
func fgyBlockU8AVX2(dst, src []uint8, stride int, grain []int16, lut *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox bool) {
	var sc0, sc1 [2]uint64
	minV, maxV := archsimd.BroadcastInt16x16(k.minV), archsimd.BroadcastInt16x16(k.maxV)
	for y := 0; y < bh; y++ {
		off := y * stride
		s := src[off : off+32]
		gy := y * LumaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g0 := fgGrainRow(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		g1 := fgGrainRow(grain, cur+gy+16, 0, top+gy+16, 0, false, oy, k, wt, wc)
		// dav1d gathers with vpgatherdd; archsimd has no gather, so eight
		// indices come out of one 64-bit load and the eight results pack
		// into one word, as the arm64 kernel does.
		sc0[0] = fgGather8(lut, binary.LittleEndian.Uint64(s[0:]))
		sc0[1] = fgGather8(lut, binary.LittleEndian.Uint64(s[8:]))
		sc1[0] = fgGather8(lut, binary.LittleEndian.Uint64(s[16:]))
		sc1[1] = fgGather8(lut, binary.LittleEndian.Uint64(s[24:]))
		n0 := fgNoise16(fgScale16(&sc0), g0, fgWiden16(s[0:]), archsimd.LoadInt16x16Array(&k.noiseMul), minV, maxV)
		n1 := fgNoise16(fgScale16(&sc1), g1, fgWiden16(s[16:]), archsimd.LoadInt16x16Array(&k.noiseMul), minV, maxV)
		fgStore32(dst[off:off+32], n0, n1)
	}
	archsimd.ClearAVXUpperBits()
}

// fguvIndex16 is fguv's !csfl scaling index for sixteen chroma samples:
// clip_pixel(((avg*uv_luma_mult + src*uv_mult) >> 6) + uv_offset). The sum
// saturates in int16 (paddsw) like dav1d: a saturated sum >> 6 is +-512, and
// any uv_offset in [-256, 255] then clips to the same 0 or 255 as the exact
// sum.
func fguvIndex16(avg, src archsimd.Int16x16, k *fgConstsU8) archsimd.Int16x16 {
	a := avg.Mul(archsimd.BroadcastInt16x16(k.uvLumaMult))
	b := src.Mul(archsimd.BroadcastInt16x16(k.uvMult))
	v := a.AddSaturated(b).ShiftAllRight(6).Add(archsimd.BroadcastInt16x16(k.uvOffset))
	return v.Max(archsimd.Int16x16{}).Min(archsimd.BroadcastInt16x16(255))
}

// fgGatherIdx16 looks up the scaling values of sixteen int16 indices.
func fgGatherIdx16(lut *[ScalingLUTSize]uint8, idx archsimd.Int16x16, sc *[2]uint64, ix *[16]int16) {
	idx.StoreArray(ix)
	for j := range sc {
		var v uint64
		for i := 7; i >= 0; i-- {
			v = v<<8 | uint64(lut[uint8(ix[8*j+i])])
		}
		sc[j] = v
	}
}

// fguvBlockU8SX1AVX2 is fguv_32x32xn_i420/i422_8bpc (the sx loop) for one
// full-width 16xbh chroma block. lumaStep is the luma stride times 1<<ss_y.
func fguvBlockU8SX1AVX2(dst, src []uint8, stride int, luma []uint8, lumaStep int, grain []int16, lut *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	var sc [2]uint64
	minV, maxV := archsimd.BroadcastInt16x16(k.minV), archsimd.BroadcastInt16x16(k.maxV)
	ones := archsimd.BroadcastInt8x32(1)
	for y := 0; y < bh; y++ {
		off := y * stride
		gy := y * ChromaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g := fgGrainRow(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		x := fgWiden16(src[off:])
		li := y * lumaStep
		// (l[2i] + l[2i+1] + 1) >> 1: vpmaddubsw against ones sums the pairs
		// in place, as dav1d does.
		l := archsimd.LoadUint8x32(luma[li : li+32])
		avg := l.DotProductPairsSaturated(ones).Add(archsimd.BroadcastInt16x16(1)).ShiftAllRight(1)
		if !csfl {
			avg = fguvIndex16(avg, x, k)
		}
		fgGatherIdx16(lut, avg, &sc, &k.ix)
		n := fgNoise16(fgScale16(&sc), g, x, archsimd.LoadInt16x16Array(&k.noiseMul), minV, maxV)
		fgStore16(dst[off:off+16], n)
	}
	archsimd.ClearAVXUpperBits()
}

// fguvBlockU8SX0AVX2 is fguv_32x32xn_i444_8bpc for one full-width 32xbh
// chroma block. lumaStep is the luma stride.
func fguvBlockU8SX0AVX2(dst, src []uint8, stride int, luma []uint8, lumaStep int, grain []int16, lut *[ScalingLUTSize]uint8, k *fgConstsU8, cur, old, top, topOld, bh, ystart int, ox, csfl bool) {
	var sc0, sc1 [2]uint64
	minV, maxV := archsimd.BroadcastInt16x16(k.minV), archsimd.BroadcastInt16x16(k.maxV)
	for y := 0; y < bh; y++ {
		off := y * stride
		s := src[off : off+32]
		gy := y * ChromaGrainWidth
		var wt, wc archsimd.Int16x16
		oy := y < ystart
		if oy {
			wt, wc = archsimd.BroadcastInt16x16(k.oyTop[y]), archsimd.BroadcastInt16x16(k.oyCur[y])
		}
		g0 := fgGrainRow(grain, cur+gy, old+gy, top+gy, topOld+gy, ox, oy, k, wt, wc)
		g1 := fgGrainRow(grain, cur+gy+16, 0, top+gy+16, 0, false, oy, k, wt, wc)
		li := y * lumaStep
		l := luma[li : li+32]
		x0, x1 := fgWiden16(s[0:]), fgWiden16(s[16:])
		if csfl {
			sc0[0] = fgGather8(lut, binary.LittleEndian.Uint64(l[0:]))
			sc0[1] = fgGather8(lut, binary.LittleEndian.Uint64(l[8:]))
			sc1[0] = fgGather8(lut, binary.LittleEndian.Uint64(l[16:]))
			sc1[1] = fgGather8(lut, binary.LittleEndian.Uint64(l[24:]))
		} else {
			fgGatherIdx16(lut, fguvIndex16(fgWiden16(l[0:]), x0, k), &sc0, &k.ix)
			fgGatherIdx16(lut, fguvIndex16(fgWiden16(l[16:]), x1, k), &sc1, &k.ix)
		}
		n0 := fgNoise16(fgScale16(&sc0), g0, x0, archsimd.LoadInt16x16Array(&k.noiseMul), minV, maxV)
		n1 := fgNoise16(fgScale16(&sc1), g1, x1, archsimd.LoadInt16x16Array(&k.noiseMul), minV, maxV)
		fgStore32(dst[off:off+32], n0, n1)
	}
	archsimd.ClearAVXUpperBits()
}

// fgLumaBlockU8 places grain on one luma block of a row band: full-width
// blocks take the AVX2 kernel, a partial right-edge block (or a machine
// without AVX2) the scalar path.
func fgLumaBlockU8(dst, src []uint8, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	if !fgAVX2 || blockWidth != LumaBlockSize {
		applyLumaBlockU8(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
		return
	}
	cur := lumaGrainSampleIndex(offsets[0][0], 0, 0, 0, 0)
	old := lumaGrainSampleIndex(offsets[1][0], 1, 0, 0, 0)
	top := lumaGrainSampleIndex(offsets[0][1], 0, 1, 0, 0)
	topOld := lumaGrainSampleIndex(offsets[1][1], 1, 1, 0, 0)
	fgyBlockU8AVX2(dst[blockX:], src[blockX:], params.Stride, grain, (*[ScalingLUTSize]uint8)(scaling), k, cur, old, top, topOld, params.Height, yStart, xStart != 0)
}

// fgChromaBlockU8 is fgLumaBlockU8 for a chroma block.
func fgChromaBlockU8(dst, src, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	if !fgAVX2 || blockWidth != LumaBlockSize>>shiftX {
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
		fguvBlockU8SX1AVX2(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, lut, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
		return
	}
	fguvBlockU8SX0AVX2(dst[blockX:], src[blockX:], params.Stride, lumaRow, params.LumaStride<<shiftY, grain, lut, k, cur, old, top, topOld, params.Height, yStart, xStart != 0, params.ChromaScalingFromLuma)
}
