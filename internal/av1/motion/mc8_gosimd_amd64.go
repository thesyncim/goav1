// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2 8-bit translational motion compensation, ported from dav1d's
// src/x86/mc_avx2.asm put_6tap_8bpc / prep_6tap_8bpc and avg_8bpc. Like dav1d,
// the regular and smooth filters (outer taps zero), the <=4 filters and
// bilinear run as six taps over window taps 1..6. Sharp, widths that are not a
// multiple of 8 and the high-bit-depth paths keep the general kernels.
//
// Structure taken from dav1d:
//   - Horizontal: VPSHUFB builds the (x, x+1) byte pairs of each tap pair and
//     VPMADDUBSW multiplies them by the halved taps packed as int8 pairs (three
//     multiplies per eight outputs).
//   - Vertical, 1D: rows are zero-extended to words and the next row is shifted
//     into the high byte, so one VPMADDUBSW applies a vertical tap pair.
//     (amd64 archsimd has no VPUNPCKLBW; the shift-or forms the same pairs.)
//   - Vertical, 2D: the int16 intermediate rows are word-interleaved
//     (VPUNPCKLWD/HWD) and VPMADDWD applies a vertical tap pair in int32.
//   - Each 256-bit vector holds two output rows (one per 128-bit lane) or 16
//     columns of one row, so per-lane packs land in natural order.
//
// dav1d's pmulhrsw by 2^k is exactly (x + 2^(14-k)) >> (15-k); the rounding
// here is the libaom-exact goav1 form (see mc8_gosimd_arm64.go), with halved
// taps throughout:
//
//	put H:   clip((s + 34) >> 6)               s = sum(half tap * px)
//	put V:   clip((s + 32) >> 6)
//	put 2D:  m = (s + 2) >> 2;  clip((sum(half tap * m) + 512) >> 10)
//	prep H:  (s + 2 + 6144 << 2) >> 2  (unsigned)
//	prep V:  (s + 2 + 6144 << 2) >> 2  (unsigned)
//	prep 2D: (sum(half tap * m) + 32 + 6144 << 6) >> 6
//
// Like dav1d's RET (VZEROUPPER), every entry point clears the upper YMM
// halves before returning: Go code after it uses legacy-SSE encodings (MOVUPS
// copies, X15 zeroing), which carry false dependencies on dirty upper state.
// For the same reason the kernels keep loop-carried vectors 256-bit and use
// immediate shift counts only: both otherwise make the compiler emit
// legacy-SSE MOVUPS / MOVQ inside the loops.
//
// amd64 archsimd has no VPACKUSWB: bytes are narrowed by clamping to [0, 255]
// and VPSHUFB. The 2D results go through VPACKUSDW, which is exact because
// init checks every phase's output range.

const (
	mc8KindRegular = iota
	mc8KindSmooth
	mc8KindSharp
	mc8KindBilinear
	mc8KindRegular4
	mc8KindSmooth4
	mc8KindCount
)

// mc8Phase is one filter phase of the six-tap window in AVX2 form.
type mc8Phase struct {
	b [3]int16 // VPMADDUBSW tap pairs: int8 h[2i] | int8 h[2i+1] << 8 (halved taps)
	d [3]int32 // VPMADDWD tap pairs: int16 h[2i] | int16 h[2i+1] << 16 (halved taps)
	f [2]int16 // four-tap VPMADDUBSW pairs (h[1], h[2]) and (h[3], h[4])
	// The eight-tap forms (dav1d put_8tap) over the full window, for sharp.
	b8 [4]int16
	d8 [4]int32
}

var mc8Phases [mc8KindCount][16]mc8Phase

// mc8SixTap reports whether every phase of kind lies inside window taps 1..6
// and keeps every AVX2 sum exact (see init).
var mc8SixTap [mc8KindCount]bool

// mc8FourTap reports whether every phase of a six-tap kind also has zero
// window taps 0 and 5, so the w=4 horizontal pass can run four taps (dav1d
// put_6tap .h_w4).
var mc8FourTap [mc8KindCount]bool

// mc8EightTap reports whether every phase of kind keeps the eight-tap AVX2
// sums exact. A block runs the eight-tap kernels on both axes when either
// filtered axis is sharp.
var mc8EightTap [mc8KindCount]bool

// mc8AVX2 gates the kernels at run time; tests and benchmarks clear it to
// measure the general path.
var mc8AVX2 = archsimd.X86.AVX2()

func init() {
	tables := [mc8KindCount]*[16][filterTaps]int16{
		mc8KindRegular:  &subpelFilters8,
		mc8KindSmooth:   &subpelFilters8Smooth,
		mc8KindSharp:    &subpelFilters8Sharp,
		mc8KindBilinear: &bilinearFilters,
		mc8KindRegular4: &subpelFilters4,
		mc8KindSmooth4:  &subpelFilters4Smooth,
	}
	for kind, tab := range tables {
		ok, four, eight := true, true, true
		for s := range 16 {
			k := &tab[s]
			var h8 [8]int
			for i, c := range k {
				if c&1 != 0 {
					eight = false
				}
				h8[i] = int(c) / 2
			}
			if !mc8SumsExact(h8[:]) {
				eight = false
			}
			if k[0] != 0 || k[7] != 0 {
				ok = false
			}
			for _, c := range k {
				if c&1 != 0 {
					ok = false
				}
			}
			var h [6]int
			for i := range h {
				h[i] = int(k[1+i]) / 2
			}
			// Each VPMADDUBSW pair and every int16 sum must stay exact.
			sMin, sMax := 0, 0
			for i := 0; i < 6; i += 2 {
				pMin, pMax := 0, 0
				for _, c := range h[i : i+2] {
					if c < 0 {
						pMin += 255 * c
					} else {
						pMax += 255 * c
					}
				}
				if pMin < -32768 || pMax > 32767 {
					ok = false
				}
				sMin += pMin
				sMax += pMax
			}
			// put H/V add 34; prep adds 2 + 6144 << 2 as unsigned.
			if sMin < -32768 || sMax+34 > 32767 || sMin+2+6144<<2 < 0 || sMax+2+6144<<2 > 65535 {
				ok = false
			}
			if h[0] != 0 || h[5] != 0 {
				four = false
			}
			ph := &mc8Phases[kind][s]
			for i := range 4 {
				ph.b8[i] = int16(uint16(uint8(int8(h8[2*i]))) | uint16(uint8(int8(h8[2*i+1])))<<8)
				ph.d8[i] = int32(uint32(uint16(int16(h8[2*i]))) | uint32(uint16(int16(h8[2*i+1])))<<16)
			}
			ph.f[0] = int16(uint16(uint8(int8(h[1]))) | uint16(uint8(int8(h[2])))<<8)
			ph.f[1] = int16(uint16(uint8(int8(h[3]))) | uint16(uint8(int8(h[4])))<<8)
			for i := range 3 {
				ph.b[i] = int16(uint16(uint8(int8(h[2*i]))) | uint16(uint8(int8(h[2*i+1])))<<8)
				ph.d[i] = int32(uint32(uint16(int16(h[2*i]))) | uint32(uint16(int16(h[2*i+1])))<<16)
			}
		}
		mc8SixTap[kind] = ok
		mc8FourTap[kind] = ok && four
		mc8EightTap[kind] = eight
	}
	// The 2D outputs use VPACKUSDW, a saturating pack: it matches libaom's
	// uint16 truncation only while every prep value lies in [0, 65535].
	for kx := range mc8KindCount {
		for ky := range mc8KindCount {
			if mc8EightTap[kx] && mc8EightTap[ky] && !mc8PrepRangeOK(tables[kx], tables[ky]) {
				mc8EightTap[kx], mc8EightTap[ky] = false, false
			}
			if !mc8SixTap[kx] || !mc8SixTap[ky] {
				continue
			}
			for sx := range 16 {
				mMin, mMax := mc8TapRange(&tables[kx][sx], 0, 255)
				mMin, mMax = (mMin+2)>>2, (mMax+2)>>2
				for sy := range 16 {
					lo, hi := mc8TapRange(&tables[ky][sy], mMin, mMax)
					if (lo+32+6144<<6)>>6 < 0 || (hi+32+6144<<6)>>6 > 65535 {
						mc8SixTap[kx], mc8SixTap[ky] = false, false
						mc8FourTap[kx], mc8FourTap[ky] = false, false
					}
				}
			}
		}
	}
}

// mc8SumsExact reports whether every VPMADDUBSW pair of the halved taps h and
// their int16 sums stay exact for put (+34) and prep (+2 + 6144 << 2,
// unsigned) over byte samples.
func mc8SumsExact(h []int) bool {
	sMin, sMax := 0, 0
	for i := 0; i < len(h); i += 2 {
		pMin, pMax := 0, 0
		for _, c := range h[i : i+2] {
			if c < 0 {
				pMin += 255 * c
			} else {
				pMax += 255 * c
			}
		}
		if pMin < -32768 || pMax > 32767 {
			return false
		}
		sMin += pMin
		sMax += pMax
	}
	return sMin >= -32768 && sMax+34 <= 32767 && sMin+2+6144<<2 >= 0 && sMax+2+6144<<2 <= 65535
}

// mc8PrepRangeOK reports whether every 2D prep value of the two kinds lies in
// [0, 65535], where VPACKUSDW equals libaom's uint16 cast.
func mc8PrepRangeOK(tx, ty *[16][filterTaps]int16) bool {
	for sx := range 16 {
		mMin, mMax := mc8TapRange(&tx[sx], 0, 255)
		mMin, mMax = (mMin+2)>>2, (mMax+2)>>2
		for sy := range 16 {
			lo, hi := mc8TapRange(&ty[sy], mMin, mMax)
			if (lo+32+6144<<6)>>6 < 0 || (hi+32+6144<<6)>>6 > 65535 {
				return false
			}
		}
	}
	return true
}

// mc8TapRange bounds sum(half tap * v) for v in [lo, hi].
func mc8TapRange(k *[filterTaps]int16, lo, hi int) (int, int) {
	sMin, sMax := 0, 0
	for _, c := range k {
		h := int(c) / 2
		if h < 0 {
			sMin += h * hi
			sMax += h * lo
		} else {
			sMin += h * lo
			sMax += h * hi
		}
	}
	return sMin, sMax
}

// mc8Kind mirrors interpKernel's table selection.
func mc8Kind(f InterpFilter, blockSize int) int {
	if blockSize <= 4 && f != InterpBilinear {
		if f == InterpEightTapSmooth {
			return mc8KindSmooth4
		}
		return mc8KindRegular4
	}
	switch f {
	case InterpEightTapRegular:
		return mc8KindRegular
	case InterpEightTapSmooth:
		return mc8KindSmooth
	case InterpMultiTapSharp:
		return mc8KindSharp
	default:
		return mc8KindBilinear
	}
}

// mc8Source resolves the six-tap source window of a w x h block whose first
// output sample reads ref at (refX, refY); halo selects the filtered axes.
// When any tap falls outside the plane the clamped window is materialized in
// edge first (dav1d src/recon_tmpl.c mc() emu_edge). Every kernel reads at
// most max(w+8, 16) bytes per row from the window's left edge; the returned
// window is exactly bh rows of that footprint, so the kernels' checked row
// loads stay inside it (edge holds the largest one, 135 rows of 136 bytes).
func mc8Source(ref *frame.Plane, refX, refY, w, h, taps int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src []byte, stride int, ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= taps/2 - 1
		bw += taps - 1
	}
	if hy {
		y0 -= taps/2 - 1
		bh += taps - 1
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < ref.Width || ref.Stride > mc8MaxStride {
		return nil, 0, false
	}
	need := max(w+8, 16)
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		if end := start + (bh-1)*ref.Stride + need; end <= len(ref.Pix) {
			return ref.Pix[start:end], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return nil, 0, false
	}
	end := (bh-1)*emuEdgeStride + need
	if end > len(edge) {
		return nil, 0, false
	}
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	return edge[:end], emuEdgeStride, true
}

// mc8MaxStride bounds the row strides the kernels take. With it and the block
// size bounds (mc8Dims), every row offset r*stride + c is known not to
// overflow, so each checked row window costs one compare.
const mc8MaxStride = 1 << 20

// mc8Dims panics unless w, h and the strides ss and ds lie in the ranges the
// drivers admit (w <= maxBlockSize, h at most the intermediate height).
func mc8Dims(w, h, ss, ds int) {
	if uint(w) > maxBlockSize || uint(h) > 2*maxBlockSize || uint(ss) > mc8MaxStride || uint(ds) > mc8MaxStride {
		panic("motion: mc8 kernel dimensions out of range")
	}
}

// mc8IM is the 2D intermediate: ConvolveScratch.im and
// CompoundConvolveScratch.im8 share this shape.
type mc8IM = [compoundIMMaxSamples]int16

// mc8Taps picks the kernel family for a block: the six-tap kernels (four-tap
// horizontally at w=4), or the eight-tap kernels on both axes when a filtered
// axis is sharp (dav1d put_8tap), which w=4 leaves to the general path.
func mc8Taps(w, kx, ky, subX, subY int) (eight, ok bool) {
	if (subX == 0 || mc8SixTap[kx]) && (subY == 0 || mc8SixTap[ky]) {
		return false, w != 4 || subX == 0 || mc8FourTap[kx]
	}
	return true, w != 4 && (subX == 0 || mc8EightTap[kx]) && (subY == 0 || mc8EightTap[ky])
}

// Put8 is dav1d's mc() put for an unscaled 8-bit single-reference block:
// the copy (integer vector) or the put kernels for the block's filters, from
// the reference origin (refX, refY) and Q4 subpel offsets (subX, subY). It
// returns false, with nothing written, for shapes it does not cover; the
// caller then runs the general path.
func Put8(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 {
		return mc8CopyFast(dst, ref, dstX, dstY, refX, refY, w, h)
	}
	if !mc8AVX2 || (w&7 != 0 && w != 4) || w <= 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	eight, ok := mc8Taps(w, kx, ky, subX, subY)
	if !ok {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width || dst.Stride > mc8MaxStride {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	dEnd := dOff + (h-1)*dst.Stride + w
	if dEnd > len(dst.Pix) {
		return false
	}
	taps := 6
	if eight {
		taps = 8
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, taps, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	d := dst.Pix[dOff:dEnd]
	if eight {
		switch {
		case subX != 0 && subY != 0:
			mc8AVX2H8Mid(src, stride, w, h+7, &mc8Phases[kx][subX], &scratch.im)
			mc8AVX2PutV8Mid(&scratch.im, w, h, &mc8Phases[ky][subY], d, dst.Stride)
		case subX != 0:
			mc8AVX2PutH8(src, stride, w, h, &mc8Phases[kx][subX], d, dst.Stride)
		default:
			mc8AVX2PutV8(src, stride, w, h, &mc8Phases[ky][subY], d, dst.Stride)
		}
		archsimd.ClearAVXUpperBits()
		return true
	}
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			mc8AVX2MidH4W4(src, stride, h+5, &mc8Phases[kx][subX], &scratch.im)
			mc8AVX2PutV6MidW4(&scratch.im, h, &mc8Phases[ky][subY], d, dst.Stride)
		case subX != 0:
			mc8AVX2PutH4W4(src, stride, h, &mc8Phases[kx][subX], d, dst.Stride)
		default:
			mc8AVX2PutV6W4(src, stride, h, &mc8Phases[ky][subY], &scratch.im, d, dst.Stride)
		}
		archsimd.ClearAVXUpperBits()
		return true
	}
	switch {
	case subX != 0 && subY != 0:
		im := &scratch.im
		mc8AVX2H6Mid(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8AVX2PutV6Mid(im, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	case subX != 0:
		mc8AVX2PutH6(src, stride, w, h, &mc8Phases[kx][subX], d, dst.Stride)
	default:
		mc8AVX2PutV6(src, stride, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the AVX2 prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if !mc8AVX2 || (w&7 != 0 && w != 4) || w <= 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	eight, ok := mc8Taps(w, kx, ky, subX, subY)
	if !ok {
		return false
	}
	taps := 6
	if eight {
		taps = 8
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, taps, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	o := out[:w*h]
	im := (*mc8IM)(&scratch.im8)
	if subX|subY == 0 {
		mc8AVX2PrepCopy(src, stride, w, h, o)
		archsimd.ClearAVXUpperBits()
		return true
	}
	if eight {
		switch {
		case subX != 0 && subY != 0:
			mc8AVX2H8Mid(src, stride, w, h+7, &mc8Phases[kx][subX], im)
			mc8AVX2PrepV8Mid(im, w, h, &mc8Phases[ky][subY], o)
		case subX != 0:
			mc8AVX2PrepH8(src, stride, w, h, &mc8Phases[kx][subX], o)
		default:
			mc8AVX2PrepV8(src, stride, w, h, &mc8Phases[ky][subY], o)
		}
		archsimd.ClearAVXUpperBits()
		return true
	}
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			mc8AVX2MidH4W4(src, stride, h+5, &mc8Phases[kx][subX], im)
			mc8AVX2PrepV6MidW4(im, h, &mc8Phases[ky][subY], o)
		case subX != 0:
			mc8AVX2PrepH4W4(src, stride, h, &mc8Phases[kx][subX], o)
		default:
			mc8AVX2PrepV6W4(src, stride, h, &mc8Phases[ky][subY], im, o)
		}
		archsimd.ClearAVXUpperBits()
		return true
	}
	switch {
	case subX != 0 && subY != 0:
		mc8AVX2H6Mid(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8AVX2PrepV6Mid(im, w, h, &mc8Phases[ky][subY], o)
	case subX != 0:
		mc8AVX2PrepH6(src, stride, w, h, &mc8Phases[kx][subX], o)
	default:
		mc8AVX2PrepV6(src, stride, w, h, &mc8Phases[ky][subY], o)
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// VPSHUFB indices of the three horizontal tap pairs: output column i reads the
// byte pairs (i, i+1), (i+2, i+3) and (i+4, i+5) of its lane.
var (
	mc8ShufA = [32]int8{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8}
	mc8ShufB = [32]int8{2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10}
	mc8ShufC = [32]int8{4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12}
	mc8ShufD = [32]int8{6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14}
	// mc8Deint gathers the even bytes of each lane, then the odd bytes.
	mc8Deint = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}
	// mc8Q3120 is VPERMQ q3120 as dword indices.
	mc8Q3120 = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}
)

func mc8TapsB(t int16) archsimd.Int8x32 {
	return archsimd.BroadcastInt16x16(t).ToBits().ReshapeToUint8s().BitsToInt8()
}

func mc8TapsD(t int32) archsimd.Int16x16 {
	return archsimd.BroadcastInt32x8(t).ToBits().ReshapeToUint16s().BitsToInt16()
}

// Row windows. Offsets are in elements of the slice; the kernels bound their
// dimensions and strides first (mc8Dims), so each checked window below is one
// compare.

func mc8W16(s []byte, o int) *[16]byte {
	return (*[16]byte)(s[uint(o) : uint(o)+16])
}

// mc8W24 is the 24-byte window holding the 16-byte loads at o and o+8.
func mc8W24(s []byte, o int) *[24]byte {
	return (*[24]byte)(s[uint(o) : uint(o)+24])
}

func mc8Lo16(x *[24]byte) *[16]byte { return (*[16]byte)(x[:16]) }

func mc8Hi16(x *[24]byte) *[16]byte { return (*[16]byte)(x[8:]) }

// mc8Ld2 loads p into the low lane and q into the high lane.
func mc8Ld2(p, q *[16]byte) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	return z.SetLo(archsimd.LoadUint8x16Array(p)).SetHi(archsimd.LoadUint8x16Array(q))
}

// mc8Ld2At loads the 16 bytes at s[p:] into the low lane and at s[q:] into
// the high lane.
func mc8Ld2At(s []byte, p, q int) archsimd.Uint8x32 {
	return mc8Ld2(mc8W16(s, p), mc8W16(s, q))
}

func mc8Ld64(s []byte, o int) uint64 {
	return binary.LittleEndian.Uint64(s[uint(o) : uint(o)+8])
}

func mc8Ld32(s []byte, o int) uint32 {
	return binary.LittleEndian.Uint32(s[uint(o) : uint(o)+4])
}

func mc8St16(d []byte, o int, v archsimd.Uint8x16) {
	v.StoreArray((*[16]uint8)(d[uint(o) : uint(o)+16]))
}

func mc8St8(d []byte, o int, v archsimd.Uint8x16) {
	mc8St64(d, o, v.ReshapeToUint64s().GetElem(0))
}

func mc8St64(d []byte, o int, x uint64) {
	binary.LittleEndian.PutUint64(d[uint(o):uint(o)+8], x)
}

func mc8St4(d []byte, o int, x uint32) {
	binary.LittleEndian.PutUint32(d[uint(o):uint(o)+4], x)
}

func mc8StU16(out []uint16, o int, v archsimd.Uint16x16) {
	v.StoreArray((*[16]uint16)(out[uint(o) : uint(o)+16]))
}

func mc8StU16x8(out []uint16, o int, v archsimd.Uint16x8) {
	v.StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
}

// mc8StU16x4 stores the four words of x at out[o:].
func mc8StU16x4(out []uint16, o int, x uint64) {
	p := (*[4]uint16)(out[uint(o) : uint(o)+4])
	p[0], p[1], p[2], p[3] = uint16(x), uint16(x>>16), uint16(x>>32), uint16(x>>48)
}

func mc8LdI16(im *mc8IM, o int) archsimd.Int16x16 {
	return archsimd.LoadInt16x16Array((*[16]int16)(im[uint(o) : uint(o)+16]))
}

func mc8StI16(im *mc8IM, o int, v archsimd.Int16x16) {
	v.StoreArray((*[16]int16)(im[uint(o) : uint(o)+16]))
}

func mc8StI16x8(im *mc8IM, o int, v archsimd.Int16x8) {
	v.StoreArray((*[8]int16)(im[uint(o) : uint(o)+8]))
}

// mc8H6 is the six-tap halved horizontal sum of the eight output columns of
// each lane (PUT_6TAP_H).
func mc8H6(y archsimd.Uint8x32, sA, sB, sC, t0, t1, t2 archsimd.Int8x32) archsimd.Int16x16 {
	a := y.PermuteOrZeroGrouped(sA).DotProductPairsSaturated(t0)
	b := y.PermuteOrZeroGrouped(sB).DotProductPairsSaturated(t1)
	c := y.PermuteOrZeroGrouped(sC).DotProductPairsSaturated(t2)
	return a.Add(b).Add(c)
}

// mc8Pack is VPACKUSWB: per lane, the clamped bytes of a then of b.
func mc8Pack(a, b archsimd.Int16x16, zero, c255 archsimd.Int16x16, deint archsimd.Int8x32) archsimd.Uint8x32 {
	a = a.Max(zero).Min(c255)
	b = b.Max(zero).Min(c255)
	return a.Or(b.ShiftAllLeft(8)).ToBits().ReshapeToUint8s().PermuteOrZeroGrouped(deint)
}

// mc8AVX2PutH6 is the put horizontal-only pass: clip((s + 34) >> 6).
func mc8AVX2PutH6(src []byte, ss, w, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(w, h, ss, ds)
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	for ur := uint(0); ur < uint(h); ur += 2 {
		r := int(ur)
		r1 := min(r+1, h-1) // the last row of an odd h pairs with itself
		s0, d0 := r*ss, r*ds
		s1, d1 := r1*ss, r1*ds
		if w == 8 {
			a := mc8H6(mc8Ld2At(src, s0, s1), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint)
			mc8St8(dst, d0, v.GetLo())
			mc8St8(dst, d1, v.GetHi())
			continue
		}
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x0, x1 := mc8W24(src, s0+c), mc8W24(src, s1+c)
			a := mc8H6(mc8Ld2(mc8Lo16(x0), mc8Lo16(x1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			b := mc8H6(mc8Ld2(mc8Hi16(x0), mc8Hi16(x1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint)
			mc8St16(dst, d1+c, v.GetHi())
			mc8St16(dst, d0+c, v.GetLo())
		}
		if c := w &^ 15; c < w {
			a := mc8H6(mc8Ld2At(src, s0+c, s1+c), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint)
			mc8St8(dst, d1+c, v.GetHi())
			mc8St8(dst, d0+c, v.GetLo())
		}
	}
}

// mc8AVX2PrepH6 is the prep horizontal-only pass into int16 rows of stride w:
// ((s + 2 + 6144 << 2) >> 2) as unsigned.
func mc8AVX2PrepH6(src []byte, ss, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, ss, 0)
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		for ur := uint(0); ur < uint(h&^1); ur += 2 {
			r := int(ur)
			s0 := r * ss
			a := mc8H6(mc8Ld2At(src, s0, s0+ss), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
		}
		r := h &^ 1
		if r < h {
			s0 := r * ss
			a := mc8H6(mc8Ld2At(src, s0, s0), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	for r := 0; r < h; r++ {
		s0, o := r*ss, r*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x := mc8W24(src, s0+c)
			a := mc8H6(mc8Ld2(mc8Lo16(x), mc8Hi16(x)), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, o+c, a.ToBits().ShiftAllRight(2))
		}
		if c := w &^ 15; c < w {
			x := mc8W16(src, s0+c)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, o+c, a.ToBits().ShiftAllRight(2).GetLo())
		}
	}
}

// mc8AVX2H6Mid writes rows x w rounded horizontal sums m = (s + 2) >> 2 as
// int16 with row stride w (the 2D intermediate).
func mc8AVX2H6Mid(src []byte, ss, w, rows int, ph *mc8Phase, im *mc8IM) {
	mc8Dims(w, rows, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2)
	if w == 8 {
		for ur := uint(0); ur < uint(rows&^1); ur += 2 {
			r := int(ur)
			s0 := r * ss
			a := mc8H6(mc8Ld2At(src, s0, s0+ss), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*8, a)
		}
		r := rows &^ 1
		if r < rows {
			s0 := r * ss
			a := mc8H6(mc8Ld2At(src, s0, s0), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*8, a.GetLo())
		}
		return
	}
	for r := 0; r < rows; r++ {
		s0, o := r*ss, r*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x := mc8W24(src, s0+c)
			a := mc8H6(mc8Ld2(mc8Lo16(x), mc8Hi16(x)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, o+c, a)
		}
		if c := w &^ 15; c < w {
			x := mc8W16(src, s0+c)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, o+c, a.GetLo())
		}
	}
}

// mc8Zx zero-extends the 16 bytes at s[o:] to words.
func mc8Zx(s []byte, o int) archsimd.Uint16x16 {
	return archsimd.LoadUint8x16Array(mc8W16(s, o)).ExtendToUint16()
}

// mc8Pair puts row a in the low and row b in the high byte of each word: the
// (a, b) byte pairs VPMADDUBSW multiplies by a vertical tap pair.
func mc8Pair(a, b archsimd.Uint16x16) archsimd.Uint8x32 {
	return a.Or(b.ShiftAllLeft(8)).ReshapeToUint8s()
}

func mc8V6(p0, p2, p4 archsimd.Uint8x32, t0, t1, t2 archsimd.Int8x32) archsimd.Int16x16 {
	return p0.DotProductPairsSaturated(t0).Add(p2.DotProductPairsSaturated(t1)).Add(p4.DotProductPairsSaturated(t2))
}

// mc8AVX2PutV6 is the put vertical-only pass: clip((s + 32) >> 6).
//
// The vertical kernels wider than 8 run 16-column strips; when w%16 == 8 the
// last strip starts at w-16 and rewrites eight columns of the previous one
// with the same values, so every strip stores 16 columns per row.
func mc8AVX2PutV6(src []byte, ss, w, h int, ph *mc8Phase, dst []byte, ds int) {
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	if w == 8 {
		mc8AVX2PutV6W8(src, ss, h, t0, t1, t2, dst, ds)
		return
	}
	mc8Dims(w, h, ss, ds)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		mc8AVX2PutV6Strip(src[c:], ss, h, ph, dst[c:], ds)
	}
}

// mc8AVX2PutV6Strip is one 16-column strip of mc8AVX2PutV6. Its own frame
// keeps the row loop's integer state (two windows, two strides, the row) in
// registers; each row pair loads and stores through one two-row window.
func mc8AVX2PutV6Strip(sc []byte, ss, h int, ph *mc8Phase, dc []byte, ds int) {
	mc8Dims(16, h, ss, ds)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := mc8Zx(sc, 0)
	z1 := mc8Zx(sc, ss)
	z2 := mc8Zx(sc, 2*ss)
	z3 := mc8Zx(sc, 3*ss)
	z4 := mc8Zx(sc, 4*ss)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	for ur := uint(0); ur < uint(h&^1); ur += 2 {
		r := int(ur)
		x5, x6 := mc8Rows2(sc, (r+5)*ss, ss)
		z5 := archsimd.LoadUint8x16Array(x5).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(x6).ExtendToUint16()
		p4 := mc8Pair(z4, z5)
		p5 := mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, b, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
		d0, d1 := mc8Rows2(dc, r*ds, ds)
		v.GetLo().StoreArray(d0)
		v.GetHi().StoreArray(d1)
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r := h &^ 1; r < h {
		z5 := mc8Zx(sc, (r+5)*ss)
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
		mc8St16(dc, r*ds, v.GetLo())
	}
}

// mc8Rows2 returns the 16-byte windows at s[o:] and s[o+ss:] through one
// checked slice (both rows must exist).
func mc8Rows2(s []byte, o, ss int) (*[16]byte, *[16]byte) {
	win := s[uint(o) : uint(o)+uint(ss)+16]
	return (*[16]byte)(win), (*[16]byte)(win[ss:])
}

// The eight-column vertical helpers stay in 256-bit registers (a 128-bit
// value carried across iterations makes the compiler emit legacy-SSE MOVUPS
// register copies, which stall on x86 while the upper halves are dirty).

// mc8Join puts the low lane of a in the low and the low lane of b in the high
// lane (VPERM2I128).
func mc8Join(a, b archsimd.Uint8x32) archsimd.Uint8x32 {
	return a.ConcatPermute128Scalars(0, 2, b)
}

// mc8AVX2PutV6W8 runs the eight-column vertical pass two rows per vector (row
// r in the low lane, r+1 in the high lane).
func mc8AVX2PutV6W8(src []byte, ss, h int, t0, t1, t2 archsimd.Int8x32, dst []byte, ds int) {
	mc8Dims(8, h, ss, ds)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := mc8Zx(src, 0)
	z1 := mc8Zx(src, ss)
	z2 := mc8Zx(src, 2*ss)
	z3 := mc8Zx(src, 3*ss)
	z4 := mc8Zx(src, 4*ss)
	p2 := mc8Pair(z2, z3)
	q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
	q2 := mc8Join(p2, mc8Pair(z3, z4))
	for ur := uint(0); ur < uint(h); ur += 2 {
		r := int(ur)
		s, d := (r+5)*ss, r*ds
		z5 := mc8Zx(src, s)
		z6 := z5
		if r+1 < h {
			z6 = mc8Zx(src, s+ss)
		}
		p4 := mc8Pair(z4, z5)
		q4 := mc8Join(p4, mc8Pair(z5, z6))
		a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		mc8St8(dst, d, v.GetLo())
		if r+1 < h {
			mc8St8(dst, d+ds, v.GetHi())
		}
		q0, q2, z4 = q2, q4, z6
	}
}

// mc8AVX2PrepV6 is the prep vertical-only pass: ((s + 2) >> 2) + 6144.
func mc8AVX2PrepV6(src []byte, ss, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, ss, 0)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		z0 := mc8Zx(src, 0)
		z1 := mc8Zx(src, ss)
		z2 := mc8Zx(src, 2*ss)
		z3 := mc8Zx(src, 3*ss)
		z4 := mc8Zx(src, 4*ss)
		q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
		q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
		for ur := uint(0); ur < uint(h&^1); ur += 2 {
			r := int(ur)
			x5, x6 := mc8Rows2(src, (r+5)*ss, ss)
			z5 := archsimd.LoadUint8x16Array(x5).ExtendToUint16()
			z6 := archsimd.LoadUint8x16Array(x6).ExtendToUint16()
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
			q0, q2, z4 = q2, q4, z6
		}
		r := h &^ 1
		if r < h {
			z5 := mc8Zx(src, (r+5)*ss)
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z5))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		mc8AVX2PrepV6Strip(src[c:], ss, w, h, ph, out[c:])
	}
}

// mc8AVX2PrepV6Strip is one 16-column strip of mc8AVX2PrepV6 (out row
// stride w), two rows per step through two-row windows, in its own frame so
// the row loop's integer state stays in registers.
func mc8AVX2PrepV6Strip(sc []byte, ss, w, h int, ph *mc8Phase, oc []uint16) {
	mc8Dims(w, h, ss, 0)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	z0 := mc8Zx(sc, 0)
	z1 := mc8Zx(sc, ss)
	z2 := mc8Zx(sc, 2*ss)
	z3 := mc8Zx(sc, 3*ss)
	z4 := mc8Zx(sc, 4*ss)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	for ur := uint(0); ur < uint(h&^1); ur += 2 {
		r := int(ur)
		x5, x6 := mc8Rows2(sc, (r+5)*ss, ss)
		z5 := archsimd.LoadUint8x16Array(x5).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(x6).ExtendToUint16()
		p4, p5 := mc8Pair(z4, z5), mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		win := oc[uint(r*w) : uint(r*w)+uint(w)+16]
		a.StoreArray((*[16]uint16)(win))
		b.StoreArray((*[16]uint16)(win[w:]))
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r := h &^ 1; r < h {
		z5 := mc8Zx(sc, (r+5)*ss)
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		mc8StU16(oc, r*w, a)
	}
}

// mc8VMid is the six-tap vertical VPMADDWD sum over word-interleaved row
// pairs, plus rnd.
func mc8VMid(p0, p2, p4, d0, d1, d2 archsimd.Int16x16, rnd archsimd.Int32x8) archsimd.Int32x8 {
	return p0.DotProductPairs(d0).Add(rnd).Add(p2.DotProductPairs(d1)).Add(p4.DotProductPairs(d2))
}

// mc8AVX2PutV6Mid runs the put 2D vertical pass over the int16 intermediate
// (row stride w): clip((sum(half tap * m) + 512) >> 10).
//
// For w == 8 a 32-byte load at row k holds rows k and k+1, so the per-lane
// word interleave of rows (k, k+1) gives row pairs (k, k+1) in the low lane and
// (k+1, k+2) in the high lane: output rows r and r+1 come out one per lane.
// Wider blocks run 16-column strips one row at a time, where the per-lane
// interleave and VPACKUSDW undo each other's lane split.
func mc8AVX2PutV6Mid(im *mc8IM, w, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(w, h, 0, ds)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	rnd := archsimd.BroadcastInt32x8(512)
	c255 := archsimd.BroadcastUint16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	if w == 8 {
		l0 := mc8LdI16(im, 0)
		l1 := mc8LdI16(im, 8)
		l2 := mc8LdI16(im, 16)
		l3 := mc8LdI16(im, 24)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			q, d := 32+r*8, r*ds
			l4 := mc8LdI16(im, q)
			l5 := mc8LdI16(im, q+8)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			lo := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(10)
			hi := mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(10)
			u := lo.SaturateToUint16ConcatGrouped(hi).Min(c255)
			v := u.ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			mc8St8(dst, d, v.GetLo())
			if r+1 < h {
				mc8St8(dst, d+ds, v.GetHi())
			}
			a0, b0, a2, b2 = a2, b2, a4, b4
		}
		return
	}
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		dc := dst[c:]
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			q, d := c+(r+5)*w, r*ds
			l5 := mc8LdI16(im, q)
			l6 := l5
			if r+1 < h {
				l6 = mc8LdI16(im, q+w)
			}
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			a5, b5 := l5.InterleaveLoGrouped(l6), l5.InterleaveHiGrouped(l6)
			x := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(10).SaturateToUint16ConcatGrouped(mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(10)).Min(c255)
			y := mc8VMid(a1, a3, a5, d0, d1, d2, rnd).ShiftAllRight(10).SaturateToUint16ConcatGrouped(mc8VMid(b1, b3, b5, d0, d1, d2, rnd).ShiftAllRight(10)).Min(c255)
			// x and y are rows r and r+1 in natural order, each word < 256.
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			mc8St16(dc, d, v.GetLo())
			if r+1 < h {
				mc8St16(dc, d+ds, v.GetHi())
			}
			a0, b0, a1, b1, a2, b2, a3, b3, l4 = a2, b2, a3, b3, a4, b4, a5, b5, l6
		}
	}
}

// mc8AVX2PrepV6Mid is the prep 2D vertical pass:
// (sum(half tap * m) + 32 + 6144 << 6) >> 6.
func mc8AVX2PrepV6Mid(im *mc8IM, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, 0, 0)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	rnd := archsimd.BroadcastInt32x8(32 + 6144<<6)
	if w == 8 {
		l0 := mc8LdI16(im, 0)
		l1 := mc8LdI16(im, 8)
		l2 := mc8LdI16(im, 16)
		l3 := mc8LdI16(im, 24)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		for ur := uint(0); ur < uint(h&^1); ur += 2 {
			r := int(ur)
			q := 32 + r*8
			l4 := mc8LdI16(im, q)
			l5 := mc8LdI16(im, q+8)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			lo := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(6)
			hi := mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(6)
			mc8StU16(out, r*8, lo.SaturateToUint16ConcatGrouped(hi))
			a0, b0, a2, b2 = a2, b2, a4, b4
		}
		r := h &^ 1
		if r < h {
			q := 32 + r*8
			l4 := mc8LdI16(im, q)
			l5 := mc8LdI16(im, q+8)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			lo := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(6)
			hi := mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(6)
			mc8StU16x8(out, r*8, lo.SaturateToUint16ConcatGrouped(hi).GetLo())
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		oc := out[c:]
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		for r := 0; r < h; r++ {
			l5 := mc8LdI16(im, c+(r+5)*w)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			x := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(6).SaturateToUint16ConcatGrouped(mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(6))
			o := r * w
			mc8StU16(oc, o, x)
			a0, b0, a1, b1, a2, b2, a3, b3, l4 = a1, b1, a2, b2, a3, b3, a4, b4, l5
		}
	}
}

// mc8BlendAvgFast is the 8-bit plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (dav1d avg_8bpc):
// clip(round4(((a*8 + b*8) >> 4) - 6144)) is (a + b - 12272) >> 5 by nested
// floor division, in int16 lanes (8-bit CONV_BUF values stay below 2^14).
func mc8BlendAvgFast(dst *frame.Plane, a, b []uint16, dstX, dstY, w, h int) bool {
	if !mc8AVX2 || w <= 0 || (w != 4 && w&7 != 0) || w > maxBlockSize || h <= 0 || h > maxBlockSize || len(a) < w*h || len(b) < w*h || dst.Stride > mc8MaxStride {
		return false
	}
	off := dstY*dst.Stride + dstX
	end := off + (h-1)*dst.Stride + w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || end > len(dst.Pix) {
		return false
	}
	mc8AVX2BlendAvg(dst.Pix[off:end], dst.Stride, a[:w*h], b[:w*h], w, h)
	archsimd.ClearAVXUpperBits()
	return true
}

func mc8AvgSum(a, b []uint16, o int, bias archsimd.Int16x16) archsimd.Int16x16 {
	x := archsimd.LoadUint16x16Array((*[16]uint16)(a[uint(o) : uint(o)+16]))
	y := archsimd.LoadUint16x16Array((*[16]uint16)(b[uint(o) : uint(o)+16]))
	return x.Add(y).BitsToInt16().Add(bias).ShiftAllRight(5)
}

// mc8AVX2BlendAvg writes the averages; a and b hold exactly w*h samples.
func mc8AVX2BlendAvg(dst []byte, ds int, a, b []uint16, w, h int) {
	mc8Dims(w, h, 0, ds)
	bias := archsimd.BroadcastInt16x16(-12272)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	switch w {
	case 4:
		// Four rows per vector; the last group may be partial.
		for ur := uint(0); ur < uint(h&^3); ur += 4 {
			r := int(ur)
			x := mc8AvgSum(a, b, r*4, bias)
			v := mc8Pack(x, x, zero, c255, deint).ReshapeToUint32s()
			lo, hi := v.GetLo(), v.GetHi()
			d := r * ds
			mc8St4(dst, d, lo.GetElem(0))
			mc8St4(dst, d+ds, lo.GetElem(1))
			mc8St4(dst, d+2*ds, hi.GetElem(0))
			mc8St4(dst, d+3*ds, hi.GetElem(1))
		}
		r := h &^ 3
		for ; r < h; r++ {
			d := r * ds
			for c := range 4 {
				v := (int(a[r*4+c]) + int(b[r*4+c]) - 12272) >> 5
				dst[d+c] = byte(min(max(v, 0), 255))
			}
		}
	case 8:
		for ur := uint(0); ur < uint(h&^3); ur += 4 {
			r := int(ur)
			x := mc8AvgSum(a, b, r*8, bias)
			y := mc8AvgSum(a, b, r*8+16, bias)
			// lanes: [r, r+2 | r+1, r+3]
			v := mc8Pack(x, y, zero, c255, deint).ReshapeToUint64s()
			lo, hi := v.GetLo(), v.GetHi()
			d := r * ds
			mc8St64(dst, d, lo.GetElem(0))
			mc8St64(dst, d+ds, hi.GetElem(0))
			mc8St64(dst, d+2*ds, lo.GetElem(1))
			mc8St64(dst, d+3*ds, hi.GetElem(1))
		}
		r := h &^ 3
		for ; r+2 <= h; r += 2 {
			x := mc8AvgSum(a, b, r*8, bias)
			v := mc8Pack(x, x, zero, c255, deint).ReshapeToUint64s()
			d := r * ds
			mc8St64(dst, d, v.GetLo().GetElem(0))
			mc8St64(dst, d+ds, v.GetHi().GetElem(0))
		}
		if r < h {
			d := r * ds
			for c := range 8 {
				v := (int(a[r*8+c]) + int(b[r*8+c]) - 12272) >> 5
				dst[d+c] = byte(min(max(v, 0), 255))
			}
		}
	case 16:
		for ur := uint(0); ur < uint(h&^1); ur += 2 {
			r := int(ur)
			x := mc8AvgSum(a, b, r*16, bias)
			y := mc8AvgSum(a, b, r*16+16, bias)
			// lanes: [r c0-7, r+1 c0-7 | r c8-15, r+1 c8-15]
			q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
			v := mc8Pack(x, y, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ds
			mc8St16(dst, d, v.GetLo())
			mc8St16(dst, d+ds, v.GetHi())
		}
		r := h &^ 1
		if r < h {
			x := mc8AvgSum(a, b, r*16, bias)
			q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
			v := mc8Pack(x, x, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			mc8St16(dst, r*ds, v.GetLo())
		}
	default:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for r := 0; r < h; r++ {
			pa := a[r*w : r*w+w]
			pb := b[r*w : r*w+w]
			d := r * ds
			for uc := uint(0); uc < uint(w&^31); uc += 32 {
				c := int(uc)
				x := mc8AvgSum(pa, pb, c, bias)
				y := mc8AvgSum(pa, pb, c+16, bias)
				v := mc8Pack(x, y, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(dst[uint(d+c) : uint(d+c)+32]))
			}
			for c := w &^ 31; c < w; c += 8 {
				for k := range 8 {
					v := (int(pa[c+k]) + int(pb[c+k]) - 12272) >> 5
					dst[d+c+k] = byte(min(max(v, 0), 255))
				}
			}
		}
	}
}

// mc8AVX2PutH8 is mc8AVX2PutH6 with the eight-tap window (dav1d put_8tap/prep_8tap).
func mc8AVX2PutH8(src []byte, ss, w, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(w, h, ss, ds)
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	sD := archsimd.LoadInt8x32Array(&mc8ShufD)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1, t2, t3 := mc8TapsB(ph.b8[0]), mc8TapsB(ph.b8[1]), mc8TapsB(ph.b8[2]), mc8TapsB(ph.b8[3])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	for ur := uint(0); ur < uint(h); ur += 2 {
		r := int(ur)
		r1 := min(r+1, h-1) // the last row of an odd h pairs with itself
		s0, d0 := r*ss, r*ds
		s1, d1 := r1*ss, r1*ds
		if w == 8 {
			a := mc8H8(mc8Ld2At(src, s0, s1), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint)
			mc8St8(dst, d0, v.GetLo())
			mc8St8(dst, d1, v.GetHi())
			continue
		}
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x0, x1 := mc8W24(src, s0+c), mc8W24(src, s1+c)
			a := mc8H8(mc8Ld2(mc8Lo16(x0), mc8Lo16(x1)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			b := mc8H8(mc8Ld2(mc8Hi16(x0), mc8Hi16(x1)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint)
			mc8St16(dst, d1+c, v.GetHi())
			mc8St16(dst, d0+c, v.GetLo())
		}
		if c := w &^ 15; c < w {
			a := mc8H8(mc8Ld2At(src, s0+c, s1+c), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint)
			mc8St8(dst, d1+c, v.GetHi())
			mc8St8(dst, d0+c, v.GetLo())
		}
	}
}

// mc8AVX2PrepH8 is mc8AVX2PrepH6 with the eight-tap window (dav1d put_8tap/prep_8tap).
func mc8AVX2PrepH8(src []byte, ss, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, ss, 0)
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	sD := archsimd.LoadInt8x32Array(&mc8ShufD)
	t0, t1, t2, t3 := mc8TapsB(ph.b8[0]), mc8TapsB(ph.b8[1]), mc8TapsB(ph.b8[2]), mc8TapsB(ph.b8[3])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		for ur := uint(0); ur < uint(h&^1); ur += 2 {
			r := int(ur)
			s0 := r * ss
			a := mc8H8(mc8Ld2At(src, s0, s0+ss), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
		}
		r := h &^ 1
		if r < h {
			s0 := r * ss
			a := mc8H8(mc8Ld2At(src, s0, s0), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	for r := 0; r < h; r++ {
		s0, o := r*ss, r*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x := mc8W24(src, s0+c)
			a := mc8H8(mc8Ld2(mc8Lo16(x), mc8Hi16(x)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd)
			mc8StU16(out, o+c, a.ToBits().ShiftAllRight(2))
		}
		if c := w &^ 15; c < w {
			x := mc8W16(src, s0+c)
			a := mc8H8(mc8Ld2(x, x), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd)
			mc8StU16x8(out, o+c, a.ToBits().ShiftAllRight(2).GetLo())
		}
	}
}

// mc8AVX2H8Mid is mc8AVX2H6Mid with the eight-tap window (dav1d put_8tap/prep_8tap).
func mc8AVX2H8Mid(src []byte, ss, w, rows int, ph *mc8Phase, im *mc8IM) {
	mc8Dims(w, rows, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	sD := archsimd.LoadInt8x32Array(&mc8ShufD)
	t0, t1, t2, t3 := mc8TapsB(ph.b8[0]), mc8TapsB(ph.b8[1]), mc8TapsB(ph.b8[2]), mc8TapsB(ph.b8[3])
	rnd := archsimd.BroadcastInt16x16(2)
	if w == 8 {
		for ur := uint(0); ur < uint(rows&^1); ur += 2 {
			r := int(ur)
			s0 := r * ss
			a := mc8H8(mc8Ld2At(src, s0, s0+ss), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*8, a)
		}
		r := rows &^ 1
		if r < rows {
			s0 := r * ss
			a := mc8H8(mc8Ld2At(src, s0, s0), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*8, a.GetLo())
		}
		return
	}
	for r := 0; r < rows; r++ {
		s0, o := r*ss, r*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x := mc8W24(src, s0+c)
			a := mc8H8(mc8Ld2(mc8Lo16(x), mc8Hi16(x)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, o+c, a)
		}
		if c := w &^ 15; c < w {
			x := mc8W16(src, s0+c)
			a := mc8H8(mc8Ld2(x, x), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, o+c, a.GetLo())
		}
	}
}

// mc8H8 is mc8H6 over the eight-tap window (PUT_8TAP_H).
func mc8H8(y archsimd.Uint8x32, sA, sB, sC, sD, t0, t1, t2, t3 archsimd.Int8x32) archsimd.Int16x16 {
	a := y.PermuteOrZeroGrouped(sA).DotProductPairsSaturated(t0)
	b := y.PermuteOrZeroGrouped(sB).DotProductPairsSaturated(t1)
	c := y.PermuteOrZeroGrouped(sC).DotProductPairsSaturated(t2)
	d := y.PermuteOrZeroGrouped(sD).DotProductPairsSaturated(t3)
	return a.Add(b).Add(c.Add(d))
}

func mc8V8(p0, p2, p4, p6 archsimd.Uint8x32, t0, t1, t2, t3 archsimd.Int8x32) archsimd.Int16x16 {
	a := p0.DotProductPairsSaturated(t0).Add(p2.DotProductPairsSaturated(t1))
	return a.Add(p4.DotProductPairsSaturated(t2).Add(p6.DotProductPairsSaturated(t3)))
}

func mc8VMid8(p0, p2, p4, p6, d0, d1, d2, d3 archsimd.Int16x16, rnd archsimd.Int32x8) archsimd.Int32x8 {
	a := p0.DotProductPairs(d0).Add(rnd).Add(p2.DotProductPairs(d1))
	return a.Add(p4.DotProductPairs(d2).Add(p6.DotProductPairs(d3)))
}

// mc8AVX2PutV8 is the eight-tap put vertical-only pass, clip((s + 32) >> 6),
// laid out as mc8AVX2PutV6.
func mc8AVX2PutV8(src []byte, ss, w, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(w, h, ss, ds)
	t0, t1, t2, t3 := mc8TapsB(ph.b8[0]), mc8TapsB(ph.b8[1]), mc8TapsB(ph.b8[2]), mc8TapsB(ph.b8[3])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	if w == 8 {
		z0 := mc8Zx(src, 0)
		z1 := mc8Zx(src, ss)
		z2 := mc8Zx(src, 2*ss)
		z3 := mc8Zx(src, 3*ss)
		z4 := mc8Zx(src, 4*ss)
		z5 := mc8Zx(src, 5*ss)
		z6 := mc8Zx(src, 6*ss)
		q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
		q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			s := (r + 7) * ss
			z7 := mc8Zx(src, s)
			z8 := z7
			if r+1 < h {
				z8 = mc8Zx(src, s+ss)
			}
			q6 := mc8Join(mc8Pair(z6, z7), mc8Pair(z7, z8))
			a := mc8V8(q0, q2, q4, q6, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint)
			d := r * ds
			mc8St8(dst, d, v.GetLo())
			if r+1 < h {
				mc8St8(dst, d+ds, v.GetHi())
			}
			q0, q2, q4, z6 = q2, q4, q6, z8
		}
		return
	}
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		sc, dc := src[c:], dst[c:]
		z0 := mc8Zx(sc, 0)
		z1 := mc8Zx(sc, ss)
		z2 := mc8Zx(sc, 2*ss)
		z3 := mc8Zx(sc, 3*ss)
		z4 := mc8Zx(sc, 4*ss)
		z5 := mc8Zx(sc, 5*ss)
		z6 := mc8Zx(sc, 6*ss)
		p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
		p4, p5 := mc8Pair(z4, z5), mc8Pair(z5, z6)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			two := r+1 < h
			s := (r + 7) * ss
			z7 := mc8Zx(sc, s)
			z8 := z7
			if two {
				z8 = mc8Zx(sc, s+ss)
			}
			p6, p7 := mc8Pair(z6, z7), mc8Pair(z7, z8)
			a := mc8V8(p0, p2, p4, p6, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			b := mc8V8(p1, p3, p5, p7, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ds
			mc8St16(dc, d, v.GetLo())
			if two {
				mc8St16(dc, d+ds, v.GetHi())
			}
			p0, p1, p2, p3, p4, p5, z6 = p2, p3, p4, p5, p6, p7, z8
		}
	}
}

// mc8AVX2PrepV8 is the eight-tap prep vertical-only pass, ((s + 2) >> 2) +
// 6144, laid out as mc8AVX2PutV8.
func mc8AVX2PrepV8(src []byte, ss, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, ss, 0)
	t0, t1, t2, t3 := mc8TapsB(ph.b8[0]), mc8TapsB(ph.b8[1]), mc8TapsB(ph.b8[2]), mc8TapsB(ph.b8[3])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		z0 := mc8Zx(src, 0)
		z1 := mc8Zx(src, ss)
		z2 := mc8Zx(src, 2*ss)
		z3 := mc8Zx(src, 3*ss)
		z4 := mc8Zx(src, 4*ss)
		z5 := mc8Zx(src, 5*ss)
		z6 := mc8Zx(src, 6*ss)
		q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
		q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			s := (r + 7) * ss
			z7 := mc8Zx(src, s)
			z8 := z7
			if r+1 < h {
				z8 = mc8Zx(src, s+ss)
			}
			q6 := mc8Join(mc8Pair(z6, z7), mc8Pair(z7, z8))
			u := mc8V8(q0, q2, q4, q6, t0, t1, t2, t3).Add(rnd).ToBits().ShiftAllRight(2)
			if r+1 < h {
				mc8StU16(out, r*8, u)
			} else {
				mc8StU16x8(out, r*8, u.GetLo())
			}
			q0, q2, q4, z6 = q2, q4, q6, z8
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		sc, oc := src[c:], out[c:]
		z0 := mc8Zx(sc, 0)
		z1 := mc8Zx(sc, ss)
		z2 := mc8Zx(sc, 2*ss)
		z3 := mc8Zx(sc, 3*ss)
		z4 := mc8Zx(sc, 4*ss)
		z5 := mc8Zx(sc, 5*ss)
		z6 := mc8Zx(sc, 6*ss)
		p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
		p4, p5 := mc8Pair(z4, z5), mc8Pair(z5, z6)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			two := r+1 < h
			s := (r + 7) * ss
			z7 := mc8Zx(sc, s)
			z8 := z7
			if two {
				z8 = mc8Zx(sc, s+ss)
			}
			p6, p7 := mc8Pair(z6, z7), mc8Pair(z7, z8)
			x := mc8V8(p0, p2, p4, p6, t0, t1, t2, t3).Add(rnd).ToBits().ShiftAllRight(2)
			y := mc8V8(p1, p3, p5, p7, t0, t1, t2, t3).Add(rnd).ToBits().ShiftAllRight(2)
			o := r * w
			mc8StU16(oc, o, x)
			if two {
				mc8StU16(oc, o+w, y)
			}
			p0, p1, p2, p3, p4, p5, z6 = p2, p3, p4, p5, p6, p7, z8
		}
	}
}

// mc8AVX2PutV8Mid is the eight-tap put 2D vertical pass over the int16
// intermediate (h+7 rows, stride w), laid out as mc8AVX2PutV6Mid.
func mc8AVX2PutV8Mid(im *mc8IM, w, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(w, h, 0, ds)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2, d3 := mc8TapsD(ph.d8[0]), mc8TapsD(ph.d8[1]), mc8TapsD(ph.d8[2]), mc8TapsD(ph.d8[3])
	// clip((S + 512) >> 10) is ((S + 512) >> 6) >> 4 after the
	// unsigned-saturating pack (nested floor division; the >> 6 values fit).
	// Both shifts stay immediates: a variable count makes the compiler emit a
	// legacy-SSE MOVQ into an XMM register.
	rnd := archsimd.BroadcastInt32x8(512)
	c255 := archsimd.BroadcastUint16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	if w == 8 {
		l0 := mc8LdI16(im, 0)
		l1 := mc8LdI16(im, 8)
		l2 := mc8LdI16(im, 16)
		l3 := mc8LdI16(im, 24)
		l4 := mc8LdI16(im, 32)
		l5 := mc8LdI16(im, 40)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			q := 48 + r*8
			l6 := mc8LdI16(im, q)
			l7 := mc8LdI16(im, q+8)
			a6, b6 := l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
			lo := mc8VMid8(a0, a2, a4, a6, d0, d1, d2, d3, rnd).ShiftAllRight(6)
			hi := mc8VMid8(b0, b2, b4, b6, d0, d1, d2, d3, rnd).ShiftAllRight(6)
			v := lo.SaturateToUint16ConcatGrouped(hi).ShiftAllRight(4).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			d := r * ds
			mc8St8(dst, d, v.GetLo())
			if r+1 < h {
				mc8St8(dst, d+ds, v.GetHi())
			}
			a0, b0, a2, b2, a4, b4 = a2, b2, a4, b4, a6, b6
		}
		return
	}
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		dc := dst[c:]
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		l5 := mc8LdI16(im, c+5*w)
		l6 := mc8LdI16(im, c+6*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		a5, b5 := l5.InterleaveLoGrouped(l6), l5.InterleaveHiGrouped(l6)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			two := r+1 < h
			q := c + (r+7)*w
			l7 := mc8LdI16(im, q)
			l8 := l7
			if two {
				l8 = mc8LdI16(im, q+w)
			}
			a6, b6 := l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
			a7, b7 := l7.InterleaveLoGrouped(l8), l7.InterleaveHiGrouped(l8)
			x := mc8VMid8(a0, a2, a4, a6, d0, d1, d2, d3, rnd).ShiftAllRight(6).SaturateToUint16ConcatGrouped(mc8VMid8(b0, b2, b4, b6, d0, d1, d2, d3, rnd).ShiftAllRight(6))
			y := mc8VMid8(a1, a3, a5, a7, d0, d1, d2, d3, rnd).ShiftAllRight(6).SaturateToUint16ConcatGrouped(mc8VMid8(b1, b3, b5, b7, d0, d1, d2, d3, rnd).ShiftAllRight(6))
			x, y = x.ShiftAllRight(4).Min(c255), y.ShiftAllRight(4).Min(c255)
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ds
			mc8St16(dc, d, v.GetLo())
			if two {
				mc8St16(dc, d+ds, v.GetHi())
			}
			a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, l6 = a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, l8
		}
	}
}

// mc8AVX2PrepV8Mid is the eight-tap prep 2D vertical pass,
// (sum(half tap * m) + 32 + 6144 << 6) >> 6, laid out as mc8AVX2PutV8Mid.
func mc8AVX2PrepV8Mid(im *mc8IM, w, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(w, h, 0, 0)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2, d3 := mc8TapsD(ph.d8[0]), mc8TapsD(ph.d8[1]), mc8TapsD(ph.d8[2]), mc8TapsD(ph.d8[3])
	rnd := archsimd.BroadcastInt32x8(32 + 6144<<6)
	if w == 8 {
		l0 := mc8LdI16(im, 0)
		l1 := mc8LdI16(im, 8)
		l2 := mc8LdI16(im, 16)
		l3 := mc8LdI16(im, 24)
		l4 := mc8LdI16(im, 32)
		l5 := mc8LdI16(im, 40)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			q := 48 + r*8
			l6 := mc8LdI16(im, q)
			l7 := mc8LdI16(im, q+8)
			a6, b6 := l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
			lo := mc8VMid8(a0, a2, a4, a6, d0, d1, d2, d3, rnd).ShiftAllRight(6)
			hi := mc8VMid8(b0, b2, b4, b6, d0, d1, d2, d3, rnd).ShiftAllRight(6)
			u := lo.SaturateToUint16ConcatGrouped(hi)
			if r+1 < h {
				mc8StU16(out, r*8, u)
			} else {
				mc8StU16x8(out, r*8, u.GetLo())
			}
			a0, b0, a2, b2, a4, b4 = a2, b2, a4, b4, a6, b6
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		oc := out[c:]
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		l5 := mc8LdI16(im, c+5*w)
		l6 := mc8LdI16(im, c+6*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		a5, b5 := l5.InterleaveLoGrouped(l6), l5.InterleaveHiGrouped(l6)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			two := r+1 < h
			q := c + (r+7)*w
			l7 := mc8LdI16(im, q)
			l8 := l7
			if two {
				l8 = mc8LdI16(im, q+w)
			}
			a6, b6 := l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
			a7, b7 := l7.InterleaveLoGrouped(l8), l7.InterleaveHiGrouped(l8)
			x := mc8VMid8(a0, a2, a4, a6, d0, d1, d2, d3, rnd).ShiftAllRight(6).SaturateToUint16ConcatGrouped(mc8VMid8(b0, b2, b4, b6, d0, d1, d2, d3, rnd).ShiftAllRight(6))
			y := mc8VMid8(a1, a3, a5, a7, d0, d1, d2, d3, rnd).ShiftAllRight(6).SaturateToUint16ConcatGrouped(mc8VMid8(b1, b3, b5, b7, d0, d1, d2, d3, rnd).ShiftAllRight(6))
			o := r * w
			mc8StU16(oc, o, x)
			if two {
				mc8StU16(oc, o+w, y)
			}
			a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, l6 = a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, l8
		}
	}
}

// Width-4 kernels (dav1d put_6tap/prep_6tap .h_w4, .v_w4, .hv_w4). Each
// 256-bit vector holds four rows of four samples: rows r, r+1 in the low lane
// and r+2, r+3 in the high lane.

var (
	mc8Shuf4A = [32]int8{0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12, 0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12}
	mc8Shuf4B = [32]int8{2, 3, 3, 4, 4, 5, 5, 6, 10, 11, 11, 12, 12, 13, 13, 14, 2, 3, 3, 4, 4, 5, 5, 6, 10, 11, 11, 12, 12, 13, 13, 14}
)

// mc8H4x4 is the four-tap horizontal sum of four rows of four outputs; each
// offset addresses the first sample of its row's window (window tap 0), the
// loads start at window tap 1.
func mc8H4x4(s []byte, o0, o1, o2, o3 int, sA, sB, t0, t1 archsimd.Int8x32) archsimd.Int16x16 {
	var z2 archsimd.Uint64x2
	var z4 archsimd.Uint64x4
	lo := z2.SetElem(0, mc8Ld64(s, o0+1)).SetElem(1, mc8Ld64(s, o1+1))
	hi := z2.SetElem(0, mc8Ld64(s, o2+1)).SetElem(1, mc8Ld64(s, o3+1))
	y := z4.SetLo(lo).SetHi(hi).ReshapeToUint8s()
	return y.PermuteOrZeroGrouped(sA).DotProductPairsSaturated(t0).Add(y.PermuteOrZeroGrouped(sB).DotProductPairsSaturated(t1))
}

// mc8Rows4 returns the offsets of the source rows r..r+3 of a rows-row window,
// repeating the last row past the end.
func mc8Rows4(ss, r, rows int) (int, int, int, int) {
	return r * ss, min(r+1, rows-1) * ss, min(r+2, rows-1) * ss, min(r+3, rows-1) * ss
}

// mc8StPutW4 stores the rows r..r+3 (r+3 < h only) of v: dwords 0, 1 of the
// low lane and 0, 1 of the high lane.
func mc8StPutW4(dst []byte, ds, r, h int, v archsimd.Uint8x32) {
	u := v.ReshapeToUint32s()
	lo, hi := u.GetLo(), u.GetHi()
	d := r * ds
	mc8St4(dst, d, lo.GetElem(0))
	if r+1 < h {
		mc8St4(dst, d+ds, lo.GetElem(1))
	}
	if r+2 < h {
		mc8St4(dst, d+2*ds, hi.GetElem(0))
	}
	if r+3 < h {
		mc8St4(dst, d+3*ds, hi.GetElem(1))
	}
}

// mc8StPrepW4 stores the rows r..r+3 (r+3 < h only) of u to a stride-4
// CONV_BUF.
func mc8StPrepW4(out []uint16, r, h int, u archsimd.Uint16x16) {
	o := r * 4
	if r+4 <= h {
		mc8StU16(out, o, u)
		return
	}
	lo := u.GetLo()
	if r+2 <= h {
		mc8StU16x8(out, o, lo)
		if r+3 == h {
			mc8StU16x4(out, o+8, u.GetHi().ReshapeToUint64s().GetElem(0))
		}
		return
	}
	mc8StU16x4(out, o, lo.ReshapeToUint64s().GetElem(0))
}

// mc8AVX2PutH4W4 is the w=4 put horizontal-only pass.
func mc8AVX2PutH4W4(src []byte, ss, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(4, h, ss, ds)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		o0, o1, o2, o3 := mc8Rows4(ss, r, h)
		a := mc8H4x4(src, o0, o1, o2, o3, sA, sB, t0, t1).Add(rnd).ShiftAllRight(6)
		mc8StPutW4(dst, ds, r, h, mc8Pack(a, a, zero, c255, deint))
	}
}

// mc8AVX2PrepH4W4 is the w=4 prep horizontal-only pass.
func mc8AVX2PrepH4W4(src []byte, ss, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(4, h, ss, 0)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		o0, o1, o2, o3 := mc8Rows4(ss, r, h)
		a := mc8H4x4(src, o0, o1, o2, o3, sA, sB, t0, t1).Add(rnd)
		mc8StPrepW4(out, r, h, a.ToBits().ShiftAllRight(2))
	}
}

// mc8AVX2MidH4W4 writes the w=4 2D intermediate (row stride 4). The last
// group may write up to three rows past rows inside the scratch block.
func mc8AVX2MidH4W4(src []byte, ss, rows int, ph *mc8Phase, im *mc8IM) {
	mc8Dims(4, rows, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2)
	for ur := uint(0); ur < uint(rows); ur += 4 {
		r := int(ur)
		o0, o1, o2, o3 := mc8Rows4(ss, r, rows)
		a := mc8H4x4(src, o0, o1, o2, o3, sA, sB, t0, t1).Add(rnd).ShiftAllRight(2)
		mc8StI16(im, r*4, a)
	}
}

// mc8PackW4 packs the h+5 four-byte source rows into im at stride 4 bytes (two
// words per row), so a 16-byte load holds four consecutive rows.
func mc8PackW4(src []byte, ss, h int, im *mc8IM) {
	for k := range h + 5 {
		v := mc8Ld32(src, k*ss)
		p := (*[2]int16)(im[2*k : 2*k+2])
		p[0], p[1] = int16(v), int16(v>>16) // one 32-bit store
	}
}

// mc8ZxW4 zero-extends the 16 packed bytes at word i of im.
func mc8ZxW4(im *mc8IM, i int) archsimd.Uint16x16 {
	return archsimd.LoadInt16x8Array((*[8]int16)(im[uint(i) : uint(i)+8])).ToBits().ReshapeToUint8s().ExtendToUint16()
}

// mc8AVX2PutV6W4 is the w=4 put vertical-only pass. The h+5 source rows are
// first packed into im (mc8PackW4); zero-extended, row k+1 shifted into the
// high bytes forms the vertical byte pairs for four output rows at once.
func mc8AVX2PutV6W4(src []byte, ss, h int, ph *mc8Phase, im *mc8IM, dst []byte, ds int) {
	mc8Dims(4, h, ss, ds)
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(src, ss, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	rnd := archsimd.BroadcastInt16x16(32)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		q := 2 * (r + 4)
		pc := mc8Pair(mc8ZxW4(im, q), mc8ZxW4(im, q+2))
		a := mc8V6(pa, pb, pc, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		mc8StPutW4(dst, ds, r, h, mc8Pack(a, a, zero, c255, deint))
		pa = pc
		pb = mc8Pair(mc8ZxW4(im, q+4), mc8ZxW4(im, q+6))
	}
}

// mc8AVX2PrepV6W4 is the w=4 prep vertical-only pass, laid out as
// mc8AVX2PutV6W4.
func mc8AVX2PrepV6W4(src []byte, ss, h int, ph *mc8Phase, im *mc8IM, out []uint16) {
	mc8Dims(4, h, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(src, ss, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		q := 2 * (r + 4)
		pc := mc8Pair(mc8ZxW4(im, q), mc8ZxW4(im, q+2))
		s := mc8V6(pa, pb, pc, t0, t1, t2)
		mc8StPrepW4(out, r, h, s.Add(rnd).ToBits().ShiftAllRight(2))
		pa = pc
		pb = mc8Pair(mc8ZxW4(im, q+4), mc8ZxW4(im, q+6))
	}
}

// mc8AVX2PutV6MidW4 is the w=4 put 2D vertical pass. A 32-byte load at row k
// of the stride-4 intermediate holds rows k..k+3; the per-lane word interleave
// with the load at row k+1 gives the row pairs (k, k+1) | (k+2, k+3) (low
// halves) and (k+1, k+2) | (k+3, k+4) (high halves), so VPACKUSDW of the two
// sums is rows r..r+3 in order.
func mc8AVX2PutV6MidW4(im *mc8IM, h int, ph *mc8Phase, dst []byte, ds int) {
	mc8Dims(4, h, 0, ds)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	// clip((S + 512) >> 10) is ((S + 512) >> 6) >> 4 after the
	// unsigned-saturating pack (nested floor division; the >> 6 values fit).
	// Both shifts stay immediates: a variable count makes the compiler emit a
	// legacy-SSE MOVQ into an XMM register.
	rnd := archsimd.BroadcastInt32x8(512)
	c255 := archsimd.BroadcastUint16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	l0, l1 := mc8LdI16(im, 0), mc8LdI16(im, 4)
	l2, l3 := mc8LdI16(im, 8), mc8LdI16(im, 12)
	pa, qa := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
	pb, qb := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		q := 4 * (r + 4)
		l4, l5 := mc8LdI16(im, q), mc8LdI16(im, q+4)
		pc, qc := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		x := mc8VMid(pa, pb, pc, d0, d1, d2, rnd).ShiftAllRight(6)
		y := mc8VMid(qa, qb, qc, d0, d1, d2, rnd).ShiftAllRight(6)
		u := x.SaturateToUint16ConcatGrouped(y)
		mc8StPutW4(dst, ds, r, h, u.ShiftAllRight(4).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint))
		pa, qa = pc, qc
		l6, l7 := mc8LdI16(im, q+8), mc8LdI16(im, q+12)
		pb, qb = l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
	}
}

// mc8AVX2PrepV6MidW4 is the w=4 prep 2D vertical pass,
// (sum(half tap * m) + 32 + 6144 << 6) >> 6.
func mc8AVX2PrepV6MidW4(im *mc8IM, h int, ph *mc8Phase, out []uint16) {
	mc8Dims(4, h, 0, 0)
	_ = im[0] // one nil check, ahead of the loops
	d0, d1, d2 := mc8TapsD(ph.d[0]), mc8TapsD(ph.d[1]), mc8TapsD(ph.d[2])
	rnd := archsimd.BroadcastInt32x8(32 + 6144<<6)
	l0, l1 := mc8LdI16(im, 0), mc8LdI16(im, 4)
	l2, l3 := mc8LdI16(im, 8), mc8LdI16(im, 12)
	pa, qa := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
	pb, qb := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
	for ur := uint(0); ur < uint(h); ur += 4 {
		r := int(ur)
		q := 4 * (r + 4)
		l4, l5 := mc8LdI16(im, q), mc8LdI16(im, q+4)
		pc, qc := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		x := mc8VMid(pa, pb, pc, d0, d1, d2, rnd).ShiftAllRight(6)
		y := mc8VMid(qa, qb, qc, d0, d1, d2, rnd).ShiftAllRight(6)
		mc8StPrepW4(out, r, h, x.SaturateToUint16ConcatGrouped(y))
		pa, qa = pc, qc
		l6, l7 := mc8LdI16(im, q+8), mc8LdI16(im, q+12)
		pb, qb = l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
	}
}

// mc8AVX2PrepCopy is the prep copy (dav1d prep_8bpc): 16 * px + 6144.
func mc8AVX2PrepCopy(src []byte, ss, w, h int, out []uint16) {
	mc8Dims(w, h, ss, 0)
	off := archsimd.BroadcastUint16x16(6144)
	switch w {
	case 4:
		for ur := uint(0); ur < uint(h); ur += 4 {
			r := int(ur)
			o0, o1, o2, o3 := mc8Rows4(ss, r, h)
			var z archsimd.Uint32x4
			v := z.SetElem(0, mc8Ld32(src, o0)).SetElem(1, mc8Ld32(src, o1)).SetElem(2, mc8Ld32(src, o2)).SetElem(3, mc8Ld32(src, o3))
			mc8StPrepW4(out, r, h, v.ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off))
		}
	case 8:
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			p0 := r * ss
			p1 := min(r+1, h-1) * ss
			var z archsimd.Uint64x2
			u := z.SetElem(0, mc8Ld64(src, p0)).SetElem(1, mc8Ld64(src, p1)).ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off)
			if r+1 < h {
				mc8StU16(out, r*8, u)
			} else {
				mc8StU16x8(out, r*8, u.GetLo())
			}
		}
	default:
		for r := 0; r < h; r++ {
			s, o := r*ss, r*w
			for uc := uint(0); uc < uint(w&^15); uc += 16 {
				c := int(uc)
				mc8StU16(out, o+c, mc8Zx(src, s+c).ShiftAllLeft(4).Add(off))
			}
			if c := w &^ 15; c < w {
				mc8StU16x8(out, o+c, mc8Zx(src, s+c).ShiftAllLeft(4).Add(off).GetLo())
			}
		}
	}
}

// The remaining dav1d-style fast paths are arm64-only for now.

func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	return false
}
