// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
	"simd/archsimd"
)

func u8HalfMAC(x, coefficient, sum archsimd.Int16x8) archsimd.Int16x8 {
	return x.Mul(coefficient).Add(sum)
}

func u8HalfSaturate(x archsimd.Int16x8) archsimd.Uint8x16 {
	x = x.Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	return x.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8EvenBytesIdx))
}

// VPSHUFB indices: the low byte of each word into bytes 0..7, and the low
// byte of each dword into bytes 0..3 (lo) or 4..7 (hi); -1 zeroes the rest.
var (
	u8EvenBytesIdx = [16]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	u8LoDwordBytes = [16]int8{0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	u8HiDwordBytes = [16]int8{-1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1}
)

func u8WideMAC8(lo, hi archsimd.Int32x4, samples, coefficient archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return hbdMAC8(lo, hi, samples, coefficient)
}

// u8PackClipped32 truncates the eight int32 lanes of lo and hi to bytes 0..7
// (the scalar byte cast), zeroing bytes 8..15.
func u8PackClipped32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	a := lo.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8LoDwordBytes))
	b := hi.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8HiDwordBytes))
	return a.Or(b)
}

func u8PackU16From32(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return hbdJoinU16(lo, hi)
}

func u8MulAdd32(x, coefficient, sum archsimd.Int32x4) archsimd.Int32x4 {
	return x.Mul(coefficient).Add(sum)
}

func u8WidenU16(x archsimd.Uint16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return hbdWidenU16(x)
}

func u8PackSaturate32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	zero, c255 := archsimd.Int32x4{}, archsimd.BroadcastInt32x4(255)
	return u8PackClipped32(lo.Max(zero).Min(c255), hi.Max(zero).Min(c255))
}

func u8Right16(x, amount archsimd.Int16x8, count uint64) archsimd.Int16x8 {
	return x.ShiftAllRight(count)
}

func u8Right32(x, amount archsimd.Int32x4, count uint64) archsimd.Int32x4 {
	return x.ShiftAllRight(count)
}

// scaledStep2Perm holds, per tap and odd-phase offset, the byte shuffle that
// gathers the eight output columns of a 2:1 scaled row: lane x reads byte
// x/2 + (x&1)*odd + t of the load.
var scaledStep2Perm = func() (p [2][filterTaps][16]int8) {
	for odd := range 2 {
		for t := range filterTaps {
			for x := range 8 {
				p[odd][t][x] = int8(x>>1 + (x&1)*odd + t)
				p[odd][t][x+8] = p[odd][t][x]
			}
		}
	}
	return p
}()

// scaledStep2HorizontalRow filters one source row into eight output columns
// at a time for the 2:1 scaled convolve, gathering each tap's samples with a
// byte shuffle (PSHUFB; Permute would emit AVX-512 VBMI VPERMB).
func scaledStep2HorizontalRow(src []byte, w8 int, odd int, coef *[filterTaps]archsimd.Int16x8, row []int16) {
	var perm [filterTaps]archsimd.Int8x16
	for t := range filterTaps {
		perm[t] = archsimd.LoadInt8x16(scaledStep2Perm[odd][t][:])
	}
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	for x := 0; x < w8; x += 8 {
		a := archsimd.LoadUint8x16Array((*[16]byte)(src[x/2 : x/2+16]))
		sum := u8HalfMAC(a.PermuteOrZero(perm[0]).ExtendLo8ToUint16().BitsToInt16(), coef[0], xRound)
		for t := 1; t < filterTaps; t++ {
			sum = u8HalfMAC(a.PermuteOrZero(perm[t]).ExtendLo8ToUint16().BitsToInt16(), coef[t], sum)
		}
		u8Right16(sum, right2, 2).Store(row[x : x+8])
	}
}

// hbdSIMDAvailable reports whether the Go SIMD high-bit-depth kernels may run.
// The amd64 kernels are AVX2-only; AVX-512 is never required.
func hbdSIMDAvailable() bool {
	return archsimd.X86.AVX2()
}

// hbdJoinU16 packs the four low lanes of lo and hi into eight uint16 lanes
// (lo first) with the scalar uint16 cast: VPSHUFB keeps the low word of each
// dword (the AVX2 packs saturate, and the truncating forms need AVX-512).
func hbdJoinU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	a := lo.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&hbdLoWordsLo))
	b := hi.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&hbdLoWordsHi))
	return a.Or(b).ReshapeToUint16s()
}

var (
	hbdLoWordsLo = [16]int8{0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1}
	hbdLoWordsHi = [16]int8{-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13}
)

// hbdMAC8 accumulates the products of eight int16 samples s with the broadcast
// coefficient c into lo (lanes 0..3) and hi (lanes 4..7). The samples are
// zero-extended, which equals sign extension for HBD sample values (<= 4095).
func hbdMAC8(lo, hi archsimd.Int32x4, s, c archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	zero := archsimd.BroadcastInt16x8(0)
	c32 := c.ExtendLo4ToInt32()
	lo32 := s.InterleaveLo(zero).ToBits().ReshapeToUint32s().BitsToInt32()
	hi32 := s.InterleaveHi(zero).ToBits().ReshapeToUint32s().BitsToInt32()
	return lo.Add(lo32.Mul(c32)), hi.Add(hi32.Mul(c32))
}

func hbdMAC4(sum archsimd.Int32x4, s, c archsimd.Int16x8) archsimd.Int32x4 {
	zero := archsimd.BroadcastInt16x8(0)
	x := s.InterleaveLo(zero).ToBits().ReshapeToUint32s().BitsToInt32()
	return sum.Add(x.Mul(c.ExtendLo4ToInt32()))
}

// hbdMulAdd32 returns x*y + z for int32 lanes.
func hbdMulAdd32(x, y, z archsimd.Int32x4) archsimd.Int32x4 {
	return x.Mul(y).Add(z)
}

// amd64 archsimd has no signed per-lane Shift operation. The scalar amount is
// invariant for this kernel and retains the same arithmetic right shift.
func hbdShiftRight(v, _ archsimd.Int32x4, n int) archsimd.Int32x4 {
	return v.ShiftAllRight(uint64(n))
}

func hbdShiftRightU16(v archsimd.Uint16x8, _ archsimd.Int16x8, n int) archsimd.Uint16x8 {
	return v.ShiftAllRight(uint64(n))
}

// hbdWidenU16 zero-extends eight uint16 lanes to two int32 vectors (lanes 0..3
// and 4..7) by interleaving with zero.
func hbdWidenU16(u archsimd.Uint16x8) (lo, hi archsimd.Int32x4) {
	zero := archsimd.BroadcastUint16x8(0)
	return u.InterleaveLo(zero).ReshapeToUint32s().BitsToInt32(), u.InterleaveHi(zero).ReshapeToUint32s().BitsToInt32()
}

// narrowU8x8 packs the eight int32 lanes of lo and hi (already clipped to
// [0, 255]) into the low eight bytes of the result (scalar byte cast).
func narrowU8x8(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return u8PackClipped32(lo, hi)
}

// AVX2 16bpc (10- and 12-bit) translational motion compensation, ported from
// dav1d's src/x86/mc16_avx2.asm put_8tap_16bpc / prep_8tap_16bpc: every
// filter runs as eight taps whose pairs feed vpmaddwd against the
// interleaved (s[i], s[i+1]) sample pairs, exact in int32 (samples are at
// most 4095, taps fit int16). Sixteen outputs per ymm pair; the in-lane
// unpack order is undone by the in-lane pack. The arithmetic is the
// libaom-exact one of the scalar paths on the full-tap sum S (round0/round1
// from highBDRoundBits, rc = compoundRound0):
//
//	put H:   clip((S + 2^(round0-1) + 64) >> 7)       (nested round0, 7-round0)
//	put V:   clip((S + 64) >> 7)
//	put 2D:  m = (S + 2^(round0-1)) >> round0
//	         clip((sum(f * m) + 2^(round1-1)) >> round1)  (libaom's offsets cancel)
//	prep H, V: ((S + 2^(rc-1)) >> rc) + 2^14 + 2^13, as uint16
//	prep 2D: m = (S + 2^(rc-1)) >> rc
//	         (sum(f * m) + 2^21 + 2^20 + 64) >> 7, as uint16
//
// Every shift count is an immediate or a hoisted vpsravd vector, and every
// kernel returns through vzeroupper: a legacy-SSE instruction with dirty
// upper ymm state stalls on Zen 3, and the Go compiler emits neither.

// mc16Pairs[kind][phase][i] is the vpmaddwd tap pair (f[2i], f[2i+1]) of the
// full eight-tap filter.
var mc16Pairs [mc8KindCount][16][4]int32

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
		for s := range 16 {
			for i := range 4 {
				mc16Pairs[kind][s][i] = int32(uint32(uint16(tab[s][2*i])) | uint32(uint16(tab[s][2*i+1]))<<16)
			}
		}
	}
}

const mc16PrepOffset = 1<<14 + 1<<13

// mc16IM is the int32 2D intermediate: ConvolveScratch.imHBD and
// CompoundConvolveScratch.im share this shape.
type mc16IM = [compoundIMMaxSamples]int32

// mc16SourceAVX2 resolves the eight-tap source window of a w x h block: an
// axis with a filter reads from three samples before the block. The kernels
// load sixteen (or eight, for the tails and w == 4) samples per tap, so the
// window must stay readable for max(w, 8) + 7 samples per row when filtered
// horizontally (max(w, 8) otherwise). Overhanging blocks materialize the
// window with emuEdge16. The returned window is exactly bh rows of that
// footprint, so the kernels' checked row loads stay inside it.
func mc16SourceAVX2(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *emuEdge16Buf) (src []byte, stride int, ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	read := max(w, 8)
	if hx {
		x0 -= 3
		bw += 7
		read += 7
	}
	if hy {
		y0 -= 3
		bh += 7
	}
	read *= 2
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width || ref.Stride > mc8MaxStride {
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		if end := start + (bh-1)*ref.Stride + read; end <= len(ref.Pix) {
			return ref.Pix[start:end], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return nil, 0, false
	}
	// emuEdge16Rows (136) rows of emuEdge16Stride (144) samples hold the
	// window (bw <= 135, bh <= 135) and the kernels' over-read.
	end := (bh-1)*2*emuEdge16Stride + read
	if end > len(edge) {
		return nil, 0, false
	}
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	return edge[:end], 2 * emuEdge16Stride, true
}

// mc16Shape reports whether the fast kernels cover a w x h block.
func mc16Shape(bitDepth, w, h int) bool {
	return mc8AVX2 && (bitDepth == 10 || bitDepth == 12) && (w&7 == 0 || w == 4) && w <= maxBlockSize && h > 0 && h <= maxBlockSize
}

// mc16PutFast predicts a w x h block of a 10- or 12-bit plane. It returns
// false for shapes it does not cover (the caller runs the general path).
func mc16PutFast(dst *frame.Plane, ref *frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 || !mc16Shape(bitDepth, w, h) || scratch == nil {
		return false
	}
	dOff := dstY*dst.Stride + 2*dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width || dst.Stride > mc8MaxStride ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) {
		return false
	}
	src, stride, ok := mc16SourceAVX2(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	dg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h)
	maxV := int16(1)<<bitDepth - 1
	round0, round1 := highBDRoundBits(uint8(bitDepth))
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		mc16HToI32(src, stride, w, h+7, px, round0, &scratch.imHBD)
		mc16VFromI32(&scratch.imHBD, w, h, py, true, round1, maxV, dg, nil)
	case subX != 0:
		mc16H(src, stride, w, h, px, true, round0, maxV, dst.Pix[dOff:dOff+(h-1)*dst.Stride+2*w], nil, dst.Stride)
	default:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h+7) {
			return false
		}
		mc16V(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h+7), w, h, py, true, 0, maxV, dg, nil)
	}
	return true
}

// mc16PrepFast fills a w x h CONV_BUF (row stride w) from a 10- or 12-bit
// reference. It returns false for shapes it does not cover.
func mc16PrepFast(out []uint16, ref *frame.Plane, bitDepth, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if !mc16Shape(bitDepth, w, h) || scratch == nil || len(out) < w*h {
		return false
	}
	src, stride, ok := mc16SourceAVX2(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	rc := compoundRound0(uint8(bitDepth))
	o := out[:w*h]
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		im := (*mc16IM)(&scratch.im)
		mc16HToI32(src, stride, w, h+7, px, rc, im)
		mc16VFromI32(im, w, h, py, false, 0, 0, unsafeutil.Grid[uint8]{}, o)
	case subX != 0:
		mc16H(src, stride, w, h, px, false, rc, 0, nil, o, w)
	case subY != 0:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h+7) {
			return false
		}
		mc16V(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h+7), w, h, py, false, rc, 0, unsafeutil.Grid[uint8]{}, o)
	default:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h) {
			return false
		}
		mc16PrepCopy(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h), w, h, 7-rc, o)
	}
	return true
}

// mc16TapPairs16 broadcasts the four tap pairs.
func mc16TapPairs16(p *[4]int32) (k0, k1, k2, k3 archsimd.Int16x16) {
	return archsimd.BroadcastInt32x8(p[0]).AsInt16x16(), archsimd.BroadcastInt32x8(p[1]).AsInt16x16(),
		archsimd.BroadcastInt32x8(p[2]).AsInt16x16(), archsimd.BroadcastInt32x8(p[3]).AsInt16x16()
}

// Sample windows: offsets are in bytes of the 16-bit plane rows (or in
// elements of the uint16 / int32 buffers); the kernels bound their dimensions
// first (mc8Dims), so each checked window is one compare.

// mc16Dot8 is the eight-tap sum of the sample rows (or shifted sample
// windows) a0..a7: lo holds outputs 0-3 and 8-11, hi 4-7 and 12-15 (the
// in-lane unpack order).
func mc16Dot8(a0, a1, a2, a3, a4, a5, a6, a7, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	lo = a0.InterleaveLoGrouped(a1).DotProductPairs(k0).Add(a2.InterleaveLoGrouped(a3).DotProductPairs(k1)).
		Add(a4.InterleaveLoGrouped(a5).DotProductPairs(k2)).Add(a6.InterleaveLoGrouped(a7).DotProductPairs(k3))
	hi = a0.InterleaveHiGrouped(a1).DotProductPairs(k0).Add(a2.InterleaveHiGrouped(a3).DotProductPairs(k1)).
		Add(a4.InterleaveHiGrouped(a5).DotProductPairs(k2)).Add(a6.InterleaveHiGrouped(a7).DotProductPairs(k3))
	return lo, hi
}

// mc16HDot loads the eight shifted windows of the sixteen outputs at s[o:]
// (one checked 46-byte window) and returns their tap sums.
func mc16HDot(s []byte, o int, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	x := (*[46]byte)(s[uint(o) : uint(o)+46])
	l := func(k int) archsimd.Int16x16 {
		return archsimd.LoadUint8x32Array((*[32]byte)(x[k : k+32])).AsInt16x16()
	}
	return mc16Dot8(l(0), l(2), l(4), l(6), l(8), l(10), l(12), l(14), k0, k1, k2, k3)
}

// mc16HDot8 is mc16HDot for eight outputs (one checked 30-byte window).
func mc16HDot8(s []byte, o int, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	x := (*[30]byte)(s[uint(o) : uint(o)+30])
	var z archsimd.Int16x16
	l := func(k int) archsimd.Int16x16 {
		return z.SetLo(archsimd.LoadUint8x16Array((*[16]byte)(x[k : k+16])).AsInt16x8())
	}
	return mc16Dot8(l(0), l(2), l(4), l(6), l(8), l(10), l(12), l(14), k0, k1, k2, k3)
}

// mc16Out finishes sixteen (n = 16), eight or four sums in the in-lane order
// and stores them: put clips (lo, hi) >> sh to [0, maxV] into the plane bytes
// d at byte offset do; prep adds the CONV_BUF offset, keeps the low 16 bits
// (libaom's uint16 cast) and stores into out at element offset oo.
func mc16Out(lo, hi archsimd.Int32x8, n int, put bool, shV archsimd.Uint32x8, maxV archsimd.Int16x16, d []byte, do int, out []uint16, oo int) {
	lo, hi = lo.ShiftRight(shV), hi.ShiftRight(shV)
	var v archsimd.Int16x16
	if put {
		v = lo.SaturateToInt16ConcatGrouped(hi).Max(archsimd.Int16x16{}).Min(maxV)
		switch n {
		case 16:
			v.AsUint8x32().StoreArray((*[32]byte)(d[uint(do) : uint(do)+32]))
		case 8:
			v.GetLo().AsUint8x16().StoreArray((*[16]byte)(d[uint(do) : uint(do)+16]))
		default:
			binary.LittleEndian.PutUint64(d[uint(do):uint(do)+8], v.GetLo().AsUint64x2().GetElem(0))
		}
		return
	}
	off := archsimd.BroadcastInt32x8(mc16PrepOffset)
	mask := archsimd.BroadcastInt32x8(0xffff)
	u := lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask))
	switch n {
	case 16:
		u.StoreArray((*[16]uint16)(out[uint(oo) : uint(oo)+16]))
	case 8:
		u.GetLo().StoreArray((*[8]uint16)(out[uint(oo) : uint(oo)+8]))
	default:
		mc8StU16x4(out, oo, u.GetLo().ReshapeToUint64s().GetElem(0))
	}
}

// mc16H is the horizontal-only pass: put (clip((S + rnd) >> 7)) into dst
// (row stride ds bytes) or prep (((S + rnd) >> rc) + offset) into out (row
// stride ds elements). r is round0 (put) or rc (prep).
func mc16H(src []byte, ss, w, h int, pairs *[4]int32, put bool, r int, maxPix int16, dst []byte, out []uint16, ds int) {
	mc8Dims(w, h, ss, ds)
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = int32(1)<<(r-1) + 64
		shV = archsimd.BroadcastUint32x8(7)
	} else {
		rnd = int32(1) << (r - 1)
		shV = archsimd.BroadcastUint32x8(uint32(r))
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	for y := 0; y < h; y++ {
		s, d := y*ss, y*ds
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			lo, hi := mc16HDot(src, s+2*c, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), 16, put, shV, maxV, dst, d+2*c, out, d+c)
		}
		if c := w &^ 15; c < w {
			lo, hi := mc16HDot8(src, s+2*c, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), w-c, put, shV, maxV, dst, d+2*c, out, d+c)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16V is the vertical-only pass (put: clip((S + 64) >> 7) into dg; prep:
// as mc16H into out), sliding an eight-row window down each column strip of
// sg, the h+7 source rows. The strips are Grid views (sixteen samples, or
// eight for the tail), their rows named by the output row.
func mc16V(sg unsafeutil.Grid[uint8], w, h int, pairs *[4]int32, put bool, rc int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	mc16VCols(sg, w, h, pairs, put, rc, maxPix, dg, out)
	archsimd.ClearAVXUpperBits()
}

func mc16VCols(sg unsafeutil.Grid[uint8], w, h int, pairs *[4]int32, put bool, rc int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = 64
		shV = archsimd.BroadcastUint32x8(7)
	} else {
		rnd = int32(1) << (rc - 1)
		shV = archsimd.BroadcastUint32x8(uint32(rc))
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	off := archsimd.BroadcastInt32x8(mc16PrepOffset)
	mask := archsimd.BroadcastInt32x8(0xffff)
	var zero archsimd.Int16x16
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, 2*c, 32, h+7) || (put && !dg.Fits(0, 2*c, 32, h)) {
			return
		}
		s := sg.Sub(0, 2*c, 32, h+7)
		r0 := archsimd.LoadUint8x32Array(s.Row32(0, 0)).AsInt16x16()
		r1 := archsimd.LoadUint8x32Array(s.Row32(1, 0)).AsInt16x16()
		r2 := archsimd.LoadUint8x32Array(s.Row32(2, 0)).AsInt16x16()
		r3 := archsimd.LoadUint8x32Array(s.Row32(3, 0)).AsInt16x16()
		r4 := archsimd.LoadUint8x32Array(s.Row32(4, 0)).AsInt16x16()
		r5 := archsimd.LoadUint8x32Array(s.Row32(5, 0)).AsInt16x16()
		r6 := archsimd.LoadUint8x32Array(s.Row32(6, 0)).AsInt16x16()
		t := s.Sub(7, 0, 32, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 32, h)
		}
		for y := 0; y < h; y++ {
			r7 := archsimd.LoadUint8x32Array(t.Row32(y, 0)).AsInt16x16()
			lo, hi := mc16Dot8(r0, r1, r2, r3, r4, r5, r6, r7, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			if put {
				lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).AsUint8x32().StoreArray(d.Row32(y, 0))
			} else {
				o := y*w + c
				lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).StoreArray((*[16]uint16)(out[uint(o) : uint(o)+16]))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
	if n := w - c; n > 0 {
		// Eight (or, at w == 4, four) columns: sixteen-byte rows.
		if !sg.Fits(0, 2*c, 16, h+7) || (put && !dg.Fits(0, 2*c, 2*n, h)) {
			return
		}
		s := sg.Sub(0, 2*c, 16, h+7)
		r0 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(0, 0)).AsInt16x8())
		r1 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(1, 0)).AsInt16x8())
		r2 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(2, 0)).AsInt16x8())
		r3 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(3, 0)).AsInt16x8())
		r4 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(4, 0)).AsInt16x8())
		r5 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(5, 0)).AsInt16x8())
		r6 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(6, 0)).AsInt16x8())
		t := s.Sub(7, 0, 16, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 2*n, h)
		}
		for y := 0; y < h; y++ {
			r7 := zero.SetLo(archsimd.LoadUint8x16Array(t.Row16(y, 0)).AsInt16x8())
			lo, hi := mc16Dot8(r0, r1, r2, r3, r4, r5, r6, r7, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			o := y*w + c
			switch {
			case put && n == 8:
				lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).GetLo().AsUint8x16().StoreArray(d.Row16(y, 0))
			case put:
				binary.LittleEndian.PutUint64(d.Row8(y, 0)[:], lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).GetLo().AsUint64x2().GetElem(0))
			case n == 8:
				lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).GetLo().StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			default:
				mc8StU16x4(out, o, lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).GetLo().ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
}

// mc16HToI32 writes rows x w rounded horizontal sums m = (S + 2^(r-1)) >> r
// as int32 with row stride w (r is round0 for put, rc for prep).
func mc16HToI32(src []byte, ss, w, rows int, pairs *[4]int32, r int, im *mc16IM) {
	mc8Dims(w, rows, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	rndV := archsimd.BroadcastInt32x8(int32(1) << (r - 1))
	shV := archsimd.BroadcastUint32x8(uint32(r))
	for y := 0; y < rows; y++ {
		s, o := y*ss, y*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			lo, hi := mc16HDot(src, s+2*c, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			// Undo the in-lane order: outputs 0-7, then 8-15.
			p := (*[16]int32)(im[uint(o+c) : uint(o+c)+16])
			lo.ConcatPermute128Scalars(0, 2, hi).StoreArray((*[8]int32)(p[:8]))
			lo.ConcatPermute128Scalars(1, 3, hi).StoreArray((*[8]int32)(p[8:]))
		}
		if c := w &^ 15; c < w {
			lo, hi := mc16HDot8(src, s+2*c, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			v := lo.ConcatPermute128Scalars(0, 2, hi)
			if w-c == 8 {
				v.StoreArray((*[8]int32)(im[uint(o+c) : uint(o+c)+8]))
			} else {
				v.GetLo().StoreArray((*[4]int32)(im[uint(o+c) : uint(o+c)+4]))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16PairOrder is the vpermd order that restores sample order after a
// vpackssdw/vpackusdw of two eight-column int32 vectors.
var mc16PairOrder = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// mc16VFromI32 is the 2D vertical pass over the int32 intermediate (row
// stride w, h+7 rows, a Grid view whose eight tap rows of output row y are
// row y of eight h-row sub-views, one per tap): sum(f * m) in int32 (vpmulld), then put
// clips (sum + 2^(round1-1)) >> round1 into dg and prep stores
// (sum + 2^21 + 2^20 + 64) >> 7 as uint16 into out (row stride w).
func mc16VFromI32(im *mc16IM, w, h int, pairs *[4]int32, put bool, round1 int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	mc16VFromI32Cols(im, w, h, pairs, put, round1, maxPix, dg, out)
	archsimd.ClearAVXUpperBits()
}

func mc16VFromI32Cols(im *mc16IM, w, h int, pairs *[4]int32, put bool, round1 int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	if uint(h-1) >= mc8MaxRows || uint(w-1) >= maxBlockSize || !unsafeutil.GridFit(len(im), 0, w, w, h+7) {
		return
	}
	g := unsafeutil.NewGrid(im[:], 0, w, w, h+7)
	var f [8]archsimd.Int32x8
	for i := range 4 {
		f[2*i] = archsimd.BroadcastInt32x8(int32(int16(pairs[i])))
		f[2*i+1] = archsimd.BroadcastInt32x8(pairs[i] >> 16)
	}
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = int32(1) << (round1 - 1)
		shV = archsimd.BroadcastUint32x8(uint32(round1))
	} else {
		rnd = 1<<21 + 1<<20 + 64
		shV = archsimd.BroadcastUint32x8(7)
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	mask := archsimd.BroadcastInt32x8(0xffff)
	perm := archsimd.LoadUint32x8Array(&mc16PairOrder)
	c := 0
	for ; c+8 <= w; c += 8 {
		if !g.Fits(0, c, 8, h+7) || (put && !dg.Fits(0, 2*c, 16, h)) {
			return
		}
		s0 := g.Sub(0, c, 8, h)
		s1 := g.Sub(1, c, 8, h)
		s2 := g.Sub(2, c, 8, h)
		s3 := g.Sub(3, c, 8, h)
		s4 := g.Sub(4, c, 8, h)
		s5 := g.Sub(5, c, 8, h)
		s6 := g.Sub(6, c, 8, h)
		s7 := g.Sub(7, c, 8, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 16, h)
		}
		for y := 0; y < h; y++ {
			acc := rndV.Add(archsimd.LoadInt32x8Array(s0.Row8(y, 0)).Mul(f[0]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s1.Row8(y, 0)).Mul(f[1]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s2.Row8(y, 0)).Mul(f[2]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s3.Row8(y, 0)).Mul(f[3]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s4.Row8(y, 0)).Mul(f[4]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s5.Row8(y, 0)).Mul(f[5]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s6.Row8(y, 0)).Mul(f[6]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s7.Row8(y, 0)).Mul(f[7]))
			acc = acc.ShiftRight(shV)
			if put {
				acc.SaturateToInt16ConcatGrouped(acc).AsInt32x8().Permute(perm).AsInt16x16().Max(archsimd.Int16x16{}).Min(maxV).GetLo().AsUint8x16().StoreArray(d.Row16(y, 0))
			} else {
				a := acc.And(mask)
				o := y*w + c
				a.SaturateToUint16ConcatGrouped(a).AsInt32x8().Permute(perm).AsUint16x16().GetLo().StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			}
		}
	}
	if c < w {
		// Four columns (w == 4).
		if !g.Fits(0, c, 4, h+7) || (put && !dg.Fits(0, 2*c, 8, h)) {
			return
		}
		s0 := g.Sub(0, c, 4, h)
		s1 := g.Sub(1, c, 4, h)
		s2 := g.Sub(2, c, 4, h)
		s3 := g.Sub(3, c, 4, h)
		s4 := g.Sub(4, c, 4, h)
		s5 := g.Sub(5, c, 4, h)
		s6 := g.Sub(6, c, 4, h)
		s7 := g.Sub(7, c, 4, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 8, h)
		}
		var z archsimd.Int32x8
		for y := 0; y < h; y++ {
			acc := rndV.Add(z.SetLo(archsimd.LoadInt32x4Array(s0.Row4(y, 0))).Mul(f[0]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s1.Row4(y, 0))).Mul(f[1]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s2.Row4(y, 0))).Mul(f[2]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s3.Row4(y, 0))).Mul(f[3]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s4.Row4(y, 0))).Mul(f[4]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s5.Row4(y, 0))).Mul(f[5]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s6.Row4(y, 0))).Mul(f[6]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s7.Row4(y, 0))).Mul(f[7]))
			acc = acc.ShiftRight(shV)
			if put {
				v := acc.SaturateToInt16ConcatGrouped(acc).AsInt32x8().Permute(perm).AsInt16x16().Max(archsimd.Int16x16{}).Min(maxV)
				binary.LittleEndian.PutUint64(d.Row8(y, 0)[:], v.GetLo().AsUint64x2().GetElem(0))
			} else {
				a := acc.And(mask)
				v := a.SaturateToUint16ConcatGrouped(a).AsInt32x8().Permute(perm).AsUint16x16()
				mc8StU16x4(out, y*w+c, v.GetLo().ReshapeToUint64s().GetElem(0))
			}
		}
	}
}

// mc16PrepCopy is the prep copy: (px << shl) + offset, in uint16 lanes (the
// result stays below 2^16 at 10 and 12 bits), from the h rows of sg.
func mc16PrepCopy(sg unsafeutil.Grid[uint8], w, h, shl int, out []uint16) {
	mc16PrepCopyCols(sg, w, h, shl, out)
	archsimd.ClearAVXUpperBits()
}

func mc16PrepCopyCols(sg unsafeutil.Grid[uint8], w, h, shl int, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	mul := archsimd.BroadcastInt16x16(int16(1) << shl)
	off := archsimd.BroadcastInt16x16(mc16PrepOffset) // wraps mod 2^16 like the uint16 sum
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, 2*c, 32, h) {
			return
		}
		s := sg.Sub(0, 2*c, 32, h)
		for y := 0; y < h; y++ {
			o := y*w + c
			archsimd.LoadUint8x32Array(s.Row32(y, 0)).AsInt16x16().Mul(mul).Add(off).AsUint16x16().StoreArray((*[16]uint16)(out[uint(o) : uint(o)+16]))
		}
	}
	if n := w - c; n > 0 {
		if !sg.Fits(0, 2*c, 16, h) {
			return
		}
		s := sg.Sub(0, 2*c, 16, h)
		for y := 0; y < h; y++ {
			o := y*w + c
			v := archsimd.Int16x16{}.SetLo(archsimd.LoadUint8x16Array(s.Row16(y, 0)).AsInt16x8()).Mul(mul).Add(off).GetLo().AsUint16x8()
			if n == 8 {
				v.StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			} else {
				mc8StU16x4(out, o, v.ReshapeToUint64s().GetElem(0))
			}
		}
	}
}

// AVX2 16bpc compound blends (dav1d src/x86/mc16_avx2.asm avg / mask /
// w_mask), the 10- and 12-bit siblings of mc8mask_gosimd_amd64.go. They
// compute libaom's CONV_BUF formulas exactly for every uint16 input, with
// rb = 2*7 - round0 - 7 (4 at 10 bits, 2 at 12) and roundOffset = 2^14 + 2^13:
//
//	avg:  clip((((a + b) >> 1) - roundOffset + 2^(rb-1)) >> rb)
//	      floor((a + b) / 2) is the rounding vpavgw minus (a ^ b) & 1, and the
//	      bias is a saturating subtract (a negative result clips to 0 either
//	      way); >> rb is vpmulhuw by 2^(16-rb).
//	mask: clip((m*a + (64-m)*b - 64*(roundOffset - 2^(rb-1))) >> (6 + rb)),
//	      vpmaddwd on the samples biased by -2^15 (exact in int32), the bias
//	      folded into the constant.
//	diff-weighted mask: min(38 + (((|a-b| >> 5) + 1) >> 5), 64).
//
// The kernels take the block's rows as slices (the destination window, the
// w*h CONV_BUFs, the mask window) and load and store through one checked
// window per vector, with the dimensions bounded first (mc8Dims).

// mc16BlendRoundBits returns rb and the avg bias roundOffset - 2^(rb-1).
func mc16BlendRoundBits(bitDepth int) (rb int, bias uint16) {
	rb = 2*filterBits - compoundRound0(uint8(bitDepth)) - compoundRound1Bits
	return rb, uint16(mc16PrepOffset - 1<<(rb-1))
}

// mc16BlendDst checks the destination window of a w x h 16bpc blend and
// returns it.
func mc16BlendDst(dst *frame.Plane, dstX, dstY, w, h int) ([]byte, bool) {
	off := dstY*dst.Stride + 2*dstX
	end := off + (h-1)*dst.Stride + 2*w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride || w > maxBlockSize || h > maxBlockSize {
		return nil, false
	}
	return dst.Pix[off:end], true
}

// mc16BlendAvgFast is the 16bpc plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (w a multiple of 8, or 4 with an
// even height).
func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	d, ok := mc16BlendDst(dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendAvgAVX2(d, dst.Stride, a[:w*h], b[:w*h], w, h, rb, bias, uint16(1)<<bitDepth-1)
	return true
}

// mc16Avg16 is the avg formula on sixteen lanes.
func mc16Avg16(va, vb, one, bias, mul, maxV archsimd.Uint16x16) archsimd.Uint16x16 {
	return va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).MulHigh(mul).Min(maxV)
}

func mc16LdU16(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(s[uint(o) : uint(o)+16]))
}

func mc16LdU16x8(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(s[uint(o) : uint(o)+8])))
}

func mc16BlendAvgAVX2(dst []byte, ds int, a, b []uint16, w, h, rb int, biasC, maxC uint16) {
	mc8Dims(w, h, 0, ds)
	one := archsimd.BroadcastUint16x16(1)
	bias := archsimd.BroadcastUint16x16(biasC)
	mul := archsimd.BroadcastUint16x16(uint16(1) << (16 - rb))
	maxV := archsimd.BroadcastUint16x16(maxC)
	switch {
	case w == 4:
		// Four rows of four per vector (the CONV_BUF rows are contiguous);
		// a height of 2 mod 4 finishes with eight lanes.
		for ur := uint(0); ur < uint(h&^3); ur += 4 {
			r := int(ur)
			v := mc16Avg16(mc16LdU16(a, 4*r), mc16LdU16(b, 4*r), one, bias, mul, maxV).AsUint64x4()
			a0, a1 := mc8Rows2x8(dst, r*ds, ds)
			b0, b1 := mc8Rows2x8(dst, (r+2)*ds, ds)
			binary.LittleEndian.PutUint64(a0[:], v.GetLo().GetElem(0))
			binary.LittleEndian.PutUint64(a1[:], v.GetLo().GetElem(1))
			binary.LittleEndian.PutUint64(b0[:], v.GetHi().GetElem(0))
			binary.LittleEndian.PutUint64(b1[:], v.GetHi().GetElem(1))
		}
		if r := h &^ 3; r < h {
			v := mc16Avg16(mc16LdU16x8(a, 4*r), mc16LdU16x8(b, 4*r), one, bias, mul, maxV).GetLo().AsUint64x2()
			y0, y1 := mc8Rows2x8(dst, r*ds, ds)
			binary.LittleEndian.PutUint64(y0[:], v.GetElem(0))
			binary.LittleEndian.PutUint64(y1[:], v.GetElem(1))
		}
	default:
		for r := 0; r < h; r++ {
			o, d := r*w, r*ds
			for uc := uint(0); uc < uint(w&^15); uc += 16 {
				c := int(uc)
				v := mc16Avg16(mc16LdU16(a, o+c), mc16LdU16(b, o+c), one, bias, mul, maxV)
				v.AsUint8x32().StoreArray((*[32]byte)(dst[uint(d+2*c) : uint(d+2*c)+32]))
			}
			if c := w &^ 15; c < w {
				v := mc16Avg16(mc16LdU16x8(a, o+c), mc16LdU16x8(b, o+c), one, bias, mul, maxV)
				v.GetLo().AsUint8x16().StoreArray((*[16]byte)(dst[uint(d+2*c) : uint(d+2*c)+16]))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16BlendMaskFast blends with a full-resolution (4:4:4 / luma) or 4:2:0
// subsampled mask; width must be a multiple of 8.
func mc16BlendMaskFast(dst frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	d, ok := mc16BlendDst(&dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	// The kernels load sixteen mask bytes per eight (unsubsampled) or
	// sixteen (subsampled) output columns.
	mEnd := (mh-1)*maskStride + mw + 8
	if maskStride < mw || maskStride > mc8MaxStride || mEnd-8 > len(mask) || mEnd > cap(mask) {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendMaskAVX2(d, dst.Stride, a[:w*h], b[:w*h], w, h, mask[:mEnd], maskStride, subX,
		64<<15-64*int32(bias), uint32(6+rb), int16(1)<<bitDepth-1)
	return true
}

// mc16MaskBlend16 blends sixteen samples with the int16 weights m (0..64).
func mc16MaskBlend16(va, vb, m, m64, bias16 archsimd.Int16x16, c, z archsimd.Int32x8, shV archsimd.Uint32x8, maxV archsimd.Int16x16) archsimd.Int16x16 {
	// Biased samples a - 2^15 fit int16: m*a + (64-m)*b
	// = m*a' + (64-m)*b' + 64*2^15.
	sa, sb := va.Sub(bias16), vb.Sub(bias16)
	mi := m64.Sub(m)
	lo := sa.InterleaveLoGrouped(sb).DotProductPairs(m.InterleaveLoGrouped(mi)).Add(c).ShiftRight(shV)
	hi := sa.InterleaveHiGrouped(sb).DotProductPairs(m.InterleaveHiGrouped(mi)).Add(c).ShiftRight(shV)
	return lo.SaturateToInt16ConcatGrouped(hi).Max(archsimd.Int16x16{}).Min(maxV)
}

func mc16BlendMaskAVX2(dst []byte, ds int, a, b []uint16, w, h int, mask []byte, maskStride int, sub bool, cC int32, sh uint32, maxPix int16) {
	mc8Dims(w, h, maskStride, ds)
	m64 := archsimd.BroadcastInt16x16(64)
	bias16 := archsimd.BroadcastInt16x16(-1 << 15)
	c := archsimd.BroadcastInt32x8(cC)
	shV := archsimd.BroadcastUint32x8(sh)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	var z archsimd.Int32x8
	for r := 0; r < h; r++ {
		o, d := r*w, r*ds
		mrow, mrow1 := r*maskStride, 0
		if sub {
			mrow = 2 * r * maskStride
			mrow1 = mrow + maskStride
		}
		for ucol := uint(0); ucol < uint(w); ucol += 16 {
			col := int(ucol)
			full := w-col >= 16
			var m archsimd.Int16x16
			if sub {
				// 2x2 sums: vpmaddubsw against ones adds the column pairs of
				// each row in place.
				var u0, u1 archsimd.Uint8x32
				if full {
					u0 = mc8Ld32B(mask, mrow+2*col)
					u1 = mc8Ld32B(mask, mrow1+2*col)
				} else {
					u0 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(mc8W16(mask, mrow+2*col)))
					u1 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(mc8W16(mask, mrow1+2*col)))
				}
				v0 := u0.DotProductPairsSaturated(ones)
				v1 := u1.DotProductPairsSaturated(ones)
				m = v0.Add(v1).Add(two).ShiftAllRight(2)
			} else {
				m = mc8Zx(mask, mrow+col).AsInt16x16()
			}
			if full {
				va := mc16LdU16(a, o+col).AsInt16x16()
				vb := mc16LdU16(b, o+col).AsInt16x16()
				v := mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV)
				v.AsUint8x32().StoreArray((*[32]byte)(dst[uint(d+2*col) : uint(d+2*col)+32]))
			} else {
				va := mc16LdU16x8(a, o+col).AsInt16x16()
				vb := mc16LdU16x8(b, o+col).AsInt16x16()
				v := mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV)
				v.GetLo().AsUint8x16().StoreArray((*[16]byte)(dst[uint(d+2*col) : uint(d+2*col)+16]))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16DiffWtdMaskFast builds the 10- or 12-bit difference-weighted mask; width
// must be a multiple of 8.
func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || len(a) < w*h || len(b) < w*h ||
		maskStride < w || maskStride > mc8MaxStride || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc16DiffWtdMaskAVX2(mask[:(h-1)*maskStride+w], maskStride, a[:w*h], b[:w*h], w, h, invert)
	return true
}

// mc16LowBytes gathers the low byte of every int16 lane into the low half of
// its 128-bit lane.
var mc16LowBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

func mc16DiffWtdMaskAVX2(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) {
	mc8Dims(w, h, 0, maskStride)
	one := archsimd.BroadcastUint16x16(1)
	base := archsimd.BroadcastUint16x16(38)
	maxM := archsimd.BroadcastUint16x16(64)
	lowBytes := archsimd.LoadInt8x32Array(&mc16LowBytes)
	for r := 0; r < h; r++ {
		o, mrow := r*w, r*maskStride
		for ucol := uint(0); ucol < uint(w); ucol += 16 {
			col := int(ucol)
			full := w-col >= 16
			var va, vb archsimd.Uint16x16
			if full {
				va, vb = mc16LdU16(a, o+col), mc16LdU16(b, o+col)
			} else {
				va, vb = mc16LdU16x8(a, o+col), mc16LdU16x8(b, o+col)
			}
			diff := va.Max(vb).Sub(va.Min(vb))
			m := diff.ShiftAllRight(5).Add(one).ShiftAllRight(5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			v := m.AsUint8x32().PermuteOrZeroGrouped(lowBytes)
			if full {
				binary.LittleEndian.PutUint64(mask[uint(mrow+col):uint(mrow+col)+8], v.GetLo().AsUint64x2().GetElem(0))
				binary.LittleEndian.PutUint64(mask[uint(mrow+col+8):uint(mrow+col+8)+8], v.GetHi().AsUint64x2().GetElem(0))
			} else {
				binary.LittleEndian.PutUint64(mask[uint(mrow+col):uint(mrow+col)+8], v.GetLo().AsUint64x2().GetElem(0))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

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

// mc8Put8Tap is Put8's eight-tap (sharp) path, dav1d put_8tap: the checked
// destination window and the clamped source window as slices (mc8Source).
func mc8Put8Tap(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width || dst.Stride > mc8MaxStride {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	dEnd := dOff + (h-1)*dst.Stride + w
	if dEnd > len(dst.Pix) {
		return false
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, 8, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	d := dst.Pix[dOff:dEnd]
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

// mc8Prep8Tap is mc8PrepFast's eight-tap (sharp) path into the w x h
// CONV_BUF o.
func mc8Prep8Tap(o []uint16, ref *frame.Plane, refX, refY, w, h, subX, subY, kx, ky int, scratch *CompoundConvolveScratch) bool {
	src, stride, ok := mc8Source(ref, refX, refY, w, h, 8, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	im := (*mc8IM)(&scratch.im8)
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

// mc8Rows2 returns the 16-byte windows at s[o:] and s[o+ss:] through one
// checked slice (both rows must exist).
func mc8Rows2(s []byte, o, ss int) (*[16]byte, *[16]byte) {
	win := s[uint(o) : uint(o)+uint(ss)+16]
	return (*[16]byte)(win), (*[16]byte)(win[ss:])
}

// mc8Rows2x4 is mc8Rows2 for 4-byte rows.
func mc8Rows2x4(s []byte, o, ss int) (*[4]byte, *[4]byte) {
	win := s[uint(o) : uint(o)+uint(ss)+4]
	return (*[4]byte)(win), (*[4]byte)(win[ss:])
}

// mc8StPutW4 stores the rows r..r+3 (r+3 < h only) of v: dwords 0, 1 of the
// low lane and 0, 1 of the high lane.
func mc8StPutW4(dst []byte, ds, r, h int, v archsimd.Uint8x32) {
	u := v.ReshapeToUint32s()
	lo, hi := u.GetLo(), u.GetHi()
	d := r * ds
	if r+4 <= h { // a full group: two two-row windows
		a0, a1 := mc8Rows2x4(dst, d, ds)
		b0, b1 := mc8Rows2x4(dst, d+2*ds, ds)
		binary.LittleEndian.PutUint32(a0[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(a1[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(b0[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(b1[:], hi.GetElem(1))
		return
	}
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

// mc8Rows2x8 is mc8Rows2 for 8-byte rows.
func mc8Rows2x8(s []byte, o, ss int) (*[8]byte, *[8]byte) {
	win := s[uint(o) : uint(o)+uint(ss)+8]
	return (*[8]byte)(win), (*[8]byte)(win[ss:])
}

// mc8Rows2x24 is mc8Rows2 for 24-byte rows.
func mc8Rows2x24(s []byte, o, ss int) (*[24]byte, *[24]byte) {
	win := s[uint(o) : uint(o)+uint(ss)+24]
	return (*[24]byte)(win), (*[24]byte)(win[ss:])
}

// The eight-column vertical helpers stay in 256-bit registers (a 128-bit
// value carried across iterations makes the compiler emit legacy-SSE MOVUPS
// register copies, which stall on x86 while the upper halves are dirty).

// mc8Join puts the low lane of a in the low and the low lane of b in the high
// lane (VPERM2I128).
func mc8Join(a, b archsimd.Uint8x32) archsimd.Uint8x32 {
	return a.ConcatPermute128Scalars(0, 2, b)
}

// mc8VMid is the six-tap vertical VPMADDWD sum over word-interleaved row
// pairs, plus rnd.
func mc8VMid(p0, p2, p4, d0, d1, d2 archsimd.Int16x16, rnd archsimd.Int32x8) archsimd.Int32x8 {
	return p0.DotProductPairs(d0).Add(rnd).Add(p2.DotProductPairs(d1)).Add(p4.DotProductPairs(d2))
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

func mc8BlendAvgCopyFast(dst, ref0, ref1 *frame.Plane, dstX, dstY, refX0, refY0, refX1, refY1, w, h int) bool {
	return false
}

func mc8BlendDistWtdCopyFast(dst, ref0, ref1 *frame.Plane, dstX, dstY, refX0, refY0, refX1, refY1, w, h, fwd, bck int) bool {
	return false
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
		// The second row of the pair is st (dt) bytes down; the last row of
		// an odd h pairs with itself. One two-row window covers both rows.
		st, dt := ss, ds
		if r+1 >= h {
			st, dt = 0, 0
		}
		s0, d0 := r*ss, r*ds
		if w == 8 {
			a := mc8H8(mc8Ld2(mc8Rows2(src, s0, st)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s()
			y0, y1 := mc8Rows2x8(dst, d0, dt)
			binary.LittleEndian.PutUint64(y0[:], v.GetLo().GetElem(0))
			binary.LittleEndian.PutUint64(y1[:], v.GetHi().GetElem(0))
			continue
		}
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			x0, x1 := mc8Rows2x24(src, s0+c, st)
			a := mc8H8(mc8Ld2(mc8Lo16(x0), mc8Lo16(x1)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			b := mc8H8(mc8Ld2(mc8Hi16(x0), mc8Hi16(x1)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint)
			y0, y1 := mc8Rows2(dst, d0+c, dt)
			v.GetHi().StoreArray(y1)
			v.GetLo().StoreArray(y0)
		}
		if c := w &^ 15; c < w {
			a := mc8H8(mc8Ld2(mc8Rows2(src, s0+c, st)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s()
			y0, y1 := mc8Rows2x8(dst, d0+c, dt)
			binary.LittleEndian.PutUint64(y1[:], v.GetHi().GetElem(0))
			binary.LittleEndian.PutUint64(y0[:], v.GetLo().GetElem(0))
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
			a := mc8H8(mc8Ld2(mc8Rows2(src, s0, ss)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd)
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
			a := mc8H8(mc8Ld2(mc8Rows2(src, s0, ss)), sA, sB, sC, sD, t0, t1, t2, t3).Add(rnd).ShiftAllRight(2)
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

// mc8H4x4Q is mc8H4x4 over the eight bytes at window tap 1 of four rows.
func mc8H4x4Q(q0, q1, q2, q3 uint64, sA, sB, t0, t1 archsimd.Int8x32) archsimd.Int16x16 {
	var z2 archsimd.Uint64x2
	var z4 archsimd.Uint64x4
	lo := z2.SetElem(0, q0).SetElem(1, q1)
	hi := z2.SetElem(0, q2).SetElem(1, q3)
	y := z4.SetLo(lo).SetHi(hi).ReshapeToUint8s()
	return y.PermuteOrZeroGrouped(sA).DotProductPairsSaturated(t0).Add(y.PermuteOrZeroGrouped(sB).DotProductPairsSaturated(t1))
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

// mc8ZxW4 zero-extends the 16 packed bytes at word i of im.
func mc8ZxW4(im *mc8IM, i int) archsimd.Uint16x16 {
	return archsimd.LoadInt16x8Array((*[8]int16)(im[uint(i) : uint(i)+8])).ToBits().ReshapeToUint8s().ExtendToUint16()
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

// The put copies (dav1d put_avx2 / put_16bpc_avx2): an in-plane w x h block
// moved as vector rows over footprint-checked Grid views instead of one
// memmove per row. unsafeutil.CopyRect is not used here: on amd64 the
// compiler lowers its 32- and 64-byte array copies to runtime.memmove calls,
// so the rows go through archsimd loads and stores instead (VMOVDQU ymm, then
// VZEROUPPER at exit as the AVX2 kernels do).

// mc8CopyFast is the 8-bit put copy for an in-plane w x h block with w a
// multiple of 4. It returns false when the source leaves the plane (the
// clamped copy runs).
func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+dstX, refY*ref.Stride+refX
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, w, h) {
		return false
	}
	mc8CopyGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, w, h))
	return true
}

// mc16CopyFast is the 16bpc put copy: an in-plane w x h block of a
// 2-byte-sample plane, w a multiple of 4, copied as rows of 2w bytes. It
// returns false when the source leaves the plane (the clamped copy runs).
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width ||
		refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < 2*ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+2*dstX, refY*ref.Stride+2*refX
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, 2*w, h) {
		return false
	}
	mc8CopyGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, 2*w, h))
	return true
}

// mc8CopyGrid copies the rows of sg onto dg, two views of the same shape
// whose width is a multiple of 4 bytes: 4- and 8-byte rows as scalar moves,
// 16-byte rows as one xmm load and store, wider rows in 32-byte ymm pieces
// (with a 16-byte tail for widths that are 16 mod 32). Every row index is
// the loop variable, bounded by the views' common height, and every column
// a constant or a strip view's, so the rows pay no checks.
func mc8CopyGrid(dg, sg unsafeutil.Grid[uint8]) {
	h, n := dg.Height(), dg.Width()
	if !sg.Fits(0, 0, n, h) {
		return
	}
	switch {
	case n == 4:
		for r := 0; r < h; r++ {
			*dg.Row4(r, 0) = *sg.Row4(r, 0)
		}
	case n == 8:
		for r := 0; r < h; r++ {
			*dg.Row8(r, 0) = *sg.Row8(r, 0)
		}
	case n == 16:
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x16Array(sg.Row16(r, 0)).StoreArray(dg.Row16(r, 0))
		}
	case n == 32:
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x32Array(sg.Row32(r, 0)).StoreArray(dg.Row32(r, 0))
		}
		archsimd.ClearAVXUpperBits()
	case n == 64:
		for r := 0; r < h; r++ {
			s, d := sg.Row64(r, 0), dg.Row64(r, 0)
			a := archsimd.LoadUint8x32Array((*[32]byte)(s[0:32]))
			b := archsimd.LoadUint8x32Array((*[32]byte)(s[32:64]))
			a.StoreArray((*[32]byte)(d[0:32]))
			b.StoreArray((*[32]byte)(d[32:64]))
		}
		archsimd.ClearAVXUpperBits()
	default:
		// Wider rows (and widths that are not powers of two): 64-byte strips,
		// then 32-, 16-, 8- and 4-byte ones.
		c := 0
		for ; c+64 <= n; c += 64 {
			mc8CopyGrid(dg.Sub(0, c, 64, h), sg.Sub(0, c, 64, h))
		}
		for _, k := range [...]int{32, 16, 8, 4} {
			if c+k <= n {
				mc8CopyGrid(dg.Sub(0, c, k, h), sg.Sub(0, c, k, h))
				c += k
			}
		}
	}
}

// The six-tap entries and kernels over footprint-checked unsafeutil.Grid
// views, as on arm64 (mc8_gosimd_arm64.go, mc8grid_gosimd_arm64.go): Put8
// checks the common case (destination and source window inside their planes)
// in line and runs the kernel with no call before it; everything else is
// mc8PutEdge, which materializes the clamped window (mc8SourceGrid).
//
// Each kernel takes fixed-width column strips of its views (Fits, then Sub:
// one test per strip) and names every row by the row loop variable, bounded
// by the strip's height, or by a constant, so the rows pay no check. A
// kernel restates what it relies on with Fits and returns early when it does
// not hold, which the entries make unreachable. The 2D intermediate (int16)
// and the prep CONV_BUF keep their fixed-array and slice windows; the
// eight-tap (sharp) kernels keep their slice windows (mc8Source).

// mc8MaxRows bounds the row counts the kernels take (h+5 rows of a block at
// most maxBlockSize high). The kernels test it first, which tells the compiler
// their row loops cannot overflow, so the row checks drop.
const mc8MaxRows = maxBlockSize + 8

// mc8G4..mc8G24 are row r of g from column 0, as a fixed-size window.
func mc8G4(g unsafeutil.Grid[uint8], r int) *[4]byte   { return g.Row4(r, 0) }
func mc8G8(g unsafeutil.Grid[uint8], r int) *[8]byte   { return g.Row8(r, 0) }
func mc8G16(g unsafeutil.Grid[uint8], r int) *[16]byte { return g.Row16(r, 0) }
func mc8G24(g unsafeutil.Grid[uint8], r int) *[24]byte { return g.Row24(r, 0) }

// mc8Halo is the source window halo of an axis: 2 when its phase is nonzero.
func mc8Halo(sub int) int {
	if sub != 0 {
		return 2
	}
	return 0
}

// mc8SourceGrid resolves the six-tap source window of a w x h block whose
// first output sample reads ref at (refX, refY) as a Grid view from its
// top-left tap sample: bh rows (h, plus 5 on a filtered vertical axis) of
// max(w+8, 16) bytes, the bytes every kernel may load. When any tap falls
// outside the plane the clamped window is first materialized in edge (dav1d
// src/recon_tmpl.c mc() emu_edge). ok is false only when the reference plane
// itself is unusable.
func mc8SourceGrid(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src unsafeutil.Grid[uint8], ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= 2
		bw += 5
	}
	if hy {
		y0 -= 2
		bh += 5
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < ref.Width {
		return src, false
	}
	sw := max(w+8, 16)
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		if unsafeutil.GridFit(len(ref.Pix), start, ref.Stride, sw, bh) {
			return unsafeutil.NewGrid(ref.Pix, start, ref.Stride, sw, bh), true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return src, false
	}
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the kernels
	// read at most max(w+8, 16) <= 136 bytes per row.
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	if !unsafeutil.GridFit(len(edge), 0, emuEdgeStride, sw, bh) {
		return src, false
	}
	return unsafeutil.NewGrid(edge[:], 0, emuEdgeStride, sw, bh), true
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
	if eight {
		return mc8Put8Tap(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window, in place when it lies inside the plane:
	// halo 2 before and 3 after each sample on a filtered axis. The common
	// case, destination and source window both in their planes, is checked
	// in line and runs the kernel with no call before it.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, sw := dstY*dst.Stride+dstX, y0*ref.Stride+x0, max(w+8, 16)
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < ref.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		return mc8PutEdge(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	dg, sg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	// mc8PutGrid written out: one call level less for the common kernels.
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX != 0 && subY != 0:
		im := &scratch.im
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PutV6MidW4(im, phy, dg)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PutV6Mid(im, phy, dg)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PutH4W4(sg, phx, dg)
		} else {
			mc8AVX2PutH6(sg, phx, dg)
		}
	default:
		if w == 4 {
			mc8AVX2PutV6W4(sg, phy, &scratch.im, dg)
		} else {
			mc8AVX2PutV6(sg, phy, dg)
		}
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// mc8PutEdge is Put8's six-tap path for a block whose destination or source
// window is not inside its plane (or whose views do not fit): the checked
// destination view and the source window from mc8SourceGrid, which clamps it.
func mc8PutEdge(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dOff := dstY*dst.Stride + dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) {
		return false
	}
	sg, ok := mc8SourceGrid(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PutGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8PutGrid runs the six-tap put kernel of a validated block (w 4 or a
// multiple of 8, a nonzero phase on some axis): dg is the w x h destination,
// sg the source window. Like dav1d's RET (VZEROUPPER) it clears the upper
// YMM halves before returning.
func mc8PutGrid(dg, sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *ConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		im := &scratch.im
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PutV6MidW4(im, phy, dg)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PutV6Mid(im, phy, dg)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PutH4W4(sg, phx, dg)
		} else {
			mc8AVX2PutH6(sg, phx, dg)
		}
	default:
		if w == 4 {
			mc8AVX2PutV6W4(sg, phy, &scratch.im, dg)
		} else {
			mc8AVX2PutV6(sg, phy, dg)
		}
	}
	archsimd.ClearAVXUpperBits()
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
	o := out[:w*h]
	if eight {
		return mc8Prep8Tap(o, ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	sOff, sw := y0*ref.Stride+x0, max(w+8, 16)
	var sg unsafeutil.Grid[uint8]
	if x0 >= 0 && y0 >= 0 && x0+w+hx+hx+hx>>1 <= ref.Width && y0+bh <= ref.Height && ref.Stride >= ref.Width &&
		unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		sg = unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	} else if sg, ok = mc8SourceGrid(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge); !ok {
		return false
	}
	im := (*mc8IM)(&scratch.im8)
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX|subY == 0:
		mc8AVX2PrepCopy(sg, w, h, o)
	case subX != 0 && subY != 0:
		if w == 4 {
			mc8AVX2MidH4W4(sg, h+5, phx, im)
			mc8AVX2PrepV6MidW4(im, h, phy, o)
		} else {
			mc8AVX2H6Mid(sg, w, h+5, phx, im)
			mc8AVX2PrepV6Mid(im, w, h, phy, o)
		}
	case subX != 0:
		if w == 4 {
			mc8AVX2PrepH4W4(sg, h, phx, o)
		} else {
			mc8AVX2PrepH6(sg, w, h, phx, o)
		}
	default:
		if w == 4 {
			mc8AVX2PrepV6W4(sg, h, phy, im, o)
		} else {
			mc8AVX2PrepV6(sg, w, h, phy, o)
		}
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// mc8AVX2PutH6 is the put horizontal-only pass, clip((s + 34) >> 6): sg
// holds the h = dg.Height() source rows (max(w+8, 16) bytes each), dg the
// destination block.
func mc8AVX2PutH6(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().GetElem(0))
			binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().GetElem(0))
		}
		if r < h {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], mc8Pack(a, a, zero, c255, deint).ReshapeToUint64s().GetLo().GetElem(0))
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) || !dg.Fits(0, c, 16, h) {
			return
		}
		s, d := sg.Sub(0, c, 24, h), dg.Sub(0, c, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			x0, x1 := mc8G24(s, r), mc8G24(s, r+1)
			a := mc8H6(mc8Ld2((*[16]byte)(x0[0:16]), (*[16]byte)(x1[0:16])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			b := mc8H6(mc8Ld2((*[16]byte)(x0[8:24]), (*[16]byte)(x1[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			v := mc8Pack(a, b, zero, c255, deint)
			v.GetLo().StoreArray(mc8G16(d, r))
			v.GetHi().StoreArray(mc8G16(d, r+1))
		}
		if r < h {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[0:16])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			b := mc8H6(mc8Ld2((*[16]byte)(x[8:24]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(6)
			mc8Pack(a, b, zero, c255, deint).GetLo().StoreArray(mc8G16(d, r))
		}
	}
	if c < w && sg.Fits(0, c, 16, h) && dg.Fits(0, c, 8, h) {
		// The last eight columns of a 16k+8-wide block (no AV1 block is).
		mc8AVX2PutH6(sg.Sub(0, c, 16, h), ph, dg.Sub(0, c, 8, h))
	}
}

// mc8AVX2PrepH6 is the prep horizontal-only pass into int16 rows of stride w:
// ((s + 2 + 6144 << 2) >> 2) as unsigned.
func mc8AVX2PrepH6(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		r := 0
		for ; r < h-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
		}
		if r < h {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) {
			return
		}
		s := sg.Sub(0, c, 24, h)
		for r := 0; r < h; r++ {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*w+c, a.ToBits().ShiftAllRight(2))
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, h) {
			return
		}
		s := sg.Sub(0, c, 16, h)
		for r := 0; r < h; r++ {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*w+c, a.ToBits().ShiftAllRight(2).GetLo())
		}
	}
}

// mc8AVX2H6Mid writes rows x w rounded horizontal sums m = (s + 2) >> 2 as
// int16 with row stride w (the 2D intermediate) from the rows of sg.
func mc8AVX2H6Mid(sg unsafeutil.Grid[uint8], w, rows int, ph *mc8Phase, im *mc8IM) {
	if uint(rows-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	sA := archsimd.LoadInt8x32Array(&mc8ShufA)
	sB := archsimd.LoadInt8x32Array(&mc8ShufB)
	sC := archsimd.LoadInt8x32Array(&mc8ShufC)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, rows) {
			return
		}
		s := sg.Sub(0, 0, 16, rows)
		r := 0
		for ; r < rows-1; r += 2 {
			a := mc8H6(mc8Ld2(mc8G16(s, r), mc8G16(s, r+1)), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*8, a)
		}
		if r < rows {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*8, a.GetLo())
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, rows) {
			return
		}
		s := sg.Sub(0, c, 24, rows)
		for r := 0; r < rows; r++ {
			x := mc8G24(s, r)
			a := mc8H6(mc8Ld2((*[16]byte)(x[0:16]), (*[16]byte)(x[8:24])), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16(im, r*w+c, a)
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, rows) {
			return
		}
		s := sg.Sub(0, c, 16, rows)
		for r := 0; r < rows; r++ {
			x := mc8G16(s, r)
			a := mc8H6(mc8Ld2(x, x), sA, sB, sC, t0, t1, t2).Add(rnd).ShiftAllRight(2)
			mc8StI16x8(im, r*w+c, a.GetLo())
		}
	}
}

// mc8AVX2PutV6 is the put vertical-only pass: clip((s + 32) >> 6). sg holds
// the h+5 source rows, dg the h destination rows.
//
// The kernels wider than 8 run 16-column strips; when w%16 == 8 the last
// strip starts at w-16 and rewrites eight columns of the previous one with
// the same values, so every strip stores 16 columns per row.
func mc8AVX2PutV6(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if w == 8 {
		mc8AVX2PutV6W8(sg, ph, dg)
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !sg.Fits(0, c, 16, h+5) || !dg.Fits(0, c, 16, h) {
			return
		}
		mc8AVX2PutV6Strip(sg.Sub(0, c, 16, h+5), ph, dg.Sub(0, c, 16, h))
	}
}

// mc8AVX2PutV6Strip is one 16-column strip of mc8AVX2PutV6: s holds the h+5
// source rows of d's h rows. The five leading rows are rows 0..4 of s, the
// rest the rows of t (s from row 5), indexed by the output row.
func mc8AVX2PutV6Strip(s unsafeutil.Grid[uint8], ph *mc8Phase, d unsafeutil.Grid[uint8]) {
	h := d.Height()
	if uint(h-1) >= mc8MaxRows || !s.Fits(0, 0, 16, h+5) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := archsimd.LoadUint8x16Array(mc8G16(s, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(s, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(s, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(s, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(s, 4)).ExtendToUint16()
	t := s.Sub(5, 0, 16, h)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		p4 := mc8Pair(z4, z5)
		p5 := mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, b, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
		v.GetLo().StoreArray(mc8G16(d, r))
		v.GetHi().StoreArray(mc8G16(d, r+1))
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ShiftAllRight(6)
		mc8Pack(a, a, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s().GetLo().StoreArray(mc8G16(d, r))
	}
}

// mc8AVX2PutV6W8 runs the eight-column vertical pass two rows per vector (row
// r in the low lane, r+1 in the high lane).
func mc8AVX2PutV6W8(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h+5) || !dg.Fits(0, 0, 8, h) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	rnd := archsimd.BroadcastInt16x16(32)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	z0 := archsimd.LoadUint8x16Array(mc8G16(sg, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(sg, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(sg, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(sg, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(sg, 4)).ExtendToUint16()
	t := sg.Sub(5, 0, 16, h)
	q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
	q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
		a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().ReshapeToUint64s().GetElem(0))
		binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().ReshapeToUint64s().GetElem(0))
		q0, q2, z4 = q2, q4, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z5))
		a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], mc8Pack(a, a, zero, c255, deint).GetLo().ReshapeToUint64s().GetElem(0))
	}
}

// mc8AVX2PrepV6 is the prep vertical-only pass, ((s + 2) >> 2) + 6144, from
// the h+5 rows of sg into the w x h CONV_BUF out.
func mc8AVX2PrepV6(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	if w == 8 {
		if !sg.Fits(0, 0, 16, h+5) {
			return
		}
		z0 := archsimd.LoadUint8x16Array(mc8G16(sg, 0)).ExtendToUint16()
		z1 := archsimd.LoadUint8x16Array(mc8G16(sg, 1)).ExtendToUint16()
		z2 := archsimd.LoadUint8x16Array(mc8G16(sg, 2)).ExtendToUint16()
		z3 := archsimd.LoadUint8x16Array(mc8G16(sg, 3)).ExtendToUint16()
		z4 := archsimd.LoadUint8x16Array(mc8G16(sg, 4)).ExtendToUint16()
		t := sg.Sub(5, 0, 16, h)
		q0 := mc8Join(mc8Pair(z0, z1), mc8Pair(z1, z2))
		q2 := mc8Join(mc8Pair(z2, z3), mc8Pair(z3, z4))
		r := 0
		for ; r < h-1; r += 2 {
			z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
			z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z6))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16(out, r*8, a.ToBits().ShiftAllRight(2))
			q0, q2, z4 = q2, q4, z6
		}
		if r < h {
			z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
			q4 := mc8Join(mc8Pair(z4, z5), mc8Pair(z5, z5))
			a := mc8V6(q0, q2, q4, t0, t1, t2).Add(rnd)
			mc8StU16x8(out, r*8, a.ToBits().ShiftAllRight(2).GetLo())
		}
		return
	}
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !sg.Fits(0, c, 16, h+5) {
			return
		}
		mc8AVX2PrepV6Strip(sg.Sub(0, c, 16, h+5), w, h, ph, out[c:])
	}
}

// mc8AVX2PrepV6Strip is one 16-column strip of mc8AVX2PrepV6 (out row
// stride w), two rows per step.
func mc8AVX2PrepV6Strip(s unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, oc []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !s.Fits(0, 0, 16, h+5) {
		return
	}
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	z0 := archsimd.LoadUint8x16Array(mc8G16(s, 0)).ExtendToUint16()
	z1 := archsimd.LoadUint8x16Array(mc8G16(s, 1)).ExtendToUint16()
	z2 := archsimd.LoadUint8x16Array(mc8G16(s, 2)).ExtendToUint16()
	z3 := archsimd.LoadUint8x16Array(mc8G16(s, 3)).ExtendToUint16()
	z4 := archsimd.LoadUint8x16Array(mc8G16(s, 4)).ExtendToUint16()
	t := s.Sub(5, 0, 16, h)
	p0, p1, p2, p3 := mc8Pair(z0, z1), mc8Pair(z1, z2), mc8Pair(z2, z3), mc8Pair(z3, z4)
	r := 0
	for ; r < h-1; r += 2 {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		z6 := archsimd.LoadUint8x16Array(mc8G16(t, r+1)).ExtendToUint16()
		p4, p5 := mc8Pair(z4, z5), mc8Pair(z5, z6)
		a := mc8V6(p0, p2, p4, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		b := mc8V6(p1, p3, p5, t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		win := oc[uint(r*w) : uint(r*w)+uint(w)+16]
		a.StoreArray((*[16]uint16)(win))
		b.StoreArray((*[16]uint16)(win[w:]))
		p0, p1, p2, p3, z4 = p2, p3, p4, p5, z6
	}
	if r < h {
		z5 := archsimd.LoadUint8x16Array(mc8G16(t, r)).ExtendToUint16()
		a := mc8V6(p0, p2, mc8Pair(z4, z5), t0, t1, t2).Add(rnd).ToBits().ShiftAllRight(2)
		mc8StU16(oc, r*w, a)
	}
}

// mc8AVX2PutV6Mid runs the put 2D vertical pass over the int16 intermediate
// (row stride w = dg.Width(), h+5 rows): clip((sum(half tap * m) + 512) >> 10)
// into dg.
//
// For w == 8 a 32-byte load at row k holds rows k and k+1, so the per-lane
// word interleave of rows (k, k+1) gives row pairs (k, k+1) in the low lane and
// (k+1, k+2) in the high lane: output rows r and r+1 come out one per lane.
// Wider blocks run 16-column strips one row at a time, where the per-lane
// interleave and VPACKUSDW undo each other's lane split.
func mc8AVX2PutV6Mid(im *mc8IM, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h, w := dg.Height(), dg.Width()
	if uint(h-1) >= mc8MaxRows {
		return
	}
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
		r := 0
		for ; r < h; r += 2 {
			q := 32 + r*8
			l4 := mc8LdI16(im, q)
			l5 := mc8LdI16(im, q+8)
			a4, b4 := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
			lo := mc8VMid(a0, a2, a4, d0, d1, d2, rnd).ShiftAllRight(10)
			hi := mc8VMid(b0, b2, b4, d0, d1, d2, rnd).ShiftAllRight(10)
			v := lo.SaturateToUint16ConcatGrouped(hi).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			binary.LittleEndian.PutUint64(mc8G8(dg, r)[:], v.GetLo().ReshapeToUint64s().GetElem(0))
			if r+1 < h {
				binary.LittleEndian.PutUint64(mc8G8(dg, r+1)[:], v.GetHi().ReshapeToUint64s().GetElem(0))
			}
			a0, b0, a2, b2 = a2, b2, a4, b4
		}
		return
	}
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	for c0 := 0; c0 < w; c0 += 16 {
		c := max(min(c0, w-16), 0)
		if !dg.Fits(0, c, 16, h) {
			return
		}
		dc := dg.Sub(0, c, 16, h)
		l0 := mc8LdI16(im, c)
		l1 := mc8LdI16(im, c+w)
		l2 := mc8LdI16(im, c+2*w)
		l3 := mc8LdI16(im, c+3*w)
		l4 := mc8LdI16(im, c+4*w)
		a0, b0 := l0.InterleaveLoGrouped(l1), l0.InterleaveHiGrouped(l1)
		a1, b1 := l1.InterleaveLoGrouped(l2), l1.InterleaveHiGrouped(l2)
		a2, b2 := l2.InterleaveLoGrouped(l3), l2.InterleaveHiGrouped(l3)
		a3, b3 := l3.InterleaveLoGrouped(l4), l3.InterleaveHiGrouped(l4)
		for r := 0; r < h; r += 2 {
			q := c + (r+5)*w
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
			v.GetLo().StoreArray(mc8G16(dc, r))
			if r+1 < h {
				v.GetHi().StoreArray(mc8G16(dc, r+1))
			}
			a0, b0, a1, b1, a2, b2, a3, b3, l4 = a2, b2, a3, b3, a4, b4, a5, b5, l6
		}
	}
}

// Width-4 kernels (dav1d put_6tap/prep_6tap .h_w4, .v_w4, .hv_w4). Each
// 256-bit vector holds four rows of four samples: rows r, r+1 in the low lane
// and r+2, r+3 in the high lane. The source view's rows are sixteen bytes
// (max(w+8, 16)); the four-tap horizontal pass reads bytes 1..8 of each.

// mc8H4Row is the eight bytes at window tap 1 of row r of s.
func mc8H4Row(s unsafeutil.Grid[uint8], r int) uint64 {
	return binary.LittleEndian.Uint64(mc8G16(s, r)[1:9])
}

// mc8StPutW4G stores the rows r..r+3 (r+k < h only) of v to dg: dwords 0, 1
// of the low lane and 0, 1 of the high lane.
func mc8StPutW4G(dg unsafeutil.Grid[uint8], r, h int, v archsimd.Uint8x32) {
	u := v.ReshapeToUint32s()
	lo, hi := u.GetLo(), u.GetHi()
	le := binary.LittleEndian
	if r+4 <= h && dg.Fits(r, 0, 4, 4) { // a full group
		d := dg.Sub(r, 0, 4, 4)
		le.PutUint32(mc8G4(d, 0)[:], lo.GetElem(0))
		le.PutUint32(mc8G4(d, 1)[:], lo.GetElem(1))
		le.PutUint32(mc8G4(d, 2)[:], hi.GetElem(0))
		le.PutUint32(mc8G4(d, 3)[:], hi.GetElem(1))
		return
	}
	le.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
	if r+1 < h {
		le.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
	}
	if r+2 < h {
		le.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
	}
	if r+3 < h {
		le.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
	}
}

// mc8AVX2PutH4W4 is the w=4 put horizontal-only pass.
func mc8AVX2PutH4W4(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h) {
		return
	}
	s := sg.Sub(0, 0, 16, h)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(34)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	r := 0
	for ; r < h-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
	}
	if r < h {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, h-1)), mc8H4Row(s, min(r+2, h-1)), mc8H4Row(s, h-1), sA, sB, t0, t1).Add(rnd).ShiftAllRight(6)
		mc8StPutW4G(dg, r, h, mc8Pack(a, a, zero, c255, deint))
	}
}

// mc8AVX2PrepH4W4 is the w=4 prep horizontal-only pass.
func mc8AVX2PrepH4W4(sg unsafeutil.Grid[uint8], h int, ph *mc8Phase, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 16, h) {
		return
	}
	s := sg.Sub(0, 0, 16, h)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	r := 0
	for ; r < h-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd)
		mc8StU16(out, r*4, a.ToBits().ShiftAllRight(2))
	}
	if r < h {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, h-1)), mc8H4Row(s, min(r+2, h-1)), mc8H4Row(s, h-1), sA, sB, t0, t1).Add(rnd)
		mc8StPrepW4(out, r, h, a.ToBits().ShiftAllRight(2))
	}
}

// mc8AVX2MidH4W4 writes the w=4 2D intermediate (row stride 4) from the rows
// rows of sg. The last group may write up to three rows past rows inside the
// scratch block.
func mc8AVX2MidH4W4(sg unsafeutil.Grid[uint8], rows int, ph *mc8Phase, im *mc8IM) {
	if uint(rows-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	if !sg.Fits(0, 0, 16, rows) {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	sA := archsimd.LoadInt8x32Array(&mc8Shuf4A)
	sB := archsimd.LoadInt8x32Array(&mc8Shuf4B)
	t0, t1 := mc8TapsB(ph.f[0]), mc8TapsB(ph.f[1])
	rnd := archsimd.BroadcastInt16x16(2)
	r := 0
	for ; r < rows-3; r += 4 {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, r+1), mc8H4Row(s, r+2), mc8H4Row(s, r+3), sA, sB, t0, t1).Add(rnd).ShiftAllRight(2)
		mc8StI16(im, r*4, a)
	}
	if r < rows {
		a := mc8H4x4Q(mc8H4Row(s, r), mc8H4Row(s, min(r+1, rows-1)), mc8H4Row(s, min(r+2, rows-1)), mc8H4Row(s, rows-1), sA, sB, t0, t1).Add(rnd).ShiftAllRight(2)
		mc8StI16(im, r*4, a)
	}
}

// mc8PackW4 packs the h+5 four-byte source rows of sg into im at stride 4
// bytes (two words per row), so a 16-byte load holds four consecutive rows.
func mc8PackW4(sg unsafeutil.Grid[uint8], h int, im *mc8IM) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	if !sg.Fits(0, 0, 4, h+5) {
		return
	}
	s := sg.Sub(0, 0, 4, h+5)
	for k := 0; k < h+5; k++ {
		v := binary.LittleEndian.Uint32(mc8G4(s, k)[:])
		p := (*[2]int16)(im[2*k : 2*k+2])
		p[0], p[1] = int16(v), int16(v>>16) // one 32-bit store
	}
}

// mc8AVX2PutV6W4 is the w=4 put vertical-only pass. The h+5 source rows are
// first packed into im (mc8PackW4); zero-extended, row k+1 shifted into the
// high bytes forms the vertical byte pairs for four output rows at once.
func mc8AVX2PutV6W4(sg unsafeutil.Grid[uint8], ph *mc8Phase, im *mc8IM, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(sg, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	rnd := archsimd.BroadcastInt16x16(32)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for r := 0; r < h; r += 4 {
		q := 2 * (r + 4)
		pc := mc8Pair(mc8ZxW4(im, q), mc8ZxW4(im, q+2))
		a := mc8V6(pa, pb, pc, t0, t1, t2).Add(rnd).ShiftAllRight(6)
		v := mc8Pack(a, a, zero, c255, deint)
		if r+4 > h {
			mc8StPutW4G(dg, r, h, v)
			return
		}
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
		pa = pc
		pb = mc8Pair(mc8ZxW4(im, q+4), mc8ZxW4(im, q+6))
	}
}

// mc8AVX2PrepV6W4 is the w=4 prep vertical-only pass, laid out as
// mc8AVX2PutV6W4.
func mc8AVX2PrepV6W4(sg unsafeutil.Grid[uint8], h int, ph *mc8Phase, im *mc8IM, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	_ = im[0] // one nil check, ahead of the loops
	mc8PackW4(sg, h, im)
	t0, t1, t2 := mc8TapsB(ph.b[0]), mc8TapsB(ph.b[1]), mc8TapsB(ph.b[2])
	rnd := archsimd.BroadcastInt16x16(2 + 6144<<2)
	pa := mc8Pair(mc8ZxW4(im, 0), mc8ZxW4(im, 2))
	pb := mc8Pair(mc8ZxW4(im, 4), mc8ZxW4(im, 6))
	for r := 0; r < h; r += 4 {
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
func mc8AVX2PutV6MidW4(im *mc8IM, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	h := dg.Height()
	if uint(h-1) >= mc8MaxRows {
		return
	}
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
	for r := 0; r < h; r += 4 {
		q := 4 * (r + 4)
		l4, l5 := mc8LdI16(im, q), mc8LdI16(im, q+4)
		pc, qc := l4.InterleaveLoGrouped(l5), l4.InterleaveHiGrouped(l5)
		x := mc8VMid(pa, pb, pc, d0, d1, d2, rnd).ShiftAllRight(6)
		y := mc8VMid(qa, qb, qc, d0, d1, d2, rnd).ShiftAllRight(6)
		u := x.SaturateToUint16ConcatGrouped(y)
		v := u.ShiftAllRight(4).Min(c255).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
		if r+4 > h {
			mc8StPutW4G(dg, r, h, v)
			return
		}
		u4 := v.ReshapeToUint32s()
		lo, hi := u4.GetLo(), u4.GetHi()
		binary.LittleEndian.PutUint32(mc8G4(dg, r)[:], lo.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+1)[:], lo.GetElem(1))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+2)[:], hi.GetElem(0))
		binary.LittleEndian.PutUint32(mc8G4(dg, r+3)[:], hi.GetElem(1))
		pa, qa = pc, qc
		l6, l7 := mc8LdI16(im, q+8), mc8LdI16(im, q+12)
		pb, qb = l6.InterleaveLoGrouped(l7), l6.InterleaveHiGrouped(l7)
	}
}

// mc8AVX2PrepCopy is the prep copy (dav1d prep_8bpc), 16 * px + 6144, from
// the h rows of sg into the w x h CONV_BUF out.
func mc8AVX2PrepCopy(sg unsafeutil.Grid[uint8], w, h int, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	off := archsimd.BroadcastUint16x16(6144)
	le := binary.LittleEndian
	switch w {
	case 4:
		if !sg.Fits(0, 0, 4, h) {
			return
		}
		s := sg.Sub(0, 0, 4, h)
		for r := 0; r < h; r += 4 {
			var z archsimd.Uint32x4
			v := z.SetElem(0, le.Uint32(mc8G4(s, r)[:])).SetElem(1, le.Uint32(mc8G4(s, min(r+1, h-1))[:])).
				SetElem(2, le.Uint32(mc8G4(s, min(r+2, h-1))[:])).SetElem(3, le.Uint32(mc8G4(s, min(r+3, h-1))[:]))
			mc8StPrepW4(out, r, h, v.ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off))
		}
	case 8:
		if !sg.Fits(0, 0, 8, h) {
			return
		}
		s := sg.Sub(0, 0, 8, h)
		r := 0
		for ; r < h-1; r += 2 {
			var z archsimd.Uint64x2
			u := z.SetElem(0, le.Uint64(mc8G8(s, r)[:])).SetElem(1, le.Uint64(mc8G8(s, r+1)[:])).ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off)
			mc8StU16(out, r*8, u)
		}
		if r < h {
			var z archsimd.Uint64x2
			u := z.SetElem(0, le.Uint64(mc8G8(s, r)[:])).ReshapeToUint8s().ExtendToUint16().ShiftAllLeft(4).Add(off)
			mc8StU16x8(out, r*8, u.GetLo())
		}
	default:
		c := 0
		for ; c+16 <= w; c += 16 {
			if !sg.Fits(0, c, 16, h) {
				return
			}
			s := sg.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				mc8StU16(out, r*w+c, archsimd.LoadUint8x16Array(mc8G16(s, r)).ExtendToUint16().ShiftAllLeft(4).Add(off))
			}
		}
		if c < w {
			if !sg.Fits(0, c, 16, h) {
				return
			}
			s := sg.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				mc8StU16x8(out, r*w+c, archsimd.LoadUint8x16Array(mc8G16(s, r)).ExtendToUint16().ShiftAllLeft(4).Add(off).GetLo())
			}
		}
	}
}

// AVX2 8-bit compound blends over two CONV_BUFs (dav1d mc_avx2.asm mask_8bpc,
// w_avg_8bpc and the w_mask difference mask), the vector forms of
// BlendCompoundMaskD16, BlendCompoundAvg's distance-weighted branch and
// BuildDiffWtdMaskD16 at 8 bits.
//
// Mask blend: libaom's clip(round4(((m*a + (64-m)*b) >> 6) - 6144)) equals
// (X - 392704) >> 10 with X = m*a + (64-m)*b (nested floor division). One
// VPMADDWD per four samples forms X from the word-interleaved (a, b) and
// (m, 64-m) pairs; a and b are biased by -32768 so any uint16 sample is exact
// as int16, which subtracts 64 * 32768 from X.
//
// Distance-weighted average: clip(round4(((a*fwd + b*bck) >> 4) - 6144)) is
// (S - 98176) >> 8 with S = a*fwd + b*bck, formed the same way.
//
// Difference mask: 38 + (round4(|a - b|) >> 4), capped at 64, is
// 38 + ((|a - b| >> 3) + 1) >> 5 by nested floor division.
//
// Each step works on 16 CONV_BUF samples: one row of 16 columns, two rows of
// 8 or four rows of 4.

const (
	mc8OpMask = iota
	mc8OpWAvg
)

// mc8BlendShapeOK reports whether the AVX2 blends cover a w x h block: rows of
// 16 or 32k columns, or whole groups of rows of 8 (pairs) or 4 (quads).
func mc8BlendShapeOK(w, h int) bool {
	switch {
	case w <= 0 || h <= 0 || w > maxBlockSize || h > maxBlockSize:
		return false
	case w == 4:
		return h&3 == 0
	case w == 8:
		return h&1 == 0
	default:
		return w == 16 || w&31 == 0
	}
}

// mc8BlendMaskFast blends with a full-resolution or 4:2:0 subsampled mask.
// It returns false for shapes it does not cover.
func mc8BlendMaskFast(dst frame.Plane, a, b []uint16, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if !mc8AVX2 || !mc8BlendShapeOK(w, h) || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	off := dstY*dst.Stride + dstX
	end := off + (h-1)*dst.Stride + w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	mEnd := (mh-1)*maskStride + mw
	if maskStride < mw || maskStride > mc8MaxStride || mEnd > len(mask) {
		return false
	}
	mc8AVX2Blend(dst.Pix[off:end], dst.Stride, a[:w*h], b[:w*h], w, h, mask[:mEnd], maskStride, subX, mc8OpMask, 0)
	archsimd.ClearAVXUpperBits()
	return true
}

// blendCompoundAvg8AVX2 is the blendCompoundAvg8Impl slot on AVX2 machines:
// the distance-weighted 8-bit average, with the general kernel for anything
// it does not cover.
func blendCompoundAvg8AVX2(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	off := dstY*dst.Stride + dstX
	end := off + (height-1)*dst.Stride + width
	if !mc8AVX2 || roundOffset != 6144 || roundBits != 4 || fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		!mc8BlendShapeOK(width, height) || len(src0) < width*height || len(src1) < width*height ||
		dstX < 0 || dstY < 0 || dstX+width > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride {
		blendCompoundAvg8GoSIMD(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	wts := int32(uint32(uint16(fwdOffset)) | uint32(uint16(bckOffset))<<16)
	mc8AVX2Blend(dst.Pix[off:end], dst.Stride, src0[:width*height], src1[:width*height], width, height, nil, 0, false, mc8OpWAvg, wts)
	archsimd.ClearAVXUpperBits()
}

func init() {
	if mc8AVX2 {
		blendCompoundAvg8Impl = blendCompoundAvg8AVX2
	}
}

// mc8Ld16U16 loads the 16 samples at s[o:].
func mc8Ld16U16(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(s[uint(o) : uint(o)+16]))
}

// The mask getters return the 16 mask values of one step, laid out like the
// samples. The 4:2:0 forms average 2x2 mask blocks ((sum + 2) >> 2) with
// VPMADDUBSW against ones, as dav1d's mask_420.

func mc8MaskW4(mask []byte, o, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint32x4
	v := z.SetElem(0, mc8Ld32(mask, o)).SetElem(1, mc8Ld32(mask, o+ms)).SetElem(2, mc8Ld32(mask, o+2*ms)).SetElem(3, mc8Ld32(mask, o+3*ms))
	return v.ReshapeToUint8s().ExtendToUint16()
}

func mc8MaskW8(mask []byte, o, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint64x2
	return z.SetElem(0, mc8Ld64(mask, o)).SetElem(1, mc8Ld64(mask, o+ms)).ReshapeToUint8s().ExtendToUint16()
}

// mc8Ld4x8 gathers the 8 bytes at o, o+2*ms, o+4*ms and o+6*ms.
func mc8Ld4x8(mask []byte, o, ms int) archsimd.Uint8x32 {
	var z2 archsimd.Uint64x2
	var z4 archsimd.Uint64x4
	lo := z2.SetElem(0, mc8Ld64(mask, o)).SetElem(1, mc8Ld64(mask, o+2*ms))
	hi := z2.SetElem(0, mc8Ld64(mask, o+4*ms)).SetElem(1, mc8Ld64(mask, o+6*ms))
	return z4.SetLo(lo).SetHi(hi).ReshapeToUint8s()
}

// mc8Mask16 returns the mask of the 16 columns at row r, column c.
func mc8Mask16(mask []byte, ms, r, c int, sub bool, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	if !sub {
		return mc8Zx(mask, r*ms+c)
	}
	p := 2 * (r*ms + c)
	return mc8Mask420(mc8Ld32B(mask, p), mc8Ld32B(mask, p+ms), ones, two)
}

func mc8Mask420(x, y archsimd.Uint8x32, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	s := x.DotProductPairsSaturated(ones).Add(y.DotProductPairsSaturated(ones))
	return s.Add(two).ShiftAllRight(2).ToBits()
}

func mc8Ld32B(s []byte, o int) archsimd.Uint8x32 {
	return archsimd.LoadUint8x32Array((*[32]uint8)(s[uint(o) : uint(o)+32]))
}

// mc8BlendStep blends the 16 samples at a[o:] and b[o:]: with mask m
// (mc8OpMask) or with the broadcast (fwd, bck) word pair wts (mc8OpWAvg).
func mc8BlendStep(a, b []uint16, o int, m archsimd.Uint16x16, op int, wts archsimd.Int16x16, sign, m64 archsimd.Uint16x16, kMask, kWAvg archsimd.Int32x8) archsimd.Int16x16 {
	x := mc8Ld16U16(a, o).Xor(sign).BitsToInt16()
	y := mc8Ld16U16(b, o).Xor(sign).BitsToInt16()
	lo, hi := x.InterleaveLoGrouped(y), x.InterleaveHiGrouped(y)
	if op == mc8OpWAvg {
		return lo.DotProductPairs(wts).Add(kWAvg).ShiftAllRight(8).SaturateToInt16ConcatGrouped(hi.DotProductPairs(wts).Add(kWAvg).ShiftAllRight(8))
	}
	mi := m64.Sub(m)
	ml := m.InterleaveLoGrouped(mi).BitsToInt16()
	mh := m.InterleaveHiGrouped(mi).BitsToInt16()
	return lo.DotProductPairs(ml).Add(kMask).ShiftAllRight(10).SaturateToInt16ConcatGrouped(hi.DotProductPairs(mh).Add(kMask).ShiftAllRight(10))
}

// mc8AVX2Blend runs the mask blend or the distance-weighted average over a
// block mc8BlendShapeOK accepts; a and b hold w*h samples and dst and mask
// the block's rows.
func mc8AVX2Blend(dst []byte, ds int, a, b []uint16, w, h int, mask []byte, ms int, sub bool, op int, wts int32) {
	mc8Dims(w, h, ms, ds)
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	sign := archsimd.BroadcastUint16x16(0x8000)
	m64 := archsimd.BroadcastUint16x16(64)
	kMask := archsimd.BroadcastInt32x8(64*32768 - 392704)
	kWAvg := archsimd.BroadcastInt32x8(32768*(int32(uint16(wts))+wts>>16) - 98176)
	wv := mc8TapsD(wts)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	var m archsimd.Uint16x16
	switch w {
	case 4, 8:
		rows := 2
		if w == 4 {
			rows = 4
		}
		for r := 0; r < h; r += rows {
			if op == mc8OpMask {
				switch {
				case !sub && w == 4:
					m = mc8MaskW4(mask, r*ms, ms)
				case !sub:
					m = mc8MaskW8(mask, r*ms, ms)
				case w == 4:
					p := 2 * r * ms
					m = mc8Mask420(mc8Ld4x8(mask, p, ms), mc8Ld4x8(mask, p+ms, ms), ones, two)
				default:
					p := 2 * r * ms
					m = mc8Mask420(mc8Ld2At(mask, p, p+2*ms), mc8Ld2At(mask, p+ms, p+3*ms), ones, two)
				}
			}
			u := mc8BlendStep(a, b, r*w, m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u, u, zero, c255, deint)
			if w == 4 {
				mc8StPutW4(dst, ds, r, h, v)
			} else {
				y0, y1 := mc8Rows2x8(dst, r*ds, ds)
				u := v.ReshapeToUint64s()
				binary.LittleEndian.PutUint64(y0[:], u.GetLo().GetElem(0))
				binary.LittleEndian.PutUint64(y1[:], u.GetHi().GetElem(0))
			}
		}
	case 16:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			r1 := min(r+1, h-1)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r, 0, sub, ones, two)
			}
			u0 := mc8BlendStep(a, b, 16*r, m, op, wv, sign, m64, kMask, kWAvg)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r1, 0, sub, ones, two)
			}
			u1 := mc8BlendStep(a, b, 16*r1, m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ds
			mc8St16(dst, d, v.GetLo())
			if r+1 < h {
				mc8St16(dst, d+ds, v.GetHi())
			}
		}
	default:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for r := 0; r < h; r++ {
			d := r * ds
			for uc := uint(0); uc < uint(w); uc += 32 {
				c := int(uc)
				o := r*w + c
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c, sub, ones, two)
				}
				u0 := mc8BlendStep(a, b, o, m, op, wv, sign, m64, kMask, kWAvg)
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c+16, sub, ones, two)
				}
				u1 := mc8BlendStep(a, b, o+16, m, op, wv, sign, m64, kMask, kWAvg)
				v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(dst[uint(d+c) : uint(d+c)+32]))
			}
		}
	}
}

// mc8DiffWtdMaskFast builds the 8-bit difference-weighted mask. It returns
// false for shapes it does not cover.
func mc8DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) bool {
	if !mc8AVX2 || !mc8BlendShapeOK(w, h) || len(a) < w*h || len(b) < w*h ||
		maskStride < w || maskStride > mc8MaxStride || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc8AVX2DiffWtd(mask[:(h-1)*maskStride+w], maskStride, a[:w*h], b[:w*h], w, h, invert)
	archsimd.ClearAVXUpperBits()
	return true
}

func mc8DiffStep(a, b []uint16, o int, invert bool, one, base, m64 archsimd.Uint16x16) archsimd.Uint16x16 {
	x, y := mc8Ld16U16(a, o), mc8Ld16U16(b, o)
	m := x.Max(y).Sub(x.Min(y)).ShiftAllRight(3).Add(one).ShiftAllRight(5).Add(base).Min(m64)
	if invert {
		m = m64.Sub(m)
	}
	return m
}

func mc8AVX2DiffWtd(mask []byte, ms int, a, b []uint16, w, h int, invert bool) {
	mc8Dims(w, h, 0, ms)
	one := archsimd.BroadcastUint16x16(1)
	base := archsimd.BroadcastUint16x16(38)
	m64 := archsimd.BroadcastUint16x16(64)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	switch w {
	case 4, 8:
		rows := 2
		if w == 4 {
			rows = 4
		}
		for r := 0; r < h; r += rows {
			v := mc8DiffStep(a, b, r*w, invert, one, base, m64).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			if w == 4 {
				mc8StPutW4(mask, ms, r, h, v)
			} else {
				y0, y1 := mc8Rows2x8(mask, r*ms, ms)
				u := v.ReshapeToUint64s()
				binary.LittleEndian.PutUint64(y0[:], u.GetLo().GetElem(0))
				binary.LittleEndian.PutUint64(y1[:], u.GetHi().GetElem(0))
			}
		}
	case 16:
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			r1 := min(r+1, h-1)
			x := mc8DiffStep(a, b, 16*r, invert, one, base, m64)
			y := mc8DiffStep(a, b, 16*r1, invert, one, base, m64)
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ms
			mc8St16(mask, d, v.GetLo())
			if r+1 < h {
				mc8St16(mask, d+ms, v.GetHi())
			}
		}
	default:
		for r := 0; r < h; r++ {
			d := r * ms
			for uc := uint(0); uc < uint(w); uc += 32 {
				c := int(uc)
				o := r*w + c
				x := mc8DiffStep(a, b, o, invert, one, base, m64)
				y := mc8DiffStep(a, b, o+16, invert, one, base, m64)
				v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(mask[uint(d+c) : uint(d+c)+32]))
			}
		}
	}
}

// AVX2 8-bit warped motion (dav1d src/x86/mc_avx2.asm warp_affine_8x8): the
// horizontal pass forms each output's eight-tap dot product with vpmaddwd
// on centred int16 samples and its own filter row (two outputs per ymm, one
// per 128-bit lane), and two vphaddd levels gather the eight outputs in
// order; the vertical pass pairs the int16 intermediate rows and multiplies
// them against the per-column filters, transposed into tap pairs. Both are
// exact in int32 and compute the scalar warpHorizontal8Resident and
// warpVertical8Full formulas.

// warp16Filter is warpedFilter widened to int16 with 256 rows, indexed as
// [i&255] (the routing preconditions keep every index inside the filter rows).
var warp16Filter [256][8]int16

func init() {
	for i := range warpedFilter {
		warp16Filter[i] = warpedFilter[i]
	}
}

var warpAVX2 = archsimd.X86.AVX2()

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if !warpAVX2 || reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 || !warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	warpHorizontal8AVX2(tmp, ref, ix4, sx4, iy4, alpha, beta)
	return sy4
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8AVX2(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8AVX2(dst, tmp, i, j, rowShift, colShift, baseSY, 0, delta)
}

// warpCompound8x8Dispatch has no AVX2 kernel yet: the scalar compound block
// runs.
func warpCompound8x8Dispatch(out []uint16, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	return false
}

// warpFilterPair loads the int16 filter rows of the phases a and b into the
// low and high 128-bit lanes.
func warpFilterPair(a, b int) archsimd.Int16x16 {
	return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array(&warp16Filter[a&255])).SetHi(archsimd.LoadInt16x8Array(&warp16Filter[b&255]))
}

// warpPhase is the filter row of a phase: 64 + round2(s, 10).
func warpPhase(s int) int { return 64 + ((s + 512) >> 10) }

var (
	// Dword selections of the widened 16-sample row: the low lane holds
	// samples 0..7 (outputs 0..3 read windows 0..10) and the high lane
	// samples 4..11 (outputs 4..7); the second vector continues each lane.
	warpWinLo = [8]uint32{0, 1, 2, 3, 2, 3, 4, 5}
	warpWinHi = [8]uint32{4, 5, 6, 7, 6, 7, 7, 7}
)

// warpHorizontal8AVX2 computes the resident 15x8 intermediate:
// tmp = (sum(s * c) + 2^14 + 4) >> 3, with s centred by -128 (every filter
// row sums to 128, so the centring removes 2^14 and the offset adds it back:
// the constant is 2^15 + 4).
func warpHorizontal8AVX2(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, alpha, beta int) {
	centre := archsimd.BroadcastUint8x16(128)
	bias := archsimd.BroadcastInt32x8(1<<15 + 4)
	winLo := archsimd.LoadUint32x8Array(&warpWinLo)
	winHi := archsimd.LoadUint32x8Array(&warpWinHi)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + ix4 - 7
		sx := sx4 + beta*(k+4)
		a := u8Bytes16(ref.Pix[rowByte:]).Xor(centre).AsInt8x16().ExtendToInt16().AsInt32x8()
		b := a.Permute(winLo).AsUint8x32()
		c := a.Permute(winHi).AsUint8x32()
		// Window j of each lane is b:c shifted by 2j bytes.
		w0 := b.AsInt16x16()
		w1 := c.ConcatShiftBytesRightGrouped(b, 2).AsInt16x16()
		w2 := c.ConcatShiftBytesRightGrouped(b, 4).AsInt16x16()
		w3 := c.ConcatShiftBytesRightGrouped(b, 6).AsInt16x16()
		p0 := w0.DotProductPairs(warpFilterPair(warpPhase(sx), warpPhase(sx+4*alpha)))
		p1 := w1.DotProductPairs(warpFilterPair(warpPhase(sx+alpha), warpPhase(sx+5*alpha)))
		p2 := w2.DotProductPairs(warpFilterPair(warpPhase(sx+2*alpha), warpPhase(sx+6*alpha)))
		p3 := w3.DotProductPairs(warpFilterPair(warpPhase(sx+3*alpha), warpPhase(sx+7*alpha)))
		// vphaddd twice: lane 0 ends as outputs 0..3, lane 1 as 4..7.
		sum := p0.ConcatAddPairsGrouped(p1).ConcatAddPairsGrouped(p2.ConcatAddPairsGrouped(p3))
		out := (k + 7) * warpedIntermediateColumns
		sum.Add(bias).ShiftAllRight(3).StoreArray((*[8]int32)(tmp[out : out+8]))
	}
	archsimd.ClearAVXUpperBits()
}

// warpByteIdx gathers the low byte of every dword into the low four bytes of
// its 128-bit lane.
var warpByteIdx = [32]int8{
	0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
}

// warpVertical8AVX2 computes the 8x8 vertical output block:
// clip(((2^19 + sum(c * tmp) + 2^10) >> 11) - 384). The intermediate fits
// int16, so the rows narrow exactly and pair up for vpmaddwd; the eight
// column filters of a row are transposed into four tap-pair vectors (with
// gamma = 0 every column shares one filter).
func warpVertical8AVX2(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x8(seed)
	var zero archsimd.Int32x8
	c255 := archsimd.BroadcastInt32x8(255)
	byteIdx := archsimd.LoadInt8x32Array(&warpByteIdx)
	// pairs[t]: the int16 rows t and t+1 interleaved, columns 0..3 in the low
	// lane and 4..7 in the high lane.
	var rows [15]archsimd.Int16x8
	for t := range rows {
		rows[t] = archsimd.LoadInt32x4Array((*[4]int32)(tmp[8*t : 8*t+4])).SaturateToInt16Concat(archsimd.LoadInt32x4Array((*[4]int32)(tmp[8*t+4 : 8*t+8])))
	}
	var pairs [14]archsimd.Int16x16
	for t := range pairs {
		pairs[t] = archsimd.Int16x16{}.SetLo(rows[t].InterleaveLo(rows[t+1])).SetHi(rows[t].InterleaveHi(rows[t+1]))
	}
	for row := 0; row < 8; row++ {
		sy := baseSY + delta*row
		// Columns c and c+4 share a ymm (low and high lane); a 4x4 dword
		// transpose per lane turns the filter rows into tap pairs.
		f0 := warpFilterPair(warpPhase(sy), warpPhase(sy+4*gamma)).AsInt32x8()
		f1 := warpFilterPair(warpPhase(sy+gamma), warpPhase(sy+5*gamma)).AsInt32x8()
		f2 := warpFilterPair(warpPhase(sy+2*gamma), warpPhase(sy+6*gamma)).AsInt32x8()
		f3 := warpFilterPair(warpPhase(sy+3*gamma), warpPhase(sy+7*gamma)).AsInt32x8()
		a0, a1 := f0.InterleaveLoGrouped(f1).AsInt64x4(), f0.InterleaveHiGrouped(f1).AsInt64x4()
		a2, a3 := f2.InterleaveLoGrouped(f3).AsInt64x4(), f2.InterleaveHiGrouped(f3).AsInt64x4()
		k01 := a0.InterleaveLoGrouped(a2).AsInt16x16()
		k23 := a0.InterleaveHiGrouped(a2).AsInt16x16()
		k45 := a1.InterleaveLoGrouped(a3).AsInt16x16()
		k67 := a1.InterleaveHiGrouped(a3).AsInt16x16()
		sum := bias.Add(pairs[row].DotProductPairs(k01)).Add(pairs[row+2].DotProductPairs(k23)).
			Add(pairs[row+4].DotProductPairs(k45)).Add(pairs[row+6].DotProductPairs(k67))
		v := sum.ShiftAllRight(round1Bits).Max(zero).Min(c255).AsUint8x32().PermuteOrZeroGrouped(byteIdx).AsUint32x8()
		px := uint64(v.GetLo().GetElem(0)) | uint64(v.GetHi().GetElem(0))<<32
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		binary.LittleEndian.PutUint64(dst.Pix[dstRow:dstRow+8], px)
	}
	archsimd.ClearAVXUpperBits()
}

// AVX2 10/12-bit warped motion (dav1d src/x86/mc16_avx2.asm
// warp_affine_8x8_16bpc). The horizontal pass is warpHorizontal8AVX2's on
// the samples centred by -2^15; the vertical
// intermediate does not fit int16 at these depths, so the vertical pass
// multiplies the int32 rows by per-column coefficient vectors (vpmulld),
// the eight column filters transposed into tap pairs and split into their
// sign-extended halves. Both compute the scalar formulas exactly.

func warpHorizontalHighBDResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	// The sixteen-sample loads read one sample past the resident window.
	if !warpAVX2 || !warpHorizResidentOffsInRange(sx4, alpha, beta) || ref.Stride&1 != 0 ||
		(iy4+7)*ref.Stride+(ix4+9)*2 > len(ref.Pix) {
		return warpHorizontalHighBDResident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	warpHorizontalHighBDAVX2(tmp, ref, ix4, sx4, iy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	return sy4
}

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDAVX2(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDAVX2(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, 0, delta, reduceBitsVert, offsetBitsVert)
}

// warpHighBDSampleCenterBias restores the 2^15 * 128 that centring the
// samples removes from each dot product.
const warpHighBDSampleCenterBias = 1 << 22

// warpHorizontalHighBDAVX2 computes the resident 15x8 intermediate:
// tmp = (2^offset + sum(s * c) + 2^(reduce-1)) >> reduce.
func warpHorizontalHighBDAVX2(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, alpha, beta, reduce, offset int) {
	// Samples are centred by -2^15 so every uint16 value is an exact int16;
	// each filter row sums to 128, so the centring removes 2^22, added back
	// in the bias.
	bias := archsimd.BroadcastInt32x8(int32(1)<<offset + int32(1)<<(reduce-1) + warpHighBDSampleCenterBias)
	centre := archsimd.BroadcastInt16x16(-1 << 15)
	sh := archsimd.BroadcastUint32x8(uint32(reduce))
	winLo := archsimd.LoadUint32x8Array(&warpWinLo)
	winHi := archsimd.LoadUint32x8Array(&warpWinHi)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + (ix4-7)*2
		a := archsimd.LoadUint8x32(ref.Pix[rowByte : rowByte+32]).AsInt16x16().Xor(centre).AsInt32x8()
		b := a.Permute(winLo).AsUint8x32()
		c := a.Permute(winHi).AsUint8x32()
		sx := sx4 + beta*(k+4)
		w0 := b.AsInt16x16()
		w1 := c.ConcatShiftBytesRightGrouped(b, 2).AsInt16x16()
		w2 := c.ConcatShiftBytesRightGrouped(b, 4).AsInt16x16()
		w3 := c.ConcatShiftBytesRightGrouped(b, 6).AsInt16x16()
		p0 := w0.DotProductPairs(warpFilterPair(warpPhase(sx), warpPhase(sx+4*alpha)))
		p1 := w1.DotProductPairs(warpFilterPair(warpPhase(sx+alpha), warpPhase(sx+5*alpha)))
		p2 := w2.DotProductPairs(warpFilterPair(warpPhase(sx+2*alpha), warpPhase(sx+6*alpha)))
		p3 := w3.DotProductPairs(warpFilterPair(warpPhase(sx+3*alpha), warpPhase(sx+7*alpha)))
		sum := p0.ConcatAddPairsGrouped(p1).ConcatAddPairsGrouped(p2.ConcatAddPairsGrouped(p3))
		out := (k + 7) * warpedIntermediateColumns
		sum.Add(bias).ShiftRight(sh).StoreArray((*[8]int32)(tmp[out : out+8]))
	}
	archsimd.ClearAVXUpperBits()
}

// warpVerticalHighBDAVX2 computes the 8x8 output block at (x0, y0):
// clip(((2^offset + sum(c * tmp) + 2^(reduce-1)) >> reduce) - 2^(bd-1) - 2^bd).
func warpVerticalHighBDAVX2(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, gamma, delta, reduce, offset int) {
	bias := archsimd.BroadcastInt32x8(int32(1)<<offset + int32(1)<<(reduce-1))
	sh := archsimd.BroadcastUint32x8(uint32(reduce))
	sub := archsimd.BroadcastInt32x8(int32(1)<<(bitDepth-1) + int32(1)<<bitDepth)
	var zero archsimd.Int32x8
	maxV := archsimd.BroadcastInt32x8(int32(max))
	for row := 0; row < 8; row++ {
		sy := baseSY + delta*row
		f0 := warpFilterPair(warpPhase(sy), warpPhase(sy+4*gamma)).AsInt32x8()
		f1 := warpFilterPair(warpPhase(sy+gamma), warpPhase(sy+5*gamma)).AsInt32x8()
		f2 := warpFilterPair(warpPhase(sy+2*gamma), warpPhase(sy+6*gamma)).AsInt32x8()
		f3 := warpFilterPair(warpPhase(sy+3*gamma), warpPhase(sy+7*gamma)).AsInt32x8()
		a0, a1 := f0.InterleaveLoGrouped(f1).AsInt64x4(), f0.InterleaveHiGrouped(f1).AsInt64x4()
		a2, a3 := f2.InterleaveLoGrouped(f3).AsInt64x4(), f2.InterleaveHiGrouped(f3).AsInt64x4()
		// Tap pairs (t, t+1) per column, columns 0..3 low lane, 4..7 high.
		k01 := a0.InterleaveLoGrouped(a2).AsInt32x8()
		k23 := a0.InterleaveHiGrouped(a2).AsInt32x8()
		k45 := a1.InterleaveLoGrouped(a3).AsInt32x8()
		k67 := a1.InterleaveHiGrouped(a3).AsInt32x8()
		// Column order of the coefficient vectors: c0..c3 | c4..c7, the
		// order of a tmp row.
		ld := func(t int) archsimd.Int32x8 {
			return archsimd.LoadInt32x8Array((*[8]int32)(tmp[(row+t)*8 : (row+t)*8+8]))
		}
		sum := bias.Add(ld(0).Mul(k01.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(1).Mul(k01.ShiftAllRight(16))).
			Add(ld(2).Mul(k23.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(3).Mul(k23.ShiftAllRight(16))).
			Add(ld(4).Mul(k45.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(5).Mul(k45.ShiftAllRight(16))).
			Add(ld(6).Mul(k67.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(7).Mul(k67.ShiftAllRight(16)))
		v := sum.ShiftRight(sh).Sub(sub).Max(zero).Min(maxV)
		p := v.SaturateToInt16ConcatGrouped(v)
		off := (y0+row)*dst.Stride + 2*x0
		d := dst.Pix[off : off+16]
		binary.LittleEndian.PutUint64(d[0:], p.GetLo().AsUint64x2().GetElem(0))
		binary.LittleEndian.PutUint64(d[8:], p.GetHi().AsUint64x2().GetElem(0))
	}
	archsimd.ClearAVXUpperBits()
}
