// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"encoding/binary"
	"math/bits"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// bitLenFloat32 computes bits.Len32 for lanes below 2^24 through the float32
// exponent: a value below 2^24 converts exactly, so its biased exponent is
// 127 + floor(log2 v) for v >= 1, and 0 for v = 0 (which the max clamps to 0).
func bitLenFloat32(v archsimd.Uint32x4, sh23, c126, zero32 archsimd.Int32x4) archsimd.Int32x4 {
	f := v.BitsToInt32().ConvertToFloat32()
	exp := encShrU32x4(f.ToBits(), sh23, 23).BitsToInt32()
	return exp.Sub(c126).Max(zero32)
}

// pixelstats_gosimd.go hosts the Go-native SIMD pixel-domain statistics
// (pixelStats*): sse = sum((src-ref)^2) and sum = sum(src-ref) over a block.
// Each 16-byte vector widens to int16 lanes, and |diff| <= 255 keeps every
// square (<= 65025) exact in int32. The per-lane int32 accumulators cannot
// overflow for blocks up to 64x64: the largest total is 4096*65025 < 2^31.
//
// Blocks 16 or more samples wide take 16 columns per vector. The 8-wide shape
// packs two rows into one vector, so both 8-byte rows fill the 128-bit lanes.
// Narrower shapes (4 wide) stay scalar, as an 8-byte row cannot be loaded
// without reading past it.

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	pixelStats8x8Impl = pixelStats8x8SIMD
	pixelStats8x16Impl = pixelStats8x16SIMD
	pixelStats16x8Impl = pixelStats16x8SIMD
	pixelStats16x16Impl = pixelStats16x16SIMD
	pixelStats16x32Impl = pixelStats16x32SIMD
	pixelStats32x16Impl = pixelStats32x16SIMD
	pixelStats32x32Impl = pixelStats32x32SIMD
	pixelStats32x64Impl = pixelStats32x64SIMD
	pixelStats64x32Impl = pixelStats64x32SIMD
}

// pixelDiffAcc folds one 16-sample pair of blocks into the running sse and sum
// int32 lane accumulators.
func pixelDiffAcc(sse, sum archsimd.Int32x4, a, b archsimd.Uint8x16) (archsimd.Int32x4, archsimd.Int32x4) {
	lo := a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16())
	hi := widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16())
	d0 := lo.ExtendLo4ToInt32()
	d1 := hiInt16(lo).ExtendLo4ToInt32()
	d2 := hi.ExtendLo4ToInt32()
	d3 := hiInt16(hi).ExtendLo4ToInt32()
	sse = sse.Add(d0.Mul(d0)).Add(d1.Mul(d1)).Add(d2.Mul(d2)).Add(d3.Mul(d3))
	sum = sum.Add(d0).Add(d1).Add(d2).Add(d3)
	return sse, sum
}

// reducePixelAcc returns the total sse and the total signed sum.
func reducePixelAcc(sse, sum archsimd.Int32x4) (total, signed int64) {
	total = int64(sse.GetElem(0)) + int64(sse.GetElem(1)) + int64(sse.GetElem(2)) + int64(sse.GetElem(3))
	signed = int64(sum.GetElem(0)) + int64(sum.GetElem(1)) + int64(sum.GetElem(2)) + int64(sum.GetElem(3))
	return total, signed
}

// pixelStatsRows16 handles w >= 16 with w a multiple of 16.
func pixelStatsRows16(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	return pixelStatsRows16Arch(src, srcStride, ref, refStride, w, h)
}

// pixelStatsPairs8 handles the 8-wide shape: rows r and r+1 share one vector.
func pixelStatsPairs8(src []byte, srcStride int, ref []byte, refStride int, h int) (sse uint32, sum int32) {
	sseAcc := archsimd.BroadcastInt32x4(0)
	sumAcc := archsimd.BroadcastInt32x4(0)
	r := 0
	for ; r+2 <= h; r += 2 {
		sseAcc, sumAcc = pixelDiffAcc(sseAcc, sumAcc, packRows8(src, r*srcStride, srcStride), packRows8(ref, r*refStride, refStride))
	}
	total, signed := reducePixelAcc(sseAcc, sumAcc)
	if r < h {
		t, sg := pixelStatsPureGo(src[r*srcStride:], srcStride, ref[r*refStride:], refStride, 8, 1)
		total += int64(t)
		signed += int64(sg)
	}
	return uint32(total), int32(signed)
}

// packRows8 loads the 8-byte rows at off and off+stride into one 16-byte vector.
func packRows8(p []byte, off, stride int) archsimd.Uint8x16 {
	lo := binary.LittleEndian.Uint64(p[off:])
	hi := binary.LittleEndian.Uint64(p[off+stride:])
	return archsimd.BroadcastUint64x2(lo).SetElem(1, hi).ReshapeToUint8s()
}

func pixelStatsSIMD(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	switch {
	case w == 8:
		return pixelStatsPairs8(src, srcStride, ref, refStride, h)
	case w >= 16 && w&15 == 0:
		return pixelStatsRows16(src, srcStride, ref, refStride, w, h)
	}
	return pixelStatsPureGo(src, srcStride, ref, refStride, w, h)
}

func pixelStats8x8SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 8, 8)
}

func pixelStats8x16SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats16x8SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats16x16SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats16x32SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats32x16SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats32x32SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats32x64SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 32, 64)
}

func pixelStats64x32SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 64, 32)
}

func init() {
	rdoqEOBImpl = rdoqEOBSIMD
	levelsAllZeroImpl = levelsAllZeroSIMD
}

// levelsAllZeroSIMD is levelsAllZeroPureGo sixteen levels at a time.
func levelsAllZeroSIMD(q []int16) bool {
	if len(q)&15 != 0 {
		return levelsAllZeroPureGo(q)
	}
	var a0, a1 archsimd.Int16x8
	for i := 0; i <= len(q)-16; i += 16 {
		w := (*[16]int16)(q[i:])
		a0 = a0.Or(archsimd.LoadInt16x8Array((*[8]int16)(w[0:8])))
		a1 = a1.Or(archsimd.LoadInt16x8Array((*[8]int16)(w[8:16])))
	}
	u := a0.Or(a1).ToBits().ReshapeToUint64s()
	return u.GetElem(0)|u.GetElem(1) == 0
}

// rdoqEOBSIMD is rdoqEOBPureGo eight levels at a time: the non-zero lanes
// select iscan+1 into a running maximum.
func rdoqEOBSIMD(qcoeff []int16, iscan []int16) int {
	n := len(qcoeff)
	if n&7 != 0 {
		return rdoqEOBPureGo(qcoeff, iscan)
	}
	iscan = iscan[:n]
	zero := archsimd.BroadcastInt16x8(0)
	one := archsimd.BroadcastInt16x8(1)
	acc := zero
	for i := 0; i <= len(qcoeff)-8 && i <= len(iscan)-8; i += 8 {
		q := archsimd.LoadInt16x8Array((*[8]int16)(qcoeff[i:]))
		s := archsimd.LoadInt16x8Array((*[8]int16)(iscan[i:]))
		acc = acc.Max(s.Add(one).Masked(q.NotEqual(zero)))
	}
	var lanes [8]int16
	acc.StoreArray(&lanes)
	eob := int16(0)
	for _, v := range lanes {
		eob = max(eob, v)
	}
	return int(eob)
}

// rdStatsDiffSIMD mirrors rdStatsDiffPureGo eight levels per step. A step
// whose levels are all zero contributes nothing and is skipped; otherwise
// dq*(2c - dq) is exact in int64 lanes (dq is zero on zero levels) and the
// rate lanes follow rdStatsBlockSIMD.
func rdStatsDiffSIMD(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (diff, rate int64, allZero bool) {
	stepV := archsimd.BroadcastInt32x4(step)
	tsV := archsimd.BroadcastInt32x4(-int32(ts))
	sh23 := archsimd.BroadcastInt32x4(-23)
	c126 := archsimd.BroadcastInt32x4(126)
	zero32 := archsimd.Int32x4{}
	one := archsimd.BroadcastUint32x4(1)
	diffAcc := archsimd.BroadcastInt64x2(0)
	rateAcc := archsimd.BroadcastInt32x4(0)
	allZero = true
	i := 0
	if count > 0 {
		t := tran[:count]
		qc := qcoeff[:count]
		for ; i <= len(qc)-8 && i <= len(t)-8; i += 8 {
			q := archsimd.LoadInt16x8Array((*[8]int16)(qc[i:]))
			u := q.ToBits().ReshapeToUint64s()
			if u.GetElem(0)|u.GetElem(1) == 0 {
				continue
			}
			allZero = false
			tw := (*[8]int32)(t[i:])
			c0 := archsimd.LoadInt32x4Array((*[4]int32)(tw[0:4]))
			c1 := archsimd.LoadInt32x4Array((*[4]int32)(tw[4:8]))
			dq0 := encSarI32x4(q.ExtendLo4ToInt32().Mul(stepV), tsV, uint64(ts))
			dq1 := encSarI32x4(hiInt16(q).ExtendLo4ToInt32().Mul(stepV), tsV, uint64(ts))
			diffAcc = mulSumInt32(mulSumInt32(diffAcc, dq0, c0.Add(c0).Sub(dq0)), dq1, c1.Add(c1).Sub(dq1))
			qAbs := q.Abs()
			lo := qAbs.ToBits().ExtendLo4ToUint32()
			hiMag := hiInt16(qAbs).ToBits().ExtendLo4ToUint32()
			nzLo := lo.Min(one).BitsToInt32()
			nzHi := hiMag.Min(one).BitsToInt32()
			rateAcc = rateAcc.Add(bitLenFloat32(lo, sh23, c126, zero32)).Add(bitLenFloat32(hiMag, sh23, c126, zero32)).
				Add(nzLo.Add(nzLo)).Add(nzHi.Add(nzHi))
		}
	}
	var lanes int64
	for ; i < count; i++ {
		v := qcoeff[i]
		if v == 0 {
			continue
		}
		allZero = false
		c := int64(tran[i])
		dq := (int64(v) * int64(step)) >> ts
		diff += dq * (2*c - dq)
		level := v
		if level < 0 {
			level = -level
		}
		lanes += int64(2 + bits.Len16(uint16(level)))
	}
	for l := range uint8(4) {
		lanes += int64(rateAcc.GetElem(l))
	}
	diff += diffAcc.GetElem(0) + diffAcc.GetElem(1)
	return diff, lanes << 9, allZero
}

// rdstats_gosimd.go hosts Go-native SIMD (simd/archsimd) forms of the encoder's
// residual extraction and skip-decision RD statistics.
// They are exact against the scalar references (residualBlockPureGo,
// rdStatsBlockPureGo): products that can exceed 32 bits widen
// to int64 before they are accumulated, and the int16 narrowing that SVT's
// load_tran_low_to_s16q applies is reproduced with a shift pair on int32 lanes.
// The lane moves, widening squares and bit lengths are shared helpers with
// arch-specific forms (gosimd_ops_*.go, gosimd_bitlen.go).

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	residualBlockImpl = residualBlockSIMD
	rdStatsBlockImpl = rdStatsBlockSIMD
	rdStatsDiffImpl = rdStatsDiffSIMD
}

// residualBlockSIMD extracts dst[r*w+c] = src - pred with one kernel per block
// width, as svt_residual_kernel8bit_neon dispatches: each row is taken as a
// fixed-size array once, so the vector loads and stores inside it carry no
// bounds checks. Eight-wide rows pair two rows per vector; other widths are
// scalar.
func residualBlockSIMD(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	if h <= 0 || w <= 0 {
		return
	}
	switch w {
	case 8:
		if h&1 == 0 {
			residualRows8(dst, srcPlane, srcOff, stride, pred, predStride, h)
			return
		}
	case 16:
		if unsafeutil.GridFit(len(dst), 0, 16, 16, h) &&
			unsafeutil.GridFit(len(srcPlane), srcOff, stride, 16, h) &&
			unsafeutil.GridFit(len(pred), 0, predStride, 16, h) {
			gd := unsafeutil.NewGrid(dst, 0, 16, 16, h)
			gs := unsafeutil.NewGrid(srcPlane, srcOff, stride, 16, h)
			gp := unsafeutil.NewGrid(pred, 0, predStride, 16, h)
			for r := 0; r < gd.Height(); r++ {
				residualRow16(gd.Row16(r, 0), gs.Row16(r, 0), gp.Row16(r, 0))
			}
			return
		}
		for r := range h {
			residualRow16((*[16]int16)(dst[r*16:]), (*[16]uint8)(srcPlane[srcOff+r*stride:]), (*[16]uint8)(pred[r*predStride:]))
		}
		return
	case 32:
		if unsafeutil.GridFit(len(dst), 0, 32, 32, h) &&
			unsafeutil.GridFit(len(srcPlane), srcOff, stride, 32, h) &&
			unsafeutil.GridFit(len(pred), 0, predStride, 32, h) {
			gd := unsafeutil.NewGrid(dst, 0, 32, 32, h)
			gs := unsafeutil.NewGrid(srcPlane, srcOff, stride, 32, h)
			gp := unsafeutil.NewGrid(pred, 0, predStride, 32, h)
			for r := 0; r < gd.Height(); r++ {
				d, s, p := gd.Row32(r, 0), gs.Row32(r, 0), gp.Row32(r, 0)
				residualRow16((*[16]int16)(d[:16]), (*[16]uint8)(s[:16]), (*[16]uint8)(p[:16]))
				residualRow16((*[16]int16)(d[16:]), (*[16]uint8)(s[16:]), (*[16]uint8)(p[16:]))
			}
			return
		}
		for r := range h {
			d := (*[32]int16)(dst[r*32:])
			s := (*[32]uint8)(srcPlane[srcOff+r*stride:])
			p := (*[32]uint8)(pred[r*predStride:])
			residualRow16((*[16]int16)(d[:16]), (*[16]uint8)(s[:16]), (*[16]uint8)(p[:16]))
			residualRow16((*[16]int16)(d[16:]), (*[16]uint8)(s[16:]), (*[16]uint8)(p[16:]))
		}
		return
	case 64:
		if unsafeutil.GridFit(len(dst), 0, 64, 64, h) &&
			unsafeutil.GridFit(len(srcPlane), srcOff, stride, 64, h) &&
			unsafeutil.GridFit(len(pred), 0, predStride, 64, h) {
			gd := unsafeutil.NewGrid(dst, 0, 64, 64, h)
			gs := unsafeutil.NewGrid(srcPlane, srcOff, stride, 64, h)
			gp := unsafeutil.NewGrid(pred, 0, predStride, 64, h)
			for r := 0; r < gd.Height(); r++ {
				d, s, p := gd.Row64(r, 0), gs.Row64(r, 0), gp.Row64(r, 0)
				residualRow16((*[16]int16)(d[:16]), (*[16]uint8)(s[:16]), (*[16]uint8)(p[:16]))
				residualRow16((*[16]int16)(d[16:32]), (*[16]uint8)(s[16:32]), (*[16]uint8)(p[16:32]))
				residualRow16((*[16]int16)(d[32:48]), (*[16]uint8)(s[32:48]), (*[16]uint8)(p[32:48]))
				residualRow16((*[16]int16)(d[48:]), (*[16]uint8)(s[48:]), (*[16]uint8)(p[48:]))
			}
			return
		}
		for r := range h {
			d := (*[64]int16)(dst[r*64:])
			s := (*[64]uint8)(srcPlane[srcOff+r*stride:])
			p := (*[64]uint8)(pred[r*predStride:])
			residualRow16((*[16]int16)(d[:16]), (*[16]uint8)(s[:16]), (*[16]uint8)(p[:16]))
			residualRow16((*[16]int16)(d[16:32]), (*[16]uint8)(s[16:32]), (*[16]uint8)(p[16:32]))
			residualRow16((*[16]int16)(d[32:48]), (*[16]uint8)(s[32:48]), (*[16]uint8)(p[32:48]))
			residualRow16((*[16]int16)(d[48:]), (*[16]uint8)(s[48:]), (*[16]uint8)(p[48:]))
		}
		return
	}
	if w < 8 {
		residualBlockPureGo(dst, srcPlane, srcOff, stride, pred, predStride, w, h)
		return
	}
	for r := 0; r < h; r++ {
		srow := srcPlane[srcOff+r*stride:]
		prow := pred[r*predStride:]
		drow := dst[r*w : r*w+w]
		c := 0
		for ; c+16 <= w; c += 16 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(srow[c : c+16]))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(prow[c : c+16]))
			a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).StoreArray((*[8]int16)(drow[c : c+8]))
			widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).StoreArray((*[8]int16)(drow[c+8 : c+16]))
		}
		for ; c < w; c++ {
			drow[c] = int16(srow[c]) - int16(prow[c])
		}
	}
}

// residualRow16 is one sixteen-sample residual row.
func residualRow16(d *[16]int16, s, p *[16]uint8) {
	a := archsimd.LoadUint8x16Array(s)
	b := archsimd.LoadUint8x16Array(p)
	a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).StoreArray((*[8]int16)(d[:8]))
	widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).StoreArray((*[8]int16)(d[8:]))
}

// residualRows8 is the eight-wide kernel over an even row count, two rows per
// vector. A contiguous prediction (stride 8) loads both rows at once; a source
// row pair with 16 readable bytes from each row start loads two vectors and
// joins their low halves, which keeps the rows out of general registers.
func residualRows8(dst []int16, src []byte, off, stride int, pred []byte, predStride, h int) {
	if h <= 0 {
		return
	}
	if predStride == 8 && off >= 0 && stride >= 0 && off+(h-1)*stride+16 <= len(src) && h*8 <= len(pred) && h*8 <= len(dst) {
		s := src[off : off+(h-1)*stride+16]
		p := pred[:h*8]
		d := dst[:h*8]
		for r := 0; r < h; r += 2 {
			so := r * stride
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(s[so:]))
			a1 := archsimd.LoadUint8x16Array((*[16]uint8)(s[so+stride:]))
			a := a0.ReshapeToUint64s().InterleaveLo(a1.ReshapeToUint64s()).ReshapeToUint8s()
			b := archsimd.LoadUint8x16Array((*[16]uint8)(p[r*8:]))
			dr := (*[16]int16)(d[r*8:])
			a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).StoreArray((*[8]int16)(dr[:8]))
			widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).StoreArray((*[8]int16)(dr[8:]))
		}
		return
	}
	for r := 0; r < h; r += 2 {
		so := off + r*stride
		var a, b archsimd.Uint8x16
		if so+stride+16 <= len(src) {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(src[so:]))
			a1 := archsimd.LoadUint8x16Array((*[16]uint8)(src[so+stride:]))
			a = a0.ReshapeToUint64s().InterleaveLo(a1.ReshapeToUint64s()).ReshapeToUint8s()
		} else {
			a = packRows8(src, so, stride)
		}
		if predStride == 8 {
			b = archsimd.LoadUint8x16Array((*[16]uint8)(pred[r*8:]))
		} else {
			b = packRows8(pred, r*predStride, predStride)
		}
		d := (*[16]int16)(dst[r*8:])
		a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).StoreArray((*[8]int16)(d[:8]))
		widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).StoreArray((*[8]int16)(d[8:]))
	}
}

// rdStatsBlockSIMD mirrors rdStatsBlockPureGo. For a zero level the AC-step
// dequant is zero, so e = c and the code term is c*c, matching the branch in
// the reference without a branch in the vector body. The 32-bit dequant product
// is exact: |level| <= 2^15 and step < 2^16 at every bit depth. The rate term
// per lane is 2 + bits.Len16(|level|) for a non-zero level and 0 otherwise.
func rdStatsBlockSIMD(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (dskip, dcode int64, rate int64, allZero bool) {
	stepV := archsimd.BroadcastInt32x4(step)
	tsV := archsimd.BroadcastInt32x4(-int32(ts))
	sh23 := archsimd.BroadcastInt32x4(-23)
	c126 := archsimd.BroadcastInt32x4(126)
	zero32 := archsimd.Int32x4{}
	one := archsimd.BroadcastUint32x4(1)
	skipAcc := archsimd.BroadcastInt64x2(0)
	codeAcc := archsimd.BroadcastInt64x2(0)
	orAcc := archsimd.BroadcastInt16x8(0)
	rateAcc := archsimd.BroadcastInt32x4(0)
	i := 0
	if count > 0 {
		t := tran[:count]
		qc := qcoeff[:count]
		for ; i <= len(qc)-8 && i <= len(t)-8; i += 8 {
			tw := (*[8]int32)(t[i:])
			c0 := archsimd.LoadInt32x4Array((*[4]int32)(tw[0:4]))
			c1 := archsimd.LoadInt32x4Array((*[4]int32)(tw[4:8]))
			q := archsimd.LoadInt16x8Array((*[8]int16)(qc[i:]))
			skipAcc = sqSumInt32(sqSumInt32(skipAcc, c0), c1)
			dq0 := encSarI32x4(q.ExtendLo4ToInt32().Mul(stepV), tsV, uint64(ts))
			dq1 := encSarI32x4(hiInt16(q).ExtendLo4ToInt32().Mul(stepV), tsV, uint64(ts))
			codeAcc = sqSumInt32(sqSumInt32(codeAcc, c0.Sub(dq0)), c1.Sub(dq1))
			orAcc = orAcc.Or(q)
			// |level| as unsigned 16-bit (|-32768| = 32768), split into 32-bit lanes.
			qAbs := q.Abs()
			lo := qAbs.ToBits().ExtendLo4ToUint32()
			hiMag := hiInt16(qAbs).ToBits().ExtendLo4ToUint32()
			nzLo := lo.Min(one).BitsToInt32()
			nzHi := hiMag.Min(one).BitsToInt32()
			rateAcc = rateAcc.Add(bitLenFloat32(lo, sh23, c126, zero32)).Add(bitLenFloat32(hiMag, sh23, c126, zero32)).
				Add(nzLo.Add(nzLo)).Add(nzHi.Add(nzHi))
		}
	}
	var lanes int64
	allZero = true
	for ; i < count; i++ {
		c := int64(tran[i])
		dskip += c * c
		v := qcoeff[i]
		if v == 0 {
			dcode += c * c
			continue
		}
		e := c - (int64(v)*int64(step))>>ts
		dcode += e * e
		level := v
		if level < 0 {
			level = -level
		}
		lanes += int64(2 + bits.Len16(uint16(level)))
		allZero = false
	}
	for l := range uint8(4) {
		lanes += int64(rateAcc.GetElem(l))
	}
	dskip += skipAcc.GetElem(0) + skipAcc.GetElem(1)
	dcode += codeAcc.GetElem(0) + codeAcc.GetElem(1)
	for l := range uint8(8) {
		if orAcc.GetElem(l) != 0 {
			allZero = false
		}
	}
	return dskip, dcode, lanes << 9, allZero
}

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	copyPredToReconBlockImpl = copyPredToReconBlockSIMD
	addConstantResidualToPredReconBlockImpl = addConstantResidualToPredReconBlockSIMD
}

// copyPredToReconBlockSIMD is copyPredToReconBlockPureGo with fixed-size row
// copies (vector moves) instead of a memmove call per row.
func copyPredToReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int) {
	if h <= 0 || w&7 != 0 {
		copyPredToReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h)
		return
	}
	copyBlockRows(reconPlane[py*stride+px:], stride, pred, predStride, w, h)
}

// copyBlockRows copies the w x h block src (stride ss) to dst (stride ds), w a
// multiple of 8: one grid bounds check per call, then unchecked fixed-width row copies.
func copyBlockRows(dst []byte, ds int, src []byte, ss, w, h int) {
	if h <= 0 || w <= 0 {
		return
	}
	if uint(ds) > rowStrideMax || uint(ss) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	if !unsafeutil.GridFit(len(dst), 0, ds, w, h) || !unsafeutil.GridFit(len(src), 0, ss, w, h) {
		panic(errKernelArgs)
	}
	gd, gs := unsafeutil.NewGrid(dst, 0, ds, w, h), unsafeutil.NewGrid(src, 0, ss, w, h)
	switch w {
	case 8:
		for r := 0; r < gd.Height(); r++ {
			*gd.Row8(r, 0) = *gs.Row8(r, 0)
		}
	case 16:
		for r := 0; r < gd.Height(); r++ {
			*gd.Row16(r, 0) = *gs.Row16(r, 0)
		}
	case 32:
		for r := 0; r < gd.Height(); r++ {
			*gd.Row32(r, 0) = *gs.Row32(r, 0)
		}
	case 64:
		for r := 0; r < gd.Height(); r++ {
			*gd.Row64(r, 0) = *gs.Row64(r, 0)
		}
	default:
		for r := 0; r < gd.Height(); r++ {
			x := 0
			for ; x+16 <= w; x += 16 {
				*gd.Row16(r, x) = *gs.Row16(r, x)
			}
			if x+8 <= w {
				*gd.Row8(r, x) = *gs.Row8(r, x)
			}
		}
	}
}

// addConstantResidualToPredReconBlockSIMD is
// addConstantResidualToPredReconBlockPureGo on 16-byte vectors: with the
// residual clamped to [-255, 255] (beyond which the clamp already pins every
// sample), pred + r clamped to [0, 255] is a saturating byte add of r or a
// saturating byte subtract of -r. Rows are bounds-checked windows: one per
// row pair for 8-wide blocks, one per row otherwise.
func addConstantResidualToPredReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int, residual int16) {
	if h <= 0 || (w&15 != 0 && (w != 8 || h&1 != 0)) {
		addConstantResidualToPredReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h, residual)
		return
	}
	if uint(stride) > rowStrideMax || uint(predStride) > rowStrideMax || uint(w) > rowsMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	dst := reconPlane[py*stride+px:]
	r := max(min(int(residual), 255), -255)
	add := archsimd.BroadcastUint8x16(uint8(max(r, 0)))
	sub := archsimd.BroadcastUint8x16(uint8(max(-r, 0)))
	if w == 8 {
		// Two 8-byte rows per vector, staged through memory (cheaper than
		// lane inserts on Apple cores).
		var pair [16]uint8
		dn, sn := stride+8, predStride+8
		for i := 0; i < h/2; i++ {
			do, so := 2*i*stride, 2*i*predStride
			dw, sw := dst[do:do+dn:do+dn], pred[so:so+sn:so+sn]
			*(*[8]byte)(pair[0:8]) = *(*[8]byte)(sw)
			*(*[8]byte)(pair[8:16]) = *(*[8]byte)(sw[predStride:])
			archsimd.LoadUint8x16Array(&pair).AddSaturated(add).SubSaturated(sub).StoreArray(&pair)
			*(*[8]byte)(dw) = *(*[8]byte)(pair[0:8])
			*(*[8]byte)(dw[stride:]) = *(*[8]byte)(pair[8:16])
		}
		return
	}
	switch w {
	case 16:
		dn, sn := stride+16, predStride+16
		for i := 0; i < h/2; i++ {
			do, so := 2*i*stride, 2*i*predStride
			dw, sw := dst[do:do+dn:do+dn], pred[so:so+sn:so+sn]
			addConst16((*[16]uint8)(dw), (*[16]uint8)(sw), add, sub, r >= 0)
			addConst16((*[16]uint8)(dw[stride:]), (*[16]uint8)(sw[predStride:]), add, sub, r >= 0)
		}
		if h&1 != 0 {
			y := h - 1
			addConst16((*[16]uint8)(dst[y*stride:y*stride+16]), (*[16]uint8)(pred[y*predStride:y*predStride+16]), add, sub, r >= 0)
		}
		return
	case 32:
		for y := 0; y < h; y++ {
			d, s := (*[32]uint8)(dst[y*stride:y*stride+32]), (*[32]uint8)(pred[y*predStride:y*predStride+32])
			addConst16((*[16]uint8)(d[0:16]), (*[16]uint8)(s[0:16]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[16:32]), (*[16]uint8)(s[16:32]), add, sub, r >= 0)
		}
		return
	case 64:
		for y := 0; y < h; y++ {
			d, s := (*[64]uint8)(dst[y*stride:y*stride+64]), (*[64]uint8)(pred[y*predStride:y*predStride+64])
			addConst16((*[16]uint8)(d[0:16]), (*[16]uint8)(s[0:16]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[16:32]), (*[16]uint8)(s[16:32]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[32:48]), (*[16]uint8)(s[32:48]), add, sub, r >= 0)
			addConst16((*[16]uint8)(d[48:64]), (*[16]uint8)(s[48:64]), add, sub, r >= 0)
		}
		return
	}
	for y := 0; y < h; y++ {
		d, s := dst[y*stride:y*stride+w], pred[y*predStride:y*predStride+w]
		for c := 0; c+16 <= w; c += 16 {
			addConst16((*[16]uint8)(d[c:c+16]), (*[16]uint8)(s[c:c+16]), add, sub, r >= 0)
		}
	}
}

// addConst16 stores the 16 samples s plus the clamped residual into d: a
// saturating add of add when pos, else a saturating subtract of sub.
func addConst16(d, s *[16]uint8, add, sub archsimd.Uint8x16, pos bool) {
	v := archsimd.LoadUint8x16Array(s)
	if pos {
		v = v.AddSaturated(add)
	} else {
		v = v.SubSaturated(sub)
	}
	v.StoreArray(d)
}

// sad8x8SIMD is the 8x8 SAD with independent source and reference strides.
func sad8x8SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8Rows(src, srcStride, ref, refStride)
}

// sad8x8x4SIMD computes four 8x8 SADs of one source block against four
// reference origins sharing stride.
func sad8x8x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4Rows(src, ref0, ref1, ref2, ref3, stride)
}

// sad8x8x4Step4SIMD computes four 8x8 SADs against reference origins ref+0,
// ref+4, ref+8, and ref+12 (the raster step-4 candidate group).
func sad8x8x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4Rows(src, ref, stride)
}

// sad8x8CompoundAvgSIMD computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block with independent source and reference strides.
func sad8x8CompoundAvgSIMD(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundRows(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}

// sad16x16SIMD is the 16x16 SAD with independent source and reference strides.
func sad16x16SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	if refStride == 16 && len(ref) >= 256 {
		return sad16x16Packed(src, srcStride, (*[256]uint8)(ref[:256]))
	}
	return sad16ColsDual(src, srcStride, ref, refStride, 16)
}

// sad16x16x4SIMD computes four 16x16 SADs against reference origins sharing
// stride.
func sad16x16x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16ColsX4(src, ref0, ref1, ref2, ref3, stride, 16)
}

// sad16x16x4Step4SIMD computes four 16x16 SADs against ref+0, ref+4, ref+8, and
// ref+12. The last candidate reads 12 bytes past each 16-byte row.
func sad16x16x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	return sad16ColsX4(src, ref, ref[4:], ref[8:], ref[12:], stride, 16)
}

// sad32x32SIMD is the 32x32 SAD with independent source and reference strides.
// It takes one 32-byte window per row pair.
func sad32x32SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32Rows(src, srcStride, ref, refStride)
}

// sad32x32x4SIMD computes four 32x32 SADs against reference origins sharing
// stride, as two 16-column strips.
func sad32x32x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	l0, l1, l2, l3 := sad16ColsX4(src, ref0, ref1, ref2, ref3, stride, 32)
	r0, r1, r2, r3 := sad16ColsX4(src[16:], ref0[16:], ref1[16:], ref2[16:], ref3[16:], stride, 32)
	return l0 + r0, l1 + r1, l2 + r2, l3 + r3
}

// sad32x32x4Step4SIMD computes four 32x32 SADs against ref+0, ref+4, ref+8, and
// ref+12, as two 16-column strips. The last candidate reads 12 bytes past each
// 32-byte row.
func sad32x32x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	l0, l1, l2, l3 := sad16ColsX4(src, ref, ref[4:], ref[8:], ref[12:], stride, 32)
	r0, r1, r2, r3 := sad16ColsX4(src[16:], ref[16:], ref[20:], ref[24:], ref[28:], stride, 32)
	return l0 + r0, l1 + r1, l2 + r2, l3 + r3
}

// sad64x64SIMD is the 64x64 SAD with independent source and reference strides.
// It takes one 64-byte window per row pair.
func sad64x64SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad64Rows(src, srcStride, ref, refStride)
}

// Single-stride adapters match the sadNxNImpl signatures in sad.go.

func sad8x8SingleSIMD(src, ref []byte, stride int, limit int) int {
	return sad8x8Same(src, ref, stride)
}

func sad16x16SingleSIMD(src, ref []byte, stride int) int {
	return sad16x16Same(src, ref, stride)
}

func sad32x32SingleSIMD(src, ref []byte, stride int) int {
	return sad32x32SIMD(src, stride, ref, stride)
}

// bindSIMDSAD points the SAD dispatch variables at the Go-native-SIMD kernels.
// Callers gate it on the CPU features the kernels need.
func bindSIMDSAD() {
	sad8x8Impl = sad8x8SingleSIMD
	sad16x16Impl = sad16x16SingleSIMD
	sad32x32Impl = sad32x32SingleSIMD
	sad8x8x4Step4Impl = sad8x8x4Step4SIMD
	sad8x8x4Impl = sad8x8x4SIMD
	sad16x16x4Impl = sad16x16x4SIMD
	sad16x16x4Step4Impl = sad16x16x4Step4SIMD
	sad32x32x4Impl = sad32x32x4SIMD
	sad32x32x4Step4Impl = sad32x32x4Step4SIMD
	sad8x8DualImpl = sad8x8SIMD
	sad16x16DualImpl = sad16x16SIMD
	sad32x32DualImpl = sad32x32SIMD
	sad8x8CompoundAvgBlockImpl = sad8x8CompoundAvgSIMD
}
