//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
)

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
	switch w {
	case 8:
		if h&1 == 0 {
			residualRows8(dst, srcPlane, srcOff, stride, pred, predStride, h)
			return
		}
	case 16:
		for r := range h {
			residualRow16((*[16]int16)(dst[r*16:]), (*[16]uint8)(srcPlane[srcOff+r*stride:]), (*[16]uint8)(pred[r*predStride:]))
		}
		return
	case 32:
		for r := range h {
			d := (*[32]int16)(dst[r*32:])
			s := (*[32]uint8)(srcPlane[srcOff+r*stride:])
			p := (*[32]uint8)(pred[r*predStride:])
			residualRow16((*[16]int16)(d[:16]), (*[16]uint8)(s[:16]), (*[16]uint8)(p[:16]))
			residualRow16((*[16]int16)(d[16:]), (*[16]uint8)(s[16:]), (*[16]uint8)(p[16:]))
		}
		return
	case 64:
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
			a := archsimd.LoadUint8x16(srow[c : c+16])
			b := archsimd.LoadUint8x16(prow[c : c+16])
			a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).Store(drow[c : c+8])
			widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).Store(drow[c+8 : c+16])
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
	tsU := uint64(ts)
	one := archsimd.BroadcastUint32x4(1)
	skipAcc := archsimd.BroadcastInt64x2(0)
	codeAcc := archsimd.BroadcastInt64x2(0)
	orAcc := archsimd.BroadcastInt16x8(0)
	rateAcc := archsimd.BroadcastInt32x4(0)
	i := 0
	for ; i+8 <= count; i += 8 {
		c0 := archsimd.LoadInt32x4(tran[i : i+4])
		c1 := archsimd.LoadInt32x4(tran[i+4 : i+8])
		q := archsimd.LoadInt16x8(qcoeff[i : i+8])
		skipAcc = sqSumInt32(sqSumInt32(skipAcc, c0), c1)
		dq0 := q.ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		dq1 := hiInt16(q).ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		codeAcc = sqSumInt32(sqSumInt32(codeAcc, c0.Sub(dq0)), c1.Sub(dq1))
		orAcc = orAcc.Or(q)
		// |level| as unsigned 16-bit (|-32768| = 32768), split into 32-bit lanes.
		mag := q.Abs().ToBits()
		lo := mag.ExtendLo4ToUint32()
		hiMag := hiInt16(q.Abs()).ToBits().ExtendLo4ToUint32()
		nzLo := lo.Min(one).BitsToInt32()
		nzHi := hiMag.Min(one).BitsToInt32()
		rateAcc = rateAcc.Add(bitLenFloat32(lo)).Add(bitLenFloat32(hiMag)).
			Add(nzLo.Add(nzLo)).Add(nzHi.Add(nzHi))
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
