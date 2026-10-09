//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"encoding/binary"
	"simd/archsimd"
)

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
	pixelStats8x4Impl = pixelStats8x4SIMD
	pixelStats8x16Impl = pixelStats8x16SIMD
	pixelStats8x32Impl = pixelStats8x32SIMD
	pixelStats16x4Impl = pixelStats16x4SIMD
	pixelStats16x8Impl = pixelStats16x8SIMD
	pixelStats16x16Impl = pixelStats16x16SIMD
	pixelStats16x32Impl = pixelStats16x32SIMD
	pixelStats16x64Impl = pixelStats16x64SIMD
	pixelStats32x8Impl = pixelStats32x8SIMD
	pixelStats32x16Impl = pixelStats32x16SIMD
	pixelStats32x32Impl = pixelStats32x32SIMD
	pixelStats32x64Impl = pixelStats32x64SIMD
	pixelStats64x16Impl = pixelStats64x16SIMD
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
	sseAcc := archsimd.BroadcastInt32x4(0)
	sumAcc := archsimd.BroadcastInt32x4(0)
	for r := range h {
		srow := src[r*srcStride : r*srcStride+w]
		rrow := ref[r*refStride : r*refStride+w]
		for c := 0; c < w; c += 16 {
			sseAcc, sumAcc = pixelDiffAcc(sseAcc, sumAcc, archsimd.LoadUint8x16(srow[c:c+16]), archsimd.LoadUint8x16(rrow[c:c+16]))
		}
	}
	total, signed := reducePixelAcc(sseAcc, sumAcc)
	return uint32(total), int32(signed)
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

func pixelStats8x4SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 8, 4)
}

func pixelStats8x16SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats8x32SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 8, 32)
}

func pixelStats16x4SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 16, 4)
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

func pixelStats16x64SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 16, 64)
}

func pixelStats32x8SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 32, 8)
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

func pixelStats64x16SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 64, 16)
}

func pixelStats64x32SIMD(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsSIMD(src, srcStride, ref, refStride, 64, 32)
}
