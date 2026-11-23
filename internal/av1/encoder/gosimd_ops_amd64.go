//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// gosimd_ops_amd64.go: see gosimd_ops_arm64.go for the shared contract.

// Byte-shuffle indices. A negative index selects zero.
var (
	hiBytesIdx   = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	hi16WordsIdx = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	oddDwordsIdx = [16]int8{4, 5, 6, 7, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1}
)

// widenHi8 zero-extends bytes 8..15 of v to 16-bit lanes.
func widenHi8(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.PermuteOrZero(archsimd.LoadInt8x16Array(&hiBytesIdx)).ExtendLo8ToUint16()
}

// hiInt16 moves 16-bit lanes 4..7 of v into lanes 0..3 (upper lanes zero).
func hiInt16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&hi16WordsIdx)).
		ReshapeToUint16s().BitsToInt16()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc. Even lanes
// widen directly; odd lanes are first moved into the even positions.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	odd := v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	return acc.Add(v.MulWidenEven(v)).Add(odd.MulWidenEven(odd))
}

// mulSumInt32 adds the exact products a[i]*b[i] of the four int32 lane pairs
// to acc.
func mulSumInt32(acc archsimd.Int64x2, a, b archsimd.Int32x4) archsimd.Int64x2 {
	oddA := a.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	oddB := b.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	return acc.Add(a.MulWidenEven(b)).Add(oddA.MulWidenEven(oddB))
}

// packU16x16Idx keeps the low byte of each 16-bit lane in the low eight bytes
// of its 128-bit lane; packU16x16Lanes then joins the two lanes' halves.
var (
	packU16x16Idx   = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	packU16x16Lanes = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}
)

// packU16x16ToU8 narrows sixteen 16-bit lanes holding values 0..255 to bytes
// in lane order (VPSHUFB, then VPERMD across the 128-bit halves): amd64
// archsimd has no VPACKUSWB.
func packU16x16ToU8(v archsimd.Uint16x16) archsimd.Uint8x16 {
	b := v.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&packU16x16Idx))
	return b.AsUint32x8().Permute(archsimd.LoadUint32x8Array(&packU16x16Lanes)).AsUint8x32().GetLo()
}
