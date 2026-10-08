//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// gosimd_ops_amd64.go: see gosimd_ops_arm64.go for the shared contract.

// Byte-shuffle indices. A negative index selects zero.
var (
	hiBytesIdx    = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	hi16WordsIdx  = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	hi32DwordsIdx = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	oddDwordsIdx  = [16]int8{4, 5, 6, 7, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1}
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

// hiInt32 moves 32-bit lanes 2..3 of v into lanes 0..1 (upper lanes zero).
func hiInt32(v archsimd.Int32x4) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&hi32DwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc. Even lanes
// widen directly; odd lanes are first moved into the even positions.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	odd := v.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&oddDwordsIdx)).
		ReshapeToUint32s().BitsToInt32()
	return acc.Add(v.MulWidenEven(v)).Add(odd.MulWidenEven(odd))
}

// Even/odd element shuffles: each source keeps its even (or odd) lanes packed
// into the low half via PSHUFB, and PUNPCKL zips the two packed halves.
var (
	evenBytes16Idx = [16]int8{0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1}
	oddBytes16Idx  = [16]int8{2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	evenBytes32Idx = [16]int8{0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1}
	oddBytes32Idx  = [16]int8{4, 5, 6, 7, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1}
)

// packEven16 keeps lanes 0,2,4,6 of x in the low four lanes (zero above).
func packEven16(x archsimd.Int16x8, idx *[16]int8) archsimd.Int16x8 {
	return x.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(idx)).
		ReshapeToUint16s().BitsToInt16()
}

func interleaveEven16(x, y archsimd.Int16x8) archsimd.Int16x8 {
	return packEven16(x, &evenBytes16Idx).InterleaveLo(packEven16(y, &evenBytes16Idx))
}

func interleaveOdd16(x, y archsimd.Int16x8) archsimd.Int16x8 {
	return packEven16(x, &oddBytes16Idx).InterleaveLo(packEven16(y, &oddBytes16Idx))
}

// packEven32 keeps lanes 0,2 of x in the low two lanes (zero above).
func packEven32(x archsimd.Int32x4, idx *[16]int8) archsimd.Int32x4 {
	return x.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(idx)).
		ReshapeToUint32s().BitsToInt32()
}

func interleaveEven32(x, y archsimd.Int32x4) archsimd.Int32x4 {
	return packEven32(x, &evenBytes32Idx).InterleaveLo(packEven32(y, &evenBytes32Idx))
}

func interleaveOdd32(x, y archsimd.Int32x4) archsimd.Int32x4 {
	return packEven32(x, &oddBytes32Idx).InterleaveLo(packEven32(y, &oddBytes32Idx))
}

// shiftRight16 and shiftRight32 are arithmetic right shifts by the uniform
// negative count in lane 0 of rsh (the same value in every lane).
func shiftRight16(v, rsh archsimd.Int16x8) archsimd.Int16x8 {
	return v.ShiftAllRight(uint64(-int64(rsh.GetElem(0))))
}

func shiftRight32(v, rsh archsimd.Int32x4) archsimd.Int32x4 {
	return v.ShiftAllRight(uint64(-int64(rsh.GetElem(0))))
}
