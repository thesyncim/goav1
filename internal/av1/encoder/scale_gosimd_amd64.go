//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// amd64 has no ConcatEven in this archsimd, so each output vector gathers its
// samples with PermuteOrZero (PSHUFB): every source vector contributes the lanes
// it owns and writes zero elsewhere, and the partial results are ORed together.
// The index tables are built once from the sample-position arithmetic in the
// comments of each kernel.

// scaleDown2Byte[m] picks bytes for source vector m (0..1) of a 2x byte row:
// output lane l = 8m + i takes source byte 2i of that vector.
var scaleDown2Byte = [2][16]int8{
	{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14},
}

// scaleDown4Byte[m] picks bytes for source vector m (0..3) of a 4x byte row:
// output lane l = 4m + i takes source byte 4i of that vector.
var scaleDown4Byte = [4][16]int8{
	{0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12},
}

// scaleDown2Word[m] picks bytes for source vector m (0..1) of a 2x 16-bit row:
// output byte k = 2l + b (l is the output sample) takes source byte 4(l%4)+b
// when l/4 == m.
var scaleDown2Word = [2][16]int8{
	{0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13},
}

// scaleDown4Word[m] picks bytes for source vector m (0..3) of a 4x 16-bit row:
// output byte k = 2l + b takes source byte 8(l%2)+b when k/4 == m.
var scaleDown4Word = [4][16]int8{
	{0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9},
}

// scaleRow8Down2 writes dst[x] = src[2x] for len(dst) a multiple of 16.
func scaleRow8Down2(dst, src []byte) {
	for x := 0; x < len(dst); x += 16 {
		s := src[2*x : 2*x+32]
		v0 := archsimd.LoadUint8x16(s[0:16])
		v1 := archsimd.LoadUint8x16(s[16:32])
		out := v0.PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown2Byte[0])).
			Or(v1.PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown2Byte[1])))
		out.Store(dst[x : x+16])
	}
}

// scaleRow8Down4 writes dst[x] = src[4x] for len(dst) a multiple of 16.
func scaleRow8Down4(dst, src []byte) {
	for x := 0; x < len(dst); x += 16 {
		s := src[4*x : 4*x+64]
		out := archsimd.LoadUint8x16(s[0:16]).PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Byte[0])).
			Or(archsimd.LoadUint8x16(s[16:32]).PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Byte[1]))).
			Or(archsimd.LoadUint8x16(s[32:48]).PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Byte[2]))).
			Or(archsimd.LoadUint8x16(s[48:64]).PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Byte[3])))
		out.Store(dst[x : x+16])
	}
}

// scaleRow16Down2 writes dst[x] = src[2x] for len(dst) a multiple of 8.
func scaleRow16Down2(dst, src []uint16) {
	for x := 0; x < len(dst); x += 8 {
		s := src[2*x : 2*x+16]
		v0 := archsimd.LoadUint16x8(s[0:8]).ReshapeToUint8s()
		v1 := archsimd.LoadUint16x8(s[8:16]).ReshapeToUint8s()
		out := v0.PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown2Word[0])).
			Or(v1.PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown2Word[1])))
		out.ReshapeToUint16s().Store(dst[x : x+8])
	}
}

// scaleRow16Down4 writes dst[x] = src[4x] for len(dst) a multiple of 8.
func scaleRow16Down4(dst, src []uint16) {
	for x := 0; x < len(dst); x += 8 {
		s := src[4*x : 4*x+32]
		out := archsimd.LoadUint16x8(s[0:8]).ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Word[0])).
			Or(archsimd.LoadUint16x8(s[8:16]).ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Word[1]))).
			Or(archsimd.LoadUint16x8(s[16:24]).ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Word[2]))).
			Or(archsimd.LoadUint16x8(s[24:32]).ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&scaleDown4Word[3])))
		out.ReshapeToUint16s().Store(dst[x : x+8])
	}
}
