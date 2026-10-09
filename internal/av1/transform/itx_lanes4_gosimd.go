// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// Go-native SIMD four-lane inverse DCT64 (dav1d's high-bitdepth itx shape,
// src/arm/64/itx16.S): four adjacent columns, or four transposed rows, ride
// the four int32 lanes of every vector. The butterfly kernels are generated
// (dct64_lanes4_gosimd.go) in two flavours: Narrow for clamp bounds within
// +/-2^17 (8- and 10-bit streams, and 12-bit column passes) and Wide for the
// +/-2^19 envelope of every supported bit depth. Both are bit-identical to
// inverseDCT64 on each lane when the inputs are pre-clamped to [min, max],
// which the row staging and the mid-pass clamp (hybrid.go) guarantee.

const (
	itxNarrowBound = 1 << 17
	itxWideBound   = 1 << 19
)

// inverseDCT64Lanes4 picks the kernel for the clamp bounds; it reports false
// when the bounds exceed the int32-exact envelope.
func inverseDCT64Lanes4(p unsafe.Pointer, stride uintptr, min, max int32) bool {
	switch {
	case min >= -itxNarrowBound && max < itxNarrowBound:
		inverseDCT64Lanes4Narrow(p, stride, min, max)
	case min >= -itxWideBound && max < itxWideBound:
		inverseDCT64Lanes4Wide(p, stride, min, max)
	default:
		return false
	}
	return true
}

func inverseDCT64Col4SIMD(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < 4 || rowStride > (len(buf)-4)/(dct64Size-1) ||
		!inverseDCT64Lanes4(unsafe.Pointer(&buf[0]), uintptr(rowStride)*4, min, max) {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
	}
}

// inverseDCT64Row4SIMD transposes the four rows into 64 four-lane vectors,
// runs the column kernel on them and transposes back.
func inverseDCT64Row4SIMD(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size || len(r2) < dct64Size || len(r3) < dct64Size ||
		min < -itxWideBound || max >= itxWideBound {
		inverseDCT64Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	var t [dct64Size][4]int32
	a, b, c, d := (*[dct64Size]int32)(r0), (*[dct64Size]int32)(r1), (*[dct64Size]int32)(r2), (*[dct64Size]int32)(r3)
	for i := 0; i < dct64Size; i += 4 {
		t0, t1, t2, t3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(a[i:])), archsimd.LoadInt32x4Array((*[4]int32)(b[i:])),
			archsimd.LoadInt32x4Array((*[4]int32)(c[i:])), archsimd.LoadInt32x4Array((*[4]int32)(d[i:])))
		t0.StoreArray(&t[i])
		t1.StoreArray(&t[i+1])
		t2.StoreArray(&t[i+2])
		t3.StoreArray(&t[i+3])
	}
	inverseDCT64Lanes4(unsafe.Pointer(&t), 16, min, max)
	for i := 0; i < dct64Size; i += 4 {
		o0, o1, o2, o3 := itxTranspose4(
			archsimd.LoadInt32x4Array(&t[i]), archsimd.LoadInt32x4Array(&t[i+1]),
			archsimd.LoadInt32x4Array(&t[i+2]), archsimd.LoadInt32x4Array(&t[i+3]))
		o0.StoreArray((*[4]int32)(a[i:]))
		o1.StoreArray((*[4]int32)(b[i:]))
		o2.StoreArray((*[4]int32)(c[i:]))
		o3.StoreArray((*[4]int32)(d[i:]))
	}
}

// itxTranspose4 transposes the 4x4 int32 matrix whose rows are v0..v3.
func itxTranspose4(v0, v1, v2, v3 archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	e0 := v0.InterleaveLo(v1).ToBits().ReshapeToUint64s()
	e1 := v0.InterleaveHi(v1).ToBits().ReshapeToUint64s()
	e2 := v2.InterleaveLo(v3).ToBits().ReshapeToUint64s()
	e3 := v2.InterleaveHi(v3).ToBits().ReshapeToUint64s()
	return e0.InterleaveLo(e2).ReshapeToUint32s().BitsToInt32(), e0.InterleaveHi(e2).ReshapeToUint32s().BitsToInt32(),
		e1.InterleaveLo(e3).ReshapeToUint32s().BitsToInt32(), e1.InterleaveHi(e3).ReshapeToUint32s().BitsToInt32()
}
