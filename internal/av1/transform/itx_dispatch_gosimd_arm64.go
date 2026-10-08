// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// init binds the Go SIMD batched inverse kernels that replaced the arm64 NEON
// assembly. The column kernels run four (or two) columns in int32 lanes and the
// row kernels run rows through a 4x4 transpose; the adapters keep the same
// envelope gating the assembly adapters had and fall back to the pure-Go
// reference outside it, so output bytes do not depend on the path taken.
func init() {
	inverseDCT4Col4Impl = inverseDCT4Col4SIMDAdapter
	inverseDCT8Col2Impl = inverseDCT8Col2SIMDAdapter
	inverseDCT8Col4Impl = inverseDCT8Col4SIMDAdapter
	inverseDCT16Col2Impl = inverseDCT16Col2SIMDAdapter
	inverseDCT16Col4Impl = inverseDCT16Col4SIMDAdapter
	inverseADST16Col4Impl = inverseADST16Col4SIMDAdapter
	inverseADST16Col4FlipImpl = inverseADST16Col4FlipSIMDAdapter
	inverseDCT8Row2Impl = inverseDCT8Row2SIMDAdapter
	inverseDCT16Row2Impl = inverseDCT16Row2SIMDAdapter
	inverseADST16Row4Impl = inverseADST16Row4SIMDAdapter
	inverseADST16Row4FlipImpl = inverseADST16Row4FlipSIMDAdapter
}

// simdInEnvelope reports whether every value lies inside the +/-2^19 stage
// envelope the int32-lane kernels are exact for. Row kernels read unclamped
// coefficients, so they check their inputs; column kernels rely on the staged
// [min, max] pre-clamp exactly as the assembly kernels did.
func simdInEnvelope(v []int32) bool {
	for _, x := range v {
		if x < -colClampBoundNEON || x >= colClampBoundNEON {
			return false
		}
	}
	return true
}

func simdColEnvelope(min, max int32) bool {
	return min >= -colClampBoundNEON && max < colClampBoundNEON
}

func inverseDCT4Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct4Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT4Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT4Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct8Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT8Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT8Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT8Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT16Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT16Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4FlipSIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4FlipPureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4FlipSIMD(buf, rowStride, min, max)
}

func inverseDCT8Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size || !simdColEnvelope(min, max) ||
		!simdInEnvelope(r0[:dct8Size]) || !simdInEnvelope(r1[:dct8Size]) {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT8Row2SIMD(r0[:dct8Size], r1[:dct8Size], min, max)
}

func inverseDCT16Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || !simdColEnvelope(min, max) ||
		!simdInEnvelope(r0[:dct16Size]) || !simdInEnvelope(r1[:dct16Size]) {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT16Row2SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
}

func inverseADST16Row4SIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) || !simdInEnvelope(r0[:adst16Size]) || !simdInEnvelope(r1[:adst16Size]) ||
		!simdInEnvelope(r2[:adst16Size]) || !simdInEnvelope(r3[:adst16Size]) {
		inverseADST16Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4SIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}

func inverseADST16Row4FlipSIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) || !simdInEnvelope(r0[:adst16Size]) || !simdInEnvelope(r1[:adst16Size]) ||
		!simdInEnvelope(r2[:adst16Size]) || !simdInEnvelope(r3[:adst16Size]) {
		inverseADST16Row4FlipPureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4FlipSIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}
