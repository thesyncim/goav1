//go:build goexperiment.simd && arm64 && !purego

package tile

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	txbEOBVectorImpl = txbEOBSIMD
}

// txbEOBSIMD is the vector form of the txbEOB* scans: eight coefficients per
// step, keeping iscanP1 where the coefficient is non-zero (BIC with the
// equal-to-zero mask) and folding it into a running maximum.
func txbEOBSIMD(coeffs, iscanP1 []int16) int {
	n := len(coeffs)
	_ = iscanP1[n-1]
	cp := unsafe.Pointer(unsafe.SliceData(coeffs))
	ip := unsafe.Pointer(unsafe.SliceData(iscanP1))
	zero := archsimd.BroadcastInt16x8(0)
	acc0, acc1 := zero, zero
	i := 0
	for ; i+16 <= n; i += 16 {
		v0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(cp, 2*i)))
		v1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(cp, 2*i+16)))
		s0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ip, 2*i)))
		s1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ip, 2*i+16)))
		acc0 = acc0.Max(s0.AndNot(v0.Equal(zero).ToInt16x8()))
		acc1 = acc1.Max(s1.AndNot(v1.Equal(zero).ToInt16x8()))
	}
	for ; i < n; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(cp, 2*i)))
		s := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ip, 2*i)))
		acc0 = acc0.Max(s.AndNot(v.Equal(zero).ToInt16x8()))
	}
	return int(acc0.Max(acc1).ReduceMax())
}
