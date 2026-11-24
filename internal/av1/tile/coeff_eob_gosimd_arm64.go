//go:build goexperiment.simd && arm64 && !purego

package tile

import "simd/archsimd"

func init() {
	txbEOBVectorImpl = txbEOBSIMD
}

// txbEOBSIMD is the vector form of the txbEOB* scans: eight coefficients per
// step, keeping iscanP1 where the coefficient is non-zero (BIC with the
// equal-to-zero mask) and folding it into a running maximum.
func txbEOBSIMD(coeffs, iscanP1 []int16) int {
	iscanP1 = iscanP1[:len(coeffs)]
	zero := archsimd.BroadcastInt16x8(0)
	acc0, acc1 := zero, zero
	i := 0
	for ; i <= len(coeffs)-16; i += 16 {
		c, s := (*[16]int16)(coeffs[i:]), (*[16]int16)(iscanP1[i:])
		v0 := archsimd.LoadInt16x8Array((*[8]int16)(c[0:8]))
		v1 := archsimd.LoadInt16x8Array((*[8]int16)(c[8:16]))
		s0 := archsimd.LoadInt16x8Array((*[8]int16)(s[0:8]))
		s1 := archsimd.LoadInt16x8Array((*[8]int16)(s[8:16]))
		acc0 = acc0.Max(s0.AndNot(v0.Equal(zero).ToInt16x8()))
		acc1 = acc1.Max(s1.AndNot(v1.Equal(zero).ToInt16x8()))
	}
	for ; i <= len(coeffs)-8; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(coeffs[i:]))
		s := archsimd.LoadInt16x8Array((*[8]int16)(iscanP1[i:]))
		acc0 = acc0.Max(s.AndNot(v.Equal(zero).ToInt16x8()))
	}
	return int(acc0.Max(acc1).ReduceMax())
}
