//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import "simd/archsimd"

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
	for i := 0; i < len(q); i += 16 {
		a0 = a0.Or(archsimd.LoadInt16x8(q[i : i+8]))
		a1 = a1.Or(archsimd.LoadInt16x8(q[i+8 : i+16]))
	}
	var lanes [8]int16
	a0.Or(a1).StoreArray(&lanes)
	var any int16
	for _, v := range lanes {
		any |= v
	}
	return any == 0
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
	for i := 0; i < n; i += 8 {
		q := archsimd.LoadInt16x8(qcoeff[i : i+8])
		s := archsimd.LoadInt16x8(iscan[i : i+8])
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
