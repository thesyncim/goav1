// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// AVX2 forms of the directional intra edge filter and upsampler (dav1d's
// ipred_filter_edge / ipred_upsample_edge in ipred_avx2.asm). Both work on
// a padded stack copy of the edge, so the clamped end taps of the scalar
// code become plain loads, and both are exact: the filter's weights are
// nonnegative and sum to 16, so 16*4095+8 fits uint16, and the upsampler's
// |-a+9b+9c-d| <= 18*1023 fits int16 up to 10 bits.

var intraEdgeAVX2 = archsimd.X86.AVX2()

// intraEdgeFilterK holds the five filter taps of each strength, broadcast.
var intraEdgeFilterK = func() (k [3][intraEdgeTaps][8]uint16) {
	for s, taps := range intraEdgeFilterKernels {
		for j, t := range taps {
			for l := range k[s][j] {
				k[s][j][l] = uint16(t)
			}
		}
	}
	return k
}()

// filterIntraEdgeArch is the body of FilterIntraEdgeBounded after its checks
// (1 <= strength <= 3); it reports false when the scalar code must run.
func filterIntraEdgeArch(edge []uint16, strength uint8) bool {
	n := len(edge)
	if !intraEdgeAVX2 || n < 2 || n > intraEdgeMaxSize {
		return false
	}
	// pad[m] is the clamped scratch[m-2].
	var pad [intraEdgeMaxSize + 32]uint16
	pad[0], pad[1] = edge[0], edge[0]
	copy(pad[2:], edge)
	last := edge[n-1]
	for i := n + 2; i < n+24; i++ {
		pad[i] = last
	}
	k := &intraEdgeFilterK[strength-1]
	k1 := archsimd.LoadUint16x8Array(&k[1])
	k2 := archsimd.LoadUint16x8Array(&k[2])
	k3 := archsimd.LoadUint16x8Array(&k[3])
	round := archsimd.BroadcastUint16x8(8)
	for i := 1; i < n; i += 8 {
		acc := archsimd.LoadUint16x8(pad[i+1:]).Mul(k1).
			Add(archsimd.LoadUint16x8(pad[i+2:]).Mul(k2)).
			Add(archsimd.LoadUint16x8(pad[i+3:]).Mul(k3)).
			Add(round)
		if strength == 3 {
			acc = acc.Add(archsimd.LoadUint16x8(pad[i:]).Add(archsimd.LoadUint16x8(pad[i+4:])).ShiftAllLeft(1))
		}
		acc = acc.ShiftAllRight(4)
		if i+8 <= n {
			acc.Store(edge[i:])
		} else {
			var t [8]uint16
			acc.StoreArray(&t)
			copy(edge[i:], t[:n-i])
		}
	}
	return true
}

// upsampleIntraEdgeArch is the body of UpsampleIntraEdgeBounded after its
// checks; it reports false when the scalar code must run.
func upsampleIntraEdgeArch(edge []uint16, origin, size int, max uint16) bool {
	if !intraEdgeAVX2 || max > 1023 || size > intraEdgeMaxUpsampleSize || origin < 2 {
		return false
	}
	// s is the scalar scratch: s[0] = s[1] = p[-1], s[2..size+1] = p[0..size-1],
	// s[size+2] = p[size-1], zero past it.
	var s [intraEdgeMaxUpsampleSize + 16]int16
	s[0] = int16(edge[origin-1])
	s[1] = s[0]
	for i := 0; i < size; i++ {
		s[i+2] = int16(edge[origin+i])
	}
	s[size+2] = int16(edge[origin+size-1])
	edge[origin-2] = uint16(s[0])
	nine := archsimd.BroadcastInt16x8(9)
	round := archsimd.BroadcastInt16x8(8)
	maxV := archsimd.BroadcastInt16x8(int16(max))
	var zero archsimd.Int16x8
	out := edge[origin-1:]
	for i := 0; i < size; i += 8 {
		a := archsimd.LoadInt16x8(s[i:])
		b := archsimd.LoadInt16x8(s[i+1:])
		c := archsimd.LoadInt16x8(s[i+2:])
		d := archsimd.LoadInt16x8(s[i+3:])
		f := b.Add(c).Mul(nine).Sub(a).Sub(d).Add(round).ShiftAllRight(4).Max(zero).Min(maxV)
		lo, hi := f.InterleaveLo(c), f.InterleaveHi(c)
		cnt := min(size-i, 8) * 2
		var t [16]int16
		lo.StoreArray((*[8]int16)(t[:8]))
		hi.StoreArray((*[8]int16)(t[8:]))
		for j := 0; j < cnt; j++ {
			out[2*i+j] = uint16(t[j])
		}
	}
	return true
}
