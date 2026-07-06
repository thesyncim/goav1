// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

// benchFilterBlockU8Params is the worst-case 8-bit 8x8 shape: both primary
// and secondary taps active so the clip path runs.
func benchFilterBlockU8Params() ([]byte, []uint16, int, BlockFilterParams) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{
		PrimaryStrength:   15,
		SecondaryStrength: 4,
		Direction:         4,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		CoeffShift:        0,
		Width:             8,
		Height:            8,
	}
	return dst, input, cdefBlockOrigin(), params
}

// BenchmarkFilterBlockU8PureGo / BenchmarkFilterBlockU8Dispatch bracket the
// 8-bit-dst kernel speedup; compare against BenchmarkFilterBlockDispatch to
// see the uint8-store vs uint16-store delta at 8 bits.
func BenchmarkFilterBlockU8PureGo(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8PureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockU8Dispatch(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockU8Dispatch4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockU8Params()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, origin, params)
	}
}
