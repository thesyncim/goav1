// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"simd/archsimd"
)

func TestDCT2LaneSIMDDispatchBound(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"DCT8 row", inverseDCT8Row2Impl, inverseDCT8Row2SIMDAdapter},
		{"DCT16 row", inverseDCT16Row2Impl, inverseDCT16Row2SIMDAdapter},
		{"DCT8 column", inverseDCT8Col2Impl, inverseDCT8Col2SIMDAdapter},
		{"DCT16 column", inverseDCT16Col2Impl, inverseDCT16Col2SIMDAdapter},
	}
	for _, check := range checks {
		if reflect.ValueOf(check.got).Pointer() != reflect.ValueOf(check.want).Pointer() {
			t.Errorf("%s dispatch is not bound to Go SIMD", check.name)
		}
	}
}

func TestDCT2LaneSIMDMatchesScalarFullRange(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	rng := rand.New(rand.NewSource(0xDC7216))
	clamps := [][2]int32{
		{-32768, 32767},
		{-(1 << 19), (1 << 19) - 1},
		{math.MinInt32, math.MaxInt32},
		{-1, 1},
	}
	for _, tc := range []struct {
		name string
		n    int
		row  func([]int32, []int32, int32, int32)
		col  func([]int32, int, int32, int32)
		one  func([]int32, int, int32, int32)
	}{
		{"DCT8", dct8Size, inverseDCT8Row2SIMDAdapter, inverseDCT8Col2SIMDAdapter, func(x []int32, stride int, lo, hi int32) { inverseDCT8(x, stride, lo, hi) }},
		{"DCT16", dct16Size, inverseDCT16Row2SIMDAdapter, inverseDCT16Col2SIMDAdapter, func(x []int32, stride int, lo, hi int32) { inverseDCT16(x, stride, lo, hi) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for iter := 0; iter < 80; iter++ {
				lo, hi := clamps[iter%len(clamps)][0], clamps[iter%len(clamps)][1]
				for offset := 0; offset < 4; offset++ {
					got := make([]int32, offset+2*tc.n+5)
					for i := range got {
						switch i % 11 {
						case 0:
							got[i] = math.MinInt32
						case 1:
							got[i] = math.MaxInt32
						case 2:
							got[i] = lo
						case 3:
							got[i] = hi
						default:
							got[i] = int32(rng.Uint32())
						}
					}
					want := append([]int32(nil), got...)
					tc.row(got[offset:offset+tc.n], got[offset+tc.n+1:offset+2*tc.n+1], lo, hi)
					tc.one(want[offset:offset+tc.n], 1, lo, hi)
					tc.one(want[offset+tc.n+1:offset+2*tc.n+1], 1, lo, hi)
					if !reflect.DeepEqual(got, want) {
						for i := range got {
							if got[i] != want[i] {
								t.Fatalf("row mismatch iter=%d offset=%d clamp=[%d,%d] index=%d got=%d want=%d", iter, offset, lo, hi, i, got[i], want[i])
							}
						}
					}
				}

				rowStride := 2 + iter%9
				offset := iter % 4
				length := (tc.n-1)*rowStride + 2
				got := make([]int32, offset+length+5)
				for i := range got {
					switch i % 13 {
					case 0:
						got[i] = math.MinInt32
					case 1:
						got[i] = math.MaxInt32
					case 2:
						got[i] = lo
					case 3:
						got[i] = hi
					default:
						got[i] = int32(rng.Uint32())
					}
				}
				want := append([]int32(nil), got...)
				tc.col(got[offset:offset+length], rowStride, lo, hi)
				tc.one(want[offset:], rowStride, lo, hi)
				tc.one(want[offset+1:], rowStride, lo, hi)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("column mismatch iter=%d offset=%d stride=%d clamp=[%d,%d]", iter, offset, rowStride, lo, hi)
				}
			}
		})
	}
}
