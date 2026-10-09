// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestFilterSIMDExactMinimumWindow covers the final load and store at the end
// of a slice with no padding. The vertical cases also hit one full tile plus
// an odd tail, and the fourteen-tap cases exercise overlapping tiles.
func TestFilterSIMDExactMinimumWindow(t *testing.T) {
	type edgeFn func([]byte, int, int, int, int, int, filter4Params)
	for _, tc := range []struct {
		name           string
		before, taps   int
		pure8, simd8   edgeFn
		pure16, simd16 edgeFn
	}{
		{"4", 2, 4,
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4EdgePureGo(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4EdgeSIMD(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4Edge16PureGo(p, q, s, o, n, f) },
			func(p []byte, q, s, o, n, _ int, f filter4Params) { filter4Edge16SIMD(p, q, s, o, n, f) }},
		{"6", 3, 6, filter6EdgePureGo, filter6EdgeSIMD, filter6Edge16PureGo, filter6Edge16SIMD},
		{"8", 4, 8, filter8EdgePureGo, filter8EdgeSIMD, filter8Edge16PureGo, filter8Edge16SIMD},
		{"14", 7, 14, filter14EdgePureGo, filter14EdgeSIMD, filter14Edge16PureGo, filter14Edge16SIMD},
	} {
		for _, depth := range []uint8{8, 10, 12} {
			sz := 1
			if depth > 8 {
				sz = 2
			}
			maxVal := (1 << depth) - 1
			for _, vertical := range []bool{false, true} {
				for _, length := range []int{8, 9, 15, 32} {
					for _, thr := range []Thresholds{
						{Limit: 0, BlockLimit: 0, HighEdgeVariance: 0},
						{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10},
						{Limit: 255, BlockLimit: 255, HighEdgeVariance: 255},
					} {
						scale, params := filter4ParamsFor(depth, thr)
						step, outer := length*sz, sz
						q0 := tc.before * step
						if vertical {
							step, outer = sz, tc.taps*sz
							q0 = tc.before * sz
						}
						buf := make([]byte, length*tc.taps*sz)
						rng := rand.New(rand.NewSource(int64(1000 + int(depth)*100 + length*17 + tc.taps)))
						for pattern := 0; pattern < 4; pattern++ {
							for i := 0; i < len(buf); i += sz {
								v := 0
								switch pattern {
								case 1:
									v = maxVal
								case 2:
									if (i/sz)&1 != 0 {
										v = maxVal
									}
								case 3:
									v = rng.Intn(maxVal + 1)
								}
								buf[i] = byte(v)
								if sz == 2 {
									buf[i+1] = byte(v >> 8)
								}
							}
							want := bytes.Clone(buf)
							got := bytes.Clone(buf)
							pure, simd := tc.pure8, tc.simd8
							if sz == 2 {
								pure, simd = tc.pure16, tc.simd16
							}
							pure(want, q0, step, outer, length, scale, params)
							simd(got, q0, step, outer, length, scale, params)
							if !bytes.Equal(got, want) {
								t.Fatalf("filter%s depth=%d vertical=%v len=%d thr=%+v pattern=%d", tc.name, depth, vertical, length, thr, pattern)
							}
						}
					}
				}
			}
		}
	}
}
