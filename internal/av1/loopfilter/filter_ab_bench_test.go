// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// BenchmarkFilterAB exercises the public dispatch with the same edge geometry
// in the assembly reference and the Go SIMD tree. The near-flat alternating
// samples keep the filter active, including the wide fourteen-tap path.
func BenchmarkFilterAB(b *testing.B) {
	type filterFn func(frame.Plane, int, uint8, Edge, int32, int32, int32, Thresholds) error
	for _, f := range []struct {
		name string
		fn   filterFn
	}{
		{"4", Filter4Edge}, {"6", Filter6Edge}, {"8", Filter8Edge}, {"14", Filter14Edge},
	} {
		for _, depth := range []uint8{8, 10, 12} {
			for _, dir := range []struct {
				name string
				edge Edge
				x, y int32
			}{{"H", EdgeHorizontal, 0, 32}, {"V", EdgeVertical, 32, 0}} {
				b.Run(fmt.Sprintf("F%s/%dbit/%s", f.name, depth, dir.name), func(b *testing.B) {
					bytesPerSample := 1
					if depth > 8 {
						bytesPerSample = 2
					}
					plane := testPlane(64, 64, bytesPerSample, 64*bytesPerSample)
					mid := uint16(1 << (depth - 1))
					for y := 0; y < 64; y++ {
						for x := 0; x < 64; x++ {
							setSample(plane, bytesPerSample, x, y, mid+uint16((x+y)&1))
						}
					}
					thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
					b.ReportAllocs()
					for b.Loop() {
						if err := f.fn(plane, bytesPerSample, depth, dir.edge, dir.x, dir.y, 64, thresholds); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
