// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import "testing"

// Vertical fourteen-sample benchmarks over a flat plane (the all-flat wide
// path is the heaviest branch and the common case on smooth content). The
// 10-bit case covers the two-byte transpose path; the short case covers a
// single-group batch.

func BenchmarkFilter14EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter14EdgeVerticalShort(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeVertical, 32, 0, 8, thresholds)
	}
}
