// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import "testing"

// Vertical six/eight-sample benchmarks over a flat 10-bit plane (the all-flat
// path is the heaviest branch and the common case on smooth content). These
// cover the two-byte transpose path around the horizontal 16-bit kernels; the
// short cases cover a single-group batch.

func BenchmarkFilter6EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter6EdgeVertical10bitShort(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 2, 10, EdgeVertical, 32, 0, 8, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical10bitShort(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 2, 10, EdgeVertical, 32, 0, 8, thresholds)
	}
}
