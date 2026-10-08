// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package dsp

import (
	"fmt"
	"testing"
)

// planeShapeBenchStride is a plane-like destination stride, as the decoder's
// reconstruct path writes transform blocks into full frame planes.
const planeShapeBenchStride = 512

var planeShapeBenchSizes = []int{4, 8, 16, 32, 64}

// BenchmarkPlaneResidualShapes measures AddResidualPlaneBlockTrusted through
// the dispatched kernel for the square transform shapes the decoder uses.
func BenchmarkPlaneResidualShapes(b *testing.B) {
	for _, bps := range []int{1, 2} {
		for _, n := range planeShapeBenchSizes {
			b.Run(fmt.Sprintf("bps%d/%dx%d", bps, n, n), func(b *testing.B) {
				max := uint16(0xff)
				if bps == 2 {
					max = 0x3ff
				}
				dst := make([]byte, planeShapeBenchStride*n)
				for i := range dst {
					dst[i] = byte(i * 7)
					if bps == 2 && i&1 == 1 {
						dst[i] &= 3
					}
				}
				res := make([]int16, n*n)
				for i := range res {
					res[i] = int16((i*37)%61) - 30
				}
				b.ReportAllocs()
				for b.Loop() {
					AddResidualPlaneBlockTrusted(dst, planeShapeBenchStride, bps, max, n, n, res, n)
				}
			})
		}
	}
}

// BenchmarkPlaneRawShapes measures AddRawTransformPlaneBlockTrusted through
// the dispatched kernel for the square transform shapes the decoder uses.
func BenchmarkPlaneRawShapes(b *testing.B) {
	for _, bps := range []int{1, 2} {
		for _, n := range planeShapeBenchSizes {
			b.Run(fmt.Sprintf("bps%d/%dx%d", bps, n, n), func(b *testing.B) {
				max := uint16(0xff)
				if bps == 2 {
					max = 0x3ff
				}
				dst := make([]byte, planeShapeBenchStride*n)
				for i := range dst {
					dst[i] = byte(i * 7)
					if bps == 2 && i&1 == 1 {
						dst[i] &= 3
					}
				}
				raw := make([]int32, n*n)
				for i := range raw {
					raw[i] = int32((i*37)%977) - 488
				}
				b.ReportAllocs()
				for b.Loop() {
					AddRawTransformPlaneBlockTrusted(dst, planeShapeBenchStride, bps, max, n, n, raw, n)
				}
			})
		}
	}
}
