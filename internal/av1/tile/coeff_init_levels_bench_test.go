// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package tile

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func BenchmarkCoeffInitLevels(b *testing.B) {
	sizes := [...]TransformSize{
		TransformSize4x4,
		TransformSize8x8,
		TransformSize16x16,
		TransformSize32x32,
		TransformSize64x64,
		TransformSize4x16,
		TransformSize16x4,
		TransformSize8x32,
		TransformSize32x8,
	}
	for _, size := range sizes {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB := int(geo.maxEOB)
		scratchLen := int(geo.scratchLen)
		coeffs := make([]int16, maxEOB)
		levels := make([]uint8, scratchLen)
		for i := range coeffs {
			switch i & 7 {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(i%9 - 4)
			default:
				coeffs[i] = int16((i*53)%4095 - 2047)
			}
		}
		dims, _ := size.Dimensions()
		name := fmt.Sprintf("%dx%d_scan%dx%d", int(dims.W4)*4, int(dims.H4)*4, geo.scanWidth, geo.scanHeight)
		b.Run(name+"/impl", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := CoeffInitLevels(coeffs, size, levels); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/scalar", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				coeffInitLevelsPureGo(coeffs, int(geo.scanWidth), int(geo.scanHeight), levels, scratchLen)
			}
		})
	}
}

func BenchmarkCoeffNZMapContexts(b *testing.B) {
	sizes := [...]TransformSize{
		TransformSize16x16,
		TransformSize32x32,
		TransformSize64x64,
		TransformSize4x16,
		TransformSize8x32,
	}
	for _, size := range sizes {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB := int(geo.maxEOB)
		scratchLen := int(geo.scratchLen)
		coeffs := make([]int16, maxEOB)
		levels := make([]uint8, scratchLen)
		contexts := make([]int8, maxEOB)
		scan := coeffScanTable[size][transform.Class2D]
		for i := range coeffs {
			switch i & 7 {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(i%9 - 4)
			default:
				coeffs[i] = int16((i*53)%4095 - 2047)
			}
		}
		if err := CoeffInitLevels(coeffs, size, levels); err != nil {
			b.Fatal(err)
		}
		dims, _ := size.Dimensions()
		name := fmt.Sprintf("%dx%d_scan%dx%d", int(dims.W4)*4, int(dims.H4)*4, geo.scanWidth, geo.scanHeight)
		b.Run(name+"/impl-full", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := CoeffNZMapContexts(levels, size, transform.Class2D, scan, maxEOB, contexts); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/scalar-full", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := coeffNZMapContextsScalar(levels, size, transform.Class2D, scan, maxEOB, contexts, maxEOB); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
