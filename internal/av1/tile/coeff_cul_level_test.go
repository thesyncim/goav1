// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package tile

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffCulLevelMatchesScalar(t *testing.T) {
	rnd := newCoeffContextRandom(0x43554c31)
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB := int(geo.maxEOB)
		coeffs := make([]int16, maxEOB)
		for _, class := range classes {
			scan := coeffScanTable[size][class]
			if len(scan) < maxEOB {
				t.Fatalf("size=%d class=%d missing scan", size, class)
			}
			for iter := 0; iter < 32; iter++ {
				for i := range coeffs {
					switch rnd.u8() & 7 {
					case 0, 1:
						coeffs[i] = 0
					case 2:
						coeffs[i] = int16(int(rnd.u8()%7) - 3)
					case 3:
						coeffs[i] = int16(int(rnd.u16()%260) - 130)
					default:
						coeffs[i] = int16(int(rnd.u16()%8192) - 4096)
					}
				}
				for _, eob := range coeffCulLevelEOBCases(maxEOB) {
					got := coeffCulLevel(coeffs, scan, eob)
					want := coeffCulLevelScalar(coeffs, scan, eob)
					if got != want {
						t.Fatalf("size=%d class=%d iter=%d eob=%d cul=%d want %d", size, class, iter, eob, got, want)
					}
				}
			}
		}
	}
}

func BenchmarkCoeffCulLevel(b *testing.B) {
	sizes := [...]TransformSize{
		TransformSize4x4,
		TransformSize8x8,
		TransformSize16x16,
		TransformSize32x32,
		TransformSize64x64,
		TransformSize8x4,
		TransformSize16x8,
		TransformSize4x16,
		TransformSize8x32,
	}
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	rnd := newCoeffContextRandom(0x43554c42)
	for _, size := range sizes {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB := int(geo.maxEOB)
		coeffs := make([]int16, maxEOB)
		for i := range coeffs {
			switch rnd.u8() & 7 {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(int(rnd.u8()%7) - 3)
			default:
				coeffs[i] = int16(int(rnd.u16()%2048) - 1024)
			}
		}
		dims, _ := size.Dimensions()
		name := fmt.Sprintf("%dx%d_scan%dx%d", int(dims.W4)*4, int(dims.H4)*4, geo.scanWidth, geo.scanHeight)
		for _, class := range classes {
			class := class
			scan := coeffScanTable[size][class]
			if len(scan) < maxEOB {
				b.Fatalf("size=%d class=%d missing scan", size, class)
			}
			b.Run(name+"/"+coeffBenchClassName(class)+"/impl", func(b *testing.B) {
				var sink uint8
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					sink = coeffCulLevel(coeffs, scan, maxEOB)
				}
				if sink == 0xff {
					b.Fatal(sink)
				}
			})
			b.Run(name+"/"+coeffBenchClassName(class)+"/scalar", func(b *testing.B) {
				var sink uint8
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					sink = coeffCulLevelScalar(coeffs, scan, maxEOB)
				}
				if sink == 0xff {
					b.Fatal(sink)
				}
			})
			b.Run(name+"/"+coeffBenchClassName(class)+"/arch-direct", func(b *testing.B) {
				if _, ok := coeffCulLevelArch(coeffs, scan, maxEOB); !ok {
					b.Skip("arch path unavailable")
				}
				var sink uint8
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					v, ok := coeffCulLevelArch(coeffs, scan, maxEOB)
					if !ok {
						b.Fatal("arch path became unavailable")
					}
					sink = v
				}
				if sink == 0xff {
					b.Fatal(sink)
				}
			})
		}
	}
}

func coeffCulLevelEOBCases(maxEOB int) []int {
	cases := []int{0, 1}
	if maxEOB > 2 {
		cases = append(cases, 2)
	}
	if maxEOB > 3 {
		cases = append(cases, 3)
	}
	for _, eob := range []int{4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024} {
		if eob <= maxEOB {
			cases = append(cases, eob)
		}
	}
	if cases[len(cases)-1] != maxEOB {
		cases = append(cases, maxEOB)
	}
	return cases
}
