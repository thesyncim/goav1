//go:build goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"bytes"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"testing"
)

// TestCoeffInitLevelsSIMDRandomMatchesPureGo drives the architecture's Go SIMD
// level-init kernel directly over randomized full-range int16 coefficients and
// saturation edges, for every valid geometry with a SIMD-supported height, and
// compares it byte-for-byte with the pure-Go reference.
func TestCoeffInitLevelsSIMDRandomMatchesPureGo(t *testing.T) {
	rnd := newCoeffContextRandom(0x494e4c31)
	edges := [...]int16{-32768, -32767, -16384, -129, -128, -127, -1, 0, 1, 126, 127, 128, 129, 255, 256, 32767}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		switch geo.scanHeight {
		case 4, 8, 16, 32:
		default:
			continue
		}
		maxEOB := int(geo.maxEOB)
		scratchLen := int(geo.scratchLen)
		coeffs := make([]int16, maxEOB)
		got := make([]uint8, scratchLen)
		want := make([]uint8, scratchLen)
		for iter := 0; iter < 64; iter++ {
			for i := range coeffs {
				switch rnd.u8() & 3 {
				case 0:
					coeffs[i] = edges[int(rnd.u8())%len(edges)]
				case 1:
					coeffs[i] = int16(rnd.u16()%512) - 256
				default:
					coeffs[i] = int16(rnd.u16())
				}
			}
			for i := range got {
				got[i] = rnd.u8()
				want[i] = got[i]
			}
			clear(got)
			coeffInitLevelsSIMD(coeffs, int(geo.scanWidth), int(geo.scanHeight), got)
			coeffInitLevelsPureGo(coeffs, int(geo.scanWidth), int(geo.scanHeight), want, scratchLen)
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("size=%d iter=%d levels[%d]=%d want %d", size, iter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestCoeffNZMapContextsSIMDFullMatchesScalar drives the Go SIMD full-map kernel
// directly (no CPU gate, so it runs on every host the architecture build
// supports) and compares every dense context with the scalar reference.
func TestCoeffNZMapContextsSIMDFullMatchesScalar(t *testing.T) {
	rnd := newCoeffContextRandom(0x4e5a4d42)
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid || (geo.scanHeight != 4 && geo.scanHeight != 8 && geo.scanHeight != 16 && geo.scanHeight != 32) {
			continue
		}
		maxEOB := int(geo.maxEOB)
		levels := make([]uint8, int(geo.scratchLen))
		for _, class := range classes {
			for iter := 0; iter < 32; iter++ {
				for i := range levels {
					switch iter & 3 {
					case 0:
						levels[i] = rnd.u8()
					case 1:
						levels[i] = rnd.u8() & 0x7f
					default:
						levels[i] = rnd.u8() & 3
					}
				}
				got := make([]int8, maxEOB)
				want := make([]int8, maxEOB)
				for i := range got {
					got[i] = int8(rnd.u8())
				}
				// The per-position reference: the eob-tail rule in the scalar
				// scan loop does not apply to the full map, so query each
				// raster position directly.
				for pos := range want {
					ctx, err := CoeffLowerLevelsContext(levels, size, class, pos)
					if err != nil {
						t.Fatal(err)
					}
					want[pos] = int8(ctx)
				}
				if !coeffNZMapContextsSIMDFull(levels, size, class, got) {
					t.Fatalf("size=%d class=%d did not use SIMD full map", size, class)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("size=%d class=%d iter=%d context[%d]=%d want %d", size, class, iter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestCoeffNZMapContextsSIMDFullZeroAlloc covers the plain shape and the
// windowed tail groups of the 8-row and 4-row paths (Horiz reaches the furthest
// neighbour, so its last group is staged through the window).
func TestCoeffNZMapContextsSIMDFullZeroAlloc(t *testing.T) {
	shapes := [...]struct {
		size  TransformSize
		class transform.Class
	}{
		{TransformSize32x32, transform.Class2D},
		{TransformSize8x8, transform.ClassHoriz},
		{TransformSize4x4, transform.ClassHoriz},
		{TransformSize16x8, transform.ClassHoriz},
	}
	for _, s := range shapes {
		geo := coeffGeometryTable[s.size]
		levels := make([]uint8, int(geo.scratchLen))
		for i := range levels {
			levels[i] = uint8(i*37) & 0x7f
		}
		contexts := make([]int8, int(geo.maxEOB))
		allocs := testing.AllocsPerRun(200, func() {
			coeffNZMapContextsSIMDFull(levels, s.size, s.class, contexts)
		})
		if allocs != 0 {
			t.Fatalf("coeffNZMapContextsSIMDFull size=%d class=%d allocs = %v, want 0", s.size, s.class, allocs)
		}
	}
}

// BenchmarkCoeffNZMapContexts2D times the encoder's lower-context maps.
func BenchmarkCoeffNZMapContexts2D(b *testing.B) {
	for _, size := range [...]TransformSize{TransformSize16x16, TransformSize32x32} {
		geo := coeffGeometryTable[size]
		levels := make([]uint8, int(geo.scratchLen))
		for i := range levels {
			levels[i] = uint8(i*37) & 3
		}
		contexts := make([]int8, int(geo.maxEOB))
		b.Run(fmt.Sprintf("%dx%d", geo.scanWidth, geo.scanHeight), func(b *testing.B) {
			for b.Loop() {
				coeffNZMapContextsSIMDFull(levels, size, transform.Class2D, contexts)
			}
		})
	}
}
