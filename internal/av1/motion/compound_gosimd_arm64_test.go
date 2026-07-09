//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

// TestCompoundX8GoSIMDMatchesPureGo checks the Go-native SIMD 8-bit compound
// horizontal CONV_BUF pass is byte-identical to the pure-Go reference over every
// subpel phase and a range of resident block shapes (the SIMD-covered case), and
// that unsupported shapes fall back byte-exactly.
func TestCompoundX8GoSIMDMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {16, 8}, {24, 32}, {32, 32}, {32, 16}, {64, 64}, {40, 8}, {4, 4},
	}
	roundOffset := compoundRoundOffset8()
	for ti, table := range tables {
		for _, size := range sizes {
			for subX := 0; subX < 16; subX++ {
				kernel := table[subX]
				pad := filterTaps
				refSide := size.w + 2*pad
				refH := size.h + 2*pad
				ref, _ := testPlane(refSide, refH, 1, refSide)
				fillMotionTestPlane(ref)
				want := make([]uint16, size.w*size.h)
				got := make([]uint16, size.w*size.h)
				predictInterCompoundRef8ToConvBufXPureGo(want, ref, pad, pad, size.w, size.h, kernel, roundOffset)
				compoundX8GoSIMD(got, ref, pad, pad, size.w, size.h, kernel, roundOffset)
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("table %d size %dx%d subX %d: mismatch at %d: got %d want %d", ti, size.w, size.h, subX, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkCompoundConvBufX8GoSIMD_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 3)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		compoundX8GoSIMD(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
}
