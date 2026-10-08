//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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

// TestCompound2D8GoSIMDMatchesPureGo is the 3-way differential for the 8-bit
// compound 2D CONV_BUF pass: the Go-native SIMD kernel and the I8MM/NEON asm
// front door must both match the pure-Go reference over every (subX, subY)
// phase pair spread, all filter tables (regular/sharp/smooth/4-tap/bilinear —
// the sharp family exercises the nonzero-f0 H pass), resident and fallback
// shapes.
func TestCompound2D8GoSIMDMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {24, 32}, {32, 16}, {64, 64}, {40, 8}, {4, 4}, {4, 8},
	}
	const offsetBits = 19
	var scratch CompoundConvolveScratch
	for ti, table := range tables {
		for _, size := range sizes {
			for _, phase := range [][2]int{{1, 1}, {7, 7}, {15, 15}, {1, 15}, {15, 1}, {8, 8}, {3, 12}} {
				xKernel := table[phase[0]]
				yKernel := table[phase[1]]
				pad := filterTaps
				refSide := size.w + 2*pad
				refH := size.h + 2*pad
				ref, _ := testPlane(refSide, refH, 1, refSide)
				fillMotionTestPlane(ref)
				want := make([]uint16, size.w*size.h)
				got := make([]uint16, size.w*size.h)
				asm := make([]uint16, size.w*size.h)
				predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, pad, pad, size.w, size.h, xKernel, yKernel, offsetBits, nil)
				compound2D8GoSIMD(got, ref, pad, pad, size.w, size.h, xKernel, yKernel, offsetBits, &scratch)
				predictInterCompoundRef8ToConvBuf2DI8MM(asm, ref, pad, pad, size.w, size.h, xKernel, yKernel, offsetBits, &scratch)
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("SIMD table %d size %dx%d phase %v: mismatch at %d: got %d want %d", ti, size.w, size.h, phase, i, got[i], want[i])
					}
					if want[i] != asm[i] {
						t.Fatalf("ASM table %d size %dx%d phase %v: mismatch at %d: got %d want %d", ti, size.w, size.h, phase, i, asm[i], want[i])
					}
				}
			}
		}
	}
}

// TestCompound2D8GoSIMDEdgeOverhang pins the edge-overhanging (extreme-MV)
// windows: references clamped past every plane border route through the
// emu-edge fallback and must stay byte-exact.
func TestCompound2D8GoSIMDEdgeOverhang(t *testing.T) {
	const offsetBits = 19
	var scratch CompoundConvolveScratch
	ref, _ := testPlane(48, 48, 1, 48)
	fillMotionTestPlane(ref)
	xKernel := subpelFilters8[5]
	yKernel := subpelFilters8Sharp[11]
	positions := [][2]int{{-20, -20}, {-3, 10}, {10, -3}, {44, 44}, {60, 60}, {44, 10}, {10, 44}, {-20, 44}}
	for _, pos := range positions {
		for _, size := range []struct{ w, h int }{{8, 8}, {16, 16}, {32, 8}} {
			want := make([]uint16, size.w*size.h)
			got := make([]uint16, size.w*size.h)
			predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, pos[0], pos[1], size.w, size.h, xKernel, yKernel, offsetBits, nil)
			compound2D8GoSIMD(got, ref, pos[0], pos[1], size.w, size.h, xKernel, yKernel, offsetBits, &scratch)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("pos %v size %dx%d: mismatch at %d: got %d want %d", pos, size.w, size.h, i, got[i], want[i])
				}
			}
		}
	}
}

// TestCompound2D8GoSIMDExtremePixels drives min/max content through both
// H-pass variants (f0 zero and nonzero) so the intermediate hits its domain
// ceiling and floor.
func TestCompound2D8GoSIMDExtremePixels(t *testing.T) {
	const offsetBits = 19
	var scratch CompoundConvolveScratch
	fills := []func(x, y int) byte{
		func(x, y int) byte { return 255 },
		func(x, y int) byte { return 0 },
		func(x, y int) byte { return byte((x % 2) * 255) },
		func(x, y int) byte { return byte((y % 2) * 255) },
	}
	kernels := [][2][filterTaps]int16{
		{subpelFilters8[8], subpelFilters8[8]},
		{subpelFilters8Sharp[8], subpelFilters8Sharp[8]}, // nonzero f0
		{subpelFilters8Sharp[15], subpelFilters8Sharp[1]},
	}
	for fi, fill := range fills {
		ref, _ := testPlane(48, 48, 1, 48)
		for y := 0; y < 48; y++ {
			for x := 0; x < 48; x++ {
				ref.Pix[y*48+x] = fill(x, y)
			}
		}
		for ki, k := range kernels {
			want := make([]uint16, 16*16)
			got := make([]uint16, 16*16)
			predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, 16, 16, 16, 16, k[0], k[1], offsetBits, nil)
			compound2D8GoSIMD(got, ref, 16, 16, 16, 16, k[0], k[1], offsetBits, &scratch)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("fill %d kernel %d: mismatch at %d: got %d want %d", fi, ki, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompound2D8GoSIMDZeroAlloc(t *testing.T) {
	ref, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(ref)
	out := make([]uint16, 32*32)
	var scratch CompoundConvolveScratch
	xKernel := subpelFilters8[5]
	yKernel := subpelFilters8[9]
	if a := testing.AllocsPerRun(20, func() {
		compound2D8GoSIMD(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, 19, &scratch)
	}); a != 0 {
		t.Errorf("compound2D8GoSIMD allocated %.1f objects/run, want 0", a)
	}
}

func benchCompound2D(b *testing.B, w, h int, fn func([]uint16, frame.Plane, int, int, int, int, [filterTaps]int16, [filterTaps]int16, int, *CompoundConvolveScratch)) {
	_, ref := benchPlanes(w, 8)
	var buf CompoundConvBuf
	out, ok := compoundConvBufView(&buf, w, h)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xKernel, err := interpKernel(InterpEightTapRegular, 32, 3)
	if err != nil {
		b.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapRegular, 32, 11)
	if err != nil {
		b.Fatal(err)
	}
	var scratch CompoundConvolveScratch
	runConvolveBench(b, w, h, func() {
		fn(out, ref, filterTaps, filterTaps, w, h, xKernel, yKernel, 19, &scratch)
	})
}

func BenchmarkCompound2D8I8MM_32(b *testing.B) {
	benchCompound2D(b, 32, 32, predictInterCompoundRef8ToConvBuf2DI8MM)
}

func BenchmarkCompound2D8GoSIMD_32(b *testing.B) {
	benchCompound2D(b, 32, 32, compound2D8GoSIMD)
}

func BenchmarkCompound2D8I8MM_16(b *testing.B) {
	benchCompound2D(b, 16, 16, predictInterCompoundRef8ToConvBuf2DI8MM)
}

func BenchmarkCompound2D8GoSIMD_16(b *testing.B) {
	benchCompound2D(b, 16, 16, compound2D8GoSIMD)
}

func BenchmarkCompound2D8I8MM_8(b *testing.B) {
	benchCompound2D(b, 8, 8, predictInterCompoundRef8ToConvBuf2DI8MM)
}

func BenchmarkCompound2D8GoSIMD_8(b *testing.B) {
	benchCompound2D(b, 8, 8, compound2D8GoSIMD)
}

// TestCompound2D8GoSIMDDispatchBound is the FuncForPC probe for the 2D
// CONV_BUF dispatch slot under the goexperiment.simd build.
func TestCompound2D8GoSIMDDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(predictInterCompoundRef8ToConvBuf2DImpl), nameOf(compound2D8GoSIMD); got != want {
		t.Errorf("predictInterCompoundRef8ToConvBuf2DImpl = %s, want %s", got, want)
	}
}
