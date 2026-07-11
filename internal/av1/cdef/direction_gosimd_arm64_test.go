// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"reflect"
	"runtime"
	"testing"
)

// TestFindDirectionSIMDMatchesScalar fuzz-sweeps random blocks over every
// coeffShift and stride and asserts the Go-native SIMD kernel is byte-identical
// to the scalar reference (which matches libaom's cdef_find_dir_c).
func TestFindDirectionSIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x53494d44)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{8, 9, 13, 16, 31} {
			for iter := range 256 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
				gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
				if gotDir != wantDir || gotVar != wantVar {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dir,var=%d,%d want %d,%d",
						coeffShift, stride, iter, gotDir, gotVar, wantDir, wantVar)
				}
			}
		}
	}
}

// TestFindDirectionSIMDEdges covers all-flat, min/max-contrast and each direction
// dominant so the argmax + tie-break and the variance paths are all exercised.
func TestFindDirectionSIMDEdges(t *testing.T) {
	const stride = 8
	patterns := []func(row, col int) uint16{
		func(row, col int) uint16 { return 128 },                            // flat -> dir 0, var 0
		func(row, col int) uint16 { return 0 },                              // min flat
		func(row, col int) uint16 { return 255 },                           // max flat (shift 0)
		func(row, col int) uint16 { return uint16((row + col) * 16) },       // diag 0
		func(row, col int) uint16 { return uint16((2*row + col) * 8) },      // dir 1
		func(row, col int) uint16 { return uint16(row * 32) },               // horizontal 2
		func(row, col int) uint16 { return uint16((2*row - col + 8) * 8) },  // dir 3
		func(row, col int) uint16 { return uint16((row - col + 8) * 16) },   // anti-diag 4
		func(row, col int) uint16 { return uint16((row - 2*col + 16) * 8) }, // dir 5
		func(row, col int) uint16 { return uint16(col * 32) },               // vertical 6
		func(row, col int) uint16 { return uint16((-row + 2*col + 8) * 8) }, // dir 7
	}
	for coeffShift := range 5 {
		maxSample := uint16((1 << (8 + coeffShift)) - 1)
		for pi, fill := range patterns {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					v := uint32(fill(row, col)) << coeffShift
					if v > uint32(maxSample) {
						v = uint32(maxSample)
					}
					img[row*stride+col] = uint16(v)
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("pattern=%d coeffShift=%d dir,var=%d,%d want %d,%d",
					pi, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

// TestFindDirectionSIMDMaxContrast drives extreme checkerboards (full-range) so
// the partial sums reach their largest magnitudes and any int32 overflow in the
// cost arithmetic must match the scalar reference bit-for-bit.
func TestFindDirectionSIMDMaxContrast(t *testing.T) {
	const stride = 8
	for coeffShift := range 5 {
		maxSample := uint16((1 << (8 + coeffShift)) - 1)
		for variant := 0; variant < 4; variant++ {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					var hi bool
					switch variant {
					case 0:
						hi = (row+col)&1 == 0
					case 1:
						hi = row&1 == 0
					case 2:
						hi = col&1 == 0
					case 3:
						hi = row < 4
					}
					if hi {
						img[row*stride+col] = maxSample
					}
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("variant=%d coeffShift=%d dir,var=%d,%d want %d,%d",
					variant, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionSIMDZeroAlloc(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = findDirectionSIMD(img, 8, 4)
	})
	if allocs != 0 {
		t.Fatalf("findDirectionSIMD allocated: %f", allocs)
	}
}

func BenchmarkFindDirectionSIMD(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = findDirectionSIMD(img, 8, 4)
	}
}

func BenchmarkFindDirectionSIMD8(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = findDirectionSIMD(img, 8, 0)
	}
}

// BenchmarkFindDirectionNEONAsm benches the hand NEON asm kernel directly for a
// like-for-like comparison against BenchmarkFindDirectionSIMD.
func BenchmarkFindDirectionNEONAsm(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = findDirectionNEON(img, 8, 4)
	}
}

// TestFindDirectionU8SIMDMatchesScalar is the 3-way u8 differential: the
// Go-native SIMD u8 direction search and the NEON asm must both match the
// scalar reference (direction + variance) across strides and content regimes,
// including flat, near-max and full-random blocks.
func TestFindDirectionU8SIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x11223344)
	for _, stride := range []int{8, 17, 160, 640} {
		for iter := range 256 {
			img := make([]byte, stride*8+8)
			for i := range img {
				switch iter % 4 {
				case 0:
					img[i] = byte(rnd.generate(256))
				case 1:
					img[i] = byte(rnd.generate(6))
				case 2:
					img[i] = byte(250 + rnd.generate(6))
				default:
					if (i/stride)%2 == 0 {
						img[i] = 255
					}
				}
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			asmDir, asmVar := findDirectionU8NEON(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("SIMD stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if asmDir != wantDir || asmVar != wantVar {
				t.Fatalf("NEON stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, asmDir, asmVar, wantDir, wantVar)
			}
		}
	}
}

// TestFindDirectionU8SIMDTightTail pins the row-7 hi-half load: the block's
// last row ends exactly at the end of the backing slice (no byte after it),
// so any overreading load would fault or diverge.
func TestFindDirectionU8SIMDTightTail(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x777)
	for _, stride := range []int{8, 16, 33} {
		for iter := 0; iter < 64; iter++ {
			img := make([]byte, 7*stride+8) // last row has exactly 8 bytes
			for i := range img {
				img[i] = byte(rnd.generate(256))
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionU8SIMDDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(findDirectionU8Impl), nameOf(findDirectionU8SIMD); got != want {
		t.Errorf("findDirectionU8Impl = %s, want %s", got, want)
	}
	if got, want := nameOf(findDirectionDualU8Impl), nameOf(findDirectionDualU8SIMD); got != want {
		t.Errorf("findDirectionDualU8Impl = %s, want %s", got, want)
	}
}

func TestFindDirectionU8SIMDZeroAlloc(t *testing.T) {
	img := make([]byte, 640*8+8)
	rnd := newCDEFRandom(1)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	if a := testing.AllocsPerRun(50, func() { findDirectionU8SIMD(img, 640) }); a != 0 {
		t.Errorf("findDirectionU8SIMD allocated %.1f objects/run, want 0", a)
	}
}

func BenchmarkFindDirectionU8NEON(b *testing.B) {
	img := make([]byte, 640*8+8)
	rnd := newCDEFRandom(2)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	b.ReportAllocs()
	for b.Loop() {
		findDirectionU8NEON(img, 640)
	}
}

func BenchmarkFindDirectionU8SIMD(b *testing.B) {
	img := make([]byte, 640*8+8)
	rnd := newCDEFRandom(2)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	b.ReportAllocs()
	for b.Loop() {
		findDirectionU8SIMD(img, 640)
	}
}
