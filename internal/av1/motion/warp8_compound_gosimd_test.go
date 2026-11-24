// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// warpCompoundScalar runs the public compound warp with the Go SIMD kernels
// off: the pure-Go oracle.
func warpCompoundScalar(t *testing.T, buf *CompoundConvBuf, ref frame.Plane, matX, matY, w, h int, m [6]int32, a, b, g, d int16, ssX, ssY bool) error {
	t.Helper()
	saved := warpGoSIMDEnabled
	warpGoSIMDEnabled = false
	defer func() { warpGoSIMDEnabled = saved }()
	return PredictWarpedCompoundToConvBuf(buf, ref, 1, 8, matX, matY, w, h, m, a, b, g, d, ssX, ssY)
}

func warpCompoundRef(rng *rand.Rand, w, h, pattern int) frame.Plane {
	stride := w + rng.Intn(3)*8
	ref := frame.Plane{Pix: make([]byte, stride*h), Stride: stride, Width: w, Height: h}
	for y := range h {
		for x := range w {
			var v byte
			switch pattern {
			case 0:
				v = byte(rng.Intn(256))
			case 1:
				v = 255
			case 2:
				v = 0
			default:
				v = byte(((x + y) & 1) * 255)
			}
			ref.Pix[y*stride+x] = v
		}
	}
	return ref
}

// TestWarpCompound8GoSIMDMatchesScalar compares the public compound warp with
// and without the Go SIMD kernel over random warp parameters (including ones
// that push filter indices out of range, which must fall back), every block
// size, interior, edge and far-outside positions, chroma subsampling and
// extreme sample patterns. It also checks the SIMD kernel actually ran.
func TestWarpCompound8GoSIMDMatchesScalar(t *testing.T) {
	if !warpGoSIMDEnabled {
		t.Skip("Go SIMD warp disabled")
	}
	rng := rand.New(rand.NewSource(7))
	sizes := [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 32}, {64, 64}, {128, 64}, {128, 128}, {8, 32}, {32, 8}, {4, 4}, {12, 8}}
	used, fell := 0, 0
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		iters := 160
		if w*h >= 64*64 {
			iters = 24
		}
		for iter := range iters {
			pattern := iter % 4
			ssX, ssY := iter%3 == 1, iter%5 == 2
			rw, rh := 24+rng.Intn(160), 24+rng.Intn(160)
			ref := warpCompoundRef(rng, rw, rh, pattern)
			// Positions span the interior, all four edges and well outside.
			matX := rng.Intn(rw+64) - 32
			matY := rng.Intn(rh+64) - 32
			span, div := 8192, 8
			if iter%7 == 0 {
				span, div = 1<<15, 1 // drives some filter indices out of range
			}
			r := func() int32 { return int32(rng.Intn(span) - span/2) }
			matrix := [6]int32{
				int32(rng.Intn(1<<20)) - 1<<19,
				int32(rng.Intn(1<<20)) - 1<<19,
				1<<16 + r(), r(), r(), 1<<16 + r(),
			}
			alpha, beta := warpReduce(int(r())/div), warpReduce(int(r())/div)
			gamma, delta := warpReduce(int(r())/div), warpReduce(int(r())/div)
			if iter%6 == 3 {
				gamma = 0
			}
			if iter%11 == 5 {
				alpha, beta, gamma, delta = 0, 0, 0, 0
			}
			var want, got CompoundConvBuf
			werr := warpCompoundScalar(t, &want, ref, matX, matY, w, h, matrix, alpha, beta, gamma, delta, ssX, ssY)
			gerr := PredictWarpedCompoundToConvBuf(&got, ref, 1, 8, matX, matY, w, h, matrix, alpha, beta, gamma, delta, ssX, ssY)
			name := fmt.Sprintf("%dx%d iter=%d", w, h, iter)
			if (werr == nil) != (gerr == nil) {
				t.Fatalf("%s: error mismatch %v / %v", name, werr, gerr)
			}
			for k := range w * h {
				if got.Data[k] != want.Data[k] {
					t.Fatalf("%s: sample %d (%d,%d) got %d want %d", name, k, k%w, k/w, got.Data[k], want.Data[k])
				}
			}
			if w >= 8 && h >= 8 {
				sx := 0
				if ssX {
					sx = 1
				}
				sy := 0
				if ssY {
					sy = 1
				}
				ix4, sx4, iy4, sy4 := warpBlockOrigin(matY, matX, matrix, int(alpha), int(beta), int(gamma), int(delta), sx, sy)
				var probe [64]uint16
				if warpCompound8x8Dispatch(probe[:], 8, ref, ix4, sx4, iy4, sy4, int(alpha), int(beta), int(gamma), int(delta)) {
					used++
				} else {
					fell++
				}
			}
		}
	}
	if used == 0 || fell == 0 {
		t.Fatalf("kernel coverage: used=%d fell back=%d", used, fell)
	}
	t.Logf("first blocks: SIMD %d, scalar fallback %d", used, fell)
}

// warpCompound8x8Scalar is the scalar compound 8x8 block from the resident
// horizontal pass: the oracle for direct kernel calls.
func warpCompound8x8Scalar(out *[64]uint16, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) {
	var tmp warpTmp
	warpHorizontal8Resident(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, round0Bits, 8+filterBits-1)
	for r := range 8 {
		for c := range 8 {
			coeffs := warpedFilter[roundPowerOfTwo(sy4+delta*r+gamma*c, warpedDiffPrecBits)+warpedPixelPrecShifts]
			sum := 1 << (8 + 2*filterBits - round0Bits)
			for m := range filterTaps {
				sum += int(coeffs[m]) * int(tmp[(r+m)*warpedIntermediateColumns+c])
			}
			out[r*8+c] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
		}
	}
}

// TestWarpCompound8GoSIMDAllPhases drives every filter row through both
// passes (horizontal with alpha = beta = 0, vertical with gamma = delta = 0,
// then both swept together) with extreme sample patterns.
func TestWarpCompound8GoSIMDAllPhases(t *testing.T) {
	if !warpGoSIMDEnabled {
		t.Skip("Go SIMD warp disabled")
	}
	rng := rand.New(rand.NewSource(3))
	for pattern := range 4 {
		ref := warpCompoundRef(rng, 48, 48, pattern)
		for hp := range warpedFilter {
			for _, vp := range []int{hp, 0, len(warpedFilter) - 1, 64} {
				sx4 := (hp - warpedPixelPrecShifts) << warpedDiffPrecBits
				sy4 := (vp - warpedPixelPrecShifts) << warpedDiffPrecBits
				var want, got [64]uint16
				warpCompound8x8Scalar(&want, ref, 20, sx4, 20, sy4, 0, 0, 0, 0)
				if !warpCompound8x8Dispatch(got[:], 8, ref, 20, sx4, 20, sy4, 0, 0, 0, 0) {
					t.Fatalf("pattern=%d phases=%d/%d: kernel declined", pattern, hp, vp)
				}
				if got != want {
					t.Fatalf("pattern=%d phases=%d/%d mismatch", pattern, hp, vp)
				}
			}
		}
	}
}

// TestWarpCompound8GoSIMDNoAllocs covers the interior and the edge-copy path.
func TestWarpCompound8GoSIMDNoAllocs(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ref := warpCompoundRef(rng, 64, 64, 0)
	matrix := [6]int32{0, 0, 1<<16 + 300, 200, -100, 1<<16 - 300}
	a, b, g, d := warpReduce(300), warpReduce(200), warpReduce(-100), warpReduce(-300)
	var buf CompoundConvBuf
	for _, pos := range [][2]int{{24, 24}, {0, 0}, {56, 56}, {-20, 30}} {
		n := testing.AllocsPerRun(50, func() {
			_ = PredictWarpedCompoundToConvBuf(&buf, ref, 1, 8, pos[0], pos[1], 16, 16, matrix, a, b, g, d, false, false)
		})
		if n != 0 {
			t.Fatalf("pos %v: %v allocs per call", pos, n)
		}
	}
}

func BenchmarkWarpCompound8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	ref := warpCompoundRef(rng, 256, 256, 0)
	matrix := [6]int32{64 << 16, 64 << 16, 1<<16 + 900, -700, 500, 1<<16 - 400}
	alpha, beta, gamma, delta := warpReduce(900), warpReduce(-700), warpReduce(500), warpReduce(-400)
	for _, simd := range []bool{false, true} {
		b.Run(fmt.Sprintf("simd=%v", simd), func(b *testing.B) {
			saved := warpGoSIMDEnabled
			warpGoSIMDEnabled = simd
			defer func() { warpGoSIMDEnabled = saved }()
			var buf CompoundConvBuf
			for b.Loop() {
				_ = PredictWarpedCompoundToConvBuf(&buf, ref, 1, 8, 64, 64, 16, 16, matrix, alpha, beta, gamma, delta, false, false)
			}
		})
	}
}
