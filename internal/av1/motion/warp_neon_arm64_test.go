// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestWarpedFilterI8Exact guards the two properties the NEON warp math depends
// on: the int8 narrow of warpedFilter is lossless, and every row still sums to
// 128 (the identity behind the horizontal SMULL bias fold).
func TestWarpedFilterI8Exact(t *testing.T) {
	for i := range warpedFilter {
		sum := 0
		for j := range warpedFilter[i] {
			if int16(warpedFilterI8[i][j]) != warpedFilter[i][j] {
				t.Fatalf("warpedFilterI8[%d][%d]=%d lossy vs %d", i, j, warpedFilterI8[i][j], warpedFilter[i][j])
			}
			sum += int(warpedFilter[i][j])
		}
		if sum != 128 {
			t.Fatalf("warpedFilter[%d] sums to %d, want 128", i, sum)
		}
	}
}

// residentRefPlane builds an 8-bit reference plane large enough that the
// resident horizontal window (and its one-byte right overshoot) is always in
// bounds for interior positions.
func residentRefPlane(rng *rand.Rand) frame.Plane {
	const w, h, stride = 80, 80, 96
	pix := make([]byte, stride*(h+2)) // +2 rows of border for the 16-byte load overshoot
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

func TestWarpHorizontal8ResidentNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5A1D0FF))
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1

	tested := 0
	for draw := 0; draw < 20000 && tested < 1500; draw++ {
		ref := residentRefPlane(rng)
		ix4 := 7 + rng.Intn(ref.Width-15) // keeps ix4>=7 and ix4+8<=Width
		iy4 := 7 + rng.Intn(ref.Height-15)
		sx4 := rng.Intn(1 << 16)
		sy4 := rng.Intn(1 << 16)
		alpha := rng.Intn(513) - 256
		beta := rng.Intn(513) - 256

		if !warpHorizResidentOffsInRange(sx4, alpha, beta) {
			continue
		}
		tested++

		var want, got warpTmp
		wantSY := warpHorizontal8Resident(&want, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
		gotSY := warpHorizontal8ResidentNEON(&got, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)

		if gotSY != wantSY {
			t.Fatalf("draw %d: sy4 got %d want %d", draw, gotSY, wantSY)
		}
		if got != want {
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("draw %d ix4=%d iy4=%d sx4=%d alpha=%d beta=%d: tmp[%d]=%d want %d",
						draw, ix4, iy4, sx4, alpha, beta, i, got[i], want[i])
				}
			}
		}
	}
	if tested == 0 {
		t.Fatal("no in-range horizontal cases exercised the NEON path")
	}
	t.Logf("compared %d in-range resident horizontal blocks", tested)
}

func TestWarpVertical8FullNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xBEEF77))
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits

	testedFull, testedGamma0 := 0, 0
	for draw := 0; draw < 40000 && (testedFull < 1200 || testedGamma0 < 400); draw++ {
		// Produce a realistic int32 tmp from a resident horizontal pass.
		ref := residentRefPlane(rng)
		ix4 := 7 + rng.Intn(ref.Width-15)
		iy4 := 7 + rng.Intn(ref.Height-15)
		hsx4 := rng.Intn(1 << 16)
		halpha := rng.Intn(257) - 128
		hbeta := rng.Intn(257) - 128
		if !warpHorizResidentOffsInRange(hsx4, halpha, hbeta) {
			continue
		}
		var tmp warpTmp
		warpHorizontal8Resident(&tmp, ref, ix4, hsx4, iy4, 0, halpha, hbeta, reduceBitsHoriz, offsetBitsHoriz)

		baseSY := rng.Intn(1 << 16)
		gamma := rng.Intn(257) - 128
		delta := rng.Intn(513) - 256

		gamma0 := draw%4 == 0
		if gamma0 {
			gamma = 0
		}
		if !warpVertFullOffsInRange(baseSY, gamma, delta) {
			continue
		}

		want, _ := testPlane(32, 32, 1, 32)
		got, _ := testPlane(32, 32, 1, 32)
		for i := range want.Pix {
			want.Pix[i] = 0xAA
			got.Pix[i] = 0xAA
		}

		if gamma0 {
			warpVertical8FullGamma0(want, &tmp, 8, 8, 0, 0, baseSY, delta, reduceBitsVert, offsetBitsVert)
			warpVertical8FullGamma0NEON(got, &tmp, 8, 8, 0, 0, baseSY, delta, reduceBitsVert, offsetBitsVert)
			testedGamma0++
		} else {
			warpVertical8Full(want, &tmp, 8, 8, 0, 0, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
			warpVertical8FullNEON(got, &tmp, 8, 8, 0, 0, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
			testedFull++
		}

		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("draw %d gamma0=%v baseSY=%d gamma=%d delta=%d pix[%d]=%d want %d",
					draw, gamma0, baseSY, gamma, delta, i, got.Pix[i], want.Pix[i])
			}
		}
	}
	if testedFull == 0 || testedGamma0 == 0 {
		t.Fatalf("insufficient vertical coverage: full=%d gamma0=%d", testedFull, testedGamma0)
	}
	t.Logf("compared %d full + %d gamma0 vertical blocks", testedFull, testedGamma0)
}

func benchWarpHorizInputs() (frame.Plane, int, int, int, int, int, int) {
	rng := rand.New(rand.NewSource(1))
	ref := residentRefPlane(rng)
	return ref, 40, 40, 32768, 0, 96, -64
}

func BenchmarkWarpHorizontal8ResidentScalar(b *testing.B) {
	ref, ix4, iy4, sx4, sy4, alpha, beta := benchWarpHorizInputs()
	var tmp warpTmp
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpHorizontal8Resident(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, round0Bits, 8+filterBits-1)
	}
}

func BenchmarkWarpHorizontal8ResidentNEON(b *testing.B) {
	ref, ix4, iy4, sx4, sy4, alpha, beta := benchWarpHorizInputs()
	if !warpHorizResidentOffsInRange(sx4, alpha, beta) {
		b.Skip("bench inputs out of range")
	}
	var tmp warpTmp
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpHorizontal8ResidentNEON(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, round0Bits, 8+filterBits-1)
	}
}

func benchWarpVertTmp() warpTmp {
	rng := rand.New(rand.NewSource(2))
	ref := residentRefPlane(rng)
	var tmp warpTmp
	warpHorizontal8Resident(&tmp, ref, 40, 32768, 40, 0, 96, -64, round0Bits, 8+filterBits-1)
	return tmp
}

func BenchmarkWarpVertical8FullScalar(b *testing.B) {
	tmp := benchWarpVertTmp()
	dst, _ := testPlane(32, 32, 1, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8Full(dst, &tmp, 8, 8, 0, 0, 32768, 96, -64, round1Bits, 8+2*filterBits-round0Bits)
	}
}

func BenchmarkWarpVertical8FullNEON(b *testing.B) {
	tmp := benchWarpVertTmp()
	if !warpVertFullOffsInRange(32768, 96, -64) {
		b.Skip("bench inputs out of range")
	}
	dst, _ := testPlane(32, 32, 1, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8FullNEON(dst, &tmp, 8, 8, 0, 0, 32768, 96, -64, round1Bits, 8+2*filterBits-round0Bits)
	}
}
