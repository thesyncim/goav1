//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"slices"
	"testing"
)

// These AVX2 Hadamard tests call the kernels directly (never gated on
// cpu.Detected.AVX2) so hosts that hide AVX2 (Rosetta 2) still prove
// bit-exactness against the accepted portable/NEON reference orders.

func hadamardTestSrc(rng *rand.Rand, stride, height int) []int16 {
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	return src
}

func TestHadamard4x4AVX2MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4A4))
	const stride, height = 17, 14
	src := hadamardTestSrc(rng, stride, height)
	for range 2000 {
		row := rng.Intn(height - 4)
		col := rng.Intn(stride - 4)
		for r := range 4 {
			for c := range 4 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var wantC, wantNEON, got [16]int32
		off := row*stride + col
		hadamard4x4PureGo(src[off:], stride, wantC[:])
		hadamard4x4SVTNEONReference(src[off:], stride, wantNEON[:])
		hadamard4x4AVX2(src[off:], stride, got[:])
		if !slices.Equal(got[:], wantC[:]) && !slices.Equal(got[:], wantNEON[:]) {
			t.Fatalf("off=%d got %v wantC %v wantNEON %v", off, got, wantC, wantNEON)
		}
		if satdCoeffsPureGo(got[:], 16) != satdCoeffsPureGo(wantC[:], 16) {
			t.Fatalf("off=%d SATD mismatch", off)
		}
	}
}

func TestHadamard8x8AVX2MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x8A8))
	const stride, height = 23, 18
	src := hadamardTestSrc(rng, stride, height)
	for range 2000 {
		row := rng.Intn(height - 8)
		col := rng.Intn(stride - 8)
		for r := range 8 {
			for c := range 8 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var want, got [64]int32
		off := row*stride + col
		hadamard8x8PureGo(src[off:], stride, want[:])
		hadamard8x8AVX2(src[off:], stride, got[:])
		if !sameHadamard8x8Order(got[:], want[:]) {
			t.Fatalf("off=%d got %v want %v", off, got, want)
		}
		if satdCoeffsPureGo(got[:], 64) != satdCoeffsPureGo(want[:], 64) {
			t.Fatalf("off=%d SATD mismatch", off)
		}
	}
}

func TestHadamard16x16AVX2MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1616))
	const stride, height = 37, 34
	src := hadamardTestSrc(rng, stride, height)
	for range 1500 {
		row := rng.Intn(height - 16)
		col := rng.Intn(stride - 16)
		for r := range 16 {
			for c := range 16 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var wantC, wantNEON, got [256]int32
		off := row*stride + col
		hadamard16x16PureGo(src[off:], stride, wantC[:])
		hadamard16x16SVTNEONReference(src[off:], stride, wantNEON[:])
		hadamard16x16AVX2(src[off:], stride, got[:])
		if !slices.Equal(got[:], wantC[:]) && !slices.Equal(got[:], wantNEON[:]) {
			t.Fatalf("off=%d 16x16 order mismatch", off)
		}
		if satdCoeffsPureGo(got[:], 256) != satdCoeffsPureGo(wantC[:], 256) {
			t.Fatalf("off=%d SATD mismatch", off)
		}
	}
}

func TestHadamard32x32AVX2MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x3232))
	const stride, height = 69, 66
	src := hadamardTestSrc(rng, stride, height)
	for range 800 {
		row := rng.Intn(height - 32)
		col := rng.Intn(stride - 32)
		for r := range 32 {
			for c := range 32 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var wantC, wantNEON, got [1024]int32
		off := row*stride + col
		hadamard32x32PureGo(src[off:], stride, wantC[:])
		hadamard32x32SVTNEONReference(src[off:], stride, wantNEON[:])
		hadamard32x32AVX2(src[off:], stride, got[:])
		if !slices.Equal(got[:], wantC[:]) && !slices.Equal(got[:], wantNEON[:]) {
			t.Fatalf("off=%d 32x32 order mismatch", off)
		}
		if satdCoeffsPureGo(got[:], 1024) != satdCoeffsPureGo(wantC[:], 1024) {
			t.Fatalf("off=%d SATD mismatch", off)
		}
	}
}
