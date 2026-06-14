package encoder

import (
	"math/rand"
	"slices"
	"testing"
)

func TestPixelStatsImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6101))
	const (
		srcStride = 151
		refStride = 157
		height    = 144
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, sh := range []struct {
		w, h int
		pure func([]byte, int, []byte, int) (uint32, int32)
		impl func([]byte, int, []byte, int) (uint32, int32)
	}{
		{w: 8, h: 8, pure: pixelStats8x8PureGo, impl: pixelStats8x8Impl},
		{w: 16, h: 16, pure: pixelStats16x16PureGo, impl: pixelStats16x16Impl},
		{w: 32, h: 32, pure: pixelStats32x32PureGo, impl: pixelStats32x32Impl},
	} {
		for range 1000 {
			srow := rng.Intn(height - sh.h)
			scol := rng.Intn(srcStride - sh.w)
			rrow := rng.Intn(height - sh.h)
			rcol := rng.Intn(refStride - sh.w)
			srcOff := srow*srcStride + scol
			refOff := rrow*refStride + rcol
			wantSSE, wantSum := sh.pure(src[srcOff:], srcStride, ref[refOff:], refStride)
			gotSSE, gotSum := sh.impl(src[srcOff:], srcStride, ref[refOff:], refStride)
			if gotSSE != wantSSE || gotSum != wantSum {
				t.Fatalf("%dx%d srcOff=%d refOff=%d: got sse=%d sum=%d want sse=%d sum=%d",
					sh.w, sh.h, srcOff, refOff, gotSSE, gotSum, wantSSE, wantSum)
			}
		}
	}
}

func TestSSEVarianceMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6102))
	const (
		srcStride = 192
		refStride = 208
		height    = 128
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, sh := range []struct {
		n     int
		fn    func([]byte, int, []byte, int) (uint32, uint32)
		shift uint
	}{
		{n: 8, fn: sseVariance8x8, shift: 6},
		{n: 16, fn: sseVariance16x16, shift: 8},
		{n: 32, fn: sseVariance32x32, shift: 10},
		{n: 64, fn: sseVariance64x64, shift: 12},
	} {
		for range 500 {
			srow := rng.Intn(height - sh.n)
			scol := rng.Intn(srcStride - sh.n)
			rrow := rng.Intn(height - sh.n)
			rcol := rng.Intn(refStride - sh.n)
			srcOff := srow*srcStride + scol
			refOff := rrow*refStride + rcol
			wantSSE, wantSum := pixelStatsPureGo(src[srcOff:], srcStride, ref[refOff:], refStride, sh.n, sh.n)
			wantVar := varianceFromStats(wantSSE, wantSum, sh.shift)
			gotSSE, gotVar := sh.fn(src[srcOff:], srcStride, ref[refOff:], refStride)
			if gotSSE != wantSSE || gotVar != wantVar {
				t.Fatalf("%dx%d srcOff=%d refOff=%d: got sse=%d var=%d want sse=%d var=%d",
					sh.n, sh.n, srcOff, refOff, gotSSE, gotVar, wantSSE, wantVar)
			}
		}
	}
}

func TestSATDCoeffsImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6105))
	coeff := make([]int32, 1024)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(65281) - 32640)
	}
	for _, count := range []int{16, 64, 256, 1024} {
		for range 200 {
			for i := range coeff[:count] {
				coeff[i] = int32(rng.Intn(65281) - 32640)
			}
			want := satdCoeffsPureGo(coeff, count)
			got := satdCoeffsImpl(coeff, count)
			if got != want {
				t.Fatalf("count=%d got %d want %d", count, got, want)
			}
		}
	}
}

func TestSATDCoeffsMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6106))
	coeff := make([]int32, 1024)
	for _, count := range []int{16, 64, 256, 1024} {
		for range 200 {
			for i := range coeff[:count] {
				coeff[i] = int32(rng.Intn(65281) - 32640)
			}
			want := satdCoeffsPureGo(coeff, count)
			got := satdCoeffs(coeff, count)
			if got != want {
				t.Fatalf("count=%d got %d want %d", count, got, want)
			}
		}
	}
}

func TestHadamard8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6108))
	const (
		stride = 19
		height = 16
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	for range 500 {
		row := rng.Intn(height - 8)
		col := rng.Intn(stride - 8)
		for r := range 8 {
			for c := range 8 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var want [64]int32
		var got [64]int32
		srcOff := row*stride + col
		hadamard8x8PureGo(src[srcOff:], stride, want[:])
		hadamard8x8Impl(src[srcOff:], stride, got[:])
		if !sameHadamard8x8Order(got[:], want[:]) {
			t.Fatalf("offset=%d got %v want %v", srcOff, got, want)
		}
	}
}

func TestHadamard8x8MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6109))
	const (
		stride = 23
		height = 18
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	for range 500 {
		row := rng.Intn(height - 8)
		col := rng.Intn(stride - 8)
		for r := range 8 {
			for c := range 8 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var want [64]int32
		var got [64]int32
		srcOff := row*stride + col
		hadamard8x8PureGo(src[srcOff:], stride, want[:])
		hadamard8x8(src[srcOff:], stride, got[:])
		if !sameHadamard8x8Order(got[:], want[:]) {
			t.Fatalf("offset=%d got %v want %v", srcOff, got, want)
		}
	}
}

func sameHadamard8x8Order(got, want []int32) bool {
	if slices.Equal(got, want) {
		return true
	}
	for r := range 8 {
		for c := range 8 {
			if got[r*8+c] != want[c*8+r] {
				return false
			}
		}
	}
	return true
}
