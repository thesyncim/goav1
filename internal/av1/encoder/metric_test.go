package encoder

import (
	"math/rand"
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
