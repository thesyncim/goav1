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

func TestHadamard16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6111))
	const (
		stride = 29
		height = 25
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	for range 300 {
		row := rng.Intn(height - 16)
		col := rng.Intn(stride - 16)
		for r := range 16 {
			for c := range 16 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var wantC [256]int32
		var wantNEON [256]int32
		var got [256]int32
		srcOff := row*stride + col
		hadamard16x16PureGo(src[srcOff:], stride, wantC[:])
		hadamard16x16SVTNEONReference(src[srcOff:], stride, wantNEON[:])
		hadamard16x16Impl(src[srcOff:], stride, got[:])
		if !slices.Equal(got[:], wantC[:]) && !slices.Equal(got[:], wantNEON[:]) {
			t.Fatalf("offset=%d got %v wantC %v wantNEON %v", srcOff, got, wantC, wantNEON)
		}
		if satdCoeffsPureGo(got[:], 256) != satdCoeffsPureGo(wantC[:], 256) {
			t.Fatalf("offset=%d SATD got %d want %d", srcOff, satdCoeffsPureGo(got[:], 256), satdCoeffsPureGo(wantC[:], 256))
		}
	}
}

func TestHadamard16x16MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6112))
	const (
		stride = 31
		height = 27
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	for range 300 {
		row := rng.Intn(height - 16)
		col := rng.Intn(stride - 16)
		for r := range 16 {
			for c := range 16 {
				src[(row+r)*stride+col+c] = int16(rng.Intn(511) - 255)
			}
		}
		var wantC [256]int32
		var wantNEON [256]int32
		var got [256]int32
		srcOff := row*stride + col
		hadamard16x16PureGo(src[srcOff:], stride, wantC[:])
		hadamard16x16SVTNEONReference(src[srcOff:], stride, wantNEON[:])
		hadamard16x16(src[srcOff:], stride, got[:])
		if !slices.Equal(got[:], wantC[:]) && !slices.Equal(got[:], wantNEON[:]) {
			t.Fatalf("offset=%d got %v wantC %v wantNEON %v", srcOff, got, wantC, wantNEON)
		}
		if satdCoeffsPureGo(got[:], 256) != satdCoeffsPureGo(wantC[:], 256) {
			t.Fatalf("offset=%d SATD got %d want %d", srcOff, satdCoeffsPureGo(got[:], 256), satdCoeffsPureGo(wantC[:], 256))
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

func hadamard16x16SVTNEONReference(src []int16, srcStride int, coeff []int32) {
	hadamard8x8TransposeReference(src, srcStride, coeff)
	hadamard8x8TransposeReference(src[8:], srcStride, coeff[64:])
	hadamard8x8TransposeReference(src[8*srcStride:], srcStride, coeff[128:])
	hadamard8x8TransposeReference(src[8*srcStride+8:], srcStride, coeff[192:])

	for base := 0; base < 64; base += 16 {
		var c [4][4][4]int32
		for g, off := range []int{0, 4, 8, 12} {
			for lane := range 4 {
				idx := base + off + lane
				a0 := coeff[idx]
				a1 := coeff[64+idx]
				a2 := coeff[128+idx]
				a3 := coeff[192+idx]

				b0 := (a0 + a1) >> 1
				b1 := (a0 - a1) >> 1
				b2 := (a2 + a3) >> 1
				b3 := (a2 - a3) >> 1

				c[g][0][lane] = b0 + b2
				c[g][1][lane] = b1 + b3
				c[g][2][lane] = b0 - b2
				c[g][3][lane] = b1 - b3
			}
		}
		for lane := range 4 {
			coeff[base+0+lane] = c[0][0][lane]
			coeff[base+4+lane] = c[2][0][lane]
			coeff[base+8+lane] = c[1][0][lane]
			coeff[base+12+lane] = c[3][0][lane]

			coeff[64+base+0+lane] = c[0][1][lane]
			coeff[64+base+4+lane] = c[2][1][lane]
			coeff[64+base+8+lane] = c[1][1][lane]
			coeff[64+base+12+lane] = c[3][1][lane]

			coeff[128+base+0+lane] = c[0][2][lane]
			coeff[128+base+4+lane] = c[2][2][lane]
			coeff[128+base+8+lane] = c[1][2][lane]
			coeff[128+base+12+lane] = c[3][2][lane]

			coeff[192+base+0+lane] = c[0][3][lane]
			coeff[192+base+4+lane] = c[2][3][lane]
			coeff[192+base+8+lane] = c[1][3][lane]
			coeff[192+base+12+lane] = c[3][3][lane]
		}
	}
}

func hadamard8x8TransposeReference(src []int16, srcStride int, coeff []int32) {
	var tmp [64]int32
	hadamard8x8PureGo(src, srcStride, tmp[:])
	for r := range 8 {
		for c := range 8 {
			coeff[r*8+c] = tmp[c*8+r]
		}
	}
}
