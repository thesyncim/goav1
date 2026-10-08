// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import (
	"bytes"
	"math/rand"
	"testing"
)

// refLoadSampleRows16 is the byte-exact scalar reference for the 16-bit sample
// staging load, independent of the build-tag-selected loadSampleRows16 (the
// little-endian per-row memmove on samples_le.go arches, the scalar loop on
// samples_generic.go arches). The differential tests below assert the active
// implementation is bit-identical to this reference.
func refLoadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	srcOff := 0
	dstOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		srcLine := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
		dstLine := dst[dstOff : dstOff+width : dstOff+width]
		for x := 0; x < width; x++ {
			off := x * 2
			dstLine[x] = uint16(srcLine[off]) | uint16(srcLine[off+1])<<8
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

// refStoreSampleRows16 is the byte-exact scalar reference for the 16-bit sample
// staging store.
func refStoreSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	dstOff := 0
	srcOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		dstLine := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
		srcLine := src[srcOff : srcOff+width : srcOff+width]
		for x, sample := range srcLine {
			off := x * 2
			dstLine[off] = byte(sample)
			dstLine[off+1] = byte(sample >> 8)
		}
		dstOff += dstStride
		srcOff += srcStride
	}
}

// TestLoadSampleRows16MatchesScalar fuzzes the active loadSampleRows16 against
// the scalar reference over randomized widths, heights, and strides, including
// non-contiguous rows whose stride exceeds the visible width so the untouched
// stride padding is exercised. The whole destination buffer (padding included)
// must stay byte-identical to the reference run.
func TestLoadSampleRows16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x10adbeef))
	const sentinel = uint16(0xa5c3)
	for iter := 0; iter < 4000; iter++ {
		width := rng.Intn(65)         // 0..64 visible samples
		height := rng.Intn(49)        // 0..48 rows
		dstPad := rng.Intn(9)         // extra dst samples of stride padding
		srcPad := rng.Intn(17)        // extra src bytes of stride padding
		dstStride := width + dstPad   // samples
		srcStride := width*2 + srcPad // bytes

		srcLen := height*srcStride + width*2 + 8
		src := make([]byte, srcLen)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}

		dstLen := height*dstStride + width + 8
		got := make([]uint16, dstLen)
		want := make([]uint16, dstLen)
		for i := range got {
			got[i] = sentinel
			want[i] = sentinel
		}

		loadSampleRows16(got, dstStride, src, srcStride, width, height)
		refLoadSampleRows16(want, dstStride, src, srcStride, width, height)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("iter=%d w=%d h=%d dstStride=%d srcStride=%d: dst[%d]=%#x want %#x",
					iter, width, height, dstStride, srcStride, i, got[i], want[i])
			}
		}
	}
}

// TestStoreSampleRows16MatchesScalar mirrors the load test for the store path,
// spanning the full 16-bit sample range so both serialized bytes are exercised.
func TestStoreSampleRows16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5701e12e))
	const sentinel = byte(0x5a)
	for iter := 0; iter < 4000; iter++ {
		width := rng.Intn(65)
		height := rng.Intn(49)
		srcPad := rng.Intn(9)         // extra src samples of stride padding
		dstPad := rng.Intn(17)        // extra dst bytes of stride padding
		srcStride := width + srcPad   // samples
		dstStride := width*2 + dstPad // bytes

		srcLen := height*srcStride + width + 8
		src := make([]uint16, srcLen)
		for i := range src {
			src[i] = uint16(rng.Intn(65536))
		}

		dstLen := height*dstStride + width*2 + 8
		got := make([]byte, dstLen)
		want := make([]byte, dstLen)
		for i := range got {
			got[i] = sentinel
			want[i] = sentinel
		}

		storeSampleRows16(got, dstStride, src, srcStride, width, height)
		refStoreSampleRows16(want, dstStride, src, srcStride, width, height)

		if !bytes.Equal(got, want) {
			t.Fatalf("iter=%d w=%d h=%d dstStride=%d srcStride=%d: store mismatch",
				iter, width, height, dstStride, srcStride)
		}
	}
}

// benchSampleRows16 sizes a representative 1080p luma plane with a little
// stride padding so the per-row copy path (not one big contiguous copy) is
// exercised, matching how the decoder stages one plane at a time.
const (
	benchW      = 1920
	benchH      = 1080
	benchDstPad = 8
)

func BenchmarkLoadSampleRows16Scalar(b *testing.B) {
	dstStride := benchW + benchDstPad
	srcStride := benchW * 2
	src := make([]byte, benchH*srcStride)
	dst := make([]uint16, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refLoadSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

func BenchmarkLoadSampleRows16Fast(b *testing.B) {
	dstStride := benchW + benchDstPad
	srcStride := benchW * 2
	src := make([]byte, benchH*srcStride)
	dst := make([]uint16, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

func BenchmarkStoreSampleRows16Scalar(b *testing.B) {
	srcStride := benchW + benchDstPad
	dstStride := benchW * 2
	src := make([]uint16, benchH*srcStride)
	dst := make([]byte, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refStoreSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

func BenchmarkStoreSampleRows16Fast(b *testing.B) {
	srcStride := benchW + benchDstPad
	dstStride := benchW * 2
	src := make([]uint16, benchH*srcStride)
	dst := make([]byte, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

// TestSampleRows16ZeroAlloc pins that neither staging helper allocates: the
// unsafe row view on little-endian arches must not force dst/src to escape.
func TestSampleRows16ZeroAlloc(t *testing.T) {
	const width, height = 64, 48
	dstStride := width + 4
	srcStride := width*2 + 6
	src8 := make([]byte, height*srcStride+width*2)
	dst16 := make([]uint16, height*dstStride+width)
	if n := testing.AllocsPerRun(50, func() {
		loadSampleRows16(dst16, dstStride, src8, srcStride, width, height)
	}); n != 0 {
		t.Fatalf("loadSampleRows16 allocated %.0f times, want 0", n)
	}

	srcStrideS := width + 4
	dstStrideB := width*2 + 6
	src16 := make([]uint16, height*srcStrideS+width)
	dst8 := make([]byte, height*dstStrideB+width*2)
	if n := testing.AllocsPerRun(50, func() {
		storeSampleRows16(dst8, dstStrideB, src16, srcStrideS, width, height)
	}); n != 0 {
		t.Fatalf("storeSampleRows16 allocated %.0f times, want 0", n)
	}
}
