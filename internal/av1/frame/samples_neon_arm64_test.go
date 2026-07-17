// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package frame

import (
	"math/rand"
	"testing"
)

func TestLoadSampleRows8NEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x10ad5a))
	for _, tc := range [...]struct {
		width  int
		height int
	}{
		{1, 3}, {4, 4}, {7, 2}, // pure-Go fallback widths
		{8, 1}, {8, 5}, {9, 3}, {15, 4}, {16, 2}, {17, 7},
		{23, 3}, {24, 4}, {31, 2}, {33, 5},
		{40, 7}, {48, 3}, {72, 9}, {80, 5},
		{160, 9}, {1289, 3},
	} {
		srcStride := tc.width + rng.Intn(24)
		dstStride := tc.width + rng.Intn(16)
		src := make([]byte, tc.height*srcStride+tc.width)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		got := make([]uint16, tc.height*dstStride+tc.width)
		want := make([]uint16, len(got))
		for i := range got {
			got[i] = 0xbeef
			want[i] = 0xbeef
		}
		loadSampleRows8PureGo(want, dstStride, src, srcStride, tc.width, tc.height)
		loadSampleRows8NEON(got, dstStride, src, srcStride, tc.width, tc.height)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%dx%d (srcStride=%d dstStride=%d) element %d: NEON=%#x PureGo=%#x",
					tc.width, tc.height, srcStride, dstStride, i, got[i], want[i])
			}
		}
	}
}

func TestLoadSampleRows8NEONZeroAlloc(t *testing.T) {
	const width, height, stride = 129, 16, 144
	src := make([]byte, height*stride)
	for i := range src {
		src[i] = byte(i * 31)
	}
	dst := make([]uint16, height*stride)
	allocs := testing.AllocsPerRun(100, func() {
		loadSampleRows8NEON(dst, stride, src, stride, width, height)
	})
	if allocs != 0 {
		t.Fatalf("loadSampleRows8NEON allocates: %v allocs/run", allocs)
	}
}

func benchLoadSampleRows8(b *testing.B, fn func(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int)) {
	// One 720p luma plane worth of staging (stride-width Full load shape).
	const width, height = 1280, 720
	src := make([]byte, height*width)
	for i := range src {
		src[i] = byte(i * 31)
	}
	dst := make([]uint16, height*width)
	b.SetBytes(int64(width * height))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, width, src, width, width, height)
	}
}

func BenchmarkLoadSampleRows8NEON_720p(b *testing.B) {
	benchLoadSampleRows8(b, loadSampleRows8NEON)
}

func BenchmarkLoadSampleRows8NEON_CDEF(b *testing.B) {
	for _, tc := range []struct {
		name          string
		width, height int
	}{
		{"8x64", 8, 64},
		{"36x32", 36, 32},
		{"40x2", 40, 2},
		{"48x2", 48, 2},
		{"72x64", 72, 64},
		{"80x2", 80, 2},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const stride = 144
			src := make([]byte, tc.height*stride)
			dst := make([]uint16, tc.height*stride)
			b.SetBytes(int64(tc.width * tc.height))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				loadSampleRows8NEON(dst, stride, src, stride, tc.width, tc.height)
			}
		})
	}
}

func BenchmarkLoadSampleRows8PureGo_720p(b *testing.B) {
	benchLoadSampleRows8(b, loadSampleRows8PureGo)
}
