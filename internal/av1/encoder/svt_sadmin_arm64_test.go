//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

func TestSVTSAD16xhx8MinMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	const stride = 96
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 96; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		switch trial % 7 {
		case 5: // the largest SADs: every difference 255
			for i := range src {
				src[i], ref[i] = 0, 255
			}
		case 6:
			for i := range src {
				src[i], ref[i] = 255, 0
			}
		}
		for rows := 1; rows <= 16; rows++ {
			so := rng.Intn(16)
			ro := rng.Intn(16)
			s := src[so : so+(rows-1)*2*stride+16]
			r := ref[ro : ro+(rows-1)*stride+23]
			var got [8]uint32
			got[5] = 11 // accumulates
			svtSAD16xhx8Min(&got, s, 2*stride, r, stride, rows)
			want := svtSAD16xhx8Scalar(s, 2*stride, r, stride, rows)
			want[5] += 11
			if got != want {
				t.Fatalf("trial %d rows %d: got %v want %v", trial, rows, got, want)
			}
		}
	}
}

func BenchmarkSVTSAD16xhx8Min(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const stride = 416
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	fillMEPattern(rng, src, 0)
	fillMEPattern(rng, ref, 0)
	for _, rows := range []int{8, 16} {
		b.Run(map[int]string{8: "cur_rows8", 16: "cur_rows16"}[rows], func(b *testing.B) {
			var sink uint32
			for b.Loop() {
				var a [8]uint32
				svtSAD16xhx8(&a, src, 2*stride, ref, 2*stride, rows)
				sink += a[3]
			}
			_ = sink
		})
		b.Run(map[int]string{8: "min_rows8", 16: "min_rows16"}[rows], func(b *testing.B) {
			var sink uint32
			for b.Loop() {
				var a [8]uint32
				svtSAD16xhx8Min(&a, src, 2*stride, ref, 2*stride, rows)
				sink += a[3]
			}
			_ = sink
		})
	}
}
