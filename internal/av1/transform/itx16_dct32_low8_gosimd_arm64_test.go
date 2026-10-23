//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseDCT32Col8RawLow8SIMD16MatchesScalar: every column group the
// int16 Low8 kernel accepts equals the scalar inverseDCT32 (the int32 path's
// result on 8-bit column bounds), a declined group is left untouched, and
// every group inside the per-column L1 guard is accepted.
func TestInverseDCT32Col8RawLow8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const width, height = 8, 32
	want := make([]int16, width*height)
	buf := make([]int32, width*height)
	keep := make([]int32, width*height)
	inGuard, accepted, declined := 0, 0, 0
	for trial := 0; trial < 6000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		for r := 8; r < height; r++ {
			clear(want[r*width : (r+1)*width])
		}
		for i, v := range want {
			buf[i] = int32(v)
		}
		copy(keep, buf)
		guard := itx16ColL1OK(want, width, height, itx16ColumnL1Bound)
		for c := 0; c < width; c++ {
			inverseDCT32(want[c:], width, minInt16, maxInt16)
		}
		if !inverseDCT32Col8RawLow8SIMD16(buf, width) {
			declined++
			for i := range buf {
				if buf[i] != keep[i] {
					t.Fatalf("trial %d: declined group modified at %d", trial, i)
				}
			}
			if guard {
				t.Fatalf("trial %d (mode %d): in-guard group declined", trial, trial%6)
			}
			continue
		}
		accepted++
		if guard {
			inGuard++
		}
		for i := range buf {
			if buf[i] != int32(want[i]) {
				t.Fatalf("trial %d (mode %d, guard %v): mismatch at row %d col %d: simd %d scalar %d",
					trial, trial%6, guard, i/width, i%width, buf[i], want[i])
			}
		}
	}
	t.Logf("accepted %d (in guard %d), declined %d", accepted, inGuard, declined)
}

// TestITX16RawColumnPass32Low8MatchesScalar: the pass over a 16 x 32 block
// with rows 8..31 zero, including column groups that fail the guard (falling
// back to the int32 kernels), equals the scalar column pass.
func TestITX16RawColumnPass32Low8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const width, height = 16, 32
	want := make([]int16, width*height)
	buf := make([]int32, width*height)
	for trial := 0; trial < 3000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		for r := 8; r < height; r++ {
			clear(want[r*width : (r+1)*width])
		}
		if trial%7 == 0 {
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					want[r*width+c] = int16(rng.Intn(65536) - 32768)
				}
			}
		}
		for i, v := range want {
			buf[i] = int32(v)
		}
		for c := 0; c < width; c++ {
			inverseDCT32(want[c:], width, minInt16, maxInt16)
		}
		itx16RawColumnPass32Low8SIMD(buf, width, minInt16, maxInt16)
		for i := range buf {
			if buf[i] != int32(want[i]) {
				t.Fatalf("trial %d: mismatch at row %d col %d: pass %d scalar %d", trial, i/width, i%width, buf[i], want[i])
			}
		}
	}
}

func itx32Low8BenchInput() []int32 {
	rng := rand.New(rand.NewSource(9))
	src := make([]int32, 32*8)
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			src[r*8+c] = int32(rng.Intn(1<<12) - (1 << 11))
		}
	}
	return src
}

func BenchmarkDCT32x8Low8_Int16Raw(b *testing.B) {
	src := itx32Low8BenchInput()
	buf := make([]int32, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		if !inverseDCT32Col8RawLow8SIMD16(buf, 8) {
			b.Fatal("guard declined the benchmark input")
		}
	}
}

func BenchmarkDCT32x8Low8_Int32Col4x2(b *testing.B) {
	src := itx32Low8BenchInput()
	buf := make([]int32, len(src))
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		inverseDCT32Col4Low8SIMD(buf, 8, minInt16, maxInt16)
		inverseDCT32Col4Low8SIMD(buf[4:], 8, minInt16, maxInt16)
	}
}
