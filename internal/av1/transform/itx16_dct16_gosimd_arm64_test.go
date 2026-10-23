//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// itx16ColL1OK reports whether every column of the width x height int16 block
// has sum |x| <= bound.
func itx16ColL1OK(buf []int16, width, height int, bound int) bool {
	for c := 0; c < width; c++ {
		l1 := 0
		for r := 0; r < height; r++ {
			v := int(buf[r*width+c])
			if v < 0 {
				v = -v
			}
			l1 += v
		}
		if l1 > bound {
			return false
		}
	}
	return true
}

func itx16FillAdversarial(rng *rand.Rand, buf []int16, width, height, mode int) {
	for i := range buf {
		buf[i] = 0
	}
	switch mode % 6 {
	case 0: // dense uniform, within the guard
		for i := range buf {
			buf[i] = int16(rng.Intn(4081) - 2040)
		}
	case 1: // one full-range row (sparse: eob-like)
		r := rng.Intn(height)
		for c := 0; c < width; c++ {
			buf[r*width+c] = int16(rng.Intn(65536) - 32768)
		}
	case 2: // two rows at +-32767 with random signs
		r0, r1 := rng.Intn(height), rng.Intn(height)
		for c := 0; c < width; c++ {
			buf[r0*width+c] = int16(16383 * (2*rng.Intn(2) - 1))
			buf[r1*width+c] = int16(16383 * (2*rng.Intn(2) - 1))
		}
	case 3: // full-range random everywhere (outside the guard: report only)
		for i := range buf {
			buf[i] = int16(rng.Intn(65536) - 32768)
		}
	case 4: // extremes
		for i := range buf {
			if rng.Intn(2) == 0 {
				buf[i] = -32768
			} else {
				buf[i] = 32767
			}
		}
	default: // mixed magnitudes with zero rows
		for r := 0; r < height; r++ {
			if rng.Intn(3) == 0 {
				continue
			}
			m := 1 << uint(rng.Intn(15))
			for c := 0; c < width; c++ {
				buf[r*width+c] = int16(rng.Intn(2*m+1) - m)
			}
		}
	}
}

func TestInverseDCT16Col8SIMD16v2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const width, height = 8, 16
	const bound = 32767 - 64
	got := make([]int16, width*height)
	want := make([]int16, width*height)
	inGuard, matchOut, totalOut := 0, 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, want, width, height, trial)
		copy(got, want)
		guard := itx16ColL1OK(want, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		inverseDCT16Col8SIMD16v2(got, width, minInt16, maxInt16)
		equal := true
		for i := range got {
			if got[i] != want[i] {
				equal = false
				break
			}
		}
		if guard {
			inGuard++
			if !equal {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial %d (mode %d) in guard: mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
					}
				}
			}
		} else {
			totalOut++
			if equal {
				matchOut++
			}
		}
	}
	t.Logf("in-guard trials %d (all exact); out-of-guard trials %d, of which %d still exact", inGuard, totalOut, matchOut)
}

func BenchmarkDCT16x8_Int16v2(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int16, 16*8+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8SIMD16v2(buf, 8, minInt16, maxInt16)
	}
}

func BenchmarkDCT16x8_Int16Old(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int16, 16*8+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8SIMD16(buf, 8, minInt16, maxInt16)
	}
}

func BenchmarkDCT16x8_Col4x2(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col4SIMDAdapter(buf, 8, minInt16, maxInt16)
		inverseDCT16Col4SIMDAdapter(buf[4:], 8, minInt16, maxInt16)
	}
}

func TestInverseDCT16Col8RawSIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	const width, height = 8, 16
	const bound = 32767 - 64
	src := make([]int16, width*height)
	got := make([]int32, width*height)
	want := make([]int32, width*height)
	inGuard, rejected := 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, src, width, height, trial)
		for i, v := range src {
			got[i] = int32(v)
			want[i] = int32(v)
		}
		guard := itx16ColL1OK(src, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		ok := inverseDCT16Col8RawSIMD16(got, width)
		if ok != guard {
			t.Fatalf("trial %d: kernel guard %v, reference guard %v", trial, ok, guard)
		}
		if !ok {
			rejected++
			for i, v := range src {
				if got[i] != int32(v) {
					t.Fatalf("trial %d: rejected block modified at %d", trial, i)
				}
			}
			continue
		}
		inGuard++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d): mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
			}
		}
	}
	t.Logf("in-guard trials %d (all exact), rejected %d (untouched)", inGuard, rejected)
}

func BenchmarkDCT16x8_RawInt16(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inverseDCT16Col8RawSIMD16(buf, 8)
	}
}

func testDCT16RawLow(t *testing.T, lowN int, kernel func([]int32, int) bool) {
	rng := rand.New(rand.NewSource(int64(100 + lowN)))
	const width, height = 8, 16
	const bound = 32767 - 64
	src := make([]int16, width*height)
	got := make([]int32, width*height)
	want := make([]int32, width*height)
	inGuard, rejected := 0, 0
	for trial := 0; trial < 4000; trial++ {
		itx16FillAdversarial(rng, src, width, height, trial)
		for i := lowN * width; i < len(src); i++ {
			src[i] = 0
		}
		for i, v := range src {
			got[i] = int32(v)
			want[i] = int32(v)
		}
		guard := itx16ColL1OK(src, width, height, bound)
		for c := 0; c < width; c++ {
			inverseDCT16(want[c:], width, minInt16, maxInt16)
		}
		ok := kernel(got, width)
		if ok != guard {
			t.Fatalf("trial %d: kernel guard %v, reference guard %v", trial, ok, guard)
		}
		if !ok {
			rejected++
			for i, v := range src {
				if got[i] != int32(v) {
					t.Fatalf("trial %d: rejected block modified at %d", trial, i)
				}
			}
			continue
		}
		inGuard++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d): mismatch at %d: simd %d scalar %d", trial, trial%6, i, got[i], want[i])
			}
		}
	}
	t.Logf("low%d: in-guard trials %d (all exact), rejected %d (untouched)", lowN, inGuard, rejected)
}

func TestInverseDCT16Col8RawLow4SIMD16MatchesScalar(t *testing.T) {
	testDCT16RawLow(t, 4, inverseDCT16Col8RawLow4SIMD16)
}

func TestInverseDCT16Col8RawLow8SIMD16MatchesScalar(t *testing.T) {
	testDCT16RawLow(t, 8, inverseDCT16Col8RawLow8SIMD16)
}

func benchRawLow(b *testing.B, lowN int, kernel func([]int32, int) bool) {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		if i/8 < lowN {
			buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kernel(buf, 8)
		for j := lowN * 8; j < 16*8; j++ {
			buf[j] = 0
		}
	}
}

func BenchmarkDCT16x8_RawInt16Low4(b *testing.B) { benchRawLow(b, 4, inverseDCT16Col8RawLow4SIMD16) }
func BenchmarkDCT16x8_RawInt16Low8(b *testing.B) { benchRawLow(b, 8, inverseDCT16Col8RawLow8SIMD16) }
