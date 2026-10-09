//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// TestPixelStatsSIMDMatchesPureGo checks every bound pixel-statistics shape
// against the scalar reference, including extremes and strided windows.
func TestPixelStatsSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9151))
	shapes := []struct{ w, h int }{
		{4, 4}, {4, 8}, {4, 16}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 8}, {8, 16}, {8, 32},
		{16, 4}, {16, 8}, {16, 16}, {16, 32}, {16, 64}, {32, 8}, {32, 16}, {32, 32}, {32, 64},
		{64, 16}, {64, 32}, {48, 4},
	}
	for _, sh := range shapes {
		for trial := range 300 {
			srcStride := sh.w + rng.Intn(17)
			refStride := sh.w + rng.Intn(17)
			src := make([]byte, srcStride*sh.h+3)
			ref := make([]byte, refStride*sh.h+3)
			fillPixelStatsBytes(rng, src, trial)
			fillPixelStatsBytes(rng, ref, trial+1)
			wantSSE, wantSum := pixelStatsPureGo(src, srcStride, ref, refStride, sh.w, sh.h)
			gotSSE, gotSum := pixelStatsSIMD(src, srcStride, ref, refStride, sh.w, sh.h)
			if gotSSE != wantSSE || gotSum != wantSum {
				t.Fatalf("%dx%d trial %d: got (%d,%d) want (%d,%d)", sh.w, sh.h, trial, gotSSE, gotSum, wantSSE, wantSum)
			}
		}
	}
}

// TestPixelStatsSIMDExtremes covers all-0 against all-255 (the largest
// per-sample difference) and the 64x64 block (the largest accumulation).
func TestPixelStatsSIMDExtremes(t *testing.T) {
	for _, sh := range []struct{ w, h int }{{8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		src := make([]byte, sh.w*sh.h)
		ref := make([]byte, sh.w*sh.h)
		for i := range ref {
			ref[i] = 255
		}
		for _, pair := range [][2][]byte{{src, ref}, {ref, src}} {
			wantSSE, wantSum := pixelStatsPureGo(pair[0], sh.w, pair[1], sh.w, sh.w, sh.h)
			gotSSE, gotSum := pixelStatsSIMD(pair[0], sh.w, pair[1], sh.w, sh.w, sh.h)
			if gotSSE != wantSSE || gotSum != wantSum {
				t.Fatalf("%dx%d extreme: got (%d,%d) want (%d,%d)", sh.w, sh.h, gotSSE, gotSum, wantSSE, wantSum)
			}
		}
	}
}

func TestPixelStatsDispatchNoAlloc(t *testing.T) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = byte(i * 5)
		ref[i] = byte(i * 9)
	}
	if a := testing.AllocsPerRun(100, func() { _, _ = pixelStats16x16Impl(src, 64, ref, 64) }); a != 0 {
		t.Errorf("pixelStats16x16Impl: %.1f allocs/op, want 0", a)
	}
	if a := testing.AllocsPerRun(100, func() { _, _ = pixelStats8x8Impl(src, 64, ref, 64) }); a != 0 {
		t.Errorf("pixelStats8x8Impl: %.1f allocs/op, want 0", a)
	}
}

// fillPixelStatsBytes fills b with random bytes; odd trials use the 0/255
// extremes so every saturated difference is reached.
func fillPixelStatsBytes(rng *rand.Rand, b []byte, trial int) {
	for i := range b {
		if trial%2 == 1 && rng.Intn(2) == 0 {
			b[i] = byte(255 * rng.Intn(2))
			continue
		}
		b[i] = byte(rng.Intn(256))
	}
}
