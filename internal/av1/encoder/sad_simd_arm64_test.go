// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// funcName returns the fully-qualified name of a function value.
func funcName(f any) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

// TestSADDispatchBoundToSIMD proves that under GOEXPERIMENT=simd the ported
// dispatch vars point at the Go-native SIMD kernels (not the NEON asm) — the
// FuncForPC probe the wiring plan calls for. The 8-wide shapes stay on NEON by
// design, so we assert those are NOT SIMD.
func TestSADDispatchBoundToSIMD(t *testing.T) {
	cases := []struct {
		name    string
		fn      any
		wantSIMD bool
	}{
		{"sad16x16Impl", sad16x16Impl, true},
		{"sad32x32Impl", sad32x32Impl, true},
		{"sad16x16x4Impl", sad16x16x4Impl, true},
		{"sad32x32x4Impl", sad32x32x4Impl, true},
		{"sad8x8x4Impl", sad8x8x4Impl, true},  // x4 reuse amortizes the pack -> beats asm
		{"sad8x8Impl", sad8x8Impl, false},     // single-block kept on NEON (pack tax, no reuse)
	}
	for _, c := range cases {
		n := funcName(c.fn)
		isSIMD := strings.Contains(n, "SIMD")
		if isSIMD != c.wantSIMD {
			t.Errorf("%s bound to %q: SIMD=%v, want SIMD=%v", c.name, n, isSIMD, c.wantSIMD)
		}
	}
	// The hot-path lowercase wrappers route to SIMD for the ported shapes.
	if n := funcName(sad16x16); strings.Contains(n, "SIMD") {
		t.Logf("sad16x16 wrapper -> %s", n) // wrapper itself, calls sad16x16SIMD
	}
}

// makeSADPlane fills a source/reference pair of the given size with random
// bytes for the differential sweeps.
func makeSADPlane(seed int64, n int) ([]byte, []byte) {
	rng := rand.New(rand.NewSource(seed))
	src := make([]byte, n)
	ref := make([]byte, n)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	return src, ref
}

// TestSAD8x8SIMDByteExact proves sad8x8SIMD is byte-identical to sad8x8PureGo
// over random data, several odd strides, and many offsets. The limit is set
// huge so the scalar early exit never fires (the SIMD kernel ignores it, so
// both must agree on the full-block total).
func TestSAD8x8SIMDByteExact(t *testing.T) {
	for _, stride := range []int{8, 9, 16, 33, 64, 73} {
		src, ref := makeSADPlane(int64(stride)*7+1, stride*80)
		rng := rand.New(rand.NewSource(int64(stride)))
		for range 3000 {
			off := rng.Intn(stride*(80-8) - 8)
			if off < 0 {
				off = 0
			}
			want := sad8x8PureGo(src[off:], ref[off:], stride, 1<<30)
			got := sad8x8SIMD(src[off:], ref[off:], stride, 1<<30)
			if got != want {
				t.Fatalf("stride %d off %d: SIMD %d want %d", stride, off, got, want)
			}
		}
	}
}

func TestSAD16x16SIMDByteExact(t *testing.T) {
	for _, stride := range []int{16, 17, 32, 48, 64, 91} {
		src, ref := makeSADPlane(int64(stride)*11+3, stride*96)
		rng := rand.New(rand.NewSource(int64(stride) + 100))
		for range 3000 {
			off := rng.Intn(stride*(96-16) - 16)
			if off < 0 {
				off = 0
			}
			want := sad16x16PureGo(src[off:], ref[off:], stride)
			got := sad16x16SIMD(src[off:], ref[off:], stride)
			if got != want {
				t.Fatalf("stride %d off %d: SIMD %d want %d", stride, off, got, want)
			}
		}
	}
}

func TestSAD32x32SIMDByteExact(t *testing.T) {
	for _, stride := range []int{32, 33, 48, 64, 96, 127} {
		src, ref := makeSADPlane(int64(stride)*13+5, stride*160)
		rng := rand.New(rand.NewSource(int64(stride) + 200))
		for range 2000 {
			off := rng.Intn(stride*(160-32) - 32)
			if off < 0 {
				off = 0
			}
			want := sad32x32PureGo(src[off:], ref[off:], stride)
			got := sad32x32SIMD(src[off:], ref[off:], stride)
			if got != want {
				t.Fatalf("stride %d off %d: SIMD %d want %d", stride, off, got, want)
			}
		}
	}
}

// TestSAD64x64SIMDByteExact exercises the composed 64x64 path (which now routes
// through sad32x32SIMD under simd).
func TestSAD64x64SIMDByteExact(t *testing.T) {
	for _, stride := range []int{64, 65, 96, 128, 191} {
		src, ref := makeSADPlane(int64(stride)*17+7, stride*192)
		rng := rand.New(rand.NewSource(int64(stride) + 300))
		for range 1000 {
			off := rng.Intn(stride*(192-64) - 64)
			if off < 0 {
				off = 0
			}
			want := sad64x64Composed(src[off:], ref[off:], stride) // both use the same 32x32 under simd
			got := sad64x64(src[off:], ref[off:], stride)
			if got != want {
				t.Fatalf("stride %d off %d: SIMD %d want %d", stride, off, got, want)
			}
			// Also check against the pure scalar 32x32 composition explicitly.
			want2 := sad32x32PureGo(src[off:], ref[off:], stride) +
				sad32x32PureGo(src[off+32:], ref[off+32:], stride) +
				sad32x32PureGo(src[off+32*stride:], ref[off+32*stride:], stride) +
				sad32x32PureGo(src[off+32*stride+32:], ref[off+32*stride+32:], stride)
			if got != want2 {
				t.Fatalf("stride %d off %d: SIMD %d want(scalar-compose) %d", stride, off, got, want2)
			}
		}
	}
}

func TestSAD8x8x4SIMDByteExact(t *testing.T) {
	for _, stride := range []int{24, 40, 79, 96} {
		src, ref := makeSADPlane(int64(stride)*19+9, stride*128)
		rng := rand.New(rand.NewSource(int64(stride) + 400))
		for range 2000 {
			row := rng.Intn(96) + 8
			col := rng.Intn(stride-16) + 8
			off := row*stride + col
			r0, r1, r2, r3 := off+2, off-2, off+2*stride, off-2*stride
			w0, w1, w2, w3 := sad8x8x4PureGo(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			g0, g1, g2, g3 := sad8x8x4SIMD(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("stride %d off %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
					stride, off, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

func TestSAD16x16x4SIMDByteExact(t *testing.T) {
	for _, stride := range []int{48, 64, 91, 128} {
		src, ref := makeSADPlane(int64(stride)*23+11, stride*192)
		rng := rand.New(rand.NewSource(int64(stride) + 500))
		for range 2000 {
			row := rng.Intn(150) + 16
			col := rng.Intn(stride-40) + 16
			off := row*stride + col
			r0, r1, r2, r3 := off+3, off-3, off+2*stride, off-2*stride
			w0, w1, w2, w3 := sad16x16x4PureGo(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			g0, g1, g2, g3 := sad16x16x4SIMD(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("stride %d off %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
					stride, off, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

func TestSAD32x32x4SIMDByteExact(t *testing.T) {
	for _, stride := range []int{96, 127, 160} {
		src, ref := makeSADPlane(int64(stride)*29+13, stride*288)
		rng := rand.New(rand.NewSource(int64(stride) + 600))
		for range 1500 {
			row := rng.Intn(220) + 32
			col := rng.Intn(stride-72) + 32
			off := row*stride + col
			r0, r1, r2, r3 := off+4, off-4, off+2*stride, off-2*stride
			w0, w1, w2, w3 := sad32x32x4PureGo(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			g0, g1, g2, g3 := sad32x32x4SIMD(src[off:], ref[r0:], ref[r1:], ref[r2:], ref[r3:], stride)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("stride %d off %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
					stride, off, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

// TestSAD8x8SIMDExtremes checks the all-0 / all-255 corners where per-lane
// abs-diffs are maximal (255) — the accumulate-without-overflow gate.
func TestSAD8x8SIMDExtremes(t *testing.T) {
	const stride = 32
	src := make([]byte, stride*32)
	ref := make([]byte, stride*32)
	for i := range src {
		src[i] = 255
		ref[i] = 0
	}
	if got, want := sad8x8SIMD(src, ref, stride, 1<<30), sad8x8PureGo(src, ref, stride, 1<<30); got != want {
		t.Fatalf("8x8 extreme: SIMD %d want %d", got, want)
	}
	if got, want := sad16x16SIMD(src, ref, stride), sad16x16PureGo(src, ref, stride); got != want {
		t.Fatalf("16x16 extreme: SIMD %d want %d", got, want)
	}
	if got, want := sad32x32SIMD(src, ref, stride), sad32x32PureGo(src, ref, stride); got != want {
		t.Fatalf("32x32 extreme: SIMD %d want %d (SIMD=%d, expected 32*32*255=%d)", got, want, got, 32*32*255)
	}
}

// TestSADSIMDWidenFallbackByteExact forces the non-DOTPROD widen accumulate
// path (used on ARM cores without the dot-product extension) and proves it is
// byte-identical to the scalar reference. On DOTPROD hardware the default tests
// only exercise the UDOT path, so this guards the fallback's overflow handling.
func TestSADSIMDWidenFallbackByteExact(t *testing.T) {
	saved := useDotProdSADSIMD
	useDotProdSADSIMD = false
	defer func() { useDotProdSADSIMD = saved }()

	for _, stride := range []int{16, 33, 64, 96, 128} {
		src, ref := makeSADPlane(int64(stride)*31+17, stride*192)
		rng := rand.New(rand.NewSource(int64(stride) + 700))
		for range 1500 {
			off := rng.Intn(stride*(192-32) - 32)
			if off < 0 {
				off = 0
			}
			if g, w := sad16x16SIMD(src[off:], ref[off:], stride), sad16x16PureGo(src[off:], ref[off:], stride); g != w {
				t.Fatalf("widen 16x16 stride %d off %d: %d want %d", stride, off, g, w)
			}
			if g, w := sad32x32SIMD(src[off:], ref[off:], stride), sad32x32PureGo(src[off:], ref[off:], stride); g != w {
				t.Fatalf("widen 32x32 stride %d off %d: %d want %d", stride, off, g, w)
			}
		}
	}
	// x4 widen path, incl. the all-255 corner that stresses the periodic
	// uint16->uint32 flush.
	const stride = 128
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = 255
		ref[i] = 0
	}
	g0, g1, g2, g3 := sad32x32x4SIMD(src, ref, ref, ref, ref, stride)
	w0, w1, w2, w3 := sad32x32x4PureGo(src, ref, ref, ref, ref, stride)
	if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
		t.Fatalf("widen 32x32x4 extreme: (%d,%d,%d,%d) want (%d,%d,%d,%d)", g0, g1, g2, g3, w0, w1, w2, w3)
	}
	h0, h1, h2, h3 := sad16x16x4SIMD(src, ref, ref, ref, ref, stride)
	x0, x1, x2, x3 := sad16x16x4PureGo(src, ref, ref, ref, ref, stride)
	if h0 != x0 || h1 != x1 || h2 != x2 || h3 != x3 {
		t.Fatalf("widen 16x16x4 extreme: (%d,%d,%d,%d) want (%d,%d,%d,%d)", h0, h1, h2, h3, x0, x1, x2, x3)
	}
}

// TestSADSIMDZeroAlloc asserts the SIMD kernels do not heap-allocate.
func TestSADSIMDZeroAlloc(t *testing.T) {
	const stride = 64
	src, ref := makeSADPlane(1, stride*64)
	var sink int
	if a := testing.AllocsPerRun(200, func() {
		sink += sad8x8SIMD(src, ref, stride, 1<<30)
		sink += sad16x16SIMD(src, ref, stride)
		sink += sad32x32SIMD(src, ref, stride)
		a, b, c, d := sad16x16x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], stride)
		sink += a + b + c + d
		e, f, g, h := sad32x32x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], stride)
		sink += e + f + g + h
	}); a != 0 {
		t.Fatalf("SIMD SAD allocated %.2f objects/op, want 0 (sink=%d)", a, sink)
	}
	_ = sink
}

// --- benchmarks: scalar reference vs Go-native SIMD vs NEON asm --------------

func benchPlane() ([]byte, []byte) {
	src := make([]byte, 128*128)
	ref := make([]byte, 128*128)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	return src, ref
}

func BenchmarkSAD16x16_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad16x16PureGo(src, ref, 128)
	}
}
func BenchmarkSAD16x16_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad16x16SIMD(src, ref, 128)
	}
}
func BenchmarkSAD16x16_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad16x16NEON(src, ref, 128)
	}
}

func BenchmarkSAD32x32_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad32x32PureGo(src, ref, 128)
	}
}
func BenchmarkSAD32x32_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad32x32SIMD(src, ref, 128)
	}
}
func BenchmarkSAD32x32_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad32x32NEON(src, ref, 128)
	}
}
func BenchmarkSAD32x32_DotProdNEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad32x32DotProd(src, ref, 128)
	}
}

func BenchmarkSAD8x8_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad8x8PureGo(src, ref, 128, 1<<30)
	}
}
func BenchmarkSAD8x8_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad8x8SIMD(src, ref, 128, 1<<30)
	}
}
func BenchmarkSAD8x8_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		sad8x8NEON(src, ref, 128, 1<<30)
	}
}

func BenchmarkSAD16x16x4_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4PureGo(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD16x16x4_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD16x16x4_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4NEON(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD16x16x4_DotProdNEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4DotProd(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}

func BenchmarkSAD32x32x4_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4PureGo(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD32x32x4_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD32x32x4_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4NEON(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD32x32x4_DotProdNEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4DotProd(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}

func BenchmarkSAD8x8x4_Scalar(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad8x8x4PureGo(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD8x8x4_SIMD(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad8x8x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
func BenchmarkSAD8x8x4_NEON(b *testing.B) {
	src, ref := benchPlane()
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad8x8x4NEON(src, ref, ref[4:], ref[8:], ref[12:], 128)
	}
}
