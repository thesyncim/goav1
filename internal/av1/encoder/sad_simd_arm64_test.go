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

	"simd/archsimd"
)

func funcName(f any) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

// TestSADDispatchBoundToSIMD proves only the promoted 8x8 four-reference
// kernel is bound to Go-native SIMD. Other SAD shapes retain their measured
// NEON implementations.
func TestSADDispatchBoundToSIMD(t *testing.T) {
	cases := []struct {
		name     string
		fn       any
		wantSIMD bool
	}{
		{"sad8x8Impl", sad8x8Impl, false},
		{"sad16x16Impl", sad16x16Impl, false},
		{"sad32x32Impl", sad32x32Impl, false},
		{"sad8x8x4Impl", sad8x8x4Impl, true},
		{"sad16x16x4Impl", sad16x16x4Impl, false},
		{"sad32x32x4Impl", sad32x32x4Impl, false},
	}
	for _, c := range cases {
		name := funcName(c.fn)
		isSIMD := strings.Contains(name, "SIMD")
		if isSIMD != c.wantSIMD {
			t.Errorf("%s bound to %q: SIMD=%v, want SIMD=%v", c.name, name, isSIMD, c.wantSIMD)
		}
	}
}

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

// TestSAD8x8x4SIMDByteExact covers random data, odd strides, and unaligned
// block origins against the scalar oracle.
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

func TestSAD8x8x4SIMDExtremes(t *testing.T) {
	const stride = 32
	src := make([]byte, stride*32)
	ref := make([]byte, stride*32)
	for i := range src {
		src[i] = 255
	}
	g0, g1, g2, g3 := sad8x8x4SIMD(src, ref, ref, ref, ref, stride)
	w0, w1, w2, w3 := sad8x8x4PureGo(src, ref, ref, ref, ref, stride)
	if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
		t.Fatalf("all-255 vs all-0: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
			g0, g1, g2, g3, w0, w1, w2, w3)
	}
}

func TestSADSIMDAbsDiffUnsignedEdges(t *testing.T) {
	var a, b, got [16]uint8
	for i := range a {
		a[i] = uint8(i * 17)
		b[i] = 255 - a[i]
	}
	absd := absDiffU8x16(archsimd.LoadUint8x16Array(&a), archsimd.LoadUint8x16Array(&b))
	absd.StoreArray(&got)
	for i := range got {
		want := int(a[i]) - int(b[i])
		if want < 0 {
			want = -want
		}
		if int(got[i]) != want {
			t.Fatalf("lane %d: absdiff(%d,%d)=%d, want %d", i, a[i], b[i], got[i], want)
		}
	}
}

func TestSADSIMDZeroAlloc(t *testing.T) {
	const stride = 64
	src, ref := makeSADPlane(1, stride*64)
	var sink int
	if allocs := testing.AllocsPerRun(200, func() {
		a, b, c, d := sad8x8x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], stride)
		sink += a + b + c + d
	}); allocs != 0 {
		t.Fatalf("8x8x4 SIMD allocated %.2f objects/op, want 0 (sink=%d)", allocs, sink)
	}
	_ = sink
}

var sadSIMDBenchSink int

func benchPlane() ([]byte, []byte) {
	src := make([]byte, 128*128)
	ref := make([]byte, 128*128)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	return src, ref
}

func BenchmarkSAD8x8x4_Scalar(b *testing.B) {
	src, ref := benchPlane()
	r0, r1, r2, r3 := ref, ref[4:], ref[8:], ref[12:]
	var sums [4]int
	b.ReportAllocs()
	for b.Loop() {
		s0, s1, s2, s3 := sad8x8x4PureGo(src, r0, r1, r2, r3, 128)
		sums[0] += s0
		sums[1] += s1
		sums[2] += s2
		sums[3] += s3
	}
	sadSIMDBenchSink = sums[0] + sums[1] + sums[2] + sums[3]
	if sadSIMDBenchSink == 0 {
		b.Fatal("unexpected zero SAD")
	}
}

func BenchmarkSAD8x8x4_SIMD(b *testing.B) {
	src, ref := benchPlane()
	r0, r1, r2, r3 := ref, ref[4:], ref[8:], ref[12:]
	var sums [4]int
	b.ReportAllocs()
	for b.Loop() {
		s0, s1, s2, s3 := sad8x8x4SIMD(src, r0, r1, r2, r3, 128)
		sums[0] += s0
		sums[1] += s1
		sums[2] += s2
		sums[3] += s3
	}
	sadSIMDBenchSink = sums[0] + sums[1] + sums[2] + sums[3]
	if sadSIMDBenchSink == 0 {
		b.Fatal("unexpected zero SAD")
	}
}
