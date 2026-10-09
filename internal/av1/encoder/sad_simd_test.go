// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// Differential tests for the shared Go-native-SIMD SAD kernels. Every input
// window is sized to exactly the bytes its kernel reads (for the step-4 family,
// including the 12-byte tail of the last candidate), so under checkptr=2 an
// over-read of any row or column pair fails instead of reading slack.

const (
	sadFillRandom = iota
	sadFillMax
	sadFillZero
	sadFillBinary
)

// sadFillPairs cycles the source/reference fill modes so every kernel sees the
// extreme absolute differences (255 against 0) as well as random data.
var sadFillPairs = [][2]int{
	{sadFillRandom, sadFillRandom},
	{sadFillMax, sadFillZero},
	{sadFillZero, sadFillMax},
	{sadFillBinary, sadFillBinary},
	{sadFillRandom, sadFillMax},
}

func fillSADWindow(rng *rand.Rand, b []byte, mode int) {
	for i := range b {
		switch mode {
		case sadFillMax:
			b[i] = 255
		case sadFillZero:
			b[i] = 0
		case sadFillBinary:
			b[i] = uint8(rng.Intn(2)) * 255
		default:
			b[i] = uint8(rng.Intn(256))
		}
	}
}

// sadWindow returns exactly n bytes starting at offset within a backing array
// of length offset+n. The capacity is clipped to n as well, so the window's
// last byte is the end of the allocation.
func sadWindow(rng *rand.Rand, n, offset, mode int) []byte {
	backing := make([]byte, offset+n)
	fillSADWindow(rng, backing, mode)
	return backing[offset : offset+n : offset+n]
}

// funcName returns the symbol name of a function value.
func funcName(f any) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

// TestSADDispatchBoundToSIMD proves that when the CPU supports the Go-native
// kernels, every SAD dispatch variable is bound to one of them.
func TestSADDispatchBoundToSIMD(t *testing.T) {
	if !sadSIMDBound {
		t.Skip("CPU lacks the SIMD features the SAD kernels need")
	}
	bound := []struct {
		name string
		fn   any
	}{
		{"sad8x8Impl", sad8x8Impl},
		{"sad16x16Impl", sad16x16Impl},
		{"sad32x32Impl", sad32x32Impl},
		{"sad8x8x4Impl", sad8x8x4Impl},
		{"sad8x8x4Step4Impl", sad8x8x4Step4Impl},
		{"sad16x16x4Impl", sad16x16x4Impl},
		{"sad16x16x4Step4Impl", sad16x16x4Step4Impl},
		{"sad32x32x4Impl", sad32x32x4Impl},
		{"sad32x32x4Step4Impl", sad32x32x4Step4Impl},
		{"sad8x8DualImpl", sad8x8DualImpl},
		{"sad16x16DualImpl", sad16x16DualImpl},
		{"sad32x32DualImpl", sad32x32DualImpl},
		{"sad8x8CompoundAvgBlockImpl", sad8x8CompoundAvgBlockImpl},
	}
	for _, c := range bound {
		if name := funcName(c.fn); !strings.Contains(name, "SIMD") {
			t.Errorf("%s bound to %q, want a Go-native SIMD kernel", c.name, name)
		}
	}
}

type sadDualKernel func(src []byte, srcStride int, ref []byte, refStride int) int

// TestSADSIMDDualKernelsMatchPureGo covers the two-stride leaves, including the
// single-stride use (stride equal on both sides) that the 8x8, 16x16, and
// 32x32 wrappers take.
func TestSADSIMDDualKernelsMatchPureGo(t *testing.T) {
	cases := []struct {
		name       string
		h, w       int
		simd, want sadDualKernel
	}{
		{"8x8", 8, 8, sad8x8SIMD, sad8x8DualPureGo},
		{"16x16", 16, 16, sad16x16SIMD, sad16x16DualPureGo},
		{"32x32", 32, 32, sad32x32SIMD, sad32x32DualPureGo},
		{"64x64", 64, 64, sad64x64SIMD, sad64x64DualPureGo},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(c.h)*131 + 7))
			for iter := range 400 {
				srcStride := c.w + rng.Intn(24)
				refStride := c.w + rng.Intn(24)
				pair := sadFillPairs[iter%len(sadFillPairs)]
				src := sadWindow(rng, (c.h-1)*srcStride+c.w, rng.Intn(9), pair[0])
				ref := sadWindow(rng, (c.h-1)*refStride+c.w, rng.Intn(9), pair[1])
				if got, want := c.simd(src, srcStride, ref, refStride), c.want(src, srcStride, ref, refStride); got != want {
					t.Fatalf("%s iter %d strides %d/%d: SIMD %d, PureGo %d", c.name, iter, srcStride, refStride, got, want)
				}
			}
		})
	}
}

// TestSADSIMDSingleStrideWrappersMatchPureGo checks the single-stride Impl
// adapters bound into the dispatch variables.
func TestSADSIMDSingleStrideWrappersMatchPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	for _, c := range []struct {
		n    int
		simd func(src, ref []byte, stride int) int
		want func(src, ref []byte, stride int) int
	}{
		{16, sad16x16SingleSIMD, sad16x16PureGo},
		{32, sad32x32SingleSIMD, sad32x32PureGo},
	} {
		for iter := range 200 {
			stride := c.n + rng.Intn(16)
			pair := sadFillPairs[iter%len(sadFillPairs)]
			src := sadWindow(rng, (c.n-1)*stride+c.n, rng.Intn(5), pair[0])
			ref := sadWindow(rng, (c.n-1)*stride+c.n, rng.Intn(5), pair[1])
			if got, want := c.simd(src, ref, stride), c.want(src, ref, stride); got != want {
				t.Fatalf("%dx%d single stride %d iter %d: SIMD %d, PureGo %d", c.n, c.n, stride, iter, got, want)
			}
		}
	}
	for iter := range 200 {
		stride := 8 + rng.Intn(16)
		pair := sadFillPairs[iter%len(sadFillPairs)]
		src := sadWindow(rng, 7*stride+8, rng.Intn(5), pair[0])
		ref := sadWindow(rng, 7*stride+8, rng.Intn(5), pair[1])
		if got, want := sad8x8SingleSIMD(src, ref, stride, 1<<30), sad8x8PureGo(src, ref, stride, 1<<30); got != want {
			t.Fatalf("8x8 single stride %d iter %d: SIMD %d, PureGo %d", stride, iter, got, want)
		}
	}
}

type sadX4Kernel func(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int)

type sadStep4Kernel func(src, ref []byte, stride int) (int, int, int, int)

// TestSADSIMDX4KernelsMatchPureGo covers the four-reference kernels with a
// shared stride. Each reference is an independent exact window.
func TestSADSIMDX4KernelsMatchPureGo(t *testing.T) {
	cases := []struct {
		name string
		h, w int
		simd sadX4Kernel
		want sadX4Kernel
	}{
		{"8x8x4", 8, 8, sad8x8x4SIMD, sad8x8x4PureGo},
		{"16x16x4", 16, 16, sad16x16x4SIMD, sad16x16x4PureGo},
		{"32x32x4", 32, 32, sad32x32x4SIMD, sad32x32x4PureGo},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(c.h)*17 + 3))
			for iter := range 300 {
				stride := c.w + rng.Intn(24)
				pair := sadFillPairs[iter%len(sadFillPairs)]
				src := sadWindow(rng, (c.h-1)*stride+c.w, rng.Intn(9), pair[0])
				var refs [4][]byte
				for i := range refs {
					refs[i] = sadWindow(rng, (c.h-1)*stride+c.w, rng.Intn(9), pair[1])
				}
				g0, g1, g2, g3 := c.simd(src, refs[0], refs[1], refs[2], refs[3], stride)
				w0, w1, w2, w3 := c.want(src, refs[0], refs[1], refs[2], refs[3], stride)
				if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
					t.Fatalf("%s iter %d stride %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
						c.name, iter, stride, g0, g1, g2, g3, w0, w1, w2, w3)
				}
			}
		})
	}
}

// TestSADSIMDStep4KernelsMatchPureGo covers the raster step-4 candidate groups.
// The reference window includes the 12-byte run past the block that the last
// candidate reads.
func TestSADSIMDStep4KernelsMatchPureGo(t *testing.T) {
	cases := []struct {
		name string
		h, w int
		simd sadStep4Kernel
		want sadStep4Kernel
	}{
		{"8x8x4step4", 8, 8, sad8x8x4Step4SIMD, sad8x8x4Step4PureGo},
		{"16x16x4step4", 16, 16, sad16x16x4Step4SIMD, sad16x16x4Step4PureGo},
		{"32x32x4step4", 32, 32, sad32x32x4Step4SIMD, sad32x32x4Step4PureGo},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(c.h)*29 + 5))
			for iter := range 300 {
				stride := c.w + rng.Intn(24)
				pair := sadFillPairs[iter%len(sadFillPairs)]
				src := sadWindow(rng, (c.h-1)*stride+c.w, rng.Intn(9), pair[0])
				ref := sadWindow(rng, (c.h-1)*stride+c.w+12, rng.Intn(9), pair[1])
				g0, g1, g2, g3 := c.simd(src, ref, stride)
				w0, w1, w2, w3 := c.want(src, ref, stride)
				if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
					t.Fatalf("%s iter %d stride %d: SIMD (%d,%d,%d,%d) want (%d,%d,%d,%d)",
						c.name, iter, stride, g0, g1, g2, g3, w0, w1, w2, w3)
				}
			}
		})
	}
}

// TestSADSIMDCompoundAvgMatchesPureGo covers the compound-average precheck,
// including the (a+b+1)>>1 rounding on odd sums.
func TestSADSIMDCompoundAvgMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	for iter := range 400 {
		srcStride := 8 + rng.Intn(24)
		ref0Stride := 8 + rng.Intn(24)
		ref1Stride := 8 + rng.Intn(24)
		pair := sadFillPairs[iter%len(sadFillPairs)]
		src := sadWindow(rng, 7*srcStride+8, rng.Intn(9), pair[0])
		ref0 := sadWindow(rng, 7*ref0Stride+8, rng.Intn(9), pair[1])
		ref1 := sadWindow(rng, 7*ref1Stride+8, rng.Intn(9), sadFillBinary)
		got := sad8x8CompoundAvgSIMD(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
		want := sad8x8CompoundAvgBlockPureGo(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
		if got != want {
			t.Fatalf("iter %d: SIMD %d, PureGo %d", iter, got, want)
		}
	}
}

// TestSADSIMDExtremeTotals pins the largest totals each shape can produce
// (every byte pair differs by 255), which exceed 16 bits for the wide shapes.
func TestSADSIMDExtremeTotals(t *testing.T) {
	fill := func(n int, v byte) []byte {
		b := make([]byte, n)
		for i := range b {
			b[i] = v
		}
		return b
	}
	for _, c := range []struct {
		name string
		n    int
		got  func(src, ref []byte, stride int) int
	}{
		{"8x8", 8, func(s, r []byte, st int) int { return sad8x8SIMD(s, st, r, st) }},
		{"16x16", 16, sad16x16SingleSIMD},
		{"32x32", 32, sad32x32SingleSIMD},
		{"64x64", 64, func(s, r []byte, st int) int { return sad64x64SIMD(s, st, r, st) }},
	} {
		src := fill(c.n*c.n, 255)
		ref := fill(c.n*c.n, 0)
		want := c.n * c.n * 255
		if g := c.got(src, ref, c.n); g != want {
			t.Errorf("%s all-255 vs all-0: SIMD %d, want %d", c.name, g, want)
		}
	}
}

// TestSADSIMDZeroAlloc keeps the hot-path kernels allocation-free.
func TestSADSIMDZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const stride = 80
	src := sadWindow(rng, 63*stride+64, 0, sadFillRandom)
	ref := sadWindow(rng, 63*stride+64+12, 0, sadFillRandom)
	var sink int
	allocs := testing.AllocsPerRun(100, func() {
		sink += sad8x8SIMD(src, stride, ref, stride)
		sink += sad16x16SIMD(src, stride, ref, stride)
		sink += sad32x32SIMD(src, stride, ref, stride)
		sink += sad64x64SIMD(src, stride, ref, stride)
		a, b, c, d := sad8x8x4SIMD(src, ref, ref[4:], ref[8:], ref[12:], stride)
		sink += a + b + c + d
		a, b, c, d = sad8x8x4Step4SIMD(src, ref, stride)
		sink += a + b + c + d
		a, b, c, d = sad16x16x4Step4SIMD(src, ref, stride)
		sink += a + b + c + d
		a, b, c, d = sad32x32x4Step4SIMD(src, ref, stride)
		sink += a + b + c + d
	})
	if allocs != 0 {
		t.Fatalf("SAD SIMD kernels allocated %.2f objects/op, want 0 (sink=%d)", allocs, sink)
	}
}
