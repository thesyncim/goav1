//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// TestMetricSIMDBinding confirms the goexperiment.simd init bound the
// pixelStats dispatch to the pixelStatsSIMD wrappers.
func TestMetricSIMDBinding(t *testing.T) {
	if !gosimdKernelsSupported() {
		t.Skip("Go SIMD kernels unsupported on this CPU")
	}
	psName := runtime.FuncForPC(reflect.ValueOf(pixelStats8x8Impl).Pointer()).Name()
	if !strings.Contains(psName, "pixelStats8x8SIMD") {
		t.Errorf("pixelStats8x8Impl bound to %q, want pixelStats8x8SIMD", psName)
	}
}

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

// TestRDStatsGosimdBinding confirms the goexperiment.simd init bound the
// residual and RD-statistics dispatch vars to the Go SIMD kernels
// wherever the CPU supports them.
func TestRDStatsGosimdBinding(t *testing.T) {
	if !gosimdKernelsSupported() {
		t.Skip("Go SIMD kernels unsupported on this CPU")
	}
	cases := []struct {
		name string
		fn   any
		want string
	}{
		{"residualBlockImpl", residualBlockImpl, "residualBlockSIMD"},
		{"rdStatsBlockImpl", rdStatsBlockImpl, "rdStatsBlockSIMD"},
	}
	for _, tc := range cases {
		name := runtime.FuncForPC(reflect.ValueOf(tc.fn).Pointer()).Name()
		if !strings.Contains(name, tc.want) {
			t.Errorf("%s bound to %q, want %q", tc.name, name, tc.want)
		}
	}
}

func TestResidualBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5D01))
	shapes := []struct{ w, h int }{{4, 4}, {8, 8}, {8, 16}, {12, 4}, {16, 16}, {16, 8}, {24, 3}, {32, 32}, {64, 16}, {40, 2}}
	for _, sh := range shapes {
		for trial := range 200 {
			stride := sh.w + rng.Intn(9)
			predStride := sh.w + rng.Intn(9)
			src := make([]byte, stride*sh.h+8)
			pred := make([]byte, predStride*sh.h+8)
			fillRDStatsBytes(rng, src)
			fillRDStatsBytes(rng, pred)
			srcOff := rng.Intn(4)
			got := make([]int16, sh.w*sh.h)
			want := make([]int16, sh.w*sh.h)
			residualBlockSIMD(got, src, srcOff, stride, pred, predStride, sh.w, sh.h)
			residualBlockPureGo(want, src, srcOff, stride, pred, predStride, sh.w, sh.h)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%dx%d trial %d [%d]: got %d want %d", sh.w, sh.h, trial, i, got[i], want[i])
				}
			}
		}
	}
}

func TestRDStatsBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5D02))
	for _, count := range []int{0, 1, 3, 7, 8, 9, 15, 16, 31, 64, 256, 1024} {
		for trial := range 300 {
			tran := make([]int32, count)
			qcoeff := make([]int16, count)
			fillRDStatsCoeffs(rng, tran, qcoeff, trial)
			step := int32(rng.Intn(29247) + 1)
			ts := uint8(rng.Intn(4))
			if trial%3 == 0 {
				step = 1 << 15
			}
			wd, wc, wr, wz := rdStatsBlockPureGo(tran, qcoeff, count, step, ts)
			gd, gc, gr, gz := rdStatsBlockSIMD(tran, qcoeff, count, step, ts)
			if gd != wd || gc != wc || gr != wr || gz != wz {
				t.Fatalf("count %d trial %d step %d ts %d: got (%d,%d,%d,%v) want (%d,%d,%d,%v)",
					count, trial, step, ts, gd, gc, gr, gz, wd, wc, wr, wz)
			}
		}
	}
}

func TestRDStatsBlockSIMDSingleNonzero(t *testing.T) {
	for _, count := range []int{8, 9, 16, 17, 31, 64} {
		tran := make([]int32, count)
		qcoeff := make([]int16, count)
		for index := range tran {
			tran[index] = int32(index*37 - 1000)
		}
		for active := range qcoeff {
			for _, level := range []int16{-32768, -1, 1, 32767} {
				qcoeff[active] = level
				wantSkip, wantCode, wantRate, wantZero := rdStatsBlockPureGo(tran, qcoeff, count, 32768, 3)
				gotSkip, gotCode, gotRate, gotZero := rdStatsBlockSIMD(tran, qcoeff, count, 32768, 3)
				if gotSkip != wantSkip || gotCode != wantCode || gotRate != wantRate || gotZero != wantZero {
					t.Fatalf("count=%d active=%d level=%d mismatch", count, active, level)
				}
				qcoeff[active] = 0
			}
		}
	}
}

func TestRDStatsBlockSIMDNoAlloc(t *testing.T) {
	tran := make([]int32, 256)
	qcoeff := make([]int16, 256)
	for i := range tran {
		tran[i] = int32(i*37 - 4000)
		qcoeff[i] = int16(i%7 - 3)
	}
	if a := testing.AllocsPerRun(100, func() { _, _, _, _ = rdStatsBlockSIMD(tran, qcoeff, 256, 21, 1) }); a != 0 {
		t.Errorf("rdStatsBlockSIMD: %.1f allocs/op, want 0", a)
	}
}

// fillRDStatsBytes fills b with random bytes, with saturated runs so the
// widening edges are reached.
func fillRDStatsBytes(rng *rand.Rand, b []byte) {
	for i := range b {
		switch rng.Intn(4) {
		case 0:
			b[i] = 255
		case 1:
			b[i] = 0
		default:
			b[i] = byte(rng.Intn(256))
		}
	}
}

// fillRDStatsCoeffs fills transform and quantized coefficients. Some trials
// hold all-zero levels, and some hold the int16 extremes.
func fillRDStatsCoeffs(rng *rand.Rand, tran []int32, qcoeff []int16, trial int) {
	for i := range tran {
		switch {
		case trial%5 == 0:
			tran[i] = int32(rng.Intn(1<<20) - 1<<19)
		default:
			tran[i] = int32(rng.Intn(8192) - 4096)
		}
		switch {
		case trial%4 == 0:
			qcoeff[i] = 0
		case trial%4 == 1 && rng.Intn(2) == 0:
			qcoeff[i] = int16(rng.Intn(65536) - 32768)
		default:
			qcoeff[i] = int16(rng.Intn(41) - 20)
		}
	}
}

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
