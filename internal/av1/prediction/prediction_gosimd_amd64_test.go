//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"fmt"
	"math/rand"
	"simd/archsimd"
	"slices"
	"testing"
)

// TestAMD64CFL16MatchesPureGo compares the AVX 16bpc CfL subsampling and
// average subtraction with the scalar kernels, including out-of-range
// samples (both must report the error).
func TestAMD64CFL16MatchesPureGo(t *testing.T) {
	assertDispatchTarget(t, "subsampleLuma16Impl", subsampleLuma16Impl, "subsampleLuma16AVX")
	assertDispatchTarget(t, "subtractCFLAverageImpl", subtractCFLAverageImpl, "subtractCFLAverageAVX")
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 3000; trial++ {
		max := uint16(1023)
		if trial&1 != 0 {
			max = 4095
		}
		subX, subY := trial%3 != 0, trial%3 == 2
		w := 4 << rng.Intn(4)
		h := 4 << rng.Intn(4)
		outW, outH := w, h
		if subX {
			outW = w / 2
		}
		if subY {
			outH = h / 2
		}
		stride := w + 8
		input := make([]uint16, stride*h+16)
		for i := range input {
			input[i] = uint16(rng.Intn(int(max) + 1))
		}
		if trial%50 == 0 {
			input[rng.Intn(stride*h)] = max + 1
		}
		got := make([]uint16, CFLBufSquare)
		want := make([]uint16, CFLBufSquare)
		errG := subsampleLuma16AVX(got, input, stride, w, h, outW, outH, subX, subY, max)
		errW := subsampleLuma16PureGo(want, input, stride, w, h, outW, outH, subX, subY, max)
		if (errG == nil) != (errW == nil) {
			t.Fatalf("trial %d: error mismatch %v vs %v", trial, errG, errW)
		}
		if errW != nil {
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d %dx%d sub=%v,%v: idx %d got %d want %d", trial, w, h, subX, subY, i, got[i], want[i])
			}
		}
		log2, _ := log2PowerOfTwoInt(outW * outH)
		g, wv := make([]int16, CFLBufSquare), make([]int16, CFLBufSquare)
		subtractCFLAverageAVX(got, g, outW, outH, log2)
		subtractCFLAveragePureGo(want, wv, outW, outH, log2)
		for i := range g {
			if g[i] != wv[i] {
				t.Fatalf("trial %d: subtract idx %d got %d want %d", trial, i, g[i], wv[i])
			}
		}
	}
}

// filterIntra16Depths are the high-bit-depth clamps the SIMD kernel must honour:
// 10-bit (max 1023) and 12-bit (max 4095, whose tap sums overflow int16 and so
// exercise the 32-bit widening path).
var filterIntra16Depths = [...]int{1023, 4095}

func filterIntra16SIMDEdges(width, height int, max int, seed uint32) IntraEdges {
	above, s1 := samplesFor(width, uint16(max), seed)
	left, _ := samplesFor(height, uint16(max), s1)
	return IntraEdges{
		Above:              above,
		Left:               left,
		AboveLeft:          uint16((seed >> 3) % uint32(max+1)),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
}

// TestFilterIntraSIMDBinding asserts the arm64 SIMD build binds the Go-native
// filter-intra kernels, so the differential tests below exercise the production
// dispatch target.
func TestAMD64FilterIntra16Binding(t *testing.T) {

	assertDispatchTarget(t, "predictFilterIntra16Impl", predictFilterIntra16Impl, "predictFilterIntraBlockDirect16AVX2")
}

// TestFilterIntra16SIMDMatchesPureGo executes the 16-bit SIMD kernel directly
// against the pure-Go reference for every mode, every valid transform size, both
// high bit depths and a spread of pseudo-random edges.
func TestAMD64FilterIntra16MatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
					edges := filterIntra16SIMDEdges(sz.width, sz.height, max, seed+uint32(mode)*131)
					base := makeDispatchBlock(sz.width, sz.height, 2)
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictFilterIntraBlockDirect16AVX2(got, sz.width, sz.height, mode, edges, max)
					predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
					diffBlocks(t, "filter-intra16-simd", max, sz.width*2, sz.height, got, want)
				}
			}
		}
	}
}

// TestFilterIntra16SIMDEdgeValues stresses the [0,max] clamp corners at both
// high bit depths, where the 32-bit tap sums saturate in both directions.
func TestAMD64FilterIntra16EdgeValues(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, pat := range []uint16{0, uint16(max), uint16(max / 2), uint16(max - 1)} {
					above := make([]uint16, sz.width)
					left := make([]uint16, sz.height)
					for i := range above {
						above[i] = pat
						if i%3 == 0 {
							above[i] = uint16(max) - pat
						}
					}
					for i := range left {
						left[i] = uint16(max) - pat
						if i%2 == 0 {
							left[i] = pat
						}
					}
					for _, al := range []uint16{0, uint16(max)} {
						edges := IntraEdges{
							Above:              above,
							Left:               left,
							AboveLeft:          al,
							AboveAvailable:     true,
							LeftAvailable:      true,
							AboveLeftAvailable: true,
						}
						base := makeDispatchBlock(sz.width, sz.height, 2)
						got := cloneBlock(base)
						want := cloneBlock(base)
						predictFilterIntraBlockDirect16AVX2(got, sz.width, sz.height, mode, edges, max)
						predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
						diffBlocks(t, "filter-intra16-simd-edge", max, sz.width*2, sz.height, got, want)
					}
				}
			}
		}
	}
}

// TestFilterIntra16SIMDZeroAlloc protects the hot-path contract: the 16-bit SIMD
// predictor must not allocate per call.
func TestAMD64FilterIntra16ZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntra16SIMDEdges(w, h, 1023, 0x24)
	block := makeDispatchBlock(w, h, 2)
	fn := func() {
		predictFilterIntraBlockDirect16AVX2(block, w, h, FilterIntraModePaeth, edges, 1023)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra16 SIMD allocated %f times per call", allocs)
	}
}

// BenchmarkAMD64FilterIntra16 compares the AVX2 10-bit filter-intra kernel
// with the scalar predictor it replaces.
func BenchmarkAMD64FilterIntra16(b *testing.B) {
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}} {
		w, h := sz[0], sz[1]
		edges := filterIntra16SIMDEdges(w, h, 1023, 0x31)
		block := makeDispatchBlock(w, h, 2)
		for _, impl := range []struct {
			name string
			fn   func(planeBlock, int, int, FilterIntraMode, IntraEdges, int)
		}{{"generic", predictFilterIntraBlockDirect16}, {"avx2", predictFilterIntraBlockDirect16AVX2}} {
			b.Run(fmt.Sprintf("shape=%dx%d/impl=%s", w, h, impl.name), func(b *testing.B) {
				for b.Loop() {
					impl.fn(block, w, h, FilterIntraModePaeth, edges, 1023)
				}
			})
		}
	}
}

// TestFilterIntra8SIMDBinding asserts that on an AVX2 machine the amd64 SIMD
// build binds the Go-native 8-bit filter-intra kernel to the dispatch slot.
func TestFilterIntra8SIMDBinding(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not available; the SIMD kernel is not bound")
	}
	assertDispatchTarget(t, "predictFilterIntra8Impl", predictFilterIntra8Impl, "predictFilterIntraBlockDirect8SIMD")
}

// withIntraEdgeScalar runs f with the AVX2 edge kernels disabled.
func withIntraEdgeScalar(f func()) {
	saved := intraEdgeAVX2
	intraEdgeAVX2 = false
	defer func() { intraEdgeAVX2 = saved }()
	f()
}

func TestFilterIntraEdgeAVX2MatchesScalar(t *testing.T) {
	if !intraEdgeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(17))
	scratch := make([]uint16, intraEdgeMaxSize)
	for trial := 0; trial < 20000; trial++ {
		n := 1 + rng.Intn(intraEdgeMaxSize)
		if trial%3 != 0 {
			n = 1 + rng.Intn(40)
		}
		bd := []uint8{8, 10, 12}[trial%3]
		strength := uint8(1 + rng.Intn(3))
		src := make([]uint16, n)
		for i := range src {
			src[i] = uint16(rng.Intn(1 << bd))
			if trial%7 == 0 {
				src[i] = 1<<bd - 1
			}
		}
		want, got := slices.Clone(src), slices.Clone(src)
		withIntraEdgeScalar(func() {
			if err := FilterIntraEdgeBounded(want, scratch, strength, bd, false); err != nil {
				t.Fatal(err)
			}
		})
		if err := FilterIntraEdgeBounded(got, scratch, strength, bd, false); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d n=%d bd=%d strength=%d:\n got %v\nwant %v", trial, n, bd, strength, got, want)
		}
	}
}

func TestUpsampleIntraEdgeAVX2MatchesScalar(t *testing.T) {
	if !intraEdgeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(18))
	scratch := make([]uint16, intraEdgeMaxSize)
	for trial := 0; trial < 20000; trial++ {
		size := 1 + rng.Intn(intraEdgeMaxUpsampleSize)
		origin := 2 + rng.Intn(4)
		bd := []uint8{8, 10, 12}[trial%3]
		src := make([]uint16, origin+2*size+3)
		for i := range src {
			src[i] = uint16(rng.Intn(1 << bd))
			if trial%5 == 0 && i%2 == 0 {
				src[i] = 1<<bd - 1
			}
		}
		want, got := slices.Clone(src), slices.Clone(src)
		withIntraEdgeScalar(func() {
			if err := UpsampleIntraEdgeBounded(want, origin, size, scratch, bd, false); err != nil {
				t.Fatal(err)
			}
		})
		if err := UpsampleIntraEdgeBounded(got, origin, size, scratch, bd, false); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d size=%d origin=%d bd=%d:\n got %v\nwant %v", trial, size, origin, bd, got, want)
		}
	}
}

// BenchmarkAMD64IntraEdge compares the scalar and AVX2 edge filter and
// upsampler on the edge lengths of small directional blocks.
func BenchmarkAMD64IntraEdge(b *testing.B) {
	scratch := make([]uint16, intraEdgeMaxSize)
	src := make([]uint16, 64)
	for i := range src {
		src[i] = uint16(i * 7 % 256)
	}
	edge := make([]uint16, 64)
	run := func(name string, f func()) {
		b.Run(name+"/impl=generic", func(b *testing.B) {
			withIntraEdgeScalar(func() {
				for i := 0; i < b.N; i++ {
					f()
				}
			})
		})
		b.Run(name+"/impl=avx2", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f()
			}
		})
	}
	run("shape=filter17/s1", func() {
		copy(edge, src)
		_ = FilterIntraEdgeBounded(edge[:17], scratch, 1, 8, true)
	})
	run("shape=filter33/s3", func() {
		copy(edge, src)
		_ = FilterIntraEdgeBounded(edge[:33], scratch, 3, 8, true)
	})
	run("shape=upsample8", func() {
		copy(edge, src)
		_ = UpsampleIntraEdgeBounded(edge, 2, 8, scratch, 8, true)
	})
}

// The amd64 SIMD kernels are differentially validated against the pure-Go
// references directly, not through the dispatch slot, so the kernels are tested on
// every amd64 build that has SIMD (native or Rosetta). Dispatch binding is covered
// by TestFilterIntra8SIMDBinding and TestStaticSIMDBinding.

func TestPaethSIMDMatchesPureGo(t *testing.T) {
	for _, w := range []int{16, 32, 48, 64} {
		for _, h := range []int{4, 8, 16, 32, 64} {
			above, s1 := samplesFor(w, 0xff, 0x111+uint32(w))
			left, _ := samplesFor(h, 0xff, s1)
			for _, al := range []uint16{0, 1, 128, 255} {
				base := makeDispatchBlock(w, h, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethSIMD(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2", 0xff, w, h, got, want)
			}
		}
	}
}

func TestPaethSIMDEdgeValues(t *testing.T) {
	const w, h = 32, 16
	patterns := []func(i int) uint16{
		func(i int) uint16 { return 0 },
		func(i int) uint16 { return 255 },
		func(i int) uint16 {
			if i%2 == 0 {
				return 0
			}
			return 255
		},
		func(i int) uint16 { return uint16(i * 17 % 256) },
	}
	for _, ap := range patterns {
		for _, lp := range patterns {
			above := make([]uint16, w)
			left := make([]uint16, h)
			for i := range above {
				above[i] = ap(i)
			}
			for i := range left {
				left[i] = lp(i)
			}
			for _, al := range []uint16{0, 128, 255} {
				base := makeDispatchBlock(w, h, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethSIMD(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2-edge", 0xff, w, h, got, want)
			}
		}
	}
}

func TestSmoothSIMDMatchesPureGo(t *testing.T) {
	for _, w := range []int{16, 32, 64} {
		for _, h := range []int{4, 8, 16, 32, 64} {
			above, s1 := samplesFor(w, 0xff, 0x222+uint32(w))
			left, _ := samplesFor(h, 0xff, s1)
			weightsW, _ := smoothWeightsForSize(w)
			weightsH, _ := smoothWeightsForSize(h)
			belowPred := left[h-1]
			rightPred := above[w-1]
			base := makeDispatchBlock(w, h, 1)

			got := cloneBlock(base)
			want := cloneBlock(base)
			predictSmoothSIMD(got, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			predictSmoothPureGo(want, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			diffBlocks(t, "smooth-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothVerticalSIMD(got, 1, weightsH, above, belowPred)
			predictSmoothVerticalPureGo(want, 1, weightsH, above, belowPred)
			diffBlocks(t, "smooth_v-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothHorizontalSIMD(got, 1, weightsW, left, rightPred)
			predictSmoothHorizontalPureGo(want, 1, weightsW, left, rightPred)
			diffBlocks(t, "smooth_h-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestApplyCFLSIMDMatchesPureGo(t *testing.T) {
	dims := [][2]int{{16, 16}, {32, 32}, {16, 8}, {32, 16}, {16, 32}, {4, 4}, {4, 8}, {4, 16}}
	var seed uint32 = 0x7c3
	for _, dim := range dims {
		w, h := dim[0], dim[1]
		ac := make([]int16, CFLBufSquare)
		for i := range ac {
			seed = seed*1664525 + 1013904223
			ac[i] = int16(int32(seed>>8)%4096 - 2048)
		}
		for _, alpha := range []int{-16, -7, -1, 0, 1, 5, 16} {
			base := makeDispatchBlock(w, h, 1)
			for i := range base.pix {
				base.pix[i] = byte((i*13 + 7) & 0xff)
			}
			got := cloneBlock(base)
			want := cloneBlock(base)
			applyCFLSIMD(got, 1, w, h, ac, alpha, 0xff)
			applyCFLPureGo(want, 1, w, h, ac, alpha, 0xff)
			diffBlocks(t, "applycfl-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestSumSamplesSIMDMatchesPureGo(t *testing.T) {
	for _, n := range []int{16, 32, 48, 64, 20, 4, 100} {
		for _, max := range []uint16{0xff, 0x3ff, 0xfff} {
			s, _ := samplesFor(n, max, 0x5a5+uint32(n)+uint32(max))
			if got, want := sumSamplesSIMD(s), sumSamplesPureGo(s); got != want {
				t.Fatalf("sumSamplesSIMD n=%d max=%d got=%d want=%d", n, max, got, want)
			}
		}
	}
}

func TestSubsampleLuma8SIMDMatchesPureGo(t *testing.T) {
	type cfg struct {
		w, h       int
		subX, subY bool
	}
	cfgs := []cfg{
		{8, 8, false, false}, {16, 16, false, false}, {32, 32, false, false},
		{24, 16, false, false},
		{16, 16, true, false}, {32, 16, true, false}, {16, 8, true, false},
		{48, 16, true, false},
		{16, 16, true, true}, {32, 32, true, true}, {16, 32, true, true},
		{8, 8, true, true}, {4, 4, true, true}, // width 4 -> pure-Go fallback
		{12, 8, false, false}, // outW=12 not %8 -> fallback
	}
	var seed uint32 = 0x1234
	for _, c := range cfgs {
		stride := c.w + 5
		input := make([]uint8, stride*c.h)
		for i := range input {
			seed = seed*1664525 + 1013904223
			input[i] = uint8(seed >> 13)
		}
		outW, outH := c.w, c.h
		if c.subX {
			outW >>= 1
		}
		if c.subY {
			outH >>= 1
		}
		got := make([]uint16, CFLBufSquare)
		want := make([]uint16, CFLBufSquare)
		subsampleLuma8SIMD(got, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		subsampleLuma8PureGo(want, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subsample %+v idx=%d got=%d want=%d", c, i, got[i], want[i])
			}
		}
	}
}

// dirEdge builds a uint16 edge slice of length n holding 8-bit content.
func dirEdge(n int, seed uint32) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		seed = seed*1664525 + 1013904223
		out[i] = uint16(seed >> 24)
	}
	return out
}

func TestDirRowInterp8SIMDMatchesPureGo(t *testing.T) {
	for _, width := range []int{8, 16, 24, 32, 64, 4, 12, 20} {
		above := dirEdge(width+8, 0x77+uint32(width))
		for _, shift := range []int{0, 1, 5, 12, 16, 31} {
			for _, base := range []int{0, 1, 3} {
				maxBase := width + base // fully interpolated (asm path when width%8==0)
				got := make([]byte, width)
				want := make([]byte, width)
				dirRowInterp8SIMD(got, above, base, shift, maxBase, width)
				dirRowInterp8PureGo(want, above, base, shift, maxBase, width)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("dirRowInterp w=%d shift=%d base=%d i=%d got=%d want=%d", width, shift, base, i, got[i], want[i])
					}
				}
				// Also exercise the clamp path (base+width > maxBase) -> pure-Go fallback.
				clampMax := base + width/2
				got2 := make([]byte, width)
				want2 := make([]byte, width)
				dirRowInterp8SIMD(got2, above, base, shift, clampMax, width)
				dirRowInterp8PureGo(want2, above, base, shift, clampMax, width)
				for i := range got2 {
					if got2[i] != want2[i] {
						t.Fatalf("dirRowInterp clamp w=%d shift=%d i=%d got=%d want=%d", width, shift, i, got2[i], want2[i])
					}
				}
			}
		}
	}
}

func TestDirAboveRun8SIMDMatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15, 33} {
		ref := dirEdge(count+8, 0x314+uint32(count))
		for _, shift := range []int{0, 1, 7, 16, 30, 31} {
			got := make([]byte, count)
			want := make([]byte, count)
			dirAboveRun8SIMD(got, ref, shift, count)
			dirAboveRun8PureGo(want, ref, shift, count)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("dirAboveRun count=%d shift=%d i=%d got=%d want=%d", count, shift, i, got[i], want[i])
				}
			}
		}
	}
}

func TestDirLeftCol8SIMDMatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15} {
		ref := dirEdge(count+8, 0x9a1+uint32(count))
		for _, stride := range []int{1, 5, 33} {
			for _, shift := range []int{0, 3, 16, 29, 31} {
				got := make([]byte, count*stride+8)
				want := make([]byte, count*stride+8)
				dirLeftCol8SIMD(got, stride, ref, shift, count)
				dirLeftCol8PureGo(want, stride, ref, shift, count)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("dirLeftCol count=%d stride=%d shift=%d i=%d got=%d want=%d", count, stride, shift, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestKernelsSIMDZeroAlloc(t *testing.T) {
	// subsample
	input := make([]uint8, 40*32)
	for i := range input {
		input[i] = byte(i)
	}
	out := make([]uint16, CFLBufSquare)
	ss := func() { subsampleLuma8SIMD(out, input, 40, 32, 32, 16, 16, true, true) }
	if a := testing.AllocsPerRun(1000, ss); a != 0 {
		t.Fatalf("subsampleLuma8SIMD allocated %f/call", a)
	}
	// dirRowInterp / dirAboveRun
	above := dirEdge(72, 0x5)
	dst := make([]byte, 64)
	ri := func() { dirRowInterp8SIMD(dst, above, 0, 11, 64, 64) }
	if a := testing.AllocsPerRun(1000, ri); a != 0 {
		t.Fatalf("dirRowInterp8SIMD allocated %f/call", a)
	}
	ar := func() { dirAboveRun8SIMD(dst, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, ar); a != 0 {
		t.Fatalf("dirAboveRun8SIMD allocated %f/call", a)
	}
	// dirLeftCol
	col := make([]byte, 64*5+8)
	lc := func() { dirLeftCol8SIMD(col, 5, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, lc); a != 0 {
		t.Fatalf("dirLeftCol8SIMD allocated %f/call", a)
	}
}
