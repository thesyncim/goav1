//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"slices"
	"testing"
)

// withSIMDSGR temporarily forces the SGR dispatch slots to the Go-native SIMD
// kernels (box sums + blend), independent of the detection gate, so the SIMD path
// runs regardless of the host. It restores the previous bindings afterwards.
func withSIMDSGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = sgrSIMDBoxsumKernel, selfguidedSIMD, selfguidedFastSIMD
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

// TestSGRSIMDIsBound guards the wiring: when the host detects the SIMD features the
// self-guided dispatch slots resolve to the Go-native SIMD kernels.
func TestSGRSIMDIsBound(t *testing.T) {
	if !sgrSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, c := range []struct {
		name string
		got  any
		want any
	}{
		{"boxsumImpl", boxsumImpl, sgrSIMDBoxsumKernel},
		{"selfguidedImpl", selfguidedImpl, selfguidedSIMD},
		{"selfguidedFastImpl", selfguidedFastImpl, selfguidedFastSIMD},
	} {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Fatalf("%s = %s, want %s", c.name, got, want)
		}
	}
}

// TestSGRBlendRowSIMDMatchesScalar is the direct differential for the nine-tap
// blend: sgrBlendRowSIMD must equal the scalar stencil formula for every weight
// set the driver uses, over random and extreme int32 A/B/dgd values, with
// lane-aligned and ragged column counts.
func TestSGRBlendRowSIMDMatchesScalar(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x51d1)
	weightSets := [][9]int32{
		{3, 4, 3, 4, 4, 4, 3, 4, 3},
		{5, 6, 5, 0, 0, 0, 5, 6, 5},
		{0, 0, 0, 5, 6, 5, 0, 0, 0},
	}
	shifts := []int{SGRProjSgrBits + 5 - SGRProjRstBits, SGRProjSgrBits + 4 - SGRProjRstBits, SGRProjSgrBits + 5 - SGRProjRstBits + 1}
	for iter := 0; iter < 200; iter++ {
		cols := 1 + rnd.pseudoUniform(40)
		n := cols + 4
		fill := func() []int32 {
			s := make([]int32, n+2)
			for i := range s {
				switch rnd.pseudoUniform(8) {
				case 0:
					s[i] = 0x7fffffff >> rnd.pseudoUniform(4)
				case 1:
					s[i] = -0x7fffffff >> rnd.pseudoUniform(4)
				default:
					s[i] = int32(rnd.pseudoUniform(1<<20)) - (1 << 19)
				}
			}
			return s
		}
		aPrev, aCur, aNext := fill(), fill(), fill()
		bPrev, bCur, bNext := fill(), fill(), fill()
		dgd := fill()
		for _, w := range weightSets {
			for _, shift := range shifts {
				lanes := cols &^ (sgrBlendLanes - 1)
				if lanes == 0 {
					continue
				}
				got := make([]int32, lanes)
				sgrBlendRowSIMD(got, dgd, aPrev, aCur, aNext, bPrev, bCur, bNext, &w, shift, lanes)
				for c := 0; c < lanes; c++ {
					a := aPrev[c]*w[0] + aPrev[c+1]*w[1] + aPrev[c+2]*w[2] +
						aCur[c]*w[3] + aCur[c+1]*w[4] + aCur[c+2]*w[5] +
						aNext[c]*w[6] + aNext[c+1]*w[7] + aNext[c+2]*w[8]
					b := bPrev[c]*w[0] + bPrev[c+1]*w[1] + bPrev[c+2]*w[2] +
						bCur[c]*w[3] + bCur[c+1]*w[4] + bCur[c+2]*w[5] +
						bNext[c]*w[6] + bNext[c+1]*w[7] + bNext[c+2]*w[8]
					want := roundPowerOfTwo(a*dgd[c]+b, shift)
					if got[c] != want {
						t.Fatalf("iter=%d w=%v shift=%d col=%d got=%d want=%d", iter, w, shift, c, got[c], want)
					}
				}
			}
		}
	}
}

// TestSGRSIMDMatchesPureGo drives the full ApplySelfguidedRestoration with the SIMD
// kernels forced (box sums + blend) and again with the pure-Go reference, asserting
// the destinations are byte-identical across bit depths, all parameter sets, and a
// mix of lane-aligned and ragged widths.
func TestSGRSIMDMatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xCAFE)
	sizes := []struct{ w, h int }{{4, 4}, {5, 5}, {8, 8}, {13, 11}, {31, 64}, {64, 16}, {64, 64}, {1, 1}, {2, 3}, {7, 7}, {16, 9}, {40, 40}}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, sz := range sizes {
			for eps := range SGRProjParams {
				stride := sz.w + 2*SGRProjBorderHorz + 5
				origin := SGRProjBorderVert*stride + SGRProjBorderHorz
				src := make([]uint16, stride*(sz.h+2*SGRProjBorderVert))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				scratchLen, err := SelfguidedScratchLen(sz.w, sz.h)
				if err != nil {
					t.Fatal(err)
				}
				xqd := [2]int8{int8(rnd.pseudoUniform(96) - 48), int8(rnd.pseudoUniform(96) - 48)}

				gotDst := make([]uint16, sz.w*sz.h)
				scratch := make([]int32, scratchLen)
				withSIMDSGR(func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, scratch); err != nil {
						t.Fatalf("simd bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
					}
				})

				wantDst := make([]uint16, sz.w*sz.h)
				wantScratch := make([]int32, scratchLen)
				withPureGoSGR(func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, wantDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, wantScratch); err != nil {
						t.Fatalf("purego bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
					}
				})

				if !slices.Equal(gotDst, wantDst) {
					for i := range wantDst {
						if gotDst[i] != wantDst[i] {
							t.Fatalf("bd=%d sz=%dx%d eps=%d dst[%d]=%d want %d", bitDepth, sz.w, sz.h, eps, i, gotDst[i], wantDst[i])
						}
					}
				}
			}
		}
	}
}

// TestSGRSIMDIsZeroAlloc protects the hot-path contract that the SIMD self-guided
// drivers do not allocate per call beyond the scratch the caller owns.
func TestSGRSIMDIsZeroAlloc(t *testing.T) {
	const w, h = 64, 64
	const bitDepth uint8 = 12
	max := uint16((1 << bitDepth) - 1)
	stride := w + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(h+2*SGRProjBorderVert))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x9977)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
	dst := make([]uint16, w*h)
	scratchLen, _ := SelfguidedScratchLen(w, h)
	scratch := make([]int32, scratchLen)
	withSIMDSGR(func() {
		if allocs := testing.AllocsPerRun(50, func() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, w, w, h, 15, [2]int8{8, 11}, bitDepth, scratch)
		}); allocs != 0 {
			t.Fatalf("SIMD SGR allocated %f times per call", allocs)
		}
	})
}

func BenchmarkApplySelfguidedRestorationSIMD(b *testing.B) {
	scratchLen, _ := SelfguidedScratchLen(64, 64)
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(64+2*SGRProjBorderVert))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	b.SetBytes(int64(64 * 64 * 2))
	b.ReportAllocs()
	withSIMDSGR(func() {
		for b.Loop() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch)
		}
	})
}

// Direct-call differential for the 8-bit Go-native SIMD self-guided projection,
// independent of the dispatch binding so the SIMD path runs on every host. Widths
// off the vector span must hit the scalar tail and still match.

func TestSGRWeightedRowU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8b3)
	widths := []int{1, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 33, 64}
	xqs := [][2]int32{{0, 128}, {31, 0}, {-96, 256}, {56, 72}, {-128, 0}, {127, 127}}
	for _, width := range widths {
		for _, xq := range xqs {
			src := randomU8Plane(rnd, width, 1)
			f0 := make([]int32, width)
			f1 := make([]int32, width)
			for i := range f0 {
				f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
				f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
			}
			want := make([]uint8, width)
			got := make([]uint8, width)
			sgrWeightedRowU8(want, src, f0, f1, xq[0], xq[1])
			sgrWeightedRowU8SIMD(got, src, f0, f1, xq[0], xq[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
				}
			}
		}
	}
}

// TestSGRWeightedRowU8SIMDExtremes hardens the projection at the int16 wrap and
// the [0,255] clamp boundaries: every sample value against extreme flt rows.
func TestSGRWeightedRowU8SIMDExtremes(t *testing.T) {
	const width = 48
	flts := []int32{-(1 << 20), -1, 0, 1, 1 << 20, 1<<21 - 1}
	xqs := [][2]int32{{0, 128}, {-96, 256}, {127, 127}, {-128, 0}}
	for s := 0; s < 256; s++ {
		src := make([]uint8, width)
		for i := range src {
			src[i] = uint8(s)
		}
		for _, f := range flts {
			f0 := make([]int32, width)
			f1 := make([]int32, width)
			for i := range f0 {
				f0[i] = f
				f1[i] = -f
			}
			for _, xq := range xqs {
				want := make([]uint8, width)
				got := make([]uint8, width)
				sgrWeightedRowU8(want, src, f0, f1, xq[0], xq[1])
				sgrWeightedRowU8SIMD(got, src, f0, f1, xq[0], xq[1])
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("s=%d f=%d xq=%v dst[%d]=%d want %d", s, f, xq, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestSGRWeightedRowU8SIMDIsZeroAlloc protects the hot-path contract that the
// 8-bit SIMD projection does not allocate per call.
func TestSGRWeightedRowU8SIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a4)
	src := randomU8Plane(rnd, 64, 1)
	dst := make([]uint8, 64)
	f0 := make([]int32, 64)
	f1 := make([]int32, 64)
	if allocs := testing.AllocsPerRun(200, func() {
		sgrWeightedRowU8SIMD(dst, src, f0, f1, 12, 116)
	}); allocs != 0 {
		t.Fatalf("u8 SIMD projection allocated %f times per call", allocs)
	}
}

func TestSGRWeightedRowU8SIMDIsBound(t *testing.T) {
	if !sgrSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(sgrWeightedRowU8Impl), nameOf(sgrWeightedRowU8SIMD); got != want {
		t.Fatalf("sgrWeightedRowU8Impl = %s, want %s", got, want)
	}
}

func TestWienerDispatchUsesSIMD(t *testing.T) {
	if !wienerSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, slot := range []struct {
		name string
		got  any
		want any
	}{
		{"wienerHorizontalTrustedImpl", wienerHorizontalTrustedImpl, wienerHorizontalSIMDTrusted},
		{"wienerVerticalImpl", wienerVerticalImpl, wienerVerticalSIMD},
		{"wienerHorizontalU8Impl", wienerHorizontalU8Impl, wantWienerHorizontalU8Impl},
		{"wienerVerticalU8Impl", wienerVerticalU8Impl, wantWienerVerticalU8Impl},
	} {
		if got, want := nameOf(slot.got), nameOf(slot.want); got != want {
			t.Errorf("%s = %s, want %s", slot.name, got, want)
		}
	}
}

// TestWienerSIMDIsBound guards the wiring: under the goexperiment.simd build
// every Wiener dispatch slot must resolve to its Go-native SIMD kernel.
func TestWienerSIMDIsBound(t *testing.T) {
	if !wienerSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, c := range []struct {
		name string
		got  any
		want any
	}{
		{"wienerVerticalImpl", wienerVerticalImpl, wienerVerticalSIMD},
		{"wienerHorizontalTrustedImpl", wienerHorizontalTrustedImpl, wienerHorizontalSIMDTrusted},
	} {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Fatalf("%s = %s, want %s", c.name, got, want)
		}
	}
}

// TestWienerVerticalSIMDMatchesPureGo is the byte-exactness differential test:
// the Go-native SIMD vertical kernel must equal the scalar wienerVertical
// sample-for-sample over random temp buffers, filters, bit depths, widths and
// heights. Widths cover 8-aligned (SIMD) and non-8-aligned (scalar fallback)
// shapes; temp values span the valid [0,maxClamp] range produced by the
// horizontal pass.
func TestWienerVerticalSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5115))
	sizes := []struct{ width, height int }{
		{8, 8}, {16, 12}, {64, 64}, {64, 31}, {24, 7},
		{13, 9}, {1, 1}, {40, 64}, {8, 1}, {56, 3}, {72, 5},
	}
	for iter := 0; iter < 400; iter++ {
		bitDepth := []int{8, 10, 12}[rng.Intn(3)]
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		// maxClamp is the horizontal pass's output ceiling; the vertical pass
		// only ever sees temp samples in [0,maxClamp].
		maxClamp := int(1<<(bitDepth+1+WienerFilterBits-round0)) - 1

		var filter WienerFilter
		switch rng.Intn(2) {
		case 0:
			// A random valid Wiener filter (taps sum to zero, mirrored).
			t0 := int16(WienerTap0Min + rng.Intn(WienerTap0Max-WienerTap0Min+1))
			t1 := int16(WienerTap1Min + rng.Intn(WienerTap1Max-WienerTap1Min+1))
			t2 := int16(WienerTap2Min + rng.Intn(WienerTap2Max-WienerTap2Min+1))
			filter = NewWienerFilter(t0, t1, t2)
		default:
			// A general 7-tap filter over the int16 tap window the reference
			// accepts, exercising the kernel beyond the canonical Wiener set.
			for i := 0; i < 7; i++ {
				filter[i] = int16(-64 + rng.Intn(129))
			}
			filter[7] = 0
		}

		sz := sizes[rng.Intn(len(sizes))]
		tempStride := sz.width
		temp := make([]uint16, tempStride*(sz.height+2*WienerHalfwin))
		for i := range temp {
			temp[i] = uint16(rng.Intn(maxClamp + 1))
		}
		dstStride := sz.width + rng.Intn(5)
		want := make([]uint16, dstStride*sz.height)
		got := make([]uint16, dstStride*sz.height)

		wienerVertical(temp, tempStride, want, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
		wienerVerticalSIMD(temp, tempStride, got, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)

		for r := 0; r < sz.height; r++ {
			for c := 0; c < sz.width; c++ {
				i := r*dstStride + c
				if want[i] != got[i] {
					t.Fatalf("iter=%d bd=%d sz=%dx%d filter=%v pos=(%d,%d) got=%d want=%d",
						iter, bitDepth, sz.width, sz.height, filter, r, c, got[i], want[i])
				}
			}
		}
	}
}

// TestWienerVerticalSIMDIsZeroAlloc protects the hot-path zero-allocation
// contract for the SIMD vertical kernel.
func TestWienerVerticalSIMDIsZeroAlloc(t *testing.T) {
	const width, height = 64, 64
	const bitDepth = 12
	max := uint16((1 << bitDepth) - 1)
	_, round1 := wienerRounds(bitDepth)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	rng := rand.New(rand.NewSource(0x2727))
	for i := range temp {
		temp[i] = uint16(rng.Intn(int(max) + 1))
	}
	dst := make([]uint16, width*height)
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerVerticalSIMD(temp, width, dst, width, width, height, info.VFilter, bitDepth, round1, max)
	}); allocs != 0 {
		t.Fatalf("wienerVerticalSIMD allocated %f times per call", allocs)
	}
}

func benchWienerVertical(b *testing.B, fn func([]uint16, int, []uint16, int, int, int, WienerFilter, int, int, uint16), width int) {
	const height = 64
	const bitDepth = 12
	max := uint16((1 << bitDepth) - 1)
	_, round1 := wienerRounds(bitDepth)
	rng := rand.New(rand.NewSource(0x6363))
	maxClamp := int(1<<(bitDepth+1+WienerFilterBits-5)) - 1
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16(rng.Intn(maxClamp + 1))
	}
	dst := make([]uint16, width*height)
	filter := DefaultWienerInfo().VFilter
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(temp, width, dst, width, width, height, filter, bitDepth, round1, max)
	}
}

func BenchmarkWienerVertical8_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 8) }
func BenchmarkWienerVertical8_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 8) }

func BenchmarkWienerVertical64_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 64) }
func BenchmarkWienerVertical64_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 64) }

// TestWienerHorizontalU8SIMDIsBound guards the wiring: under the
// goexperiment.simd build the u8 horizontal dispatch slot must resolve to the
// Go-native SIMD kernel (the scalar binding in wiener_dispatch_generic.go is excluded
// there).
func TestWienerHorizontalU8SIMDIsBound(t *testing.T) {
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got := nameOf(wienerHorizontalU8Impl); got != nameOf(wantWienerHorizontalU8Impl) {
		t.Fatalf("wienerHorizontalU8Impl = %s, want %s", got, nameOf(wantWienerHorizontalU8Impl))
	}
}

// TestWienerHorizontalU8SIMDMatchesReference is the byte-exactness
// differential: the Go-SIMD horizontal kernel must equal the scalar
// wienerHorizontalU8 reference sample-for-sample
// across the size corpus (vector widths, %16==8 tails, non-multiple-of-8
// fallback widths) and the full dispatch filter set, over random planes plus
// extreme pixel patterns (all-0, all-255, alternating 0/255) that maximize the
// symmetric tap-pair sums and the clamp edges.
func TestWienerHorizontalU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c1)
	round0, _ := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {62, 12}, {17, 4}, {31, 7}, {7, 5}, {48, 3}}, u8DiffSizes...)
	fill := func(src []uint8, mode int) {
		switch mode {
		case 0:
			for i := range src {
				src[i] = uint8(rnd.pseudoUniform(256))
			}
		case 1:
			for i := range src {
				src[i] = 255
			}
		case 2:
			for i := range src {
				src[i] = 0
			}
		default:
			for i := range src {
				src[i] = uint8(255 * (i & 1))
			}
		}
	}
	for mode := 0; mode < 4; mode++ {
		for _, sz := range sizes {
			for fi, info := range wienerDispatchFilters() {
				stride := sz.width + 2*WienerHalfwin + 3
				origin := WienerHalfwin*stride + WienerHalfwin + 1
				src := make([]uint8, stride*(sz.height+2*WienerHalfwin))
				fill(src, mode)
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				want := make([]uint16, tempLen)
				gotSIMD := make([]uint16, tempLen)
				wienerHorizontalU8(src, stride, origin, sz.width, sz.height, info.HFilter, round0, want)
				wienerHorizontalU8SIMD(src, stride, origin, sz.width, sz.height, info.HFilter, round0, gotSIMD)
				for i := range want {
					if gotSIMD[i] != want[i] {
						t.Fatalf("SIMD: mode=%d sz=%dx%d f=%d temp[%d]=%d want %d",
							mode, sz.width, sz.height, fi, i, gotSIMD[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerHorizontalU8SIMDIsZeroAlloc protects the hot-path zero-allocation
// contract for the SIMD horizontal kernel.
func TestWienerHorizontalU8SIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c2)
	round0, _ := wienerRounds(8)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	temp := make([]uint16, 64*(64+2*WienerHalfwin))
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerHorizontalU8SIMDChecked(src, stride, origin, 64, 64, info.HFilter, round0, temp)
	}); allocs != 0 {
		t.Fatalf("wienerHorizontalU8SIMDChecked allocated %f times per call", allocs)
	}
}

// TestWienerHorizontalSIMDExtremeSamples hardens the uint16 horizontal kernel at
// the value and clamp boundaries: all-zero, all-max, alternating 0/max, and a
// single max spike, for every bit depth and the symmetric filter corpus. The
// symmetric pair sums and the [0,maxClamp] clamp are exercised at their limits.
func TestWienerHorizontalSIMDExtremeSamples(t *testing.T) {
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, _ := wienerRounds(bitDepth)
		for _, sz := range []struct{ width, height int }{{8, 1}, {8, 8}, {16, 5}, {24, 9}, {64, 4}} {
			stride := sz.width + 2*WienerHalfwin + 2
			origin := WienerHalfwin*stride + WienerHalfwin
			rows := sz.height + 2*WienerHalfwin
			for mode := 0; mode < 4; mode++ {
				src := make([]uint16, stride*rows+2)
				for i := range src {
					switch mode {
					case 0:
						src[i] = 0
					case 1:
						src[i] = max
					case 2:
						src[i] = max * uint16(i&1)
					default:
						src[i] = 0
						if i == len(src)/2 {
							src[i] = max
						}
					}
				}
				for fi, info := range wienerDispatchFilters() {
					want := make([]uint16, sz.width*rows)
					got := make([]uint16, sz.width*rows)
					if !wienerHorizontal(src, stride, origin, sz.width, sz.height, info.HFilter, bitDepth, round0, max, want) {
						t.Fatalf("scalar rejected in-range extremes bd=%d mode=%d", bitDepth, mode)
					}
					wienerHorizontalSIMDTrusted(src, stride, origin, sz.width, sz.height, info.HFilter, bitDepth, round0, max, got)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d mode=%d sz=%dx%d f=%d temp[%d]=%d want %d",
								bitDepth, mode, sz.width, sz.height, fi, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestWienerHorizontalU8SIMDLoadBounds(t *testing.T) {
	const width, height = 16, 5
	round0, _ := wienerRounds(8)
	filter := DefaultWienerInfo().HFilter
	for _, fixture := range []struct {
		name         string
		rowPad       int
		canVectorize bool
	}{
		{"minimal-border", 2 * WienerHalfwin, false},
		{"vector-border", 2*WienerHalfwin + 2, true},
	} {
		stride := width + fixture.rowPad
		origin := WienerHalfwin*stride + WienerHalfwin
		src := make([]uint8, stride*(height+2*WienerHalfwin))
		for i := range src {
			src[i] = uint8((i*37 + i/stride*19) & 0xff)
		}
		if got := wienerHorizontalU8SIMDCanRun(len(src), stride, origin, width, height); got != fixture.canVectorize {
			t.Fatalf("%s canVectorize=%v want %v", fixture.name, got, fixture.canVectorize)
		}
		want := make([]uint16, width*(height+2*WienerHalfwin))
		got := make([]uint16, len(want))
		wienerHorizontalU8(src, stride, origin, width, height, filter, round0, want)
		wienerHorizontalU8SIMDChecked(src, stride, origin, width, height, filter, round0, got)
		if !slices.Equal(got, want) {
			t.Fatalf("%s: SIMD wrapper differs from scalar fallback", fixture.name)
		}
	}
}

func benchWienerHorizontalU8SIMD(b *testing.B, fn func([]uint8, int, int, int, int, WienerFilter, int, []uint16), width int) {
	const height = 64
	round0, _ := wienerRounds(8)
	stride := width + 2*WienerHalfwin + 2
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c3)
	src := make([]uint8, stride*(height+2*WienerHalfwin))
	for i := range src {
		src[i] = uint8(rnd.pseudoUniform(256))
	}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	filter := DefaultWienerInfo().HFilter
	if !wienerHorizontalU8SIMDCanRun(len(src), stride, origin, width, height) {
		b.Fatal("benchmark fixture does not enable the SIMD vector kernel")
	}
	b.ReportAllocs()
	for b.Loop() {
		fn(src, stride, origin, width, height, filter, round0, temp)
	}
}

func BenchmarkWienerHorizontalU8_SIMD(b *testing.B) {
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			benchWienerHorizontalU8SIMD(b, wienerHorizontalU8SIMDChecked, width)
		})
	}
}

func BenchmarkWienerHorizontalU8_PureGo(b *testing.B) {
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			benchWienerHorizontalU8SIMD(b, wienerHorizontalU8, width)
		})
	}
}

// These tests are the byte-exactness gate for the Wiener symmetric-pair MAC
// reduction (see wiener_neon_arm64.s / wiener_u8_neon_arm64.s). They sweep
// randomized valid Wiener coefficient sets -- every filter NewWienerFilter can
// produce, i.e. every mirrored tap triple in [Min,Max] -- against randomized
// samples, and require the NEON kernels to match the pure-Go reference sample
// for sample. The pairing (f0==f6, f1==f5, f2==f4) is only bit-identical for
// symmetric filters, so this exercises exactly the property the rewrite relies
// on across the full coefficient space, not just the fixed dispatch corpus.

// randomWienerFilter draws a random valid Wiener filter: each of the three
// distinct taps is uniform over its [Min,Max] range and NewWienerFilter mirrors
// them and derives the center so the taps sum to zero (validWienerFilter holds).
func randomWienerFilter(rnd *restorationRandom) WienerFilter {
	t0 := int16(WienerTap0Min + rnd.pseudoUniform(WienerTap0Max-WienerTap0Min+1))
	t1 := int16(WienerTap1Min + rnd.pseudoUniform(WienerTap1Max-WienerTap1Min+1))
	t2 := int16(WienerTap2Min + rnd.pseudoUniform(WienerTap2Max-WienerTap2Min+1))
	return NewWienerFilter(t0, t1, t2)
}

var wienerSymmetrySizes = []struct{ width, height int }{
	{8, 8}, {16, 5}, {24, 9}, {40, 12}, {64, 64},
}

// TestWienerHorizontalSIMDSymmetryRandomTaps checks the u16 horizontal
// symmetric-pair kernel against the reference over random valid taps and random
// samples, at every bit depth (the horizontal pair-sums stay in the int16 lane
// for all of 8/10/12-bit since source samples are <=4095).
func TestWienerHorizontalSIMDSymmetryRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5117)
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, _ := wienerRounds(bitDepth)
		for _, sz := range wienerSymmetrySizes {
			for trial := 0; trial < 24; trial++ {
				filter := randomWienerFilter(rnd)
				stride := sz.width + 2*WienerHalfwin + 4
				origin := WienerHalfwin*stride + WienerHalfwin + 2
				src := make([]uint16, stride*(sz.height+2*WienerHalfwin))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				want := make([]uint16, tempLen)
				got := make([]uint16, tempLen)
				wienerHorizontal(src, stride, origin, sz.width, sz.height, filter, bitDepth, round0, max, want)
				wienerHorizontalSIMDTrusted(src, stride, origin, sz.width, sz.height, filter, bitDepth, round0, max, got)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bd=%d sz=%dx%d filter=%v temp[%d]=%d want %d", bitDepth, sz.width, sz.height, filter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerVerticalSIMDRandomTaps checks the general u16 vertical kernel (which
// deliberately keeps the plain 7-tap MAC) against the reference over random
// valid taps and random temp values spanning the full [0,maxClamp] intermediate
// range, at every bit depth.
func TestWienerVerticalSIMDRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5118)
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
		for _, sz := range wienerSymmetrySizes {
			for trial := 0; trial < 24; trial++ {
				filter := randomWienerFilter(rnd)
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				temp := make([]uint16, tempLen)
				for i := range temp {
					temp[i] = uint16(rnd.pseudoUniform(int(limit))) // [0, maxClamp]
				}
				dstStride := sz.width + 3
				want := make([]uint16, dstStride*sz.height)
				got := make([]uint16, dstStride*sz.height)
				wienerVertical(temp, sz.width, want, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
				wienerVerticalSIMD(temp, sz.width, got, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bd=%d sz=%dx%d filter=%v dst[%d]=%d want %d", bitDepth, sz.width, sz.height, filter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerU8SIMDSymmetryRandomTaps checks the 8-bit symmetric-pair kernels
// (both horizontal and vertical) against the pure-Go u8 references over random
// valid taps. The vertical pair-sums are safe because temp is clamped to
// [0,8191] at 8-bit, so a pair-sum (<=16382) stays in the int16 lane.
func TestWienerU8SIMDSymmetryRandomTaps(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5119)
	round0, round1 := wienerRounds(8)
	for _, sz := range wienerSymmetrySizes {
		for trial := 0; trial < 24; trial++ {
			hFilter := randomWienerFilter(rnd)
			vFilter := randomWienerFilter(rnd)
			stride := sz.width + 2*WienerHalfwin + 3
			origin := WienerHalfwin*stride + WienerHalfwin + 1
			src := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)

			// Horizontal: NEON vs reference.
			wantTemp := make([]uint16, tempLen)
			gotTemp := make([]uint16, tempLen)
			wienerHorizontalU8(src, stride, origin, sz.width, sz.height, hFilter, round0, wantTemp)
			wienerHorizontalU8SIMDChecked(src, stride, origin, sz.width, sz.height, hFilter, round0, gotTemp)
			for i := range wantTemp {
				if gotTemp[i] != wantTemp[i] {
					t.Fatalf("horiz sz=%dx%d filter=%v temp[%d]=%d want %d", sz.width, sz.height, hFilter, i, gotTemp[i], wantTemp[i])
				}
			}

			// Vertical over the shared (reference) temp buffer.
			dstStride := sz.width + 2
			wantDst := make([]uint8, dstStride*sz.height)
			gotDst := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(wantTemp, sz.width, wantDst, dstStride, sz.width, sz.height, vFilter, round1)
			wienerVerticalU8SIMD(wantTemp, sz.width, gotDst, dstStride, sz.width, sz.height, vFilter, round1)
			for i := range wantDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("vert sz=%dx%d filter=%v dst[%d]=%d want %d", sz.width, sz.height, vFilter, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

func TestWienerVerticalU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5c3)
	_, round1 := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {62, 12}, {17, 4}, {31, 7}, {7, 5}}, u8DiffSizes...)
	for _, sz := range sizes {
		for fi, info := range wienerDispatchFilters() {
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			temp := make([]uint16, tempLen)
			for i := range temp {
				temp[i] = uint16(rnd.pseudoUniform(8192))
			}
			dstStride := sz.width + 2
			want := make([]uint8, dstStride*sz.height)
			got := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(temp, sz.width, want, dstStride, sz.width, sz.height, info.VFilter, round1)
			wienerVerticalU8SIMD(temp, sz.width, got, dstStride, sz.width, sz.height, info.VFilter, round1)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
		}
	}

	// The maximum horizontal-pass sample makes each symmetric pair sum to
	// 16382, still a positive int16 before the widening multiply.
	const width, height = 16, 8
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = 8191
	}
	filter := wienerDispatchFilters()[0].VFilter
	want := make([]uint8, width*height)
	got := make([]uint8, width*height)
	wienerVerticalU8(temp, width, want, width, width, height, filter, round1)
	wienerVerticalU8SIMD(temp, width, got, width, width, height, filter, round1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("max pair sum dst[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func TestWienerVerticalU8SIMDAsymmetricFilterFallback(t *testing.T) {
	const width, height = 8, 5
	_, round1 := wienerRounds(8)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5c4)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16(rnd.pseudoUniform(8192))
	}
	filter := WienerFilter{17, -21, 8, 112, 4, -21, 19}
	want := make([]uint8, width*height)
	got := make([]uint8, width*height)
	wienerVerticalU8(temp, width, want, width, width, height, filter, round1)
	wienerVerticalU8SIMD(temp, width, got, width, width, height, filter, round1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("asymmetric filter dst[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

var wienerU8VerticalBenchmarkSink uint8

func benchWienerVerticalU8(b *testing.B, fn func([]uint16, int, []uint8, int, int, int, WienerFilter, int), width int) {
	const height = 64
	_, round1 := wienerRounds(8)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16((i*97 + 31) % 8192)
	}
	dst := make([]uint8, width*height)
	info := wienerDispatchFilters()[0]
	b.ReportAllocs()
	b.SetBytes(int64(width * height))
	b.ResetTimer()
	for b.Loop() {
		fn(temp, width, dst, width, width, height, info.VFilter, round1)
	}
	b.StopTimer()
	wienerU8VerticalBenchmarkSink = dst[len(dst)-1]
}

func BenchmarkWienerVerticalU8_8_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 8)
}
func BenchmarkWienerVerticalU8_8_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 8)
}

func BenchmarkWienerVerticalU8_64_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 64)
}
func BenchmarkWienerVerticalU8_64_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 64)
}
