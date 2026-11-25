package restoration

import (
	"errors"
	"slices"
	"testing"
	"unsafe"
)

// TestRestorationScratchScalesWithUnitDimensions confirms that the scratch
// length helpers grow with the per-unit width/height. This is the contract
// frame-level callers (including SVC enhancement layers running at their own
// resolution) rely on when partitioning bordered uint16 buffers into
// processing-unit slices.
func TestRestorationScratchScalesWithUnitDimensions(t *testing.T) {
	for _, tc := range []struct {
		name   string
		width  int
		height int
	}{
		{"min", 1, 1},
		{"tail-h", 16, 64},
		{"tail-w", 64, 16},
		{"max-unit", ProcUnitSize, ProcUnitSize},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotW, err := WienerScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatalf("WienerScratchLen: %v", err)
			}
			wantW := tc.width * (tc.height + 2*WienerHalfwin)
			if gotW != wantW {
				t.Fatalf("WienerScratchLen=%d want %d", gotW, wantW)
			}
			gotS, err := SelfguidedScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatalf("SelfguidedScratchLen: %v", err)
			}
			dgdStride := tc.width + 2*SGRProjBorderHorz
			dgdLen := dgdStride * (tc.height + 2*SGRProjBorderVert)
			fltLen := tc.width * tc.height
			bufStride := sgrBufferStride(tc.width)
			bufLen := bufStride * (tc.height + 2*SGRProjBorderVert)
			wantS := dgdLen + 2*fltLen + 2*bufLen
			if gotS != wantS {
				t.Fatalf("SelfguidedScratchLen=%d want %d", gotS, wantS)
			}
		})
	}
	// Per-unit invariant: anything larger than the AV1 processing unit must be
	// rejected by both entry points. The frame-level scheduler must therefore
	// tile each plane into <=64x64 units before invoking restoration, no matter
	// how large the enhancement-layer frame is.
	if _, err := WienerScratchLen(ProcUnitSize+1, 1); !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("WienerScratchLen oversized=%v want %v", err, ErrInvalidRestoration)
	}
	if _, err := SelfguidedScratchLen(1, ProcUnitSize+1); !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("SelfguidedScratchLen oversized=%v want %v", err, ErrInvalidRestoration)
	}
}

// TestApplyRestorationCovers1280x720Frame walks Wiener and self-guided
// restoration over every processing unit that tiles a 1280x720 luma plane,
// which is a representative SVC enhancement-layer resolution. The 1280x720
// plane is decomposed into 64x64 processing units plus a 16-pixel-tall tail
// row (720 = 11*64 + 16). For each tile we feed a bordered slice of the
// caller-owned frame buffer to ApplyWienerRestorationTrusted / ApplySelfguidedRestoration
// and confirm that:
//   - Every unit invocation succeeds for both filters.
//   - Output samples remain inside the bit-depth range.
//   - A single shared scratch buffer sized for the worst-case unit can be
//     reused across every tile (i.e. scratch reuse is unit-bounded, not
//     frame-bounded), which is the invariant the frame-level pipeline relies
//     on when running at enhancement-layer resolution.
func TestApplyRestorationCovers1280x720Frame(t *testing.T) {
	const (
		frameWidth  = 1280
		frameHeight = 720
		bitDepth    = uint8(8)
	)
	max := uint16((1 << bitDepth) - 1)

	// Allocate one logical "frame" that has enough border for both filters.
	// SGR needs the larger border of the two.
	border := SGRProjBorderHorz
	if WienerHalfwin > border {
		border = WienerHalfwin
	}
	stride := frameWidth + 2*border
	bordered := stride * (frameHeight + 2*border)
	src := make([]uint16, bordered)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xc720)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}

	frameOrigin := border*stride + border

	// Worst-case scratch is the maximum unit size (ProcUnitSize x ProcUnitSize).
	wienerScratch, err := WienerScratchLen(ProcUnitSize, ProcUnitSize)
	if err != nil {
		t.Fatal(err)
	}
	sgrScratch, err := SelfguidedScratchLen(ProcUnitSize, ProcUnitSize)
	if err != nil {
		t.Fatal(err)
	}
	wienerBuf := make([]uint16, wienerScratch)
	sgrBuf := make([]int32, sgrScratch)
	wienerDst := make([]uint16, ProcUnitSize*ProcUnitSize)
	sgrDst := make([]uint16, ProcUnitSize*ProcUnitSize)

	wienerInfo := DefaultWienerInfo()

	tiles := 0
	for top := 0; top < frameHeight; top += ProcUnitSize {
		h := ProcUnitSize
		if top+h > frameHeight {
			h = frameHeight - top
		}
		for left := 0; left < frameWidth; left += ProcUnitSize {
			w := ProcUnitSize
			if left+w > frameWidth {
				w = frameWidth - left
			}
			tileOrigin := frameOrigin + top*stride + left
			dstW := w
			dstH := h

			// Wiener path.
			needW, err := WienerScratchLen(w, h)
			if err != nil {
				t.Fatalf("WienerScratchLen tile=%dx%d: %v", w, h, err)
			}
			if needW > wienerScratch {
				t.Fatalf("wiener scratch grew with tile: need=%d cap=%d", needW, wienerScratch)
			}
			if err := ApplyWienerRestorationTrusted(src, stride, tileOrigin, wienerDst, dstW, w, h, wienerInfo, bitDepth, wienerBuf[:needW]); err != nil {
				t.Fatalf("ApplyWienerRestoration tile=(%d,%d) %dx%d: %v", left, top, w, h, err)
			}
			for i := 0; i < dstW*dstH; i++ {
				if wienerDst[i] > max {
					t.Fatalf("wiener tile=(%d,%d) dst[%d]=%d > max=%d", left, top, i, wienerDst[i], max)
				}
			}

			// Self-guided path.
			needS, err := SelfguidedScratchLen(w, h)
			if err != nil {
				t.Fatalf("SelfguidedScratchLen tile=%dx%d: %v", w, h, err)
			}
			if needS > sgrScratch {
				t.Fatalf("sgr scratch grew with tile: need=%d cap=%d", needS, sgrScratch)
			}
			if err := ApplySelfguidedRestoration(src, stride, tileOrigin, sgrDst, dstW, w, h, 0, [2]int8{20, -7}, bitDepth, sgrBuf[:needS]); err != nil {
				t.Fatalf("ApplySelfguidedRestoration tile=(%d,%d) %dx%d: %v", left, top, w, h, err)
			}
			for i := 0; i < dstW*dstH; i++ {
				if sgrDst[i] > max {
					t.Fatalf("sgr tile=(%d,%d) dst[%d]=%d > max=%d", left, top, i, sgrDst[i], max)
				}
			}
			tiles++
		}
	}
	// 1280 = 20*64 (no width tail). 720 = 11*64 + 16, so 12 rows of tiles.
	if want := 20 * 12; tiles != want {
		t.Fatalf("tile count=%d want %d", tiles, want)
	}
}

// TestSGRq63HighBitDepthConstantRegionPreservesInput pins the 10-bit
// loop-restoration SGRProj projection for a constant region under the
// extreme xqd / params seen on the libaom q63 first-frame vector. Before
// the fix in this directory ApplySelfguidedRestoration computed an extra
// 1<<(bit_depth-8) factor inside B[k] (it read the scaled local `b`
// instead of the raw bBuf[k] entry), so a flat 10-bit region collapsed to
// ~10/256 of the input value. The libaom reference (av1/common/restoration.c
// calculate_intermediate_result) uses the raw B[k] for the post-update.
//
// The test parameters mirror exactly what the q63 luma SGRProj first
// stripe reports: eps=9, xqd=[31,-32]. For a constant-input stripe (any
// input pixel value in range) the SGR self-guided smoothed image must
// satisfy flt == input<<RST_BITS so the projection is the identity. The
// expectation is that dst[i] equals the input pixel exactly for every
// sample.
func TestSGRq63HighBitDepthConstantRegionPreservesInput(t *testing.T) {
	const bitDepth = uint8(10)
	const width = 64
	const height = 8

	// Two representative 10-bit pixel values; the bug manifested at any
	// value once the bit-depth was >8, but we use both a "dim" value
	// (within the 8-bit numeric range) and a 4x-scaled "high-bit-depth"
	// value to lock in the fix across the q63 flat-region triage points.
	for _, value := range []uint16{154, 616} {
		t.Run("value="+itoa(int(value)), func(t *testing.T) {
			stride := width + 2*SGRProjBorderHorz
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			src := make([]uint16, stride*(height+2*SGRProjBorderVert))
			for i := range src {
				src[i] = value
			}
			dst := make([]uint16, width*height)
			scratchLen, err := SelfguidedScratchLen(width, height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]int32, scratchLen)
			if err := ApplySelfguidedRestoration(src, stride, origin, dst, width, width, height, 9, [2]int8{31, -32}, bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			for i, got := range dst {
				if got != value {
					t.Fatalf("dst[%d]=%d want %d (constant-region identity broke for bitDepth=%d xqd=[31,-32] eps=9)", i, got, value, bitDepth)
				}
			}
		})
	}
}

// itoa avoids fmt.Sprintf in a hot test path. It accepts the small
// non-negative integers used by the table-driven subtest above.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// TestBoxsumSeparableMatchesBoxsum asserts the libaom-style separable running
// sum produces byte-identical A/B buffers to the brute-force reference across
// radii, sizes (including the exact decoder extents width+6/height+6 and the
// degenerate fallback sizes), and both the plain and squared variants.
func TestBoxsumSeparableMatchesBoxsum(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5EFA)
	// Decoder always drives widthExt = unit + 6, heightExt = unit + 6, unit in
	// [1,64]; include those plus tiny sizes that trip the brute-force fallback.
	sizes := []struct{ w, h int }{
		{7, 7}, {8, 8}, {10, 12}, {13, 9}, {23, 31}, {70, 70}, {70, 1 + 6}, {1 + 6, 70},
		{7, 8}, {9, 7}, {5, 5}, {4, 4}, {3, 3}, {6, 6}, {2, 9}, {9, 2},
	}
	for _, r := range []int{1, 2} {
		for _, sz := range sizes {
			width := sz.w
			height := sz.h
			stride := sgrBufferStride(width)
			srcStride := width + 5 // exercise srcStride != width and a nonzero origin
			srcOrigin := 2*srcStride + 3
			src := make([]int32, srcOrigin+srcStride*height+width+8)
			for i := range src {
				src[i] = int32(rnd.pseudoUniform(4096))
			}
			for _, squared := range []bool{false, true} {
				want := make([]int32, stride*height)
				got := make([]int32, stride*height)
				boxsum(src, srcOrigin, width, height, srcStride, r, squared, want, stride)
				boxsumSeparable(src, srcOrigin, width, height, srcStride, r, squared, got, stride)
				if !slices.Equal(want, got) {
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("r=%d sz=%dx%d squared=%v idx=%d got=%d want=%d",
								r, sz.w, sz.h, squared, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestSGRApplySeparableMatchesReference drives the full
// ApplySelfguidedRestoration with the separable box sum bound and compares the
// destination against the pure-Go brute-force reference across bit depths, all
// 16 parameter sets, and a mix of sizes.
func TestSGRApplySeparableMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x0B0C)
	sizes := []struct{ w, h int }{{1, 1}, {4, 4}, {5, 5}, {8, 8}, {13, 11}, {31, 64}, {64, 16}, {64, 64}, {2, 3}, {7, 7}}
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
				gotScratch := make([]int32, scratchLen)
				withBoxsum(boxsumSeparable, func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, gotScratch); err != nil {
						t.Fatalf("separable bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
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

// TestBoxsumSeparableIsZeroAlloc protects the hot-path contract that the
// separable box sum allocates nothing per call.
func TestBoxsumSeparableIsZeroAlloc(t *testing.T) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	allocs := testing.AllocsPerRun(50, func() {
		boxsumSeparable(src, so, w, h, ss, 2, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 2, false, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, false, dst, ds)
	})
	if allocs != 0 {
		t.Fatalf("boxsumSeparable allocated %f times per call", allocs)
	}
}

// withBoxsum temporarily binds the box-sum dispatch slot to fn's kernel while
// keeping the blend kernels as-is, runs body, then restores the binding.
func withBoxsum(kernel func([]int32, int, int, int, int, int, bool, []int32, int), body func()) {
	prev := boxsumImpl
	boxsumImpl = kernel
	defer func() { boxsumImpl = prev }()
	body()
}

// benchBoxsumInput builds a representative extended-block source for a 64x64 SGR
// unit (widthExt=heightExt=70) matching the decoder's calculateIntermediate call.
func benchBoxsumInput() (src []int32, srcOrigin, width, height, srcStride int, dst []int32, dstStride int) {
	width = 64 + 2*SGRProjBorderHorz
	height = 64 + 2*SGRProjBorderVert
	srcStride = width
	src = make([]int32, srcStride*height)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = int32(rnd.pseudoUniform(4096))
	}
	dstStride = sgrBufferStride(64)
	dst = make([]int32, dstStride*height)
	return src, 0, width, height, srcStride, dst, dstStride
}

func BenchmarkBoxsumScalar(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsum(src, so, w, h, ss, 2, true, dst, ds)
		boxsum(src, so, w, h, ss, 2, false, dst, ds)
	}
}

func BenchmarkBoxsumSeparable(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumSeparable(src, so, w, h, ss, 2, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 2, false, dst, ds)
	}
}

func BenchmarkBoxsumSeparableR1(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumSeparable(src, so, w, h, ss, 1, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, false, dst, ds)
	}
}

// withPureGoSGR temporarily forces the SGR dispatch slots to the pure-Go
// reference, runs fn, then restores the previous bindings.
func withPureGoSGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsum, selfguided, selfguidedFast
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

func TestSGRRollingIntermediateMatchesFullOracle(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x920)
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {3, 2}, {7, 9}, {16, 8}, {31, 33}, {63, 64}, {64, 63}} {
		w, h := sz[0], sz[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, bd := range []int{8, 10, 12} {
			for i := range dgd {
				dgd[i] = int32(rnd.pseudoUniform(1 << bd))
			}
			for pi, p := range SGRParameterTable {
				for ri, r := range p.Radius {
					if r == 0 {
						continue
					}
					for pass := 0; pass < 2; pass++ {
						wantA, wantB := make([]int32, bl), make([]int32, bl)
						gotA, gotB := make([]int32, bl+7), make([]int32, bl+7)
						for i := range gotA {
							gotA[i] = int32(uint32(i)*2654435761 + 0xfedc)
							gotB[i] = ^gotA[i]
						}
						tailA, tailB := slices.Clone(gotA[bl:]), slices.Clone(gotB[bl:])
						frozenSGRIntermediate(dgd, origin, w, h, stride, bd, pi, ri, pass, wantA, wantB, bs)
						calculateIntermediateRolling(dgd, origin, w, h, stride, bd, p, ri, pass, gotA[:bl], gotB[:bl], bs)
						step := 1
						if pass != 0 {
							step = 2
						}
						for y := -1; y < h+1; y += step {
							for x := -1; x < w+1; x++ {
								k := (y+3)*bs + x + 3
								if gotA[k] != wantA[k] || gotB[k] != wantB[k] {
									t.Fatalf("%dx%d bd=%d params=%d radius=%d pass=%d x=%d y=%d A=%d/%d B=%d/%d", w, h, bd, pi, ri, pass, x, y, gotA[k], wantA[k], gotB[k], wantB[k])
								}
							}
						}
						if !slices.Equal(gotA[bl:], tailA) || !slices.Equal(gotB[bl:], tailB) {
							t.Fatal("scratch tail written")
						}
					}
				}
			}
		}
	}
}

func frozenSGRIntermediate(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, bitDepth int, paramsIndex int, radiusIndex int, pass int, aBuf []int32, bBuf []int32, bufStride int) {
	clearInt32s(aBuf)
	clearInt32s(bBuf)
	params := SGRParameterTable[paramsIndex]
	r := int(params.Radius[radiusIndex])
	widthExt := width + 2*SGRProjBorderHorz
	heightExt := height + 2*SGRProjBorderVert
	srcOrigin := dgdOrigin - SGRProjBorderVert*dgdStride - SGRProjBorderHorz
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, false, bBuf, bufStride)
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, true, aBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	step := 1
	if pass != 0 {
		step = 2
	}
	n := (2*r + 1) * (2*r + 1)
	for row := -1; row < height+1; row += step {
		for col := -1; col < width+1; col++ {
			k := aOrigin + row*bufStride + col
			a := uint32(roundPowerOfTwo(aBuf[k], 2*(bitDepth-8)))
			b := uint32(roundPowerOfTwo(bBuf[k], bitDepth-8))
			p := uint32(0)
			if a*uint32(n) >= b*b {
				p = a*uint32(n) - b*b
			}
			z := min(roundPowerOfTwoUnsigned(p*uint32(params.S[radiusIndex]), SGRProjMTableBits), 255)
			aBuf[k] = xByXPlus1Value(z)
			// libaom uses the raw (unshifted) B[k] in the post-update, NOT the
			// bit-depth-scaled local `b`. Using the scaled b mistakenly scales
			// the SGR smoothed image by an extra 1<<(bit_depth-8) factor for
			// 10/12-bit input, producing the q63 first-frame divergence in flt
			// even when reconstruction and CDEF are byte-exact.
			bBuf[k] = int32(roundPowerOfTwoUnsigned(uint32(SGRProjSgr-int(aBuf[k]))*uint32(bBuf[k])*uint32(oneByX[n-1]), SGRProjRecipBits))
		}
	}
}

const restorationDeterministicSeed = 0xbaba

func TestSGRHotTableSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "SGRParams", size: unsafe.Sizeof(SGRParams{}), max: 6},
		{name: "SGRParameterTable", size: unsafe.Sizeof(SGRParameterTable), max: 96},
		{name: "xByXPlus1", size: unsafe.Sizeof(xByXPlus1), max: 512},
		{name: "oneByX", size: unsafe.Sizeof(oneByX), max: 50},
	}
	for _, tc := range tests {
		if tc.size > tc.max {
			t.Fatalf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		}
	}
}

func TestSGRParameterTableMatchesLibaom(t *testing.T) {
	want := [SGRProjParams]SGRParams{
		{Radius: [2]uint8{2, 1}, S: [2]int16{140, 3236}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{112, 2158}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{93, 1618}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{80, 1438}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{70, 1295}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{58, 1177}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{47, 1079}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{37, 996}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{30, 925}},
		{Radius: [2]uint8{2, 1}, S: [2]int16{25, 863}},
		{Radius: [2]uint8{0, 1}, S: [2]int16{-1, 2589}},
		{Radius: [2]uint8{0, 1}, S: [2]int16{-1, 1618}},
		{Radius: [2]uint8{0, 1}, S: [2]int16{-1, 1177}},
		{Radius: [2]uint8{0, 1}, S: [2]int16{-1, 925}},
		{Radius: [2]uint8{2, 0}, S: [2]int16{56, -1}},
		{Radius: [2]uint8{2, 0}, S: [2]int16{22, -1}},
	}
	if SGRParameterTable != want {
		t.Fatalf("SGRParameterTable=%+v want %+v", SGRParameterTable, want)
	}
	xByXPlus1Checks := map[uint32]int32{
		0:   1,
		1:   128,
		2:   171,
		3:   192,
		4:   205,
		16:  241,
		255: 256,
		256: 256,
	}
	for z, want := range xByXPlus1Checks {
		if got := xByXPlus1Value(z); got != want {
			t.Fatalf("xByXPlus1Value(%d)=%d want %d", z, got, want)
		}
	}
	if oneByX[0] != 4096 || oneByX[24] != 164 {
		t.Fatalf("oneByX sentinels mismatch")
	}
}

func TestDecodeSGRXQMatchesLibaom(t *testing.T) {
	tests := []struct {
		name   string
		params SGRParams
		xqd    [2]int8
		want   [2]int16
	}{
		{name: "two-radii", params: SGRParameterTable[0], xqd: [2]int8{20, -7}, want: [2]int16{20, 115}},
		{name: "first-disabled", params: SGRParameterTable[10], xqd: [2]int8{99, 13}, want: [2]int16{0, 115}},
		{name: "second-disabled", params: SGRParameterTable[14], xqd: [2]int8{-12, 44}, want: [2]int16{-12, 0}},
	}
	for _, tt := range tests {
		if got := DecodeSGRXQ(tt.xqd, tt.params); got != tt.want {
			t.Fatalf("%s xq=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestSelfguidedRestorationConstantBlocksStayInRange(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, size := range []struct {
			width  int
			height int
		}{{4, 4}, {17, 19}, {64, 64}} {
			for eps := range SGRProjParams {
				stride := size.width + 2*SGRProjBorderHorz
				origin := SGRProjBorderVert*stride + SGRProjBorderHorz
				src := make([]uint16, stride*(size.height+2*SGRProjBorderVert))
				for i := range src {
					src[i] = max / 3
				}
				dst := make([]uint16, size.width*size.height)
				scratchLen, err := SelfguidedScratchLen(size.width, size.height)
				if err != nil {
					t.Fatal(err)
				}
				scratch := make([]int32, scratchLen)
				if err := ApplySelfguidedRestoration(src, stride, origin, dst, size.width, size.width, size.height, eps, [2]int8{13, -9}, bitDepth, scratch); err != nil {
					t.Fatalf("bitDepth=%d size=%dx%d eps=%d: %v", bitDepth, size.width, size.height, eps, err)
				}
				for i, got := range dst {
					if got > max {
						t.Fatalf("bitDepth=%d size=%dx%d eps=%d dst[%d]=%d max %d", bitDepth, size.width, size.height, eps, i, got, max)
					}
					if got != dst[0] {
						t.Fatalf("bitDepth=%d size=%dx%d eps=%d dst[%d]=%d want constant %d", bitDepth, size.width, size.height, eps, i, got, dst[0])
					}
				}
			}
		}
	}
}

func TestSelfguidedRestorationDeterministicCorpus(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, tc := range []struct {
			width  int
			height int
			eps    int
			xqd    [2]int8
		}{
			{8, 8, 0, [2]int8{0, 0}},
			{13, 11, 7, [2]int8{19, -5}},
			{64, 16, 10, [2]int8{4, 33}},
			{31, 64, 14, [2]int8{-20, 0}},
		} {
			stride := tc.width + 2*SGRProjBorderHorz + 5
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			src := make([]uint16, stride*(tc.height+2*SGRProjBorderVert))
			for i := range src {
				src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			dst := make([]uint16, tc.width*tc.height)
			scratchLen, err := SelfguidedScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]int32, scratchLen)
			if err := ApplySelfguidedRestoration(src, stride, origin, dst, tc.width, tc.width, tc.height, tc.eps, tc.xqd, bitDepth, scratch); err != nil {
				t.Fatalf("bitDepth=%d tc=%+v: %v", bitDepth, tc, err)
			}
			for i, sample := range dst {
				if sample > max {
					t.Fatalf("bitDepth=%d tc=%+v dst[%d]=%d max=%d", bitDepth, tc, i, sample, max)
				}
			}
			dst2 := make([]uint16, len(dst))
			if err := ApplySelfguidedRestoration(src, stride, origin, dst2, tc.width, tc.width, tc.height, tc.eps, tc.xqd, bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(dst, dst2) {
				t.Fatalf("bitDepth=%d tc=%+v not deterministic", bitDepth, tc)
			}
		}
	}
}

func TestSelfguidedScratchLenAndAllocs(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(64+2*SGRProjBorderVert))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplySelfguidedRestoration allocated: %f", allocs)
	}
}

func BenchmarkApplySelfguidedRestoration(b *testing.B) {
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
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
	for b.Loop() {
		_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch)
	}
}

func TestSelfguidedRestorationRejectsInvalidInputs(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 8 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(8+2*SGRProjBorderVert))
	dst := make([]uint16, 8*8)
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-size", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 0, 8, 0, [2]int8{}, 8, scratch)
		}},
		{name: "bad-eps", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 16, [2]int8{}, 8, scratch)
		}},
		{name: "bad-depth", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 0, [2]int8{}, 9, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplySelfguidedRestoration(src[:origin], stride, origin, dst, 8, 8, 8, 0, [2]int8{}, 8, scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst[:8], 8, 8, 8, 0, [2]int8{}, 8, scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 0, [2]int8{}, 8, scratch[:scratchLen-1])
		}},
		{name: "sample-out-of-range", fn: func() error {
			bad := slices.Clone(src)
			bad[origin] = 256
			return ApplySelfguidedRestoration(bad, stride, origin, dst, 8, 8, 8, 0, [2]int8{}, 8, scratch)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidRestoration) {
				t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
			}
		})
	}
}

func FuzzApplySelfguidedRestoration(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(0), int16(0), int16(0), []byte{0, 64, 128, 255})
	f.Add(uint8(31), uint8(17), uint8(2), uint8(15), int16(13), int16(-9), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, rawEps uint8, rawX0 int16, rawX1 int16, data []byte) {
		width := int(rawW%ProcUnitSize) + 1
		height := int(rawH%ProcUnitSize) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 2*SGRProjBorderHorz
		origin := SGRProjBorderVert*stride + SGRProjBorderHorz
		src := make([]uint16, stride*(height+2*SGRProjBorderVert))
		for i := range src {
			src[i] = fuzzRestorationSample(data, i, max)
		}
		dst := make([]uint16, width*height)
		scratchLen, err := SelfguidedScratchLen(width, height)
		if err != nil {
			t.Fatalf("SelfguidedScratchLen err=%v", err)
		}
		scratch := make([]int32, scratchLen)
		err = ApplySelfguidedRestoration(src, stride, origin, dst, width, width, height, int(rawEps%SGRProjParams), [2]int8{int8(rawX0 % 96), int8(rawX1 % 96)}, bitDepth, scratch)
		if err != nil {
			t.Fatalf("ApplySelfguidedRestoration err=%v", err)
		}
		for i, sample := range dst {
			if sample > max {
				t.Fatalf("dst[%d]=%d max=%d", i, sample, max)
			}
		}
	})
}

func fuzzRestorationSample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}

type restorationRandom struct {
	state uint32
}

func newRestorationRandom(seed uint32) *restorationRandom {
	return &restorationRandom{state: seed}
}

func (r *restorationRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *restorationRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}

// The unchanged u16 entry independently clears its staging. Poison the u8
// workspace between calls, including inactive filter planes, and exercise
// in-place output after all source samples have been staged.
func TestApplySelfguidedU8ScratchReuse(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x919)
	for _, sz := range [][2]int{{1, 1}, {3, 7}, {8, 8}, {17, 15}, {63, 64}, {64, 63}} {
		w, h := sz[0], sz[1]
		stride := w + 11
		origin := 3*stride + 4
		src8 := randomU8Plane(rnd, stride, h+6)
		need, err := SelfguidedScratchLen(w, h)
		if err != nil {
			t.Fatal(err)
		}
		scratch := make([]int32, need+11)
		for pi := range SGRParameterTable {
			for _, inplace := range []bool{false, true} {
				for i := range scratch {
					scratch[i] = int32(uint32(i)*2654435761 + 0x91a7)
				}
				tail := slices.Clone(scratch[need:])
				gotSrc := slices.Clone(src8)
				wantSrc := widenU8(src8)
				got := make([]byte, stride*h)
				want := make([]uint16, stride*h)
				if inplace {
					got = gotSrc[origin:]
					want = wantSrc[origin:]
				}
				xqd := [2]int8{-91, 73}
				if err := ApplySelfguidedRestoration(wantSrc, stride, origin, want, stride, w, h, pi, xqd, 8, make([]int32, need)); err != nil {
					t.Fatal(err)
				}
				if err := ApplySelfguidedRestorationU8(gotSrc, stride, origin, got, stride, w, h, pi, xqd, scratch); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if uint16(got[i]) != want[i] {
						t.Fatalf("%dx%d params=%d inplace=%v sample=%d got=%d want=%d", w, h, pi, inplace, i, got[i], want[i])
					}
				}
				if !slices.Equal(scratch[need:], tail) {
					t.Fatal("scratch tail overwritten")
				}
			}
		}
	}
}

// This file is the byte-exactness contract for the 8-bit-pixel kernel family:
// for every pass, the u8 kernels (through their resolved dispatch slots, i.e.
// NEON on arm64 / AVX2 on native amd64) must produce exactly the bytes the
// existing u16 kernels produce when fed the same samples widened to uint16 at
// bitDepth=8. Sizes cover the vector widths, non-multiple-of-8 tails, unit
// edges (1-wide/1-high), and the full 64x64 processing unit (restoration
// units up to 256px are filtered as <=64x64 processing units, so 64 is the
// largest block any kernel ever sees).

var u8DiffSizes = []struct{ width, height int }{
	{1, 1}, {2, 3}, {5, 3}, {7, 7}, {8, 8}, {8, 1}, {1, 8}, {13, 9},
	{16, 12}, {17, 19}, {24, 7}, {31, 5}, {33, 17}, {40, 64}, {56, 33},
	{64, 1}, {64, 31}, {64, 64},
}

func widenU8(src []uint8) []uint16 {
	out := make([]uint16, len(src))
	for i, s := range src {
		out[i] = uint16(s)
	}
	return out
}

func randomU8Plane(rnd *restorationRandom, stride, rows int) []uint8 {
	plane := make([]uint8, stride*rows)
	for i := range plane {
		plane[i] = uint8(rnd.pseudoUniform(256))
	}
	return plane
}

// TestWienerHorizontalU8MatchesU16Widened proves the u8 horizontal pass
// (dispatched and pure-Go) emits the exact 16-bit temp the u16 pass emits on
// the widened plane, across sizes, strides, origins, and filters.
func TestWienerHorizontalU8MatchesU16Widened(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x811)
	round0, _ := wienerRounds(8)
	const max = uint16(255)
	for _, sz := range u8DiffSizes {
		for fi, info := range wienerDispatchFilters() {
			extra := rnd.pseudoUniform(9)
			stride := sz.width + 2*WienerHalfwin + extra
			origin := WienerHalfwin*stride + WienerHalfwin + rnd.pseudoUniform(extra+1)
			src8 := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			src16 := widenU8(src8)

			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			want := make([]uint16, tempLen)
			if !wienerHorizontal(src16, stride, origin, sz.width, sz.height, info.HFilter, 8, round0, max, want) {
				t.Fatalf("sz=%dx%d f=%d: u16 reference rejected valid input", sz.width, sz.height, fi)
			}

			gotRef := make([]uint16, tempLen)
			wienerHorizontalU8(src8, stride, origin, sz.width, sz.height, info.HFilter, round0, gotRef)
			gotImpl := make([]uint16, tempLen)
			wienerHorizontalU8Impl(src8, stride, origin, sz.width, sz.height, info.HFilter, round0, gotImpl)

			for i := range want {
				if gotRef[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d pure-Go temp[%d]=%d want %d", sz.width, sz.height, fi, i, gotRef[i], want[i])
				}
				if gotImpl[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d dispatched temp[%d]=%d want %d", sz.width, sz.height, fi, i, gotImpl[i], want[i])
				}
			}
		}
	}
}

// TestWienerVerticalU8MatchesU16Widened proves the u8 vertical pass emits
// exactly the bytes the u16 pass emits (all u16 outputs are <=255 at
// bitDepth=8), including untouched dst padding.
func TestWienerVerticalU8MatchesU16Widened(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x812)
	round0, round1 := wienerRounds(8)
	const max = uint16(255)
	for _, sz := range u8DiffSizes {
		for fi, info := range wienerDispatchFilters() {
			stride := sz.width + 2*WienerHalfwin
			origin := WienerHalfwin*stride + WienerHalfwin
			src8 := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			src16 := widenU8(src8)

			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			temp := make([]uint16, tempLen)
			if !wienerHorizontal(src16, stride, origin, sz.width, sz.height, info.HFilter, 8, round0, max, temp) {
				t.Fatal("u16 reference rejected valid input")
			}

			dstPad := rnd.pseudoUniform(5)
			dstStride := sz.width + dstPad
			want16 := make([]uint16, dstStride*sz.height)
			wienerVertical(temp, sz.width, want16, dstStride, sz.width, sz.height, info.VFilter, 8, round1, max)

			gotRef := make([]uint8, dstStride*sz.height)
			gotImpl := make([]uint8, dstStride*sz.height)
			for i := range gotRef {
				gotRef[i] = 0xAA
				gotImpl[i] = 0xAA
			}
			wienerVerticalU8(temp, sz.width, gotRef, dstStride, sz.width, sz.height, info.VFilter, round1)
			wienerVerticalU8Impl(temp, sz.width, gotImpl, dstStride, sz.width, sz.height, info.VFilter, round1)

			for row := 0; row < sz.height; row++ {
				for col := 0; col < dstStride; col++ {
					i := row*dstStride + col
					if col >= sz.width {
						if gotRef[i] != 0xAA || gotImpl[i] != 0xAA {
							t.Fatalf("sz=%dx%d f=%d padding overwritten at %d", sz.width, sz.height, fi, i)
						}
						continue
					}
					if want16[i] > 255 {
						t.Fatalf("u16 vertical produced %d > 255 at bitDepth 8", want16[i])
					}
					want := uint8(want16[i])
					if gotRef[i] != want {
						t.Fatalf("sz=%dx%d f=%d pure-Go dst[%d]=%d want %d", sz.width, sz.height, fi, i, gotRef[i], want)
					}
					if gotImpl[i] != want {
						t.Fatalf("sz=%dx%d f=%d dispatched dst[%d]=%d want %d", sz.width, sz.height, fi, i, gotImpl[i], want)
					}
				}
			}
		}
	}
}

// TestApplyWienerRestorationU8MatchesU16Widened is the end-to-end entry-point
// differential: ApplyWienerRestorationU8 vs ApplyWienerRestorationTrusted on
// the widened plane, byte-for-byte.
func TestApplyWienerRestorationU8MatchesU16Widened(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x813)
	for _, sz := range u8DiffSizes {
		for fi, info := range wienerDispatchFilters() {
			extra := rnd.pseudoUniform(7)
			stride := sz.width + 2*WienerHalfwin + extra
			origin := WienerHalfwin*stride + WienerHalfwin + rnd.pseudoUniform(extra+1)
			src8 := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			src16 := widenU8(src8)

			scratchLen, err := WienerScratchLen(sz.width, sz.height)
			if err != nil {
				t.Fatal(err)
			}
			dstStride := sz.width + rnd.pseudoUniform(4)
			want16 := make([]uint16, dstStride*sz.height)
			if err := ApplyWienerRestorationTrusted(src16, stride, origin, want16, dstStride, sz.width, sz.height, info, 8, make([]uint16, scratchLen)); err != nil {
				t.Fatal(err)
			}
			got8 := make([]uint8, dstStride*sz.height)
			if err := ApplyWienerRestorationU8(src8, stride, origin, got8, dstStride, sz.width, sz.height, info, make([]uint16, scratchLen)); err != nil {
				t.Fatal(err)
			}
			for row := 0; row < sz.height; row++ {
				for col := 0; col < sz.width; col++ {
					i := row*dstStride + col
					if uint16(got8[i]) != want16[i] {
						t.Fatalf("sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, got8[i], want16[i])
					}
				}
			}
		}
	}
}

// sgrDiffXQD returns deterministic valid xqd pairs spanning the projection
// coefficient ranges.
func sgrDiffXQD(rnd *restorationRandom) [][2]int8 {
	pairs := [][2]int8{
		{SGRProjPrjMin0, SGRProjPrjMin1},
		{SGRProjPrjMax0, SGRProjPrjMax1},
		{0, 0},
		{-33, 91},
	}
	for range 4 {
		pairs = append(pairs, [2]int8{
			int8(SGRProjPrjMin0 + rnd.pseudoUniform(SGRProjPrjMax0-SGRProjPrjMin0+1)),
			int8(SGRProjPrjMin1 + rnd.pseudoUniform(SGRProjPrjMax1-SGRProjPrjMin1+1)),
		})
	}
	return pairs
}

// TestApplySelfguidedRestorationU8MatchesU16Widened is the end-to-end SGR
// differential across every parameter index (covering the radius0-only,
// radius1-only, and dual-radius pipelines), xqd corner and random values, and
// the size/stride corpus.
func TestApplySelfguidedRestorationU8MatchesU16Widened(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x814)
	xqds := sgrDiffXQD(rnd)
	for paramsIndex := range SGRParameterTable {
		for _, sz := range u8DiffSizes {
			xqd := xqds[(paramsIndex+sz.width+sz.height)%len(xqds)]
			extra := rnd.pseudoUniform(7)
			stride := sz.width + 2*SGRProjBorderHorz + extra
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz + rnd.pseudoUniform(extra+1)
			src8 := randomU8Plane(rnd, stride, sz.height+2*SGRProjBorderVert)
			src16 := widenU8(src8)

			scratchLen, err := SelfguidedScratchLen(sz.width, sz.height)
			if err != nil {
				t.Fatal(err)
			}
			dstStride := sz.width + rnd.pseudoUniform(4)
			want16 := make([]uint16, dstStride*sz.height)
			if err := ApplySelfguidedRestoration(src16, stride, origin, want16, dstStride, sz.width, sz.height, paramsIndex, xqd, 8, make([]int32, scratchLen)); err != nil {
				t.Fatal(err)
			}
			got8 := make([]uint8, dstStride*sz.height)
			if err := ApplySelfguidedRestorationU8(src8, stride, origin, got8, dstStride, sz.width, sz.height, paramsIndex, xqd, make([]int32, scratchLen)); err != nil {
				t.Fatal(err)
			}
			for row := 0; row < sz.height; row++ {
				for col := 0; col < sz.width; col++ {
					i := row*dstStride + col
					if uint16(got8[i]) != want16[i] {
						t.Fatalf("params=%d xqd=%v sz=%dx%d dst[%d]=%d want %d", paramsIndex, xqd, sz.width, sz.height, i, got8[i], want16[i])
					}
				}
			}
		}
	}
}

// TestSGRWeightedRowU8DispatchMatchesReference exercises the dispatched final
// projection row against the scalar reference on synthetic flt values that
// bound the pipeline range, including widths with <8-column tails.
func TestSGRWeightedRowU8DispatchMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x815)
	widths := []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 40, 63, 64}
	xqs := [][2]int32{{0, 128}, {31, 0}, {-96, 256}, {0, 33}, {56, 72}, {-96, 2}}
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
			sgrWeightedRowU8Impl(got, src, f0, f1, xq[0], xq[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
				}
			}
		}
	}
}

// TestApplyWienerRestorationU8RejectsInvalidInputs mirrors the u16 entry's
// validation for the u8 entry.
func TestApplyWienerRestorationU8RejectsInvalidInputs(t *testing.T) {
	scratchLen, err := WienerScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 8 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint8, stride*(8+2*WienerHalfwin))
	dst := make([]uint8, 8*8)
	badFilter := DefaultWienerInfo()
	badFilter.HFilter[7] = 1
	const maxInt = int(^uint(0) >> 1)
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-size", fn: func() error {
			return ApplyWienerRestorationU8(src, stride, origin, dst, 8, 0, 8, DefaultWienerInfo(), scratch)
		}},
		{name: "bad-filter", fn: func() error {
			return ApplyWienerRestorationU8(src, stride, origin, dst, 8, 8, 8, badFilter, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplyWienerRestorationU8(src[:origin], stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplyWienerRestorationU8(src, stride, origin, dst[:8], 8, 8, 8, DefaultWienerInfo(), scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplyWienerRestorationU8(src, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), scratch[:scratchLen-1])
		}},
		{name: "overflow-stride", fn: func() error {
			return ApplyWienerRestorationU8(src, maxInt/2, 0, dst, 4, 4, 4, DefaultWienerInfo(), scratch)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidRestoration) {
				t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
			}
		})
	}
}

// TestApplySelfguidedRestorationU8RejectsInvalidInputs mirrors the u16 entry's
// validation for the u8 entry.
func TestApplySelfguidedRestorationU8RejectsInvalidInputs(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 8 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint8, stride*(8+2*SGRProjBorderVert))
	dst := make([]uint8, 8*8)
	const maxInt = int(^uint(0) >> 1)
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-params", fn: func() error {
			return ApplySelfguidedRestorationU8(src, stride, origin, dst, 8, 8, 8, len(SGRParameterTable), [2]int8{}, scratch)
		}},
		{name: "bad-size", fn: func() error {
			return ApplySelfguidedRestorationU8(src, stride, origin, dst, 8, 0, 8, 0, [2]int8{}, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplySelfguidedRestorationU8(src[:origin], stride, origin, dst, 8, 8, 8, 0, [2]int8{}, scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplySelfguidedRestorationU8(src, stride, origin, dst[:8], 8, 8, 8, 0, [2]int8{}, scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplySelfguidedRestorationU8(src, stride, origin, dst, 8, 8, 8, 0, [2]int8{}, scratch[:scratchLen-1])
		}},
		{name: "overflow-stride", fn: func() error {
			return ApplySelfguidedRestorationU8(src, maxInt/2, 0, dst, 4, 4, 4, 0, [2]int8{}, scratch)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidRestoration) {
				t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
			}
		})
	}
}

// TestApplyWienerRestorationU8IsZeroAlloc protects the hot-path contract that
// the u8 entry does not allocate per call.
func TestApplyWienerRestorationU8IsZeroAlloc(t *testing.T) {
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x816)
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	dst := make([]uint8, 64*64)
	info := DefaultWienerInfo()
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyWienerRestorationU8(src, stride, origin, dst, 64, 64, 64, info, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyWienerRestorationU8 allocated: %f", allocs)
	}
}

// TestApplySelfguidedRestorationU8IsZeroAlloc protects the hot-path contract
// that the u8 entry does not allocate per call.
func TestApplySelfguidedRestorationU8IsZeroAlloc(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x817)
	src := randomU8Plane(rnd, stride, 64+2*SGRProjBorderVert)
	dst := make([]uint8, 64*64)
	allocs := testing.AllocsPerRun(200, func() {
		if err := ApplySelfguidedRestorationU8(src, stride, origin, dst, 64, 64, 64, 0, [2]int8{-32, 31}, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplySelfguidedRestorationU8 allocated: %f", allocs)
	}
}

// Benchmarks: u8 vs u16 at bitDepth 8 on the 64x64 processing unit. The u16
// numbers are the same-input baseline for the widened pipeline the u8 kernels
// replace (excluding the whole-plane widen the tile layer also saves).

func benchWienerU8Inputs(b *testing.B) ([]uint8, int, int, []uint16, WienerInfo) {
	b.Helper()
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xabc)
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	return src, stride, origin, make([]uint16, scratchLen), DefaultWienerInfo()
}

func BenchmarkApplyWienerRestorationU8(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	dst := make([]uint8, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		if err := ApplyWienerRestorationU8(src, stride, origin, dst, 64, 64, 64, info, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyWienerRestorationU16At8Bit(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	src16 := widenU8(src)
	dst := make([]uint16, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		if err := ApplyWienerRestorationTrusted(src16, stride, origin, dst, 64, 64, 64, info, 8, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWienerHorizontalU8(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	round0, _ := wienerRounds(8)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		wienerHorizontalU8Impl(src, stride, origin, 64, 64, info.HFilter, round0, scratch)
	}
}

func BenchmarkWienerHorizontalU16At8Bit(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	src16 := widenU8(src)
	round0, _ := wienerRounds(8)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		wienerHorizontalTrustedImpl(src16, stride, origin, 64, 64, info.HFilter, 8, round0, 255, scratch)
	}
}

func BenchmarkWienerVerticalU8(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	round0, round1 := wienerRounds(8)
	wienerHorizontalU8(src, stride, origin, 64, 64, info.HFilter, round0, scratch)
	dst := make([]uint8, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		wienerVerticalU8Impl(scratch, 64, dst, 64, 64, 64, info.VFilter, round1)
	}
}

func BenchmarkWienerVerticalU16At8Bit(b *testing.B) {
	src, stride, origin, scratch, info := benchWienerU8Inputs(b)
	round0, round1 := wienerRounds(8)
	wienerHorizontalU8(src, stride, origin, 64, 64, info.HFilter, round0, scratch)
	dst := make([]uint16, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		wienerVerticalImpl(scratch, 64, dst, 64, 64, 64, info.VFilter, 8, round1, 255)
	}
}

// Plane-scale benchmarks: walk 64x64 Wiener units across a 1080p-sized plane
// so the source streams through the cache hierarchy like a real restoration
// pass (the 64x64 single-unit benchmarks above stay L1-resident and hide the
// u8 kernels' halved load bandwidth).

const (
	benchPlaneW = 1920
	benchPlaneH = 1080
)

func BenchmarkWienerPlane1080pU8(b *testing.B) {
	stride := benchPlaneW + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xfab)
	src := randomU8Plane(rnd, stride, benchPlaneH+2*WienerHalfwin)
	dst := make([]uint8, benchPlaneW*benchPlaneH)
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	info := DefaultWienerInfo()
	b.ReportAllocs()
	b.SetBytes(benchPlaneW * benchPlaneH)
	b.ResetTimer()
	for b.Loop() {
		for y := 0; y < benchPlaneH; y += 64 {
			for x := 0; x < benchPlaneW; x += 64 {
				if err := ApplyWienerRestorationU8(src, stride, origin+y*stride+x, dst[y*benchPlaneW+x:], benchPlaneW, 64, min(64, benchPlaneH-y), info, scratch); err != nil {
					b.Fatal(err)
				}
			}
		}
	}
}

func BenchmarkWienerPlane1080pU16At8Bit(b *testing.B) {
	stride := benchPlaneW + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xfab)
	src := widenU8(randomU8Plane(rnd, stride, benchPlaneH+2*WienerHalfwin))
	dst := make([]uint16, benchPlaneW*benchPlaneH)
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	info := DefaultWienerInfo()
	b.ReportAllocs()
	b.SetBytes(benchPlaneW * benchPlaneH)
	b.ResetTimer()
	for b.Loop() {
		for y := 0; y < benchPlaneH; y += 64 {
			for x := 0; x < benchPlaneW; x += 64 {
				if err := ApplyWienerRestorationTrusted(src, stride, origin+y*stride+x, dst[y*benchPlaneW+x:], benchPlaneW, 64, min(64, benchPlaneH-y), info, 8, scratch); err != nil {
					b.Fatal(err)
				}
			}
		}
	}
}

func benchSGRU8Inputs(b *testing.B) ([]uint8, int, int, []int32) {
	b.Helper()
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xdef)
	src := randomU8Plane(rnd, stride, 64+2*SGRProjBorderVert)
	return src, stride, origin, make([]int32, scratchLen)
}

func BenchmarkApplySelfguidedRestorationU8(b *testing.B) {
	src, stride, origin, scratch := benchSGRU8Inputs(b)
	dst := make([]uint8, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		if err := ApplySelfguidedRestorationU8(src, stride, origin, dst, 64, 64, 64, 0, [2]int8{-32, 31}, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplySelfguidedRestorationU16At8Bit(b *testing.B) {
	src, stride, origin, scratch := benchSGRU8Inputs(b)
	src16 := widenU8(src)
	dst := make([]uint16, 64*64)
	b.ReportAllocs()
	b.SetBytes(64 * 64)
	b.ResetTimer()
	for b.Loop() {
		if err := ApplySelfguidedRestoration(src16, stride, origin, dst, 64, 64, 64, 0, [2]int8{-32, 31}, 8, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

// wienerDispatchCorpus enumerates a representative set of Wiener filters,
// bit depths, and block sizes. Widths cover both NEON-eligible multiples of 8
// and odd widths that must hit the pure-Go fallback inside the NEON wrapper.
func wienerDispatchFilters() []WienerInfo {
	return []WienerInfo{
		DefaultWienerInfo(),
		{VFilter: NewWienerFilter(-4, -13, 29), HFilter: NewWienerFilter(7, -5, 13)},
		{VFilter: NewWienerFilter(WienerTap0Max, WienerTap1Min, 21), HFilter: NewWienerFilter(WienerTap0Min, WienerTap1Max, WienerTap2Max)},
		{VFilter: NewWienerFilter(WienerTap0Min, WienerTap1Min, WienerTap2Min), HFilter: NewWienerFilter(WienerTap0Max, WienerTap1Max, WienerTap2Max)},
		{VFilter: NewWienerFilter(10, -23, -17), HFilter: NewWienerFilter(-5, 8, 46)},
	}
}

// TestWienerDispatchMatchesPureGo is the bit-exactness guard for the dispatched
// Wiener passes. It exercises the resolved dispatch slots (which route through
// the NEON asm on arm64) against the canonical pure-Go reference across the
// matrix of bit depths, filters, and block widths/heights. Every temp and
// destination sample must match the reference exactly.
func TestWienerDispatchMatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5151)
	sizes := []struct{ width, height int }{
		{8, 8}, {16, 12}, {64, 64}, {64, 31}, {24, 7}, {13, 9}, {1, 1}, {40, 64},
	}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(int(bitDepth))
		for _, sz := range sizes {
			for fi, info := range wienerDispatchFilters() {
				stride := sz.width + 2*WienerHalfwin + 4
				origin := WienerHalfwin*stride + WienerHalfwin + 2
				src := make([]uint16, stride*(sz.height+2*WienerHalfwin))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)

				// Horizontal pass: reference vs dispatch.
				wantTemp := make([]uint16, tempLen)
				gotTemp := make([]uint16, tempLen)
				if !wienerHorizontal(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, wantTemp) {
					t.Fatalf("bd=%d sz=%dx%d f=%d horizontal reference rejected in-range input", bitDepth, sz.width, sz.height, fi)
				}
				wienerHorizontalTrustedImpl(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, gotTemp)
				for i := range wantTemp {
					if wantTemp[i] != gotTemp[i] {
						t.Fatalf("bd=%d sz=%dx%d f=%d horizontal temp[%d]=%d want %d", bitDepth, sz.width, sz.height, fi, i, gotTemp[i], wantTemp[i])
					}
				}

				// Vertical pass over the shared (reference) temp buffer.
				dstStride := sz.width + 3
				wantDst := make([]uint16, dstStride*sz.height)
				gotDst := make([]uint16, dstStride*sz.height)
				wienerVertical(wantTemp, sz.width, wantDst, dstStride, sz.width, sz.height, info.VFilter, int(bitDepth), round1, max)
				wienerVerticalImpl(wantTemp, sz.width, gotDst, dstStride, sz.width, sz.height, info.VFilter, int(bitDepth), round1, max)
				for i := range wantDst {
					if wantDst[i] != gotDst[i] {
						t.Fatalf("bd=%d sz=%dx%d f=%d vertical dst[%d]=%d want %d", bitDepth, sz.width, sz.height, fi, i, gotDst[i], wantDst[i])
					}
				}
			}
		}
	}
}

// TestWienerDispatchIsZeroAlloc protects the hot-path contract that the
// dispatched passes do not allocate per call.
func TestWienerDispatchIsZeroAlloc(t *testing.T) {
	const width, height = 64, 64
	const bitDepth uint8 = 12
	max := uint16((1 << bitDepth) - 1)
	round0, round1 := wienerRounds(int(bitDepth))
	stride := width + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(height+2*WienerHalfwin))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x2727)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	dst := make([]uint16, width*height)
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerHorizontalTrustedImpl(src, stride, origin, width, height, info.HFilter, int(bitDepth), round0, max, temp)
	}); allocs != 0 {
		t.Fatalf("horizontal dispatch allocated %f times per call", allocs)
	}
	if allocs := testing.AllocsPerRun(200, func() {
		wienerVerticalImpl(temp, width, dst, width, width, height, info.VFilter, int(bitDepth), round1, max)
	}); allocs != 0 {
		t.Fatalf("vertical dispatch allocated %f times per call", allocs)
	}
}

func benchWienerSetup() (src []uint16, temp []uint16, dst []uint16, stride, origin int, info WienerInfo) {
	const width, height = 64, 64
	const bitDepth uint8 = 12
	max := uint16((1 << bitDepth) - 1)
	stride = width + 2*WienerHalfwin
	origin = WienerHalfwin*stride + WienerHalfwin
	src = make([]uint16, stride*(height+2*WienerHalfwin))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x6363)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
	temp = make([]uint16, width*(height+2*WienerHalfwin))
	dst = make([]uint16, width*height)
	info = DefaultWienerInfo()
	return
}

func BenchmarkWienerHorizontalDispatch(b *testing.B) {
	src, temp, _, stride, origin, info := benchWienerSetup()
	round0, _ := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	b.ReportAllocs()
	for b.Loop() {
		wienerHorizontalTrustedImpl(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	}
}

func BenchmarkWienerVerticalPureGo(b *testing.B) {
	src, temp, dst, stride, origin, info := benchWienerSetup()
	round0, round1 := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	wienerHorizontal(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	b.ReportAllocs()
	for b.Loop() {
		wienerVertical(temp, 64, dst, 64, 64, 64, info.VFilter, 12, round1, max)
	}
}

func BenchmarkWienerVerticalDispatch(b *testing.B) {
	src, temp, dst, stride, origin, info := benchWienerSetup()
	round0, round1 := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	wienerHorizontal(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	b.ReportAllocs()
	for b.Loop() {
		wienerVerticalImpl(temp, 64, dst, 64, 64, 64, info.VFilter, 12, round1, max)
	}
}

func TestDefaultWienerInfoMatchesLibaom(t *testing.T) {
	want := WienerFilter{3, -7, 15, -22, 15, -7, 3, 0}
	got := DefaultWienerInfo()
	if got.VFilter != want || got.HFilter != want {
		t.Fatalf("DefaultWienerInfo=%+v want %+v", got, want)
	}
	if WienerTap0Min != -5 || WienerTap0Max != 10 ||
		WienerTap1Min != -23 || WienerTap1Max != 8 ||
		WienerTap2Min != -17 || WienerTap2Max != 46 {
		t.Fatalf("tap ranges mismatch")
	}
	roundTests := map[uint8][2]int{
		8:  {3, 11},
		10: {3, 11},
		12: {5, 9},
	}
	for bitDepth, want := range roundTests {
		got0, got1 := wienerRounds(int(bitDepth))
		if got0 != want[0] || got1 != want[1] {
			t.Fatalf("bitDepth=%d rounds=%d,%d want %d,%d", bitDepth, got0, got1, want[0], want[1])
		}
	}
	if !validWienerFilter(NewWienerFilter(WienerTap0Min, WienerTap1Min, WienerTap2Min)) ||
		!validWienerFilter(NewWienerFilter(WienerTap0Max, WienerTap1Max, WienerTap2Max)) {
		t.Fatalf("tap range endpoints should be valid")
	}
}

func TestWienerRestorationPreservesConstantBlocks(t *testing.T) {
	filters := []WienerInfo{
		DefaultWienerInfo(),
		{VFilter: NewWienerFilter(-3, -11, 22), HFilter: NewWienerFilter(6, -9, 18)},
	}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, size := range []struct {
			width  int
			height int
		}{{1, 1}, {17, 13}, {64, 64}} {
			for _, info := range filters {
				stride := size.width + 2*WienerHalfwin
				origin := WienerHalfwin*stride + WienerHalfwin
				src := make([]uint16, stride*(size.height+2*WienerHalfwin))
				for i := range src {
					src[i] = max / 3
				}
				dst := make([]uint16, size.width*size.height)
				scratchLen, err := WienerScratchLen(size.width, size.height)
				if err != nil {
					t.Fatal(err)
				}
				scratch := make([]uint16, scratchLen)
				if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, size.width, size.width, size.height, info, bitDepth, scratch); err != nil {
					t.Fatalf("bitDepth=%d size=%dx%d: %v", bitDepth, size.width, size.height, err)
				}
				for i, got := range dst {
					if got != max/3 {
						t.Fatalf("bitDepth=%d size=%dx%d dst[%d]=%d want %d", bitDepth, size.width, size.height, i, got, max/3)
					}
				}
			}
		}
	}
}

func TestApplyWienerRestorationMatchesReferenceCorpus(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x57)
	for _, tc := range []struct {
		name     string
		bitDepth uint8
		width    int
		height   int
		dstPad   int
		info     WienerInfo
	}{
		{name: "default-8bit", bitDepth: 8, width: 8, height: 8, dstPad: 0, info: DefaultWienerInfo()},
		{name: "custom-10bit", bitDepth: 10, width: 13, height: 9, dstPad: 3, info: WienerInfo{
			VFilter: NewWienerFilter(-4, -13, 29),
			HFilter: NewWienerFilter(7, -5, 13),
		}},
		{name: "large-12bit", bitDepth: 12, width: 64, height: 31, dstPad: 5, info: WienerInfo{
			VFilter: NewWienerFilter(WienerTap0Max, WienerTap1Min, 21),
			HFilter: NewWienerFilter(WienerTap0Min, WienerTap1Max, WienerTap2Max),
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			max := uint16((1 << tc.bitDepth) - 1)
			stride := tc.width + 2*WienerHalfwin + 4
			origin := WienerHalfwin*stride + WienerHalfwin + 2
			src := make([]uint16, stride*(tc.height+2*WienerHalfwin))
			for i := range src {
				src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			dstStride := tc.width + tc.dstPad
			dst := make([]uint16, dstStride*tc.height)
			for i := range dst {
				dst[i] = max
			}
			scratchLen, err := WienerScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]uint16, scratchLen)
			if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, dstStride, tc.width, tc.height, tc.info, tc.bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			for row := 0; row < tc.height; row++ {
				for col := 0; col < tc.width; col++ {
					want := referenceWienerPixel(src, stride, origin, row, col, tc.info, tc.bitDepth)
					if got := dst[row*dstStride+col]; got != want {
						t.Fatalf("row=%d col=%d got=%d want=%d", row, col, got, want)
					}
				}
				for col := tc.width; col < dstStride; col++ {
					if got := dst[row*dstStride+col]; got != max {
						t.Fatalf("padding row=%d col=%d overwritten with %d", row, col, got)
					}
				}
			}
		})
	}
}

func TestWienerScratchLenAndAllocs(t *testing.T) {
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scratchLen != 64*(64+2*WienerHalfwin) {
		t.Fatalf("scratchLen=%d", scratchLen)
	}
	scratch := make([]uint16, scratchLen)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(64+2*WienerHalfwin))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x99)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	info := WienerInfo{
		VFilter: NewWienerFilter(-5, 8, 46),
		HFilter: NewWienerFilter(10, -23, -17),
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, 64, 64, 64, info, 12, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyWienerRestoration allocated: %f", allocs)
	}
}

func TestApplyWienerRestorationRejectsInvalidInputs(t *testing.T) {
	scratchLen, err := WienerScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 8 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(8+2*WienerHalfwin))
	dst := make([]uint16, 8*8)
	badFilter := DefaultWienerInfo()
	badFilter.HFilter[7] = 1
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-size", fn: func() error {
			return ApplyWienerRestorationTrusted(src, stride, origin, dst, 8, 0, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "bad-depth", fn: func() error {
			return ApplyWienerRestorationTrusted(src, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 9, scratch)
		}},
		{name: "bad-filter", fn: func() error {
			return ApplyWienerRestorationTrusted(src, stride, origin, dst, 8, 8, 8, badFilter, 8, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplyWienerRestorationTrusted(src[:origin], stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplyWienerRestorationTrusted(src, stride, origin, dst[:8], 8, 8, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplyWienerRestorationTrusted(src, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 8, scratch[:scratchLen-1])
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidRestoration) {
				t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
			}
		})
	}
}

// TestApplyWienerRestorationRejectsOverflowStride confirms that an
// attacker-controlled srcStride/srcOrigin combination cannot wrap the
// bordered-block size check and slip past validation.
func TestApplyWienerRestorationRejectsOverflowStride(t *testing.T) {
	scratch := make([]uint16, 1024)
	dst := make([]uint16, 16)
	src := make([]uint16, 16)
	const maxInt = int(^uint(0) >> 1)
	// borderY*stride * 3 should overflow, and origin should be small enough
	// to require subtraction to remain inside bounds.
	err := ApplyWienerRestorationTrusted(src, maxInt/2, 0, dst, 4, 4, 4, DefaultWienerInfo(), 8, scratch)
	if !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
	}
}

// TestApplySelfguidedRestorationRejectsOverflowStride confirms the same
// overflow guard is reachable from the self-guided entry point.
func TestApplySelfguidedRestorationRejectsOverflowStride(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	dst := make([]uint16, 16)
	src := make([]uint16, 16)
	const maxInt = int(^uint(0) >> 1)
	err = ApplySelfguidedRestoration(src, maxInt/2, 0, dst, 4, 4, 4, 0, [2]int8{}, 8, scratch)
	if !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
	}
}

func FuzzApplyWienerRestoration(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), int16(3), int16(-7), int16(15), []byte{0, 64, 128, 255})
	f.Add(uint8(31), uint8(17), uint8(2), int16(-5), int16(8), int16(46), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, rawTap0 int16, rawTap1 int16, rawTap2 int16, data []byte) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 2*WienerHalfwin
		origin := WienerHalfwin*stride + WienerHalfwin
		src := make([]uint16, stride*(height+2*WienerHalfwin))
		for i := range src {
			src[i] = fuzzRestorationSample(data, i, max)
		}
		dst := make([]uint16, width*height)
		scratchLen, err := WienerScratchLen(width, height)
		if err != nil {
			t.Fatalf("WienerScratchLen err=%v", err)
		}
		scratch := make([]uint16, scratchLen)
		info := WienerInfo{
			VFilter: NewWienerFilter(fuzzWienerTap(rawTap0, WienerTap0Min, WienerTap0Max), fuzzWienerTap(rawTap1, WienerTap1Min, WienerTap1Max), fuzzWienerTap(rawTap2, WienerTap2Min, WienerTap2Max)),
			HFilter: NewWienerFilter(fuzzWienerTap(rawTap2, WienerTap0Min, WienerTap0Max), fuzzWienerTap(rawTap0, WienerTap1Min, WienerTap1Max), fuzzWienerTap(rawTap1, WienerTap2Min, WienerTap2Max)),
		}
		if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, width, width, height, info, bitDepth, scratch); err != nil {
			t.Fatalf("ApplyWienerRestoration err=%v", err)
		}
		for i, sample := range dst {
			if sample > max {
				t.Fatalf("dst[%d]=%d max=%d", i, sample, max)
			}
		}
		if len(dst) > 0 {
			want := referenceWienerPixel(src, stride, origin, height/2, width/2, info, bitDepth)
			if got := dst[(height/2)*width+width/2]; got != want {
				t.Fatalf("middle got=%d want=%d", got, want)
			}
		}
	})
}

func BenchmarkApplyWienerRestoration(b *testing.B) {
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(64+2*WienerHalfwin))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xabc)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	info := DefaultWienerInfo()
	b.ReportAllocs()
	b.SetBytes(64 * 64 * 2)
	b.ResetTimer()
	for b.Loop() {
		if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, 64, 64, 64, info, 12, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func referenceWienerPixel(src []uint16, stride int, origin int, row int, col int, info WienerInfo, bitDepth uint8) uint16 {
	max := int32((1 << bitDepth) - 1)
	round0, round1 := referenceWienerRounds(int(bitDepth))
	center := referenceWienerHorizontal(src, stride, origin, row, col, info.HFilter, int(bitDepth), round0)
	sum := int32(center)<<WienerFilterBits - int32(1<<(int(bitDepth)+round1-1))
	for k := range WienerWin {
		v := referenceWienerHorizontal(src, stride, origin, row-WienerHalfwin+k, col, info.HFilter, int(bitDepth), round0)
		sum += int32(v) * int32(info.VFilter[k])
	}
	return uint16(referenceClamp(referenceRoundPowerOfTwo(sum, round1), 0, max))
}

func referenceWienerHorizontal(src []uint16, stride int, origin int, row int, col int, filter WienerFilter, bitDepth int, round0 int) uint16 {
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	sum := int32(src[origin+row*stride+col])<<WienerFilterBits + int32(1<<(bitDepth+WienerFilterBits-1))
	for k := range WienerWin {
		sum += int32(src[origin+row*stride+col-WienerHalfwin+k]) * int32(filter[k])
	}
	return uint16(referenceClamp(referenceRoundPowerOfTwo(sum, round0), 0, limit-1))
}

func referenceWienerRounds(bitDepth int) (int, int) {
	round0 := 3
	round1 := 11
	if bitDepth == 12 {
		round0 = 5
		round1 = 9
	}
	return round0, round1
}

func referenceRoundPowerOfTwo(v int32, bits int) int32 {
	if bits <= 0 {
		return v
	}
	return (v + int32(1<<(bits-1))) >> bits
}

func referenceClamp(v int32, lo int32, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func fuzzWienerTap(raw int16, min int, max int) int16 {
	span := max - min + 1
	v := int(raw) % span
	if v < 0 {
		v += span
	}
	return int16(min + v)
}
