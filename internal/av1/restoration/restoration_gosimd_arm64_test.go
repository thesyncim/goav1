//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"math/rand"
	"reflect"
	"simd/archsimd"
	"slices"
	"testing"
)

// blendDiffWidths spans the 8-lane vector, its <8 tails, the odd widths that
// exercise the tail fallback, and the 64-wide processing unit.
var blendDiffWidths = []int{1, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 40, 56, 63, 64}

// blendDiffXQ mixes the decode-xq extremes so the projection lands both inside
// [0,max] and well past the int16 wrap boundary (|rounded| > 32767) and the
// clamp edges (0 and max).
var blendDiffXQ = [][2]int32{
	{0, 128}, {31, 0}, {-96, 256}, {56, 72}, {128, -64}, {-128, 255}, {96, -96}, {0, 0},
}

// TestSGRWeightedRowSIMDMatchesReference is the byte-exactness gate for the
// high-bit-depth blend across 8/10/12-bit maxima.
func TestSGRWeightedRowSIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5217)
	for _, maxI := range []int32{255, 1023, 4095} {
		for iter := 0; iter < 20; iter++ {
			for _, width := range blendDiffWidths {
				for _, xq := range blendDiffXQ {
					src := make([]uint16, width)
					for i := range src {
						src[i] = uint16(rnd.pseudoUniform(int(maxI) + 1))
					}
					f0 := make([]int32, width)
					f1 := make([]int32, width)
					for i := range f0 {
						f0[i] = int32(rnd.pseudoUniform(1<<23)) - (1 << 22)
						f1[i] = int32(rnd.pseudoUniform(1<<23)) - (1 << 22)
					}
					want := make([]uint16, width)
					got := make([]uint16, width)
					sgrWeightedRow(want, src, f0, f1, xq[0], xq[1], maxI)
					sgrWeightedRowSIMD(got, src, f0, f1, xq[0], xq[1], maxI)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("u16 max=%d width=%d xq=%v dst[%d]=%d want %d",
								maxI, width, xq, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestSGRBlendDispatchBindsExpectedKernels confirms that the 8-bit and the
// high-bit-depth final projections both bind their Go-native SIMD kernels.
func TestSGRBlendDispatchBindsExpectedKernels(t *testing.T) {
	if reflect.ValueOf(sgrWeightedRowU8Impl).Pointer() != reflect.ValueOf(sgrWeightedRowU8SIMD).Pointer() {
		t.Fatal("sgrWeightedRowU8Impl not bound to the SIMD kernel under goexperiment.simd")
	}
	if reflect.ValueOf(sgrWeightedRowImpl).Pointer() != reflect.ValueOf(sgrWeightedRowSIMD).Pointer() {
		t.Fatal("sgrWeightedRowImpl not bound to the SIMD kernel under goexperiment.simd")
	}
}

// TestSGRBlendSIMDIsZeroAlloc protects the hot-path contract: the staging
// arrays and broadcast constants stay on the stack, no per-call allocation.
func TestSGRBlendSIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5317)
	const width = 64
	src16 := make([]uint16, width)
	dst16 := make([]uint16, width)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		src16[i] = uint16(rnd.pseudoUniform(1024))
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	if allocs := testing.AllocsPerRun(200, func() {
		sgrWeightedRowSIMD(dst16, src16, f0, f1, 12, 116, 1023)
	}); allocs != 0 {
		t.Fatalf("SGR blend SIMD kernels allocated %f times per call", allocs)
	}
}

func benchU8Blend(b *testing.B, fn func([]uint8, []uint8, []int32, []int32, int32, int32)) {
	const width = 64
	rnd := newRestorationRandom(0x9931)
	src := randomU8Plane(rnd, width, 1)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	dst := make([]uint8, width)
	b.ResetTimer()
	for b.Loop() {
		fn(dst, src, f0, f1, 12, 116)
	}
}

func BenchmarkSGRWeightedRowU8_SIMD(b *testing.B)   { benchU8Blend(b, sgrWeightedRowU8SIMD) }
func BenchmarkSGRWeightedRowU8_PureGo(b *testing.B) { benchU8Blend(b, sgrWeightedRowU8) }

func benchU16Blend(b *testing.B, fn func([]uint16, []uint16, []int32, []int32, int32, int32, int32)) {
	const width = 64
	rnd := newRestorationRandom(0x9932)
	src := make([]uint16, width)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		src[i] = uint16(rnd.pseudoUniform(1024))
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	dst := make([]uint16, width)
	b.ResetTimer()
	for b.Loop() {
		fn(dst, src, f0, f1, 12, 116, 1023)
	}
}

func BenchmarkSGRWeightedRowU16_SIMD(b *testing.B)   { benchU16Blend(b, sgrWeightedRowSIMD) }
func BenchmarkSGRWeightedRowU16_PureGo(b *testing.B) { benchU16Blend(b, sgrWeightedRow) }

// Frozen ccec1451 driver bodies retain the unfactored general blend calls.
// They independently pin row scheduling, alias batching and complete scratch.

// frozenSGRGeneric is the self-guided filter with the SIMD blend, used for the
// radius-1 and radius-2 passes. Its A/B coefficients come from the scalar
// calculateIntermediate with the parameter pass selected by the radius index.
func frozenSGRGeneric(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const nb = 5
	const shift = SGRProjSgrBits + nb - SGRProjRstBits
	weights := [9]int32{3, 4, 3, 4, 4, 4, 3, 4, 3}
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		simdCols := width &^ (sgrBlendLanes - 1)
		if simdCols > 0 {
			sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-bufStride-1:], aBuf[k0-1:], aBuf[k0+bufStride-1:],
				bBuf[k0-bufStride-1:], bBuf[k0-1:], bBuf[k0+bufStride-1:], &weights, shift, simdCols)
		}
		aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
		aCur := aBuf[k0-1 : k0+width+1]
		aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
		bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
		for col := simdCols; col < width; col++ {
			j := col + 1
			a := (aCur[j]+aCur[j-1]+aCur[j+1]+aPrev[j]+aNext[j])*4 +
				(aPrev[j-1]+aNext[j-1]+aPrev[j+1]+aNext[j+1])*3
			b := (bCur[j]+bCur[j-1]+bCur[j+1]+bPrev[j]+bNext[j])*4 +
				(bPrev[j-1]+bNext[j-1]+bPrev[j+1]+bNext[j+1])*3
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shift)
		}
	}
}

// frozenSGRGenericFast is the radius-2 self-guided filter with the SIMD blend. It
// evaluates the box-sum stencil only on alternating rows (the "fast" subsampled
// variant), so even and odd rows use different weight sets.
func frozenSGRGenericFast(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const shiftEven = SGRProjSgrBits + 5 - SGRProjRstBits
	const shiftOdd = SGRProjSgrBits + 4 - SGRProjRstBits
	evenWeights := [9]int32{5, 6, 5, 0, 0, 0, 5, 6, 5}
	oddWeights := [9]int32{0, 0, 0, 5, 6, 5, 0, 0, 0}
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		simdCols := width &^ (sgrBlendLanes - 1)
		if row&1 == 0 {
			if simdCols > 0 {
				sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-bufStride-1:], aBuf[k0-1:], aBuf[k0+bufStride-1:],
					bBuf[k0-bufStride-1:], bBuf[k0-1:], bBuf[k0+bufStride-1:], &evenWeights, shiftEven, simdCols)
			}
			aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
			aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
			bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
			bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
			for col := simdCols; col < width; col++ {
				j := col + 1
				a := (aPrev[j]+aNext[j])*6 +
					(aPrev[j-1]+aPrev[j+1]+aNext[j-1]+aNext[j+1])*5
				b := (bPrev[j]+bNext[j])*6 +
					(bPrev[j-1]+bPrev[j+1]+bNext[j-1]+bNext[j+1])*5
				dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftEven)
			}
			continue
		}
		if simdCols > 0 {
			sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-1:], aBuf[k0-1:], aBuf[k0-1:],
				bBuf[k0-1:], bBuf[k0-1:], bBuf[k0-1:], &oddWeights, shiftOdd, simdCols)
		}
		aCur := aBuf[k0-1 : k0+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		for col := simdCols; col < width; col++ {
			j := col + 1
			a := aCur[j]*6 + (aCur[j-1]+aCur[j+1])*5
			b := bCur[j]*6 + (bCur[j-1]+bCur[j+1])*5
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftOdd)
		}
	}
}

// Use the same eight shapes as the existing rolling/full oracle. Compare every
// A/B cell, including poison in skipped rows and padding, to the untouched
// scalar rolling implementation. Full-width int32 values exercise wrap points
// independently of legal 8/10/12-bit image inputs.
func TestSGRCoefficientRollingMatchesScalar(t *testing.T) {
	params := append([]SGRParams(nil), SGRParameterTable[:]...)
	params = append(params,
		SGRParams{Radius: [2]uint8{2, 1}, S: [2]int16{-32768, 32767}},
		SGRParams{Radius: [2]uint8{1, 2}, S: [2]int16{0, -1}})
	for _, shape := range [][2]int{{1, 1}, {2, 3}, {3, 2}, {7, 9}, {16, 8}, {31, 33}, {63, 64}, {64, 63}} {
		w, h := shape[0], shape[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, depth := range []int{8, 10, 12} {
			for _, fullWidth := range []bool{false, true} {
				for i := range dgd {
					dgd[i] = int32(uint32(i)*2654435761 + 0x80000000)
					if !fullWidth {
						dgd[i] &= (1 << depth) - 1
					}
				}
				for pi, p := range params {
					for ri, radius := range p.Radius {
						if radius == 0 {
							continue
						}
						for pass := 0; pass < 2; pass++ {
							ga, gb := make([]int32, bl+7), make([]int32, bl+7)
							for i := range ga {
								ga[i] = int32(uint32(i)*2654435761 + 0xfedc)
								gb[i] = ^ga[i]
							}
							wa, wb := slices.Clone(ga), slices.Clone(gb)
							calculateIntermediateRolling(dgd, origin, w, h, stride, depth, p, ri, pass, wa[:bl], wb[:bl], bs)
							calculateIntermediateRollingSIMD(dgd, origin, w, h, stride, depth, p, ri, pass, ga[:bl], gb[:bl], bs)
							if !slices.Equal(ga, wa) || !slices.Equal(gb, wb) {
								t.Fatalf("shape=%v depth=%d fullWidth=%v params=%d radius=%d pass=%d", shape, depth, fullWidth, pi, radius, pass)
							}
						}
					}
				}
			}
		}
	}
}

// Reuses the preserved two-size/five-offset alias schedule, comparing complete
// backing and scratch to the original driver at 8, 10 and 12 bits. The frozen generic
// drivers retain the independent scalar coefficient producer.
func TestSGRCoefficientFrozenBacking(t *testing.T) {
	oldNormal, oldFast := selfguidedImpl, selfguidedFastImpl
	defer func() { selfguidedImpl, selfguidedFastImpl = oldNormal, oldFast }()
	for _, shape := range [][2]int{{7, 9}, {64, 64}} {
		w, h := shape[0], shape[1]
		stride := w + 11
		origin := 3*stride + 4
		need, _ := SelfguidedScratchLen(w, h)
		for _, depth := range []uint8{8, 10, 12} {
			for _, delta := range []int{0, 1, -1, stride, -stride} {
				for pi := range SGRParameterTable {
					got := make([]uint16, stride*(h+8)+13)
					for i := range got {
						got[i] = uint16(i*79+31) & uint16((1<<depth)-1)
					}
					want := slices.Clone(got)
					gs := make([]int32, need+11)
					for i := range gs {
						gs[i] = int32(uint32(i)*2654435761 + 0x91a7)
					}
					ws := slices.Clone(gs)
					selfguidedImpl, selfguidedFastImpl = frozenSGRGeneric, frozenSGRGenericFast
					we := ApplySelfguidedRestoration(want, stride, origin, want[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, depth, ws[:need])
					selfguidedImpl, selfguidedFastImpl = selfguidedSIMD, selfguidedFastSIMD
					ge := ApplySelfguidedRestoration(got, stride, origin, got[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, depth, gs[:need])
					if ge != we || !slices.Equal(got, want) || !slices.Equal(gs, ws) {
						t.Fatalf("wide backing shape%v depth%d delta%d pi%d err%v/%v", shape, depth, delta, pi, ge, we)
					}
					if depth != 8 {
						continue
					}
					gb := make([]byte, len(got))
					for i := range gb {
						gb[i] = byte(i*79 + 31)
					}
					wb := slices.Clone(gb)
					for i := range gs {
						gs[i] = int32(uint32(i)*2654435761 + 0x91a7)
					}
					copy(ws, gs)
					selfguidedImpl, selfguidedFastImpl = frozenSGRGeneric, frozenSGRGenericFast
					we = ApplySelfguidedRestorationU8(wb, stride, origin, wb[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, ws[:need])
					selfguidedImpl, selfguidedFastImpl = selfguidedSIMD, selfguidedFastSIMD
					ge = ApplySelfguidedRestorationU8(gb, stride, origin, gb[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, gs[:need])
					if ge != we || !slices.Equal(gb, wb) || !slices.Equal(gs, ws) {
						t.Fatalf("byte backing shape%v delta%d pi%d err%v/%v", shape, delta, pi, ge, we)
					}
				}
			}
		}
	}
}

func TestSGRU8XLookupMatchesTable(t *testing.T) {
	for base := 0; base < 256; base += 16 {
		var z [16]uint8
		for i := range z {
			z[i] = uint8(base + i)
		}
		var got [16]uint8
		sgrU8X(archsimd.LoadUint8x16Array(&z)).StoreArray(&got)
		for i, v := range got {
			if want := SGRProjSgr - int(xByXPlus1[base+i]); int(v) != want {
				t.Fatalf("x(%d) = %d, want %d", base+i, v, want)
			}
		}
	}
}

// sgrU8ReferencePasses is the int32 path ApplySelfguidedRestorationU8 runs
// without the vector passes.
func sgrU8ReferencePasses(src []uint8, srcStride, srcOrigin, width, height, paramsIndex int, flt0, flt1 []int32) {
	dgdStride := width + 2*SGRProjBorderHorz
	dgd := make([]int32, dgdStride*(height+2*SGRProjBorderVert))
	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		for col := -SGRProjBorderHorz; col < width+SGRProjBorderHorz; col++ {
			dgd[dgdOrigin+row*dgdStride+col] = int32(src[srcOrigin+row*srcStride+col])
		}
	}
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)
	aBuf := make([]int32, bufLen)
	bBuf := make([]int32, bufLen)
	params := SGRParameterTable[paramsIndex]
	if params.Radius[0] > 0 {
		selfguidedFast(dgd, dgdOrigin, width, height, dgdStride, flt0, width, 8, paramsIndex, 0, aBuf, bBuf, bufStride)
	}
	if params.Radius[1] > 0 {
		selfguided(dgd, dgdOrigin, width, height, dgdStride, flt1, width, 8, paramsIndex, 1, aBuf, bBuf, bufStride)
	}
}

func TestSGRU8PassesMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	sizes := [][2]int{{64, 64}, {1, 1}, {2, 3}, {7, 5}, {8, 8}, {15, 17}, {16, 1}, {33, 31}, {63, 64}, {64, 63}, {48, 64}, {64, 2}}
	for i := 0; i < 40; i++ {
		sizes = append(sizes, [2]int{1 + rng.Intn(64), 1 + rng.Intn(64)})
	}
	fills := []string{"random", "zero", "max", "checker", "ramp"}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for _, fill := range fills {
			stride := w + 2*SGRProjBorderHorz + rng.Intn(5)
			rows := h + 2*SGRProjBorderVert
			src := make([]uint8, stride*rows)
			for i := range src {
				switch fill {
				case "random":
					src[i] = uint8(rng.Intn(256))
				case "max":
					src[i] = 255
				case "checker":
					if (i/stride+i%stride)&1 != 0 {
						src[i] = 255
					}
				case "ramp":
					src[i] = uint8(i * 37)
				}
			}
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			for p := range SGRProjParams {
				want0 := make([]int32, w*h)
				want1 := make([]int32, w*h)
				sgrU8ReferencePasses(src, stride, origin, w, h, p, want0, want1)
				got0 := make([]int32, w*h)
				got1 := make([]int32, w*h)
				if !sgrU8Passes(src, stride, origin, w, h, SGRParameterTable[p], got0, got1) {
					t.Fatalf("%dx%d params %d: vector passes declined", w, h, p)
				}
				for i := range want0 {
					if SGRParameterTable[p].Radius[0] != 0 && got0[i] != want0[i] {
						t.Fatalf("%dx%d %s params %d: flt0[%d,%d] = %d, want %d", w, h, fill, p, i/w, i%w, got0[i], want0[i])
					}
					if SGRParameterTable[p].Radius[1] != 0 && got1[i] != want1[i] {
						t.Fatalf("%dx%d %s params %d: flt1[%d,%d] = %d, want %d", w, h, fill, p, i/w, i%w, got1[i], want1[i])
					}
				}
			}
		}
	}
}

func TestSGRU8PassesZeroAlloc(t *testing.T) {
	w, h := 64, 64
	stride := w + 2*SGRProjBorderHorz
	src := make([]uint8, stride*(h+2*SGRProjBorderVert))
	for i := range src {
		src[i] = uint8(i * 13)
	}
	flt0 := make([]int32, w*h)
	flt1 := make([]int32, w*h)
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	if n := testing.AllocsPerRun(10, func() {
		sgrU8Passes(src, stride, origin, w, h, SGRParameterTable[0], flt0, flt1)
	}); n != 0 {
		t.Fatalf("allocs = %v", n)
	}
}

func BenchmarkSGRU8Passes(b *testing.B) {
	w, h := 64, 64
	stride := w + 2*SGRProjBorderHorz
	rng := rand.New(rand.NewSource(1))
	src := make([]uint8, stride*(h+2*SGRProjBorderVert))
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	dst := make([]uint8, w*h)
	n, err := SelfguidedScratchLen(w, h)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]int32, n)
	b.Run("vector", func(b *testing.B) {
		for b.Loop() {
			if err := ApplySelfguidedRestorationU8(src, stride, origin, dst, w, w, h, 0, [2]int8{-32, 31}, scratch); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("int32", func(b *testing.B) {
		flt0 := make([]int32, w*h)
		flt1 := make([]int32, w*h)
		for b.Loop() {
			sgrU8ReferencePassesNoAlloc(src, stride, origin, w, h, 0, flt0, flt1, scratch)
		}
	})
}

// sgrU8ReferencePassesNoAlloc is sgrU8ReferencePasses on caller scratch (the
// dispatched int32 kernels), for the benchmark baseline.
func sgrU8ReferencePassesNoAlloc(src []uint8, srcStride, srcOrigin, width, height, paramsIndex int, flt0, flt1 []int32, scratch []int32) {
	dgdStride := width + 2*SGRProjBorderHorz
	dgdLen := dgdStride * (height + 2*SGRProjBorderVert)
	dgd := scratch[:dgdLen]
	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		s := src[srcOrigin+row*srcStride-SGRProjBorderHorz:]
		d := dgd[dgdOrigin+row*dgdStride-SGRProjBorderHorz:]
		for col := 0; col < width+2*SGRProjBorderHorz; col++ {
			d[col] = int32(s[col])
		}
	}
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)
	ab := scratch[dgdLen+2*width*height:]
	selfguidedFastImpl(dgd, dgdOrigin, width, height, dgdStride, flt0, width, 8, paramsIndex, 0, ab[:bufLen], ab[bufLen:2*bufLen], bufStride)
	selfguidedImpl(dgd, dgdOrigin, width, height, dgdStride, flt1, width, 8, paramsIndex, 1, ab[:bufLen], ab[bufLen:2*bufLen], bufStride)
}

// wienerSIMDDetected and sgrSIMDDetected report whether init bound the SIMD
// Wiener and self-guided kernels (NEON is mandatory on arm64).
func wienerSIMDDetected() bool { return cpu.Detected.NEON }

func sgrSIMDDetected() bool { return cpu.Detected.NEON }

// sgrSIMDBoxsumKernel is the box-sum kernel bound alongside the SIMD
// self-guided drivers: the separable running sum on arm64.
var sgrSIMDBoxsumKernel = boxsumSeparable

// arm64 routes five-tap filters to dedicated kernels through these wrappers;
// the seven-tap SIMD kernels sit behind them.
var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDTaps
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMDTaps
)

// TestWienerHorizontalU8Dav1d5AllFilters runs every five-tap filter AV1 can
// code (f1 in [-23, 8], f2 in [-17, 46], f3 = -2*(f1+f2)) over sample patterns
// that drive the int16 sums and the [0, 8191] clamp to both ends, against the
// scalar reference.
func TestWienerHorizontalU8Dav1d5AllFilters(t *testing.T) {
	const width, height = 32, 2
	stride := width + 2*WienerHalfwin + 16
	origin := WienerHalfwin*stride + WienerHalfwin
	rows := height + 2*WienerHalfwin + 1
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xd5)
	patterns := make([][]uint8, 0, 8)
	for p := range 8 {
		src := make([]uint8, stride*rows)
		for i := range src {
			x := i % stride
			switch p {
			case 0:
				src[i] = 0
			case 1:
				src[i] = 255
			case 2:
				src[i] = uint8(255 * (x & 1))
			case 3:
				src[i] = uint8(255 * ((x >> 1) & 1))
			case 4:
				if x%5 == 3 {
					src[i] = 255
				}
			case 5:
				if x%5 != 3 {
					src[i] = 255
				}
			default:
				src[i] = uint8(rnd.pseudoUniform(256))
			}
		}
		patterns = append(patterns, src)
	}
	round0, _ := wienerRounds(8)
	tempLen := width * (height + 2*WienerHalfwin)
	want := make([]uint16, tempLen)
	got := make([]uint16, tempLen)
	for f1 := -23; f1 <= 8; f1++ {
		for f2 := -17; f2 <= 46; f2++ {
			f3 := -2 * (f1 + f2)
			filter := WienerFilter{0, int16(f1), int16(f2), int16(f3), int16(f2), int16(f1), 0, 0}
			if !wienerU8Dav1dTapsFit(filter) {
				t.Fatalf("coded filter f1=%d f2=%d does not fit", f1, f2)
			}
			for pi, src := range patterns {
				wienerHorizontalU8(src, stride, origin, width, height, filter, round0, want)
				wienerHorizontalU8Dav1d5(src, stride, origin, width, height, filter, got)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("f1=%d f2=%d pattern %d: temp[%d] = %d, want %d", f1, f2, pi, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkWienerHorizontalU8Five(b *testing.B) {
	const width, height = 64, 64
	stride := width + 2*WienerHalfwin + 16
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint8, stride*(height+2*WienerHalfwin+1))
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint8(rnd.pseudoUniform(256))
	}
	filter := WienerFilter{0, -7, 21, -28, 21, -7, 0, 0}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	b.Run("dav1d", func(b *testing.B) {
		for b.Loop() {
			wienerHorizontalU8Dav1d5(src, stride, origin, width, height, filter, temp)
		}
	})
	b.Run("packed", func(b *testing.B) {
		for b.Loop() {
			wienerHorizontalU8SIMD5(src, stride, origin, width, height, filter, WienerRound0Bits, temp)
		}
	})
}

// TestWienerU8FiveTapMatchesReference checks the five-tap kernels against the
// scalar references on symmetric filters whose outer tap pair is zero, and
// that the dispatch really routes them to the five-tap kernels (a zeroed f[7]
// alone would leave the filter asymmetric and silently take the scalar path).
func TestWienerU8FiveTapMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5a5)
	round0, round1 := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {64, 64}, {62, 12}, {24, 7}, {48, 3}}, u8DiffSizes...)
	for _, sz := range sizes {
		for fi, info := range wienerDispatchFilters() {
			hf, vf := info.HFilter, info.VFilter
			hf[0], hf[WienerSubpelTaps-2] = 0, 0
			vf[0], vf[WienerSubpelTaps-2] = 0, 0
			if !wienerFilterFiveTap(hf) || !wienerFilterSymmetric(hf) || !wienerFilterFiveTap(vf) {
				t.Fatalf("filter %d is not a symmetric five-tap filter", fi)
			}
			stride := sz.width + 2*WienerHalfwin + 32
			origin := WienerHalfwin*stride + WienerHalfwin + 8
			src := make([]uint8, stride*(sz.height+2*WienerHalfwin+4))
			for i := range src {
				src[i] = uint8(rnd.pseudoUniform(256))
			}
			if sz.width >= 8 && sz.width%8 == 0 && !wienerHorizontalU8SIMDCanRun(len(src), stride, origin, sz.width, sz.height) {
				t.Fatalf("sz=%dx%d: SIMD horizontal cannot run; test geometry too tight", sz.width, sz.height)
			}
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			want := make([]uint16, tempLen)
			got := make([]uint16, tempLen)
			wienerHorizontalU8(src, stride, origin, sz.width, sz.height, hf, round0, want)
			wienerHorizontalU8SIMDTaps(src, stride, origin, sz.width, sz.height, hf, round0, got)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("horizontal sz=%dx%d f=%d temp[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
			dstStride := sz.width + 5
			wantDst := make([]uint8, dstStride*sz.height)
			gotDst := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(want, sz.width, wantDst, dstStride, sz.width, sz.height, vf, round1)
			wienerVerticalU8SIMDTaps(want, sz.width, gotDst, dstStride, sz.width, sz.height, vf, round1)
			for i := range wantDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("vertical sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

func TestWienerVerticalPackedEdgesAndTails(t *testing.T) {
	filters := []WienerFilter{
		NewWienerFilter(WienerTap0Min, WienerTap1Min, WienerTap2Min),
		NewWienerFilter(WienerTap0Max, WienerTap1Max, WienerTap2Max),
		DefaultWienerInfo().VFilter,
	}
	for _, info := range wienerDispatchFilters() {
		filters = append(filters, info.VFilter)
	}
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		maxClamp := uint16((1 << (bitDepth + 1 + WienerFilterBits - round0)) - 1)
		for _, width := range []int{1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 64} {
			const height = 5
			tempStride := width + 3
			dstStride := width + 2
			temp := make([]uint16, tempStride*(height+2*WienerHalfwin))
			for i := range temp {
				if i%3 != 0 {
					temp[i] = maxClamp
				}
			}
			for _, filter := range filters {
				want := make([]uint16, dstStride*height)
				got := make([]uint16, dstStride*height)
				wienerVertical(temp, tempStride, want, dstStride, width, height, filter, bitDepth, round1, max)
				wienerVerticalSIMD(temp, tempStride, got, dstStride, width, height, filter, bitDepth, round1, max)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("u16 depth=%d width=%d filter=%v index=%d got=%d want=%d", bitDepth, width, filter, i, got[i], want[i])
					}
				}
				if bitDepth == 8 {
					want8 := make([]uint8, dstStride*height)
					got8 := make([]uint8, dstStride*height)
					wienerVerticalU8(temp, tempStride, want8, dstStride, width, height, filter, round1)
					wienerVerticalU8SIMD(temp, tempStride, got8, dstStride, width, height, filter, round1)
					for i := range want8 {
						if got8[i] != want8[i] {
							t.Fatalf("u8 width=%d filter=%v index=%d got=%d want=%d", width, filter, i, got8[i], want8[i])
						}
					}
				}
			}
		}
	}
}

func BenchmarkWienerVerticalU8Shape(b *testing.B) {
	_, round1 := wienerRounds(8)
	filter := DefaultWienerInfo().VFilter
	for _, shape := range [][2]int{{8, 8}, {16, 8}, {32, 8}, {64, 8}, {64, 64}} {
		width, height := shape[0], shape[1]
		b.Run(fmt.Sprintf("%dx%d", width, height), func(b *testing.B) {
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			for i := range temp {
				temp[i] = uint16((i*97 + 31) % 8192)
			}
			dst := make([]uint8, width*height)
			b.ReportAllocs()
			for b.Loop() {
				wienerVerticalU8SIMD(temp, width, dst, width, width, height, filter, round1)
			}
		})
	}
}

func BenchmarkWienerVerticalU16Shape(b *testing.B) {
	_, round1 := wienerRounds(12)
	filter := DefaultWienerInfo().VFilter
	for _, shape := range [][2]int{{8, 8}, {16, 8}, {32, 8}, {64, 8}, {64, 64}} {
		width, height := shape[0], shape[1]
		b.Run(fmt.Sprintf("%dx%d", width, height), func(b *testing.B) {
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			for i := range temp {
				temp[i] = uint16((i*97 + 31) % 32768)
			}
			dst := make([]uint16, width*height)
			b.ReportAllocs()
			for b.Loop() {
				wienerVerticalSIMD(temp, width, dst, width, width, height, filter, 12, round1, 4095)
			}
		})
	}
}

func BenchmarkWienerHorizontalU16Shape(b *testing.B) {
	round0, _ := wienerRounds(12)
	filter := DefaultWienerInfo().HFilter
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			const height = 64
			stride := width + 2*WienerHalfwin + 2
			origin := WienerHalfwin*stride + WienerHalfwin
			src := make([]uint16, stride*(height+2*WienerHalfwin))
			for i := range src {
				src[i] = uint16((i*97 + 31) % 4096)
			}
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			b.ReportAllocs()
			for b.Loop() {
				wienerHorizontalSIMDTrusted(src, stride, origin, width, height, filter, 12, round0, 4095, temp)
			}
		})
	}
}
