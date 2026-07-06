// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

import (
	"errors"
	"testing"
)

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
