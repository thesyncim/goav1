// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

import "testing"

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
				okWant := wienerHorizontal(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, wantTemp)
				okGot := wienerHorizontalImpl(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, gotTemp)
				if okWant != okGot {
					t.Fatalf("bd=%d sz=%dx%d f=%d horizontal ok mismatch: want=%v got=%v", bitDepth, sz.width, sz.height, fi, okWant, okGot)
				}
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

// TestWienerDispatchRejectsOutOfRange verifies that the dispatched horizontal
// pass reproduces the reference's validity bail-out (sample > max => false).
func TestWienerDispatchRejectsOutOfRange(t *testing.T) {
	const width, height = 64, 8
	const bitDepth uint8 = 8
	max := uint16((1 << bitDepth) - 1)
	round0, _ := wienerRounds(int(bitDepth))
	stride := width + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(height+2*WienerHalfwin))
	for i := range src {
		src[i] = uint16(i % int(max+1))
	}
	src[origin+2*stride+5] = max + 1 // poison one interior sample
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	if wienerHorizontalImpl(src, stride, origin, width, height, DefaultWienerInfo().HFilter, int(bitDepth), round0, max, temp) {
		t.Fatal("dispatch accepted out-of-range sample")
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
		wienerHorizontalImpl(src, stride, origin, width, height, info.HFilter, int(bitDepth), round0, max, temp)
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

func BenchmarkWienerHorizontalPureGo(b *testing.B) {
	src, temp, _, stride, origin, info := benchWienerSetup()
	round0, _ := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	b.ReportAllocs()
	for b.Loop() {
		wienerHorizontal(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	}
}

func BenchmarkWienerHorizontalDispatch(b *testing.B) {
	src, temp, _, stride, origin, info := benchWienerSetup()
	round0, _ := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	b.ReportAllocs()
	for b.Loop() {
		wienerHorizontalImpl(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
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
