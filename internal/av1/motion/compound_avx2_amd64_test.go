// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// These differential tests assert the AVX2 8-bit compound kernels are
// bit-identical to the pure-Go references for every width/height and every
// subpel phase of each filter type. They call the AVX2 wrappers directly rather
// than through the dispatch slots, so they validate the asm even on hosts whose
// CPUID does not advertise AVX2 (e.g. amd64 under Rosetta 2, which translates
// AVX2 anyway).

func compoundAVX2Sizes() []struct{ w, h int } {
	return []struct{ w, h int }{
		{8, 1}, {8, 8}, {16, 7}, {16, 16}, {24, 3}, {32, 11}, {48, 5}, {64, 16},
	}
}

func compoundRoundParams8() (round0, offsetBits, roundOffset, roundBits int) {
	round0 = compoundRound0(8)
	offsetBits = 8 + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits = 2*filterBits - round0 - compoundRound1Bits
	return
}

func TestBlendCompoundAvg8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB1E0D))
	_, _, roundOffset, roundBits := compoundRoundParams8()
	weights := [][2]int{{8, 8}, {9, 7}, {13, 3}, {4, 12}, {16, 0}, {0, 16}}
	for _, sz := range compoundAVX2Sizes() {
		src0 := make([]uint16, sz.w*sz.h)
		src1 := make([]uint16, sz.w*sz.h)
		for i := range src0 {
			src0[i] = uint16(rng.Intn(1 << 12))
			src1[i] = uint16(rng.Intn(1 << 12))
		}
		for _, w := range weights {
			const stride = 96
			got := frame.Plane{Pix: make([]byte, stride*(sz.h+2)), Stride: stride, Width: sz.w + 4, Height: sz.h + 2}
			want := frame.Plane{Pix: make([]byte, stride*(sz.h+2)), Stride: stride, Width: sz.w + 4, Height: sz.h + 2}
			blendCompoundAvg8AVX2(got, src0, src1, 2, 1, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			blendCompoundAvg8PureGo(want, src0, src1, 2, 1, sz.w, sz.h, w[0], w[1], roundOffset, roundBits)
			for i := range got.Pix {
				if got.Pix[i] != want.Pix[i] {
					t.Fatalf("blend %dx%d w=%v byte %d: AVX2=%d PureGo=%d", sz.w, sz.h, w, i, got.Pix[i], want.Pix[i])
				}
			}
		}
	}
}

func TestCompound8CopyAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE))
	round0, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, sz := range compoundAVX2Sizes() {
		side := sz.w
		if sz.h > side {
			side = sz.h
		}
		ref := randPlane(rng, side+2*pad, 1)
		got := make([]uint16, sz.w*sz.h)
		want := make([]uint16, sz.w*sz.h)
		predictInterCompoundRef8ToConvBufCopyAVX2(got, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
		predictInterCompoundRef8ToConvBufCopyPureGo(want, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("copy %dx%d sample %d: AVX2=%d PureGo=%d", sz.w, sz.h, i, got[i], want[i])
			}
		}
	}
}

func TestCompound8XAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA11CE))
	_, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, tbl := range avx2FilterTables() {
		for ph := 1; ph < 16; ph++ {
			k := tbl[ph]
			for _, sz := range compoundAVX2Sizes() {
				side := sz.w
				if sz.h > side {
					side = sz.h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				predictInterCompoundRef8ToConvBufXAVX2(got, ref, pad, pad, sz.w, sz.h, k, roundOffset)
				predictInterCompoundRef8ToConvBufXPureGo(want, ref, pad, pad, sz.w, sz.h, k, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("X %dx%d ph=%d sample %d: AVX2=%d PureGo=%d", sz.w, sz.h, ph, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompound8YAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB0B))
	round0, _, roundOffset, _ := compoundRoundParams8()
	const pad = filterTaps
	for _, tbl := range avx2FilterTables() {
		for ph := 1; ph < 16; ph++ {
			k := tbl[ph]
			for _, sz := range compoundAVX2Sizes() {
				side := sz.w
				if sz.h > side {
					side = sz.h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				predictInterCompoundRef8ToConvBufYAVX2(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
				predictInterCompoundRef8ToConvBufYPureGo(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("Y %dx%d ph=%d sample %d: AVX2=%d PureGo=%d", sz.w, sz.h, ph, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompound82DAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2D2D))
	_, offsetBits, _, _ := compoundRoundParams8()
	const pad = filterTaps
	tables := avx2FilterTables()
	for xi, xtbl := range tables {
		for xph := 1; xph < 16; xph++ {
			xk := xtbl[xph]
			ytbl := tables[(xi+1)%len(tables)]
			for _, yph := range []int{1, 5, 9, 15} {
				yk := ytbl[yph]
				for _, sz := range compoundAVX2Sizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRef8ToConvBuf2DAVX2(got, ref, pad, pad, sz.w, sz.h, xk, yk, offsetBits, nil)
					predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, pad, pad, sz.w, sz.h, xk, yk, offsetBits, nil)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("2D %dx%d xph=%d yph=%d sample %d: AVX2=%d PureGo=%d", sz.w, sz.h, xph, yph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// The clamped (edge-overhanging) 2D path materializes a halo via emu_edge and
// reruns the resident kernel; assert it stays bit-identical to the per-tap
// clamping pure-Go reference at every block edge.
func TestCompound82DAVX2EdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xED9E))
	_, offsetBits, _, _ := compoundRoundParams8()
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	const refW, refH, stride = 40, 24, 64
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	var scGot, scWant CompoundConvolveScratch
	for _, sz := range []struct{ w, h int }{{8, 8}, {16, 8}, {8, 16}, {16, 16}} {
		for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}} {
			got := make([]uint16, sz.w*sz.h)
			want := make([]uint16, sz.w*sz.h)
			predictInterCompoundRef8ToConvBuf2DAVX2(got, ref, org[0], org[1], sz.w, sz.h, xk, yk, offsetBits, &scGot)
			predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, org[0], org[1], sz.w, sz.h, xk, yk, offsetBits, &scWant)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("2D edge %dx%d org=%v sample %d: AVX2=%d PureGo=%d", sz.w, sz.h, org, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompound8AVX2ZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA110C))
	round0, offsetBits, roundOffset, roundBits := compoundRoundParams8()
	const pad = filterTaps
	const w, h = 32, 32
	ref := randPlane(rng, w+2*pad, 1)
	out := make([]uint16, w*h)
	src0 := make([]uint16, w*h)
	src1 := make([]uint16, w*h)
	dst := frame.Plane{Pix: make([]byte, 96*(h+2)), Stride: 96, Width: w + 4, Height: h + 2}
	k := avx2FilterTables()[0][7]
	var scratch CompoundConvolveScratch
	if a := testing.AllocsPerRun(20, func() {
		blendCompoundAvg8AVX2(dst, src0, src1, 0, 0, w, h, 9, 7, roundOffset, roundBits)
		predictInterCompoundRef8ToConvBufCopyAVX2(out, ref, pad, pad, w, h, round0, roundOffset)
		predictInterCompoundRef8ToConvBufXAVX2(out, ref, pad, pad, w, h, k, roundOffset)
		predictInterCompoundRef8ToConvBufYAVX2(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRef8ToConvBuf2DAVX2(out, ref, pad, pad, w, h, k, k, offsetBits, &scratch)
	}); a != 0 {
		t.Fatalf("8-bit compound AVX2 allocates: %v allocs/run", a)
	}
}
