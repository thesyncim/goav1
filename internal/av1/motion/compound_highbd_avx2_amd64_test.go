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

// Differential tests for the AVX2 high-bit-depth compound kernels. They call
// the AVX2 wrappers directly so the asm executes (not skips) under GOARCH=amd64
// on a Rosetta host.

func compoundHBDParams(bitDepth uint8) (round0, offsetBits, roundOffset int) {
	round0 = compoundRound0(bitDepth)
	offsetBits = int(bitDepth) + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	return
}

func compoundHBDSizes() []struct{ w, h int } {
	return []struct{ w, h int }{
		{4, 4}, {4, 13}, {8, 1}, {8, 8}, {12, 3}, {16, 7}, {32, 11}, {64, 16},
	}
}

func TestCompoundHBDCopyAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE01))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, sz := range compoundHBDSizes() {
			side := sz.w
			if sz.h > side {
				side = sz.h
			}
			ref := randPlaneHBD(rng, side+2*pad, max)
			got := make([]uint16, sz.w*sz.h)
			want := make([]uint16, sz.w*sz.h)
			predictInterCompoundRefHighBDToConvBufCopyResidentAVX2(got, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(want, ref, pad, pad, sz.w, sz.h, round0, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("copy bd=%d %dx%d sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompoundHBDXAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE02))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, tbl := range avx2FilterTables() {
			for ph := 1; ph < 16; ph++ {
				k := tbl[ph]
				for _, sz := range compoundHBDSizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRefHighBDToConvBufXResidentAVX2(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufXResident(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("X bd=%d %dx%d ph=%d sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, ph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBDYAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE03))
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		for _, tbl := range avx2FilterTables() {
			for ph := 1; ph < 16; ph++ {
				k := tbl[ph]
				for _, sz := range compoundHBDSizes() {
					side := sz.w
					if sz.h > side {
						side = sz.h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					got := make([]uint16, sz.w*sz.h)
					want := make([]uint16, sz.w*sz.h)
					predictInterCompoundRefHighBDToConvBufYResidentAVX2(got, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufYResident(want, ref, pad, pad, sz.w, sz.h, k, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("Y bd=%d %dx%d ph=%d sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, ph, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBD2DAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE04))
	const pad = filterTaps
	tables := avx2FilterTables()
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, offsetBits, _ := compoundHBDParams(bd)
		for xi, xtbl := range tables {
			for xph := 1; xph < 16; xph++ {
				xk := xtbl[xph]
				ytbl := tables[(xi+1)%len(tables)]
				for _, yph := range []int{1, 5, 9, 15} {
					yk := ytbl[yph]
					for _, sz := range compoundHBDSizes() {
						side := sz.w
						if sz.h > side {
							side = sz.h
						}
						ref := randPlaneHBD(rng, side+2*pad, max)
						got := make([]uint16, sz.w*sz.h)
						want := make([]uint16, sz.w*sz.h)
						var imG, imW compoundIM
						predictInterCompoundRefHighBDToConvBuf2DResidentAVX2(got, ref, pad, pad, sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG)
						predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, pad, pad, sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imW)
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("2D bd=%d %dx%d xph=%d yph=%d sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, xph, yph, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestCompoundHBD2DClampedAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE05))
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, offsetBits, _ := compoundHBDParams(bd)
		const refW, refH = 40, 24
		stride := refW * 2
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
		for _, sz := range []struct{ w, h int }{{4, 4}, {8, 8}, {16, 8}, {8, 16}} {
			for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}} {
				got := make([]uint16, sz.w*sz.h)
				want := make([]uint16, sz.w*sz.h)
				var imG, imW compoundIM
				var edge emuEdge16Buf
				for i := range edge {
					edge[i] = 0xa5
				}
				gotEdge := make([]uint16, sz.w*sz.h)
				predictInterCompoundRefHighBDToConvBuf2DClampedAVX2(got, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG, nil)
				predictInterCompoundRefHighBDToConvBuf2DClampedAVX2(gotEdge, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imG, &edge)
				predictInterCompoundRefHighBDToConvBuf2DClamped(want, ref, org[0], org[1], sz.w, sz.h, xk, yk, round0, offsetBits, int(bd), &imW)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("2D clamped bd=%d %dx%d org=%v sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, org, i, got[i], want[i])
					}
					if gotEdge[i] != want[i] {
						t.Fatalf("2D clamped (edge) bd=%d %dx%d org=%v sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotEdge[i], want[i])
					}
				}
			}
		}
	}
}

// TestCompoundHBDXYClampedEmuEdgeAVX2MatchesPureGo runs the AVX2 X/Y resident
// kernels over emu_edge halo windows of edge-overhanging blocks (the compound
// X/Y clamped fast path) and asserts bit-identity with the pure-Go per-tap
// clamping references.
func TestCompoundHBDXYClampedEmuEdgeAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE07))
	tables := avx2FilterTables()
	xk := tables[2][11]
	yk := tables[0][7]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << int(bd)) - 1)
		round0, _, roundOffset := compoundHBDParams(bd)
		const refW, refH = 40, 24
		stride := refW * 2
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
		for _, sz := range []struct{ w, h int }{{4, 4}, {8, 8}, {16, 8}, {8, 16}, {12, 4}, {64, 32}} {
			for _, org := range [][2]int{{-3, -2}, {refW - sz.w + 2, 1}, {1, refH - sz.h + 3}, {refW - 4, refH - 4}, {-sz.w, -sz.h}} {
				gotX := make([]uint16, sz.w*sz.h)
				gotY := make([]uint16, sz.w*sz.h)
				wantX := make([]uint16, sz.w*sz.h)
				wantY := make([]uint16, sz.w*sz.h)
				var edge emuEdge16Buf
				for i := range edge {
					edge[i] = 0xa5
				}
				emu, emuX := emuEdgeWindow16X(ref, org[0], org[1], sz.w, sz.h, &edge)
				predictInterCompoundRefHighBDToConvBufXResidentAVX2(gotX, emu, emuX, 0, sz.w, sz.h, xk, round0, roundOffset)
				predictInterCompoundRefHighBDToConvBufXClamped(wantX, ref, org[0], org[1], sz.w, sz.h, xk, round0, roundOffset)
				for i := range edge {
					edge[i] = 0x5a
				}
				emu, emuY := emuEdgeWindow16Y(ref, org[0], org[1], sz.w, sz.h, &edge)
				predictInterCompoundRefHighBDToConvBufYResidentAVX2(gotY, emu, 0, emuY, sz.w, sz.h, yk, round0, roundOffset)
				predictInterCompoundRefHighBDToConvBufYClamped(wantY, ref, org[0], org[1], sz.w, sz.h, yk, round0, roundOffset)
				for i := range wantX {
					if gotX[i] != wantX[i] {
						t.Fatalf("X emu bd=%d %dx%d org=%v sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotX[i], wantX[i])
					}
					if gotY[i] != wantY[i] {
						t.Fatalf("Y emu bd=%d %dx%d org=%v sample %d: AVX2=%d PureGo=%d", bd, sz.w, sz.h, org, i, gotY[i], wantY[i])
					}
				}
			}
		}
	}
}

func TestCompoundHBDAVX2ZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0DE06))
	const pad = filterTaps
	const w, h = 32, 32
	round0, offsetBits, roundOffset := compoundHBDParams(12)
	ref := randPlaneHBD(rng, w+2*pad, (1<<12)-1)
	out := make([]uint16, w*h)
	k := avx2FilterTables()[0][7]
	var im compoundIM
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBufCopyResidentAVX2(out, ref, pad, pad, w, h, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBufXResidentAVX2(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBufYResidentAVX2(out, ref, pad, pad, w, h, k, round0, roundOffset)
		predictInterCompoundRefHighBDToConvBuf2DResidentAVX2(out, ref, pad, pad, w, h, k, k, round0, offsetBits, 12, &im)
	}); a != 0 {
		t.Fatalf("HBD compound AVX2 allocates: %v allocs/run", a)
	}
}
