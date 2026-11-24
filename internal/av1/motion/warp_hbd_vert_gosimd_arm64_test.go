// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestWarpVerticalHighBDGoSIMDMatchesScalar compares the NEON 16bpc vertical
// warp pass (full and gamma-zero) with the scalar one over random int32
// intermediates (including values past the horizontal pass's range) and
// filter phases at 10 and 12 bits, checking every byte of the plane.
func TestWarpVerticalHighBDGoSIMDMatchesScalar(t *testing.T) {
	if !warpGoSIMDEnabled {
		t.Skip("Go SIMD warp disabled")
	}
	rng := rand.New(rand.NewSource(91))
	tested := 0
	for draw := 0; draw < 6000; draw++ {
		bd := uint8(10)
		if draw&1 != 0 {
			bd = 12
		}
		round0, round1 := highBDRoundBits(bd)
		offsetV := int(bd) + 2*filterBits - round0
		max := uint16(1)<<bd - 1
		var tmp warpTmp
		lim := 1 << (int(bd) + 7 - round0 + 1)
		if draw%5 == 0 {
			lim = 1 << 20
		}
		for i := range tmp {
			tmp[i] = int32(rng.Intn(2*lim) - lim/2)
		}
		baseSY := rng.Intn(1<<17) - 1<<16
		gamma, delta := rng.Intn(1<<13)-1<<12, rng.Intn(1<<13)-1<<12
		gamma0 := draw%4 >= 2
		if gamma0 {
			gamma = 0
		}
		if !warpVertFullOffsInRange(baseSY, gamma, delta) {
			continue
		}
		mk := func() frame.Plane {
			p := frame.Plane{Pix: make([]byte, 2*24*24), Stride: 2 * 24, Width: 24, Height: 24}
			for i := range p.Pix {
				p.Pix[i] = 0xa5
			}
			return p
		}
		want, got := mk(), mk()
		if gamma0 {
			warpVerticalHighBDFullGamma0(want, bd, max, &tmp, 8, 8, 1, 2, baseSY, delta, round1, offsetV)
			warpVerticalHighBDFullGamma0Dispatch(got, bd, max, &tmp, 8, 8, 1, 2, baseSY, delta, round1, offsetV)
		} else {
			warpVerticalHighBDFull(want, bd, max, &tmp, 8, 8, 1, 2, baseSY, gamma, delta, round1, offsetV)
			warpVerticalHighBDFullDispatch(got, bd, max, &tmp, 8, 8, 1, 2, baseSY, gamma, delta, round1, offsetV)
		}
		tested++
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("draw %d bd %d gamma0=%v: byte %d got %d want %d", draw, bd, gamma0, i, got.Pix[i], want.Pix[i])
			}
		}
	}
	if tested < 1000 {
		t.Fatalf("only %d blocks tested", tested)
	}
}

// TestWarpVerticalHighBDGoSIMDZeroAlloc checks the NEON vertical passes do
// not allocate.
func TestWarpVerticalHighBDGoSIMDZeroAlloc(t *testing.T) {
	var tmp warpTmp
	dst := frame.Plane{Pix: make([]byte, 2*24*24), Stride: 2 * 24, Width: 24, Height: 24}
	round0, round1 := highBDRoundBits(10)
	offsetV := 10 + 2*filterBits - round0
	if n := testing.AllocsPerRun(100, func() {
		warpVerticalHighBDFullDispatch(dst, 10, 1023, &tmp, 8, 8, 0, 0, 0, 100, -200, round1, offsetV)
		warpVerticalHighBDFullGamma0Dispatch(dst, 10, 1023, &tmp, 8, 8, 0, 0, 0, -200, round1, offsetV)
	}); n != 0 {
		t.Fatalf("%v allocs per run", n)
	}
}

// BenchmarkWarpVerticalHighBD compares the NEON and scalar 10-bit vertical
// warp passes on one 8x8 block.
func BenchmarkWarpVerticalHighBD(b *testing.B) {
	var tmp warpTmp
	rng := rand.New(rand.NewSource(5))
	for i := range tmp {
		tmp[i] = int32(rng.Intn(1 << 15))
	}
	dst := frame.Plane{Pix: make([]byte, 2*24*24), Stride: 2 * 24, Width: 24, Height: 24}
	round0, round1 := highBDRoundBits(10)
	offsetV := 10 + 2*filterBits - round0
	b.Run("impl=scalar", func(b *testing.B) {
		for b.Loop() {
			warpVerticalHighBDFull(dst, 10, 1023, &tmp, 8, 8, 0, 0, 0, 100, -200, round1, offsetV)
		}
	})
	b.Run("impl=neon", func(b *testing.B) {
		for b.Loop() {
			warpVerticalHighBDFullDispatch(dst, 10, 1023, &tmp, 8, 8, 0, 0, 0, 100, -200, round1, offsetV)
		}
	})
}
