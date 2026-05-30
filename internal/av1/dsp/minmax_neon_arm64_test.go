// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestMinMaxAbsDiff8x8NEONMatchesPureGo calls the NEON variant directly (not
// via the dispatch slot) and asserts bit-exactness with the pure-Go reference
// across the full (stride, bytesPerSample) shape space plus randomised
// content. This guards the assembly independently of the dispatch wiring.
func TestMinMaxAbsDiff8x8NEONMatchesPureGo(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
	for _, seed := range []uint32{0xdeadbeef, 0xfeedface, 0x00000000, 0xffffffff, 0x12345678} {
		for _, bytesPerSample := range []int{1, 2} {
			var a [8 * 64]byte
			var b [8 * 64]byte
			fillMinMaxBytes(a[:], seed)
			fillMinMaxBytes(b[:], seed^0xa5a5a5a5)
			minStride := 8 * bytesPerSample
			for aStride := minStride; aStride <= 64; aStride += minStride {
				for bStride := minStride; bStride <= 64; bStride += minStride {
					wantMin, wantMax, wantErr := minMaxAbsDiff8x8PureGo(a[:], aStride, b[:], bStride, bytesPerSample)
					gotMin, gotMax, gotErr := minMaxAbsDiff8x8NEON(a[:], aStride, b[:], bStride, bytesPerSample)
					if wantErr != gotErr {
						t.Fatalf("seed=%#x bps=%d aStride=%d bStride=%d err: neon=%v ref=%v", seed, bytesPerSample, aStride, bStride, gotErr, wantErr)
					}
					if gotMin != wantMin || gotMax != wantMax {
						t.Fatalf("seed=%#x bps=%d aStride=%d bStride=%d neon=(%d,%d) ref=(%d,%d)",
							seed, bytesPerSample, aStride, bStride, gotMin, gotMax, wantMin, wantMax)
					}
				}
			}
		}
	}
}

// TestMinMaxAbsDiff8x8NEONRejectsInvalid confirms the NEON wrapper enforces the
// same validation contract as the reference, returning ErrInvalidBlock rather
// than touching out-of-range memory.
func TestMinMaxAbsDiff8x8NEONRejectsInvalid(t *testing.T) {
	var a [64]byte
	var b [64]byte
	for _, tc := range []struct {
		name           string
		aStride        int
		bStride        int
		bytesPerSample int
		a              []byte
		b              []byte
	}{
		{name: "bad sample width", aStride: 8, bStride: 8, bytesPerSample: 3, a: a[:], b: b[:]},
		{name: "short stride", aStride: 7, bStride: 8, bytesPerSample: 1, a: a[:], b: b[:]},
		{name: "short source", aStride: 8, bStride: 8, bytesPerSample: 1, a: a[:63], b: b[:]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := minMaxAbsDiff8x8NEON(tc.a, tc.aStride, tc.b, tc.bStride, tc.bytesPerSample); err != ErrInvalidBlock {
				t.Fatalf("err=%v want %v", err, ErrInvalidBlock)
			}
		})
	}
}

func TestMinMaxAbsDiff8x8NEONIsZeroAlloc(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
	var a [8 * 64]byte
	var b [8 * 64]byte
	fillMinMaxBytes(a[:], 0x11111111)
	fillMinMaxBytes(b[:], 0x22222222)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := minMaxAbsDiff8x8NEON(a[:], 64, b[:], 64, 2); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("minMaxAbsDiff8x8NEON allocated: %f", allocs)
	}
}
