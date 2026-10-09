// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import "testing"

// minMaxSIMDCheck compares the Go SIMD kernel with the pure-Go reference for one
// argument set, including the error verdict.
func minMaxSIMDCheck(t *testing.T, a []byte, aStride int, b []byte, bStride int, bytesPerSample int) {
	t.Helper()
	wantMin, wantMax, wantErr := minMaxAbsDiff8x8PureGo(a, aStride, b, bStride, bytesPerSample)
	gotMin, gotMax, gotErr := minMaxAbsDiff8x8SIMD(a, aStride, b, bStride, bytesPerSample)
	if gotErr != wantErr {
		t.Fatalf("bps=%d aStride=%d bStride=%d err: simd=%v reference=%v", bytesPerSample, aStride, bStride, gotErr, wantErr)
	}
	if gotMin != wantMin || gotMax != wantMax {
		t.Fatalf("bps=%d aStride=%d bStride=%d simd=(%d,%d) reference=(%d,%d)",
			bytesPerSample, aStride, bStride, gotMin, gotMax, wantMin, wantMax)
	}
}

// TestMinMaxAbsDiff8x8SIMDMatchesPureGo sweeps every stride both samples can use
// with random data, then uses the sample extremes so that the lane-wise
// differences saturate at zero and at the maximum.
func TestMinMaxAbsDiff8x8SIMDMatchesPureGo(t *testing.T) {
	for _, bytesPerSample := range []int{1, 2} {
		minStride := 8 * bytesPerSample
		var a [8 * 64]byte
		var b [8 * 64]byte
		for seed := range uint32(4) {
			fillMinMaxBytes(a[:], 0x9e3779b9*(seed+1))
			fillMinMaxBytes(b[:], 0x7f4a7c15*(seed+1))
			for aStride := minStride; aStride <= 64; aStride += minStride {
				for bStride := minStride; bStride <= 64; bStride += minStride {
					minMaxSIMDCheck(t, a[:], aStride, b[:], bStride, bytesPerSample)
					minMaxSIMDCheck(t, a[:], aStride+1, b[:], bStride, bytesPerSample)
				}
			}
		}
		// Extreme sample patterns: all-zero against all-ones, identical blocks,
		// and alternating extremes in each lane.
		extremes := [][2]byte{{0x00, 0xff}, {0xff, 0x00}, {0x80, 0x7f}, {0x55, 0x55}}
		for _, pair := range extremes {
			for i := range a {
				a[i] = pair[0]
				b[i] = pair[1]
				if i%3 == 0 {
					a[i], b[i] = pair[1], pair[0]
				}
			}
			for aStride := minStride; aStride <= 64; aStride += minStride {
				minMaxSIMDCheck(t, a[:], aStride, b[:], aStride, bytesPerSample)
			}
		}
	}
}

// TestMinMaxAbsDiff8x8SIMDMatchesPureGoSixteenBit checks the 16-bit path at the
// full u16 range, where the lane differences reach 0xffff.
func TestMinMaxAbsDiff8x8SIMDMatchesPureGoSixteenBit(t *testing.T) {
	var a [8 * 32]byte
	var b [8 * 32]byte
	patterns := []uint16{0, 1, 0x7fff, 0x8000, 0xfffe, 0xffff, 0x0fff, 0x1000}
	for i := range 8 * 16 {
		put16(a[:], i, patterns[i%len(patterns)])
		put16(b[:], i, patterns[(i*5+3)%len(patterns)])
	}
	for aStride := 16; aStride <= 32; aStride += 16 {
		for bStride := 16; bStride <= 32; bStride += 16 {
			minMaxSIMDCheck(t, a[:], aStride, b[:], bStride, 2)
		}
	}
}

// TestMinMaxAbsDiff8x8SIMDRejectsInvalid confirms the SIMD variant reports the
// same invalid verdicts as the reference for bad sample widths, short buffers and
// strides below the row size.
func TestMinMaxAbsDiff8x8SIMDRejectsInvalid(t *testing.T) {
	buf := make([]byte, 8*64)
	cases := []struct {
		name           string
		aStride        int
		bytesPerSample int
		a              []byte
	}{
		{"bps-0", 8, 0, buf},
		{"bps-3", 8, 3, buf},
		{"stride-short-8bit", 7, 1, buf},
		{"stride-short-16bit", 15, 2, buf},
		{"short-buffer", 8, 1, buf[:40]},
		{"short-buffer-16bit", 16, 2, buf[:100]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, gotErr := minMaxAbsDiff8x8SIMD(tc.a, tc.aStride, buf, tc.aStride, tc.bytesPerSample)
			_, _, wantErr := minMaxAbsDiff8x8PureGo(tc.a, tc.aStride, buf, tc.aStride, tc.bytesPerSample)
			if gotErr != ErrInvalidBlock || wantErr != ErrInvalidBlock {
				t.Fatalf("errors: simd=%v reference=%v, want ErrInvalidBlock for both", gotErr, wantErr)
			}
		})
	}
}

// TestMinMaxAbsDiff8x8SIMDIsZeroAlloc keeps the Go SIMD kernel allocation-free.
func TestMinMaxAbsDiff8x8SIMDIsZeroAlloc(t *testing.T) {
	var a [8 * 64]byte
	var b [8 * 64]byte
	fillMinMaxBytes(a[:], 0x11111111)
	fillMinMaxBytes(b[:], 0x22222222)
	for _, bytesPerSample := range []int{1, 2} {
		allocs := testing.AllocsPerRun(1000, func() {
			_, _, err := minMaxAbsDiff8x8SIMD(a[:], 16, b[:], 16, bytesPerSample)
			if err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("bps=%d: SIMD kernel allocated %f times per call", bytesPerSample, allocs)
		}
	}
}
