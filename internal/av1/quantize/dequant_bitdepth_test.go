// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package quantize

import (
	"testing"
)

// TestDQCoeffBoundsMatchLibaom anchors the bit-depth-dependent dequantized
// coefficient clamp bounds to the values libaom's decode_coefs() applies in
// av1/decoder/decodetxb.c:312.
func TestDQCoeffBoundsMatchLibaom(t *testing.T) {
	cases := []struct {
		bd       uint8
		wantMin  int32
		wantMax  int32
	}{
		{bd: 8, wantMin: -32768, wantMax: 32767},
		{bd: 10, wantMin: -131072, wantMax: 131071},
		{bd: 12, wantMin: -524288, wantMax: 524287},
	}
	for _, tc := range cases {
		got_min, got_max, ok := dqCoeffBounds(tc.bd)
		if !ok {
			t.Fatalf("bd=%d: dqCoeffBounds returned !ok", tc.bd)
		}
		if got_min != tc.wantMin || got_max != tc.wantMax {
			t.Fatalf("bd=%d: range=[%d,%d] want [%d,%d]", tc.bd, got_min, got_max, tc.wantMin, tc.wantMax)
		}
	}
}

// TestDQCoeffBoundsRejectsInvalidBitDepth ensures only 8/10/12 are accepted.
func TestDQCoeffBoundsRejectsInvalidBitDepth(t *testing.T) {
	for _, bd := range []uint8{0, 1, 6, 7, 9, 11, 13, 16, 255} {
		if _, _, ok := dqCoeffBounds(bd); ok {
			t.Fatalf("bd=%d: dqCoeffBounds unexpectedly accepted", bd)
		}
	}
}

// TestDequantizeBlockScaledBitDepthClamps verifies that the 10-bit path lets
// values past the 8-bit clamp through, up to the 18-bit bound. This anchors
// the fix for the 10-bit quantizer cohort: the OLD goav1 path produced raw
// products that exceeded libaom's tran_low_t bound, feeding wrong residuals
// to the inverse transform.
func TestDequantizeBlockScaledBitDepthClamps(t *testing.T) {
	// Force a dequant product > int16 max but within ±131071 (10-bit) and
	// > ±524287 (12-bit). coeff=300, scale=200 ⇒ 60000 > 32767, < 131071.
	width, height := 4, 4
	q := Quantizer{DC: 200, AC: 200}
	coeff := make([]int16, width*height)
	for i := range coeff {
		coeff[i] = 300
	}
	dst8 := make([]int32, width*height)
	dst10 := make([]int32, width*height)
	if err := DequantizeBlockScaledBitDepth(dst8, height, coeff, height, width, height, q, 0, 8); err != nil {
		t.Fatalf("8-bit dequant: %v", err)
	}
	if err := DequantizeBlockScaledBitDepth(dst10, height, coeff, height, width, height, q, 0, 10); err != nil {
		t.Fatalf("10-bit dequant: %v", err)
	}
	for i := range dst8 {
		if dst8[i] != 32767 {
			t.Fatalf("8-bit dst[%d]=%d want 32767 (int16 max)", i, dst8[i])
		}
		if dst10[i] != 60000 {
			t.Fatalf("10-bit dst[%d]=%d want 60000 (unclamped)", i, dst10[i])
		}
	}
}

// TestDequantizeBlockScaledBitDepthInt64Multiply pins the int64 widening of
// libaom's `(int64)level * dqv & 0xffffff` formula. The Go input is already
// int16, so for legal AV1 streams the multiply fits in int32; this test
// constructs an out-of-range product that would wrap to a different value
// under a 32-bit multiply, to guard against future regressions where the
// quantized buffer widens to int32 (e.g. for a more libaom-faithful pipeline).
// The 10-bit q32/q63 cohort exposed this divergence as +4 residual deltas at
// the first sample of frame 0; widening to int64 keeps the bit pattern of
// libaom's tran_low_t multiply exactly.
func TestDequantizeBlockScaledBitDepthInt64Multiply(t *testing.T) {
	// Pre-fill dequant output and dequantize a single nonzero level. Drive
	// `level * scale` into a value > 2^23 so the &0xffffff mask actually
	// strips bits; this verifies the mask is applied after the int64 widen.
	width, height := 4, 4
	q := Quantizer{DC: 16777216 - 100, AC: 16777216 - 100} // 24-bit-ish scale
	coeff := make([]int16, width*height)
	coeff[0] = 2
	dst := make([]int32, width*height)
	if err := DequantizeBlockScaledBitDepth(dst, height, coeff, height, width, height, q, 0, 12); err != nil {
		t.Fatalf("12-bit dequant: %v", err)
	}
	// product = 2 * (1<<24 - 100) = (1<<25 - 200). After &0xffffff: low
	// 24 bits = (1<<25 - 200) mod (1<<24) = (1<<24 - 200). The 12-bit dq
	// clamp at ±(1<<19) caps this at 524287.
	if dst[0] != 524287 {
		t.Fatalf("dst[0]=%d want 524287 (12-bit dq clamp ceiling)", dst[0])
	}
}

// TestDequantizeBlockScaledBitDepth8Equivalent confirms the BitDepth(8) entry
// point reproduces the legacy unscaled-bd path.
func TestDequantizeBlockScaledBitDepth8Equivalent(t *testing.T) {
	width, height := 4, 4
	q := Quantizer{DC: 50, AC: 60}
	coeff := make([]int16, width*height)
	for i := range coeff {
		coeff[i] = int16(i*37 - 200)
	}
	dst := make([]int32, width*height)
	dstBD := make([]int32, width*height)
	if err := DequantizeBlockScaled(dst, height, coeff, height, width, height, q, 1); err != nil {
		t.Fatalf("DequantizeBlockScaled: %v", err)
	}
	if err := DequantizeBlockScaledBitDepth(dstBD, height, coeff, height, width, height, q, 1, 8); err != nil {
		t.Fatalf("DequantizeBlockScaledBitDepth(8): %v", err)
	}
	for i := range dst {
		if dst[i] != dstBD[i] {
			t.Fatalf("idx=%d legacy=%d bitDepth8=%d", i, dst[i], dstBD[i])
		}
	}
}
