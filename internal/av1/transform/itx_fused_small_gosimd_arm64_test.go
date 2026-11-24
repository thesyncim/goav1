// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

type itxSmallFused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool

// testInverseAddSmall checks a generated w x h kernel set against the
// staged transform and residual add for every type but IDTX: coefficients
// inside the coded extent (zero past it, as the decoder's coefficient
// buffers are), magnitudes past the int16 clamps, and separate pred/dst
// buffers. Realistic magnitudes must never decline.
func testInverseAddSmall(t *testing.T, w, h int, fused itxSmallFused) {
	rng := rand.New(rand.NewSource(int64(97 + w*h + w)))
	stride := w + 8
	coeff := make([]int32, w*h)
	raw := make([]int32, w*h)
	pred := make([]byte, h*stride)
	got := make([]byte, h*stride)
	want := make([]byte, h*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 60000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX || !txType.Supported(Size{Width: uint8(w), Height: uint8(h)}) {
			continue
		}
		// A 16-point side puts the block in the 16x16 transform sets,
		// which have no V_ADST, H_ADST, V_FLIPADST or H_FLIPADST: the
		// kernels decline those (the staged path runs them).
		if max(w, h) == 16 && txType >= TypeVADST {
			if fused(got, stride, pred, stride, coeff, h, txType, 1, 1) {
				t.Fatalf("type %d accepted at %dx%d", txType, w, h)
			}
			continue
		}
		activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
		mode := (trial / int(TypeCount)) % 6
		mag := []int{16, 256, 1024, 4096, 32767, 70000}[mode]
		clear(coeff)
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) != 0 {
					coeff[c*h+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		if err := InverseBlockBitDepthRawBounded(raw, coeff, h, Size{Width: uint8(w), Height: uint8(h)}, txType, 8, activeRows, activeCols); err != nil {
			t.Fatal(err)
		}
		copy(want, pred)
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				v := int(pred[r*stride+c]) + int((int64(raw[r*w+c])+8)>>4)
				want[r*stride+c] = byte(min(max(v, 0), 255))
			}
		}
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, h, txType, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst", trial)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mag %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mag, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = 0xa5
		}
		if !fused(got, stride, pred, stride, coeff, h, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < h; r++ {
			for c := 0; c < stride; c++ {
				wv := want[r*stride+c]
				if c >= w {
					wv = 0xa5
				}
				if got[r*stride+c] != wv {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", w, h, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
	if fused(got, stride, pred, stride, coeff, h, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted")
	}
}

func TestInverseAdd4x4GenMatchesStaged(t *testing.T) {
	testInverseAddSmall(t, 4, 4, func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
		return InverseAdd4x4(dst, dstStride, pred, predStride, coeff, coeffStride, txType, activeRows)
	})
}
func TestInverseAdd4x8GenMatchesStaged(t *testing.T)  { testInverseAddSmall(t, 4, 8, InverseAdd4x8) }
func TestInverseAdd8x4GenMatchesStaged(t *testing.T)  { testInverseAddSmall(t, 8, 4, InverseAdd8x4) }
func TestInverseAdd4x16GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 4, 16, InverseAdd4x16) }
func TestInverseAdd16x4GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 16, 4, InverseAdd16x4) }
func TestInverseAdd8x16GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 8, 16, InverseAdd8x16) }
func TestInverseAdd16x8GenMatchesStaged(t *testing.T) { testInverseAddSmall(t, 16, 8, InverseAdd16x8) }
