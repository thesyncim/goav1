// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

type itxRectFused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool

// testInverseAddRect checks a fused w x h kernel against the staged
// transform and residual add for every type but IDTX: coefficients inside
// the coded extent, garbage in the rows past it (the staged path never
// reads them), magnitudes past the int16 clamps, and separate pred/dst
// buffers. Realistic magnitudes must never decline.
func testInverseAddRect(t *testing.T, w, h int, fused itxRectFused) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(int64(91 + w*h + w)))
	stride := w + 8
	coeff := make([]int32, w*h)
	raw := make([]int32, w*h)
	pred := make([]byte, h*stride)
	got := make([]byte, h*stride)
	want := make([]byte, h*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 100000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX {
			continue
		}
		activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
		mode := (trial / int(TypeCount)) % 6
		mag := []int{16, 256, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < w; c++ {
			for r := 0; r < h; r++ {
				v := int32(0)
				switch {
				case r >= activeRows:
					v = int32(rng.Intn(2001) - 1000)
				case c < activeCols && rng.Intn(3) != 0:
					v = int32(rng.Intn(2*mag+1) - mag)
				}
				coeff[c*h+r] = v
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
	if fused(got, stride, pred, stride, coeff, h, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted")
	}
}

func TestInverseAdd4x8MatchesStaged(t *testing.T) { testInverseAddRect(t, 4, 8, InverseAdd4x8) }
func TestInverseAdd8x4MatchesStaged(t *testing.T) { testInverseAddRect(t, 8, 4, InverseAdd8x4) }

func TestInverseAdd4x16MatchesStaged(t *testing.T) { testInverseAddRect(t, 4, 16, InverseAdd4x16) }
func TestInverseAdd16x4MatchesStaged(t *testing.T) { testInverseAddRect(t, 16, 4, InverseAdd16x4) }
func TestInverseAdd8x16MatchesStaged(t *testing.T) { testInverseAddRect(t, 8, 16, InverseAdd8x16) }
func TestInverseAdd16x8MatchesStaged(t *testing.T) { testInverseAddRect(t, 16, 8, InverseAdd16x8) }

func TestInverseAdd8x8AllTypesMatchesStaged(t *testing.T) {
	testInverseAddRect(t, 8, 8, func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
		if txType == TypeDCTDCT {
			return InverseAddDCT8x8(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols)
		}
		return InverseAdd8x8Hybrid(dst, dstStride, pred, predStride, coeff, coeffStride, txType, activeRows, activeCols)
	})
}
