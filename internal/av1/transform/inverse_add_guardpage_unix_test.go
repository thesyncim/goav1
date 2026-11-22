// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package transform

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

// inverseAddCase is one fused inverse-transform-and-add entry point: a w x h
// block of bps-byte samples whose coded coefficients are the column-major
// min(w,32) x min(h,32) corner.
type inverseAddCase struct {
	name string
	w, h int
	bps  int
	run  func(dst []byte, ds int, pred []byte, ps int, coeff []int32, cs int, typ Type, rows, cols int) bool
}

func inverseAddCases() []inverseAddCase {
	type a8 = func([]byte, int, []byte, int, []int32, int, int, int) bool
	type t8 = func([]byte, int, []byte, int, []int32, int, Type, int, int) bool
	type a16 = func([]byte, int, []byte, int, []int32, int, int, int, uint8) bool
	type t16 = func([]byte, int, []byte, int, []int32, int, Type, int, int, uint8) bool
	dct := func(name string, w, h int, f a8) inverseAddCase {
		return inverseAddCase{name, w, h, 1, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, _ Type, r, k int) bool {
			return f(d, ds, p, ps, c, cs, r, k)
		}}
	}
	typed := func(name string, w, h int, f t8) inverseAddCase {
		return inverseAddCase{name, w, h, 1, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, t Type, r, k int) bool {
			return f(d, ds, p, ps, c, cs, t, r, k)
		}}
	}
	dctHBD := func(name string, w, h int, f a16) inverseAddCase {
		return inverseAddCase{name, w, h, 2, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, _ Type, r, k int) bool {
			return f(d, ds, p, ps, c, cs, r, k, 10)
		}}
	}
	typedHBD := func(name string, w, h int, f t16) inverseAddCase {
		return inverseAddCase{name, w, h, 2, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, t Type, r, k int) bool {
			return f(d, ds, p, ps, c, cs, t, r, k, 10)
		}}
	}
	return []inverseAddCase{
		{"4x4", 4, 4, 1, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, t Type, r, _ int) bool {
			return InverseAdd4x4(d, ds, p, ps, c, cs, t, r)
		}},
		typed("4x8", 4, 8, InverseAdd4x8),
		typed("8x4", 8, 4, InverseAdd8x4),
		typed("4x16", 4, 16, InverseAdd4x16),
		typed("16x4", 16, 4, InverseAdd16x4),
		typed("8x16", 8, 16, InverseAdd8x16),
		typed("16x8", 16, 8, InverseAdd16x8),
		typed("8x8hybrid", 8, 8, InverseAdd8x8Hybrid),
		typed("16x16hybrid", 16, 16, InverseAdd16x16Hybrid),
		dct("dct8x8", 8, 8, InverseAddDCT8x8),
		dct("dct16x16", 16, 16, InverseAddDCT16x16),
		dct("dct16x32", 16, 32, InverseAddDCT16x32),
		dct("dct32x16", 32, 16, InverseAddDCT32x16),
		dct("dct32x32", 32, 32, InverseAddDCT32x32Full),
		{"dct32x32low16", 32, 32, 1, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, _ Type, r, _ int) bool {
			return InverseAddDCT32x32Low16(d, ds, p, ps, c, cs, min(r, 16))
		}},
		{"dct32x32low8", 32, 32, 1, func(d []byte, ds int, p []byte, ps int, c []int32, cs int, _ Type, _, _ int) bool {
			return InverseAddDCT32x32Low8(d, ds, p, ps, c, cs)
		}},
		dct("dct64x64", 64, 64, InverseAddDCT64x64),
		typedHBD("8x8hybridHBD", 8, 8, InverseAdd8x8HybridHBD),
		typedHBD("16x16hybridHBD", 16, 16, InverseAdd16x16HybridHBD),
		dctHBD("dct8x8HBD", 8, 8, InverseAddDCT8x8HBD),
		dctHBD("dct16x16HBD", 16, 16, InverseAddDCT16x16HBD),
		dctHBD("dct16x32HBD", 16, 32, InverseAddDCT16x32HBD),
		dctHBD("dct32x16HBD", 32, 16, InverseAddDCT32x16HBD),
		dctHBD("dct32x32HBD", 32, 32, InverseAddDCT32x32HBD),
		dctHBD("dct64x64HBD", 64, 64, InverseAddDCT64x64HBD),
	}
}

// TestInverseAddStaysInsideGuardedBlocks runs every fused inverse-add entry
// point on destination, prediction and coefficient buffers exactly as long
// as its length check requires, ending at (or starting right after) an
// inaccessible page, and compares the result with a heap-allocated run.
func TestInverseAddStaysInsideGuardedBlocks(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	types := []Type{TypeDCTDCT, TypeADSTDCT, TypeFlipADSTDCT}
	for _, tc := range inverseAddCases() {
		cw, ch := min(tc.w, 32), min(tc.h, 32)
		for iter := range 24 {
			before := iter%2 == 1
			alloc := func(n int) []byte {
				if before {
					return guardpage.Before[byte](t, n)
				}
				return guardpage.After[byte](t, n)
			}
			stride := tc.w * tc.bps
			n := (tc.h-1)*stride + tc.w*tc.bps
			dst, pred := alloc(n), alloc(n)
			for i := 0; i < n; i += tc.bps {
				v := rng.Intn(256)
				if tc.bps == 2 {
					v = rng.Intn(1 << 10)
					dst[i+1], pred[i+1] = byte(v>>8), byte(v>>8)
				}
				dst[i], pred[i] = byte(v), byte(v)
			}
			var coeff []int32
			if before {
				coeff = guardpage.Before[int32](t, (cw-1)*ch+ch)
			} else {
				coeff = guardpage.After[int32](t, (cw-1)*ch+ch)
			}
			rows, cols := 1+rng.Intn(ch), 1+rng.Intn(cw)
			for c := range cols {
				for r := range rows {
					coeff[c*ch+r] = int32(rng.Intn(512) - 256)
				}
			}
			typ := types[iter%len(types)]
			wantDst, wantPred, wantCoeff := bytes.Clone(dst), bytes.Clone(pred), append([]int32(nil), coeff...)
			name := fmt.Sprintf("%s iter %d before=%v rows=%d cols=%d type=%d", tc.name, iter, before, rows, cols, typ)
			var ok bool
			guardpage.Check(t, name, func() {
				ok = tc.run(dst, stride, pred, stride, coeff, ch, typ, rows, cols)
			})
			wantOK := tc.run(wantDst, stride, wantPred, stride, wantCoeff, ch, typ, rows, cols)
			if ok != wantOK || !bytes.Equal(dst, wantDst) {
				t.Fatalf("%s: guarded run differs from the heap run", name)
			}
		}
	}
}
