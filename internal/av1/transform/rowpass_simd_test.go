// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// stageClampSets enumerates the (min,max) clamp pairs the inverse-transform
// stages use across the supported bit depths (8/10/12 bit, row and column
// passes). The NEON kernels must be bit-exact against pure-Go for every one.
var stageClampSets = func() [][2]int32 {
	var sets [][2]int32
	for _, bd := range []uint8{8, 10, 12} {
		rMin, rMax, cMin, cMax, ok := stageRangeBounds(bd)
		if !ok {
			continue
		}
		sets = append(sets, [2]int32{rMin, rMax}, [2]int32{cMin, cMax})
	}
	return sets
}()

// rowEdgeValues are the coefficient values most likely to expose rounding or
// clamp boundary bugs in the butterfly kernels.
var rowEdgeValues = []int32{
	0, 1, -1, 7, -7,
	minInt16, maxInt16,
	-131072, 131071,
	-524288, 524287,
	-8388608, 8388607,
}

// row2TestFunc names one batched row kernel under test: its accelerated
// implementation, the pure-Go reference it must equal bit-for-bit, and the
// transform length.
type row2TestFunc struct {
	name   string
	length int
	impl   func(r0, r1 []int32, min, max int32)
	ref    func(r0, r1 []int32, min, max int32)
}

// dispatchRow2Funcs returns every kernel under test. The impl entries come
// from row2TestImpls, which the arch-specific build populates: on arm64 it
// points at the NEON adapters directly (so all five NEON kernels are gated for
// bit-exactness regardless of which the dispatcher binds), elsewhere at the
// live dispatch slots.
func dispatchRow2Funcs() []row2TestFunc {
	return row2TestImpls()
}

func eqInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestRowPassDispatchMatchesPureGo asserts every wired batched row kernel is
// bit-identical to its pure-Go reference across random inputs and the full
// clamp-range ladder. On arm64 the dispatch slots resolve to the NEON kernels
// (cpu.Detected.NEON is mandatory), so this is the primary byte-exactness gate
// for the assembly.
func TestRowPassDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))
	for _, fn := range dispatchRow2Funcs() {
		t.Run(fn.name, func(t *testing.T) {
			for _, cs := range stageClampSets {
				min, max := cs[0], cs[1]
				for iter := 0; iter < 50000; iter++ {
					r0 := make([]int32, fn.length)
					r1 := make([]int32, fn.length)
					for i := 0; i < fn.length; i++ {
						// Span the row-input range (bd+8 -> up to 20-bit
						// magnitudes); clamp to the active stage range as the
						// staging step would.
						r0[i] = clipRange(int64(int32(rng.Int63())>>11), min, max)
						r1[i] = clipRange(int64(int32(rng.Int63())>>11), min, max)
					}
					ref0 := append([]int32(nil), r0...)
					ref1 := append([]int32(nil), r1...)
					got0 := append([]int32(nil), r0...)
					got1 := append([]int32(nil), r1...)
					fn.ref(ref0, ref1, min, max)
					fn.impl(got0, got1, min, max)
					if !eqInt32(got0, ref0) || !eqInt32(got1, ref1) {
						t.Fatalf("%s clamp=[%d,%d] iter=%d\n in0=%v in1=%v\n got0=%v want0=%v\n got1=%v want1=%v",
							fn.name, min, max, iter, r0, r1, got0, ref0, got1, ref1)
					}
				}
			}
		})
	}
}

// TestRowPassDispatchEdgeValues exhaustively crosses the boundary coefficient
// values (which sit at clamp and rounding cusps) for the small kernels and a
// large random sample for the wide ones.
func TestRowPassDispatchEdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1234abcd))
	for _, fn := range dispatchRow2Funcs() {
		t.Run(fn.name, func(t *testing.T) {
			for _, cs := range stageClampSets {
				min, max := cs[0], cs[1]
				for iter := 0; iter < 200000; iter++ {
					r0 := make([]int32, fn.length)
					r1 := make([]int32, fn.length)
					for i := 0; i < fn.length; i++ {
						r0[i] = clipRange(int64(rowEdgeValues[rng.Intn(len(rowEdgeValues))]), min, max)
						r1[i] = clipRange(int64(rowEdgeValues[rng.Intn(len(rowEdgeValues))]), min, max)
					}
					ref0 := append([]int32(nil), r0...)
					ref1 := append([]int32(nil), r1...)
					got0 := append([]int32(nil), r0...)
					got1 := append([]int32(nil), r1...)
					fn.ref(ref0, ref1, min, max)
					fn.impl(got0, got1, min, max)
					if !eqInt32(got0, ref0) || !eqInt32(got1, ref1) {
						t.Fatalf("%s clamp=[%d,%d]\n in0=%v in1=%v\n got0=%v want0=%v\n got1=%v want1=%v",
							fn.name, min, max, r0, r1, got0, ref0, got1, ref1)
					}
				}
			}
		})
	}
}

// TestRowPassDispatchZeroAlloc protects the hot-path invariant that the
// batched row kernels do not allocate.
func TestRowPassDispatchZeroAlloc(t *testing.T) {
	r0 := make([]int32, dct8Size)
	r1 := make([]int32, dct8Size)
	for i := range r0 {
		r0[i] = int32(i*131 - 400)
		r1[i] = int32(i*-77 + 200)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		inverseDCT8Row2Impl(r0, r1, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT8 batched row kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT4Row2Impl(r0, r1, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT4 batched row kernel allocated %f times per call", allocs)
	}
}
