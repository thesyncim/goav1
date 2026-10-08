// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// col2TestFunc names one batched column kernel under test: its accelerated
// implementation, the pure-Go reference it must equal bit-for-bit, and the
// transform length.
type col2TestFunc struct {
	name   string
	length int
	impl   func(buf []int32, rowStride int, min, max int32)
	ref    func(buf []int32, rowStride int, min, max int32)
}

// runCol2 lays out a length x (rowStride>=2) row-major buffer, transforms the
// first two columns with both impl and ref, and reports any mismatch. The
// buffer is sized so the two columns plus their tail row exactly fit; the
// padding columns are left untouched and compared too, guarding against any
// stray write outside the targeted column pair.
func runCol2(t *testing.T, fn col2TestFunc, rng *rand.Rand, fill func() int32) {
	t.Helper()
	const rowStride = 4 // exercise a real stride (two target columns + padding)
	bufLen := (fn.length-1)*rowStride + 2
	for _, cs := range stageClampSets {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 30000; iter++ {
			buf := make([]int32, bufLen)
			for i := range buf {
				buf[i] = clipRange(int64(fill()), min, max)
			}
			ref := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			fn.ref(ref, rowStride, min, max)
			fn.impl(got, rowStride, min, max)
			if !eqInt32(got, ref) {
				t.Fatalf("%s clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v",
					fn.name, min, max, iter, buf, got, ref)
			}
		}
	}
}

// runCol4 is runCol2 for the batched four-column kernels: it transforms the
// first four columns of a length x rowStride row-major buffer with both impl
// and ref and reports any mismatch, including in the untouched padding
// columns. Inputs are pre-clamped to [min, max], the four-column kernels'
// documented precondition (the decode path establishes it via the mid-pass
// clamp in hybrid.go).
func runCol4(t *testing.T, fn col2TestFunc, rng *rand.Rand, fill func() int32) {
	t.Helper()
	const rowStride = 6 // four target columns + padding
	bufLen := (fn.length-1)*rowStride + 4
	for _, cs := range stageClampSets {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 30000; iter++ {
			buf := make([]int32, bufLen)
			for i := range buf {
				buf[i] = clipRange(int64(fill()), min, max)
			}
			ref := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			fn.ref(ref, rowStride, min, max)
			fn.impl(got, rowStride, min, max)
			if !eqInt32(got, ref) {
				t.Fatalf("%s clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v",
					fn.name, min, max, iter, buf, got, ref)
			}
		}
	}
}

// TestColPassDispatchMatchesPureGo asserts every column kernel is bit-identical
// to its pure-Go reference across random inputs and the full clamp ladder. On
// arm64 the entries resolve to the NEON kernels, so this is the byte-exactness
// gate for the column assembly.
func TestColPassDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc01))
	for _, fn := range colPass2TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol2(t, fn, rng, func() int32 { return int32(int32(rng.Int63()) >> 11) })
		})
	}
	for _, fn := range colPass4TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol4(t, fn, rng, func() int32 { return int32(int32(rng.Int63()) >> 11) })
		})
	}
}

// TestColPassDispatchEdgeValues crosses the boundary coefficient values that
// sit at clamp and rounding cusps.
func TestColPassDispatchEdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(0xfeed))
	for _, fn := range colPass2TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol2(t, fn, rng, func() int32 { return rowEdgeValues[rng.Intn(len(rowEdgeValues))] })
		})
	}
	for _, fn := range colPass4TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol4(t, fn, rng, func() int32 { return rowEdgeValues[rng.Intn(len(rowEdgeValues))] })
		})
	}
}

// TestColPassDispatchZeroAlloc protects the hot-path invariant that the batched
// column kernels do not allocate.
func TestColPassDispatchZeroAlloc(t *testing.T) {
	const rowStride = 4
	buf := make([]int32, (dct16Size-1)*rowStride+2)
	for i := range buf {
		buf[i] = int32(i*131 - 400)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		inverseDCT8Col2Impl(buf, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT8 batched column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT16Col2Impl(buf, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT16 batched column kernel allocated %f times per call", allocs)
	}
	buf4 := make([]int32, (dct64Size-1)*rowStride+4)
	for i := range buf4 {
		buf4[i] = int32(i*131 - 400)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT16Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT16 batched four-column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT32Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT32 batched four-column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT64Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT64 batched four-column kernel allocated %f times per call", allocs)
	}
}
