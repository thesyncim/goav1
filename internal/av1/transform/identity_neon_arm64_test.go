// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

func TestInverseIdentity16Rows4NEONMatchesPureGo(t *testing.T) {
	const (
		width       = 16
		height      = 16
		coeffStride = 19
		dstStride   = 21
	)
	impl := inverseIdentity16Rows4Impl
	if impl == nil {
		t.Fatal("16x16 identity kernel is not installed")
	}
	defer func() { inverseIdentity16Rows4Impl = impl }()
	rng := rand.New(rand.NewSource(0x1d7a16))
	activeRows := []int{0, 1, 3, 4, 5, 7, 8, 11, 12, 15, 16}
	for _, bounds := range stageClampSets {
		rowMin, rowMax := bounds[0], bounds[1]
		for _, colBounds := range stageClampSets {
			colMin, colMax := colBounds[0], colBounds[1]
			for _, active := range activeRows {
				for iter := 0; iter < 100; iter++ {
					coeff := make([]int32, (width-1)*coeffStride+height)
					for i := range coeff {
						coeff[i] = int32(rng.Uint32())
					}
					want := make([]int16, dstStride*height)
					got := make([]int16, dstStride*height)
					for i := range want {
						want[i] = int16(0x3535)
						got[i] = want[i]
					}

					inverseIdentity16Rows4Impl = nil
					if err := inverseIdentityBlockClampedRows(want, dstStride, coeff, coeffStride, Size{Width: width, Height: height}, rowMin, rowMax, colMin, colMax, active); err != nil {
						t.Fatal(err)
					}
					inverseIdentity16Rows4Impl = impl
					if err := inverseIdentityBlockClampedRows(got, dstStride, coeff, coeffStride, Size{Width: width, Height: height}, rowMin, rowMax, colMin, colMax, active); err != nil {
						t.Fatal(err)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("rowClamp=[%d,%d] colClamp=[%d,%d] active=%d iter=%d dst[%d]=%d want %d", rowMin, rowMax, colMin, colMax, active, iter, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestInverseIdentity16Rows4NEONZeroAlloc(t *testing.T) {
	coeff := make([]int32, 16*16)
	dst := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := inverseIdentityBlockClampedRows(dst, 16, coeff, 16, Size{Width: 16, Height: 16}, minInt16, maxInt16, minInt16, maxInt16, 0); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("16x16 identity kernel allocated %f times per call", allocs)
	}
}

func BenchmarkInverseIdentity16ActiveRows1NEON(b *testing.B) {
	benchmarkInverseIdentity16ActiveRows(b, 1, inverseIdentity16Rows4NEON)
}

func BenchmarkInverseIdentity16ActiveRows1PureGo(b *testing.B) {
	benchmarkInverseIdentity16ActiveRows(b, 1, nil)
}

func BenchmarkInverseIdentity16ActiveRows3NEON(b *testing.B) {
	benchmarkInverseIdentity16ActiveRows(b, 3, inverseIdentity16Rows4NEON)
}

func BenchmarkInverseIdentity16ActiveRows3PureGo(b *testing.B) {
	benchmarkInverseIdentity16ActiveRows(b, 3, nil)
}

func benchmarkInverseIdentity16ActiveRows(b *testing.B, activeRows int, impl func([]int16, int, []int32, int, int, int32, int32, int32, int32)) {
	old := inverseIdentity16Rows4Impl
	inverseIdentity16Rows4Impl = impl
	b.Cleanup(func() { inverseIdentity16Rows4Impl = old })
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32((i*37)%1021 - 510)
	}
	dst := make([]int16, 16*16)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = inverseIdentityBlockClampedRows(dst, 16, coeff, 16, Size{Width: 16, Height: 16}, minInt16, maxInt16, minInt16, maxInt16, activeRows)
	}
}
