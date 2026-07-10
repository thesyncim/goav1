// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math"
	"math/rand"
	"testing"
)

// TestStageTransposeClampSIMDDifferential locksteps the SIMD staging kernel
// against stageTransposeClampScalar over every block shape the row pass can
// request, both rect2 variants, adversarial extremes (full int32 range, well
// past the dequant envelope) and random dequant-realistic values.
func TestStageTransposeClampSIMDDifferential(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	extremes := []int32{
		math.MinInt32, math.MinInt32 + 1, -(1 << 22) - 1, -(1 << 22), -(1 << 22) + 1,
		-524288, -181, -1, 0, 1, 181, 524287,
		(1 << 22) - 1, 1 << 22, (1 << 22) + 1, math.MaxInt32 - 1, math.MaxInt32,
	}
	clamps := []struct{ lo, hi int32 }{
		{-32768, 32767},           // 8-bit row clamp
		{-131072, 131071},         // 10-bit
		{-524288, 524287},         // 12-bit
		{math.MinInt32, math.MaxInt32}, // degenerate wide clamp (rect2 falls back)
	}
	shapes := []struct{ rows, cols, width, stride int }{
		{4, 4, 4, 4}, {4, 4, 8, 8}, {8, 8, 8, 8}, {16, 16, 16, 16},
		{32, 32, 32, 32}, {32, 32, 64, 32}, {5, 4, 8, 8}, {4, 5, 8, 8},
		{7, 6, 8, 8}, {13, 9, 16, 16}, {3, 3, 4, 4}, {2, 7, 8, 8},
		{16, 4, 16, 16}, {4, 16, 16, 16}, {31, 17, 32, 32},
	}
	for _, sh := range shapes {
		coeff := make([]int32, sh.stride*sh.cols+sh.rows)
		want := make([]int32, sh.width*sh.rows)
		got := make([]int32, sh.width*sh.rows)
		for _, cl := range clamps {
			for _, rect2 := range []bool{false, true} {
				for trial := 0; trial < 8; trial++ {
					for i := range coeff {
						switch trial {
						case 0:
							coeff[i] = extremes[i%len(extremes)]
						case 1:
							coeff[i] = extremes[rng.Intn(len(extremes))]
						default:
							coeff[i] = int32(rng.Intn(1<<20) - 1<<19)
						}
					}
					for i := range want {
						want[i] = math.MaxInt32
						got[i] = math.MaxInt32
					}
					stageTransposeClampScalar(want, sh.width, coeff, sh.stride, sh.rows, sh.cols, rect2, cl.lo, cl.hi)
					stageTransposeClamp(got, sh.width, coeff, sh.stride, sh.rows, sh.cols, rect2, cl.lo, cl.hi)
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("shape %+v clamp %+v rect2=%v trial=%d: mismatch at %d: scalar %d simd %d",
								sh, cl, rect2, trial, i, want[i], got[i])
						}
					}
				}
			}
		}
	}
}

func BenchmarkStageTransposeClamp32x32(b *testing.B) {
	benchStage(b, 32, 32, 32, false)
}

func BenchmarkStageTransposeClamp32x32Rect2(b *testing.B) {
	benchStage(b, 32, 32, 32, true)
}

func BenchmarkStageTransposeClamp16x16(b *testing.B) {
	benchStage(b, 16, 16, 16, false)
}

func BenchmarkStageTransposeClampScalar32x32(b *testing.B) {
	benchStageScalar(b, 32, 32, 32, false)
}

func BenchmarkStageTransposeClampScalar16x16(b *testing.B) {
	benchStageScalar(b, 16, 16, 16, false)
}

func benchStage(b *testing.B, rows, cols, width int, rect2 bool) {
	coeff := make([]int32, cols*width)
	scratch := make([]int32, rows*width)
	rng := rand.New(rand.NewSource(2))
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<16) - 1<<15)
	}
	b.ReportAllocs()
	for b.Loop() {
		stageTransposeClamp(scratch, width, coeff, width, rows, cols, rect2, -32768, 32767)
	}
}

func benchStageScalar(b *testing.B, rows, cols, width int, rect2 bool) {
	coeff := make([]int32, cols*width)
	scratch := make([]int32, rows*width)
	rng := rand.New(rand.NewSource(2))
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<16) - 1<<15)
	}
	b.ReportAllocs()
	for b.Loop() {
		stageTransposeClampScalar(scratch, width, coeff, width, rows, cols, rect2, -32768, 32767)
	}
}
