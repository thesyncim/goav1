// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import (
	"slices"
	"testing"
)

// TestSGRRollingIntermediateAVX2MatchesFullOracle checks the AVX2 A/B producer
// against the frozen full box-sum oracle for every parameter set, radius,
// pass and bit depth, on random and on all-maximum samples, and that it
// writes nothing past the coefficient buffers.
func TestSGRRollingIntermediateAVX2MatchesFullOracle(t *testing.T) {
	if !sgrIntermediateAVX2 {
		t.Skip("no AVX2")
	}
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xa52)
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {6, 6}, {7, 9}, {8, 8}, {14, 5}, {16, 8}, {31, 33}, {32, 64}, {63, 64}, {64, 63}, {64, 64}} {
		w, h := sz[0], sz[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, bd := range []int{8, 10, 12} {
			for fill := 0; fill < 2; fill++ {
				for i := range dgd {
					if fill == 0 {
						dgd[i] = int32(rnd.pseudoUniform(1 << bd))
					} else {
						dgd[i] = int32(1<<bd - 1)
					}
				}
				for pi, p := range SGRParameterTable {
					for ri, r := range p.Radius {
						if r == 0 {
							continue
						}
						for pass := 0; pass < 2; pass++ {
							wantA, wantB := make([]int32, bl), make([]int32, bl)
							gotA, gotB := make([]int32, bl+7), make([]int32, bl+7)
							for i := range gotA {
								gotA[i] = int32(uint32(i)*2654435761 + 0xfedc)
								gotB[i] = ^gotA[i]
							}
							tailA, tailB := slices.Clone(gotA[bl:]), slices.Clone(gotB[bl:])
							frozenSGRIntermediate(dgd, origin, w, h, stride, bd, pi, ri, pass, wantA, wantB, bs)
							calculateIntermediateRollingAVX2(dgd, origin, w, h, stride, bd, p, ri, pass, gotA[:bl], gotB[:bl], bs)
							step := 1
							if pass != 0 {
								step = 2
							}
							for y := -1; y < h+1; y += step {
								for x := -1; x < w+1; x++ {
									k := (y+3)*bs + x + 3
									if gotA[k] != wantA[k] || gotB[k] != wantB[k] {
										t.Fatalf("%dx%d bd=%d fill=%d params=%d radius=%d pass=%d x=%d y=%d A=%d/%d B=%d/%d", w, h, bd, fill, pi, ri, pass, x, y, gotA[k], wantA[k], gotB[k], wantB[k])
									}
								}
							}
							if !slices.Equal(gotA[bl:], tailA) || !slices.Equal(gotB[bl:], tailB) {
								t.Fatal("scratch tail written")
							}
						}
					}
				}
			}
		}
	}
}

// BenchmarkAMD64SGRIntermediate compares the scalar rolling A/B producer
// with the AVX2 one on a 64x64 processing unit.
func BenchmarkAMD64SGRIntermediate(b *testing.B) {
	const w, h = 64, 64
	stride := w + 9
	origin := 3*stride + 4
	bs := sgrBufferStride(w)
	bl := bs * (h + 6)
	dgd := make([]int32, stride*(h+6))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xb11)
	for i := range dgd {
		dgd[i] = int32(rnd.pseudoUniform(256))
	}
	aBuf, bBuf := make([]int32, bl), make([]int32, bl)
	p := SGRParameterTable[0]
	for _, c := range []struct {
		name     string
		ri, pass int
	}{{"r2/pass1", 0, 1}, {"r1/pass0", 1, 0}} {
		b.Run("shape="+c.name+"/impl=generic", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				calculateIntermediateRolling(dgd, origin, w, h, stride, 8, p, c.ri, c.pass, aBuf, bBuf, bs)
			}
		})
		b.Run("shape="+c.name+"/impl=avx2", func(b *testing.B) {
			if !sgrIntermediateAVX2 {
				b.Skip("no AVX2")
			}
			for i := 0; i < b.N; i++ {
				calculateIntermediateRollingAVX2(dgd, origin, w, h, stride, 8, p, c.ri, c.pass, aBuf, bBuf, bs)
			}
		})
	}
}
