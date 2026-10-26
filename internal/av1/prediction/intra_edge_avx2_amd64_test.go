// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"math/rand"
	"slices"
	"testing"
)

// withIntraEdgeScalar runs f with the AVX2 edge kernels disabled.
func withIntraEdgeScalar(f func()) {
	saved := intraEdgeAVX2
	intraEdgeAVX2 = false
	defer func() { intraEdgeAVX2 = saved }()
	f()
}

func TestFilterIntraEdgeAVX2MatchesScalar(t *testing.T) {
	if !intraEdgeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(17))
	scratch := make([]uint16, intraEdgeMaxSize)
	for trial := 0; trial < 20000; trial++ {
		n := 1 + rng.Intn(intraEdgeMaxSize)
		if trial%3 != 0 {
			n = 1 + rng.Intn(40)
		}
		bd := []uint8{8, 10, 12}[trial%3]
		strength := uint8(1 + rng.Intn(3))
		src := make([]uint16, n)
		for i := range src {
			src[i] = uint16(rng.Intn(1 << bd))
			if trial%7 == 0 {
				src[i] = 1<<bd - 1
			}
		}
		want, got := slices.Clone(src), slices.Clone(src)
		withIntraEdgeScalar(func() {
			if err := FilterIntraEdgeBounded(want, scratch, strength, bd, false); err != nil {
				t.Fatal(err)
			}
		})
		if err := FilterIntraEdgeBounded(got, scratch, strength, bd, false); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d n=%d bd=%d strength=%d:\n got %v\nwant %v", trial, n, bd, strength, got, want)
		}
	}
}

func TestUpsampleIntraEdgeAVX2MatchesScalar(t *testing.T) {
	if !intraEdgeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(18))
	scratch := make([]uint16, intraEdgeMaxSize)
	for trial := 0; trial < 20000; trial++ {
		size := 1 + rng.Intn(intraEdgeMaxUpsampleSize)
		origin := 2 + rng.Intn(4)
		bd := []uint8{8, 10, 12}[trial%3]
		src := make([]uint16, origin+2*size+3)
		for i := range src {
			src[i] = uint16(rng.Intn(1 << bd))
			if trial%5 == 0 && i%2 == 0 {
				src[i] = 1<<bd - 1
			}
		}
		want, got := slices.Clone(src), slices.Clone(src)
		withIntraEdgeScalar(func() {
			if err := UpsampleIntraEdgeBounded(want, origin, size, scratch, bd, false); err != nil {
				t.Fatal(err)
			}
		})
		if err := UpsampleIntraEdgeBounded(got, origin, size, scratch, bd, false); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("trial %d size=%d origin=%d bd=%d:\n got %v\nwant %v", trial, size, origin, bd, got, want)
		}
	}
}

// BenchmarkAMD64IntraEdge compares the scalar and AVX2 edge filter and
// upsampler on the edge lengths of small directional blocks.
func BenchmarkAMD64IntraEdge(b *testing.B) {
	scratch := make([]uint16, intraEdgeMaxSize)
	src := make([]uint16, 64)
	for i := range src {
		src[i] = uint16(i * 7 % 256)
	}
	edge := make([]uint16, 64)
	run := func(name string, f func()) {
		b.Run(name+"/impl=generic", func(b *testing.B) {
			withIntraEdgeScalar(func() {
				for i := 0; i < b.N; i++ {
					f()
				}
			})
		})
		b.Run(name+"/impl=avx2", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f()
			}
		})
	}
	run("shape=filter17/s1", func() {
		copy(edge, src)
		_ = FilterIntraEdgeBounded(edge[:17], scratch, 1, 8, true)
	})
	run("shape=filter33/s3", func() {
		copy(edge, src)
		_ = FilterIntraEdgeBounded(edge[:33], scratch, 3, 8, true)
	})
	run("shape=upsample8", func() {
		copy(edge, src)
		_ = UpsampleIntraEdgeBounded(edge, 2, 8, scratch, 8, true)
	})
}
