// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestDirectionalZ1Z3AVX2MatchesScalar predicts every zone-1 and zone-3
// angle for 4..16 block sizes, with and without upsampled edges and with
// predicted extents past the visible block, with the AVX2 kernels on and
// off.
func TestDirectionalZ1Z3AVX2MatchesScalar(t *testing.T) {
	if !intraEdgeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(23))
	sizes := []int{4, 8, 16}
	compared := 0
	const origin = 16
	for trial := 0; trial < 30000; trial++ {
		w, h := sizes[rng.Intn(3)], sizes[rng.Intn(3)]
		pw, ph := w, h
		if trial%5 == 0 {
			pw, ph = w+4*rng.Intn(2), h+4*rng.Intn(2)
		}
		angle := 1 + rng.Intn(268)
		if angle == 90 || angle == 180 || (angle > 90 && angle < 180) {
			continue
		}
		edges := DirectionalEdges{AboveOrigin: origin, LeftOrigin: origin}
		if trial%2 == 0 {
			edges.UpsampleAbove, edges.UpsampleLeft = true, true
		}
		n := origin + 2*(pw+ph) + 8
		edges.Above, edges.Left = make([]uint16, n), make([]uint16, n)
		for i := range edges.Above {
			edges.Above[i] = uint16(rng.Intn(256))
			edges.Left[i] = uint16(rng.Intn(256))
		}
		stride := w + 5
		mk := func() frame.Plane {
			return frame.Plane{Pix: make([]byte, stride*h), Stride: stride, Width: w, Height: h}
		}
		want, got := mk(), mk()
		var errWant, errGot error
		withIntraEdgeScalar(func() {
			errWant = PredictDirectionalIntraPlaneBlockWithExtentBounded(want, 1, 8, 0, 0, w, h, pw, ph, angle, edges, true)
		})
		errGot = PredictDirectionalIntraPlaneBlockWithExtentBounded(got, 1, 8, 0, 0, w, h, pw, ph, angle, edges, true)
		if (errWant == nil) != (errGot == nil) {
			t.Fatalf("trial %d: error mismatch %v / %v", trial, errGot, errWant)
		}
		if errGot == nil {
			compared++
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("trial %d %dx%d (pred %dx%d) angle %d up=%v: mismatch\n got %v\nwant %v", trial, w, h, pw, ph, angle, edges.UpsampleAbove, got.Pix, want.Pix)
		}
	}
	t.Logf("compared %d predictions", compared)
	if compared < 5000 {
		t.Fatalf("only %d predictions compared", compared)
	}
}

// BenchmarkAMD64DirectionalZ1Z3 compares the scalar and AVX2 zone-1 and
// zone-3 predictors on the 4x4 upsampled blocks that dominate intra streams.
func BenchmarkAMD64DirectionalZ1Z3(b *testing.B) {
	const origin = 16
	edges := DirectionalEdges{AboveOrigin: origin, LeftOrigin: origin, UpsampleAbove: true, UpsampleLeft: true}
	edges.Above, edges.Left = make([]uint16, 64), make([]uint16, 64)
	for i := range edges.Above {
		edges.Above[i] = uint16(i * 13 % 256)
		edges.Left[i] = uint16(i * 29 % 256)
	}
	p := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 4, Height: 4}
	for _, c := range []struct {
		name  string
		angle int
	}{{"z1/4x4/up", 45}, {"z3/4x4/up", 203}} {
		b.Run("shape="+c.name+"/impl=generic", func(b *testing.B) {
			withIntraEdgeScalar(func() {
				for i := 0; i < b.N; i++ {
					_ = PredictDirectionalIntraPlaneBlockWithExtentBounded(p, 1, 8, 0, 0, 4, 4, 4, 4, c.angle, edges, true)
				}
			})
		})
		b.Run("shape="+c.name+"/impl=avx2", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = PredictDirectionalIntraPlaneBlockWithExtentBounded(p, 1, 8, 0, 0, 4, 4, 4, 4, c.angle, edges, true)
			}
		})
	}
}
