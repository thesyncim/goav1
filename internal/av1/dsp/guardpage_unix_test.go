// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package dsp

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/guardpage"
)

func guardBuf[T any](t *testing.T, n int, before bool) []T {
	if before {
		return guardpage.Before[T](t, n)
	}
	return guardpage.After[T](t, n)
}

// TestDSPStaysInsideGuardedBlocks runs the plane block kernels on planes,
// and residuals exactly as large as the block (stride = width), at
// both page edges, and compares with heap-allocated runs.
func TestDSPStaysInsideGuardedBlocks(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for iter := range 600 {
		w, h := 1+rng.Intn(66), 1+rng.Intn(66)
		before := iter%2 == 1
		bps := 1 + iter%4/2
		maxV := uint16(255)
		if bps == 2 {
			maxV = 1023
		}
		name := fmt.Sprintf("iter %d %dx%d bps=%d before=%v", iter, w, h, bps, before)
		n := (h-1)*w*bps + w*bps
		pix := guardBuf[byte](t, n, before)
		for i := 0; i < n; i += bps {
			v := rng.Intn(int(maxV) + 1)
			pix[i] = byte(v)
			if bps == 2 {
				pix[i+1] = byte(v >> 8)
			}
		}
		dst := frame.Plane{Pix: pix, Stride: w * bps, Width: w, Height: h}
		heap := frame.Plane{Pix: bytes.Clone(pix), Stride: w * bps, Width: w, Height: h}
		res := guardBuf[int16](t, (h-1)*w+w, before)
		for i := range res {
			res[i] = int16(rng.Intn(512) - 256)
		}
		raw := guardBuf[int32](t, (h-1)*w+w, before)
		for i := range raw {
			raw[i] = int32(rng.Intn(4096) - 2048)
		}
		src := frame.Plane{Pix: bytes.Clone(pix), Stride: w * bps, Width: w, Height: h}
		check := func(op string, f func(p frame.Plane) error) {
			var err error
			guardpage.Check(t, name+" "+op, func() { err = f(dst) })
			herr := f(heap)
			if (err == nil) != (herr == nil) || !bytes.Equal(dst.Pix, heap.Pix) {
				t.Fatalf("%s %s: guarded run differs (%v / %v)", name, op, err, herr)
			}
		}
		check("add residual trusted", func(p frame.Plane) error {
			AddResidualPlaneBlockTrusted(p.Pix, p.Stride, bps, maxV, w, h, res, w)
			return nil
		})
		check("add constant", func(p frame.Plane) error {
			AddConstantResidualPlaneBlockTrusted(p.Pix, p.Stride, bps, maxV, w, h, res[0])
			return nil
		})
		check("add raw", func(p frame.Plane) error {
			AddRawTransformPlaneBlockTrusted(p.Pix, p.Stride, bps, maxV, w, h, raw, w)
			return nil
		})
		check("copy", func(p frame.Plane) error {
			return CopyPlaneBlock(p, src, bps, 0, 0, 0, 0, w, h)
		})
	}
}
