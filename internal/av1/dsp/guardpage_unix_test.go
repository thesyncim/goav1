// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package dsp

import (
	"bytes"
	"fmt"
	"math/rand"
	"slices"
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
// residuals and masks exactly as large as the block (stride = width), at
// both page edges, and compares with heap-allocated runs.
func TestDSPStaysInsideGuardedBlocks(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for iter := range 600 {
		w, h := 1+rng.Intn(66), 1+rng.Intn(66)
		before := iter%2 == 1
		bps := 1 + iter%4/2
		bd, maxV := uint8(8), uint16(255)
		if bps == 2 {
			bd, maxV = 10, 1023
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
		check("add residual", func(p frame.Plane) error {
			return AddResidualPlaneBlock(p, bps, bd, 0, 0, w, h, res, w)
		})
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
		check("fill", func(p frame.Plane) error {
			return FillPlaneBlock(p, bps, 0, 0, w, h, maxV/3)
		})
		if w >= 8 && h >= 8 {
			var lo, hi uint16
			var err error
			guardpage.Check(t, name+" minmax", func() {
				lo, hi, err = MinMaxAbsDiff8x8(dst.Pix[(h-8)*dst.Stride+(w-8)*bps:], dst.Stride, src.Pix[(h-8)*src.Stride+(w-8)*bps:], src.Stride, bps)
			})
			hlo, hhi, herr := MinMaxAbsDiff8x8(heap.Pix[(h-8)*heap.Stride+(w-8)*bps:], heap.Stride, src.Pix[(h-8)*src.Stride+(w-8)*bps:], src.Stride, bps)
			if lo != hlo || hi != hhi || (err == nil) != (herr == nil) {
				t.Fatalf("%s minmax: guarded run differs", name)
			}
		}
		subX, subY := rng.Intn(2) == 1, rng.Intn(2) == 1
		mw, mh := w, h
		if subX {
			mw *= 2
		}
		if subY {
			mh *= 2
		}
		mask := guardBuf[uint8](t, (mh-1)*mw+mw, before)
		for i := range mask {
			mask[i] = uint8(rng.Intn(65))
		}
		b0, b1 := guardBuf[uint16](t, (h-1)*w+w, before), guardBuf[uint16](t, (h-1)*w+w, before)
		for i := range b0 {
			b0[i], b1[i] = uint16(rng.Intn(int(maxV)+1)), uint16(rng.Intn(int(maxV)+1))
		}
		out := guardBuf[uint16](t, (h-1)*w+w, before)
		want := make([]uint16, len(out))
		var err error
		guardpage.Check(t, name+" blend", func() {
			err = BlendA64Mask(out, w, b0, w, b1, w, mask, mw, w, h, subX, subY, bd)
		})
		herr := BlendA64Mask(want, w, slices.Clone(b0), w, slices.Clone(b1), w, slices.Clone(mask), mw, w, h, subX, subY, bd)
		if (err == nil) != (herr == nil) || !slices.Equal(out, want) {
			t.Fatalf("%s blend: guarded run differs (%v / %v)", name, err, herr)
		}
	}
}
