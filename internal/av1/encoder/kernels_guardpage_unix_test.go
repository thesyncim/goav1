// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package encoder

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/guardpage"
)

func guardBytes(t *testing.T, rng *rand.Rand, n int, before bool) []byte {
	var b []byte
	if before {
		b = guardpage.Before[byte](t, n)
	} else {
		b = guardpage.After[byte](t, n)
	}
	for i := range b {
		b[i] = byte(rng.Intn(256))
	}
	return b
}

// blockLen is the shortest buffer holding a w x h block at the given stride.
func blockLen(w, h, stride int) int { return (h-1)*stride + w }

// TestEncoderKernelsStayInsideGuardedBlocks runs the SAD, pixel statistics
// and reconstruction kernels on buffers exactly as long as the block they cover (stride equal to and wider than the width), against
// a guard page on either side, and compares each result with a heap run.
func TestEncoderKernelsStayInsideGuardedBlocks(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	type stats func([]byte, int, []byte, int) (uint32, int32)
	pixelStats := []struct {
		w, h int
		f    stats
	}{
		{4, 4, pixelStats4x4}, {8, 8, pixelStats8x8}, {16, 8, pixelStats16x8}, {8, 16, pixelStats8x16},
		{16, 16, pixelStats16x16}, {32, 16, pixelStats32x16}, {16, 32, pixelStats16x32}, {32, 32, pixelStats32x32},
		{64, 32, pixelStats64x32}, {32, 64, pixelStats32x64},
	}
	sads := []struct {
		n int
		f func(src, ref []byte, stride int) int
	}{
		{8, func(s, r []byte, st int) int { return sad8x8(s, r, st, 1<<30) }},
		{16, sad16x16}, {32, sad32x32}, {64, sad64x64},
	}
	duals := []struct {
		n int
		f func([]byte, int, []byte, int) int
	}{{8, sad8x8Dual}, {16, sad16x16Dual}, {32, sad32x32Dual}, {64, sad64x64Dual}}
	step4s := []struct {
		n int
		f func(src, ref []byte, stride int) (int, int, int, int)
	}{{8, sad8x8x4Step4}, {16, sad16x16x4Step4}, {32, sad32x32x4Step4}}
	x4s := []struct {
		n int
		f func(src, r0, r1, r2, r3 []byte, stride int) (int, int, int, int)
	}{{8, sad8x8x4}, {16, sad16x16x4}, {32, sad32x32x4}}

	for _, before := range []bool{false, true} {
		for _, pad := range []int{0, 24} {
			name := fmt.Sprintf("before=%v pad=%d", before, pad)
			for _, k := range sads {
				st := k.n + pad
				s, r := guardBytes(t, rng, blockLen(k.n, k.n, st), before), guardBytes(t, rng, blockLen(k.n, k.n, st), before)
				var got int
				guardpage.Check(t, fmt.Sprintf("%s sad%d", name, k.n), func() { got = k.f(s, r, st) })
				if want := k.f(slices.Clone(s), slices.Clone(r), st); got != want {
					t.Fatalf("%s sad%d: %d want %d", name, k.n, got, want)
				}
			}
			for _, k := range duals {
				ss, rs := k.n+pad, k.n+2*pad+8
				s, r := guardBytes(t, rng, blockLen(k.n, k.n, ss), before), guardBytes(t, rng, blockLen(k.n, k.n, rs), before)
				var got int
				guardpage.Check(t, fmt.Sprintf("%s sad%d dual", name, k.n), func() { got = k.f(s, ss, r, rs) })
				if want := k.f(slices.Clone(s), ss, slices.Clone(r), rs); got != want {
					t.Fatalf("%s sad%d dual: %d want %d", name, k.n, got, want)
				}
			}
			for _, k := range step4s {
				st := k.n + 12 + pad
				s, r := guardBytes(t, rng, blockLen(k.n, k.n, st), before), guardBytes(t, rng, blockLen(k.n+12, k.n, st), before)
				var a [4]int
				guardpage.Check(t, fmt.Sprintf("%s sad%d step4", name, k.n), func() { a[0], a[1], a[2], a[3] = k.f(s, r, st) })
				var b [4]int
				b[0], b[1], b[2], b[3] = k.f(slices.Clone(s), slices.Clone(r), st)
				if a != b {
					t.Fatalf("%s sad%d step4 differs", name, k.n)
				}
			}
			for _, k := range x4s {
				st := k.n + pad
				n := blockLen(k.n, k.n, st)
				s := guardBytes(t, rng, n, before)
				r := [4][]byte{guardBytes(t, rng, n, before), guardBytes(t, rng, n, before), guardBytes(t, rng, n, before), guardBytes(t, rng, n, before)}
				var a [4]int
				guardpage.Check(t, fmt.Sprintf("%s sad%d x4", name, k.n), func() { a[0], a[1], a[2], a[3] = k.f(s, r[0], r[1], r[2], r[3], st) })
				var b [4]int
				b[0], b[1], b[2], b[3] = k.f(slices.Clone(s), slices.Clone(r[0]), slices.Clone(r[1]), slices.Clone(r[2]), slices.Clone(r[3]), st)
				if a != b {
					t.Fatalf("%s sad%d x4 differs", name, k.n)
				}
			}
			for _, k := range pixelStats {
				ss, rs := k.w+pad, k.w+2*pad
				s, r := guardBytes(t, rng, blockLen(k.w, k.h, ss), before), guardBytes(t, rng, blockLen(k.w, k.h, rs), before)
				var e, m uint32
				var c, d int32
				guardpage.Check(t, fmt.Sprintf("%s stats%dx%d", name, k.w, k.h), func() { e, c = k.f(s, ss, r, rs) })
				m, d = k.f(slices.Clone(s), ss, slices.Clone(r), rs)
				if e != m || c != d {
					t.Fatalf("%s stats%dx%d differs", name, k.w, k.h)
				}
			}
			for _, w := range []int{4, 8, 16, 32, 64} {
				for _, h := range []int{4, 8, 16, 32, 64} {
					stride, ps := w+pad+5, w+pad
					px, py := 3, 2
					recon := guardBytes(t, rng, (py+h-1)*stride+px+w, before)
					pred := guardBytes(t, rng, blockLen(w, h, ps), before)
					raw := make([]int32, w*h)
					for i := range raw {
						raw[i] = int32(rng.Intn(8192) - 4096)
					}
					ref := slices.Clone(recon)
					tag := fmt.Sprintf("%s recon %dx%d", name, w, h)
					guardpage.Check(t, tag+" copy", func() { copyPredToReconBlock(recon, pred, ps, stride, px, py, w, h) })
					copyPredToReconBlock(ref, slices.Clone(pred), ps, stride, px, py, w, h)
					guardpage.Check(t, tag+" const", func() { addConstantResidualToPredReconBlock(recon, pred, ps, stride, px, py, w, h, -37) })
					addConstantResidualToPredReconBlock(ref, slices.Clone(pred), ps, stride, px, py, w, h, -37)
					dst := recon[py*stride+px:]
					guardpage.Check(t, tag+" raw", func() { addRawResidualToPredReconBlock(dst, stride, pred, ps, raw, w, h) })
					addRawResidualToPredReconBlock(ref[py*stride+px:], stride, slices.Clone(pred), ps, raw, w, h)
					if !slices.Equal(recon, ref) {
						t.Fatalf("%s differs", tag)
					}
				}
			}
		}
	}
}

// TestEncoderPlaneKernelsStayInsideGuardedPlanes covers the kernels that read
// whole planes: the 2x2 downsample of the HME pyramid (on source planes
// exactly 2h rows of 2w samples) and the 8-tap predictor (on a reference whose
// last byte is the last one pred8Fits allows).
func TestEncoderPlaneKernelsStayInsideGuardedPlanes(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, before := range []bool{false, true} {
		for _, sz := range [][2]int{{1, 1}, {7, 3}, {16, 8}, {17, 9}, {33, 20}, {64, 36}, {183, 103}} {
			w, h := sz[0], sz[1]
			for _, pad := range []int{0, 13} {
				ss, ds := 2*w+pad, w+pad
				src := guardBytes(t, rng, blockLen(2*w, 2*h, ss), before)
				var dst []byte
				if before {
					dst = guardpage.Before[byte](t, blockLen(w, h, ds))
				} else {
					dst = guardpage.After[byte](t, blockLen(w, h, ds))
				}
				want := make([]byte, len(dst))
				name := fmt.Sprintf("downsample %dx%d pad=%d before=%v", w, h, pad, before)
				guardpage.Check(t, name, func() { svtDownsample2DArch(dst, ds, src, ss, w, h) })
				svtDownsample2D(want, ds, slices.Clone(src), ss, w, h)
				if !slices.Equal(dst, want) {
					t.Fatalf("%s differs from the scalar reference", name)
				}
			}
		}
		for _, w := range []int{8, 16, 24, 32, 64} {
			for _, h := range []int{8, 16, 32, 64} {
				refX, refY := 3, 3
				width, height := refX+w+4, refY+h+5
				stride := width + 12
				pix := guardBytes(t, rng, (refY+h+4)*stride+refX+w+16, before)
				ref := frame.Plane{Pix: pix, Stride: stride, Width: width, Height: height}
				var im [pred8IMLen]int16
				for _, sub := range [][2]int{{0, 0}, {5, 0}, {0, 9}, {7, 3}} {
					name := fmt.Sprintf("pred8 %dx%d sub=%v before=%v", w, h, sub, before)
					dst := guardBytes(t, rng, blockLen(w, h, w), before)
					want := slices.Clone(dst)
					var ok bool
					guardpage.Check(t, name, func() {
						ok = pred8Predict(dst, w, &ref, refX, refY, w, h, sub[0], sub[1], motion.RegularFilters, &im)
					})
					heapRef := frame.Plane{Pix: slices.Clone(pix), Stride: stride, Width: width, Height: height}
					if ok != pred8Predict(want, w, &heapRef, refX, refY, w, h, sub[0], sub[1], motion.RegularFilters, &im) || !slices.Equal(dst, want) {
						t.Fatalf("%s differs from the heap run", name)
					}
				}
			}
		}
	}
}
