// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"bytes"
	"fmt"
	"math/rand"
	"simd/archsimd"
	"testing"
)

// TestAMD64LPFSBDispatch checks that the 8-bit superblock filters run the AVX2
// kernels on an AVX2 host: the gate is bound, and checkasm-shaped calls are
// handled by the vector groups (not the unit-by-unit fallback) and match the
// C reference.
func TestAMD64LPFSBDispatch(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	if !sbAVX2 {
		t.Fatal("AVX2 superblock loop filter is not bound")
	}
	for _, vertical := range []bool{true, false} {
		for _, luma := range []bool{true, false} {
			n := 3
			if !luma {
				n = 2
			}
			for s := 0; s < n; s++ {
				c := newCheckasmSBCase(int64(77+s), vertical, luma, s)
				want := append([]byte(nil), c.pix...)
				got := append([]byte(nil), c.pix...)
				var ws, gs SBStats
				var simd int
				if vertical {
					ws = filterSBVertical8Scalar(want, c.off, c.stride, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					gs, simd = avxSBVertical8(got, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					ws = filterSBHorizontal8Scalar(want, c.off, c.stride, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					gs, simd = avxSBHorizontal8(got, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
				if simd == 0 {
					t.Fatalf("vertical=%v luma=%v s=%d: no group took the AVX2 kernel", vertical, luma, s)
				}
				if ws != gs || !bytes.Equal(got, want) {
					t.Fatalf("vertical=%v luma=%v s=%d: AVX2 differs from the reference", vertical, luma, s)
				}
			}
		}
	}
}

// TestAMD64LPFSBMatchesScalar drives the AVX2 superblock filters directly on
// random geometry: mixed widths and empty units inside one vmask byte, row
// ends that cut a 32-sample group (rowBytes not a multiple of 32), rows close
// to the plane end, and every sharpness and level component.
func TestAMD64LPFSBMatchesScalar(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	rng := rand.New(rand.NewSource(3))
	const w, h = 200, 200
	simdGroups := 0
	for iter := 0; iter < 6000; iter++ {
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		var pix []byte
		if rng.Intn(2) == 0 {
			pix = sbTestPlane(rng, w, h)
		} else {
			// Mostly flat content so the 6/8/16-tap paths fire.
			pix = make([]byte, w*h)
			for y := 0; y < h; y++ {
				base := rng.Intn(256)
				for x := 0; x < w; x++ {
					v := base + rng.Intn(3) - 1
					if rng.Intn(40) == 0 {
						v += rng.Intn(60) - 30
					}
					pix[y*w+x] = uint8(min(max(v, 0), 255))
				}
			}
		}
		lvlStride := w / 4
		lvl := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		var ws, gs SBStats
		var simd int
		if rng.Intn(2) == 0 {
			x := 4 * (2 + rng.Intn(w/4-4))
			if vm[2] != 0 {
				x = 16 * (1 + rng.Intn(w/16-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*w + x
			lvlOff := (y0/4)*lvlStride + x/4
			ws = filterSBVertical8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs, simd = avxSBVertical8(got, off, w, w-x, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (2 + rng.Intn(h/4-4))
			if vm[2] != 0 {
				y = 16 * (1 + rng.Intn(h/16-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*w + x0
			lvlOff := (y/4)*lvlStride + x0/4
			ws = filterSBHorizontal8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs, simd = avxSBHorizontal8(got, off, w, w-x0, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		simdGroups += simd
		if ws != gs {
			t.Fatalf("iter %d: stats %+v, want %+v", iter, gs, ws)
		}
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("iter %d luma=%v vm=%x: first diff at (%d,%d): got %d want %d", iter, luma, vm, i%w, i/w, got[i], want[i])
				}
			}
		}
	}
	if simdGroups == 0 {
		t.Fatal("no group took the AVX2 kernel")
	}
}

// BenchmarkAMD64LPFSB times the checkasm-shaped superblock calls on the
// unit-by-unit reference (impl=generic) and the AVX2 kernels (impl=avx2).
// dir=h is the vertical-edge column filter (dav1d lpf_h_sb), dir=v the
// horizontal-edge row filter (lpf_v_sb).
func BenchmarkAMD64LPFSB(b *testing.B) {
	if !archsimd.X86.AVX2() {
		b.Skip("AVX2 unavailable")
	}
	for _, vertical := range []bool{true, false} {
		for _, luma := range []bool{true, false} {
			widths := []int{4, 8, 16}
			plane := "y"
			if !luma {
				widths = []int{4, 6}
				plane = "uv"
			}
			dir := "v"
			if vertical {
				dir = "h"
			}
			for s, wd := range widths {
				const cases = 16
				var cs [cases]checkasmSBCase
				for i := range cs {
					cs[i] = newCheckasmSBCase(int64(1000*s+i), vertical, luma, s)
				}
				for _, impl := range []string{"generic", "avx2"} {
					b.Run(fmt.Sprintf("dir=%s/plane=%s/w=%d/impl=%s", dir, plane, wd, impl), func(b *testing.B) {
						saved := sbAVX2
						sbAVX2 = impl == "avx2"
						defer func() { sbAVX2 = saved }()
						var work [cases][]byte
						for i := range cs {
							work[i] = append([]byte(nil), cs[i].pix...)
						}
						i := 0
						for b.Loop() {
							c := &cs[i&(cases-1)]
							w := work[i&(cases-1)]
							if vertical {
								FilterSBVertical8(w, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
							} else {
								FilterSBHorizontal8(w, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
							}
							i++
						}
					})
				}
			}
		}
	}
}

// TestAMD64LPFSBZeroAlloc keeps the AVX2 superblock filters allocation-free
// (stack tap arrays must not escape, also under checkptr builds).
func TestAMD64LPFSBZeroAlloc(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	for _, vertical := range []bool{true, false} {
		for _, luma := range []bool{true, false} {
			c := newCheckasmSBCase(5, vertical, luma, 2-btoi(!luma))
			allocs := testing.AllocsPerRun(50, func() {
				if vertical {
					FilterSBVertical8(c.pix, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					FilterSBHorizontal8(c.pix, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
			})
			if allocs != 0 {
				t.Fatalf("vertical=%v luma=%v: %v allocs per call", vertical, luma, allocs)
			}
		}
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
