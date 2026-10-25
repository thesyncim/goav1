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

// TestAMD64LPFSB16Dispatch checks that the 10/12-bit superblock filters run
// the AVX2 kernels on an AVX2 host: checkasm-shaped calls are handled by the
// vector groups and match the C reference.
func TestAMD64LPFSB16Dispatch(t *testing.T) {
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
				c := newCheckasmSB16Case(int64(77+s), vertical, luma, s)
				want := append([]byte(nil), c.pix...)
				got := append([]byte(nil), c.pix...)
				var ws, gs SBStats
				var simd int
				if vertical {
					ws = filterSBVertical16Scalar(want, c.off, c.stride, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					gs, simd = avx16SBVertical(got, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					ws = filterSBHorizontal16Scalar(want, c.off, c.stride, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					gs, simd = avx16SBHorizontal(got, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
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

// TestAMD64LPFSB16MatchesScalar drives the AVX2 10/12-bit superblock filters
// directly on random geometry, content, masks and levels.
func TestAMD64LPFSB16MatchesScalar(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	rng := rand.New(rand.NewSource(5))
	const w, h = 160, 160
	const stride = 2 * w
	simdGroups := 0
	for iter := 0; iter < 6000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		pix := sbTestPlane16(rng, w, h, bitDepth)
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
			x := 4 * (1 + rng.Intn(w/4-1))
			if vm[2] != 0 {
				x = 8 * (1 + rng.Intn(w/8-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*stride + 2*x
			lvlOff := (y0/4)*lvlStride + x/4
			ws = filterSBVertical16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs, simd = avx16SBVertical(got, off, stride, 2*(w-x), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (1 + rng.Intn(h/4-1))
			if vm[2] != 0 {
				y = 8 * (1 + rng.Intn(h/8-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*stride + 2*x0
			lvlOff := (y/4)*lvlStride + x0/4
			ws = filterSBHorizontal16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs, simd = avx16SBHorizontal(got, off, stride, 2*(w-x0), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		simdGroups += simd
		if ws != gs {
			t.Fatalf("iter %d: stats %+v, want %+v", iter, gs, ws)
		}
		if !bytes.Equal(got, want) {
			for i := 0; i < len(got); i += 2 {
				if got[i] != want[i] || got[i+1] != want[i+1] {
					px := i / 2
					t.Fatalf("iter %d luma=%v bd=%d vm=%x: first diff at (%d,%d): got %d want %d", iter, luma, bitDepth, vm, px%w, px/w, ld16(got, i), ld16(want, i))
				}
			}
		}
	}
	if simdGroups == 0 {
		t.Fatal("no group took the AVX2 kernel")
	}
}

// TestAMD64LPFSB16ZeroAlloc keeps the AVX2 10/12-bit filters allocation-free.
func TestAMD64LPFSB16ZeroAlloc(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	for _, vertical := range []bool{true, false} {
		for _, luma := range []bool{true, false} {
			c := newCheckasmSB16Case(5, vertical, luma, 1)
			allocs := testing.AllocsPerRun(50, func() {
				if vertical {
					FilterSBVertical16(c.pix, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					FilterSBHorizontal16(c.pix, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
			})
			if allocs != 0 {
				t.Fatalf("vertical=%v luma=%v: %v allocs per call", vertical, luma, allocs)
			}
		}
	}
}

// BenchmarkAMD64LPFSB16 times the checkasm-shaped 10/12-bit superblock calls
// on the unit-by-unit reference (impl=generic) and the AVX2 kernels
// (impl=avx2); dir=h is lpf_h_sb (vertical edges), dir=v lpf_v_sb.
func BenchmarkAMD64LPFSB16(b *testing.B) {
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
				var cs [cases]checkasmSB16Case
				for i := range cs {
					cs[i] = newCheckasmSB16Case(int64(1000*s+i), vertical, luma, s)
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
								FilterSBVertical16(w, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
							} else {
								FilterSBHorizontal16(w, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
							}
							i++
						}
					})
				}
			}
		}
	}
}
