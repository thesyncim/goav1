//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"bytes"
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"testing"
)

// TestFilterSIMDDispatchBoundAMD64 checks that the amd64 dispatch slots resolve
// to the Go-native SIMD kernels whenever the CPU reports AVX2. Rosetta does not
// advertise AVX2, so the check skips there; the direct kernel tests still run.
func TestFilterSIMDDispatchBoundAMD64(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not reported by CPUID; dispatch stays on pure Go")
	}
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter4EdgeImpl", filter4EdgeImpl, filter4EdgeSIMD},
		{"filter4Edge16Impl", filter4Edge16Impl, filter4Edge16SIMD},
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeSIMD},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeSIMD},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeSIMD},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}

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
				var simd int
				if vertical {
					filterSBVertical16Scalar(want, c.off, c.stride, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					simd = avx16SBVertical(got, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					filterSBHorizontal16Scalar(want, c.off, c.stride, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					simd = avx16SBHorizontal(got, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
				if simd == 0 {
					t.Fatalf("vertical=%v luma=%v s=%d: no group took the AVX2 kernel", vertical, luma, s)
				}
				if !bytes.Equal(got, want) {
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
		var simd int
		if rng.Intn(2) == 0 {
			x := 4 * (1 + rng.Intn(w/4-1))
			if vm[2] != 0 {
				x = 8 * (1 + rng.Intn(w/8-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*stride + 2*x
			lvlOff := (y0/4)*lvlStride + x/4
			filterSBVertical16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			simd = avx16SBVertical(got, off, stride, 2*(w-x), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (1 + rng.Intn(h/4-1))
			if vm[2] != 0 {
				y = 8 * (1 + rng.Intn(h/8-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*stride + 2*x0
			lvlOff := (y/4)*lvlStride + x0/4
			filterSBHorizontal16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			simd = avx16SBHorizontal(got, off, stride, 2*(w-x0), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		simdGroups += simd
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
				var simd int
				if vertical {
					filterSBVertical8Scalar(want, c.off, c.stride, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					simd = avxSBVertical8(got, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					filterSBHorizontal8Scalar(want, c.off, c.stride, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
					simd = avxSBHorizontal8(got, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
				if simd == 0 {
					t.Fatalf("vertical=%v luma=%v s=%d: no group took the AVX2 kernel", vertical, luma, s)
				}
				if !bytes.Equal(got, want) {
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
		var simd int
		if rng.Intn(2) == 0 {
			x := 4 * (2 + rng.Intn(w/4-4))
			if vm[2] != 0 {
				x = 16 * (1 + rng.Intn(w/16-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*w + x
			lvlOff := (y0/4)*lvlStride + x/4
			filterSBVertical8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			simd = avxSBVertical8(got, off, w, w-x, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (2 + rng.Intn(h/4-4))
			if vm[2] != 0 {
				y = 16 * (1 + rng.Intn(h/16-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*w + x0
			lvlOff := (y/4)*lvlStride + x0/4
			filterSBHorizontal8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			simd = avxSBHorizontal8(got, off, w, w-x0, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		simdGroups += simd
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

// BenchmarkAMD64LPFSB times superblock calls on the unit-by-unit reference
// (impl=generic) and the AVX2 kernels (impl=avx2). dir=h is the vertical-edge
// column filter (dav1d lpf_h_sb), dir=v the horizontal-edge row filter
// (lpf_v_sb). content=checkasm reproduces dav1d checkasm's check_lpf_sb draws
// (a quarter of the edges are random pixels that fail the filter mask at the
// first tap); content=video is decoded-like: smooth blocks with a small step
// at every edge, so most edges filter, as in real streams.
func BenchmarkAMD64LPFSB(b *testing.B) {
	if !archsimd.X86.AVX2() {
		b.Skip("AVX2 unavailable")
	}
	for _, content := range []string{"checkasm", "video"} {
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
						if content == "video" {
							cs[i] = newVideoSBCase(int64(1000*s+i), vertical, luma, s)
						} else {
							cs[i] = newCheckasmSBCase(int64(1000*s+i), vertical, luma, s)
						}
					}
					for _, impl := range []string{"generic", "avx2"} {
						b.Run(fmt.Sprintf("content=%s/dir=%s/plane=%s/w=%d/impl=%s", content, dir, plane, wd, impl), func(b *testing.B) {
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
								if i&(cases-1) == 0 && i > 0 {
									// Refresh the inputs so repeated filtering does
									// not flatten them.
									b.StopTimer()
									for k := range cs {
										copy(work[k], cs[k].pix)
									}
									b.StartTimer()
								}
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
}

// newVideoSBCase is newCheckasmSBCase's geometry with decoded-like content:
// every unit is an edge of the widest width the strength allows, levels are
// mid-range, and the pixels are smooth with a small step across the edge.
func newVideoSBCase(seed int64, vertical, luma bool, strength int) checkasmSBCase {
	c := newCheckasmSBCase(seed, vertical, luma, strength)
	rng := rand.New(rand.NewSource(seed + 99))
	nBlks := 32
	if !luma {
		nBlks = 16
	}
	c.vm = [3]uint32{}
	for j := 0; j < nBlks; j++ {
		c.vm[strength] |= 1 << uint(j)
	}
	for i := range len(c.lvl) / 4 {
		c.lvl[4*(i)+0] = uint8(16 + rng.Intn(24))
	}
	for i := 0; i < 4*nBlks; i++ {
		base := 60 + rng.Intn(120)
		step := rng.Intn(9) - 4
		for k := -8; k < 8; k++ {
			v := base + rng.Intn(3) - 1
			if k >= 0 {
				v += step
			}
			var o int
			if vertical {
				o = i*c.stride + c.off + k
			} else {
				o = c.off + i + k*c.stride
			}
			c.pix[o] = uint8(v)
		}
	}
	return c
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
