// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// These benchmarks reproduce dav1d's tests/checkasm/loopfilter.c check_lpf_sb
// for 8-bit luma and chroma: one superblock call over 32 (luma) or 16
// (chroma) units, edge pixels shaped by init_lpf_border so that the long-flat,
// short-flat, normal/hev and unfiltered cases all occur, unit widths drawn as
// checkasm draws them, and levels in 0..63 with dav1d's level-from-previous
// fallback layout. The ns/op is directly comparable with
// `checkasm -t loopfilter_8bpc -b` (lpf_h_sb_* is FilterSBVertical8, the
// vertical-edge column filter; lpf_v_sb_* is FilterSBHorizontal8).

func checkasmClip(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

type checkasmSBCase struct {
	pix       []byte
	off       int
	stride    int
	rowBytes  int
	vm        [3]uint32
	lvl       [][4]uint8
	lvlOff    int
	lvlStride int
	lut       SBLUT
}

// newCheckasmSBCase builds one check_lpf_sb call. vertical selects the
// vertical-edge (column) filter; strength is checkasm's loop index i.
func newCheckasmSBCase(seed int64, vertical, luma bool, strength int) checkasmSBCase {
	rng := rand.New(rand.NewSource(seed))
	nBlks := 32
	if !luma {
		nBlks = 16
	}
	sharp := rng.Intn(8)
	c := checkasmSBCase{lut: NewSBLUT(uint8(sharp))}
	// checkasm: l[32*2][4]; vertical edges read l[2j+1] (fallback l[2j]),
	// horizontal edges read l[32+j] (fallback l[j]).
	c.lvl = make([][4]uint8, 64)
	for j := 0; j < nBlks; j++ {
		idx := rng.Intn(strength + 2)
		if idx != 0 {
			c.vm[idx-1] |= 1 << uint(j)
		}
		if vertical {
			c.lvl[2*j][0] = uint8(rng.Intn(64))
			c.lvl[2*j+1][0] = uint8(rng.Intn(64))
		} else {
			c.lvl[j][0] = uint8(rng.Intn(64))
			c.lvl[32+j][0] = uint8(rng.Intn(64))
		}
	}
	if vertical {
		// 16 taps per row around the edge; a wider row keeps the kernel on the
		// in-frame (16-byte row) path the frame apply takes.
		c.stride = 32
		c.pix = make([]byte, c.stride*(4*nBlks))
		c.off = 8
		c.rowBytes = c.stride - 8
		c.lvlOff, c.lvlStride = 1, 2
	} else {
		c.stride = 4 * nBlks
		c.pix = make([]byte, c.stride*16)
		c.off = 8 * c.stride
		c.rowBytes = c.stride
		c.lvlOff, c.lvlStride = 32, 32
	}
	for i := 0; i < 4*nBlks; i++ {
		x := i >> 2
		var L uint8
		if vertical {
			L = c.lvl[2*x+1][0]
			if L == 0 {
				L = c.lvl[2*x][0]
			}
		} else {
			L = c.lvl[32+x][0]
			if L == 0 {
				L = c.lvl[x][0]
			}
		}
		E, I := int(c.lut.E[L]), int(c.lut.I[L])
		var base, step int
		if vertical {
			base, step = i*c.stride+c.off, 1
		} else {
			base, step = c.off+i, c.stride
		}
		set := func(k int, v uint8) { c.pix[base+k*step] = v }
		get := func(k int) uint8 { return c.pix[base+k*step] }
		checkasmLPFBorderFull(rng, set, get, E, I)
	}
	return c
}

// checkasmLPFBorderFull is the complete init_lpf_border (all four cases).
func checkasmLPFBorderFull(rng *rand.Rand, set func(i int, v uint8), get func(i int) uint8, E, I int) {
	const F = 1
	filterType := rng.Intn(4)
	edgeDiff := rng.Intn((E+2)*4) - 2*(E+2)
	switch filterType {
	case 0:
		for i := -8; i < 8; i++ {
			set(i, uint8(rng.Intn(256)))
		}
	case 1:
		set(-8, uint8(rng.Intn(256)))
		set(7, uint8(rng.Intn(256)))
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 7; i++ {
			set(-(1 + i), checkasmClip(int(get(-1))+rng.Intn(2*(F+1))-(F+1)))
			set(i, checkasmClip(int(get(0))+rng.Intn(2*(F+1))-(F+1)))
		}
	case 2:
		for i := 4; i < 8; i++ {
			set(-(1 + i), uint8(rng.Intn(256)))
			set(i, uint8(rng.Intn(256)))
		}
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 4; i++ {
			set(-(1 + i), checkasmClip(int(get(-1))+rng.Intn(2*(F+1))-(F+1)))
			set(i, checkasmClip(int(get(0))+rng.Intn(2*(F+1))-(F+1)))
		}
	case 3:
		for i := 4; i < 8; i++ {
			set(-(1 + i), uint8(rng.Intn(256)))
			set(i, uint8(rng.Intn(256)))
		}
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 4; i++ {
			set(-(1 + i), checkasmClip(int(get(-i))+rng.Intn(2*(I+1))-(I+1)))
			set(i, checkasmClip(int(get(i-1))+rng.Intn(2*(I+1))-(I+1)))
		}
	}
}

func benchmarkCheckasmSB(b *testing.B, vertical, luma bool) {
	n := 3
	widths := []int{4, 8, 16}
	if !luma {
		n = 2
		widths = []int{4, 6}
	}
	for s := 0; s < n; s++ {
		b.Run(fmt.Sprintf("w%d", widths[s]), func(b *testing.B) {
			// checkasm's bench alternates between two identical buffers; keep a
			// pool of independent draws so one random case does not dominate.
			const cases = 16
			var cs [cases]checkasmSBCase
			var work [cases][]byte
			for i := range cs {
				cs[i] = newCheckasmSBCase(int64(1000*s+i), vertical, luma, s)
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

func BenchmarkCheckasmLPFHSBY8(b *testing.B)  { benchmarkCheckasmSB(b, true, true) }
func BenchmarkCheckasmLPFVSBY8(b *testing.B)  { benchmarkCheckasmSB(b, false, true) }
func BenchmarkCheckasmLPFHSBUV8(b *testing.B) { benchmarkCheckasmSB(b, true, false) }
func BenchmarkCheckasmLPFVSBUV8(b *testing.B) { benchmarkCheckasmSB(b, false, false) }

// TestFilterSB8CheckasmMatchesScalar runs the dispatched superblock filters
// against the unit-by-unit C reference on checkasm-shaped content, where the
// flat, normal and hev paths all fire (random pixels mostly fail the filter
// mask). Levels are moved to a random component and the level slice is cut
// right after the last unit, so the vector level loads hit their bounds
// fallback as well.
func TestFilterSB8CheckasmMatchesScalar(t *testing.T) {
	for seed := int64(0); seed < 600; seed++ {
		for _, vertical := range []bool{true, false} {
			for _, luma := range []bool{true, false} {
				n := 3
				if !luma {
					n = 2
				}
				for s := 0; s < n; s++ {
					c := newCheckasmSBCase(seed*7+int64(s), vertical, luma, s)
					comp := int(seed & 3)
					lvl := make([][4]uint8, len(c.lvl))
					for i := range c.lvl {
						lvl[i][comp] = c.lvl[i][0]
						lvl[i][(comp+1)&3] = uint8(i) // junk in the other components
					}
					if seed&1 == 1 && !vertical {
						lvl = lvl[:c.lvlOff+31]
					}
					want := append([]byte(nil), c.pix...)
					got := append([]byte(nil), c.pix...)
					var ws, gs SBStats
					vm := c.vm
					if seed&1 == 1 && !vertical {
						for k := range vm {
							vm[k] &= 0x7fffffff
						}
					}
					if vertical {
						ws = filterSBVertical8Scalar(want, c.off, c.stride, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						gs = FilterSBVertical8(got, c.off, c.stride, c.rowBytes, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					} else {
						ws = filterSBHorizontal8Scalar(want, c.off, c.stride, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						gs = FilterSBHorizontal8(got, c.off, c.stride, c.rowBytes, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					}
					if ws != gs {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d: stats %+v, want %+v", seed, vertical, luma, s, gs, ws)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d: pixels differ", seed, vertical, luma, s)
					}
				}
			}
		}
	}
}
