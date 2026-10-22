// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

// checkasmSB16Case is checkasmSBCase for 10/12-bit planes: dav1d's
// tests/checkasm/loopfilter.c check_lpf_sb at 16bpc, with init_lpf_border's
// limits and flatness scaled by bitdepth_min_8 and samples stored as
// little-endian uint16 (off, stride and rowBytes in bytes).
type checkasmSB16Case struct {
	checkasmSBCase
	bitDepth uint8
}

func newCheckasmSB16Case(seed int64, vertical, luma bool, strength int) checkasmSB16Case {
	rng := rand.New(rand.NewSource(seed))
	nBlks := 32
	if !luma {
		nBlks = 16
	}
	sharp := rng.Intn(8)
	c := checkasmSB16Case{checkasmSBCase: checkasmSBCase{lut: NewSBLUT(uint8(sharp))}, bitDepth: 10}
	if rng.Intn(2) == 1 {
		c.bitDepth = 12
	}
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
		c.stride = 2 * 32
		c.pix = make([]byte, c.stride*(4*nBlks))
		c.off = 2 * 8
		c.rowBytes = c.stride - c.off
		c.lvlOff, c.lvlStride = 1, 2
	} else {
		c.stride = 2 * 4 * nBlks
		c.pix = make([]byte, c.stride*16)
		c.off = 8 * c.stride
		c.rowBytes = c.stride
		c.lvlOff, c.lvlStride = 32, 32
	}
	bd8 := int(c.bitDepth) - 8
	maxV := 1<<c.bitDepth - 1
	clip := func(v int) int { return min(max(v, 0), maxV) }
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
		E, I := int(c.lut.E[L])<<bd8, int(c.lut.I[L])<<bd8
		F := 1 << bd8
		var base, step int
		if vertical {
			base, step = i*c.stride+c.off, 2
		} else {
			base, step = c.off+2*i, c.stride
		}
		set := func(k, v int) { binary.LittleEndian.PutUint16(c.pix[base+k*step:], uint16(v)) }
		get := func(k int) int { return int(binary.LittleEndian.Uint16(c.pix[base+k*step:])) }
		edgeDiff := rng.Intn((E+2)*4) - 2*(E+2)
		switch rng.Intn(4) {
		case 0:
			for k := -8; k < 8; k++ {
				set(k, rng.Intn(maxV+1))
			}
		case 1:
			set(-8, rng.Intn(maxV+1))
			set(7, rng.Intn(maxV+1))
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 7; k++ {
				set(-(1 + k), clip(get(-1)+rng.Intn(2*(F+1))-(F+1)))
				set(k, clip(get(0)+rng.Intn(2*(F+1))-(F+1)))
			}
		case 2:
			for k := 4; k < 8; k++ {
				set(-(1 + k), rng.Intn(maxV+1))
				set(k, rng.Intn(maxV+1))
			}
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 4; k++ {
				set(-(1 + k), clip(get(-1)+rng.Intn(2*(F+1))-(F+1)))
				set(k, clip(get(0)+rng.Intn(2*(F+1))-(F+1)))
			}
		case 3:
			for k := 4; k < 8; k++ {
				set(-(1 + k), rng.Intn(maxV+1))
				set(k, rng.Intn(maxV+1))
			}
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 4; k++ {
				set(-(1 + k), clip(get(-k)+rng.Intn(2*(I+1))-(I+1)))
				set(k, clip(get(k-1)+rng.Intn(2*(I+1))-(I+1)))
			}
		}
	}
	return c
}

// TestFilterSB16CheckasmMatchesScalar is TestFilterSB8CheckasmMatchesScalar
// for the 10/12-bit superblock filters.
func TestFilterSB16CheckasmMatchesScalar(t *testing.T) {
	for seed := int64(0); seed < 400; seed++ {
		for _, vertical := range []bool{true, false} {
			for _, luma := range []bool{true, false} {
				n := 3
				if !luma {
					n = 2
				}
				for s := 0; s < n; s++ {
					c := newCheckasmSB16Case(seed*7+int64(s), vertical, luma, s)
					comp := int(seed & 3)
					lvl := make([][4]uint8, len(c.lvl))
					for i := range c.lvl {
						lvl[i][comp] = c.lvl[i][0]
						lvl[i][(comp+1)&3] = uint8(i)
					}
					vm := c.vm
					if seed&1 == 1 && !vertical {
						lvl = lvl[:c.lvlOff+31]
						for k := range vm {
							vm[k] &= 0x7fffffff
						}
					}
					want := append([]byte(nil), c.pix...)
					got := append([]byte(nil), c.pix...)
					var ws, gs SBStats
					if vertical {
						ws = filterSBVertical16Scalar(want, c.off, c.stride, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						gs = FilterSBVertical16(got, c.off, c.stride, c.rowBytes, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					} else {
						ws = filterSBHorizontal16Scalar(want, c.off, c.stride, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						gs = FilterSBHorizontal16(got, c.off, c.stride, c.rowBytes, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					}
					if ws != gs {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d: stats %+v, want %+v", seed, vertical, luma, s, gs, ws)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d bd=%d: pixels differ", seed, vertical, luma, s, c.bitDepth)
					}
				}
			}
		}
	}
}

func benchmarkCheckasmSB16(b *testing.B, vertical, luma bool) {
	n := 3
	widths := []int{4, 8, 16}
	if !luma {
		n = 2
		widths = []int{4, 6}
	}
	for s := 0; s < n; s++ {
		b.Run(fmt.Sprintf("w%d", widths[s]), func(b *testing.B) {
			const cases = 16
			var cs [cases]checkasmSB16Case
			var work [cases][]byte
			for i := range cs {
				cs[i] = newCheckasmSB16Case(int64(1000*s+i), vertical, luma, s)
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

func BenchmarkCheckasmLPFHSBY16(b *testing.B)  { benchmarkCheckasmSB16(b, true, true) }
func BenchmarkCheckasmLPFVSBY16(b *testing.B)  { benchmarkCheckasmSB16(b, false, true) }
func BenchmarkCheckasmLPFHSBUV16(b *testing.B) { benchmarkCheckasmSB16(b, true, false) }
func BenchmarkCheckasmLPFVSBUV16(b *testing.B) { benchmarkCheckasmSB16(b, false, false) }
