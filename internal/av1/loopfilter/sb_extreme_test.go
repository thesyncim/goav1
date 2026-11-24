// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"bytes"
	"math/rand"
	"testing"
)

// sbExtremeRun filters one superblock call of 32 units (128 lanes) whose
// taps come from tap(lane, k) (k = -8..7, k < 0 above/left of the edge) with
// every unit at the maximum level and the given sharpness, and checks the
// dispatched filter against the unit-by-unit reference in both edge
// directions.
func sbExtremeRun(t *testing.T, name string, sharp uint8, luma bool, vm [3]uint32, tap func(lane, k int) uint8) {
	t.Helper()
	lut := NewSBLUT(sharp)
	for _, vertical := range []bool{false, true} {
		var pix []byte
		var off, stride, rowBytes, lvlOff, lvlStride int
		if vertical {
			// 128 rows of 32 bytes, edge at x = 8 (16-byte rows fit).
			stride, off, rowBytes = 32, 8, 24
			pix = make([]byte, stride*128)
			lvlStride, lvlOff = 8, 2
			for lane := 0; lane < 128; lane++ {
				for k := -8; k < 8; k++ {
					pix[lane*stride+off+k] = tap(lane, k)
				}
			}
		} else {
			// 16 rows of 128 lanes, edge at y = 8.
			stride, off, rowBytes = 128, 8*128, 128
			pix = make([]byte, stride*16)
			lvlStride, lvlOff = 32, 2*32
			for lane := 0; lane < 128; lane++ {
				for k := -8; k < 8; k++ {
					pix[off+k*stride+lane] = tap(lane, k)
				}
			}
		}
		lvl := make([]uint8, 4*(256))
		for i := range len(lvl) / 4 {
			lvl[4*(i)+0] = MaxLevel
		}
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		if vertical {
			filterSBVertical8Scalar(want, off, stride, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
			FilterSBVertical8(got, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
		} else {
			filterSBHorizontal8Scalar(want, off, stride, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
			FilterSBHorizontal8(got, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
		}
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%s vertical=%v luma=%v vm=%x: first diff at byte %d: got %d want %d", name, vertical, luma, vm, i, got[i], want[i])
				}
			}
		}
	}
}

// TestFilterSB8ExtremeTaps drives the 8-bit superblock kernels at the
// boundaries of their exact rewrites: flat lanes whose taps sit at 254/255
// (the scaled flat accumulators peak 128 below 2^16) or 0/1, and the narrow
// filter's saturating f = iclip_diff(3*(q0-p0) + iclip_diff(p1-q1)) over every
// p0, q0 pair with p1, q1 at the int8 clamp points.
func TestFilterSB8ExtremeTaps(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	widthMasks := func(luma bool) [][3]uint32 {
		all := ^uint32(0)
		m := [][3]uint32{{all, 0, 0}, {0, all, 0}}
		if luma {
			m = append(m, [3]uint32{0, 0, all})
		}
		var mixed [3]uint32
		for u := 0; u < 32; u++ {
			w := rng.Intn(len(m))
			mixed[w] |= 1 << uint(u)
		}
		return append(m, mixed)
	}
	for _, luma := range []bool{true, false} {
		for _, vm := range widthMasks(luma) {
			for iter := 0; iter < 8; iter++ {
				for _, lo := range []uint8{0, 254} {
					sbExtremeRun(t, "flat", uint8(rng.Intn(MaxSharpness+1)), luma, vm, func(lane, k int) uint8 {
						return lo + uint8(rng.Intn(2))
					})
				}
				// Flat inner taps around a step: flat8in passes, flat8out
				// does not on some lanes.
				sbExtremeRun(t, "flatin", uint8(rng.Intn(MaxSharpness+1)), luma, vm, func(lane, k int) uint8 {
					if k < -4 || k > 3 {
						return uint8(rng.Intn(256))
					}
					return 254 + uint8(rng.Intn(2))
				})
			}
			// Sharpness 0 keeps I = 63, so lanes with |q0 - p0| up to 96 and
			// p1, q1 up to 63 away still pass the filter mask: 3*(q0-p0) and
			// p1 - q1 both reach the int8 clamp.
			dev := []int{-63, -40, 0, 40, 63}
			clip := func(v int) uint8 { return uint8(max(0, min(255, v))) }
			for rep := 0; rep < 2; rep++ {
				for p0 := 0; p0 < 256; p0++ {
					for half := 0; half < 2; half++ {
						d1 := make([]int, 128)
						d2 := make([]int, 128)
						for i := range d1 {
							d1[i], d2[i] = dev[rng.Intn(len(dev))], dev[rng.Intn(len(dev))]
						}
						q0 := func(lane int) int { return (lane*2 + half) & 255 }
						if rep == 1 {
							// Concentrate q0 within the fm window around p0.
							q0 = func(lane int) int { return p0 - 96 + (lane*3+half)%193 }
						}
						sbExtremeRun(t, "narrow", 0, luma, vm, func(lane, k int) uint8 {
							switch k {
							case -1:
								return uint8(p0)
							case 0:
								return clip(q0(lane))
							case -2:
								return clip(p0 + d1[lane])
							case 1:
								return clip(q0(lane) + d2[lane])
							}
							return uint8(rng.Intn(256))
						})
					}
				}
			}
		}
	}
}
