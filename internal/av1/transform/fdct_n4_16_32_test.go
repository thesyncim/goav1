// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// n4Residuals yields 8-bit residual blocks: random ranges, the extremes and
// constant blocks.
func n4Residuals(n, count int) [][]int16 {
	rng := rand.New(rand.NewSource(7))
	var out [][]int16
	for i := range count {
		res := make([]int16, n*n)
		switch i % 6 {
		case 0:
			for j := range res {
				res[j] = int16(rng.Intn(511)) - 255
			}
		case 1:
			for j := range res {
				res[j] = int16(rng.Intn(9)) - 4
			}
		case 2:
			for j := range res {
				res[j] = 255
				if rng.Intn(2) == 0 {
					res[j] = -255
				}
			}
		case 3:
			v := int16(rng.Intn(511)) - 255
			for j := range res {
				res[j] = v
			}
		case 4:
			for j := range res {
				res[j] = int16((j%n)*16 - 255)
			}
		default:
			for j := range res {
				res[j] = int16(rng.Intn(61)) - 30
			}
		}
		out = append(out, res)
	}
	return out
}

func TestForwardDCTN4MatchesFull(t *testing.T) {
	for _, n := range []int{16, 32} {
		k := n / 4
		for i, res := range n4Residuals(n, 600) {
			full := make([]int32, n*n)
			var err error
			if n == 16 {
				err = ForwardDCT16x16(full, 16, res, 16)
			} else {
				err = ForwardDCT32x32(full, 32, res, 32)
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, impl := range []string{"pure", "impl"} {
				got := make([]int32, n*n)
				for j := range got {
					got[j] = 12345 // only the N4 block is written
				}
				switch {
				case n == 16 && impl == "pure":
					var buf [64]int32
					forwardDCT16x16N4ColPureGo(&buf, (*[256]int16)(res))
					fwd16x16N4RowsPureGo((*[256]int32)(got), &buf, 0xf)
				case n == 16:
					ForwardDCT16x16N48BitTrusted((*[256]int32)(got), (*[256]int16)(res))
				case impl == "pure":
					var buf [256]int32
					forwardDCT32x32N4ColPureGo(&buf, (*[1024]int16)(res))
					fwd32x32N4RowsPureGo((*[1024]int32)(got), &buf, 0xff)
				default:
					ForwardDCT32x32N48BitTrusted((*[1024]int32)(got), (*[1024]int16)(res))
				}
				for c := range n {
					for r := range n {
						g := got[c*n+r]
						if c < k && r < k {
							if g != full[c*n+r] {
								t.Fatalf("n=%d case %d %s: coeff[col %d][row %d] = %d, full %d", n, i, impl, c, r, g, full[c*n+r])
							}
						} else if g != 12345 {
							t.Fatalf("n=%d case %d %s: wrote outside the N4 block at col %d row %d", n, i, impl, c, r)
						}
					}
				}
			}
		}
	}
}

// TestForwardDCTN4RowSkipMatchesRowSkip pins the N4 row-skip kernels to the
// scalar row-skipping kernels followed by the encoder's N4 restriction: the same
// group mask over the N4 rows and the same N4 coefficients.
func TestForwardDCTN4RowSkipMatchesRowSkip(t *testing.T) {
	for _, n := range []int{16, 32} {
		k := n / 4
		groups := uint8(1)<<((k+3)>>2) - 1
		for i, res := range n4Residuals(n, 600) {
			for _, thr := range []int32{0, 1, 20, 40, 70, 150, 220, 600, 5000} {
				thr0 := thr / 2
				want := make([]int32, n*n)
				got := make([]int32, n*n)
				var wm, gm uint8
				if n == 16 {
					wm = forwardDCT16x16RowSkipPureGo((*[256]int32)(want), (*[256]int16)(res), thr0, thr)
					gm = ForwardDCT16x16N4RowSkip8BitTrusted((*[256]int32)(got), (*[256]int16)(res), thr0, thr)
				} else {
					wm = forwardDCT32x32RowSkipPureGo((*[1024]int32)(want), (*[1024]int16)(res), thr0, thr)
					gm = ForwardDCT32x32N4RowSkip8BitTrusted((*[1024]int32)(got), (*[1024]int16)(res), thr0, thr)
				}
				wm &= groups
				if gm != wm {
					t.Fatalf("n=%d case %d thr %d: mask %b, row-skip kernel %b", n, i, thr, gm, wm)
				}
				if gm == 0 {
					continue
				}
				for c := range k {
					for r := range k {
						if got[c*n+r] != want[c*n+r] {
							t.Fatalf("n=%d case %d thr %d: coeff[col %d][row %d] = %d, row-skip %d", n, i, thr, c, r, got[c*n+r], want[c*n+r])
						}
					}
				}
			}
		}
	}
}

func BenchmarkForwardDCTN4(b *testing.B) {
	for _, n := range []int{16, 32} {
		res := n4Residuals(n, 1)[0]
		coeff := make([]int32, n*n)
		b.Run("rowskip", func(b *testing.B) {
			for range b.N {
				if n == 16 {
					ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(coeff), (*[256]int16)(res), 10, 20)
				} else {
					ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(coeff), (*[1024]int16)(res), 10, 20)
				}
			}
		})
		b.Run("n4", func(b *testing.B) {
			for range b.N {
				if n == 16 {
					ForwardDCT16x16N4RowSkip8BitTrusted((*[256]int32)(coeff), (*[256]int16)(res), 10, 20)
				} else {
					ForwardDCT32x32N4RowSkip8BitTrusted((*[1024]int32)(coeff), (*[1024]int16)(res), 10, 20)
				}
			}
		})
	}
}
