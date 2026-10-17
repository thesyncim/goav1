// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// rowSkipNetwork is one forward row network the row-skip bound covers, with
// the analytic rounding-error bound its slack is derived from.
type rowSkipNetwork struct {
	n        int
	slack    int32
	errBound float64
	run      func(in []int32) []int32
}

var rowSkipNetworks = []rowSkipNetwork{
	{8, fwdRowSkipSlack8, 1.21, func(in []int32) []int32 {
		var a, out [8]int32
		copy(a[:], in)
		fwdDCT8(&a, &out)
		return out[:]
	}},
	{16, fwdRowSkipSlack16, 2.83, func(in []int32) []int32 {
		var a, out [16]int32
		copy(a[:], in)
		fwdDCT16(&a, &out, &fwdCospi12, 12)
		return out[:]
	}},
	{32, fwdRowSkipSlack32, 7.05, func(in []int32) []int32 {
		var a, out [32]int32
		copy(a[:], in)
		fwdDCT32(&a, &out, &fwdCospi12, 12)
		return out[:]
	}},
}

// TestForwardDCTRowBound pins the constants behind fwdRowSkipLive for each row
// network: every effective input weight is at most 1 (plus constant
// quantization noise far below the 1/64 the bound allows), the observed
// rounding error stays within the analytic bound the slack covers, and no
// output ever reaches S + S>>6 + slack.
func TestForwardDCTRowBound(t *testing.T) {
	for _, nw := range rowSkipNetworks {
		t.Run(fmt.Sprint(nw.n), func(t *testing.T) {
			if float64(nw.slack) < math.Ceil(nw.errBound)+1 {
				t.Fatalf("slack %d does not cover error bound %v", nw.slack, nw.errBound)
			}
			n := nw.n
			// Effective weights from scaled unit vectors: the network's rounding is
			// at most a few units, negligible against the 2^24 scale.
			const scale = 1 << 24
			w := make([][]float64, n)
			for l := range w {
				w[l] = make([]float64, n)
			}
			maxW := 0.0
			in := make([]int32, n)
			for j := range n {
				clear(in)
				in[j] = scale
				out := nw.run(in)
				for l := range n {
					w[l][j] = float64(out[l]) / scale
					maxW = max(maxW, math.Abs(w[l][j]))
				}
			}
			if maxW > 1+1.0/256 {
				t.Fatalf("max effective weight %v exceeds 1+1/256", maxW)
			}
			rng := rand.New(rand.NewSource(int64(7 + n)))
			maxErr := 0.0
			check := func(in []int32) {
				out := nw.run(in)
				s := int32(0)
				for _, v := range in {
					s += max(v, -v)
				}
				for l := range n {
					lin := 0.0
					for j := range n {
						lin += w[l][j] * float64(in[j])
					}
					maxErr = max(maxErr, math.Abs(float64(out[l])-lin))
					c := max(out[l], -out[l])
					if fwdRowSkipLive(s, c+1, nw.slack) == false {
						t.Fatalf("|C_%d|=%d breaks the bound for S=%d (in %v)", l, c, s, in)
					}
				}
			}
			for trial := range 300000 {
				amp := []int32{1, 2, 3, 8, 64, 4080}[trial%6]
				for j := range in {
					in[j] = rng.Int31n(2*amp+1) - amp
				}
				check(in)
			}
			// Sign patterns aligned with each basis row maximize the linear part.
			for l := range n {
				for j := range in {
					in[j] = 2040
					if w[l][j] < 0 {
						in[j] = -2040
					}
				}
				check(in)
			}
			t.Logf("n=%d max effective weight %v, observed rounding error %v (bound %v)", n, maxW, maxErr, nw.errBound)
			// The weight estimate itself carries ~1e-7 relative noise.
			if maxErr > nw.errBound+0.05 {
				t.Fatalf("observed rounding error %v exceeds the %v analytic bound", maxErr, nw.errBound)
			}
		})
	}
}

// rowSkipCase runs one row-skipping kernel against the full transform.
type rowSkipCase struct {
	n      int
	full   func(coeff []int32, res []int16)
	scalar func(coeff []int32, res []int16, thr0, thr int32) uint8
	impls  map[string]func(coeff []int32, res []int16, thr0, thr int32) uint8
}

func rowSkipCases() []rowSkipCase {
	return []rowSkipCase{
		{
			n:    8,
			full: func(c []int32, r []int16) { forwardDCT8x8PureGo(c, 8, r, 8) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT8x8RowSkipPureGo((*[64]int32)(c), (*[64]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT8x8RowSkipImpl((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT8x8RowSkipFull((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
			},
		},
		{
			n:    16,
			full: func(c []int32, r []int16) { forwardDCT16x16PureGo(c, 16, r, 16) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT16x16RowSkipPureGo((*[256]int32)(c), (*[256]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT16x16RowSkipImpl((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT16x16RowSkipFull((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
			},
		},
		{
			n:    32,
			full: func(c []int32, r []int16) { forwardDCT32x32PureGo(c, 32, r, 32) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT32x32RowSkipPureGo((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT32x32RowSkipImpl((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT32x32RowSkipFull((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
			},
		},
	}
}

func TestForwardDCTRowSkipMatchesFull(t *testing.T) {
	for _, tc := range rowSkipCases() {
		t.Run(fmt.Sprint(tc.n), func(t *testing.T) {
			n := tc.n
			groups := n / 4
			allMask := uint8(1<<groups - 1)
			rng := rand.New(rand.NewSource(int64(11 + n)))
			res := make([]int16, n*n)
			full := make([]int32, n*n)
			got := make([]int32, n*n)
			sawSkip, sawPartial, sawAllZero := false, false, false
			for trial := range 6000 {
				amp := []int16{0, 1, 2, 4, 12, 40, 255}[trial%7]
				sparse := trial%3 == 0
				for i := range res {
					res[i] = 0
					if amp > 0 && (!sparse || rng.Intn(16) == 0) {
						res[i] = int16(rng.Intn(2*int(amp)+1)) - amp
					}
				}
				if trial%97 == 0 {
					for i := range res {
						res[i] = 255
						if (i/n+i%n)%2 == 1 {
							res[i] = -255
						}
					}
				}
				thr := rng.Int31n(400) + 1
				thr0 := min(rng.Int31n(400)+1, thr)
				tc.full(full, res)
				wantMask := tc.scalar(make([]int32, n*n), res, thr0, thr)
				check := func(name string, mask uint8) {
					for r := range n {
						live := mask&(1<<(r>>2)) != 0
						rowThr := thr
						if r == 0 {
							rowThr = thr0
						}
						for c := range n {
							want := full[c*n+r]
							if live {
								if got[c*n+r] != want {
									t.Fatalf("%s trial %d: coeff[%d][%d]=%d want %d", name, trial, c, r, got[c*n+r], want)
								}
								continue
							}
							if max(want, -want) >= rowThr {
								t.Fatalf("%s trial %d: skipped row %d holds |%d| >= zero bin %d", name, trial, r, want, rowThr)
							}
							if mask != 0 && got[c*n+r] != 0 {
								t.Fatalf("%s trial %d: skipped coeff[%d][%d]=%d not zeroed", name, trial, c, r, got[c*n+r])
							}
						}
					}
				}
				for i := range got {
					got[i] = 0x5a5a5a5a
				}
				check("scalar", tc.scalar(got, res, thr0, thr))
				switch wantMask {
				case 0:
					sawAllZero = true
				case allMask:
				default:
					sawPartial = true
				}
				if wantMask != allMask {
					sawSkip = true
				}
				for name, impl := range tc.impls {
					for i := range got {
						got[i] = 0x5a5a5a5a
					}
					mask := impl(got, res, thr0, thr)
					if name != "full" && mask != wantMask && mask != allMask {
						t.Fatalf("%s trial %d: mask %b, scalar mask %b", name, trial, mask, wantMask)
					}
					check(name, mask)
				}
			}
			if !sawSkip || !sawPartial || !sawAllZero {
				t.Fatalf("corpus coverage skip=%v partial=%v allzero=%v", sawSkip, sawPartial, sawAllZero)
			}
		})
	}
}

func BenchmarkForwardDCTRowSkip(b *testing.B) {
	for _, tc := range rowSkipCases() {
		n := tc.n
		rng := rand.New(rand.NewSource(3))
		res := make([]int16, n*n)
		for i := range res {
			res[i] = int16(rng.Intn(9)) - 4
		}
		coeff := make([]int32, n*n)
		b.Run(fmt.Sprintf("%d/full", n), func(b *testing.B) {
			for range b.N {
				tc.impls["full"](coeff, res, 0, 0)
			}
		})
		for _, thr := range []int32{40, 70, 150, 220} {
			mask := tc.impls["dispatch"](coeff, res, thr, thr)
			b.Run(fmt.Sprintf("%d/skip-thr%d-mask%b", n, thr, mask), func(b *testing.B) {
				for range b.N {
					tc.impls["dispatch"](coeff, res, thr, thr)
				}
			})
		}
	}
}
