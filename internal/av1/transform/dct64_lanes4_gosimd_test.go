// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"reflect"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// lanes4ClampSets is the clamp ladder the four-lane kernels must be exact
// for: every bit depth's stage bounds plus the edges of both envelopes.
func lanes4ClampSets() [][2]int32 {
	sets := append([][2]int32(nil), stageClampSets...)
	return append(sets,
		[2]int32{-itxNarrowBound, itxNarrowBound - 1},
		[2]int32{-itxWideBound, itxWideBound - 1},
		[2]int32{-itxNarrowBound - 1, itxNarrowBound - 1},
		[2]int32{-100000, 70000},
		[2]int32{-3, 5},
	)
}

// lanes4Fill returns an input generator for one of several value patterns:
// uniform in [min, max], extremes only, sparse, or small.
func lanes4Fill(rng *rand.Rand, pattern int, min, max int32) func() int32 {
	switch pattern {
	case 0:
		return func() int32 { return min + int32(rng.Int63n(int64(max)-int64(min)+1)) }
	case 1:
		ext := []int32{min, max, min + 1, max - 1, 0, 1, -1}
		return func() int32 { return ext[rng.Intn(len(ext))] }
	case 2:
		return func() int32 {
			if rng.Intn(8) != 0 {
				return 0
			}
			if rng.Intn(2) == 0 {
				return min
			}
			return max
		}
	default:
		return func() int32 { return clipRange(int64(rng.Intn(512)-256), min, max) }
	}
}

// dct64Lanes4Kernels lists both generated kernels with the bound each is
// valid for, so the Wide kernel is also exercised at narrow clamp bounds.
var dct64Lanes4Kernels = []struct {
	name  string
	bound int32
	fn    func(p unsafe.Pointer, stride uintptr, min, max int32)
}{
	{"Narrow", itxNarrowBound, inverseDCT64Lanes4Narrow},
	{"Wide", itxWideBound, inverseDCT64Lanes4Wide},
}

func TestDCT64Lanes4MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6464))
	for _, kern := range dct64Lanes4Kernels {
		for _, cs := range lanes4ClampSets() {
			min, max := cs[0], cs[1]
			if min < -kern.bound || max >= kern.bound {
				continue
			}
			for _, rowStride := range []int{4, 5, 7, 64} {
				for iter := 0; iter < 400; iter++ {
					fill := lanes4Fill(rng, iter%4, min, max)
					off := rng.Intn(4)
					buf := make([]int32, off+(dct64Size-1)*rowStride+4+3)
					for i := range buf {
						buf[i] = fill()
					}
					want := append([]int32(nil), buf...)
					for c := 0; c < 4; c++ {
						inverseDCT64(want[off+c:], rowStride, min, max)
					}
					got := append([]int32(nil), buf...)
					kern.fn(unsafe.Pointer(&got[off]), uintptr(rowStride)*4, min, max)
					if !eqInt32(got, want) {
						t.Fatalf("%s clamp=[%d,%d] stride=%d iter=%d\n in=%v\n got=%v\nwant=%v",
							kern.name, min, max, rowStride, iter, buf, got, want)
					}
				}
			}
		}
	}
}

func TestDCT64Col4Row4SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4646))
	for _, cs := range lanes4ClampSets() {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 600; iter++ {
			fill := lanes4Fill(rng, iter%4, min, max)
			const rowStride = 6
			buf := make([]int32, (dct64Size-1)*rowStride+4)
			for i := range buf {
				buf[i] = fill()
			}
			want := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			inverseDCT64Col4PureGo(want, rowStride, min, max)
			inverseDCT64Col4SIMD(got, rowStride, min, max)
			if !eqInt32(got, want) {
				t.Fatalf("Col4 clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v", min, max, iter, buf, got, want)
			}

			rows := make([]int32, 4*dct64Size+5)
			for i := range rows {
				rows[i] = fill()
			}
			wantR := append([]int32(nil), rows...)
			gotR := append([]int32(nil), rows...)
			split := func(s []int32) ([]int32, []int32, []int32, []int32) {
				return s[1:65], s[66:130], s[130:194], s[195:259]
			}
			w0, w1, w2, w3 := split(wantR)
			g0, g1, g2, g3 := split(gotR)
			inverseDCT64Row4PureGo(w0, w1, w2, w3, min, max)
			inverseDCT64Row4SIMD(g0, g1, g2, g3, min, max)
			if !eqInt32(gotR, wantR) {
				t.Fatalf("Row4 clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v", min, max, iter, rows, gotR, wantR)
			}
		}
	}
}

func TestDCT64Lanes4SIMDDispatchBound(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON unavailable")
	}
	if reflect.ValueOf(inverseDCT64Col4Impl).Pointer() != reflect.ValueOf(inverseDCT64Col4SIMD).Pointer() {
		t.Fatal("DCT64 column dispatch did not bind the Go SIMD kernel")
	}
	if reflect.ValueOf(inverseDCT64Row4Impl).Pointer() != reflect.ValueOf(inverseDCT64Row4SIMD).Pointer() {
		t.Fatal("DCT64 row dispatch did not bind the Go SIMD kernel")
	}
}

// TestDCT64Lanes4OutOfEnvelopeFallsBack checks bounds beyond the int32-exact
// envelope and short buffers route to the scalar reference.
func TestDCT64Lanes4OutOfEnvelopeFallsBack(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	min, max := int32(-(1 << 22)), int32(1<<22-1)
	buf := make([]int32, (dct64Size-1)*4+4)
	for i := range buf {
		buf[i] = min + int32(rng.Int63n(int64(max)-int64(min)+1))
	}
	want := append([]int32(nil), buf...)
	inverseDCT64Col4PureGo(want, 4, min, max)
	inverseDCT64Col4SIMD(buf, 4, min, max)
	if !eqInt32(buf, want) {
		t.Fatal("out-of-envelope Col4 mismatch")
	}
	short := make([]int32, 3*dct64Size+60)
	wantS := append([]int32(nil), short...)
	inverseDCT64Row4SIMD(short[:60], short[64:128], short[128:192], short[192:], -100, 100)
	inverseDCT64Row4PureGo(wantS[:60], wantS[64:128], wantS[128:192], wantS[192:], -100, 100)
	if !eqInt32(short, wantS) {
		t.Fatal("short-row Row4 mismatch")
	}
}
