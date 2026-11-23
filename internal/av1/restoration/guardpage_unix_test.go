// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package restoration

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

func guardSlice[T any](t *testing.T, n int, before bool) []T {
	if before {
		return guardpage.Before[T](t, n)
	}
	return guardpage.After[T](t, n)
}

// TestRestorationStaysInsideGuardedBuffers runs the Wiener and self-guided
// filters with source, destination and scratch buffers exactly as long as
// the entry points require (the bordered source window, the output block and
// the reported scratch length), each ending at or starting right after an
// inaccessible page.
func TestRestorationStaysInsideGuardedBuffers(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const b = 3 // WienerHalfwin == SGRProjBorderHorz == SGRProjBorderVert
	for iter := range 400 {
		w, h := 1+rng.Intn(ProcUnitSize), 1+rng.Intn(ProcUnitSize)
		before := iter%2 == 1
		hbd := iter%4 >= 2
		bd := uint8(8)
		if hbd {
			bd = 10
		}
		stride := w + 2*b
		origin := b*stride + b
		srcN := (h+2*b-1)*stride + w + 2*b
		dstN := (h-1)*w + w
		name := fmt.Sprintf("iter %d %dx%d hbd=%v before=%v", iter, w, h, hbd, before)
		info := WienerInfo{
			HFilter: NewWienerFilter(int16(rng.Intn(16)-5), int16(rng.Intn(32)-23), int16(rng.Intn(64)-17)),
			VFilter: NewWienerFilter(int16(rng.Intn(16)-5), int16(rng.Intn(32)-23), int16(rng.Intn(64)-17)),
		}
		pi := rng.Intn(len(SGRParameterTable))
		xqd := [2]int8{int8(rng.Intn(128) - 96), int8(rng.Intn(128) - 32)}
		wn, err := WienerScratchLen(w, h)
		if err != nil {
			t.Fatal(err)
		}
		sn, err := SelfguidedScratchLen(w, h)
		if err != nil {
			t.Fatal(err)
		}
		if !hbd {
			src := guardSlice[uint8](t, srcN, before)
			for i := range src {
				src[i] = uint8(rng.Intn(256))
			}
			dst := guardSlice[uint8](t, dstN, before)
			want := make([]uint8, dstN)
			guardpage.Check(t, name+" wiener", func() {
				if err := ApplyWienerRestorationU8(src, stride, origin, dst, w, w, h, info, guardSlice[uint16](t, wn, before)); err != nil {
					t.Fatal(err)
				}
			})
			if err := ApplyWienerRestorationU8(slices.Clone(src), stride, origin, want, w, w, h, info, make([]uint16, wn)); err != nil || !slices.Equal(dst, want) {
				t.Fatalf("%s: guarded Wiener differs (%v)", name, err)
			}
			guardpage.Check(t, name+" sgr", func() {
				if err := ApplySelfguidedRestorationU8(src, stride, origin, dst, w, w, h, pi, xqd, guardSlice[int32](t, sn, before)); err != nil {
					t.Fatal(err)
				}
			})
			if err := ApplySelfguidedRestorationU8(slices.Clone(src), stride, origin, want, w, w, h, pi, xqd, make([]int32, sn)); err != nil || !slices.Equal(dst, want) {
				t.Fatalf("%s: guarded self-guided differs (%v)", name, err)
			}
			continue
		}
		src := guardSlice[uint16](t, srcN, before)
		for i := range src {
			src[i] = uint16(rng.Intn(1 << bd))
		}
		dst := guardSlice[uint16](t, dstN, before)
		want := make([]uint16, dstN)
		if err := ApplyWienerRestorationTrusted(slices.Clone(src), stride, origin, want, w, w, h, info, bd, make([]uint16, wn)); err != nil {
			t.Fatal(err)
		}
		guardpage.Check(t, name+" wiener trusted", func() {
			if err := ApplyWienerRestorationTrusted(src, stride, origin, dst, w, w, h, info, bd, guardSlice[uint16](t, wn, before)); err != nil || !slices.Equal(dst, want) {
				t.Fatalf("%s: guarded trusted Wiener differs (%v)", name, err)
			}
		})
		guardpage.Check(t, name+" sgr", func() {
			if err := ApplySelfguidedRestoration(src, stride, origin, dst, w, w, h, pi, xqd, bd, guardSlice[int32](t, sn, before)); err != nil {
				t.Fatal(err)
			}
		})
		if err := ApplySelfguidedRestoration(slices.Clone(src), stride, origin, want, w, w, h, pi, xqd, bd, make([]int32, sn)); err != nil || !slices.Equal(dst, want) {
			t.Fatalf("%s: guarded self-guided differs (%v)", name, err)
		}
	}
}
