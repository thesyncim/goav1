package restoration

import (
	"slices"
	"testing"
)

// The unchanged u16 entry independently clears its staging. Poison the u8
// workspace between calls, including inactive filter planes, and exercise
// in-place output after all source samples have been staged.
func TestApplySelfguidedU8ScratchReuse(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x919)
	for _, sz := range [][2]int{{1, 1}, {3, 7}, {8, 8}, {17, 15}, {63, 64}, {64, 63}} {
		w, h := sz[0], sz[1]
		stride := w + 11
		origin := 3*stride + 4
		src8 := randomU8Plane(rnd, stride, h+6)
		need, err := SelfguidedScratchLen(w, h)
		if err != nil {
			t.Fatal(err)
		}
		scratch := make([]int32, need+11)
		for pi := range SGRParameterTable {
			for _, inplace := range []bool{false, true} {
				for i := range scratch {
					scratch[i] = int32(uint32(i)*2654435761 + 0x91a7)
				}
				tail := slices.Clone(scratch[need:])
				gotSrc := slices.Clone(src8)
				wantSrc := widenU8(src8)
				got := make([]byte, stride*h)
				want := make([]uint16, stride*h)
				if inplace {
					got = gotSrc[origin:]
					want = wantSrc[origin:]
				}
				xqd := [2]int8{-91, 73}
				if err := ApplySelfguidedRestoration(wantSrc, stride, origin, want, stride, w, h, pi, xqd, 8, make([]int32, need)); err != nil {
					t.Fatal(err)
				}
				if err := ApplySelfguidedRestorationU8(gotSrc, stride, origin, got, stride, w, h, pi, xqd, scratch); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if uint16(got[i]) != want[i] {
						t.Fatalf("%dx%d params=%d inplace=%v sample=%d got=%d want=%d", w, h, pi, inplace, i, got[i], want[i])
					}
				}
				if !slices.Equal(scratch[need:], tail) {
					t.Fatal("scratch tail overwritten")
				}
			}
		}
	}
}
