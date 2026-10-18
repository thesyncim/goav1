//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"slices"
	"testing"
)

// Use the same eight shapes as the existing rolling/full oracle. Compare every
// A/B cell, including poison in skipped rows and padding, to the untouched
// scalar rolling implementation. Full-width int32 values exercise wrap points
// independently of legal 8/10/12-bit image inputs.
func TestSGRCoefficientRollingMatchesScalar(t *testing.T) {
	params := append([]SGRParams(nil), SGRParameterTable[:]...)
	params = append(params,
		SGRParams{Radius: [2]uint8{2, 1}, S: [2]int16{-32768, 32767}},
		SGRParams{Radius: [2]uint8{1, 2}, S: [2]int16{0, -1}})
	for _, shape := range [][2]int{{1, 1}, {2, 3}, {3, 2}, {7, 9}, {16, 8}, {31, 33}, {63, 64}, {64, 63}} {
		w, h := shape[0], shape[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, depth := range []int{8, 10, 12} {
			for _, fullWidth := range []bool{false, true} {
				for i := range dgd {
					dgd[i] = int32(uint32(i)*2654435761 + 0x80000000)
					if !fullWidth {
						dgd[i] &= (1 << depth) - 1
					}
				}
				for pi, p := range params {
					for ri, radius := range p.Radius {
						if radius == 0 {
							continue
						}
						for pass := 0; pass < 2; pass++ {
							ga, gb := make([]int32, bl+7), make([]int32, bl+7)
							for i := range ga {
								ga[i] = int32(uint32(i)*2654435761 + 0xfedc)
								gb[i] = ^ga[i]
							}
							wa, wb := slices.Clone(ga), slices.Clone(gb)
							calculateIntermediateRolling(dgd, origin, w, h, stride, depth, p, ri, pass, wa[:bl], wb[:bl], bs)
							calculateIntermediateRollingSIMD(dgd, origin, w, h, stride, depth, p, ri, pass, ga[:bl], gb[:bl], bs)
							if !slices.Equal(ga, wa) || !slices.Equal(gb, wb) {
								t.Fatalf("shape=%v depth=%d fullWidth=%v params=%d radius=%d pass=%d", shape, depth, fullWidth, pi, radius, pass)
							}
						}
					}
				}
			}
		}
	}
}

// Reuses the preserved two-size/five-offset alias schedule, comparing complete
// backing and scratch to the original driver at 8, 10 and 12 bits. The frozen generic
// drivers retain the independent scalar coefficient producer.
func TestSGRCoefficientFrozenBacking(t *testing.T) {
	oldNormal, oldFast := selfguidedImpl, selfguidedFastImpl
	defer func() { selfguidedImpl, selfguidedFastImpl = oldNormal, oldFast }()
	for _, shape := range [][2]int{{7, 9}, {64, 64}} {
		w, h := shape[0], shape[1]
		stride := w + 11
		origin := 3*stride + 4
		need, _ := SelfguidedScratchLen(w, h)
		for _, depth := range []uint8{8, 10, 12} {
			for _, delta := range []int{0, 1, -1, stride, -stride} {
				for pi := range SGRParameterTable {
					got := make([]uint16, stride*(h+8)+13)
					for i := range got {
						got[i] = uint16(i*79+31) & uint16((1<<depth)-1)
					}
					want := slices.Clone(got)
					gs := make([]int32, need+11)
					for i := range gs {
						gs[i] = int32(uint32(i)*2654435761 + 0x91a7)
					}
					ws := slices.Clone(gs)
					selfguidedImpl, selfguidedFastImpl = frozenSGRGeneric, frozenSGRGenericFast
					we := ApplySelfguidedRestoration(want, stride, origin, want[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, depth, ws[:need])
					selfguidedImpl, selfguidedFastImpl = selfguidedSIMD, selfguidedFastSIMD
					ge := ApplySelfguidedRestoration(got, stride, origin, got[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, depth, gs[:need])
					if ge != we || !slices.Equal(got, want) || !slices.Equal(gs, ws) {
						t.Fatalf("wide backing shape%v depth%d delta%d pi%d err%v/%v", shape, depth, delta, pi, ge, we)
					}
					if depth != 8 {
						continue
					}
					gb := make([]byte, len(got))
					for i := range gb {
						gb[i] = byte(i*79 + 31)
					}
					wb := slices.Clone(gb)
					for i := range gs {
						gs[i] = int32(uint32(i)*2654435761 + 0x91a7)
					}
					copy(ws, gs)
					selfguidedImpl, selfguidedFastImpl = frozenSGRGeneric, frozenSGRGenericFast
					we = ApplySelfguidedRestorationU8(wb, stride, origin, wb[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, ws[:need])
					selfguidedImpl, selfguidedFastImpl = selfguidedSIMD, selfguidedFastSIMD
					ge = ApplySelfguidedRestorationU8(gb, stride, origin, gb[origin+delta:], stride, w, h, pi, [2]int8{-91, 73}, gs[:need])
					if ge != we || !slices.Equal(gb, wb) || !slices.Equal(gs, ws) {
						t.Fatalf("byte backing shape%v delta%d pi%d err%v/%v", shape, delta, pi, ge, we)
					}
				}
			}
		}
	}
}
