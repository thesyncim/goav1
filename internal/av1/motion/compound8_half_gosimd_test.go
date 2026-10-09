// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestCompound8HalfExtremesAndExactWindows(t *testing.T) {
	round0, offsetBits, roundOffset, _ := compoundRoundParams8()
	for _, tbl := range avx2FilterTables() {
		for phase := 0; phase < 16; phase++ {
			k := tbl[phase]
			for _, shape := range [][2]int{{4, 4}, {8, 16}, {32, 32}, {128, 128}} {
				w, h := shape[0], shape[1]
				for pattern := 0; pattern < 3; pattern++ {
					stride := w + 9
					ref := frame.Plane{Pix: make([]byte, (h+6)*stride+w+8), Stride: stride, Width: w + 8, Height: h + 7}
					for y := 0; y < ref.Height; y++ {
						for x := 0; x < ref.Width; x++ {
							if pattern == 1 || pattern == 2 && (x+y)&1 != 0 {
								ref.Pix[y*stride+x] = 255
							}
						}
					}
					var scratch CompoundConvolveScratch
					for axis := 0; axis < 4; axis++ {
						got := make([]uint16, w*h)
						want := make([]uint16, w*h)
						switch axis {
						case 0:
							predictInterCompoundRef8ToConvBufCopyGoSIMD(got, ref, 3, 3, w, h, round0, roundOffset)
							predictInterCompoundRef8ToConvBufCopyPureGo(want, ref, 3, 3, w, h, round0, roundOffset)
						case 1:
							predictInterCompoundRef8ToConvBufXGoSIMD(got, ref, 3, 3, w, h, k, roundOffset)
							predictInterCompoundRef8ToConvBufXPureGo(want, ref, 3, 3, w, h, k, roundOffset)
						case 2:
							predictInterCompoundRef8ToConvBufYGoSIMD(got, ref, 3, 3, w, h, k, round0, roundOffset)
							predictInterCompoundRef8ToConvBufYPureGo(want, ref, 3, 3, w, h, k, round0, roundOffset)
						default:
							predictInterCompoundRef8ToConvBuf2DGoSIMD(got, ref, 3, 3, w, h, k, k, offsetBits, &scratch)
							predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, 3, 3, w, h, k, k, offsetBits, &scratch)
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("axis=%d phase=%d shape=%dx%d pattern=%d sample=%d: got=%d want=%d", axis, phase, w, h, pattern, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}
