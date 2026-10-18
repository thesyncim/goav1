package restoration

import (
	"slices"
	"testing"
)

func TestSGRRollingIntermediateMatchesFullOracle(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x920)
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {3, 2}, {7, 9}, {16, 8}, {31, 33}, {63, 64}, {64, 63}} {
		w, h := sz[0], sz[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, bd := range []int{8, 10, 12} {
			for i := range dgd {
				dgd[i] = int32(rnd.pseudoUniform(1 << bd))
			}
			for pi, p := range SGRParameterTable {
				for ri, r := range p.Radius {
					if r == 0 {
						continue
					}
					for pass := 0; pass < 2; pass++ {
						wantA, wantB := make([]int32, bl), make([]int32, bl)
						gotA, gotB := make([]int32, bl+7), make([]int32, bl+7)
						for i := range gotA {
							gotA[i] = int32(uint32(i)*2654435761 + 0xfedc)
							gotB[i] = ^gotA[i]
						}
						tailA, tailB := slices.Clone(gotA[bl:]), slices.Clone(gotB[bl:])
						frozenSGRIntermediate(dgd, origin, w, h, stride, bd, pi, ri, pass, wantA, wantB, bs)
						calculateIntermediateRolling(dgd, origin, w, h, stride, bd, p, ri, pass, gotA[:bl], gotB[:bl], bs)
						step := 1
						if pass != 0 {
							step = 2
						}
						for y := -1; y < h+1; y += step {
							for x := -1; x < w+1; x++ {
								k := (y+3)*bs + x + 3
								if gotA[k] != wantA[k] || gotB[k] != wantB[k] {
									t.Fatalf("%dx%d bd=%d params=%d radius=%d pass=%d x=%d y=%d A=%d/%d B=%d/%d", w, h, bd, pi, ri, pass, x, y, gotA[k], wantA[k], gotB[k], wantB[k])
								}
							}
						}
						if !slices.Equal(gotA[bl:], tailA) || !slices.Equal(gotB[bl:], tailB) {
							t.Fatal("scratch tail written")
						}
					}
				}
			}
		}
	}
}

func frozenSGRIntermediate(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, bitDepth int, paramsIndex int, radiusIndex int, pass int, aBuf []int32, bBuf []int32, bufStride int) {
	clearInt32s(aBuf)
	clearInt32s(bBuf)
	params := SGRParameterTable[paramsIndex]
	r := int(params.Radius[radiusIndex])
	widthExt := width + 2*SGRProjBorderHorz
	heightExt := height + 2*SGRProjBorderVert
	srcOrigin := dgdOrigin - SGRProjBorderVert*dgdStride - SGRProjBorderHorz
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, false, bBuf, bufStride)
	boxsum(dgd, srcOrigin, widthExt, heightExt, dgdStride, r, true, aBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	step := 1
	if pass != 0 {
		step = 2
	}
	n := (2*r + 1) * (2*r + 1)
	for row := -1; row < height+1; row += step {
		for col := -1; col < width+1; col++ {
			k := aOrigin + row*bufStride + col
			a := uint32(roundPowerOfTwo(aBuf[k], 2*(bitDepth-8)))
			b := uint32(roundPowerOfTwo(bBuf[k], bitDepth-8))
			p := uint32(0)
			if a*uint32(n) >= b*b {
				p = a*uint32(n) - b*b
			}
			z := min(roundPowerOfTwoUnsigned(p*uint32(params.S[radiusIndex]), SGRProjMTableBits), 255)
			aBuf[k] = xByXPlus1Value(z)
			// libaom uses the raw (unshifted) B[k] in the post-update, NOT the
			// bit-depth-scaled local `b`. Using the scaled b mistakenly scales
			// the SGR smoothed image by an extra 1<<(bit_depth-8) factor for
			// 10/12-bit input, producing the q63 first-frame divergence in flt
			// even when reconstruction and CDEF are byte-exact.
			bBuf[k] = int32(roundPowerOfTwoUnsigned(uint32(SGRProjSgr-int(aBuf[k]))*uint32(bBuf[k])*uint32(oneByX[n-1]), SGRProjRecipBits))
		}
	}
}
