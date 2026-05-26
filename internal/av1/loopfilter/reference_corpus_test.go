package loopfilter

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

const (
	libaomLPFDeterministicSeed = 0xbaba
	libaomLPFNumCoeffs         = 1024
	libaomLPFStride            = 32
	libaomLPFIterations        = 64
)

func TestLoopFilterEdgesMatchLibaomOperationCorpus(t *testing.T) {
	for _, width := range []int{4, 6, 8, 14} {
		for _, bitDepth := range []uint8{8, 10, 12} {
			bytesPerSample := 1
			if bitDepth != 8 {
				bytesPerSample = 2
			}
			mask := (1 << int(bitDepth)) - 1
			for _, edge := range []Edge{EdgeHorizontal, EdgeVertical} {
				rnd := newLibaomTestRandom(libaomLPFDeterministicSeed)
				for i := range libaomLPFIterations {
					blimit := uint8(rnd.pseudoUniform(3*MaxLevel + 5))
					limit := uint8(rnd.pseudoUniform(MaxLevel + 1))
					hev := uint8(rnd.pseudoUniform(MaxLevel+1) >> 4)
					thresholds := Thresholds{Limit: limit, BlockLimit: blimit, HighEdgeVariance: hev}

					got := libaomInitLoopFilterInput(bytesPerSample, rnd, limit, mask, i)
					want := cloneTestPlane(got)
					applyCReferenceFilterByWidth(want, bytesPerSample, bitDepth, edge, width, 8, 8, 4, thresholds)

					if err := applyLoopFilterByWidth(width, got, bytesPerSample, bitDepth, edge, 8, 8, 4, thresholds); err != nil {
						t.Fatalf("width=%d bitdepth=%d edge=%d iteration=%d: %v", width, bitDepth, edge, i, err)
					}
					if !bytes.Equal(got.Pix, want.Pix) {
						t.Fatalf("width=%d bitdepth=%d edge=%d iteration=%d got=%v want=%v", width, bitDepth, edge, i, got.Pix, want.Pix)
					}
				}
			}
		}
	}
}

type libaomTestRandom struct {
	state uint32
}

func newLibaomTestRandom(seed uint32) *libaomTestRandom {
	return &libaomTestRandom{state: seed}
}

func (r *libaomTestRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *libaomTestRandom) rand16() uint16 {
	return uint16((r.generate(1<<31) >> 15) & 0xffff)
}

func (r *libaomTestRandom) rand8() uint8 {
	return uint8((r.generate(1<<31) >> 23) & 0xff)
}

func (r *libaomTestRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}

func libaomInitLoopFilterInput(bytesPerSample int, rnd *libaomTestRandom, limit uint8, mask int, iteration int) frame.Plane {
	tmp := make([]uint16, libaomLPFNumCoeffs)
	for j := 0; j < libaomLPFNumCoeffs; {
		val := rnd.rand8()
		if val&0x80 != 0 {
			tmp[j] = rnd.rand16()
			j++
			continue
		}
		for k := 0; k < int(val&0x1f)+1 && j < libaomLPFNumCoeffs; k++ {
			if j < 1 {
				tmp[j] = rnd.rand16()
			} else if val&0x20 != 0 {
				tmp[j] = uint16(int(tmp[j-1]) + int(limit) - 1)
			} else {
				tmp[j] = uint16(int(tmp[j-1]) - (int(limit) - 1))
			}
			j++
		}
	}

	for j := 0; j < libaomLPFNumCoeffs; {
		val := rnd.rand8()
		if val&0x80 != 0 {
			j++
			continue
		}
		for k := 0; k < int(val&0x1f)+1 && j < libaomLPFNumCoeffs; k++ {
			if j < 1 {
				tmp[j] = rnd.rand16()
			} else {
				dst := (j%libaomLPFStride)*libaomLPFStride + j/libaomLPFStride
				prev := ((j - 1) % libaomLPFStride * libaomLPFStride) + (j-1)/libaomLPFStride
				if val&0x20 != 0 {
					tmp[dst] = uint16(int(tmp[prev]) + int(limit) - 1)
				} else {
					tmp[dst] = uint16(int(tmp[prev]) - (int(limit) - 1))
				}
			}
			j++
		}
	}

	plane := testPlane(libaomLPFStride, libaomLPFStride, bytesPerSample, libaomLPFStride*bytesPerSample)
	for j := range libaomLPFNumCoeffs {
		src := j
		if iteration%2 == 0 {
			src = libaomLPFStride*(j%libaomLPFStride) + j/libaomLPFStride
		}
		value := tmp[src] & uint16(mask)
		setSample(plane, bytesPerSample, j%libaomLPFStride, j/libaomLPFStride, value)
	}
	return plane
}

func applyCReferenceFilterByWidth(plane frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, width int, x int, y int, length int, thresholds Thresholds) {
	for i := range length {
		q0, step := filter4SampleOffset(plane, bytesPerSample, edge, x, y, i)
		sample := make([]int, width)
		for j := range width {
			sample[j] = readSample(plane.Pix, bytesPerSample, q0+(j-width/2)*step)
		}
		ref := cReferenceFilterByWidth(width, sample, bitDepth, thresholds)
		for j, value := range ref {
			writeSample(plane.Pix, bytesPerSample, q0+(j-width/2)*step, value)
		}
	}
}

func applyLoopFilterByWidth(width int, plane frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	switch width {
	case 4:
		return Filter4Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 6:
		return Filter6Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 8:
		return Filter8Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 14:
		return Filter14Edge(plane, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	default:
		return ErrInvalidFilter
	}
}

func cReferenceFilterByWidth(width int, sample []int, bitDepth uint8, thresholds Thresholds) []int {
	out := make([]int, width)
	switch width {
	case 4:
		var in [4]int
		copy(in[:], sample)
		ref := cReferenceFilter4Samples(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 6:
		var in [6]int
		copy(in[:], sample)
		ref := cReferenceFilter6(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 8:
		var in [8]int
		copy(in[:], sample)
		ref := cReferenceFilter8(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 14:
		var in [14]int
		copy(in[:], sample)
		ref := cReferenceFilter14(in, bitDepth, thresholds)
		copy(out, ref[:])
	}
	return out
}

func cReferenceFilter4Samples(in [4]int, bitDepth uint8, thresholds Thresholds) [4]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale
	hev := int(thresholds.HighEdgeVariance) * scale
	min := -128 * scale
	max := 128*scale - 1
	center := 128 * scale

	p1, p0 := in[0], in[1]
	q0, q1 := in[2], in[3]
	if !cReferenceFilterMask2(limit, blimit, p1, p0, q0, q1) {
		return in
	}
	in[0], in[1], in[2], in[3] = cReferenceFilter4(p1, p0, q0, q1, hev, min, max, center)
	return in
}

func cReferenceFilterMask2(limit int, blimit int, p1 int, p0 int, q0 int, q1 int) bool {
	return cAbs(p1-p0) <= limit &&
		cAbs(q1-q0) <= limit &&
		cAbs(p0-q0)*2+cAbs(p1-q1)/2 <= blimit
}
