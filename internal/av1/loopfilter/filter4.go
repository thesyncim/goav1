// Ported from libaom: aom_dsp/loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/frame"

// Filter4Edge applies AV1's narrow four-sample deblocking filter along one
// horizontal or vertical edge. The x/y coordinate identifies q0, the first
// sample on the current block side of the edge.
func Filter4Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	if err := validateFilter4Edge(dst, bytesPerSample, bitDepth, edge, x, y, length); err != nil {
		return err
	}
	shift := int(bitDepth - 8)
	scale := 1 << shift
	params := filter4Params{
		limit:  int(thresholds.Limit) * scale,
		blimit: int(thresholds.BlockLimit) * scale,
		hev:    int(thresholds.HighEdgeVariance) * scale,
		min:    -128 * scale,
		max:    128*scale - 1,
		center: 128 * scale,
	}

	q0Base, step := filter4SampleOffset(dst, bytesPerSample, edge, x, y, 0)
	outer := edgeOuterStride(dst, bytesPerSample, edge)
	pix := dst.Pix
	if bytesPerSample == 1 {
		for i := 0; i < length; i++ {
			q0 := q0Base + i*outer
			p1 := int(pix[q0-2*step])
			p0 := int(pix[q0-step])
			q0Sample := int(pix[q0])
			q1 := int(pix[q0+step])
			if !needsFilter4(p1, p0, q0Sample, q1, params) {
				continue
			}
			p1, p0, q0Sample, q1 = filter4Samples(p1, p0, q0Sample, q1, params)
			pix[q0-2*step] = byte(p1)
			pix[q0-step] = byte(p0)
			pix[q0] = byte(q0Sample)
			pix[q0+step] = byte(q1)
		}
		return nil
	}
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p1 := readSample(pix, bytesPerSample, q0-2*step)
		p0 := readSample(pix, bytesPerSample, q0-step)
		q0Sample := readSample(pix, bytesPerSample, q0)
		q1 := readSample(pix, bytesPerSample, q0+step)
		if !needsFilter4(p1, p0, q0Sample, q1, params) {
			continue
		}
		p1, p0, q0Sample, q1 = filter4Samples(p1, p0, q0Sample, q1, params)
		writeSample(pix, bytesPerSample, q0-2*step, p1)
		writeSample(pix, bytesPerSample, q0-step, p0)
		writeSample(pix, bytesPerSample, q0, q0Sample)
		writeSample(pix, bytesPerSample, q0+step, q1)
	}
	return nil
}

type filter4Params struct {
	limit  int
	blimit int
	hev    int
	min    int
	max    int
	center int
}

func validateFilter4Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int) error {
	if !validEdge(edge) || length <= 0 || dst.Stride <= 0 || dst.Width <= 0 || dst.Height <= 0 {
		return ErrInvalidFilter
	}
	if !validSampleLayout(bytesPerSample, bitDepth) {
		return ErrInvalidFilter
	}
	rowBytes, ok := checkedMul(dst.Width, bytesPerSample)
	if !ok || dst.Stride < rowBytes {
		return ErrInvalidFilter
	}
	if x < 0 || y < 0 {
		return ErrInvalidFilter
	}
	switch edge {
	case EdgeHorizontal:
		if y < 2 || y+1 >= dst.Height || x > dst.Width-length {
			return ErrInvalidFilter
		}
	case EdgeVertical:
		if x < 2 || x+1 >= dst.Width || y > dst.Height-length {
			return ErrInvalidFilter
		}
	}

	lastX := x
	lastY := y
	switch edge {
	case EdgeHorizontal:
		lastX = x + length - 1
		lastY = y + 1
	case EdgeVertical:
		lastX = x + 1
		lastY = y + length - 1
	}
	lastOffset, ok := sampleOffset(dst, bytesPerSample, lastX, lastY)
	if !ok {
		return ErrInvalidFilter
	}
	if lastOffset+bytesPerSample > len(dst.Pix) {
		return ErrInvalidFilter
	}
	return nil
}

func filter4SampleOffset(dst frame.Plane, bytesPerSample int, edge Edge, x int, y int, i int) (int, int) {
	if edge == EdgeHorizontal {
		return y*dst.Stride + (x+i)*bytesPerSample, dst.Stride
	}
	return (y+i)*dst.Stride + x*bytesPerSample, bytesPerSample
}

// edgeOuterStride returns the byte distance between successive samples along an
// edge as the loop index advances (perpendicular to the per-sample step).
func edgeOuterStride(dst frame.Plane, bytesPerSample int, edge Edge) int {
	if edge == EdgeHorizontal {
		return bytesPerSample
	}
	return dst.Stride
}

func needsFilter4(p1 int, p0 int, q0 int, q1 int, params filter4Params) bool {
	return absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter4Samples(p1 int, p0 int, q0 int, q1 int, params filter4Params) (int, int, int, int) {
	ps1 := p1 - params.center
	ps0 := p0 - params.center
	qs0 := q0 - params.center
	qs1 := q1 - params.center
	hev := absInt(p1-p0) > params.hev || absInt(q1-q0) > params.hev

	filter := 0
	if hev {
		filter = signedClamp(ps1-qs1, params.min, params.max)
	}
	filter = signedClamp(filter+3*(qs0-ps0), params.min, params.max)
	filter1 := signedClamp(filter+4, params.min, params.max) >> 3
	filter2 := signedClamp(filter+3, params.min, params.max) >> 3

	p0 = signedClamp(ps0+filter2, params.min, params.max) + params.center
	q0 = signedClamp(qs0-filter1, params.min, params.max) + params.center
	if !hev {
		outer := (filter1 + 1) >> 1
		p1 = signedClamp(ps1+outer, params.min, params.max) + params.center
		q1 = signedClamp(qs1-outer, params.min, params.max) + params.center
	}
	return p1, p0, q0, q1
}

func validSampleLayout(bytesPerSample int, bitDepth uint8) bool {
	return (bytesPerSample == 1 && bitDepth == 8) ||
		(bytesPerSample == 2 && (bitDepth == 10 || bitDepth == 12))
}

func sampleOffset(plane frame.Plane, bytesPerSample int, x int, y int) (int, bool) {
	if x < 0 || y < 0 || x >= plane.Width || y >= plane.Height {
		return 0, false
	}
	rowOffset, ok := checkedMul(y, plane.Stride)
	if !ok {
		return 0, false
	}
	colOffset, ok := checkedMul(x, bytesPerSample)
	if !ok {
		return 0, false
	}
	return checkedAdd(rowOffset, colOffset)
}

func readSample(pix []byte, bytesPerSample int, offset int) int {
	if bytesPerSample == 1 {
		return int(pix[offset])
	}
	return int(uint16(pix[offset]) | uint16(pix[offset+1])<<8)
}

func writeSample(pix []byte, bytesPerSample int, offset int, sample int) {
	if bytesPerSample == 1 {
		pix[offset] = byte(sample)
		return
	}
	pix[offset] = byte(sample)
	pix[offset+1] = byte(sample >> 8)
}

func signedClamp(v int, min int, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
