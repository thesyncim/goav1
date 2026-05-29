// Ported from libaom: aom_dsp/loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/frame"

// Filter8Edge applies AV1's eight-sample deblocking filter along one
// horizontal or vertical edge. The x/y coordinate identifies q0, the first
// sample on the current block side of the edge.
func Filter8Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	if err := validateFilter8Edge(dst, bytesPerSample, bitDepth, edge, x, y, length); err != nil {
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
			p3 := int(pix[q0-4*step])
			p2 := int(pix[q0-3*step])
			p1 := int(pix[q0-2*step])
			p0 := int(pix[q0-step])
			q0Sample := int(pix[q0])
			q1 := int(pix[q0+step])
			q2 := int(pix[q0+2*step])
			q3 := int(pix[q0+3*step])
			if !needsFilter8(p3, p2, p1, p0, q0Sample, q1, q2, q3, params) {
				continue
			}
			p2, p1, p0, q0Sample, q1, q2 = filter8Samples(p3, p2, p1, p0, q0Sample, q1, q2, q3, scale, params)
			pix[q0-3*step] = byte(p2)
			pix[q0-2*step] = byte(p1)
			pix[q0-step] = byte(p0)
			pix[q0] = byte(q0Sample)
			pix[q0+step] = byte(q1)
			pix[q0+2*step] = byte(q2)
		}
		return nil
	}
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p3 := readSample(pix, bytesPerSample, q0-4*step)
		p2 := readSample(pix, bytesPerSample, q0-3*step)
		p1 := readSample(pix, bytesPerSample, q0-2*step)
		p0 := readSample(pix, bytesPerSample, q0-step)
		q0Sample := readSample(pix, bytesPerSample, q0)
		q1 := readSample(pix, bytesPerSample, q0+step)
		q2 := readSample(pix, bytesPerSample, q0+2*step)
		q3 := readSample(pix, bytesPerSample, q0+3*step)
		if !needsFilter8(p3, p2, p1, p0, q0Sample, q1, q2, q3, params) {
			continue
		}
		p2, p1, p0, q0Sample, q1, q2 = filter8Samples(p3, p2, p1, p0, q0Sample, q1, q2, q3, scale, params)
		writeSample(pix, bytesPerSample, q0-3*step, p2)
		writeSample(pix, bytesPerSample, q0-2*step, p1)
		writeSample(pix, bytesPerSample, q0-step, p0)
		writeSample(pix, bytesPerSample, q0, q0Sample)
		writeSample(pix, bytesPerSample, q0+step, q1)
		writeSample(pix, bytesPerSample, q0+2*step, q2)
	}
	return nil
}

func validateFilter8Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int) error {
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
		if y < 4 || y+3 >= dst.Height || x > dst.Width-length {
			return ErrInvalidFilter
		}
	case EdgeVertical:
		if x < 4 || x+3 >= dst.Width || y > dst.Height-length {
			return ErrInvalidFilter
		}
	}

	lastX := x
	lastY := y
	switch edge {
	case EdgeHorizontal:
		lastX = x + length - 1
		lastY = y + 3
	case EdgeVertical:
		lastX = x + 3
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

func needsFilter8(p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int, params filter4Params) bool {
	return absInt(p3-p2) <= params.limit &&
		absInt(p2-p1) <= params.limit &&
		absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(q2-q1) <= params.limit &&
		absInt(q3-q2) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter8Samples(p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int, flatThreshold int, params filter4Params) (int, int, int, int, int, int) {
	flat := absInt(p1-p0) <= flatThreshold &&
		absInt(q1-q0) <= flatThreshold &&
		absInt(p2-p0) <= flatThreshold &&
		absInt(q2-q0) <= flatThreshold &&
		absInt(p3-p0) <= flatThreshold &&
		absInt(q3-q0) <= flatThreshold
	if !flat {
		p1, p0, q0, q1 = filter4Samples(p1, p0, q0, q1, params)
		return p2, p1, p0, q0, q1, q2
	}
	return roundPowerOfTwo(p3+p3+p3+2*p2+p1+p0+q0, 3),
		roundPowerOfTwo(p3+p3+p2+2*p1+p0+q0+q1, 3),
		roundPowerOfTwo(p3+p2+p1+2*p0+q0+q1+q2, 3),
		roundPowerOfTwo(p2+p1+p0+2*q0+q1+q2+q3, 3),
		roundPowerOfTwo(p1+p0+q0+2*q1+q2+q3+q3, 3),
		roundPowerOfTwo(p0+q0+q1+2*q2+q3+q3+q3, 3)
}
