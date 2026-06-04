// Ported from libaom: aom_dsp/loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/frame"

// Filter6Edge applies AV1's six-sample deblocking filter along one horizontal
// or vertical edge. The x/y coordinate identifies q0, the first sample on the
// current block side of the edge.
func Filter6Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int32, y int32, length int32, thresholds Thresholds) error {
	xi, yi, lengthi := int(x), int(y), int(length)
	if err := validateFilter6Edge(dst, bytesPerSample, bitDepth, edge, xi, yi, lengthi); err != nil {
		return err
	}
	scale, params := filter4ParamsFor(bitDepth, thresholds)

	q0Base, step := filter4SampleOffset(dst, bytesPerSample, edge, xi, yi, 0)
	outer := edgeOuterStride(dst, bytesPerSample, edge)
	pix := dst.Pix
	if bytesPerSample == 1 {
		filter6EdgeImpl(pix, q0Base, step, outer, lengthi, scale, params)
		return nil
	}
	wide := params.widen()
	for i := range lengthi {
		q0 := q0Base + i*outer
		p2 := readSample(pix, bytesPerSample, q0-3*step)
		p1 := readSample(pix, bytesPerSample, q0-2*step)
		p0 := readSample(pix, bytesPerSample, q0-step)
		q0Sample := readSample(pix, bytesPerSample, q0)
		q1 := readSample(pix, bytesPerSample, q0+step)
		q2 := readSample(pix, bytesPerSample, q0+2*step)
		if !needsFilter6(p2, p1, p0, q0Sample, q1, q2, wide) {
			continue
		}
		p1, p0, q0Sample, q1 = filter6Samples(p2, p1, p0, q0Sample, q1, q2, scale, wide)
		writeSample(pix, bytesPerSample, q0-2*step, p1)
		writeSample(pix, bytesPerSample, q0-step, p0)
		writeSample(pix, bytesPerSample, q0, q0Sample)
		writeSample(pix, bytesPerSample, q0+step, q1)
	}
	return nil
}

// filter6EdgeImpl is the dispatch slot for the 8-bit six-sample deblocking
// kernel. It is resolved once at package init (see filter_wide_dispatch_*.go);
// filter6EdgePureGo is the canonical bit-exact reference every tuned variant
// must match.
var filter6EdgeImpl = filter6EdgePureGo

// filter6EdgePureGo applies the 8-bit six-sample filter to length sample
// positions along an edge. q0Base is the byte offset of the first q0 sample,
// step is the byte stride between adjacent taps, and outer is the byte stride
// between successive positions along the edge.
func filter6EdgePureGo(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	wide := params.widen()
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p2 := int(pix[q0-3*step])
		p1 := int(pix[q0-2*step])
		p0 := int(pix[q0-step])
		q0Sample := int(pix[q0])
		q1 := int(pix[q0+step])
		q2 := int(pix[q0+2*step])
		if !needsFilter6(p2, p1, p0, q0Sample, q1, q2, wide) {
			continue
		}
		p1, p0, q0Sample, q1 = filter6Samples(p2, p1, p0, q0Sample, q1, q2, scale, wide)
		pix[q0-2*step] = byte(p1)
		pix[q0-step] = byte(p0)
		pix[q0] = byte(q0Sample)
		pix[q0+step] = byte(q1)
	}
}

func validateFilter6Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int) error {
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
		if y < 3 || y+2 >= dst.Height || x > dst.Width-length {
			return ErrInvalidFilter
		}
	case EdgeVertical:
		if x < 3 || x+2 >= dst.Width || y > dst.Height-length {
			return ErrInvalidFilter
		}
	}

	lastX := x
	lastY := y
	switch edge {
	case EdgeHorizontal:
		lastX = x + length - 1
		lastY = y + 2
	case EdgeVertical:
		lastX = x + 2
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

func needsFilter6(p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, params filter4KernelParams) bool {
	return absInt(p2-p1) <= params.limit &&
		absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(q2-q1) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter6Samples(p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, flatThreshold int, params filter4KernelParams) (int, int, int, int) {
	flat := absInt(p1-p0) <= flatThreshold &&
		absInt(q1-q0) <= flatThreshold &&
		absInt(p2-p0) <= flatThreshold &&
		absInt(q2-q0) <= flatThreshold
	if !flat {
		return filter4Samples(p1, p0, q0, q1, params)
	}
	return roundPowerOfTwo(p2*3+p1*2+p0*2+q0, 3),
		roundPowerOfTwo(p2+p1*2+p0*2+q0*2+q1, 3),
		roundPowerOfTwo(p1+p0*2+q0*2+q1*2+q2, 3),
		roundPowerOfTwo(p0+q0*2+q1*2+q2*3, 3)
}

func roundPowerOfTwo(v int, n uint) int {
	return (v + (1 << (n - 1))) >> n
}
