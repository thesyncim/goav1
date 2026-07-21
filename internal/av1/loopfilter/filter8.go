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
func Filter8Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int32, y int32, length int32, thresholds Thresholds) error {
	xi, yi, lengthi := int(x), int(y), int(length)
	if err := validateFilter8Edge(dst, bytesPerSample, bitDepth, edge, xi, yi, lengthi); err != nil {
		return err
	}
	Filter8EdgeTrusted(dst, bytesPerSample, bitDepth, edge, xi, yi, lengthi, thresholds)
	return nil
}

// Filter8EdgeTrusted applies the eight-sample filter after the caller has
// validated the plane layout, edge radius, run extent, sample width, and bit
// depth. It is the zero-validation seam used by the decoder's mask scanner.
func Filter8EdgeTrusted(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) {
	scale, params := filter4ParamsFor(bitDepth, thresholds)

	q0Base, step := filter4SampleOffset(dst, bytesPerSample, edge, x, y, 0)
	outer := edgeOuterStride(dst, bytesPerSample, edge)
	pix := dst.Pix
	if bytesPerSample == 1 {
		filter8EdgeImpl(pix, q0Base, step, outer, length, scale, params)
		return
	}
	filter8Edge16Impl(pix, q0Base, step, outer, length, scale, params)
}

// filter8Edge16Impl is the dispatch slot for the 10/12-bit (two-byte sample)
// eight-sample deblocking kernel. It is resolved once at package init (see
// filter_wide_dispatch_*.go); filter8Edge16PureGo is the canonical bit-exact
// reference every tuned variant must match.
var filter8Edge16Impl = filter8Edge16PureGo

// filter8Edge16PureGo applies the eight-sample filter to length sample
// positions on a two-byte (10/12-bit) plane. q0Base is the byte offset of the
// first q0 sample, step is the byte stride between adjacent taps, and outer is
// the byte stride between successive positions along the edge.
func filter8Edge16PureGo(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	wide := params.widen()
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p3 := readSample(pix, 2, q0-4*step)
		p2 := readSample(pix, 2, q0-3*step)
		p1 := readSample(pix, 2, q0-2*step)
		p0 := readSample(pix, 2, q0-step)
		q0Sample := readSample(pix, 2, q0)
		q1 := readSample(pix, 2, q0+step)
		q2 := readSample(pix, 2, q0+2*step)
		q3 := readSample(pix, 2, q0+3*step)
		if !needsFilter8(p3, p2, p1, p0, q0Sample, q1, q2, q3, wide) {
			continue
		}
		p2, p1, p0, q0Sample, q1, q2 = filter8Samples(p3, p2, p1, p0, q0Sample, q1, q2, q3, scale, wide)
		writeSample(pix, 2, q0-3*step, p2)
		writeSample(pix, 2, q0-2*step, p1)
		writeSample(pix, 2, q0-step, p0)
		writeSample(pix, 2, q0, q0Sample)
		writeSample(pix, 2, q0+step, q1)
		writeSample(pix, 2, q0+2*step, q2)
	}
}

// filter8EdgeImpl is the dispatch slot for the 8-bit eight-sample deblocking
// kernel. It is resolved once at package init (see filter_wide_dispatch_*.go);
// filter8EdgePureGo is the canonical bit-exact reference every tuned variant
// must match.
var filter8EdgeImpl = filter8EdgePureGo

// filter8EdgePureGo applies the 8-bit eight-sample filter to length sample
// positions along an edge. q0Base is the byte offset of the first q0 sample,
// step is the byte stride between adjacent taps, and outer is the byte stride
// between successive positions along the edge.
func filter8EdgePureGo(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	wide := params.widen()
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
		if !needsFilter8(p3, p2, p1, p0, q0Sample, q1, q2, q3, wide) {
			continue
		}
		p2, p1, p0, q0Sample, q1, q2 = filter8Samples(p3, p2, p1, p0, q0Sample, q1, q2, q3, scale, wide)
		pix[q0-3*step] = byte(p2)
		pix[q0-2*step] = byte(p1)
		pix[q0-step] = byte(p0)
		pix[q0] = byte(q0Sample)
		pix[q0+step] = byte(q1)
		pix[q0+2*step] = byte(q2)
	}
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

func needsFilter8(p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int, params filter4KernelParams) bool {
	return absInt(p3-p2) <= params.limit &&
		absInt(p2-p1) <= params.limit &&
		absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(q2-q1) <= params.limit &&
		absInt(q3-q2) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter8Samples(p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int, flatThreshold int, params filter4KernelParams) (int, int, int, int, int, int) {
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
