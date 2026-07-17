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
func Filter4Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int32, y int32, length int32, thresholds Thresholds) error {
	xi, yi, lengthi := int(x), int(y), int(length)
	if err := validateFilter4Edge(dst, bytesPerSample, bitDepth, edge, xi, yi, lengthi); err != nil {
		return err
	}
	Filter4EdgeTrusted(dst, bytesPerSample, bitDepth, edge, xi, yi, lengthi, thresholds)
	return nil
}

// Filter4EdgeTrusted applies the four-sample filter after the caller has
// validated the plane layout, edge radius, run extent, sample width, and bit
// depth. It is the zero-validation seam used by the decoder's mask scanner.
func Filter4EdgeTrusted(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) {
	_, params := filter4ParamsFor(bitDepth, thresholds)

	q0Base, step := filter4SampleOffset(dst, bytesPerSample, edge, x, y, 0)
	outer := edgeOuterStride(dst, bytesPerSample, edge)
	pix := dst.Pix
	if bytesPerSample == 1 {
		filter4EdgeImpl(pix, q0Base, step, outer, length, params)
		return
	}
	filter4Edge16Impl(pix, q0Base, step, outer, length, params)
}

type filter4Params struct {
	limit  int16
	blimit int16
	hev    int16
	min    int16
	max    int16
	center int16
}

type filter4KernelParams struct {
	limit  int
	blimit int
	hev    int
	min    int
	max    int
	center int
}

func filter4ParamsFor(bitDepth uint8, thresholds Thresholds) (int, filter4Params) {
	scale := 1 << int(bitDepth-8)
	return scale, filter4Params{
		limit:  int16(int(thresholds.Limit) * scale),
		blimit: int16(int(thresholds.BlockLimit) * scale),
		hev:    int16(int(thresholds.HighEdgeVariance) * scale),
		min:    int16(-128 * scale),
		max:    int16(128*scale - 1),
		center: int16(128 * scale),
	}
}

func (p filter4Params) widen() filter4KernelParams {
	return filter4KernelParams{
		limit:  int(p.limit),
		blimit: int(p.blimit),
		hev:    int(p.hev),
		min:    int(p.min),
		max:    int(p.max),
		center: int(p.center),
	}
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

// filter4EdgeImpl is the dispatch slot for the 8-bit narrow four-sample
// deblocking kernel, processing every sample position along one edge. It is
// resolved once at package init (see filter4_dispatch_*.go); the per-edge cost
// is one indirect call with no feature-detection branch. filter4EdgePureGo is
// the canonical bit-exact reference every tuned variant must match.
var filter4EdgeImpl = filter4EdgePureGo

// filter4EdgePureGo applies the 8-bit narrow filter to length sample positions
// along an edge. q0Base is the byte offset of the first q0 sample, step is the
// byte stride between adjacent taps (perpendicular to the edge), and outer is
// the byte stride between successive positions along the edge.
func filter4EdgePureGo(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	wide := params.widen()
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p1 := int(pix[q0-2*step])
		p0 := int(pix[q0-step])
		q0Sample := int(pix[q0])
		q1 := int(pix[q0+step])
		if !needsFilter4(p1, p0, q0Sample, q1, wide) {
			continue
		}
		p1, p0, q0Sample, q1 = filter4Samples(p1, p0, q0Sample, q1, wide)
		pix[q0-2*step] = byte(p1)
		pix[q0-step] = byte(p0)
		pix[q0] = byte(q0Sample)
		pix[q0+step] = byte(q1)
	}
}

// filter4Edge16Impl is the dispatch slot for the 10/12-bit (two-byte sample)
// narrow four-sample deblocking kernel. It is resolved once at package init
// (see filter4_dispatch_*.go); filter4Edge16PureGo is the canonical bit-exact
// reference every tuned variant must match.
var filter4Edge16Impl = filter4Edge16PureGo

// filter4Edge16PureGo applies the narrow filter to length sample positions on a
// two-byte (10/12-bit) plane. q0Base is the byte offset of the first q0 sample,
// step is the byte stride between adjacent taps, and outer is the byte stride
// between successive positions along the edge.
func filter4Edge16PureGo(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	wide := params.widen()
	for i := 0; i < length; i++ {
		q0 := q0Base + i*outer
		p1 := readSample(pix, 2, q0-2*step)
		p0 := readSample(pix, 2, q0-step)
		q0Sample := readSample(pix, 2, q0)
		q1 := readSample(pix, 2, q0+step)
		if !needsFilter4(p1, p0, q0Sample, q1, wide) {
			continue
		}
		p1, p0, q0Sample, q1 = filter4Samples(p1, p0, q0Sample, q1, wide)
		writeSample(pix, 2, q0-2*step, p1)
		writeSample(pix, 2, q0-step, p0)
		writeSample(pix, 2, q0, q0Sample)
		writeSample(pix, 2, q0+step, q1)
	}
}

func needsFilter4(p1 int, p0 int, q0 int, q1 int, params filter4KernelParams) bool {
	return absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter4Samples(p1 int, p0 int, q0 int, q1 int, params filter4KernelParams) (int, int, int, int) {
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
