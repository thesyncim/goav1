package loopfilter

import "github.com/thesyncim/goav1/internal/av1/frame"

// Filter14Edge applies AV1's wide fourteen-sample deblocking filter along one
// horizontal or vertical edge. The x/y coordinate identifies q0, the first
// sample on the current block side of the edge.
func Filter14Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	if err := validateFilter14Edge(dst, bytesPerSample, bitDepth, edge, x, y, length); err != nil {
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

	for i := range length {
		q0, step := filter4SampleOffset(dst, bytesPerSample, edge, x, y, i)
		p6 := readSample(dst.Pix, bytesPerSample, q0-7*step)
		p5 := readSample(dst.Pix, bytesPerSample, q0-6*step)
		p4 := readSample(dst.Pix, bytesPerSample, q0-5*step)
		p3 := readSample(dst.Pix, bytesPerSample, q0-4*step)
		p2 := readSample(dst.Pix, bytesPerSample, q0-3*step)
		p1 := readSample(dst.Pix, bytesPerSample, q0-2*step)
		p0 := readSample(dst.Pix, bytesPerSample, q0-step)
		q0Sample := readSample(dst.Pix, bytesPerSample, q0)
		q1 := readSample(dst.Pix, bytesPerSample, q0+step)
		q2 := readSample(dst.Pix, bytesPerSample, q0+2*step)
		q3 := readSample(dst.Pix, bytesPerSample, q0+3*step)
		q4 := readSample(dst.Pix, bytesPerSample, q0+4*step)
		q5 := readSample(dst.Pix, bytesPerSample, q0+5*step)
		q6 := readSample(dst.Pix, bytesPerSample, q0+6*step)
		if !needsFilter8(p3, p2, p1, p0, q0Sample, q1, q2, q3, params) {
			continue
		}
		p5, p4, p3, p2, p1, p0, q0Sample, q1, q2, q3, q4, q5 = filter14Samples(
			p6, p5, p4, p3, p2, p1, p0,
			q0Sample, q1, q2, q3, q4, q5, q6,
			scale, params,
		)
		writeSample(dst.Pix, bytesPerSample, q0-6*step, p5)
		writeSample(dst.Pix, bytesPerSample, q0-5*step, p4)
		writeSample(dst.Pix, bytesPerSample, q0-4*step, p3)
		writeSample(dst.Pix, bytesPerSample, q0-3*step, p2)
		writeSample(dst.Pix, bytesPerSample, q0-2*step, p1)
		writeSample(dst.Pix, bytesPerSample, q0-step, p0)
		writeSample(dst.Pix, bytesPerSample, q0, q0Sample)
		writeSample(dst.Pix, bytesPerSample, q0+step, q1)
		writeSample(dst.Pix, bytesPerSample, q0+2*step, q2)
		writeSample(dst.Pix, bytesPerSample, q0+3*step, q3)
		writeSample(dst.Pix, bytesPerSample, q0+4*step, q4)
		writeSample(dst.Pix, bytesPerSample, q0+5*step, q5)
	}
	return nil
}

func validateFilter14Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int) error {
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
		if y < 7 || y+6 >= dst.Height || x > dst.Width-length {
			return ErrInvalidFilter
		}
	case EdgeVertical:
		if x < 7 || x+6 >= dst.Width || y > dst.Height-length {
			return ErrInvalidFilter
		}
	}

	lastX := x
	lastY := y
	switch edge {
	case EdgeHorizontal:
		lastX = x + length - 1
		lastY = y + 6
	case EdgeVertical:
		lastX = x + 6
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

func filter14Samples(p6 int, p5 int, p4 int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int, q4 int, q5 int, q6 int, flatThreshold int, params filter4Params) (int, int, int, int, int, int, int, int, int, int, int, int) {
	flat := flatMask4(flatThreshold, p3, p2, p1, p0, q0, q1, q2, q3)
	flat2 := flatMask4(flatThreshold, p6, p5, p4, p0, q0, q4, q5, q6)
	if !flat || !flat2 {
		p2, p1, p0, q0, q1, q2 = filter8Samples(p3, p2, p1, p0, q0, q1, q2, q3, flatThreshold, params)
		return p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5
	}

	return roundPowerOfTwo(p6*7+p5*2+p4*2+p3+p2+p1+p0+q0, 4),
		roundPowerOfTwo(p6*5+p5*2+p4*2+p3*2+p2+p1+p0+q0+q1, 4),
		roundPowerOfTwo(p6*4+p5+p4*2+p3*2+p2*2+p1+p0+q0+q1+q2, 4),
		roundPowerOfTwo(p6*3+p5+p4+p3*2+p2*2+p1*2+p0+q0+q1+q2+q3, 4),
		roundPowerOfTwo(p6*2+p5+p4+p3+p2*2+p1*2+p0*2+q0+q1+q2+q3+q4, 4),
		roundPowerOfTwo(p6+p5+p4+p3+p2+p1*2+p0*2+q0*2+q1+q2+q3+q4+q5, 4),
		roundPowerOfTwo(p5+p4+p3+p2+p1+p0*2+q0*2+q1*2+q2+q3+q4+q5+q6, 4),
		roundPowerOfTwo(p4+p3+p2+p1+p0+q0*2+q1*2+q2*2+q3+q4+q5+q6*2, 4),
		roundPowerOfTwo(p3+p2+p1+p0+q0+q1*2+q2*2+q3*2+q4+q5+q6*3, 4),
		roundPowerOfTwo(p2+p1+p0+q0+q1+q2*2+q3*2+q4*2+q5+q6*4, 4),
		roundPowerOfTwo(p1+p0+q0+q1+q2+q3*2+q4*2+q5*2+q6*5, 4),
		roundPowerOfTwo(p0+q0+q1+q2+q3+q4*2+q5*2+q6*7, 4)
}

func flatMask4(thresh int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int) bool {
	return absInt(p1-p0) <= thresh &&
		absInt(q1-q0) <= thresh &&
		absInt(p2-p0) <= thresh &&
		absInt(q2-q0) <= thresh &&
		absInt(p3-p0) <= thresh &&
		absInt(q3-q0) <= thresh
}
