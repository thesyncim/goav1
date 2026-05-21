package loopfilter

import "github.com/thesyncim/goav1/internal/av1/frame"

// Filter6Edge applies AV1's six-sample deblocking filter along one horizontal
// or vertical edge. The x/y coordinate identifies q0, the first sample on the
// current block side of the edge.
func Filter6Edge(dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	if err := validateFilter6Edge(dst, bytesPerSample, bitDepth, edge, x, y, length); err != nil {
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

	for i := 0; i < length; i++ {
		q0, step := filter4SampleOffset(dst, bytesPerSample, edge, x, y, i)
		p2 := readSample(dst.Pix, bytesPerSample, q0-3*step)
		p1 := readSample(dst.Pix, bytesPerSample, q0-2*step)
		p0 := readSample(dst.Pix, bytesPerSample, q0-step)
		q0Sample := readSample(dst.Pix, bytesPerSample, q0)
		q1 := readSample(dst.Pix, bytesPerSample, q0+step)
		q2 := readSample(dst.Pix, bytesPerSample, q0+2*step)
		if !needsFilter6(p2, p1, p0, q0Sample, q1, q2, params) {
			continue
		}
		p1, p0, q0Sample, q1 = filter6Samples(p2, p1, p0, q0Sample, q1, q2, scale, params)
		writeSample(dst.Pix, bytesPerSample, q0-2*step, p1)
		writeSample(dst.Pix, bytesPerSample, q0-step, p0)
		writeSample(dst.Pix, bytesPerSample, q0, q0Sample)
		writeSample(dst.Pix, bytesPerSample, q0+step, q1)
	}
	return nil
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

func needsFilter6(p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, params filter4Params) bool {
	return absInt(p2-p1) <= params.limit &&
		absInt(p1-p0) <= params.limit &&
		absInt(q1-q0) <= params.limit &&
		absInt(q2-q1) <= params.limit &&
		absInt(p0-q0)*2+absInt(p1-q1)/2 <= params.blimit
}

func filter6Samples(p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, flatThreshold int, params filter4Params) (int, int, int, int) {
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
