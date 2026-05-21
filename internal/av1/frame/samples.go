package frame

// SamplePlane is a caller-owned uint16 view of a frame plane. Stride is in
// samples, not bytes.
type SamplePlane struct {
	Pix    []uint16
	Stride int
	Width  int
	Height int
}

// SamplePlaneLen reports the caller-owned uint16 scratch length required to
// hold plane with the same row stride expressed in samples.
func SamplePlaneLen(plane Plane, bytesPerSample int) (int, error) {
	_, need, err := samplePlaneLayout(plane, bytesPerSample)
	return need, err
}

// LoadSamplePlane expands an 8-bit or little-endian 16-bit byte plane into
// caller-owned uint16 sample storage.
func LoadSamplePlane(dst []uint16, src Plane, bytesPerSample int) (SamplePlane, error) {
	strideSamples, need, err := samplePlaneLayout(src, bytesPerSample)
	if err != nil {
		return SamplePlane{}, err
	}
	if len(dst) < need {
		return SamplePlane{}, ErrShortBuffer
	}
	samples := dst[:need]
	for y := 0; y < src.Height; y++ {
		srcLine := src.Pix[y*src.Stride : y*src.Stride+src.Width*bytesPerSample]
		dstLine := samples[y*strideSamples : y*strideSamples+src.Width]
		switch bytesPerSample {
		case 1:
			for x := range dstLine {
				dstLine[x] = uint16(srcLine[x])
			}
		case 2:
			for x := range dstLine {
				off := x * 2
				dstLine[x] = uint16(srcLine[off]) | uint16(srcLine[off+1])<<8
			}
		}
	}
	return SamplePlane{Pix: samples, Stride: strideSamples, Width: src.Width, Height: src.Height}, nil
}

// StoreSamplePlane writes visible samples from src into an 8-bit or
// little-endian 16-bit byte plane.
func StoreSamplePlane(dst Plane, bytesPerSample int, src SamplePlane) error {
	if _, _, err := samplePlaneLayout(dst, bytesPerSample); err != nil {
		return err
	}
	if !samplePlaneFits(src) || src.Width != dst.Width || src.Height != dst.Height {
		return ErrInvalidPlane
	}
	for y := 0; y < dst.Height; y++ {
		dstLine := dst.Pix[y*dst.Stride : y*dst.Stride+dst.Width*bytesPerSample]
		srcLine := src.Pix[y*src.Stride : y*src.Stride+src.Width]
		switch bytesPerSample {
		case 1:
			for x, sample := range srcLine {
				if sample > 0xff {
					return ErrInvalidPlane
				}
				dstLine[x] = byte(sample)
			}
		case 2:
			for x, sample := range srcLine {
				off := x * 2
				dstLine[off] = byte(sample)
				dstLine[off+1] = byte(sample >> 8)
			}
		}
	}
	return nil
}

func samplePlaneLayout(plane Plane, bytesPerSample int) (int, int, error) {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return 0, 0, ErrInvalidPlane
	}
	if plane.Width < 0 || plane.Height < 0 || plane.Stride < 0 {
		return 0, 0, ErrInvalidPlane
	}
	if plane.Width == 0 || plane.Height == 0 {
		if len(plane.Pix) != 0 {
			return 0, 0, ErrInvalidPlane
		}
		return 0, 0, nil
	}
	rowBytes, ok := checkedMul(plane.Width, bytesPerSample)
	if !ok || plane.Stride < rowBytes || plane.Stride%bytesPerSample != 0 {
		return 0, 0, ErrInvalidPlane
	}
	lastRow, ok := checkedMul(plane.Height-1, plane.Stride)
	if !ok {
		return 0, 0, ErrInvalidPlane
	}
	needBytes, ok := checkedAdd(lastRow, rowBytes)
	if !ok || len(plane.Pix) < needBytes {
		return 0, 0, ErrInvalidPlane
	}
	strideSamples := plane.Stride / bytesPerSample
	need, ok := checkedMul(strideSamples, plane.Height)
	if !ok {
		return 0, 0, ErrInvalidPlane
	}
	return strideSamples, need, nil
}

func samplePlaneFits(plane SamplePlane) bool {
	if plane.Width < 0 || plane.Height < 0 || plane.Stride < 0 {
		return false
	}
	if plane.Width == 0 || plane.Height == 0 {
		return len(plane.Pix) == 0
	}
	if plane.Stride < plane.Width {
		return false
	}
	lastRow, ok := checkedMul(plane.Height-1, plane.Stride)
	if !ok {
		return false
	}
	need, ok := checkedAdd(lastRow, plane.Width)
	return ok && len(plane.Pix) >= need
}
