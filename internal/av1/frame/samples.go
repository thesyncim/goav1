// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

// SamplePlane is a caller-owned uint16 view of a frame plane. Stride is in
// samples, not bytes.
type SamplePlane struct {
	Pix    []uint16
	Stride int
	Width  int
	Height int
}

// BorderedSamplePlane is caller-owned uint16 storage with an AV1-style visible
// origin inside border rows and columns. Stride is in samples.
type BorderedSamplePlane struct {
	Pix    []uint16
	Stride int
	Origin int
	Width  int
	Height int

	BorderHorz int
	BorderVert int
}

// BorderedSamplePlaneLayout reports the caller-owned layout for a bordered
// sample plane.
type BorderedSamplePlaneLayout struct {
	Stride int
	Origin int
	Rows   int
	Len    int
}

// SamplePlaneLen reports the caller-owned uint16 scratch length required to
// hold plane with the same row stride expressed in samples.
func SamplePlaneLen(plane Plane, bytesPerSample int) (int, error) {
	_, need, err := samplePlaneLayout(plane, bytesPerSample, false)
	return need, err
}

// BindSamplePlane binds caller-owned uint16 storage without loading visible
// samples. The returned view is suitable for output buffers that will be fully
// written before being stored.
func BindSamplePlane(dst []uint16, plane Plane, bytesPerSample int) (SamplePlane, error) {
	strideSamples, need, err := samplePlaneLayout(plane, bytesPerSample, false)
	if err != nil {
		return SamplePlane{}, err
	}
	if len(dst) < need {
		return SamplePlane{}, ErrShortBuffer
	}
	return SamplePlane{Pix: dst[:need], Stride: strideSamples, Width: plane.Width, Height: plane.Height}, nil
}

// BorderedSamplePlaneLen reports the caller-owned uint16 scratch layout
// required to hold plane plus horizontal and vertical sample borders. align is
// a power-of-two sample alignment for the returned stride; values <= 1 mean no
// extra alignment.
func BorderedSamplePlaneLen(plane Plane, bytesPerSample int, borderHorz int, borderVert int, align int) (BorderedSamplePlaneLayout, error) {
	return borderedSamplePlaneLayout(plane, bytesPerSample, borderHorz, borderVert, align)
}

// BindBorderedSamplePlane binds caller-owned bordered uint16 storage without
// loading visible samples. The returned view is suitable for restoration output
// buffers that will be fully written before their visible region is stored.
func BindBorderedSamplePlane(dst []uint16, plane Plane, bytesPerSample int, borderHorz int, borderVert int, align int) (BorderedSamplePlane, error) {
	layout, err := borderedSamplePlaneLayout(plane, bytesPerSample, borderHorz, borderVert, align)
	if err != nil {
		return BorderedSamplePlane{}, err
	}
	if len(dst) < layout.Len {
		return BorderedSamplePlane{}, ErrShortBuffer
	}
	return BorderedSamplePlane{
		Pix:        dst[:layout.Len],
		Stride:     layout.Stride,
		Origin:     layout.Origin,
		Width:      plane.Width,
		Height:     plane.Height,
		BorderHorz: borderHorz,
		BorderVert: borderVert,
	}, nil
}

// LoadSamplePlane expands an 8-bit or little-endian 16-bit byte plane into
// caller-owned uint16 sample storage.
func LoadSamplePlane(dst []uint16, src Plane, bytesPerSample int) (SamplePlane, error) {
	strideSamples, need, err := samplePlaneLayout(src, bytesPerSample, true)
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
		loadSampleLine(dstLine, srcLine, bytesPerSample)
	}
	return SamplePlane{Pix: samples, Stride: strideSamples, Width: src.Width, Height: src.Height}, nil
}

// LoadSamplePlaneFull behaves like LoadSamplePlane but also loads the
// past-visible row-stride padding columns (samples [Width, Stride/bytesPerSample))
// for each visible row directly from src.Pix. The returned SamplePlane keeps
// Width=src.Width so the visible region is reported unchanged; the second
// return value reports the number of valid columns (in samples) loaded into
// dst.
//
// This variant exists for postfilter passes such as CDEF whose superblock-edge
// reads extend past the visible plane width into the row-stride padding bytes
// written by the reconstruction stage (libaom's cdef_prepare_fb central read
// fetches hsize+HBORDER columns from the YV12 buffer past the visible crop).
// The default LoadSamplePlane keeps the past-visible columns at whatever value
// dst already holds, which leaves them at zero for fresh scratch and diverges
// from libaom.
//
// src.Pix must be at least Height*Stride bytes long so the past-visible columns
// of the final visible row remain in-bounds; this is the normal layout produced
// by Bind.
func LoadSamplePlaneFull(dst []uint16, src Plane, bytesPerSample int) (SamplePlane, int, error) {
	strideSamples, need, err := samplePlaneLayout(src, bytesPerSample, true)
	if err != nil {
		return SamplePlane{}, 0, err
	}
	if len(dst) < need {
		return SamplePlane{}, 0, ErrShortBuffer
	}
	if src.Width == 0 || src.Height == 0 {
		return SamplePlane{Pix: dst[:need], Stride: strideSamples, Width: src.Width, Height: src.Height}, 0, nil
	}
	fullBytes, ok := checkedMul(src.Height, src.Stride)
	if !ok || len(src.Pix) < fullBytes {
		return SamplePlane{}, 0, ErrInvalidPlane
	}
	samples := dst[:need]
	loadedWidth := strideSamples
	for y := 0; y < src.Height; y++ {
		srcLine := src.Pix[y*src.Stride : y*src.Stride+loadedWidth*bytesPerSample]
		dstLine := samples[y*strideSamples : y*strideSamples+loadedWidth]
		loadSampleLine(dstLine, srcLine, bytesPerSample)
	}
	return SamplePlane{Pix: samples, Stride: strideSamples, Width: src.Width, Height: src.Height}, loadedWidth, nil
}

// LoadBorderedSamplePlane expands an 8-bit or little-endian 16-bit byte plane
// into caller-owned bordered uint16 sample storage. Border samples are left
// unchanged for callers such as loop restoration to fill with their normal
// frame-extension pass.
func LoadBorderedSamplePlane(dst []uint16, src Plane, bytesPerSample int, borderHorz int, borderVert int, align int) (BorderedSamplePlane, error) {
	layout, err := borderedSamplePlaneLayout(src, bytesPerSample, borderHorz, borderVert, align)
	if err != nil {
		return BorderedSamplePlane{}, err
	}
	if _, _, err := samplePlaneLayout(src, bytesPerSample, true); err != nil {
		return BorderedSamplePlane{}, err
	}
	if len(dst) < layout.Len {
		return BorderedSamplePlane{}, ErrShortBuffer
	}
	samples := dst[:layout.Len]
	for y := 0; y < src.Height; y++ {
		srcLine := src.Pix[y*src.Stride : y*src.Stride+src.Width*bytesPerSample]
		dstStart := layout.Origin + y*layout.Stride
		dstLine := samples[dstStart : dstStart+src.Width]
		loadSampleLine(dstLine, srcLine, bytesPerSample)
	}
	return BorderedSamplePlane{
		Pix:        samples,
		Stride:     layout.Stride,
		Origin:     layout.Origin,
		Width:      src.Width,
		Height:     src.Height,
		BorderHorz: borderHorz,
		BorderVert: borderVert,
	}, nil
}

// StoreSamplePlane writes visible samples from src into an 8-bit or
// little-endian 16-bit byte plane.
func StoreSamplePlane(dst Plane, bytesPerSample int, src SamplePlane) error {
	if _, _, err := samplePlaneLayout(dst, bytesPerSample, true); err != nil {
		return err
	}
	if !samplePlaneFits(src) || src.Width != dst.Width || src.Height != dst.Height {
		return ErrInvalidPlane
	}
	for y := 0; y < dst.Height; y++ {
		dstLine := dst.Pix[y*dst.Stride : y*dst.Stride+dst.Width*bytesPerSample]
		srcLine := src.Pix[y*src.Stride : y*src.Stride+src.Width]
		if err := storeSampleLine(dstLine, srcLine, bytesPerSample); err != nil {
			return err
		}
	}
	return nil
}

// StoreBorderedSamplePlane writes the visible region from src into an 8-bit or
// little-endian 16-bit byte plane.
func StoreBorderedSamplePlane(dst Plane, bytesPerSample int, src BorderedSamplePlane) error {
	if _, _, err := samplePlaneLayout(dst, bytesPerSample, true); err != nil {
		return err
	}
	if !borderedSamplePlaneFits(src) || src.Width != dst.Width || src.Height != dst.Height {
		return ErrInvalidPlane
	}
	for y := 0; y < dst.Height; y++ {
		dstLine := dst.Pix[y*dst.Stride : y*dst.Stride+dst.Width*bytesPerSample]
		srcStart := src.Origin + y*src.Stride
		srcLine := src.Pix[srcStart : srcStart+src.Width]
		if err := storeSampleLine(dstLine, srcLine, bytesPerSample); err != nil {
			return err
		}
	}
	return nil
}

func samplePlaneLayout(plane Plane, bytesPerSample int, requirePix bool) (int, int, error) {
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
	if !ok || (requirePix && len(plane.Pix) < needBytes) {
		return 0, 0, ErrInvalidPlane
	}
	strideSamples := plane.Stride / bytesPerSample
	need, ok := checkedMul(strideSamples, plane.Height)
	if !ok {
		return 0, 0, ErrInvalidPlane
	}
	return strideSamples, need, nil
}

func borderedSamplePlaneLayout(plane Plane, bytesPerSample int, borderHorz int, borderVert int, align int) (BorderedSamplePlaneLayout, error) {
	if borderHorz < 0 || borderVert < 0 {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	if align <= 0 {
		align = 1
	}
	if align&(align-1) != 0 {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	if _, _, err := samplePlaneLayout(plane, bytesPerSample, false); err != nil {
		return BorderedSamplePlaneLayout{}, err
	}
	if plane.Width == 0 || plane.Height == 0 {
		if borderHorz != 0 || borderVert != 0 {
			return BorderedSamplePlaneLayout{}, ErrInvalidPlane
		}
		return BorderedSamplePlaneLayout{}, nil
	}
	borderWidth, ok := checkedMul(borderHorz, 2)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	width, ok := checkedAdd(plane.Width, borderWidth)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	stride, ok := checkedAlign(width, align)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	borderRows, ok := checkedMul(borderVert, 2)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	rows, ok := checkedAdd(plane.Height, borderRows)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	originRow, ok := checkedMul(borderVert, stride)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	origin, ok := checkedAdd(originRow, borderHorz)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	need, ok := checkedMul(stride, rows)
	if !ok {
		return BorderedSamplePlaneLayout{}, ErrInvalidPlane
	}
	return BorderedSamplePlaneLayout{Stride: stride, Origin: origin, Rows: rows, Len: need}, nil
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

func borderedSamplePlaneFits(plane BorderedSamplePlane) bool {
	if plane.Width < 0 || plane.Height < 0 || plane.Stride < 0 ||
		plane.Origin < 0 || plane.BorderHorz < 0 || plane.BorderVert < 0 {
		return false
	}
	if plane.Width == 0 || plane.Height == 0 {
		return len(plane.Pix) == 0 && plane.Origin == 0
	}
	if plane.Stride < plane.Width || plane.Origin >= len(plane.Pix) {
		return false
	}
	col := plane.Origin % plane.Stride
	rowEnd, ok := checkedAdd(col, plane.Width)
	if !ok || rowEnd > plane.Stride {
		return false
	}
	lastRow, ok := checkedMul(plane.Height-1, plane.Stride)
	if !ok {
		return false
	}
	visibleEnd, ok := checkedAdd(plane.Origin, lastRow)
	if !ok {
		return false
	}
	visibleEnd, ok = checkedAdd(visibleEnd, plane.Width)
	return ok && visibleEnd <= len(plane.Pix)
}

func loadSampleLine(dst []uint16, src []byte, bytesPerSample int) {
	switch bytesPerSample {
	case 1:
		for x := range dst {
			dst[x] = uint16(src[x])
		}
	case 2:
		for x := range dst {
			off := x * 2
			dst[x] = uint16(src[off]) | uint16(src[off+1])<<8
		}
	}
}

func storeSampleLine(dst []byte, src []uint16, bytesPerSample int) error {
	switch bytesPerSample {
	case 1:
		for x, sample := range src {
			if sample > 0xff {
				return ErrInvalidPlane
			}
			dst[x] = byte(sample)
		}
	case 2:
		for x, sample := range src {
			off := x * 2
			dst[off] = byte(sample)
			dst[off+1] = byte(sample >> 8)
		}
	}
	return nil
}
