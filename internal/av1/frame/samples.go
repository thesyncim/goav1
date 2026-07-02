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
	loadSampleRows(samples, strideSamples, src.Pix, src.Stride, src.Width, src.Height, bytesPerSample)
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
	loadSampleRows(samples, strideSamples, src.Pix, src.Stride, loadedWidth, src.Height, bytesPerSample)
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
	loadSampleRows(samples[layout.Origin:], layout.Stride, src.Pix, src.Stride, src.Width, src.Height, bytesPerSample)
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
	if !storeSampleRows(dst.Pix, dst.Stride, src.Pix, src.Stride, src.Width, src.Height, bytesPerSample) {
		return ErrInvalidPlane
	}
	return nil
}

// StoreSamplePlaneTrusted writes visible samples from src like StoreSamplePlane,
// but assumes every sample already fits the destination bit depth. It still
// validates plane geometry and buffer sizes. Internal postfilter callers use
// this after kernels that already clip their output.
func StoreSamplePlaneTrusted(dst Plane, bytesPerSample int, src SamplePlane) error {
	if _, _, err := samplePlaneLayout(dst, bytesPerSample, true); err != nil {
		return err
	}
	if !samplePlaneFits(src) || src.Width != dst.Width || src.Height != dst.Height {
		return ErrInvalidPlane
	}
	storeSampleRowsTrusted(dst.Pix, dst.Stride, src.Pix, src.Stride, src.Width, src.Height, bytesPerSample)
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
	if !storeSampleRows(dst.Pix, dst.Stride, src.Pix[src.Origin:], src.Stride, src.Width, src.Height, bytesPerSample) {
		return ErrInvalidPlane
	}
	return nil
}

// StoreBorderedSamplePlaneTrusted writes the visible region from src like
// StoreBorderedSamplePlane, but assumes every sample already fits the
// destination bit depth. It still validates plane geometry and buffer sizes.
func StoreBorderedSamplePlaneTrusted(dst Plane, bytesPerSample int, src BorderedSamplePlane) error {
	if _, _, err := samplePlaneLayout(dst, bytesPerSample, true); err != nil {
		return err
	}
	if !borderedSamplePlaneFits(src) || src.Width != dst.Width || src.Height != dst.Height {
		return ErrInvalidPlane
	}
	storeSampleRowsTrusted(dst.Pix, dst.Stride, src.Pix[src.Origin:], src.Stride, src.Width, src.Height, bytesPerSample)
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

func loadSampleRows(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int, bytesPerSample int) {
	switch bytesPerSample {
	case 1:
		loadSampleRows8(dst, dstStride, src, srcStride, width, height)
	case 2:
		srcOff := 0
		dstOff := 0
		rowBytes := width * 2
		for y := 0; y < height; y++ {
			srcLine := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
			dstLine := dst[dstOff : dstOff+width : dstOff+width]
			for x := 0; x < width; x++ {
				off := x * 2
				dstLine[x] = uint16(srcLine[off]) | uint16(srcLine[off+1])<<8
			}
			srcOff += srcStride
			dstOff += dstStride
		}
	}
}

// loadSampleRows8PureGo is the bit-exact reference for the 8-bit sample
// staging widen (u8 -> u16). loadSampleRows8 resolves to it on targets
// without a tuned variant (samples_widen_generic.go); tuned variants must
// write exactly dst[y*dstStride+x] = uint16(src[y*srcStride+x]) for x in
// [0, width), y in [0, height) and leave every other dst element untouched.
//
// Unlike most kernels this one dispatches by build tag rather than through a
// package-level func variable: an indirect call would make the compiler leak
// dst/src at every Load*SamplePlane entry point and break the decoder's
// zero-alloc steady-state budget. NEON is architecturally mandatory on arm64
// (cpu_arm64.go sets Detected.NEON unconditionally), so the static binding
// loses no runtime detection.
func loadSampleRows8PureGo(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	srcOff := 0
	dstOff := 0
	for y := 0; y < height; y++ {
		srcLine := src[srcOff : srcOff+width : srcOff+width]
		dstLine := dst[dstOff : dstOff+width : dstOff+width]
		x := 0
		for ; x+8 <= width; x += 8 {
			dstLine[x+0] = uint16(srcLine[x+0])
			dstLine[x+1] = uint16(srcLine[x+1])
			dstLine[x+2] = uint16(srcLine[x+2])
			dstLine[x+3] = uint16(srcLine[x+3])
			dstLine[x+4] = uint16(srcLine[x+4])
			dstLine[x+5] = uint16(srcLine[x+5])
			dstLine[x+6] = uint16(srcLine[x+6])
			dstLine[x+7] = uint16(srcLine[x+7])
		}
		for ; x < width; x++ {
			dstLine[x] = uint16(srcLine[x])
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

func storeSampleRows(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int, bytesPerSample int) bool {
	switch bytesPerSample {
	case 1:
		dstOff := 0
		srcOff := 0
		for y := 0; y < height; y++ {
			dstLine := dst[dstOff : dstOff+width : dstOff+width]
			srcLine := src[srcOff : srcOff+width : srcOff+width]
			x := 0
			for ; x+8 <= width; x += 8 {
				s0 := srcLine[x+0]
				s1 := srcLine[x+1]
				s2 := srcLine[x+2]
				s3 := srcLine[x+3]
				s4 := srcLine[x+4]
				s5 := srcLine[x+5]
				s6 := srcLine[x+6]
				s7 := srcLine[x+7]
				if s0|s1|s2|s3|s4|s5|s6|s7 > 0xff {
					return false
				}
				dstLine[x+0] = byte(s0)
				dstLine[x+1] = byte(s1)
				dstLine[x+2] = byte(s2)
				dstLine[x+3] = byte(s3)
				dstLine[x+4] = byte(s4)
				dstLine[x+5] = byte(s5)
				dstLine[x+6] = byte(s6)
				dstLine[x+7] = byte(s7)
			}
			for ; x < width; x++ {
				sample := srcLine[x]
				if sample > 0xff {
					return false
				}
				dstLine[x] = byte(sample)
			}
			dstOff += dstStride
			srcOff += srcStride
		}
	case 2:
		dstOff := 0
		srcOff := 0
		rowBytes := width * 2
		for y := 0; y < height; y++ {
			dstLine := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
			srcLine := src[srcOff : srcOff+width : srcOff+width]
			for x, sample := range srcLine {
				off := x * 2
				dstLine[off] = byte(sample)
				dstLine[off+1] = byte(sample >> 8)
			}
			dstOff += dstStride
			srcOff += srcStride
		}
	}
	return true
}

func storeSampleRowsTrusted(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int, bytesPerSample int) {
	switch bytesPerSample {
	case 1:
		dstOff := 0
		srcOff := 0
		for y := 0; y < height; y++ {
			dstLine := dst[dstOff : dstOff+width : dstOff+width]
			srcLine := src[srcOff : srcOff+width : srcOff+width]
			x := 0
			for ; x+8 <= width; x += 8 {
				dstLine[x+0] = byte(srcLine[x+0])
				dstLine[x+1] = byte(srcLine[x+1])
				dstLine[x+2] = byte(srcLine[x+2])
				dstLine[x+3] = byte(srcLine[x+3])
				dstLine[x+4] = byte(srcLine[x+4])
				dstLine[x+5] = byte(srcLine[x+5])
				dstLine[x+6] = byte(srcLine[x+6])
				dstLine[x+7] = byte(srcLine[x+7])
			}
			for ; x < width; x++ {
				dstLine[x] = byte(srcLine[x])
			}
			dstOff += dstStride
			srcOff += srcStride
		}
	case 2:
		dstOff := 0
		srcOff := 0
		rowBytes := width * 2
		for y := 0; y < height; y++ {
			dstLine := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
			srcLine := src[srcOff : srcOff+width : srcOff+width]
			for x, sample := range srcLine {
				off := x * 2
				dstLine[off] = byte(sample)
				dstLine[off+1] = byte(sample >> 8)
			}
			dstOff += dstStride
			srcOff += srcStride
		}
	}
}
