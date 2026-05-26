package goav1

import internalframe "github.com/thesyncim/goav1/internal/av1/frame"

// FrameFormat describes the geometry, bit depth, chroma subsampling, and
// alignment of a decoded AV1 frame buffer.
type FrameFormat = internalframe.Format

// FrameLayout reports the per-plane and total byte sizes for a FrameFormat,
// as computed by FrameRequiredSize.
type FrameLayout = internalframe.Layout

// FramePlane identifies one plane (Y, U, or V) of a Frame.
type FramePlane = internalframe.Plane

// Frame is a decoded AV1 picture stored in a caller-owned backing buffer.
// Instances are produced by BindFrame and acquired from a FramePool.
type Frame = internalframe.Frame

// FramePool is a fixed-size pool of preallocated Frames backed by a single
// caller-owned byte slice. Acquire/release operations are allocation-free.
type FramePool = internalframe.Pool

// FrameLayerPool aggregates several caller-owned FramePool instances keyed
// by FrameFormat. It is the production replacement for the per-spatial-layer
// pool glue that SVC and multi-resolution callers previously hand-rolled:
// layer-0 (e.g. 640x360) and layer-1 (e.g. 1280x720) surfaces coexist behind
// one set of global surface IDs that route through the right sub-pool.
type FrameLayerPool = internalframe.LayerPool

// FrameLayerFactory binds a new caller-owned sub-pool on first observation
// of a given FrameFormat. Implementations supply the backing storage for the
// new sub-pool and return BindFramePool's Pool value.
type FrameLayerFactory = internalframe.LayerFactory

// FrameLayerFactoryFunc adapts a plain function to FrameLayerFactory.
type FrameLayerFactoryFunc = internalframe.LayerFactoryFunc

// FrameSamplePlane is a writable view of one plane stored at uint16
// resolution. It is the buffer shape used by DSP and prediction helpers.
type FrameSamplePlane = internalframe.SamplePlane

// FrameBorderedSamplePlane is a FrameSamplePlane augmented with mirrored
// border samples on every side, sized for AV1 deblocking, CDEF, and loop
// restoration.
type FrameBorderedSamplePlane = internalframe.BorderedSamplePlane

// FrameBorderedSamplePlaneLayout reports the geometry of a
// FrameBorderedSamplePlane: stride, border width/height, and total length.
type FrameBorderedSamplePlaneLayout = internalframe.BorderedSamplePlaneLayout

// Frame buffer binding errors. Each Err* value is returned by the frame and
// frame-pool helpers when the input buffer is too small, the format is
// invalid, the pool index is out of range, or the requested plane does not
// exist.
var (
	ErrFrameInvalidFormat       = internalframe.ErrInvalidFormat
	ErrFrameInvalidPool         = internalframe.ErrInvalidPool
	ErrFramePoolEmpty           = internalframe.ErrPoolEmpty
	ErrFrameInvalidSlot         = internalframe.ErrInvalidSlot
	ErrFrameInvalidPlane        = internalframe.ErrInvalidPlane
	ErrFrameShortBuffer         = internalframe.ErrShortBuffer
	ErrFrameInvalidLayerPool    = internalframe.ErrInvalidLayerPool
	ErrFrameLayerPoolFull       = internalframe.ErrLayerPoolFull
	ErrFrameInvalidLayerFactory = internalframe.ErrInvalidLayerFactory
)

// FrameRequiredSize returns the per-plane and total byte requirements for a
// single frame matching format.
func FrameRequiredSize(format FrameFormat) (FrameLayout, error) {
	return internalframe.RequiredSize(format)
}

// FramePoolRequiredSize returns the per-frame layout and the total bytes
// required to back a pool of count frames matching format. It validates that
// count is positive and that the resulting allocation does not overflow int.
func FramePoolRequiredSize(format FrameFormat, count int) (FrameLayout, int, error) {
	if count <= 0 {
		return FrameLayout{}, 0, ErrFrameInvalidPool
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		return FrameLayout{}, 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if layout.Size != 0 && count > maxInt/layout.Size {
		return FrameLayout{}, 0, ErrFrameInvalidFormat
	}
	return layout, layout.Size * count, nil
}

// FrameFormatFromHeaders is an alias for FrameCodedFormatFromHeaders kept for
// callers that target the coded-pixel surface and do not need to distinguish
// the coded and upscaled output formats.
func FrameFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return FrameCodedFormatFromHeaders(sequence, size, align)
}

// FrameCodedFormatFromHeaders returns the FrameFormat sized to the coded
// (pre-super-res) frame width and the frame height taken from size, using the
// bit depth and chroma layout from sequence. align is propagated into the
// FrameFormat for downstream buffer sizing.
func FrameCodedFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.CodedWidth, size.Height, align)
}

// FrameOutputFormatFromHeaders returns the FrameFormat sized to the upscaled
// (post-super-res) frame width and the frame height. It is the format
// applied to the final output surface presented to the caller.
func FrameOutputFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.UpscaledWidth, size.Height, align)
}

func frameFormatFromHeaderWidth(sequence SequenceHeader, width uint32, height uint32, align int) (FrameFormat, error) {
	if width == 0 || height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(width) > maxInt || uint64(height) > maxInt {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	return FrameFormat{
		Width:        int(width),
		Height:       int(height),
		BitDepth:     sequence.ColorConfig.BitDepth,
		MonoChrome:   sequence.ColorConfig.MonoChrome,
		SubsamplingX: sequence.ColorConfig.SubsamplingX,
		SubsamplingY: sequence.ColorConfig.SubsamplingY,
		Align:        align,
	}, nil
}

// BindFrame interprets buffer as a Frame matching format. The returned Frame
// aliases buffer; the caller retains ownership and must keep it alive.
// buffer must be at least FrameRequiredSize(format).Size bytes long.
func BindFrame(buffer []byte, format FrameFormat) (Frame, error) {
	return internalframe.Bind(buffer, format)
}

// BindFramePool wires backing into a FramePool of len(frames) Frames matching
// format. The caller supplies the backing storage, the Frame headers, and the
// free/used bookkeeping slices. All slices must be sized via
// FramePoolRequiredSize and have len(frames) == len(free) == len(used).
func BindFramePool(backing []byte, format FrameFormat, frames []Frame, free []int, used []bool) (FramePool, error) {
	return internalframe.BindPool(backing, format, frames, free, used)
}

// BindFrameLayerPool wires caller-owned storage for up to len(pools) sub-pools
// keyed by FrameFormat. stride bounds the maximum local pool index that the
// global surface ID encoding can carry (256 is a safe default for pools sized
// up to 256 surfaces). factory binds a new caller-owned sub-pool the first
// time a previously-unseen FrameFormat is observed.
//
// After binding, FrameLayerPool.Acquire and Release are zero-alloc for any
// FrameFormat that has already been observed; the first Acquire of a new
// FrameFormat invokes factory exactly once. The same-length rule
// (len(pools) == len(formats) == len(bound)) lets every sub-pool slot be
// addressed by index without re-allocating.
func BindFrameLayerPool(pools []FramePool, formats []FrameFormat, bound []bool, stride int, factory FrameLayerFactory) (FrameLayerPool, error) {
	return internalframe.BindLayerPool(pools, formats, bound, stride, factory)
}

// FrameSamplePlaneLen reports the number of uint16 entries required to back a
// FrameSamplePlane view of plane at bytesPerSample (1 or 2).
func FrameSamplePlaneLen(plane FramePlane, bytesPerSample int) (int, error) {
	return internalframe.SamplePlaneLen(plane, bytesPerSample)
}

// FrameBorderedSamplePlaneLen reports the layout (stride, length, border
// sizes) of a FrameBorderedSamplePlane view of plane with the supplied
// border and alignment.
func FrameBorderedSamplePlaneLen(plane FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlaneLayout, error) {
	return internalframe.BorderedSamplePlaneLen(plane, bytesPerSample, borderHorz, borderVert, align)
}

// BindFrameBorderedSamplePlane wires dst into a FrameBorderedSamplePlane view
// with the requested borders and alignment. dst must be sized via
// FrameBorderedSamplePlaneLen and is caller-owned.
func BindFrameBorderedSamplePlane(dst []uint16, plane FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlane, error) {
	return internalframe.BindBorderedSamplePlane(dst, plane, bytesPerSample, borderHorz, borderVert, align)
}

// LoadFrameSamplePlane copies src into dst as uint16 samples and returns the
// resulting FrameSamplePlane view. dst must be sized via FrameSamplePlaneLen.
func LoadFrameSamplePlane(dst []uint16, src FramePlane, bytesPerSample int) (FrameSamplePlane, error) {
	return internalframe.LoadSamplePlane(dst, src, bytesPerSample)
}

// LoadFrameBorderedSamplePlane copies src into dst as uint16 samples, applies
// the requested borders, and returns the resulting FrameBorderedSamplePlane
// view. dst must be sized via FrameBorderedSamplePlaneLen.
func LoadFrameBorderedSamplePlane(dst []uint16, src FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlane, error) {
	return internalframe.LoadBorderedSamplePlane(dst, src, bytesPerSample, borderHorz, borderVert, align)
}

// StoreFrameSamplePlane copies the uint16 samples in src back into dst at the
// requested bytesPerSample, clipping each sample to the plane bit depth.
func StoreFrameSamplePlane(dst FramePlane, bytesPerSample int, src FrameSamplePlane) error {
	return internalframe.StoreSamplePlane(dst, bytesPerSample, src)
}

// StoreFrameBorderedSamplePlane copies the central (non-border) region of src
// back into dst at the requested bytesPerSample, clipping to plane bit depth.
func StoreFrameBorderedSamplePlane(dst FramePlane, bytesPerSample int, src FrameBorderedSamplePlane) error {
	return internalframe.StoreBorderedSamplePlane(dst, bytesPerSample, src)
}
