package goav1

import internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"

// DecoderFrameLayerPool adapts a root FrameLayerPool to the decoder's
// external-reference provider and releaser interfaces.
type DecoderFrameLayerPool = internaldecoder.FrameLayerPool

// DecoderTemporalMotionReferenceProvider resolves caller-owned global surface
// IDs to temporal motion-vector reference metadata for custom SVC batch runners.
type DecoderTemporalMotionReferenceProvider = internaldecoder.TemporalMotionReferenceProvider

// DecoderTemporalMotionReferenceProviderFunc adapts a function to
// DecoderTemporalMotionReferenceProvider.
type DecoderTemporalMotionReferenceProviderFunc = internaldecoder.TemporalMotionReferenceProviderFunc

// NewDecoderFrameLayerPool wraps pool as a DecoderFrameSurfaceProvider and
// DecoderFrameSurfaceReleaser. The caller still owns pool and its backing
// storage.
func NewDecoderFrameLayerPool(pool *FrameLayerPool) DecoderFrameLayerPool {
	return internaldecoder.NewFrameLayerPool(pool)
}

// DecoderAcquireLayerFrameSurface acquires a frame surface from the sub-pool
// matching the coded format implied by sequence and size. It returns the owning
// sub-pool, local surface index, LayerPool global surface ID, and frame pointer.
func DecoderAcquireLayerFrameSurface(pool *FrameLayerPool, sequence SequenceHeader, size FrameSize, align int) (*FramePool, int, int, *Frame, error) {
	return internaldecoder.AcquireLayerFrameSurface(pool, sequence, size, align)
}

// DecoderLayerPoolGlobalSurfaceID returns the LayerPool-encoded global ID for a
// local surface in sub. It returns -1 when sub is not bound to pool or local is
// invalid.
func DecoderLayerPoolGlobalSurfaceID(pool *FrameLayerPool, sub *FramePool, local int) int {
	return internaldecoder.LayerPoolGlobalSurfaceID(pool, sub, local)
}

// ResolveDecoderFrameReferencesWithProvider resolves caller-owned global
// surface IDs through provider instead of a single FramePool.
func ResolveDecoderFrameReferencesWithProvider(provider DecoderFrameSurfaceProvider, surfaces []int, frames []*Frame) (int, error) {
	return internaldecoder.ResolveFrameReferencesWithProvider(provider, surfaces, frames)
}

// PublishDecoderTemporalMotionReference stores the decoded frame's temporal
// motion-vector reference metadata under surface in store.
func PublishDecoderTemporalMotionReference(event DecoderEvent, surface int, current *TileReferenceMVFrame, store []TileTemporalMotionReferenceFrame) error {
	return internaldecoder.PublishTemporalMotionReference(event, surface, current, store)
}

// ResolveDecoderTemporalMotionReferences resolves local surface indices to
// temporal motion-vector reference metadata from store.
func ResolveDecoderTemporalMotionReferences(surfaces []int, store []TileTemporalMotionReferenceFrame, dst []TileTemporalMotionReferenceFrame) (int, error) {
	return internaldecoder.ResolveTemporalMotionReferences(surfaces, store, dst)
}

// ResolveDecoderTemporalMotionReferencesWithProvider resolves caller-owned
// global surface IDs through provider instead of a single temporal-motion store.
func ResolveDecoderTemporalMotionReferencesWithProvider(provider DecoderTemporalMotionReferenceProvider, surfaces []int, dst []TileTemporalMotionReferenceFrame) (int, error) {
	return internaldecoder.ResolveTemporalMotionReferencesWithProvider(provider, surfaces, dst)
}
