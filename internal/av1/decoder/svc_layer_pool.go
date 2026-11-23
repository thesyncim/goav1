// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// FrameLayerPool is a thin adapter over *frame.LayerPool that satisfies
// FrameSurfaceProvider and FrameSurfaceReleaser. SVC and multi-resolution
// callers can use it directly with RunEventWithContextAndExternalReferences
// without writing per-stream provider/releaser glue: surface IDs are the
// global IDs minted by LayerPool.Acquire, and routing across sub-pools is
// handled by LayerPool itself.
//
// FrameLayerPool is a value type that holds only a pointer; the caller still
// owns the underlying LayerPool storage.
type FrameLayerPool struct {
	Pool *frame.LayerPool
}

// NewFrameLayerPool wraps a caller-owned *frame.LayerPool as a
// FrameLayerPool adapter.
func NewFrameLayerPool(pool *frame.LayerPool) FrameLayerPool {
	return FrameLayerPool{Pool: pool}
}

// FrameSurface implements FrameSurfaceProvider by routing globalID through
// the underlying *frame.LayerPool.
func (p FrameLayerPool) FrameSurface(globalID int) (*frame.Frame, error) {
	if p.Pool == nil {
		return nil, ErrInvalidSurfaceReference
	}
	fr, err := p.Pool.Frame(globalID)
	if err != nil {
		return nil, err
	}
	if fr == nil {
		return nil, ErrInvalidSurfaceReference
	}
	return fr, nil
}

// ReleaseFrameSurfaces implements FrameSurfaceReleaser by routing each
// globalID through the underlying *frame.LayerPool to its owning sub-pool.
func (p FrameLayerPool) ReleaseFrameSurfaces(ids []int) error {
	if p.Pool == nil {
		return ErrInvalidSurfaceReference
	}
	if len(ids) == 0 {
		return nil
	}
	return p.Pool.ReleaseMany(ids)
}

// LayerPoolGlobalSurfaceID returns the LayerPool-encoded global ID for a
// surface that the caller already acquired from a sub-pool. Together with the
// local index returned by AcquireLayerFrameSurface, this is what callers
// thread through SurfaceReferences and the globalSurface translator passed to
// RunEventWithContextAndExternalReferences. Returns -1 when sub is not bound
// to pool or local is out of range.
func LayerPoolGlobalSurfaceID(pool *frame.LayerPool, sub *frame.Pool, local int) int {
	if pool == nil {
		return -1
	}
	id, err := pool.EncodeGlobalID(sub, local)
	if err != nil {
		return -1
	}
	return id
}
