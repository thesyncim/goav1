package decoder

import "github.com/thesyncim/goav1/internal/av1/frame"

// FinishFrameSurface applies a final decoded frame event to refs and releases
// overwritten frame-pool slots. Reference state is published only after the
// pool accepts the release batch.
func FinishFrameSurface(refs *SurfaceReferences, pool *frame.Pool, event Event, surface int, releases []int) (int, error) {
	if refs == nil {
		return 0, ErrInvalidSurfaceReference
	}

	next := *refs
	count, err := next.FinishFrame(event, surface, releases)
	if err != nil {
		return 0, err
	}
	if count != 0 {
		if pool == nil {
			return 0, frame.ErrInvalidPool
		}
		if err := pool.ReleaseMany(releases[:count]); err != nil {
			return 0, err
		}
	}

	*refs = next
	return count, nil
}

// ShowExistingFrameSurface resolves a show-existing-frame event, applies any
// key-frame reference reset, and releases overwritten frame-pool slots.
// Reference state is published only after the pool accepts the release batch.
func ShowExistingFrameSurface(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (int, int, error) {
	if refs == nil {
		return -1, 0, ErrInvalidSurfaceReference
	}

	next := *refs
	surface, count, err := next.ShowExistingFrame(event, releases)
	if err != nil {
		return -1, 0, err
	}
	if count != 0 {
		if pool == nil {
			return -1, 0, frame.ErrInvalidPool
		}
		if err := pool.ReleaseMany(releases[:count]); err != nil {
			return -1, 0, err
		}
	}

	*refs = next
	return surface, count, nil
}
