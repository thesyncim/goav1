package decoder

import "github.com/thesyncim/goav1/internal/av1/parser"

// SurfaceReferences maps AV1 reference slots to caller-owned frame-pool slots.
//
// The zero value is ready to use. Release lists are caller-provided so refresh
// and show-existing paths can return overwritten surfaces without allocating.
type SurfaceReferences struct {
	slots       [parser.RefFrames]int
	initialized bool
}

func (r *SurfaceReferences) Reset() {
	if r == nil {
		return
	}
	for i := 0; i < parser.RefFrames; i++ {
		r.slots[i] = -1
	}
	r.initialized = true
}

func (r *SurfaceReferences) ReferenceSlot(ref int) (int, bool) {
	if r == nil || ref < 0 || ref >= parser.RefFrames {
		return -1, false
	}
	r.ensureInitialized()
	slot := r.slots[ref]
	return slot, slot >= 0
}

func (r *SurfaceReferences) Holds(surface int) bool {
	if r == nil || surface < 0 {
		return false
	}
	r.ensureInitialized()
	return slotsHold(r.slots, surface)
}

// FinishFrame applies the reference refresh described by a complete decoded
// frame event.
func (r *SurfaceReferences) FinishFrame(event Event, surface int, releases []int) (int, error) {
	if (event.Kind != EventFrame && event.Kind != EventTileGroup) || !event.TileGroup.Final {
		return 0, ErrInvalidSurfaceEvent
	}
	return r.Refresh(event.FrameSize.RefreshFrameFlags, surface, releases)
}

// Refresh applies AV1 refresh_frame_flags for a decoded surface. It returns the
// unique previously referenced surfaces that are no longer held by any
// reference slot.
func (r *SurfaceReferences) Refresh(flags uint8, surface int, releases []int) (int, error) {
	if r == nil || surface < 0 {
		return 0, ErrInvalidSurfaceReference
	}
	r.ensureInitialized()

	next := r.slots
	var overwritten [parser.RefFrames]int
	overwrittenCount := 0
	for i := 0; i < parser.RefFrames; i++ {
		if (flags & (1 << uint(i))) == 0 {
			continue
		}
		old := next[i]
		if old >= 0 && old != surface && !containsSurface(overwritten[:overwrittenCount], old) {
			overwritten[overwrittenCount] = old
			overwrittenCount++
		}
		next[i] = surface
	}
	return r.apply(next, overwritten[:overwrittenCount], releases)
}

// ShowExistingFrame returns the referenced output surface for a
// show_existing_frame event and applies the key-frame reference reset when AV1
// requires it. The first return value is the surface to output.
func (r *SurfaceReferences) ShowExistingFrame(event Event, releases []int) (int, int, error) {
	if event.Kind != EventExistingFrame || !event.FrameHeader.ShowExistingFrame {
		return -1, 0, ErrInvalidSurfaceEvent
	}
	surface, ok := r.ReferenceSlot(int(event.FrameHeader.ExistingFrameIdx))
	if !ok {
		return -1, 0, ErrInvalidSurfaceReference
	}
	count, err := r.ShowExisting(surface, event.ExistingFrame.FrameType, releases)
	if err != nil {
		return -1, 0, err
	}
	return surface, count, nil
}

// ShowExisting applies the AV1 key-frame show-existing reset to the surface
// reference map. Non-key show-existing frames do not alter reference slots.
func (r *SurfaceReferences) ShowExisting(surface int, frameType parser.FrameType, releases []int) (int, error) {
	if r == nil || surface < 0 {
		return 0, ErrInvalidSurfaceReference
	}
	if frameType != parser.FrameTypeKey {
		return 0, nil
	}
	r.ensureInitialized()

	next := r.slots
	var overwritten [parser.RefFrames]int
	overwrittenCount := 0
	for i := 0; i < parser.RefFrames; i++ {
		old := next[i]
		if old >= 0 && old != surface && !containsSurface(overwritten[:overwrittenCount], old) {
			overwritten[overwrittenCount] = old
			overwrittenCount++
		}
		next[i] = surface
	}
	return r.apply(next, overwritten[:overwrittenCount], releases)
}

func (r *SurfaceReferences) ensureInitialized() {
	if !r.initialized {
		r.Reset()
	}
}

func (r *SurfaceReferences) apply(next [parser.RefFrames]int, overwritten []int, releases []int) (int, error) {
	releaseCount := 0
	for i := 0; i < len(overwritten); i++ {
		surface := overwritten[i]
		if slotsHold(next, surface) {
			continue
		}
		if releaseCount >= len(releases) {
			return 0, ErrSurfaceReleaseBufferTooSmall
		}
		releases[releaseCount] = surface
		releaseCount++
	}
	r.slots = next
	return releaseCount, nil
}

func slotsHold(slots [parser.RefFrames]int, surface int) bool {
	for i := 0; i < parser.RefFrames; i++ {
		if slots[i] == surface {
			return true
		}
	}
	return false
}

func containsSurface(slots []int, surface int) bool {
	for i := 0; i < len(slots); i++ {
		if slots[i] == surface {
			return true
		}
	}
	return false
}
