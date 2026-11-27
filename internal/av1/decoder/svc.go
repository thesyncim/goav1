// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameSurfaceProvider routes an opaque, caller-owned surface identifier to
// the *frame.Frame that holds the decoded pixels.
//
// SVC streams (and any caller that retains decoded surfaces in multiple
// frame.Pool instances, one per spatial layer) cannot use the single-pool
// resolution path baked into RunEventWithContext... because reference slots
// span pools. A provider implementation maps an opaque surface identifier —
// chosen by the caller and stored in SurfaceReferences.slots — to the
// *frame.Frame that actually holds the referenced pixels.
type FrameSurfaceProvider interface {
	FrameSurface(id int) (*frame.Frame, error)
}

// ResolveFrameReferencesWithProvider converts caller-owned surface IDs into
// checked frame pointers using a FrameSurfaceProvider rather than a single
// frame.Pool. dst is published only after every referenced surface has
// validated, matching the atomicity guarantee of ResolveFrameReferences.
func ResolveFrameReferencesWithProvider(provider FrameSurfaceProvider, surfaces []int, dst []*frame.Frame) (int, error) {
	count := len(surfaces)
	if count > parser.InterRefsPerFrame {
		return 0, ErrInvalidSurfaceReference
	}
	if len(dst) < count {
		return 0, ErrSurfaceReferenceBufferTooSmall
	}
	if provider == nil {
		return 0, ErrInvalidSurfaceReference
	}

	var resolved [parser.InterRefsPerFrame]*frame.Frame
	for i := range count {
		if surfaces[i] < 0 {
			return 0, ErrInvalidSurfaceReference
		}
		ref, err := provider.FrameSurface(surfaces[i])
		if err != nil {
			return 0, err
		}
		if ref == nil {
			return 0, ErrInvalidSurfaceReference
		}
		resolved[i] = ref
	}
	for i := range count {
		dst[i] = resolved[i]
	}
	return count, nil
}

// FrameSurfaceReleaser is the caller-owned hook RunEventWithContextAndExternalReferences
// uses to release overwritten reference surfaces at frame completion. The
// release IDs use the caller's own surface namespace, matching what is stored
// in SurfaceReferences.slots and resolved through FrameSurfaceProvider.
type FrameSurfaceReleaser interface {
	ReleaseFrameSurfaces(ids []int) error
}

// SharedFrameContextStore holds saved entropy (CDF) frame contexts keyed by
// the stream-global surface ID, mirroring libaom storing frame_context inside
// the shared RefCntBuffer pool (av1/decoder/decodeframe.c:
// cm->cur_frame->frame_context = *cm->fc at frame end). SVC enhancement-layer
// frames with primary_ref_frame != PRIMARY_REF_NONE inherit their frame
// context from the referenced buffer
// (get_primary_ref_frame_buf(cm)->frame_context), which may belong to a
// different spatial layer. A per-layer store keyed by local reference slot
// cannot reach a lower layer's saved context; keying by the shared global
// surface ID does.
//
// The store must be shared across all spatial layers that participate in the
// same AV1 stream, exactly like the shared SurfaceReferences, because both the
// reference slots and the per-buffer frame contexts are stream-global per the
// AV1 spec.
//
// The zero value is ready to use. The store is caller-owned and not safe for
// concurrent use across frames; AV1 decoding is serial across the frame-begin,
// tile, and frame-finish events that touch it.
type SharedFrameContextStore struct {
	contexts map[int]int
	backing  []threading.FrameWorkTileResidualCDFStorage
	used     int

	// motion holds each reference buffer's saved motion vectors under the same
	// global surface ID (libaom's cm->cur_frame->mvs, dav1d's f->mvs_ref), so
	// a frame with use_ref_frame_mvs projects from the buffers its reference
	// slots name, whichever spatial layer decoded them.
	motion map[int]*sharedMotionFrame
}

// sharedMotionFrame is one reference buffer's MV_REF grid and the order-hint
// metadata temporal projection needs. frame.Entries aliases backing.
type sharedMotionFrame struct {
	frame   tile.ReferenceMVFrame
	backing []tile.ReferenceMVEntry
	meta    mvFrameMeta
}

// Prealloc primes the retained frame-context map so realtime shared-reference
// decoders can keep Reset and subsequent store operations allocation-free.
func (s *SharedFrameContextStore) Prealloc(capacity int) {
	if s == nil || capacity <= 0 {
		return
	}
	if s.contexts == nil {
		s.contexts = make(map[int]int, capacity)
	}
	if len(s.contexts) == 0 && len(s.backing) < capacity {
		if cap(s.backing) >= capacity {
			s.backing = s.backing[:capacity]
		} else {
			s.backing = make([]threading.FrameWorkTileResidualCDFStorage, capacity)
		}
	}
}

// Reset clears every saved frame context. Callers invoke this at a new coded
// video sequence boundary, matching ResetFrameSurfaceReferences clearing the
// stream-global reference slots. Existing map capacity is retained for hot
// decode paths that reuse a decoder across streams.
func (s *SharedFrameContextStore) Reset() {
	if s == nil {
		return
	}
	clear(s.contexts)
	s.used = 0
	for _, m := range s.motion {
		m.frame = tile.ReferenceMVFrame{}
		m.meta = mvFrameMeta{}
	}
}

// motionFrame returns the saved motion vectors of the buffer with the global
// surface ID, or nil when that buffer never published any.
func (s *SharedFrameContextStore) motionFrame(globalID int) *sharedMotionFrame {
	if s == nil || globalID < 0 {
		return nil
	}
	if m := s.motion[globalID]; m != nil && m.meta.valid {
		return m
	}
	return nil
}

// publishMotion saves a decoded frame's motion vectors under its global
// surface ID. The store takes the grid's backing and returns the buffer's
// previous one for the caller's next frame, so no entries are copied.
func (s *SharedFrameContextStore) publishMotion(globalID int, current tile.ReferenceMVFrame, backing []tile.ReferenceMVEntry, meta mvFrameMeta) []tile.ReferenceMVEntry {
	if s == nil || globalID < 0 {
		return backing
	}
	m := s.motion[globalID]
	if m == nil {
		if s.motion == nil {
			s.motion = make(map[int]*sharedMotionFrame, parser.RefFrames)
		}
		m = new(sharedMotionFrame)
		s.motion[globalID] = m
	}
	previous := m.backing
	m.frame, m.backing, m.meta = current, backing, meta
	return previous
}

// load returns the saved frame context for the global surface ID, reporting
// whether one is present.
func (s *SharedFrameContextStore) load(globalID int) (threading.FrameWorkTileResidualCDFStorage, bool) {
	if s == nil || globalID < 0 || s.contexts == nil {
		return threading.FrameWorkTileResidualCDFStorage{}, false
	}
	index, ok := s.contexts[globalID]
	if !ok || index < 0 || index >= s.used || index >= len(s.backing) {
		return threading.FrameWorkTileResidualCDFStorage{}, false
	}
	return s.backing[index], true
}

// store saves the adapted frame context under the global surface ID, mirroring
// libaom's cm->cur_frame->frame_context = *cm->fc at frame end.
func (s *SharedFrameContextStore) store(globalID int, ctx threading.FrameWorkTileResidualCDFStorage) {
	if s == nil || globalID < 0 {
		return
	}
	if s.contexts == nil {
		s.Prealloc(parser.RefFrames)
	}
	if index, ok := s.contexts[globalID]; ok {
		if index >= 0 && index < len(s.backing) {
			s.backing[index] = ctx
		}
		return
	}
	if s.used >= len(s.backing) {
		nextLen := len(s.backing) * 2
		if nextLen < parser.RefFrames {
			nextLen = parser.RefFrames
		}
		if nextLen <= s.used {
			nextLen = s.used + 1
		}
		next := make([]threading.FrameWorkTileResidualCDFStorage, nextLen)
		copy(next, s.backing)
		s.backing = next
	}
	index := s.used
	s.used++
	s.backing[index] = ctx
	s.contexts[globalID] = index
}

func frameWorkIdentityGlobalSurface(local int) int { return local }

// RunEventWithContextAndExternalReferences is the SVC-friendly counterpart of
// RunEventWithContextAndSideDataAndPostFilterRunners. The caller is
// responsible for retaining decoded surfaces across spatial-layer pools and
// supplies provider, which maps the surface IDs stored in refs.slots to the
// *frame.Frame that actually holds the referenced pixels.
//
// The current frame's output surface is still acquired from framePool, which
// must hold the current frame's format. After acquisition the caller-supplied
// globalSurface translator converts the local pool index to the caller's
// global surface ID, and that global ID is what gets refreshed into
// refs.slots at frame completion. releaser is invoked with the overwritten
// surface IDs (in the caller's namespace) so cross-pool releases can be
// dispatched to the right pool.
//
// refs must be shared across all spatial layers that participate in the same
// AV1 stream, because reference slots are stream-global per the AV1 spec.
func (s *FrameWorkState) RunEventWithContextAndExternalReferences(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event *Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, provider FrameSurfaceProvider, globalSurface func(local int) int, releaser FrameSurfaceReleaser, frameContexts *SharedFrameContextStore, side FrameWorkSideDataRunner, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkEventResult, error) {
	return s.RunEventWithContextAndExternalReferencesWithPostPublisher(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, provider, globalSurface, releaser, frameContexts, side, runner, post, nil)
}

func (s *FrameWorkState) RunEventWithContextAndExternalReferencesWithPostPublisher(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event *Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, provider FrameSurfaceProvider, globalSurface func(local int) int, releaser FrameSurfaceReleaser, frameContexts *SharedFrameContextStore, side FrameWorkSideDataRunner, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner, postPublisher FrameWorkPostFilterGlobalSurfacePublisher) (FrameWorkEventResult, error) {
	if provider == nil {
		return FrameWorkEventResult{}, ErrInvalidSurfaceReference
	}
	if globalSurface == nil {
		// Identity: the caller is not using a global namespace.
		globalSurface = frameWorkIdentityGlobalSurface
	}
	// Route entropy (CDF) frame-context save/restore through the shared,
	// global-surface-keyed store for the duration of this event so that the
	// frame-context inheritance (primary_ref_frame) and save (refresh) read
	// and write the stream-global RefCntBuffer-equivalent slot rather than this
	// layer's local reference array. Cleared on return so a panic or error path
	// cannot leak per-event sharing config across frames.
	s.sharedFrameContexts = frameContexts
	s.sharedFrameContextRefs = refs
	s.sharedFrameContextGlobalSurface = globalSurface
	defer s.clearSharedFrameContext()

	// A new coded video sequence drops every reference. Here the reference
	// slots hold caller-namespace surface IDs, so release them through
	// releaser, as finishExternal does, before PlanEvent's reset would hand
	// them to framePool as local indices.
	if event != nil && event.NewCodedVideoSequence && releaser != nil {
		if err := releaseExternalReferences(refs, releases, releaser); err != nil {
			return FrameWorkEventResult{}, err
		}
	}

	step, output, referenceCount, references, err := s.planEventWithExternalReferenceContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, provider)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	if err := s.bindFrameWorkEventSideData(event, step, output, references, event.Unit.Payload, side); err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.runStepExternalRefresh(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, runner, post, postPublisher, globalSurface, releaser)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	return FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

// releaseExternalReferences clears every reference slot and hands the
// surfaces they held to releaser. refs is updated only after the release
// succeeds.
func releaseExternalReferences(refs *SurfaceReferences, releases []int, releaser FrameSurfaceReleaser) error {
	if refs == nil {
		return ErrInvalidSurfaceReference
	}
	next := *refs
	count, err := next.ReleaseAll(releases)
	if err != nil {
		return err
	}
	if count != 0 {
		if err := releaser.ReleaseFrameSurfaces(releases[:count]); err != nil {
			return err
		}
	}
	*refs = next
	return nil
}

func (s *FrameWorkState) clearSharedFrameContext() {
	if s == nil {
		return
	}
	s.sharedFrameContexts = nil
	s.sharedFrameContextRefs = nil
	s.sharedFrameContextGlobalSurface = nil
}

func (s *FrameWorkState) planEventWithExternalReferenceContext(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event *Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, provider FrameSurfaceProvider) (FrameWorkStep, *frame.Frame, uint8, []*frame.Frame, error) {
	step, output, err := s.PlanEvent(refs, framePool, sequence, event, align, referenceSurfaces, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkStep{}, nil, 0, nil, err
	}

	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStep{}, nil, 0, nil, err
	}
	if step.Kind == FrameWorkStepShowExisting {
		// Show-existing surfaces are resolved through the provider because
		// the referenced surface may live in a different layer's pool.
		output, err = provider.FrameSurface(step.ShowExisting.Surface)
		if err != nil {
			return FrameWorkStep{}, nil, 0, nil, err
		}
	}

	var references []*frame.Frame
	if hasTile && plan.JobCount != 0 {
		if output == nil {
			surface, err := frameWorkStepSurface(step)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
			// Output is always acquired from the current frame's pool: the
			// surface index returned by frameWorkStepSurface is the local
			// pool index that BeginFrameSurface acquired and PlanEvent
			// threaded through.
			output, err = framePool.Frame(surface)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
		}
		if referenceCount != 0 {
			referenceCountN := int(referenceCount)
			if step.Kind == FrameWorkStepTile {
				count, err := refs.FrameReferences(event, referenceSurfaces)
				if err != nil {
					return FrameWorkStep{}, nil, 0, nil, err
				}
				if count != referenceCountN {
					return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
				}
			}
			count, err := ResolveFrameReferencesWithProvider(provider, referenceSurfaces[:referenceCountN], referenceFrames)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
			if count != referenceCountN {
				return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
			}
			references = referenceFrames[:referenceCountN]
		}
	}
	return step, output, referenceCount, references, nil
}

func (s *FrameWorkState) runStepExternalRefresh(refs *SurfaceReferences, framePool *frame.Pool, event *Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner, postPublisher FrameWorkPostFilterGlobalSurfacePublisher, globalSurface func(local int) int, releaser FrameSurfaceReleaser) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	dispatch := s.payloadStepDispatch(event, step, output, references, payload, true, jobs, batches)
	executed, err := dispatch.executeRunner(workerPool, runner)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if output == nil {
		output, err = frameWorkPostFilterRunnerOutput(event, framePool, step, post)
		if err != nil {
			return FrameWorkStepResult{ExecutedTileWork: executed}, err
		}
	}
	if err := runFrameWorkPostFilterRunner(event, step, output, referenceCount, executed, dispatch.cdefIndexMap, dispatch.loopFilterMap, dispatch.loopFilterMasks, dispatch.restorationFrameBuffers, workerPool, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	publishedGlobalSurface, hasPublishedGlobalSurface := frameWorkPostFilterPublishedGlobalSurface(post, postPublisher)
	completed, releaseCount, err := s.finishIfEventCompletesFrameWorkExternal(refs, framePool, event, releases, globalSurface, releaser, publishedGlobalSurface, hasPublishedGlobalSurface)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	totalReleaseCount, err := frameWorkAddReleaseCount(step.ReleaseCount, releaseCount)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     totalReleaseCount,
	}, nil
}

func (s *FrameWorkState) finishIfEventCompletesFrameWorkExternal(refs *SurfaceReferences, framePool *frame.Pool, event *Event, releases []int, globalSurface func(local int) int, releaser FrameSurfaceReleaser, publishedGlobalSurface int, hasPublishedGlobalSurface bool) (bool, int, error) {
	if !EventCompletesFrameWork(event) {
		return false, 0, nil
	}
	if s == nil || !s.active {
		return false, 0, ErrInvalidFrameWorkState
	}
	count, err := s.finishExternal(refs, framePool, event, releases, globalSurface, releaser, publishedGlobalSurface, hasPublishedGlobalSurface)
	if err != nil {
		return false, 0, err
	}
	return true, count, nil
}

func (s *FrameWorkState) finishExternal(refs *SurfaceReferences, framePool *frame.Pool, event *Event, releases []int, globalSurface func(local int) int, releaser FrameSurfaceReleaser, publishedGlobalSurface int, hasPublishedGlobalSurface bool) (int, error) {
	if s == nil || !s.active {
		return 0, ErrInvalidFrameWorkState
	}
	if refs == nil {
		return 0, ErrInvalidSurfaceReference
	}
	if (event.Kind != EventFrame && event.Kind != EventTileGroup) || !event.TileGroup.Final {
		return 0, ErrInvalidSurfaceEvent
	}

	localSurface := s.Surface
	codedGlobalID := globalSurface(localSurface)
	if codedGlobalID < 0 {
		return 0, ErrInvalidSurfaceReference
	}
	globalID := codedGlobalID
	if hasPublishedGlobalSurface {
		if publishedGlobalSurface < 0 {
			return 0, ErrInvalidSurfaceReference
		}
		globalID = publishedGlobalSurface
	}

	next := *refs
	count, err := next.Refresh(event.FrameSize.RefreshFrameFlags, globalID, releases)
	if err != nil {
		return 0, err
	}
	releaseIDs := releases[:count]
	releaseCount := count
	if hasPublishedGlobalSurface && globalID != codedGlobalID {
		copy(s.externalReleaseScratch[:], releases[:count])
		s.externalReleaseScratch[count] = codedGlobalID
		releaseCount = count + 1
		releaseIDs = s.externalReleaseScratch[:releaseCount]
	}
	if releaseCount != 0 {
		if releaser != nil {
			if err := releaser.ReleaseFrameSurfaces(releaseIDs); err != nil {
				return 0, err
			}
		} else if framePool != nil {
			// Fallback: assume all releases live in framePool.
			if err := framePool.ReleaseMany(releaseIDs); err != nil {
				return 0, err
			}
		}
	}

	*refs = next
	s.finishTileResidualCDFs(event, globalID)
	s.publishTemporalMotionFrame(event, globalID)
	s.segMapStore().publish(event.FrameSize.RefreshFrameFlags, s.segFrame.cur)
	s.resetActive()
	return releaseCount, nil
}

func frameWorkPostFilterPublishedGlobalSurface(post FrameWorkPostFilterRunner, publisher FrameWorkPostFilterGlobalSurfacePublisher) (int, bool) {
	if publisher != nil {
		return publisher.PublishedFrameWorkGlobalSurface()
	}
	if post == nil {
		return -1, false
	}
	fallback, ok := post.(FrameWorkPostFilterGlobalSurfacePublisher)
	if !ok {
		return -1, false
	}
	return fallback.PublishedFrameWorkGlobalSurface()
}
