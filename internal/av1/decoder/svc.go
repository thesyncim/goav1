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

// FrameSurfaceProviderFunc adapts a plain function to FrameSurfaceProvider.
type FrameSurfaceProviderFunc func(int) (*frame.Frame, error)

// FrameSurface implements FrameSurfaceProvider.
func (fn FrameSurfaceProviderFunc) FrameSurface(id int) (*frame.Frame, error) {
	if fn == nil {
		return nil, ErrInvalidSurfaceReference
	}
	return fn(id)
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

// TemporalMotionReferenceProvider routes caller-owned surface IDs to the
// TemporalMotionReferenceFrame metadata that ResolveTemporalMotionReferences
// would otherwise look up in a single per-surface store. This is the SVC
// counterpart of indexing into a single per-layer mvStore by surface index.
type TemporalMotionReferenceProvider interface {
	TemporalMotionReference(id int) (tile.TemporalMotionReferenceFrame, error)
}

// TemporalMotionReferenceProviderFunc adapts a plain function to
// TemporalMotionReferenceProvider.
type TemporalMotionReferenceProviderFunc func(id int) (tile.TemporalMotionReferenceFrame, error)

// TemporalMotionReference implements TemporalMotionReferenceProvider.
func (fn TemporalMotionReferenceProviderFunc) TemporalMotionReference(id int) (tile.TemporalMotionReferenceFrame, error) {
	if fn == nil {
		return tile.TemporalMotionReferenceFrame{}, ErrInvalidSurfaceReference
	}
	return fn(id)
}

// ResolveTemporalMotionReferencesWithProvider is the multi-pool counterpart
// of ResolveTemporalMotionReferences. dst is published atomically after every
// referenced surface has validated.
func ResolveTemporalMotionReferencesWithProvider(provider TemporalMotionReferenceProvider, surfaces []int, dst []tile.TemporalMotionReferenceFrame) (int, error) {
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

	var resolved [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	for i := range count {
		surface := surfaces[i]
		if surface < 0 {
			return 0, ErrInvalidSurfaceReference
		}
		ref, err := provider.TemporalMotionReference(surface)
		if err != nil {
			return 0, err
		}
		if ref.Frame == nil {
			return 0, ErrInvalidSurfaceReference
		}
		if err := ref.Frame.Validate(); err != nil {
			return 0, ErrInvalidSurfaceReference
		}
		resolved[i] = ref
	}
	for i := range count {
		dst[i] = resolved[i]
	}
	return count, nil
}

// FrameSurfaceReleaserFunc adapts a plain function to FrameSurfaceReleaser.
type FrameSurfaceReleaserFunc func(ids []int) error

// ReleaseFrameSurfaces implements FrameSurfaceReleaser.
func (fn FrameSurfaceReleaserFunc) ReleaseFrameSurfaces(ids []int) error {
	if fn == nil || len(ids) == 0 {
		return nil
	}
	return fn(ids)
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
	contexts map[int]threading.FrameWorkTileResidualCDFStorage
}

// Reset clears every saved frame context. Callers invoke this at a new coded
// video sequence boundary, matching ResetFrameSurfaceReferences clearing the
// stream-global reference slots.
func (s *SharedFrameContextStore) Reset() {
	if s == nil {
		return
	}
	s.contexts = nil
}

// load returns the saved frame context for the global surface ID, reporting
// whether one is present.
func (s *SharedFrameContextStore) load(globalID int) (threading.FrameWorkTileResidualCDFStorage, bool) {
	if s == nil || globalID < 0 || s.contexts == nil {
		return threading.FrameWorkTileResidualCDFStorage{}, false
	}
	ctx, ok := s.contexts[globalID]
	return ctx, ok
}

// store saves the adapted frame context under the global surface ID, mirroring
// libaom's cm->cur_frame->frame_context = *cm->fc at frame end.
func (s *SharedFrameContextStore) store(globalID int, ctx threading.FrameWorkTileResidualCDFStorage) {
	if s == nil || globalID < 0 {
		return
	}
	if s.contexts == nil {
		s.contexts = make(map[int]threading.FrameWorkTileResidualCDFStorage)
	}
	s.contexts[globalID] = ctx
}

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
func (s *FrameWorkState) RunEventWithContextAndExternalReferences(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, provider FrameSurfaceProvider, globalSurface func(local int) int, releaser FrameSurfaceReleaser, frameContexts *SharedFrameContextStore, side FrameWorkSideDataRunner, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkEventResult, error) {
	if provider == nil {
		return FrameWorkEventResult{}, ErrInvalidSurfaceReference
	}
	if globalSurface == nil {
		// Identity: the caller is not using a global namespace.
		globalSurface = func(local int) int { return local }
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

	step, output, referenceCount, references, err := s.planEventWithExternalReferenceContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, provider)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	if err := s.bindFrameWorkEventSideData(event, step, output, references, event.Unit.Payload, side); err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.runStepExternalRefresh(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, runner, post, globalSurface, releaser)
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

func (s *FrameWorkState) clearSharedFrameContext() {
	if s == nil {
		return
	}
	s.sharedFrameContexts = nil
	s.sharedFrameContextRefs = nil
	s.sharedFrameContextGlobalSurface = nil
}

func (s *FrameWorkState) planEventWithExternalReferenceContext(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, provider FrameSurfaceProvider) (FrameWorkStep, *frame.Frame, int, []*frame.Frame, error) {
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
			if step.Kind == FrameWorkStepTile {
				count, err := refs.FrameReferences(event, referenceSurfaces)
				if err != nil {
					return FrameWorkStep{}, nil, 0, nil, err
				}
				if count != referenceCount {
					return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
				}
			}
			count, err := ResolveFrameReferencesWithProvider(provider, referenceSurfaces[:referenceCount], referenceFrames)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
			if count != referenceCount {
				return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
			}
			references = referenceFrames[:referenceCount]
		}
	}
	return step, output, referenceCount, references, nil
}

func (s *FrameWorkState) runStepExternalRefresh(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner, globalSurface func(local int) int, releaser FrameSurfaceReleaser) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	frameContext := frameWorkFrameContext(event, s.sequenceContext())
	var initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFsValid *bool
	if s != nil && s.tileResidualCurrentCDFsValid {
		initialTileResidualCDFs = &s.tileResidualCurrentCDFs
		retainedTileResidualCDFs = &s.tileResidualRetainedCDFs
		retainedTileResidualCDFsValid = &s.tileResidualRetainedCDFsValid
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	executed, err := executeFrameWorkStepWithPayloadRunner(step, workerPool, output, references, payload, true, frameContext, event.FrameHeader.DisableCDFUpdate, initialTileResidualCDFs, retainedTileResidualCDFs, retainedTileResidualCDFsValid, s.motionFields(), cdefIndexMap, loopFilterMap, restorationFrameBuffers, jobs, batches, runner)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if output == nil {
		output, err = frameWorkPostFilterRunnerOutput(event, framePool, step, post)
		if err != nil {
			return FrameWorkStepResult{ExecutedTileWork: executed}, err
		}
	}
	if err := runFrameWorkPostFilterRunner(event, step, output, referenceCount, executed, cdefIndexMap, loopFilterMap, restorationFrameBuffers, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	publishedGlobalSurface, hasPublishedGlobalSurface := frameWorkPostFilterPublishedGlobalSurface(post)
	completed, releaseCount, err := s.finishIfEventCompletesFrameWorkExternal(refs, framePool, event, releases, globalSurface, releaser, publishedGlobalSurface, hasPublishedGlobalSurface)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     step.ReleaseCount + releaseCount,
	}, nil
}

func (s *FrameWorkState) finishIfEventCompletesFrameWorkExternal(refs *SurfaceReferences, framePool *frame.Pool, event Event, releases []int, globalSurface func(local int) int, releaser FrameSurfaceReleaser, publishedGlobalSurface int, hasPublishedGlobalSurface bool) (bool, int, error) {
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

func (s *FrameWorkState) finishExternal(refs *SurfaceReferences, framePool *frame.Pool, event Event, releases []int, globalSurface func(local int) int, releaser FrameSurfaceReleaser, publishedGlobalSurface int, hasPublishedGlobalSurface bool) (int, error) {
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
	s.publishTemporalMotionFrame(event)
	s.resetActive()
	return releaseCount, nil
}

func frameWorkPostFilterPublishedGlobalSurface(post FrameWorkPostFilterRunner) (int, bool) {
	if post == nil {
		return -1, false
	}
	publisher, ok := post.(FrameWorkPostFilterGlobalSurfacePublisher)
	if !ok {
		return -1, false
	}
	return publisher.PublishedFrameWorkGlobalSurface()
}
