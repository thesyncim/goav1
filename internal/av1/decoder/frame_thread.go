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

// FrameThreadFrame is one in-flight frame of the frame-threaded decode
// (dav1d's Dav1dFrameContext under n_fc > 1, src/decode.c + src/thread_task.c).
// RunStepFrameThreaded runs the frame's entropy pass and publishes its parse
// outputs; Tail later finishes the reconstruction and post-filter on another
// goroutine, while the next frame parses.
//
// The frame owns copies of its post-filter side data, so the next frame's
// Begin can rebind the FrameWorkState while this frame's tail still reads the
// CDEF index map, loop-filter masks, and restoration buffers. Pool releases of
// surfaces the frame dropped from the reference slots are deferred to
// Releases: in-flight frames may still predict from them (dav1d holds those
// pictures by refcount until the frame finishes, dav1d_thread_picture_unref in
// dav1d_decode_frame_exit).
type FrameThreadFrame struct {
	Recon threading.FrameThreadRecon

	cdefIndexMap            threading.FrameWorkCDEFIndexMap
	loopFilterMap           threading.FrameWorkLoopFilterMap
	loopFilterMasks         threading.FrameWorkLoopFilterMasks
	restorationFrameBuffers threading.FrameWorkRestorationFrameBuffers
	cdefIndexMapValid       bool
	loopFilterMapValid      bool
	loopFilterMasksValid    bool
	restorationValid        bool

	post    FrameWorkPostFilterContext
	hasPost bool

	// Releases holds ReleaseCount surfaces the frame's reference refresh
	// dropped; the owner returns them to the frame pool once every frame that
	// could read them has finished.
	Releases     [parser.RefFrames]int
	ReleaseCount int
}

// EventFrameThreadable reports whether event is a coded frame that the
// frame-threaded path can split: a complete single-tile frame OBU outside an
// intra-block-copy frame (whose block copies read the frame being
// reconstructed at arbitrary distances, so the parse cannot run ahead of the
// pixels), not starting a new coded video sequence (whose reference reset
// releases surfaces immediately).
func EventFrameThreadable(event *Event) bool {
	return event != nil && event.Kind == EventFrame && event.TileGroup.Final &&
		event.TileGroup.TileCount == 1 && event.TileInfo.Cols == 1 && event.TileInfo.Rows == 1 &&
		!event.NewCodedVideoSequence && !event.FrameSize.AllowIntrabc && !event.FrameSize.SuperResEnabled &&
		!event.FrameHeader.ShowExistingFrame
}

func (f *FrameThreadFrame) captureSideData(s *FrameWorkState) {
	f.cdefIndexMap, f.cdefIndexMapValid = s.cdefIndexMap, s.cdefIndexMapValid
	f.loopFilterMap, f.loopFilterMapValid = s.loopFilterMap, s.loopFilterMapValid
	f.loopFilterMasks, f.loopFilterMasksValid = s.loopFilterMasks, s.loopFilterMasksValid
	f.restorationFrameBuffers, f.restorationValid = s.restorationFrameBuffers, s.restorationFrameBuffersValid
}

func (f *FrameThreadFrame) sideData() (*threading.FrameWorkCDEFIndexMap, *threading.FrameWorkLoopFilterMap, *threading.FrameWorkLoopFilterMasks, *threading.FrameWorkRestorationFrameBuffers) {
	var cdef *threading.FrameWorkCDEFIndexMap
	var lfMap *threading.FrameWorkLoopFilterMap
	var lfMasks *threading.FrameWorkLoopFilterMasks
	var lr *threading.FrameWorkRestorationFrameBuffers
	if f.cdefIndexMapValid {
		cdef = &f.cdefIndexMap
	}
	if f.loopFilterMapValid {
		lfMap = &f.loopFilterMap
	}
	if f.loopFilterMasksValid {
		lfMasks = &f.loopFilterMasks
	}
	if f.restorationValid {
		lr = &f.restorationFrameBuffers
	}
	return cdef, lfMap, lfMasks, lr
}

// RunStepFrameThreaded runs the entropy pass of a frame-threadable event (see
// EventFrameThreadable) planned by PlanEvent, then publishes the frame's parse
// outputs: the reference slots, the adapted CDFs and the MV_REF grid, exactly
// what FrameWorkState.Finish publishes, but before the frame is reconstructed
// and post-filtered. dav1d signals the same outputs right after the
// update-tile's entropy pass (thread_task.c, the out_cdf / progress[0] signal)
// so the next frame's entropy pass can start. The state is left inactive for
// the next Begin. Reconstruction lanes keep running; ft.Tail completes the
// frame. laneNeed bounds the reconstruction lanes live at once across every
// in-flight frame.
func (s *FrameWorkState) RunStepFrameThreaded(refs *SurfaceReferences, event *Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, runner threading.FrameWorkBatchRunner, ft *FrameThreadFrame, laneNeed int) (FrameWorkStepResult, error) {
	if s == nil || !s.active || ft == nil || refs == nil || output == nil || !EventFrameThreadable(event) || !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkState
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	ft.captureSideData(s)
	ft.hasPost = false
	ft.ReleaseCount = 0
	cdefIndexMap, loopFilterMap, loopFilterMasks, restoration := ft.sideData()
	frameContext := frameWorkFrameContext(event, s.sequenceContext())
	initialCDFs, retainedCDFs, retainedCDFsValid := s.tileResidualCDFBindings()
	prepared, err := prepareFrameWorkStepWithPayload(step, output, references, event.Unit.Payload, true, frameContext, event.FrameHeader.DisableCDFUpdate, initialCDFs, retainedCDFs, retainedCDFsValid, s.motionFields(), cdefIndexMap, loopFilterMap, loopFilterMasks, restoration, jobs, batches)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if !prepared.HasTileWork || prepared.BatchCount != 1 {
		return FrameWorkStepResult{}, ErrInvalidTileWork
	}
	prepared.Base.FrameThread = &ft.Recon
	if err := workerPool.ExecuteFrameWorkRunnerFrameThread(batches[0], jobs[:prepared.JobCount], prepared.Base, runner, laneNeed); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: true}, err
	}

	ft.post = FrameWorkPostFilterContext{
		Event:                   *event,
		Step:                    step,
		Output:                  output,
		ReferenceCount:          referenceCount,
		ExecutedTileWork:        true,
		CDEFIndexMap:            cdefIndexMap,
		LoopFilterMap:           loopFilterMap,
		LoopFilterMasks:         loopFilterMasks,
		RestorationFrameBuffers: restoration,
		pool:                    workerPool,
	}
	ft.hasPost = true

	// Publication seam: the reference-slot refresh without the pool release,
	// then the adapted CDFs and the MV_REF grid (Finish's order).
	next := *refs
	count, err := next.FinishFrame(event, s.Surface, ft.Releases[:])
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: true}, err
	}
	*refs = next
	ft.ReleaseCount = count
	s.finishTileResidualCDFs(event, -1)
	s.publishTemporalMotionFrame(event)
	s.resetActive()
	releaseCount, err := frameWorkAddReleaseCount(step.ReleaseCount, count)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: true}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: true,
		CompletedFrame:   true,
		ReleaseCount:     releaseCount,
	}, nil
}

// Tail finishes a frame-threaded frame: it joins the reconstruction lanes and
// runs the post-filter chain (dav1d's DAV1D_TASK_TYPE_TILE_RECONSTRUCTION and
// the deblock / CDEF / super-res / loop-restoration sbrow tasks).
func (f *FrameThreadFrame) Tail(post FrameWorkPostFilterRunner) error {
	if err := f.Recon.Join(); err != nil {
		return err
	}
	if !f.hasPost {
		return ErrInvalidFrameWorkState
	}
	f.hasPost = false
	if post == nil {
		return nil
	}
	return post.Apply(f.post)
}
