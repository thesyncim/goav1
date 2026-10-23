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

// FrameThreadMaxFrames bounds the frames in flight under frame threading
// (dav1d caps n_fc at 8 in get_num_threads).
const FrameThreadMaxFrames = 8

// The saved-CDF pool holds the RefFrames slot targets, the planning frame's
// initial and retained entries, and, per in-flight frame, its initial and
// retained entries, which it keeps after the slots move on. The MV-grid pool
// likewise holds the slot grids, the planning frame's grid and, per in-flight
// frame, its own grid plus the reference grids its parse projects from.
const (
	frameWorkCDFPoolEntries = parser.RefFrames + 2 + 2*FrameThreadMaxFrames
	frameWorkMVGrids        = parser.RefFrames + 1 + (1+parser.InterRefsPerFrame)*FrameThreadMaxFrames
)

// FrameThreadFrame is one in-flight frame of the frame-threaded decode
// (dav1d's Dav1dFrameContext under n_fc > 1, src/decode.c + src/thread_task.c).
//
// PublishFrameThreaded runs at the frame's Begin, the way dav1d_submit_frame
// updates c->refs[] before any tile is decoded: the reference slots, the
// slot frame contexts (CDF handles) and the slot MV_REF grids point at this
// frame immediately, and the frame takes over everything its own entropy pass
// reads (its initial CDF and retained-CDF entries, its MV grid, the reference
// MV grids it projects from, its post-filter side data). The next frame can
// therefore be planned while this one is still parsing. RunStepFrameThreaded
// then runs the entropy pass from these bindings; Tail finishes the
// reconstruction and post-filter; ReleaseFrameThreadHolds returns the held
// entries once the frame is retired.
//
// Pool releases of surfaces the frame dropped from the reference slots are
// deferred to Releases: in-flight frames may still predict from them (dav1d
// holds those pictures by refcount until the frame finishes).
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

	// Entropy-pass bindings taken over from the FrameWorkState at Begin.
	sequence       threading.FrameWorkSequenceContext
	referenceCount uint8
	initialCDF     int8
	retainedCDF    int8
	cdfHeld        bool
	willRetain     bool
	retainedValid  bool
	currentMV      tile.ReferenceMVFrame
	currentMVValid bool
	temporal       bool
	mvRefFrames    [parser.InterRefsPerFrame]tile.ReferenceMVFrame
	mvRefs         [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	currentGrid    uint8
	refGrids       [parser.InterRefsPerFrame]uint8
	heldGrids      [1 + parser.InterRefsPerFrame]uint8
	heldGridCount  int
	published      bool
	// seg holds the frame's segmentation-map handles (its current map and
	// the previous map its entropy pass reads) until it retires.
	seg      segMapFrame
	segStore *SegmentMapStore
	// segView is the block-loop view of seg, taken on the planning goroutine.
	segView threading.FrameWorkSegmentMaps

	post    FrameWorkPostFilterContext
	hasPost bool

	// Releases holds ReleaseCount surfaces the frame's reference refresh
	// dropped; the owner returns them to the frame pool once every frame that
	// could read them has finished.
	Releases     [parser.RefFrames]int
	ReleaseCount int
}

// EventFrameThreadable reports whether event is a coded frame that the
// frame-threaded path can split: a frame whose every tile arrives in one final
// tile group (dav1d submits a frame once all its tile data is in), outside an
// intra-block-copy frame (whose block copies read the frame being
// reconstructed at arbitrary distances, so the parse cannot run ahead of the
// pixels), not starting a new coded video sequence (whose reference reset
// releases surfaces immediately).
func EventFrameThreadable(event *Event) bool {
	return event != nil && event.Kind == EventFrame && event.TileGroup.Final &&
		event.TileGroup.StartTile == 0 && int(event.TileGroup.TileCount) == int(event.TileInfo.Cols)*int(event.TileInfo.Rows) &&
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

func (f *FrameThreadFrame) motionFields() frameWorkMotionFields {
	var fields frameWorkMotionFields
	if f.currentMVValid {
		fields.current = &f.currentMV
	}
	if f.temporal {
		fields.references = f.mvRefs
		fields.deferred = &f.mvRefs
	}
	if f.segView.Stride != 0 {
		fields.seg = &f.segView
	}
	return fields
}

func (s *FrameWorkState) holdMVGrid(f *FrameThreadFrame, handle uint8) {
	if handle == 0 {
		return
	}
	s.mvGridRefs[handle-1]++
	f.heldGrids[f.heldGridCount] = handle
	f.heldGridCount++
}

// PublishFrameThreaded publishes a planned frame-threadable event (see
// EventFrameThreadable) at its Begin, before its entropy pass runs: the
// reference-slot refresh without the pool release, the slot frame contexts
// and the slot MV_REF grids, exactly what FrameWorkState.Finish publishes. The
// slot CDF handle is the frame's retained entry when its update tile will
// save the adapted contexts (refresh_frame_context and CDF updates on), else
// its initial entry, so the choice is known before the parse; dav1d binds the
// same out_cdf handle in dav1d_submit_frame. The state is left inactive for
// the next Begin; ft owns the frame's entropy-pass bindings.
func (s *FrameWorkState) PublishFrameThreaded(refs *SurfaceReferences, event *Event, step FrameWorkStep, ft *FrameThreadFrame) (FrameWorkStepResult, error) {
	if s == nil || !s.active || ft == nil || refs == nil || s.sharedFrameContexts != nil || !s.ownedTemporalMotion ||
		!EventFrameThreadable(event) || !frameWorkStepMatchesEvent(event, step) || s.temporalMVsValid ||
		!s.tileResidualCurrentCDFsValid {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkState
	}
	ft.captureSideData(s)
	ft.sequence = s.sequenceContext()
	ft.referenceCount = s.ReferenceCount
	ft.hasPost = false
	ft.ReleaseCount = 0
	ft.retainedValid = false
	ft.published = false

	// Entropy bindings: the frame takes over the current/retained CDF holds
	// and its MV grid hold, and holds every reference grid it projects from.
	ft.initialCDF = s.tileResidualCurrentCDF
	ft.retainedCDF = s.tileResidualRetainedCDF
	ft.cdfHeld = true
	ft.willRetain = event.TileInfo.RefreshContext && !event.FrameHeader.DisableCDFUpdate
	ft.currentMV = s.currentMVFrame
	ft.currentMVValid = s.currentMVFrameValid
	ft.temporal = s.temporalMVsDeferred
	ft.currentGrid = s.currentMVGrid
	ft.refGrids = [parser.InterRefsPerFrame]uint8{}
	ft.heldGridCount = 0
	if s.currentMVGrid != 0 {
		ft.heldGrids[0] = s.currentMVGrid
		ft.heldGridCount = 1
	}
	ft.mvRefs = [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame{}
	if ft.temporal {
		for ref := range s.mvReferences {
			r := s.mvReferences[ref]
			if r.Frame == nil {
				continue
			}
			ft.mvRefFrames[ref] = *r.Frame
			r.Frame = &ft.mvRefFrames[ref]
			ft.mvRefs[ref] = r
			slot := int(event.FrameSize.RefFrameIdx[ref])
			if slot >= 0 && slot < parser.RefFrames {
				s.holdMVGrid(ft, s.mvSlotGrid[slot])
				ft.refGrids[ref] = s.mvSlotGrid[slot]
			}
		}
	}

	next := *refs
	count, err := next.FinishFrame(event, s.Surface, ft.Releases[:])
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	*refs = next
	ft.ReleaseCount = count
	s.tileResidualRetainedCDFsValid = ft.willRetain
	s.finishTileResidualCDFs(event, -1)
	s.publishTemporalMotionFrame(event)
	store := s.segMapStore()
	store.publish(event.FrameSize.RefreshFrameFlags, s.segFrame.cur)
	ft.seg = s.segFrame
	ft.segStore = store
	ft.segView = threading.FrameWorkSegmentMaps{}
	if s.segFrame.stride != 0 {
		ft.segView.Current, ft.segView.Previous, ft.segView.Stride = s.segFrame.fields(store)
	}
	s.segFrame = segMapFrame{}
	// The holds now belong to ft; resetActive must not drop them.
	s.tileResidualCurrentCDFsValid = false
	s.tileResidualCurrentCDF = -1
	s.tileResidualRetainedCDF = -1
	s.currentMVGrid = 0
	s.resetActive()
	ft.published = true

	releaseCount, err := frameWorkAddReleaseCount(step.ReleaseCount, count)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: true,
		CompletedFrame:   true,
		ReleaseCount:     releaseCount,
	}, nil
}

// ReleaseFrameThreadHolds returns the CDF entries and MV grids a retired
// frame held. It runs on the goroutine that plans frames.
func (s *FrameWorkState) ReleaseFrameThreadHolds(ft *FrameThreadFrame) {
	if s == nil || ft == nil {
		return
	}
	if ft.cdfHeld {
		s.releaseTileResidualCDFEntry(ft.initialCDF)
		s.releaseTileResidualCDFEntry(ft.retainedCDF)
		ft.cdfHeld = false
	}
	for i := 0; i < ft.heldGridCount; i++ {
		s.releaseMVGrid(ft.heldGrids[i])
	}
	if ft.segStore != nil {
		ft.seg.release(ft.segStore)
		ft.segView = threading.FrameWorkSegmentMaps{}
	}
	ft.heldGridCount = 0
	ft.published = false
}

// RunStepFrameThreaded runs the entropy pass of a frame published by
// PublishFrameThreaded, from the bindings the frame took over. Reconstruction
// lanes keep running when it returns; ft.Tail completes the frame. laneNeed
// bounds the reconstruction lanes live at once across every in-flight frame.
// It reads the shared CDF pool only through the entries ft holds, so it may
// run while the planning goroutine begins later frames.
func (s *FrameWorkState) RunStepFrameThreaded(event *Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, runner threading.FrameWorkBatchRunner, ft *FrameThreadFrame, laneNeed int) error {
	if s == nil || ft == nil || !ft.published || !ft.cdfHeld || output == nil {
		return ErrInvalidFrameWorkState
	}
	cdefIndexMap, loopFilterMap, loopFilterMasks, restoration := ft.sideData()
	frameContext := frameWorkFrameContext(event, ft.sequence)
	initialCDFs := s.cdfEntry(ft.initialCDF)
	retainedCDFs := s.cdfEntry(ft.retainedCDF)
	prepared, err := prepareFrameWorkStepWithPayload(step, output, references, event.Unit.Payload, true, frameContext, event.FrameHeader.DisableCDFUpdate, initialCDFs, retainedCDFs, &ft.retainedValid, ft.motionFields(), cdefIndexMap, loopFilterMap, loopFilterMasks, restoration, jobs, batches)
	if err != nil {
		return err
	}
	if !prepared.HasTileWork || prepared.BatchCount < 1 || prepared.BatchCount > len(batches) {
		return ErrInvalidTileWork
	}
	prepared.Base.FrameThread = &ft.Recon
	if err := workerPool.ExecuteFrameWorkRunnerFrameThread(batches[:prepared.BatchCount], jobs[:prepared.JobCount], prepared.Base, runner, laneNeed); err != nil {
		return err
	}
	if ft.retainedValid != ft.willRetain {
		// The slot frame contexts were bound to the retained entry at Begin;
		// the update tile must have filled it.
		return ErrInvalidFrameWorkState
	}
	ft.post = FrameWorkPostFilterContext{
		Event:                   *event,
		Step:                    step,
		Output:                  output,
		ReferenceCount:          ft.referenceCount,
		ExecutedTileWork:        true,
		CDEFIndexMap:            cdefIndexMap,
		LoopFilterMap:           loopFilterMap,
		LoopFilterMasks:         loopFilterMasks,
		RestorationFrameBuffers: restoration,
		pool:                    workerPool,
	}
	ft.hasPost = true
	return nil
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

// cdfEntry resolves a saved-CDF pool handle.
func (s *FrameWorkState) cdfEntry(idx int8) *threading.FrameWorkTileResidualCDFStorage {
	if int(idx) < len(s.tileResidualCDFPool) {
		return &s.tileResidualCDFPool[idx]
	}
	return &s.tileResidualCDFPoolExtra[int(idx)-len(s.tileResidualCDFPool)]
}

// EnableFrameThreadCDFPool gives a decoder that runs frame threading the
// extra saved-CDF entries its in-flight frames hold. It is a free function in
// the internal package so the public FrameWorkState alias gains no method.
func EnableFrameThreadCDFPool(s *FrameWorkState) {
	if s != nil && s.tileResidualCDFPoolExtra == nil {
		s.tileResidualCDFPoolExtra = new([frameWorkCDFPoolEntries - (parser.RefFrames + 2)]threading.FrameWorkTileResidualCDFStorage)
	}
}

// Pool capacities the frame-threading owner sizes its progress tables by.
const (
	FrameWorkMVGrids        = frameWorkMVGrids
	FrameWorkCDFPoolEntries = frameWorkCDFPoolEntries
)

// MotionGrids reports the frame's own MV_REF grid handle and the grid handle
// of each reference its entropy pass projects from (1-based; 0 means none).
func (f *FrameThreadFrame) MotionGrids() (uint8, [parser.InterRefsPerFrame]uint8) {
	return f.currentGrid, f.refGrids
}

// CDFEntries reports the frame's initial frame-context entry, its retained
// entry, and whether its update tile fills the retained entry.
func (f *FrameThreadFrame) CDFEntries() (initial int8, retained int8, willRetain bool) {
	return f.initialCDF, f.retainedCDF, f.willRetain
}

// SegmentMaps reports the frame's segmentation-map handles (1-based; 0 means
// none): the map its entropy pass writes (0 unless update_map) and the
// previous map it reads.
func (f *FrameThreadFrame) SegmentMaps() (written uint8, read uint8) {
	if f.seg.write {
		written = f.seg.cur
	}
	return written, f.seg.prev
}
