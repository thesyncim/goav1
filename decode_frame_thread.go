// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package goav1

import (
	"fmt"
	"sync"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// WithMaxFrameDelay bounds how many frames the decoder keeps in flight at once
// (dav1d's Dav1dSettings.max_frame_delay). With more than one frame in flight
// the decoder runs dav1d-style frame threading: the entropy pass of frame N+1
// runs while frame N is still being reconstructed and post-filtered, so a
// decoded frame is returned up to n-1 DecodeNext calls after the call that
// submitted it, and the final calls drain the pipeline. Output is byte-identical
// to the serial decode at every setting.
//
// 0 (the default) picks dav1d's automatic depth, ceil(sqrt(workers)) capped at
// 8; 1 decodes each frame completely before DecodeNext returns. The depth never
// exceeds the worker count, so a single-worker decoder is always serial. RTP
// decoders (NewDecoderFromRTPPayloads) are realtime and always use a delay of 1.
func WithMaxFrameDelay(n int) Option {
	return func(c *decoderConfig) { c.maxFrameDelay = n }
}

// decoderFrameContextCount is dav1d's get_num_threads n_fc (src/lib.c).
func decoderFrameContextCount(workers, maxFrameDelay int) int {
	if workers < 1 {
		return 1
	}
	if maxFrameDelay > 0 {
		return min(maxFrameDelay, workers, internaldecoder.FrameThreadMaxFrames)
	}
	// ceil(sqrt(n)), min 8 (fc_lut).
	n := 1
	for n*n < workers && n < 8 {
		n++
	}
	return n
}

// decoderFrameContext is one in-flight frame of the frame-threaded decode
// (dav1d's Dav1dFrameContext): every buffer the frame's parse writes and its
// reconstruction / post-filter tail still reads, so the next frame parses into
// a different context while this one finishes.
type decoderFrameContext struct {
	event       DecoderEvent
	scratch     DecoderFrameWorkResidualEventScratch
	sideScratch DecoderFrameWorkSideDataScratch
	batch       DecoderFrameWorkBatchResidualRunner
	sideData    DecoderFrameWorkSideData

	lfLevels    internalthreading.FrameWorkLoopFilterLevelTable
	lfTileEdges internalthreading.FrameWorkLoopFilterTileEdges
	lfNoSkip    internalthreading.FrameWorkLoopFilterNoSkip

	postFilter   DecoderFrameWorkReusableSupportedPostFilterRunner
	postParallel DecoderFrameWorkPostFilterParallel

	refSurface   [InterRefsPerFrame]int
	refFrames    [InterRefsPerFrame]*Frame
	planReleases [RefFrames]int

	ft internaldecoder.FrameThreadFrame

	// Entropy-pass inputs captured at submission; the context goroutine runs
	// the pass after the submitting call returned.
	payload    []byte
	step       DecoderFrameWorkStep
	target     *Frame
	references []*Frame
	show       bool
	motion     internalthreading.FrameThreadMotion
	cdfWait    *internalthreading.FrameProgress
	cdfOut     *internalthreading.FrameProgress
	segWait    *internalthreading.FrameProgress
	segOut     *internalthreading.FrameProgress

	// Tail hand-off. busy is owned by the decode goroutine; start and done
	// order the tail goroutine's writes (output, tailErr) before the next read.
	busy    bool
	surface int
	output  *Frame
	tailErr error
	start   chan struct{}
	done    chan struct{}
}

// decoderFramePipeline is the frame-threading coordinator: dav1d's
// c->fc[n_fc] ring with c->frame_thread.next and out_delayed
// (dav1d_submit_frame / drain_picture, src/decode.c and src/lib.c). The
// decode goroutine runs every frame's entropy pass in order; each context's
// tail goroutine finishes reconstruction and post-filter. Outputs leave the
// ring in submission order when a context is reused or drained.
type decoderFramePipeline struct {
	d        *Decoder
	fcs      []decoderFrameContext
	next     int
	laneNeed int

	// progress is the per-surface pixel progress (dav1d
	// Dav1dThreadPicture.progress[1]). Surfaces not owned by an in-flight frame
	// stay at FrameProgressDone.
	progress []internalthreading.FrameProgress

	// mvProgress is the entropy progress of each MV_REF grid handle (dav1d
	// progress[0], in 8x8 rows); cdfProgress marks each saved-CDF entry final
	// (dav1d's cdf_thread progress). Both stay final unless a frame in flight
	// is producing them.
	mvProgress  []internalthreading.FrameProgress
	cdfProgress []internalthreading.FrameProgress
	// segProgress marks each segmentation-map handle final once the frame
	// writing it finished its entropy pass.
	segProgress []internalthreading.FrameProgress

	// hookMu serializes submission with a test parse hook.
	hookMu sync.Mutex

	ready []*Frame
	err   error
}

func (p *decoderFramePipeline) tailLoop(fc *decoderFrameContext) {
	for range fc.start {
		err := p.parseFrame(fc)
		// The entropy outputs are final: wake the frames whose entropy pass
		// waits on this frame's CDFs or MV_REF rows.
		entropy := internalthreading.FrameProgressDone
		if err != nil {
			entropy = internalthreading.FrameProgressError
		}
		if fc.motion.Progress != nil {
			fc.motion.Progress.Publish(entropy)
		}
		if fc.cdfOut != nil {
			fc.cdfOut.Publish(entropy)
		}
		if fc.segOut != nil {
			fc.segOut.Publish(entropy)
		}
		if err != nil {
			_ = fc.ft.Recon.Join()
		} else {
			err = fc.ft.Tail(&fc.postFilter)
		}
		fc.output = nil
		if err == nil && fc.show {
			fc.output = fc.target
		}
		if err == nil && fc.output != nil {
			if out, ok := fc.postFilter.PostFilterOutput(); ok && out != nil {
				fc.output = out
			}
		}
		state := internalthreading.FrameProgressDone
		if err != nil {
			state = internalthreading.FrameProgressError
		}
		p.progress[fc.surface].Publish(state)
		fc.tailErr = err
		fc.done <- struct{}{}
	}
}

// eligible reports whether event takes the frame-threaded path.
func (p *decoderFramePipeline) eligible(sequence SequenceHeader, event *DecoderEvent) bool {
	return internaldecoder.EventFrameThreadable(event) && !sequence.FilmGrainParamsPresent &&
		!p.d.state.Active()
}

// retire waits for fc's in-flight frame, returns the surfaces it dropped from
// the reference slots to the pool, and queues its output.
func (p *decoderFramePipeline) retire(fc *decoderFrameContext) {
	if !fc.busy {
		return
	}
	<-fc.done
	fc.busy = false
	p.d.state.ReleaseFrameThreadHolds(&fc.ft)
	for _, s := range fc.ft.Releases[:fc.ft.ReleaseCount] {
		if p.readyHolds(s) {
			// Returned by this call: the shown-surface bookkeeping releases
			// it on the next call, once the caller is done with it.
			continue
		}
		_ = p.d.pool.Release(s)
	}
	fc.ft.ReleaseCount = 0
	if fc.tailErr != nil {
		if p.err == nil {
			p.err = fc.tailErr
		}
		fc.tailErr = nil
		fc.output = nil
		return
	}
	if fc.output != nil {
		p.ready = append(p.ready, fc.output)
		fc.output = nil
	}
}

func (p *decoderFramePipeline) readyHolds(surface int) bool {
	f, err := p.d.pool.Frame(surface)
	if err != nil {
		return false
	}
	for _, r := range p.ready {
		if r == f {
			return true
		}
	}
	return false
}

// deferredRelease reports whether an in-flight frame will return surface to
// the pool when it retires.
func (p *decoderFramePipeline) deferredRelease(surface int) bool {
	for i := range p.fcs {
		fc := &p.fcs[i]
		if !fc.busy {
			continue
		}
		for _, s := range fc.ft.Releases[:fc.ft.ReleaseCount] {
			if s == surface {
				return true
			}
		}
	}
	return false
}

// drain retires every in-flight frame, oldest first (dav1d drain_picture).
func (p *decoderFramePipeline) drain() {
	n := len(p.fcs)
	for k := 0; k < n; k++ {
		p.retire(&p.fcs[(p.next+k)%n])
	}
}

// retireOldest retires the oldest in-flight frame, reporting false when none
// is in flight.
func (p *decoderFramePipeline) retireOldest() bool {
	n := len(p.fcs)
	for k := 0; k < n; k++ {
		fc := &p.fcs[(p.next+k)%n]
		if fc.busy {
			p.retire(fc)
			return true
		}
	}
	return false
}

// runFrame submits one frame-threadable event (dav1d_submit_frame): it takes
// the next context, retiring the frame it held, runs the entropy pass and
// publication on the calling goroutine and hands reconstruction and the
// post-filter to the context's tail.
func (p *decoderFramePipeline) runFrame(sequence SequenceHeader, src *DecoderEvent) (DecoderFrameWorkEventResult, error) {
	d := p.d
	handed := false
	if decoderFramePipelineParseHook != nil {
		// Released by the hooked frame once its pass returned.
		p.hookMu.Lock()
		defer func() {
			if !handed {
				p.hookMu.Unlock()
			}
		}()
	}
	fc := &p.fcs[p.next]
	p.next++
	if p.next == len(p.fcs) {
		p.next = 0
	}
	p.retire(fc)
	for d.pool.Available() == 0 && p.retireOldest() {
	}

	fc.event = *src
	fc.event.SequenceHeader = sequence
	if fc.payload != nil && len(fc.event.Unit.Raw) != 0 {
		// The source reuses its payload buffer on the next call, while this
		// frame's entropy pass may still be reading it (dav1d keeps a
		// reference on the tile data instead).
		raw := fc.event.Unit.Raw
		if len(raw) > len(fc.payload) || len(fc.event.Unit.Payload) > len(raw) {
			return DecoderFrameWorkEventResult{}, ErrFrameShortBuffer
		}
		n := copy(fc.payload, raw)
		fc.event.Unit.Raw = fc.payload[:n:n]
		fc.event.Unit.Payload = fc.payload[n-len(src.Unit.Payload) : n : n]
	}
	event := &fc.event

	side, err := bindDecoderFrameWorkSideData(sequence, event.FrameSize, event.CDEF, event.Restoration, fc.sideScratch, nil, nil, nil, true)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	fc.sideData = side
	if fc.sideData.LoopFilterMasks.Valid() {
		fc.lfLevels.Bind(event.LoopFilter, event.Segmentation, event.Delta)
		fc.sideData.LoopFilterMasks.BindLevels(&fc.lfLevels)
		sbLog2 := 4
		if sequence.Use128x128Superblock {
			sbLog2 = 5
		}
		fc.sideData.LoopFilterMasks.BindTileEdges(&fc.lfTileEdges, event.TileInfo, sbLog2)
		fc.sideData.LoopFilterMasks.BindNoSkip(&fc.lfNoSkip)
	}
	if err := fc.batch.ResetStats(); err != nil {
		return DecoderFrameWorkEventResult{}, err
	}

	step, output, err := d.state.PlanEvent(&d.refs, &d.pool, sequence, event, 64, fc.refSurface[:], d.workerPool.WorkerCount(), fc.scratch.Spans, fc.scratch.Jobs, fc.scratch.Batches, fc.planReleases[:])
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	tile, referenceCount, hasTile, err := decoderFrameWorkEventStepTile(step)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if !hasTile || tile.JobCount == 0 || step.Kind != DecoderFrameWorkStepBegin {
		return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
	}
	if err := SetDecoderFrameWorkBatchResidualRunnerSideData(&fc.batch, fc.sideData); err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if err := SetDecoderFrameWorkSideData(&d.state, fc.sideData); err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	surface, err := decoderFrameWorkEventStepSurface(step)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if output == nil {
		if output, err = d.pool.Frame(surface); err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}
	fc.ft.Recon.Reset()
	var references []*Frame
	if n := int(referenceCount); n != 0 {
		count, err := ResolveDecoderFrameReferences(&d.pool, fc.refSurface[:n], fc.refFrames[:])
		if err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
		if count != n {
			return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
		}
		references = fc.refFrames[:n]
		for i, s := range fc.refSurface[:n] {
			fc.ft.Recon.RefProgress[i] = &p.progress[s]
		}
	}

	p.progress[surface].Reset()
	// dav1d_submit_frame: the reference slots, slot CDF handles and slot MV
	// grids name this frame before its entropy pass runs.
	run, err := d.state.PublishFrameThreaded(&d.refs, event, step, &fc.ft)
	if err != nil {
		p.progress[surface].Publish(internalthreading.FrameProgressError)
		return DecoderFrameWorkEventResult{}, err
	}
	fc.surface = surface

	// dav1d check_tile for DAV1D_TASK_TYPE_TILE_ENTROPY: the entropy pass
	// waits only for its initial frame context (the primary reference's
	// out_cdf) and, per superblock row, for the references' saved MV_REF
	// rows (progress[0]); it never waits for reference pixels.
	currentGrid, refGrids := fc.ft.MotionGrids()
	// Field resets keep the motion link's per-tile progress table allocated.
	fc.motion.Progress = nil
	fc.motion.RefProgress = [InterRefsPerFrame]*internalthreading.FrameProgress{}
	if currentGrid != 0 {
		p.mvProgress[currentGrid-1].Reset()
		fc.motion.Progress = &p.mvProgress[currentGrid-1]
	}
	for i, g := range refGrids {
		if g != 0 {
			fc.motion.RefProgress[i] = &p.mvProgress[g-1]
		}
	}
	fc.ft.Recon.Motion = &fc.motion
	initialCDF, retainedCDF, willRetain := fc.ft.CDFEntries()
	fc.cdfWait = &p.cdfProgress[initialCDF]
	fc.cdfOut = nil
	if willRetain {
		p.cdfProgress[retainedCDF].Reset()
		fc.cdfOut = &p.cdfProgress[retainedCDF]
	}
	// dav1d check_tile also waits on the references' entropy progress before
	// the previous segmentation map is read (get_prev_frame_segid); the pass
	// waits until the frame writing that map finished its own pass.
	segWritten, segRead := fc.ft.SegmentMaps()
	fc.segOut, fc.segWait = nil, nil
	if segWritten != 0 {
		p.segProgress[segWritten-1].Reset()
		fc.segOut = &p.segProgress[segWritten-1]
	}
	if segRead != 0 {
		fc.segWait = &p.segProgress[segRead-1]
	}
	fc.step = step
	fc.target = output
	fc.references = references
	fc.show = event.FrameHeader.ShowFrame
	handed = true
	fc.busy = true
	// The context goroutine runs the entropy pass, then the tail; the next
	// frame can be submitted at once. A failed pass reports
	// FrameProgressError to dependents and its error at retire.
	fc.start <- struct{}{}
	return DecoderFrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

// decoderFramePipelineParseHook, when set by a test, runs before a frame's
// entropy pass; the returned function runs once the pass returned, before the
// tail starts.
var decoderFramePipelineParseHook func(p *decoderFramePipeline, fc *decoderFrameContext, references []*Frame) func()

// reset drains the pipeline and marks every surface final.
func (p *decoderFramePipeline) reset() {
	p.drain()
	p.ready = p.ready[:0]
	p.err = nil
	p.next = 0
	for i := range p.progress {
		p.progress[i].Publish(internalthreading.FrameProgressDone)
	}
	for _, set := range [][]internalthreading.FrameProgress{p.mvProgress, p.cdfProgress, p.segProgress} {
		for i := range set {
			set[i].Publish(internalthreading.FrameProgressDone)
		}
	}
}

func (p *decoderFramePipeline) close() {
	p.drain()
	for i := range p.fcs {
		close(p.fcs[i].start)
		p.fcs[i].postParallel.Close()
	}
}

// newDecoderFramePipeline binds n frame contexts from their own arena.
func newDecoderFramePipeline(d *Decoder, n int, plan DecoderFrameWorkResidualStreamPlan, postArena DecoderFrameWorkPostFilterRequestScratchSize, surfaceCount int) (*decoderFramePipeline, error) {
	var size decoderArenaSize
	eventSize := plan.Size.Event
	noSkip := decoderCompactNoSkipBytes(eventSize.SideData.CDEFIndexMap)
	tileEdges := decoderLoopFilterTileEdgeBytes(eventSize.SideData.LoopFilterMasks)
	for range n {
		if err := size.addEvent(eventSize); err != nil {
			return nil, err
		}
		if err := size.addSideData(eventSize.SideData); err != nil {
			return nil, err
		}
		if err := size.addPostFilter(postArena); err != nil {
			return nil, err
		}
		if err := size.addBytes(noSkip); err != nil {
			return nil, err
		}
		if err := size.addBytes(tileEdges); err != nil {
			return nil, err
		}
	}
	arena := newDecoderArena(size)
	workers := d.workerPool.WorkerCount()
	p := &decoderFramePipeline{
		d:        d,
		fcs:      make([]decoderFrameContext, n),
		laneNeed: 2 * n * workers,
		progress: make([]internalthreading.FrameProgress, surfaceCount),
		ready:    make([]*Frame, 0, 2*(RefFrames+n)),
	}
	for i := range p.progress {
		p.progress[i].Init()
		p.progress[i].Publish(internalthreading.FrameProgressDone)
	}
	p.mvProgress = make([]internalthreading.FrameProgress, internaldecoder.FrameWorkMVGrids)
	p.cdfProgress = make([]internalthreading.FrameProgress, internaldecoder.FrameWorkCDFPoolEntries)
	p.segProgress = make([]internalthreading.FrameProgress, internaldecoder.FrameWorkSegMaps)
	for _, set := range [][]internalthreading.FrameProgress{p.mvProgress, p.cdfProgress, p.segProgress} {
		for i := range set {
			set[i].Init()
			set[i].Publish(internalthreading.FrameProgressDone)
		}
	}
	runtime := DecoderFrameWorkResidualEventRuntime{
		State:      &d.state,
		Refs:       &d.refs,
		FramePool:  &d.pool,
		Align:      64,
		WorkerPool: d.workerPool,
	}
	for i := range p.fcs {
		fc := &p.fcs[i]
		fc.scratch = newDecoderEventScratch(eventSize, &arena)
		fc.sideScratch = newDecoderSideDataScratch(eventSize.SideData, &arena)
		fc.lfNoSkip.Preseed(arena.takeBytes(noSkip))
		fc.lfTileEdges.Preseed(arena.takeBytes(tileEdges))
		fc.postFilter.size = postArena
		fc.postFilter.runner.Scratch = decoderPostFilterScratchFromArena(postArena, &arena)
		fc.postParallel.Workers = workers
		fc.postFilter.Parallel = &fc.postParallel
		rt := runtime
		rt.ReferenceSurfaces = fc.refSurface[:]
		rt.ReferenceFrames = fc.refFrames[:]
		rt.Releases = fc.planReleases[:]
		rt.SideData = &fc.sideData
		bound, _, err := BindDecoderFrameWorkResidualEventRunner(eventSize, plan.Bind.Sequence, &plan.Bind.Event, rt, fc.scratch, &fc.batch)
		if err != nil {
			return nil, err
		}
		fc.scratch.Spans = bound.Spans
		fc.scratch.Jobs = bound.Jobs
		fc.scratch.Batches = bound.Batches
		if n := d.payloadSource.payloadBufferLen(); n > 0 {
			fc.payload = make([]byte, n)
		}
		fc.start = make(chan struct{})
		fc.done = make(chan struct{})
		go p.tailLoop(fc)
	}
	return p, nil
}

// decodeNextFrameThreaded is DecodeNext under frame threading: it submits the
// next payload's frames and returns the frames that left the ring (dav1d
// dav1d_send_data + dav1d_get_picture). Once the payloads are exhausted it
// drains the ring, so the last frames arrive on the final calls.
func (d *Decoder) decodeNextFrameThreaded() (frames []*Frame, ok bool, err error) {
	p := d.fp
	p.ready = p.ready[:0]
	if d.next >= d.payloadSource.len() {
		p.drain()
		if err := p.takeErr(); err != nil {
			return nil, false, err
		}
		if len(p.ready) == 0 {
			return nil, false, nil
		}
		return d.publishFrameThreadedOutputs(), true, nil
	}
	i := d.next
	d.next++
	payload, err := d.payloadSource.payloadAt(i, d.payloadBuf)
	if err != nil {
		p.drain()
		return nil, false, fmt.Errorf("goav1: frame %d payload: %w", i, err)
	}
	var result DecoderFrameWorkResidualStreamResult
	err = d.runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, d.postFilterRunner())
	if err != nil {
		p.drain()
		p.ready = p.ready[:0]
		_ = p.takeErr()
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}
	if err := p.takeErr(); err != nil {
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}
	return d.publishFrameThreadedOutputs(), true, nil
}

func (p *decoderFramePipeline) takeErr() error {
	err := p.err
	p.err = nil
	return err
}

func (d *Decoder) publishFrameThreadedOutputs() []*Frame {
	out := d.visible[:0]
	out = append(out, d.fp.ready...)
	d.visible = out
	d.trackShownSurfaces(out)
	return out
}

// decoderFramePipelineEventNeedsDrain reports whether a non-threaded event
// touches frame or reference state and must wait for every in-flight frame.
// Temporal delimiters, metadata, padding and sequence headers that do not start
// a new coded video sequence leave the reference state alone (dav1d keeps
// parsing OBUs while frame threads run).
func decoderFramePipelineEventNeedsDrain(event *DecoderEvent) bool {
	if event.NewCodedVideoSequence {
		return true
	}
	switch event.Kind {
	case DecoderEventIgnored, DecoderEventTemporalDelimiter, DecoderEventMetadata,
		DecoderEventPadding, DecoderEventReserved, DecoderEventSequenceHeader:
		return false
	}
	return true
}

// parseFrame runs a submitted frame's entropy pass on its context goroutine
// (dav1d's DAV1D_TASK_TYPE_TILE_ENTROPY): it waits for the frame's initial
// frame context, then decodes the tile; per superblock row the block loop
// waits for the references' MV_REF rows and publishes its own.
func (p *decoderFramePipeline) parseFrame(fc *decoderFrameContext) error {
	d := p.d
	if fc.cdfWait != nil && !fc.cdfWait.Wait(internalthreading.FrameProgressDone) {
		return internalthreading.ErrFrameThreadReference
	}
	if fc.segWait != nil && !fc.segWait.Wait(internalthreading.FrameProgressDone) {
		return internalthreading.ErrFrameThreadReference
	}
	var afterParse func()
	if decoderFramePipelineParseHook != nil {
		afterParse = decoderFramePipelineParseHook(p, fc, fc.references)
		defer p.hookMu.Unlock()
	}
	err := d.state.RunStepFrameThreaded(&fc.event, fc.step, d.workerPool, fc.target, fc.references, fc.scratch.Jobs, fc.scratch.Batches, &fc.batch, &fc.ft, p.laneNeed)
	if afterParse != nil {
		afterParse()
	}
	return err
}
