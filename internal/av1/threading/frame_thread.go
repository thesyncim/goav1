// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameProgressDone and FrameProgressError are the terminal values of a
// FrameProgress, matching dav1d's FRAME_ERROR / UINT_MAX picture progress
// sentinels (src/thread_task.c, src/picture.c).
const (
	FrameProgressDone  int32 = math.MaxInt32
	FrameProgressError int32 = -1
)

// FrameProgress is the frame-threading pixel progress of one decoded surface:
// dav1d's Dav1dThreadPicture.progress[1] (src/picture.h). A frame's tail
// publishes how many luma rows of the surface are final (reconstructed and
// post-filtered); reconstruction of a later frame that predicts from the
// surface waits until the rows it reads are published (thread_task.c
// check_tile, the lowest_px reference wait).
type FrameProgress struct {
	rows atomic.Int32

	mu      sync.Mutex
	cond    sync.Cond
	waiters int32
	// tasks, when set, is woken on every publish: its reconstruction tasks
	// check this progress before they are taken.
	tasks *frameTaskPool
}

// BindTaskWake makes every publish wake pool's frame-thread workers.
func (p *FrameProgress) BindTaskWake(pool *Pool) {
	p.tasks = pool.frameTaskPool()
}

// Ready reports whether at least need rows are published or the surface's
// frame failed, so Wait would return at once.
func (p *FrameProgress) Ready(need int32) bool {
	v := p.rows.Load()
	return v >= need || v == FrameProgressError
}

// Reset marks the surface as not yet decoded: no row is available.
func (p *FrameProgress) Reset() {
	p.rows.Store(0)
}

// Publish raises the published row count to rows (FrameProgressDone or
// FrameProgressError end the frame) and wakes waiters.
func (p *FrameProgress) Publish(rows int32) {
	p.mu.Lock()
	p.rows.Store(rows)
	if p.waiters != 0 {
		p.cond.Broadcast()
	}
	p.mu.Unlock()
	p.tasks.wake()
}

// Wait blocks until at least need rows are published. It reports false when
// the surface's frame failed to decode.
func (p *FrameProgress) Wait(need int32) bool {
	if v := p.rows.Load(); v >= need || v == FrameProgressError {
		return v != FrameProgressError
	}
	p.mu.Lock()
	if p.cond.L == nil {
		p.cond.L = &p.mu
	}
	p.waiters++
	for {
		v := p.rows.Load()
		if v >= need || v == FrameProgressError {
			p.waiters--
			p.mu.Unlock()
			return v != FrameProgressError
		}
		p.cond.Wait()
	}
}

// Init prepares the condition variable; it must run before any Publish or
// Wait races with another goroutine.
func (p *FrameProgress) Init() {
	p.cond.L = &p.mu
}

// FrameThreadRecon carries one frame's reconstruction across the frame-thread
// split. dav1d's frame threading runs a frame's tile entropy passes
// (DAV1D_TASK_TYPE_TILE_ENTROPY, one task per tile) separately from their
// reconstruction (DAV1D_TASK_TYPE_TILE_RECONSTRUCTION, src/thread_task.c): the
// entropy pass of frame N+1 starts as soon as frame N's parse outputs are
// published, while N's reconstruction and post-filter still run. Every tile of
// a batch carrying a FrameThreadRecon reconstructs through its own SB-row
// wavefront lanes (the frame-thread buffers, ts->frame_thread[p].cf in dav1d);
// the tile job returns once its parse finished and leaves the lanes running;
// Join waits for every tile's lanes.
//
// Before a lane reconstructs a superblock it waits until every reference
// surface the superblock predicts from has published the rows its motion
// reaches (RefProgress, see waitSuperblockReferences), so the entropy pass
// never waits for pixels.
type FrameThreadRecon struct {
	RefProgress [parser.InterRefsPerFrame]*FrameProgress
	// Motion, when set, links the entropy pass to the saved MV_REF rows of
	// this frame and its references, so the next frame's entropy pass can
	// run concurrently a few superblock rows behind.
	Motion *FrameThreadMotion
	// CDFOut, when set, is the frame's retained frame-context progress: the
	// update tile publishes it as soon as it saved its adapted CDFs (dav1d
	// thread_task.c signals out_cdf right after that tile's entropy pass), so
	// the next frame's entropy pass starts while the other tiles still parse.
	CDFOut *FrameProgress
	// Rows, when set, receives the frame's reconstructed luma rows as the
	// lanes finish them, so the post-filter can follow reconstruction row
	// by row (dav1d's filter tasks wait on the tiles' recon progress).
	Rows *FrameThreadReconRows
	// Seq is the frame's decode order: the pool's workers take the oldest
	// frame's tasks first.
	Seq uint64
	// tasks runs the tiles' reconstruction tasks; reconTasks holds them,
	// usedTasks of them this frame (guarded by mu).
	tasks      *frameTaskPool
	reconTasks []*frameReconTask
	usedTasks  int
	// entropy is the frame's entropy task; entropyQueued records that it
	// was queued this frame, and batchWorker the worker running each batch.
	entropy       frameEntropyTask
	entropyQueued bool
	// EntropyOut are the progress of the frame's entropy outputs later
	// frames wait on (MV_REF grid, segmentation map); the worker ending the
	// pass publishes them final.
	EntropyOut  [2]*FrameProgress
	batchWorker []int

	// pending holds the tiles whose reconstruction lanes still run after
	// their parse returned; the tile parses append to it concurrently.
	mu      sync.Mutex
	pending []frameThreadPendingRecon

	// batches holds each tile batch's context, kept here so the frame-thread
	// dispatch hands the runner a pointer without a per-frame heap copy.
	batches  []FrameWorkBatch
	runner   FrameWorkBatchRunner
	parseWG  sync.WaitGroup
	parseErr atomic.Pointer[wavefrontError]
}

type frameThreadPendingRecon struct {
	wf    *frameWorkReconWavefront
	stats *FrameWorkTileResidualStats
}

// Reset clears the frame's reference waits and any pending join.
func (r *FrameThreadRecon) Reset() {
	r.RefProgress = [parser.InterRefsPerFrame]*FrameProgress{}
	r.Motion = nil
	r.CDFOut = nil
	r.Rows = nil
	r.EntropyOut = [2]*FrameProgress{}
	r.entropyQueued = false
	r.mu.Lock()
	r.usedTasks = 0
	clear(r.pending)
	r.pending = r.pending[:0]
	r.mu.Unlock()
}

// Pending reports whether reconstruction lanes are still owned by the frame.
func (r *FrameThreadRecon) Pending() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	n := len(r.pending)
	r.mu.Unlock()
	return n != 0
}

// addPending hands a parsed tile's running lanes to the frame's tail.
func (r *FrameThreadRecon) addPending(wf *frameWorkReconWavefront, stats *FrameWorkTileResidualStats) {
	r.mu.Lock()
	r.pending = append(r.pending, frameThreadPendingRecon{wf: wf, stats: stats})
	r.mu.Unlock()
}

// joinWavefront joins wf's lanes when it is still pending: a batch that
// decodes several tiles reuses one scratch, whose buffered reconstruction
// events the previous tile's lanes may still read.
func (r *FrameThreadRecon) joinWavefront(wf *frameWorkReconWavefront) error {
	r.mu.Lock()
	for i := range r.pending {
		if r.pending[i].wf != wf {
			continue
		}
		p := r.pending[i]
		last := len(r.pending) - 1
		r.pending[i] = r.pending[last]
		r.pending[last] = frameThreadPendingRecon{}
		r.pending = r.pending[:last]
		r.mu.Unlock()
		r.helpUntilDone(p.wf)
		return p.join()
	}
	r.mu.Unlock()
	return nil
}

func (p frameThreadPendingRecon) join() error {
	stats := p.stats
	var local FrameWorkTileResidualStats
	if stats == nil {
		stats = &local
	}
	return p.wf.finishLanes(stats)
}

// Join waits for every tile's reconstruction lanes and folds their counters
// into the tile stats the parses returned.
func (r *FrameThreadRecon) Join() error {
	if r == nil {
		return nil
	}
	var firstErr error
	for {
		r.mu.Lock()
		n := len(r.pending)
		if n == 0 {
			r.mu.Unlock()
			return firstErr
		}
		p := r.pending[n-1]
		r.pending[n-1] = frameThreadPendingRecon{}
		r.pending = r.pending[:n-1]
		r.mu.Unlock()
		if err := p.join(); firstErr == nil && err != nil {
			firstErr = err
		}
	}
}

// runBatchOn runs one tile batch's entropy pass on frame-thread worker w
// (with its tiles' reconstruction tasks left queued) and records the first
// error.
func (r *FrameThreadRecon) runBatchOn(w int, i int) {
	r.batchWorker[i] = w
	ctx := &r.batches[i]
	if m := r.Motion; m != nil {
		m.setWorker(ctx.Jobs, w)
	}
	var err error
	if ptr, ok := r.runner.(interface{ RunPtr(*FrameWorkBatch) error }); ok {
		err = ptr.RunPtr(ctx)
	} else {
		err = r.runner.Run(*ctx)
	}
	if err != nil {
		r.parseErr.CompareAndSwap(nil, &wavefrontError{err: err})
	}
}

// QueueFrameWorkRunnerFrameThread queues a frame's tile entropy passes
// (dav1d's per-tile DAV1D_TASK_TYPE_TILE_ENTROPY tasks, one per batch) on
// the pool's frame-thread workers, where each parse queues its tiles'
// reconstruction tasks; FrameThreadRecon.WaitParse waits for the parses. The
// parses do not hold the pool workers, which the post-filter of earlier
// frames may fan row bands across at the same time.
func (p *Pool) QueueFrameWorkRunnerFrameThread(batches []Batch, jobs []tile.Job, base FrameWorkBatch, runner FrameWorkBatchRunner) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if runner == nil || base.FrameThread == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 || len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}
	ft := base.FrameThread
	if ft.tasks = p.frameTaskPool(); ft.tasks == nil {
		return ErrPoolClosed
	}
	if m := ft.Motion; m != nil {
		m.recon = ft
		if err := m.bindTiles(base, jobs); err != nil {
			return err
		}
	}
	if rows := ft.Rows; rows != nil {
		if err := rows.bindTiles(base, jobs); err != nil {
			return err
		}
	}
	if cap(ft.batches) < len(batches) {
		ft.batches = make([]FrameWorkBatch, len(batches))
		ft.batchWorker = make([]int, len(batches))
	}
	ft.batches = ft.batches[:len(batches)]
	ft.batchWorker = ft.batchWorker[:len(batches)]
	for i := range batches {
		batchJobs, err := batches[i].jobSlice(jobs)
		if err != nil {
			return err
		}
		ctx := &ft.batches[i]
		*ctx = base
		ctx.Batch = batches[i]
		ctx.Jobs = batchJobs
		// dav1d keeps one reconstruction task per tile (t->sby advances as
		// each SB row finishes): a tile reconstructs its rows in order on one
		// worker at a time, and the workers spread over the tiles and frames
		// in flight.
		ctx.WavefrontWorkers = 1
	}
	ft.runner = runner
	ft.parseErr.Store(nil)
	ft.parseWG.Add(len(batches))
	ft.entropy.reset(ft, len(batches))
	ft.entropyQueued = true
	ft.tasks.addEntropy(ft.Seq, &ft.entropy)
	return nil
}

// WaitParse waits for the entropy passes QueueFrameWorkRunnerFrameThread
// queued and reports the first error.
func (r *FrameThreadRecon) WaitParse() error {
	r.parseWG.Wait()
	r.runner = nil
	if e := r.parseErr.Load(); e != nil {
		return e.err
	}
	return nil
}

// EndEntropy lets the frames after r start their entropy passes when r's
// pass did not run (its frame failed before QueueFrameWorkRunnerFrameThread
// queued it). Frames start their passes in decode order, so every frame
// handed a Seq calls it once its pass returned.
func (r *FrameThreadRecon) EndEntropy(p *Pool) {
	if r.entropyQueued {
		return
	}
	r.entropy.reset(r, 0)
	r.entropyQueued = true
	if tp := p.frameTaskPool(); tp != nil {
		tp.addEntropy(r.Seq, &r.entropy)
	}
}

// helpUntilDone runs other frame-thread tasks on the worker of the pass
// joining wf until wf's reconstruction task left the queue.
func (r *FrameThreadRecon) helpUntilDone(wf *frameWorkReconWavefront) {
	var t *frameReconTask
	r.mu.Lock()
	for _, x := range r.reconTasks[:r.usedTasks] {
		if x.wf == wf {
			t = x
		}
	}
	r.mu.Unlock()
	if t == nil || r.tasks == nil {
		return
	}
	r.tasks.helpUntil(r.batchWorker[wf.states[0].batch.Batch.Worker], t.finished)
}

// FrameThreadMotion is the entropy-progress side of a frame-threaded frame
// (dav1d's Dav1dThreadPicture.progress[0], in 8x8 rows): Progress is the
// frame's own saved MV_REF rows; RefProgress are the rows of the reference
// grids its parse projects temporal samples from.
//
// A multi-tile frame publishes, as dav1d's ENTROPY_PROGRESS task does from
// every tile column's progress, the largest row R such that every 8x8 row
// above R is saved in every tile: each tile reports its own absolute row end
// and the frame takes the minimum over the tiles not yet finished.
type FrameThreadMotion struct {
	Progress    *FrameProgress
	RefProgress [parser.InterRefsPerFrame]*FrameProgress

	mu        sync.Mutex
	tiles     []FrameThreadMotionTile
	multi     bool
	published int32
	// recon is the frame whose passes wait here; w is the worker of a
	// single-tile frame's pass.
	recon *FrameThreadRecon
	w     int
}

// FrameThreadMotionTile is one tile's entropy progress (dav1d
// ts->progress[1]); it implements tile.FrameMotionProgress for that tile's
// block loop.
type FrameThreadMotionTile struct {
	m    *FrameThreadMotion
	rows int32
	end  int32
	used bool
	// w is the worker running the tile's pass.
	w int
}

// bindTiles registers every job's 8x8-row span before any parse runs.
func (m *FrameThreadMotion) bindTiles(base FrameWorkBatch, jobs []tile.Job) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.published = 0
	m.multi = len(jobs) > 1
	if !m.multi {
		return nil
	}
	maxTile := 0
	for i := range jobs {
		maxTile = max(maxTile, int(jobs[i].Tile))
	}
	if cap(m.tiles) <= maxTile {
		m.tiles = make([]FrameThreadMotionTile, maxTile+1)
	}
	m.tiles = m.tiles[:maxTile+1]
	clear(m.tiles)
	ctx := base
	ctx.Jobs = jobs
	ctx.geomCache = nil
	for i := range jobs {
		region, err := ctx.JobRegion(i)
		if err != nil {
			return err
		}
		t := &m.tiles[jobs[i].Tile]
		t.m = m
		t.rows = int32(region.MIRowStart >> 1)
		t.end = int32((region.MIRowEnd + 1) >> 1)
		t.used = true
	}
	return nil
}

// publishTile records that tile t saved its MV_REF rows up to rowEnd8 and
// publishes the frame's new contiguous row count when it grew.
func (m *FrameThreadMotion) publishTile(t *FrameThreadMotionTile, rowEnd8 int32) {
	m.mu.Lock()
	t.rows = rowEnd8
	v := frameThreadMotionRows(m.tiles)
	if v > m.published && v != FrameProgressDone {
		m.published = v
		if m.Progress != nil {
			m.Progress.Publish(v)
		}
	}
	m.mu.Unlock()
}

// frameThreadMotionRows is the frame's contiguous saved row count: the minimum
// absolute row end over the tiles that have not finished, or FrameProgressDone
// once all have (the frame's tail publishes Done itself).
func frameThreadMotionRows(tiles []FrameThreadMotionTile) int32 {
	v := FrameProgressDone
	for i := range tiles {
		t := &tiles[i]
		if t.used && t.rows < t.end && t.rows < v {
			v = t.rows
		}
	}
	return v
}

// WaitReferenceRows implements tile.FrameMotionProgress.
func (t *FrameThreadMotionTile) WaitReferenceRows(rowEnd8 int32) bool {
	return t.m.waitReferenceRows(t.w, rowEnd8)
}

// PublishRows implements tile.FrameMotionProgress.
func (t *FrameThreadMotionTile) PublishRows(rowEnd8 int32) {
	t.m.publishTile(t, rowEnd8)
}

// WaitReferenceRows implements tile.FrameMotionProgress.
func (m *FrameThreadMotion) WaitReferenceRows(rowEnd8 int32) bool {
	return m.waitReferenceRows(m.w, rowEnd8)
}

// waitReferenceRows waits for the references' MV_REF rows, running other
// frame-thread tasks on worker w meanwhile.
func (m *FrameThreadMotion) waitReferenceRows(w int, rowEnd8 int32) bool {
	if m.recon != nil && m.recon.tasks != nil && !m.refRowsReady(rowEnd8) {
		m.recon.tasks.helpUntil(w, func() bool { return m.refRowsReady(rowEnd8) })
	}
	for _, p := range m.RefProgress {
		if p != nil && !p.Wait(rowEnd8) {
			return false
		}
	}
	return true
}

// PublishRows implements tile.FrameMotionProgress.
func (m *FrameThreadMotion) PublishRows(rowEnd8 int32) {
	if m.Progress != nil {
		m.Progress.Publish(rowEnd8)
	}
}

// frameThreadMotionProgress returns Jobs[index]'s entropy-progress link.
func (b *FrameWorkBatch) frameThreadMotionProgress(index int) tile.FrameMotionProgress {
	if b.FrameThread == nil || b.FrameThread.Motion == nil {
		return nil
	}
	m := b.FrameThread.Motion
	if m.multi && index >= 0 && index < len(b.Jobs) {
		if t := int(b.Jobs[index].Tile); t < len(m.tiles) && m.tiles[t].used {
			return &m.tiles[t]
		}
	}
	return m
}

// reconTask returns the reconstruction task of wf's tile, one of the
// frame's tiles, with rowCount SB rows.
func (r *FrameThreadRecon) reconTask(wf *frameWorkReconWavefront, rowCount int) *frameReconTask {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.usedTasks == len(r.reconTasks) {
		r.reconTasks = append(r.reconTasks, &frameReconTask{})
	}
	t := r.reconTasks[r.usedTasks]
	r.usedTasks++
	*t = frameReconTask{wf: wf, rowCount: rowCount, needRow: -1}
	return t
}

func (m *FrameThreadMotion) refRowsReady(rowEnd8 int32) bool {
	for _, p := range m.RefProgress {
		if p != nil && !p.Ready(rowEnd8) {
			return false
		}
	}
	return true
}

// setWorker records that worker w runs the passes of jobs.
func (m *FrameThreadMotion) setWorker(jobs []tile.Job, w int) {
	if !m.multi {
		m.w = w
		return
	}
	for i := range jobs {
		if t := int(jobs[i].Tile); t < len(m.tiles) {
			m.tiles[t].w = w
		}
	}
}

func (r *FrameThreadRecon) publishEntropyOut() {
	v := FrameProgressDone
	if r.parseErr.Load() != nil {
		v = FrameProgressError
	}
	for _, p := range r.EntropyOut {
		if p != nil {
			p.Publish(v)
		}
	}
}
