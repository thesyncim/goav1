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
// split. dav1d's frame threading runs a frame's tile entropy pass
// (DAV1D_TASK_TYPE_TILE_ENTROPY) separately from its reconstruction
// (DAV1D_TASK_TYPE_TILE_RECONSTRUCTION, src/thread_task.c): the entropy pass of
// frame N+1 starts as soon as frame N's parse outputs are published, while N's
// reconstruction and post-filter still run. A batch carrying a FrameThreadRecon
// always reconstructs through the SB-row wavefront lanes (the frame-thread
// buffers, ts->frame_thread[p].cf in dav1d), returns from the tile job once the
// parse finished, and leaves the lanes running; Join waits for them.
//
// Before a lane reconstructs its first row it waits until every reference
// surface it may predict from is final (RefProgress), so the entropy pass never
// waits for pixels.
type FrameThreadRecon struct {
	RefProgress [parser.InterRefsPerFrame]*FrameProgress

	wf    *frameWorkReconWavefront
	stats *FrameWorkTileResidualStats

	// batch is the tile job's context, kept here so the frame-thread dispatch
	// hands the runner a pointer without a per-frame heap copy.
	batch FrameWorkBatch
}

// Reset clears the frame's reference waits and any pending join.
func (r *FrameThreadRecon) Reset() {
	r.RefProgress = [parser.InterRefsPerFrame]*FrameProgress{}
	r.wf = nil
	r.stats = nil
}

// Pending reports whether reconstruction lanes are still owned by the frame.
func (r *FrameThreadRecon) Pending() bool {
	return r != nil && r.wf != nil
}

// Join waits for the frame's reconstruction lanes and folds their counters
// into the tile stats the parse returned.
func (r *FrameThreadRecon) Join() error {
	if r == nil || r.wf == nil {
		return nil
	}
	wf := r.wf
	stats := r.stats
	r.wf = nil
	r.stats = nil
	var local FrameWorkTileResidualStats
	if stats == nil {
		stats = &local
	}
	return wf.finishLanes(stats)
}

// waitReferences blocks until every reference surface is final.
func (r *FrameThreadRecon) waitReferences() bool {
	for _, p := range r.RefProgress {
		if p != nil && !p.Wait(FrameProgressDone) {
			return false
		}
	}
	return true
}

// ExecuteFrameWorkRunnerFrameThread runs a single-tile frame's tile job on the
// calling goroutine while its SB-row reconstruction lanes run on the pool's
// persistent lane group. It is the frame-threading dispatch: the parse must not
// hold the pool, because the post-filter of the previous frame fans its row
// bands across the pool workers at the same time. laneNeed is the total number
// of lanes that may be live at once across every in-flight frame, so a lane
// send always finds an idle goroutine.
func (p *Pool) ExecuteFrameWorkRunnerFrameThread(batch Batch, jobs []tile.Job, base FrameWorkBatch, runner FrameWorkBatchRunner, laneNeed int) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if runner == nil || base.FrameThread == nil {
		return ErrInvalidCallback
	}
	batches := [1]Batch{batch}
	if err := validateBatches(batches[:], jobs, len(p.workers)); err != nil {
		return err
	}
	batchJobs, err := batch.jobSlice(jobs)
	if err != nil {
		return err
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}
	lanes := p.wavefrontLaneGroupLocked(laneNeed)
	p.mu.Unlock()

	ctx := &base.FrameThread.batch
	*ctx = base
	ctx.Batch = batch
	ctx.Jobs = batchJobs
	ctx.WavefrontWorkers = uint16(min(len(p.workers), int(^uint16(0))))
	ctx.wavefrontLanes = lanes
	if ptr, ok := runner.(interface{ RunPtr(*FrameWorkBatch) error }); ok {
		return ptr.RunPtr(ctx)
	}
	return runner.Run(*ctx)
}
