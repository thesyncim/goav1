// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"sync"
	"sync/atomic"
)

// FrameTask is one queued task of the frame-threaded decode's worker pool
// (a Dav1dTask). Claim and Finish run with the pool locked; Run runs the
// claimed step on worker w (0 <= w < the pool's worker count) unlocked and
// must not block on another task.
type FrameTask interface {
	// Claim reports a step whose dependencies are met, or ok false; gone
	// removes the task from the queue.
	Claim() (step int, ok bool, gone bool)
	Run(w int, step int)
	// Finish records that step ran; gone removes the task from the queue.
	Finish(step int) (gone bool)
}

// frameTaskPool is the frame-threaded decode's worker pool, dav1d's
// TaskThreadData (src/thread_task.c): a fixed set of worker goroutines shared
// by every frame in flight. A worker takes the first ready task itself,
// scanning the frames oldest first, runs it, and scans again; it parks only
// when no task is ready and is woken when progress that a task waits on is
// published. A task never blocks: its dependencies are checked before it is
// taken (check_tile and the filter tasks' progress checks), and the tiles'
// entropy passes decode one superblock row per step, re-queued when the next
// row's dependencies are not met.
type frameTaskPool struct {
	mu     sync.Mutex
	cond   sync.Cond
	tasks  []frameTaskEntry
	parked int
	closed bool
	// signaled is dav1d's ttd->cond_signaled: set once a worker is signaled,
	// cleared by every worker before it scans the queue, so a publish while
	// a signaled worker has yet to scan sends no further signal.
	signaled atomic.Bool
}

type frameTaskEntry struct {
	seq  uint64
	task FrameTask
	init bool
}

func newFrameTaskPool(workers int) *frameTaskPool {
	tp := &frameTaskPool{}
	tp.cond.L = &tp.mu
	for w := range workers {
		go tp.worker(w)
	}
	return tp
}

// add queues task behind the tasks of its frame and every older frame
// (insert_tasks keeps the list in frame order).
func (tp *frameTaskPool) add(seq uint64, task FrameTask) {
	tp.insert(frameTaskEntry{seq: seq, task: task})
}

func (tp *frameTaskPool) insert(e frameTaskEntry) {
	tp.mu.Lock()
	i := len(tp.tasks)
	for i > 0 && tp.tasks[i-1].seq > e.seq {
		i--
	}
	tp.tasks = append(tp.tasks, frameTaskEntry{})
	copy(tp.tasks[i+1:], tp.tasks[i:])
	tp.tasks[i] = e
	tp.signalLocked()
	tp.mu.Unlock()
}

// signalLocked wakes a parked worker.
func (tp *frameTaskPool) signalLocked() {
	if tp.parked > 0 {
		tp.signaled.Store(true)
		tp.cond.Signal()
	}
}

// wake tells a parked worker that progress was published. A worker clears
// signaled before each scan, so while it is set a worker is still to scan
// the queue after this publish and the signal can be skipped.
func (tp *frameTaskPool) wake() {
	if tp == nil || tp.signaled.Load() {
		return
	}
	tp.mu.Lock()
	tp.signalLocked()
	tp.mu.Unlock()
}

func (tp *frameTaskPool) close() {
	tp.mu.Lock()
	tp.closed = true
	tp.cond.Broadcast()
	tp.mu.Unlock()
}

func (tp *frameTaskPool) remove(task FrameTask) {
	for i := range tp.tasks {
		if tp.tasks[i].task == task {
			copy(tp.tasks[i:], tp.tasks[i+1:])
			tp.tasks[len(tp.tasks)-1] = frameTaskEntry{}
			tp.tasks = tp.tasks[:len(tp.tasks)-1]
			return
		}
	}
}

// next claims the first ready step in queue order. Callers hold mu.
func (tp *frameTaskPool) next() (FrameTask, int) {
	// dav1d runs the frames' init tasks before any decoding task.
	for i := 0; i < len(tp.tasks); i++ {
		e := tp.tasks[i]
		if !e.init {
			continue
		}
		if step, ok, _ := e.task.Claim(); ok {
			tp.remove(e.task)
			return e.task, step
		}
	}
	for i := 0; i < len(tp.tasks); i++ {
		e := tp.tasks[i]
		if e.init {
			continue
		}
		step, ok, gone := e.task.Claim()
		if gone {
			tp.remove(e.task)
			i--
			continue
		}
		if ok {
			return e.task, step
		}
	}
	return nil, 0
}

func (tp *frameTaskPool) worker(w int) {
	tp.mu.Lock()
	for {
		if tp.closed {
			tp.mu.Unlock()
			return
		}
		tp.signaled.Store(false)
		t, step := tp.next()
		if t == nil {
			tp.parked++
			tp.cond.Wait()
			tp.parked--
			continue
		}
		// More tasks may be ready: pass the wake on (dav1d re-signals the task
		// cond once it found a task).
		tp.signalLocked()
		tp.mu.Unlock()
		t.Run(w, step)
		tp.mu.Lock()
		if t.Finish(step) {
			tp.remove(t)
		}
	}
}

// frameEntropyTask is a frame's tile entropy passes, one step per batch
// (DAV1D_TASK_TYPE_TILE_ENTROPY): a step decodes the batch's superblock rows
// for as long as the next one is ready (dav1d's found_unlocked) and leaves
// the batch queued, to be claimed again once its next row is ready, until
// its last row.
type frameEntropyTask struct {
	ft *FrameThreadRecon
	// state is each batch's entropyIdle, entropyRunning or entropyDone; left
	// counts the batches not done (guarded by the pool lock). ended is set by
	// the step that ran a batch's last row.
	state []uint8
	ended []bool
	left  int
	ran   atomic.Int32
}

const (
	entropyIdle uint8 = iota
	entropyRunning
	entropyDone
)

func (t *frameEntropyTask) reset(ft *FrameThreadRecon, n int) {
	t.ft = ft
	if cap(t.state) < n {
		t.state = make([]uint8, n)
		t.ended = make([]bool, n)
	}
	t.state = t.state[:n]
	t.ended = t.ended[:n]
	clear(t.state)
	clear(t.ended)
	t.left = n
	t.ran.Store(0)
}

// Claim implements FrameTask.
func (t *frameEntropyTask) Claim() (int, bool, bool) {
	for i, st := range t.state {
		if st == entropyIdle && t.ft.batchRowReady(i, true) {
			t.state[i] = entropyRunning
			return i, true, false
		}
	}
	return 0, false, t.left == 0
}

// Run implements FrameTask.
func (t *frameEntropyTask) Run(_ int, i int) {
	if !t.ft.runBatchRows(i) {
		return
	}
	t.ended[i] = true
	if int(t.ran.Add(1)) == len(t.state) {
		// The worker that ends the pass publishes its outputs final, as
		// dav1d's ENTROPY_PROGRESS task sets sr_cur.progress[0].
		t.ft.publishEntropyOut()
	}
}

// Finish implements FrameTask.
func (t *frameEntropyTask) Finish(i int) bool {
	if !t.ended[i] {
		t.state[i] = entropyIdle
		return false
	}
	t.state[i] = entropyDone
	t.left--
	t.ft.parseWG.Done()
	return t.left == 0
}

// frameReconTask is one tile's reconstruction task (DAV1D_TASK_TYPE_TILE_
// RECONSTRUCTION): row is the next SB row to reconstruct, taken once the
// parse published it and the references published the rows it reads (the
// row's lowest pixels, recorded by the parse in wf.rowNeed).
type frameReconTask struct {
	wf       *frameWorkReconWavefront
	row      int
	rowCount int
	running  bool
	ended    bool
	gone     bool
	view     frameWorkReconView
}

// Claim implements FrameTask.
func (t *frameReconTask) Claim() (int, bool, bool) {
	if t.running {
		return 0, false, false
	}
	wf := t.wf
	p := wf.progress
	p.mu.Lock()
	aborted := p.aborted.Load()
	published := wf.published
	view := wf.pub
	p.mu.Unlock()
	if aborted {
		t.gone = true
		wf.wg.Done()
		return 0, false, true
	}
	if published <= t.row {
		return 0, false, false
	}
	if need := &wf.rowNeed[t.row]; need.any && !wf.frameThread.referencesReady(&wf.states[0].batch, &need.rows) {
		return 0, false, false
	}
	t.running = true
	t.view = view
	return t.row, true, false
}

// Run implements FrameTask.
func (t *frameReconTask) Run(_ int, row int) {
	t.ended = t.wf.runTaskRow(row, t.rowCount, &t.view)
}

// Finish implements FrameTask.
func (t *frameReconTask) Finish(int) bool {
	t.running = false
	t.view = frameWorkReconView{}
	t.row++
	if t.ended || t.row >= t.rowCount {
		t.gone = true
		t.wf.wg.Done()
		return true
	}
	return false
}

// runTaskRow reconstructs SB row row of a frame-threaded tile and publishes
// it; it reports true when the tile's task ends early (an error or abort).
func (wf *frameWorkReconWavefront) runTaskRow(row int, rowCount int, view *frameWorkReconView) bool {
	st := &wf.states[0]
	ft := wf.frameThread
	if err := wf.reconstructRow(st, view, row, rowCount); err != nil {
		wf.firstErr.CompareAndSwap(nil, &wavefrontError{err: err})
		wf.abortProgress()
		if ft.Rows != nil {
			ft.Rows.Fail()
		}
		return true
	}
	if wf.progress.aborted.Load() {
		return true
	}
	if ft.Rows != nil {
		b := &wf.build
		ft.Rows.finishRows(int(wf.reconTile), int32(b.miRowStart+(row+1)*b.sbSizeMIB)*4)
	}
	return false
}

// AddFrameTask queues task, a task of the frame decoded seq-th, on the
// pool's frame-thread workers; it reports false once the pool is closed.
func (p *Pool) AddFrameTask(seq uint64, task FrameTask) bool {
	tp := p.frameTaskPool()
	if tp == nil {
		return false
	}
	tp.add(seq, task)
	return true
}

// AddFrameInitTask queues the init task of the frame decoded seq-th
// (DAV1D_TASK_TYPE_INIT / INIT_CDF): it is taken, once, before any decoding
// task as soon as its Claim reports it ready, and leaves the queue then.
func (p *Pool) AddFrameInitTask(seq uint64, task FrameTask) bool {
	tp := p.frameTaskPool()
	if tp == nil {
		return false
	}
	tp.insert(frameTaskEntry{seq: seq, task: task, init: true})
	return true
}

// RemoveFrameTask takes task out of the frame-thread queue if it is still
// queued, so its owner may reuse it.
func (p *Pool) RemoveFrameTask(task FrameTask) {
	tp := p.frameTaskPool()
	if tp == nil {
		return
	}
	tp.mu.Lock()
	tp.remove(task)
	tp.mu.Unlock()
}

// WakeFrameTasks tells the frame-thread workers that progress a task waits
// on was published.
func (p *Pool) WakeFrameTasks() {
	p.frameTaskPool().wake()
}
