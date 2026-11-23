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
// taken (check_tile and the filter tasks' progress checks).
//
// The tiles' entropy passes are the one exception: a tile's block loop runs
// to the end of the tile in one call and waits inside for the references'
// MV_REF rows (dav1d instead re-queues the entropy task per sbrow). Frames
// therefore start their entropy passes in decode order (nextEntropy): every
// pass waits only on older frames' passes, which all hold a worker, so the
// oldest one always progresses. A worker whose pass waits runs other ready
// tasks meanwhile (helpUntil), so the reconstruction and post-filter tasks
// the older frames' pixels depend on keep running.
type frameTaskPool struct {
	mu      sync.Mutex
	cond    sync.Cond
	tasks   []frameTaskEntry
	parked  int
	helpers int
	// waiting counts the entropy passes inside helpUntil, of workers.
	waiting int
	workers int
	closed  bool
	// nextEntropy is the decode order of the frame whose entropy batches
	// may be taken next.
	nextEntropy uint64
}

type frameTaskEntry struct {
	seq     uint64
	task    FrameTask
	entropy *frameEntropyTask
	init    bool
}

func newFrameTaskPool(workers int) *frameTaskPool {
	tp := &frameTaskPool{nextEntropy: 1, workers: workers}
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

// signalLocked wakes a parked worker, or every waiting one while a pass
// waits in helpUntil (its condition may be what changed).
func (tp *frameTaskPool) signalLocked() {
	if tp.helpers > 0 {
		tp.cond.Broadcast()
	} else if tp.parked > 0 {
		tp.cond.Signal()
	}
}

// wake tells a parked worker that progress was published.
func (tp *frameTaskPool) wake() {
	if tp == nil {
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

// next claims the first ready step in queue order, skipping entropy batches
// when helping. Callers hold mu.
func (tp *frameTaskPool) next(helping bool) (FrameTask, int) {
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
		t := e.task
		if e.init {
			continue
		}
		if e.entropy != nil {
			if helping || e.seq != tp.nextEntropy {
				continue
			}
			if e.entropy.n == 0 {
				// A frame whose pass did not run: the next frame may start.
				tp.nextEntropy++
				tp.remove(t)
				i = -1
				continue
			}
		}
		step, ok, gone := t.Claim()
		if gone {
			tp.remove(t)
			i--
			// A pass may be waiting in helpUntil for this task to leave.
			if tp.helpers > 0 {
				tp.cond.Broadcast()
			}
			continue
		}
		if ok {
			if e.entropy != nil && e.entropy.claimed == e.entropy.n {
				// Every batch of the frame holds a worker: the next frame's
				// pass may start.
				tp.nextEntropy++
			}
			return t, step
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
		t, step := tp.next(false)
		if t == nil {
			tp.parked++
			tp.cond.Wait()
			tp.parked--
			continue
		}
		tp.runLocked(w, t, step)
	}
}

// runLocked runs a claimed step with mu released.
func (tp *frameTaskPool) runLocked(w int, t FrameTask, step int) {
	// More tasks may be ready: pass the wake on (dav1d re-signals the task
	// cond once it found a task).
	tp.signalLocked()
	tp.mu.Unlock()
	t.Run(w, step)
	tp.mu.Lock()
	if t.Finish(step) {
		tp.remove(t)
	}
	if tp.helpers > 0 {
		tp.cond.Broadcast()
	}
}

// helpUntil runs other ready tasks on worker w, an entropy pass's worker,
// until done (called with mu held) reports true.
func (tp *frameTaskPool) helpUntil(w int, done func() bool) {
	tp.mu.Lock()
	tp.waiting++
	if tp.waiting == tp.workers {
		tp.cond.Broadcast()
	}
	for !done() && !tp.closed {
		// While another worker is free to run the other tasks, the pass
		// only waits, so it resumes as soon as its rows are in; once every
		// worker's pass waits, they run the other tasks themselves.
		var t FrameTask
		var step int
		if tp.waiting == tp.workers {
			t, step = tp.next(true)
		}
		if t == nil {
			tp.helpers++
			tp.cond.Wait()
			tp.helpers--
			continue
		}
		tp.runLocked(w, t, step)
	}
	tp.waiting--
	tp.mu.Unlock()
}

// frameEntropyTask is a frame's tile entropy passes, one step per batch
// (DAV1D_TASK_TYPE_TILE_ENTROPY).
type frameEntropyTask struct {
	ft                 *FrameThreadRecon
	n, claimed, closed int
	ran                atomic.Int32
}

func (t *frameEntropyTask) reset(ft *FrameThreadRecon, n int) {
	t.ft, t.n, t.claimed, t.closed = ft, n, 0, 0
	t.ran.Store(0)
}

// Claim implements FrameTask.
func (t *frameEntropyTask) Claim() (int, bool, bool) {
	if t.claimed == t.n {
		return 0, false, false
	}
	t.claimed++
	return t.claimed - 1, true, false
}

// Run implements FrameTask.
func (t *frameEntropyTask) Run(w int, i int) {
	t.ft.runBatchOn(w, i)
	if int(t.ran.Add(1)) == t.n {
		// The worker that ends the pass publishes its outputs final, as
		// dav1d's ENTROPY_PROGRESS task sets sr_cur.progress[0].
		t.ft.publishEntropyOut()
	}
}

// Finish implements FrameTask.
func (t *frameEntropyTask) Finish(int) bool {
	t.closed++
	t.ft.parseWG.Done()
	return t.closed == t.n
}

// addEntropy queues the frame decoded seq-th's entropy batches; n is zero
// for a frame whose pass will not run, which only lets the next frame start.
func (tp *frameTaskPool) addEntropy(seq uint64, t *frameEntropyTask) {
	tp.insert(frameTaskEntry{seq: seq, task: t, entropy: t})
}

// frameReconTask is one tile's reconstruction task (DAV1D_TASK_TYPE_TILE_
// RECONSTRUCTION): row is the next SB row to reconstruct, taken once the
// parse published it and the references published the rows it reads. need
// caches those rows (needRow is the row they were computed for).
type frameReconTask struct {
	wf       *frameWorkReconWavefront
	row      int
	rowCount int
	running  bool
	ended    bool
	gone     bool
	needRow  int
	need     frameThreadRefNeed
	needAny  bool
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
	if t.needRow != t.row {
		t.need, t.needAny = rowReferenceNeed(&view, t.row)
		t.needRow = t.row
	}
	if t.needAny && !wf.frameThread.referencesReady(&wf.states[0].batch, &t.need) {
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

// finished reports whether the task left the queue (called with the pool
// locked).
func (t *frameReconTask) finished() bool {
	return t.gone
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

// WakeFrameTasks tells the frame-thread workers that progress a task waits
// on was published.
func (p *Pool) WakeFrameTasks() {
	p.frameTaskPool().wake()
}
