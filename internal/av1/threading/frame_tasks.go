// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import "sync"

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
type frameTaskPool struct {
	mu     sync.Mutex
	cond   sync.Cond
	tasks  []frameTaskEntry
	parked int
	closed bool
}

type frameTaskEntry struct {
	seq  uint64
	task FrameTask
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
	tp.mu.Lock()
	i := len(tp.tasks)
	for i > 0 && tp.tasks[i-1].seq > seq {
		i--
	}
	tp.tasks = append(tp.tasks, frameTaskEntry{})
	copy(tp.tasks[i+1:], tp.tasks[i:])
	tp.tasks[i] = frameTaskEntry{seq: seq, task: task}
	if tp.parked > 0 {
		tp.cond.Signal()
	}
	tp.mu.Unlock()
}

// wake tells a parked worker that progress was published.
func (tp *frameTaskPool) wake() {
	if tp == nil {
		return
	}
	tp.mu.Lock()
	if tp.parked > 0 {
		tp.cond.Signal()
	}
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
	for i := 0; i < len(tp.tasks); i++ {
		t := tp.tasks[i].task
		step, ok, gone := t.Claim()
		if gone {
			tp.remove(t)
			i--
			continue
		}
		if ok {
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
		t, step := tp.next()
		if t == nil {
			tp.parked++
			tp.cond.Wait()
			tp.parked--
			continue
		}
		// More tasks may be ready: pass the wake on (dav1d re-signals the
		// task cond once it found a task).
		if tp.parked > 0 {
			tp.cond.Signal()
		}
		tp.mu.Unlock()
		t.Run(w, step)
		tp.mu.Lock()
		if t.Finish(step) {
			tp.remove(t)
		}
	}
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

// Finish implements FrameTask.
func (t *frameReconTask) Finish(int) bool {
	t.running = false
	t.view = frameWorkReconView{}
	t.row++
	if t.ended || t.row >= t.rowCount {
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

// WakeFrameTasks tells the frame-thread workers that progress a task waits
// on was published.
func (p *Pool) WakeFrameTasks() {
	p.frameTaskPool().wake()
}
