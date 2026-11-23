// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import "sync"

// frameTaskPool is the frame-threaded decode's worker pool, dav1d's
// TaskThreadData (src/thread_task.c): a fixed set of worker goroutines shared
// by every frame in flight. A worker takes the first ready task itself,
// scanning the frames oldest first, runs it, and scans again; it parks only
// when no task is ready and is woken when progress that a task waits on is
// published. A task never blocks: its dependencies are checked before it is
// taken (check_tile).
//
// The tasks are the tiles' reconstruction tasks (DAV1D_TASK_TYPE_TILE_
// RECONSTRUCTION): one per tile, advancing one SB row each time it runs.
type frameTaskPool struct {
	mu      sync.Mutex
	cond    sync.Cond
	tiles   []*frameReconTask
	parked  int
	workers int
	closed  bool
}

// frameReconTask is one tile's reconstruction task: row is the next SB row
// to reconstruct. need caches the reference rows row reads (needRow is the
// row it was computed for).
type frameReconTask struct {
	wf       *frameWorkReconWavefront
	seq      uint64
	tile     int32
	row      int
	rowCount int
	running  bool
	needRow  int
	need     frameThreadRefNeed
	needAny  bool
}

func newFrameTaskPool(workers int) *frameTaskPool {
	tp := &frameTaskPool{workers: workers}
	tp.cond.L = &tp.mu
	for range workers {
		go tp.worker()
	}
	return tp
}

// add queues wf's reconstruction task in frame order, then tile order
// (insert_tasks).
func (tp *frameTaskPool) add(t *frameReconTask) {
	tp.mu.Lock()
	i := len(tp.tiles)
	for i > 0 && (tp.tiles[i-1].seq > t.seq || tp.tiles[i-1].seq == t.seq && tp.tiles[i-1].tile > t.tile) {
		i--
	}
	tp.tiles = append(tp.tiles, nil)
	copy(tp.tiles[i+1:], tp.tiles[i:])
	tp.tiles[i] = t
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

func (tp *frameTaskPool) remove(i int) {
	copy(tp.tiles[i:], tp.tiles[i+1:])
	tp.tiles[len(tp.tiles)-1] = nil
	tp.tiles = tp.tiles[:len(tp.tiles)-1]
}

// next returns the first ready task, marked running, and the parsed rows it
// reconstructs from. Tasks of aborted tiles leave the queue. Callers hold mu.
func (tp *frameTaskPool) next() (*frameReconTask, frameWorkReconView) {
	for i := 0; i < len(tp.tiles); i++ {
		t := tp.tiles[i]
		if t.running {
			continue
		}
		wf := t.wf
		p := wf.progress
		p.mu.Lock()
		aborted := p.aborted.Load()
		published := wf.published
		view := wf.pub
		p.mu.Unlock()
		if aborted {
			tp.remove(i)
			i--
			wf.wg.Done()
			continue
		}
		if published <= t.row {
			continue
		}
		if ft := wf.frameThread; ft != nil {
			if t.needRow != t.row {
				t.need, t.needAny = rowReferenceNeed(&view, t.row)
				t.needRow = t.row
			}
			if t.needAny && !ft.referencesReady(&wf.states[0].batch, &t.need) {
				continue
			}
		}
		t.running = true
		return t, view
	}
	return nil, frameWorkReconView{}
}

func (tp *frameTaskPool) worker() {
	tp.mu.Lock()
	for {
		if tp.closed {
			tp.mu.Unlock()
			return
		}
		t, view := tp.next()
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
		done := t.wf.runTaskRow(t.row, t.rowCount, &view)
		tp.mu.Lock()
		t.running = false
		t.row++
		if done || t.row >= t.rowCount {
			for i, x := range tp.tiles {
				if x == t {
					tp.remove(i)
					break
				}
			}
			t.wf.wg.Done()
		}
	}
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
