package encoder

// pframe_wavefront.go runs the split pipeline's decision pass (pframe_split.go)
// concurrently over superblock rows with the top-right wavefront dependency —
// the SVT-AV1 ENC-DEC segment model (Source/Lib/Codec/enc_dec_process.c:
// enc_dec_kernel processes EncDecSegments with the top-right neighbor
// dependency enforced before each SB starts; entropy coding runs serially
// after) and the same dependency the decoder-side SB-row reconstruction
// wavefront in internal/av1/threading enforces. Entropy writing stays a
// serial replay per tile (adaptive CDFs are order-serial); search, prediction,
// transform/quantize, and reconstruction parallelize here.
//
// Shared state discipline (all proven against the serial walk by the
// byte-identity oracle):
//   - recon planes: a block reads top/left/top-right reconstructed pixels;
//     the top-right SB dependency covers every such read.
//   - mode contexts: the shared per-column above carriers are written by row r
//     only after row r+1 can no longer read them; each worker owns its left
//     context; the diagonal corner uses two parity planes (see
//     tile.BlockLoopWavefrontCarriers).
//   - motion/SAD grids, source-content and variance-partition caches: filled
//     serially before the walk or lazily by the worker that owns the SB.
//   - trial pricing contexts are frozen at frame start (trialBlockLocal), so
//     no decision depends on coding order.
//
// Workers are persistent and park on a channel between frames; jobs pass
// through owner fields, never per-frame closures.

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframeDecideWorker owns one wavefront lane's mutable state: a full copy of
// the tile's encode state (all per-block scratch is value arrays, so the copy
// is private), a block-loop scratch, and a carrier whose above/diagonal slots
// are rebound per row while the left context stays lane-owned.
type pframeDecideWorker struct {
	st      lossyEncodeState
	scratch tile.BlockLoopScratch
	carrier tile.BlockLoopContextCarrier

	// rec is the record of the row currently being decided.
	rec *pframeSplitRecord

	// Persistent callbacks (built once; close over the worker and wavefront).
	decideFn tile.PartitionDecider
	visitFn  tile.BlockLoopWriteVisitor

	wf  *pframeWavefront
	err error
}

// adopt refreshes the lane's encode state from the tile's frame-configured
// state. The struct copy shares the frame-level read-only slices (scan tables,
// motion/SAD grids, content caches — grid writes are per-SB disjoint) and
// privatizes every value-array scratch; the two mutable heap slices the
// decision pass touches (levels, invScratch) keep lane-owned backing.
func (w *pframeDecideWorker) adopt(src *lossyEncodeState) {
	levels := w.st.levels
	inv := w.st.invScratch
	w.st = *src
	if len(levels) < len(src.levels) {
		levels = make([]uint8, len(src.levels))
	}
	if len(inv) < len(src.invScratch) {
		inv = make([]int32, len(src.invScratch))
	}
	w.st.levels = levels
	w.st.invScratch = inv
	// The decision pass never writes symbols; poison the writer-side hooks so
	// a stray write path fails fast, and zero the accumulators this lane will
	// contribute back after the join.
	w.st.w = nil
	w.st.lfMap = nil
	w.st.decisionStats = nil
	w.st.afterSkipInter = nil
	w.st.afterSkipIntra = nil
	w.st.cntZeroMV = 0
	w.st.interpUsed = [3]uint32{}
}

// pframeWavefront owns the per-tile wavefront scheduling state and the parked
// worker goroutines. Job parameters live in owner fields set per frame.
type pframeWavefront struct {
	workers []pframeDecideWorker
	started int
	work    chan int
	wg      sync.WaitGroup

	carriers  tile.BlockLoopWavefrontCarriers
	doneAlloc []atomic.Int32
	done      []atomic.Int32
	nextRow   atomic.Int32

	mu       sync.Mutex
	cond     *sync.Cond
	waiters  atomic.Int32
	aborted  atomic.Bool
	launched int

	// Per-frame job (owner fields; no closures allocate per frame).
	coder           *pframeCoder
	src, ref        SourceFrame420
	golden          *SourceFrame420
	recon           *SourceFrame420
	referenceMode   parser.ReferenceMode
	walkReq         tile.BlockWalkRequest
	miCols, miRows  uint16
	rows, cols      int
	scaledReference bool
}

func (wf *pframeWavefront) ensureCond() {
	if wf.cond == nil {
		wf.cond = sync.NewCond(&wf.mu)
	}
}

func (wf *pframeWavefront) ensureDone(rows int) {
	if cap(wf.doneAlloc) < rows {
		wf.doneAlloc = make([]atomic.Int32, rows)
	}
	wf.done = wf.doneAlloc[:rows]
}

// publishRowProgress makes row's first cols superblocks visible to waiters.
// The sequentially-consistent atomic store orders every write the lane made
// before it (recon pixels, carrier slots, records) ahead of any reader that
// observes the new value, so the lock-free fast path in waitRowProgress is
// sound; the lock+broadcast runs only when someone is actually parked, which
// keeps the steady wavefront free of thundering-herd wakeups.
func (wf *pframeWavefront) publishRowProgress(row int, cols int32) {
	wf.done[row].Store(cols)
	if wf.waiters.Load() > 0 {
		wf.mu.Lock()
		wf.cond.Broadcast()
		wf.mu.Unlock()
	}
}

// wavefrontSpinBudget bounds the pre-park polling in waitRowProgress. A lane
// blocked on its upper-right neighbor typically waits a few microseconds (one
// superblock decides in ~30µs and rows drift apart), so yielding briefly
// before parking avoids the park/unpark storm that dominated the first
// wavefront profile (runtime wakep/semawakeup at ~45% of samples).
const wavefrontSpinBudget = 96

// waitRowProgress blocks until row has completed at least need SB columns.
// It returns false when the wavefront aborted.
func (wf *pframeWavefront) waitRowProgress(row int, need int32) bool {
	if wf.done[row].Load() >= need {
		return !wf.aborted.Load()
	}
	for spin := 0; spin < wavefrontSpinBudget; spin++ {
		runtime.Gosched()
		if wf.done[row].Load() >= need {
			return !wf.aborted.Load()
		}
		if wf.aborted.Load() {
			return false
		}
	}
	// Registration order matters for the lost-wakeup guarantee: the waiter
	// count rises before the parked re-check, and the publisher stores the
	// progress before loading the count, so either the publisher sees the
	// waiter or the waiter sees the progress.
	wf.waiters.Add(1)
	wf.mu.Lock()
	for wf.done[row].Load() < need {
		if wf.aborted.Load() {
			wf.mu.Unlock()
			wf.waiters.Add(-1)
			return false
		}
		wf.cond.Wait()
	}
	wf.mu.Unlock()
	wf.waiters.Add(-1)
	return !wf.aborted.Load()
}

func (wf *pframeWavefront) abort() {
	wf.mu.Lock()
	wf.aborted.Store(true)
	wf.cond.Broadcast()
	wf.mu.Unlock()
}

// ensureWorkers grows the lane set to n, building each lane's persistent
// callbacks once and spawning one parked helper goroutine per lane (the
// calling goroutine never decides — it runs the trailing entropy pass).
func (wf *pframeWavefront) ensureWorkers(n int, pc *pframeCoder) {
	if cap(wf.workers) < n {
		// Helpers are parked between frames when this runs, and they address
		// lanes through wf.workers by index per job, so relocating the slice
		// is safe — but the lanes' method-value callbacks bind the old
		// element addresses and must be rebuilt against the new ones.
		next := make([]pframeDecideWorker, n)
		copy(next, wf.workers)
		wf.workers = next
		for i := range wf.workers {
			wf.workers[i].wf = nil
			wf.workers[i].decideFn = nil
			wf.workers[i].visitFn = nil
		}
	}
	wf.workers = wf.workers[:n]
	for i := range wf.workers {
		w := &wf.workers[i]
		if w.wf == nil {
			w.wf = wf
			w.decideFn = w.decidePartition
			w.visitFn = w.decideVisit
		}
	}
	if wf.work == nil {
		wf.work = make(chan int, 16)
	}
	for wf.started < n {
		wf.started++
		go func() {
			for idx := range wf.work {
				wf.runWorker(idx)
				wf.wg.Done()
			}
		}()
	}
	wf.coder = pc
}

// decidePartition is the lane's partition decider: the shared content-only
// decision plus the per-row partition record append (write-pass replay input).
func (w *pframeDecideWorker) decidePartition(level tile.BlockLevel, ctx int, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
	wf := w.wf
	partition, err := w.st.decideRealtimePartition(wf.src, wf.ref, wf.scaledReference, level, miCol, miRow, haveRight, haveBottom)
	if err == nil {
		w.rec.appendPart(partition)
	}
	return partition, err
}

// decideVisit is the lane's leaf visitor: the split pipeline's decision pass
// into the row's record.
func (w *pframeDecideWorker) decideVisit(block tile.BlockVisit, scratch *tile.BlockLoopScratch) error {
	wf := w.wf
	return w.st.decidePBlock(wf.src, wf.ref, wf.golden, wf.recon, block, scratch, wf.referenceMode, wf.walkReq, wf.miCols, wf.miRows, w.rec)
}

// runWorker pulls SB rows in claim order until none remain; rows are claimed
// monotonically, so a claimed row's predecessor is always claimed and
// progressing (row 0 never waits) — the wavefront cannot deadlock.
func (wf *pframeWavefront) runWorker(idx int) {
	w := &wf.workers[idx]
	for {
		if wf.aborted.Load() {
			return
		}
		row := int(wf.nextRow.Add(1)) - 1
		if row >= wf.rows {
			return
		}
		if err := wf.runRow(w, row); err != nil {
			w.err = err
			wf.abort()
			return
		}
		if wf.aborted.Load() {
			return
		}
	}
}

// runRow decides one SB row left to right under the top-right dependency:
// before SB (row, c) starts, row-1 must have stored through column c+1 (the
// dependency the decoder recon wavefront and SVT's enc_dec segment sync both
// enforce).
func (wf *pframeWavefront) runRow(w *pframeDecideWorker, row int) error {
	w.rec = &wf.coder.splitRowRecs[row]
	wf.carriers.BindRow(&w.carrier, row)
	rootSize := uint32(16)
	miRow := uint32(wf.walkReq.MIRowStart) + uint32(row)*rootSize
	miColStart := uint32(wf.walkReq.MIColStart)
	for c := 0; c < wf.cols; c++ {
		if row > 0 {
			need := int32(c + 2)
			if need > int32(wf.cols) {
				need = int32(wf.cols)
			}
			if !wf.waitRowProgress(row-1, need) {
				return nil // aborted; the failing lane carries the error
			}
		}
		miCol := miColStart + uint32(c)*rootSize
		if err := tile.WalkBlockLoopDecideSB(&w.scratch, &w.carrier, wf.walkReq, 16, miRow, miCol, w.decideFn, w.visitFn); err != nil {
			return err
		}
		wf.publishRowProgress(row, int32(c+1))
	}
	return nil
}

// start launches the decision pass over the tile with n lanes (n >= 2), all
// running on parked helper goroutines: the calling goroutine stays free to run
// the serial entropy pass concurrently, consuming rows as they complete (the
// SVT shape — entropy_coding_process picks up EncDecSegments row results as
// enc_dec_kernel finishes them). The caller must have finished the tile's
// frame preparation (grids, content caches, rate tables) on pc.st before
// calling; every lane adopts that state here.
func (wf *pframeWavefront) start(pc *pframeCoder, n int, src, ref SourceFrame420, golden *SourceFrame420, recon *SourceFrame420,
	referenceMode parser.ReferenceMode, walkReq tile.BlockWalkRequest, miCols, miRows uint16, rows, cols int, scaledReference bool) {
	wf.ensureCond()
	wf.ensureWorkers(n, pc)
	wf.ensureDone(rows)
	wf.carriers.Reset(cols)
	wf.src, wf.ref, wf.golden, wf.recon = src, ref, golden, recon
	wf.referenceMode = referenceMode
	wf.walkReq = walkReq
	wf.miCols, wf.miRows = miCols, miRows
	wf.rows, wf.cols = rows, cols
	wf.scaledReference = scaledReference
	wf.nextRow.Store(0)
	wf.aborted.Store(false)
	for r := 0; r < rows; r++ {
		wf.done[r].Store(0)
	}
	for i := range wf.workers {
		w := &wf.workers[i]
		w.err = nil
		w.adopt(&pc.st)
	}
	wf.launched = len(wf.workers)
	wf.wg.Add(wf.launched)
	for i := 0; i < wf.launched; i++ {
		wf.work <- i
	}
}

// waitRowDecided gates the trailing entropy pass: it blocks until row's
// decision records are complete, returning an error if the wavefront aborted.
func (wf *pframeWavefront) waitRowDecided(row int) error {
	if !wf.waitRowProgress(row, int32(wf.cols)) {
		return fmt.Errorf("wavefront decide aborted")
	}
	return nil
}

// join waits for every decide lane to park and merges the lanes' zero-motion /
// filter-usage accumulators back into pc.st.
func (wf *pframeWavefront) join(pc *pframeCoder) error {
	wf.wg.Wait()
	var firstErr error
	for i := range wf.workers {
		w := &wf.workers[i]
		if w.err != nil && firstErr == nil {
			firstErr = w.err
		}
		pc.st.cntZeroMV += w.st.cntZeroMV
		for f := range pc.st.interpUsed {
			pc.st.interpUsed[f] += w.st.interpUsed[f]
		}
	}
	if firstErr != nil {
		return fmt.Errorf("wavefront decide: %w", firstErr)
	}
	return nil
}
