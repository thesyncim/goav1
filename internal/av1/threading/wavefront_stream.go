// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// frameWorkReconView is a published snapshot of the deferred reconstruction
// buffers: slice headers taken by the parsing goroutine after it finished a
// prefix of SB rows. While a tile parses, the buffers are only appended to, and
// the one element patched after it was appended (the closing SB span's end, and
// a palette visit's map pointers) is patched before its row is published. Every
// element below a published length therefore stays valid and unchanged, even
// when a later append moves the live slice to a new backing array: the snapshot
// keeps reading the old one.
type frameWorkReconView struct {
	rowStart []int32
	sbs      []frameWorkReconSB
	events   []frameWorkReconEvent
	visits   []tile.BlockLoopVisit
	blocks   []tile.BlockCoeffBlock
	cf       []frameWorkReconCF
}

// frameWorkReconSpanBuilder is the parse-side cursor that partitions the flat
// event list into per-SB spans and SB rows as block-begin events arrive. Only
// the parsing goroutine touches it.
type frameWorkReconSpanBuilder struct {
	miRowStart    int
	miColStart    int
	sbSizeMIB     int
	rowCount      int
	prevRow       int
	prevCol       int
	paletteCursor int
}

func (wf *frameWorkReconWavefront) beginSpans(region FrameWorkJobRegion, sbSizeMIB int) {
	wf.rowStart = wf.rowStart[:0]
	wf.sbSpans = wf.sbSpans[:0]
	wf.build = frameWorkReconSpanBuilder{
		miRowStart: int(region.MIRowStart),
		miColStart: int(region.MIColStart),
		sbSizeMIB:  sbSizeMIB,
		rowCount:   int(region.SBRows),
		prevRow:    -1,
		prevCol:    -1,
	}
}

// noteBlockBegin records that the block-begin event at eventIndex opens the
// block at (miRow, miCol). When the block starts a new superblock it closes the
// previous superblock's span at eventIndex and opens a new one; when it starts
// a new SB row R it returns R, the number of leading rows whose events are now
// all recorded, and -1 otherwise. Events must arrive in SB-raster order for the
// wavefront gate to be valid; anything else is rejected.
func (wf *frameWorkReconWavefront) noteBlockBegin(eventIndex int, miRow int, miCol int) (int, error) {
	b := &wf.build
	if miRow < b.miRowStart || miCol < b.miColStart || b.sbSizeMIB <= 0 {
		return -1, ErrInvalidBatch
	}
	sbRow := (miRow - b.miRowStart) / b.sbSizeMIB
	sbCol := (miCol - b.miColStart) / b.sbSizeMIB
	if sbRow >= b.rowCount {
		return -1, ErrInvalidBatch
	}
	if sbRow == b.prevRow && sbCol == b.prevCol {
		return -1, nil
	}
	if sbRow < b.prevRow || (sbRow == b.prevRow && sbCol < b.prevCol) {
		return -1, ErrInvalidBatch
	}
	idx, ok := frameWorkReconIndex(eventIndex)
	if !ok {
		return -1, ErrInvalidBatch
	}
	col, ok := frameWorkReconWavefrontCol(sbCol)
	if !ok {
		return -1, ErrInvalidBatch
	}
	if n := len(wf.sbSpans); n > 0 {
		wf.sbSpans[n-1].end = idx
	}
	for len(wf.rowStart) <= sbRow {
		if err := wf.appendRowStart(len(wf.sbSpans)); err != nil {
			return -1, err
		}
	}
	wf.sbSpans = append(wf.sbSpans, frameWorkReconSB{start: idx, end: idx, col: col})
	newRow := sbRow != b.prevRow
	b.prevRow, b.prevCol = sbRow, sbCol
	if newRow {
		return sbRow, nil
	}
	return -1, nil
}

// closeSpans ends the last superblock at eventCount and opens the row starts
// of every remaining row, so rowStart holds rowCount+1 entries.
func (wf *frameWorkReconWavefront) closeSpans(eventCount int) error {
	idx, ok := frameWorkReconIndex(eventCount)
	if !ok {
		return ErrInvalidBatch
	}
	if n := len(wf.sbSpans); n > 0 {
		wf.sbSpans[n-1].end = idx
	}
	for len(wf.rowStart) <= wf.build.rowCount {
		if err := wf.appendRowStart(len(wf.sbSpans)); err != nil {
			return err
		}
	}
	return nil
}

// publishRows makes the first rows SB rows of view available to the lanes.
func (wf *frameWorkReconWavefront) publishRows(rows int, view frameWorkReconView) {
	p := wf.ensureProgress()
	p.mu.Lock()
	wf.pub = view
	wf.published = rows
	p.cond.Broadcast()
	p.mu.Unlock()
}

// waitRows blocks until at least need SB rows are published and returns the
// latest snapshot, or false once the wavefront is aborted.
func (wf *frameWorkReconWavefront) waitRows(need int) (frameWorkReconView, bool) {
	p := wf.ensureProgress()
	p.mu.Lock()
	for wf.published < need {
		if p.aborted.Load() {
			p.mu.Unlock()
			return frameWorkReconView{}, false
		}
		p.cond.Wait()
	}
	view := wf.pub
	ok := !p.aborted.Load()
	p.mu.Unlock()
	return view, ok
}

// startLanes launches the wavefront lanes on the pool's persistent lane group
// (or on fresh goroutines when the batch did not come from a pool). Lane w
// statically owns SB rows w, w+workers, ...; each row starts once the parse
// has published it.
func (wf *frameWorkReconWavefront) startLanes(lanes chan<- wavefrontLaneTask, workers int, rowCount int) {
	wf.firstErr.Store(nil)
	wf.lanes = workers
	wf.wg.Add(workers)
	for w := 0; w < workers; w++ {
		if lanes != nil {
			lanes <- wavefrontLaneTask{wf: wf, lane: w, lanes: workers, rowCount: rowCount}
			continue
		}
		go wf.runLane(w, workers, rowCount)
	}
}

func (wf *frameWorkReconWavefront) runLane(w int, workers int, rowCount int) {
	defer wf.wg.Done()
	st := &wf.states[w]
	for row := w; row < rowCount; row += workers {
		view, ok := wf.waitRows(row + 1)
		if !ok {
			return
		}
		if err := wf.reconstructRow(st, &view, row, rowCount); err != nil {
			// Signal abort so any goroutine waiting on this row's (now
			// never-published) progress stops waiting instead of deadlocking,
			// then record the first error.
			wf.firstErr.CompareAndSwap(nil, &wavefrontError{err: err})
			wf.abortProgress()
			return
		}
	}
}

// finishLanes joins the lanes and folds their reconstruction counters into the
// tile stats.
func (wf *frameWorkReconWavefront) finishLanes(stats *FrameWorkTileResidualStats) error {
	wf.wg.Wait()
	if e := wf.firstErr.Load(); e != nil {
		return e.err
	}
	for w := 0; w < wf.lanes; w++ {
		stats.Predictions += wf.states[w].localStats.Predictions
		stats.Residuals += wf.states[w].localStats.Residuals
	}
	return nil
}

// cancelLanes aborts and joins the lanes after a parse error.
func (wf *frameWorkReconWavefront) cancelLanes() {
	wf.abortProgress()
	wf.wg.Wait()
}

// startReconStream starts the SB-row wavefront before the tile parses, so
// reconstruction of row R overlaps the entropy decode of the rows below it
// (dav1d's per-sbrow recon tasks follow the tile's parse progress the same
// way). The parse publishes each SB row when the first block of the next row
// begins; the lanes gate on that and on the row above exactly as the
// after-parse replay did.
func (c *frameWorkTileResidualLoopController) startReconStream() error {
	region, err := c.batch.JobRegion(c.index)
	if err != nil {
		return err
	}
	sbSizeMIB := int(c.batch.Sequence.SBSizeMIB)
	rowCount := int(region.SBRows)
	if sbSizeMIB == 0 || rowCount <= 0 {
		return ErrInvalidBatch
	}
	workers := min(int(c.wavefrontWorkers), rowCount)
	if c.scratch.wavefront == nil {
		c.scratch.wavefront = &frameWorkReconWavefront{}
	}
	wf := c.scratch.wavefront
	wf.beginSpans(region, sbSizeMIB)
	wf.ensureDone(rowCount)
	wf.resetProgress(rowCount)
	if err := wf.ensureStates(workers, c); err != nil {
		return err
	}
	wf.frameThread = c.batch.FrameThread
	wf.startLanes(c.batch.wavefrontLanes, workers, rowCount)
	c.streaming = true
	return nil
}

// streamBlockBegin runs before a block-begin event at the next event index is
// buffered; when the block opens a new SB row it publishes the finished rows.
func (c *frameWorkTileResidualLoopController) streamBlockBegin(visit *tile.BlockLoopVisit) error {
	wf := c.scratch.wavefront
	rows, err := wf.noteBlockBegin(len(c.scratch.reconEvents), int(visit.Block.MIRow), int(visit.Block.MICol))
	if err != nil || rows <= 0 {
		return err
	}
	return c.publishReconRows(rows)
}

func (c *frameWorkTileResidualLoopController) publishReconRows(rows int) error {
	wf := c.scratch.wavefront
	cursor, err := c.scratch.installDeferredPalettes(wf.build.paletteCursor)
	if err != nil {
		return err
	}
	wf.build.paletteCursor = cursor
	wf.publishRows(rows, frameWorkReconView{
		rowStart: wf.rowStart,
		sbs:      wf.sbSpans,
		events:   c.scratch.reconEvents,
		visits:   c.scratch.reconVisits,
		blocks:   c.scratch.reconBlocks,
		cf:       c.scratch.reconCF,
	})
	return nil
}

// finishReconStream publishes the last rows once the parse succeeded and joins
// the lanes; after a parse error it cancels them so no goroutine outlives the
// job's scratch.
func (c *frameWorkTileResidualLoopController) finishReconStream(parseErr error) error {
	wf := c.scratch.wavefront
	c.streaming = false
	if parseErr != nil {
		wf.cancelLanes()
		return parseErr
	}
	if err := wf.closeSpans(len(c.scratch.reconEvents)); err != nil {
		wf.cancelLanes()
		return err
	}
	if err := c.publishReconRows(wf.build.rowCount); err != nil {
		wf.cancelLanes()
		return err
	}
	if ft := wf.frameThread; ft != nil {
		// Frame threading: the parse is done and every row is published; the
		// lanes finish under the frame's tail (FrameThreadRecon.Join).
		ft.addPending(wf, c.stats)
		return nil
	}
	return wf.finishLanes(c.stats)
}
