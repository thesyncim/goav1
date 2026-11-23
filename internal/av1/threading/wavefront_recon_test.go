package threading

// Focused tests for the deferred SB-row reconstruction wavefront. They build a
// large single-tile job out of per-superblock intra-DC blocks (whose
// reconstruction reads the already-reconstructed top and left neighbors — the
// dependency the wavefront gate must respect), then check that replaying those
// buffered events across multiple goroutines produces byte-identical output to
// the serial replay. A separate timing smoke test logs the speedup, but only
// enforces it outside shared CI hosts where scheduler noise dominates.

import (
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

var wavefrontEnsureStatesBenchSink int

// buildWavefrontReconController builds a controller whose deferred buffers are
// pre-filled with one 64x64 intra-DC skip-transform block per superblock of an
// sbCols x sbRows single-tile frame, in SB-raster order. Skip-transform intra
// blocks predict luma+chroma directly from neighbors with no coefficients, so
// they exercise the wavefront neighbor dependency without needing a real
// entropy decode. output receives the reconstructed pixels.
func buildWavefrontReconController(t testing.TB, output *frame.Frame, sbCols int, sbRows int, scratch *FrameWorkTileResidualScratch) *frameWorkTileResidualLoopController {
	t.Helper()
	batch := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs: []tile.Job{{SBCols: uint16(sbCols), SBRows: uint16(sbRows)}},
	}
	scratch.geomCache.reset()
	scratch.resetDeferredReconBuffers()
	batch.geomCache = &scratch.geomCache

	var stats FrameWorkTileResidualStats
	c := &scratch.controller
	*c = frameWorkTileResidualLoopController{
		batch:   batch,
		index:   0,
		scratch: scratch,
		stats:   &stats,
	}

	const sbMIB = 16 // 64x64 superblock = 16 4x4 MI units
	for r := range sbRows {
		for col := range sbCols {
			miCol := uint16(col * sbMIB)
			miRow := uint16(r * sbMIB)
			visit := tile.BlockLoopVisit{
				Block: tile.BlockVisit{
					MICol: miCol, MIRow: miRow,
					MIColEnd: miCol + sbMIB, MIRowEnd: miRow + sbMIB,
					X4: uint8(miCol), Y4: uint8(miRow),
					Size:      tile.BlockSize64x64,
					VisibleW4: sbMIB, VisibleH4: sbMIB,
					HaveTop:  r > 0,
					HaveLeft: col > 0,
				},
				Prediction: tile.BlockPredictionModeResult{
					Valid:           true,
					Intra:           true,
					LumaMode:        tile.IntraModeDC,
					ChromaMode:      tile.ChromaIntraModeDC,
					ChromaModeValid: true,
				},
			}
			visit.Prefix.SkipTransform = true
			// Skip intra blocks are predicted transform block by transform
			// block over the block's transform layout (dav1d recon_b_intra):
			// a 16x16 luma / 8x8 chroma grid gives each superblock 16 luma
			// and 2x16 chroma predictions, with no cbi entries.
			visit.Coefficients.Tree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
			scratch.reconVisits = append(scratch.reconVisits, visit)
		}
	}
	return c
}

func TestReconWavefrontByteIdenticalToSerial(t *testing.T) {
	const sbCols, sbRows = 8, 8 // 512x512, 8 SB rows
	format := frame.Format{Width: sbCols * 64, Height: sbRows * 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	serialOut := testBatchFrame(t, format)
	wavefrontOut := testBatchFrame(t, format)
	// Seed the (past-visible) borders deterministically so neighbor reads are
	// well-defined and identical for both runs.
	testFillFrame(serialOut, 0)
	testFillFrame(wavefrontOut, 0)

	var predict, wfPredict FrameWorkPredictionScratch
	var serialScratch, wavefrontScratch FrameWorkTileResidualScratch

	cSerial := buildWavefrontReconController(t, serialOut, sbCols, sbRows, &serialScratch)
	cSerial.recon = frameWorkReconState{
		batch:             cSerial.batch,
		index:             0,
		stats:             cSerial.stats,
		predictionScratch: &predict,
		cflScratch:        &serialScratch.CFL,
	}
	if err := cSerial.testReplayDeferredSerial(); err != nil {
		t.Fatalf("serial replay: %v", err)
	}

	for _, workers := range []int{2, 3, 4} {
		testFillFrame(wavefrontOut, 0)
		cWave := buildWavefrontReconController(t, wavefrontOut, sbCols, sbRows, &wavefrontScratch)
		cWave.wavefrontWorkers = uint16(workers)
		cWave.recon = frameWorkReconState{
			batch:             cWave.batch,
			index:             0,
			stats:             cWave.stats,
			predictionScratch: &wfPredict,
			cflScratch:        &wavefrontScratch.CFL,
		}
		if err := cWave.testReplayDeferredWavefront(); err != nil {
			t.Fatalf("wavefront replay workers=%d: %v", workers, err)
		}
		if !equalFramePlanes(serialOut, wavefrontOut) {
			t.Fatalf("workers=%d wavefront output differs from serial replay", workers)
		}
	}
}

func TestReconWavefrontScratchUsesFlatArenas(t *testing.T) {
	var wf frameWorkReconWavefront
	c := frameWorkTileResidualLoopController{
		req: FrameWorkTileResidualRequest{
			Int32Scratch:    make([]int32, 7),
			ResidualScratch: make([]int16, 11),
		},
	}
	if err := wf.ensureStates(3, &c); err != nil {
		t.Fatal(err)
	}
	if got, want := len(wf.int32Arena), 21; got != want {
		t.Fatalf("int32 arena len=%d want %d", got, want)
	}
	if got, want := len(wf.residualArena), 33; got != want {
		t.Fatalf("residual arena len=%d want %d", got, want)
	}
	for w := range 3 {
		st := &wf.states[w]
		if got, want := len(st.int32Scratch), 7; got != want {
			t.Fatalf("worker %d int32 len=%d want %d", w, got, want)
		}
		if got, want := len(st.residualScratch), 11; got != want {
			t.Fatalf("worker %d residual len=%d want %d", w, got, want)
		}
		if &st.int32Scratch[0] != &wf.int32Arena[w*7] {
			t.Fatalf("worker %d int32 scratch is not arena-backed", w)
		}
		if &st.residualScratch[0] != &wf.residualArena[w*11] {
			t.Fatalf("worker %d residual scratch is not arena-backed", w)
		}
		if w > 0 && &wf.states[w-1].int32Scratch[6] == &st.int32Scratch[0] {
			t.Fatalf("worker %d int32 scratch overlaps previous worker", w)
		}
		if w > 0 && &wf.states[w-1].residualScratch[10] == &st.residualScratch[0] {
			t.Fatalf("worker %d residual scratch overlaps previous worker", w)
		}
	}
}

func testReconWavefrontTwoRows() *frameWorkReconWavefront {
	wf := &frameWorkReconWavefront{
		rowStart: []int32{0, 1, 2},
		sbSpans: []frameWorkReconSB{
			{start: 0, end: 0, col: 0},
			{start: 0, end: 0, col: 0},
		},
	}
	wf.ensureDone(2)
	wf.resetProgress(2)
	return wf
}

func TestReconWavefrontProgressBlocksUntilPublished(t *testing.T) {
	wf := testReconWavefrontTwoRows()
	done := make(chan error, 1)
	go func() {
		var st frameWorkReconState
		view := frameWorkReconView{rowStart: wf.rowStart, sbs: wf.sbSpans}
		done <- wf.reconstructRow(&st, &view, 1, 2)
	}()

	select {
	case err := <-done:
		t.Fatalf("row reconstructed before upper-row progress was published: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	wf.publishRowProgress(&wf.done[0], 1)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("reconstruct row after progress: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("row did not reconstruct after upper-row progress was published")
	}
}

func TestReconWavefrontAbortWakesWaiter(t *testing.T) {
	wf := testReconWavefrontTwoRows()
	done := make(chan error, 1)
	go func() {
		var st frameWorkReconState
		view := frameWorkReconView{rowStart: wf.rowStart, sbs: wf.sbSpans}
		done <- wf.reconstructRow(&st, &view, 1, 2)
	}()

	select {
	case err := <-done:
		t.Fatalf("row reconstructed before abort: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	wf.abortProgress()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("reconstruct row after abort: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("row waiter did not wake after abort")
	}
}

func BenchmarkReconWavefrontEnsureStates(b *testing.B) {
	c := frameWorkTileResidualLoopController{
		req: FrameWorkTileResidualRequest{
			Int32Scratch:    make([]int32, 4096),
			ResidualScratch: make([]int16, 4096),
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		var wf frameWorkReconWavefront
		if err := wf.ensureStates(8, &c); err != nil {
			b.Fatal(err)
		}
		wavefrontEnsureStatesBenchSink += len(wf.states) + len(wf.int32Arena) + len(wf.residualArena)
	}
}

// testReplayDeferredSerial reconstructs every buffered block in decode order on
// the controller's own reconstruction state.
func (c *frameWorkTileResidualLoopController) testReplayDeferredSerial() error {
	if err := c.scratch.finalizeDeferredPalettes(); err != nil {
		return err
	}
	cur := frameWorkReconCursor{cbi: c.scratch.reconCBI, cf: c.scratch.cfArena}
	for i := range c.scratch.reconVisits {
		if err := c.recon.reconBDeferred(&c.scratch.reconVisits[i], &cur); err != nil {
			return err
		}
	}
	return nil
}

// testReplayDeferredWavefront reconstructs a fully buffered tile of skip
// blocks (no cbi entries or coefficients) through the SB-row wavefront lanes.
func (c *frameWorkTileResidualLoopController) testReplayDeferredWavefront() error {
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
	for i := range c.scratch.reconVisits {
		v := &c.scratch.reconVisits[i]
		if !v.Prefix.SkipTransform {
			return ErrInvalidBatch
		}
		if _, err := wf.noteBlockBegin(i, 0, 0, int(v.Block.MIRow), int(v.Block.MICol)); err != nil {
			return err
		}
	}
	if err := wf.closeSpans(len(c.scratch.reconVisits)); err != nil {
		return err
	}
	wf.ensureDone(rowCount)
	wf.resetProgress(rowCount)
	if err := wf.ensureStates(workers, c); err != nil {
		return err
	}
	wf.startLanes(c.batch.wavefrontLanes, workers, rowCount)
	wf.publishRows(rowCount, frameWorkReconView{
		rowStart: wf.rowStart,
		sbs:      wf.sbSpans,
		visits:   c.scratch.reconVisits,
		cbi:      c.scratch.reconCBI,
		cfArena:  c.scratch.cfArena,
	})
	return wf.finishLanes(c.stats)
}

func equalFramePlanes(a *frame.Frame, b *frame.Frame) bool {
	return string(a.Y.Pix) == string(b.Y.Pix) &&
		string(a.U.Pix) == string(b.U.Pix) &&
		string(a.V.Pix) == string(b.V.Pix)
}

// BenchmarkReconWavefrontReplay isolates the parallelized reconstruction phase:
// the serial deferred replay against the four-lane wavefront on a large
// single-tile workload (2048x2048, 32 superblock rows).
func BenchmarkReconWavefrontReplay(b *testing.B) {
	const sbCols, sbRows = 32, 32
	format := frame.Format{Width: sbCols * 64, Height: sbRows * 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	out := testBatchFrame(b, format)

	var predict FrameWorkPredictionScratch
	var scratch FrameWorkTileResidualScratch
	for _, workers := range []int{1, 4} {
		name := "serial"
		if workers > 1 {
			name = "wavefront4"
		}
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				b.StopTimer()
				testFillFrame(out, 0)
				c := buildWavefrontReconController(b, out, sbCols, sbRows, &scratch)
				c.recon = frameWorkReconState{
					batch:             c.batch,
					index:             0,
					stats:             c.stats,
					predictionScratch: &predict,
					cflScratch:        &scratch.CFL,
				}
				b.StartTimer()
				var err error
				if workers > 1 {
					c.wavefrontWorkers = uint16(workers)
					err = c.testReplayDeferredWavefront()
				} else {
					err = c.testReplayDeferredSerial()
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestFrameWorkWavefrontLanesCapsToRows checks that a pool wider than the tile
// has wavefront rows for runs on fewer lanes instead of falling back to the
// fused serial path.
func TestFrameWorkWavefrontLanesCapsToRows(t *testing.T) {
	format := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	for _, tc := range []struct {
		sb128   bool
		workers uint16
		want    uint16
	}{
		{sb128: true, workers: 1, want: 0},  // no idle lanes
		{sb128: true, workers: 2, want: 2},  // 6 SB rows
		{sb128: true, workers: 3, want: 3},  // 6 rows / 2 per lane
		{sb128: true, workers: 4, want: 3},  // capped, not disabled
		{sb128: true, workers: 16, want: 3}, // capped, not disabled
		{sb128: false, workers: 8, want: 6}, // 12 SB rows of 64x64
	} {
		b := wavefrontLanesTestBatch(t, format, tc.sb128)
		b.WavefrontWorkers = tc.workers
		if got := frameWorkWavefrontLanes(b, 0); got != tc.want {
			t.Errorf("sb128=%v workers=%d: lanes=%d, want %d", tc.sb128, tc.workers, got, tc.want)
		}
		b.FrameSize.AllowIntrabc = true
		if got := frameWorkWavefrontLanes(b, 0); got != 0 {
			t.Errorf("sb128=%v workers=%d intrabc: lanes=%d, want 0", tc.sb128, tc.workers, got)
		}
	}
}

func wavefrontLanesTestBatch(t *testing.T, format frame.Format, sb128 bool) FrameWorkBatch {
	t.Helper()
	sb := 64
	if sb128 {
		sb = 128
	}
	b := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: sb128,
				ColorConfig: parser.ColorConfig{
					BitDepth:     format.BitDepth,
					SubsamplingX: format.SubsamplingX,
					SubsamplingY: format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(format.Width), Height: uint32(format.Height)},
		},
		Jobs: []tile.Job{{SBCols: uint16((format.Width + sb - 1) / sb), SBRows: uint16((format.Height + sb - 1) / sb)}},
	}
	return b
}
