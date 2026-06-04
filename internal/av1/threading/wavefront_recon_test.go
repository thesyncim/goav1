package threading

// Focused tests for the deferred SB-row reconstruction wavefront. They build a
// large single-tile job out of per-superblock intra-DC blocks (whose
// reconstruction reads the already-reconstructed top and left neighbors — the
// dependency the wavefront gate must respect), then check that replaying those
// buffered events across multiple goroutines produces byte-identical output to
// the serial replay, and that it is faster.

import (
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

var wavefrontEnsureStatesBenchSink int

// buildWavefrontReconController builds a controller whose deferred event list is
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
			visitIndex := len(scratch.reconVisits)
			scratch.reconVisits = append(scratch.reconVisits, visit)
			scratch.reconEvents = append(scratch.reconEvents, frameWorkReconEvent{
				kind:  frameWorkReconEventBlockBegin,
				index: int32(visitIndex),
			})
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
	if err := cSerial.replayDeferredReconstruction(); err != nil {
		t.Fatalf("serial replay: %v", err)
	}

	for _, workers := range []int{2, 3, 4} {
		testFillFrame(wavefrontOut, 0)
		cWave := buildWavefrontReconController(t, wavefrontOut, sbCols, sbRows, &wavefrontScratch)
		cWave.wavefrontWorkers = workers
		cWave.recon = frameWorkReconState{
			batch:             cWave.batch,
			index:             0,
			stats:             cWave.stats,
			predictionScratch: &wfPredict,
			cflScratch:        &wavefrontScratch.CFL,
		}
		if err := cWave.replayDeferredReconstructionWavefront(); err != nil {
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

func equalFramePlanes(a *frame.Frame, b *frame.Frame) bool {
	return string(a.Y.Pix) == string(b.Y.Pix) &&
		string(a.U.Pix) == string(b.U.Pix) &&
		string(a.V.Pix) == string(b.V.Pix)
}

// TestReconWavefrontSpeedup isolates the parallelized reconstruction phase: it
// times the serial deferred replay against the multi-goroutine wavefront on a
// large single-tile workload. It skips under -race (which serializes the heap).
func TestReconWavefrontSpeedup(t *testing.T) {
	if reconRaceEnabled {
		t.Skip("race detector serializes execution; speedup is not observable")
	}
	const sbCols, sbRows = 24, 24 // 1536x1536, 24 SB rows
	format := frame.Format{Width: sbCols * 64, Height: sbRows * 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	out := testBatchFrame(t, format)

	var predict FrameWorkPredictionScratch
	var scratch FrameWorkTileResidualScratch

	bind := func(c *frameWorkTileResidualLoopController) {
		c.recon = frameWorkReconState{
			batch:             c.batch,
			index:             0,
			stats:             c.stats,
			predictionScratch: &predict,
			cflScratch:        &scratch.CFL,
		}
	}

	measureSerial := func() time.Duration {
		best := time.Duration(1<<63 - 1)
		for range 5 {
			testFillFrame(out, 0)
			c := buildWavefrontReconController(t, out, sbCols, sbRows, &scratch)
			bind(c)
			start := time.Now()
			if err := c.replayDeferredReconstruction(); err != nil {
				t.Fatalf("serial replay: %v", err)
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}
	measureWavefront := func(workers int) time.Duration {
		best := time.Duration(1<<63 - 1)
		for range 5 {
			testFillFrame(out, 0)
			c := buildWavefrontReconController(t, out, sbCols, sbRows, &scratch)
			c.wavefrontWorkers = workers
			bind(c)
			start := time.Now()
			if err := c.replayDeferredReconstructionWavefront(); err != nil {
				t.Fatalf("wavefront replay: %v", err)
			}
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}

	serial := measureSerial()
	wave := measureWavefront(4)
	t.Logf("reconstruction best-of-5: serial %v, wavefront(4) %v (%.2fx)", serial, wave, float64(serial)/float64(wave))
	if wave >= serial {
		t.Fatalf("wavefront did not speed up reconstruction: serial %v vs wavefront %v", serial, wave)
	}
}
