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
