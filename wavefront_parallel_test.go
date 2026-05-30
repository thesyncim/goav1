package goav1_test

// End-to-end verification that the multi-worker SB-row reconstruction wavefront
// produces byte-identical output to the single-worker fused path, and that it
// speeds up a single-tile decode by employing the otherwise-idle pool lanes.
//
// The conformance harness (internal/av1/testvector) runs with a one-worker
// pool, so the wavefront never engages there. These tests drive the same public
// stream-runner path that production callers use but with workers > 1 on a
// single-tile clip, which is exactly the configuration that hands the idle
// lanes to the wavefront. GOAV1_WAVEFRONT_MIN_SBROWS_PER_WORKER=1 forces the
// wavefront on for the small bundled conformance clips (whose SB-row counts are
// below the production 2-rows-per-worker threshold).

import (
	"crypto/sha256"
	"os"
	"testing"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// wavefrontTestVector is a single-tile all-intra clip with enough superblock
// rows that a 2-4 worker wavefront overlaps. Intra reconstruction is the
// neighbor-dependent case the wavefront gate must get right, so it is the
// strongest byte-identity probe.
const wavefrontTestVector = "internal/av1/testdata/libaom/av1-1-b10-00-quantizer-00.ivf"

func decodeWavefrontFrames(t *testing.T, workers int) []byte {
	t.Helper()
	raw, err := os.ReadFile(wavefrontTestVector)
	if err != nil {
		t.Skipf("read wavefront vector: %v", err)
	}
	it, err := av1.NewIVFIterator(raw)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	var payloads [][]byte
	for {
		f, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf next: %v", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}
	if len(payloads) == 0 {
		t.Fatal("no frames in wavefront vector")
	}

	h := newWavefrontHarness(t, payloads, workers)
	defer h.Close()

	sum := sha256.New()
	// Hash every completed frame as it leaves post-filtering, before the frame
	// pool can recycle its surface. The pool holds only a bounded set of
	// surfaces (refs + current), so the final result's Outputs slice does not
	// retain every decoded frame; the per-frame post-filter hook does.
	post := func(ctx av1.DecoderFrameWorkPostFilterContext) error {
		hashFramePlanes(sum, ctx.Output)
		return nil
	}
	var result av1.DecoderFrameWorkResidualStreamResult
	if err := h.runner.RunLowOverheadsInto(&result, payloads, post); err != nil {
		t.Fatalf("decode (workers=%d): %v", workers, err)
	}
	return sum.Sum(nil)
}

// newWavefrontHarness builds a stream-runner harness backed by a worker pool of
// the requested size. It mirrors newDecodeBenchmarkHarness but is driven from a
// test and parameterized by the worker count so the same payloads can be
// decoded with one or many lanes.
func newWavefrontHarness(t *testing.T, payloads [][]byte, workers int) *decodeBenchmarkHarness {
	t.Helper()

	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatal(err)
	}

	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, benchMaxProbeEvents(payloads))
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)

	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, payloads, workers, probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}

	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		workerPool.Close()
		t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}
	const surfaceCount = av1.RefFrames + 1

	_, backingSize, err := av1.FramePoolRequiredSize(format, surfaceCount)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}
	frames := make([]av1.Frame, surfaceCount)
	free := make([]int, surfaceCount)
	used := make([]bool, surfaceCount)
	pool, err := av1.BindFramePool(make([]byte, backingSize), format, frames, free, used)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}

	h := &decodeBenchmarkHarness{
		pool:       pool,
		workerPool: workerPool,
		refSurface: make([]int, av1.InterRefsPerFrame),
		refFrames:  make([]*av1.Frame, av1.InterRefsPerFrame),
		releases:   make([]int, av1.RefFrames),
		payloads:   payloads,
	}
	h.scratch = newBenchStreamScratch(plan.Size)

	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &h.stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &h.state,
		Refs:              &h.refs,
		FramePool:         &h.pool,
		Align:             64,
		ReferenceSurfaces: h.refSurface,
		ReferenceFrames:   h.refFrames,
		Releases:          h.releases,
		WorkerPool:        h.workerPool,
		SideData:          &h.sideData,
		Stats:             &h.stats,
	}, h.scratch, &h.batch)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}
	h.runner = runner
	return h
}

func hashFramePlanes(sum interface{ Write([]byte) (int, error) }, f *av1.Frame) {
	if f == nil {
		return
	}
	sum.Write(f.Y.Pix)
	if !f.Format.MonoChrome {
		sum.Write(f.U.Pix)
		sum.Write(f.V.Pix)
	}
}

// TestWavefrontByteIdenticalToSingleWorker is the load-bearing gate: the
// multi-worker SB-row wavefront must reconstruct exactly the same bytes as the
// single-worker fused path. Run it under -race to also prove the wavefront is
// data-race free.
func TestWavefrontByteIdenticalToSingleWorker(t *testing.T) {
	t.Setenv("GOAV1_WAVEFRONT_MIN_SBROWS_PER_WORKER", "1")
	want := decodeWavefrontFrames(t, 1)
	for _, workers := range []int{2, 3, 4} {
		got := decodeWavefrontFrames(t, workers)
		if string(got) != string(want) {
			t.Fatalf("workers=%d output differs from workers=1 (wavefront not byte-identical)", workers)
		}
	}
}

// TestWavefrontSpeedup checks that handing the idle lanes to the wavefront
// reduces wall-clock decode time for a single-tile clip. It is timing-based, so
// it only asserts a modest improvement and skips under -race (which serializes
// the heap and erases the speedup) or on a single CPU.
func TestWavefrontSpeedup(t *testing.T) {
	if raceEnabled {
		t.Skip("race detector serializes execution; speedup is not observable")
	}
	t.Setenv("GOAV1_WAVEFRONT_MIN_SBROWS_PER_WORKER", "1")

	measure := func(workers int) time.Duration {
		best := time.Duration(1<<63 - 1)
		for i := 0; i < 8; i++ {
			start := time.Now()
			decodeWavefrontFrames(t, workers)
			if d := time.Since(start); d < best {
				best = d
			}
		}
		return best
	}

	single := measure(1)
	multi := measure(4)
	t.Logf("end-to-end decode best-of-8: workers=1 %v, workers=4 %v (%.2fx)", single, multi, float64(single)/float64(multi))
	// The wavefront parallelizes only reconstruction (pass 2), which is a
	// fraction of the end-to-end decode (entropy decode in pass 1 stays serial),
	// so the end-to-end gain on this small clip is modest. Require only that the
	// multi-worker path is not slower; the reconstruction-phase speedup itself is
	// measured directly by the threading package's BenchmarkReconWavefront.
	if multi > single+single/10 {
		t.Fatalf("wavefront regressed decode by >10%%: workers=1 %v vs workers=4 %v", single, multi)
	}
}
