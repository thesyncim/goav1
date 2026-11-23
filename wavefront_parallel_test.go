package goav1

// End-to-end verification that the multi-worker SB-row reconstruction wavefront
// produces byte-identical output to the single-worker fused path, with a
// benchmark of the same decode at one and four workers.
//
// The conformance harness (internal/av1/testvector) runs with a one-worker
// pool, so the wavefront never engages there. These tests drive the same public
// stream-runner path that production callers use but with workers > 1 on a
// single-tile clip, which is exactly the configuration that hands the idle
// lanes to the wavefront. The threading test hook forces the wavefront on for
// the small bundled conformance clips (whose SB-row counts are below the
// production 2-rows-per-worker threshold).

import (
	"crypto/sha256"
	"os"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"

	"github.com/thesyncim/goav1/internal/av1/threading"
)

// wavefrontTestVector is a single-tile all-intra clip with enough superblock
// rows that a 2-4 worker wavefront overlaps. Intra reconstruction is the
// neighbor-dependent case the wavefront gate must get right, so it is the
// strongest byte-identity probe.
const wavefrontTestVector = "internal/av1/testdata/libaom/av1-1-b10-00-quantizer-00.ivf"

func decodeWavefrontFrames(t testing.TB, workers int) []byte {
	t.Helper()
	raw, err := os.ReadFile(wavefrontTestVector)
	if err != nil {
		t.Skipf("read wavefront vector: %v", err)
	}
	it, err := NewIVFIterator(raw)
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
	post := func(ctx internaldecoder.FrameWorkPostFilterContext) error {
		hashFramePlanes(sum, ctx.Output)
		return nil
	}
	var result decoderFrameWorkResidualStreamResult
	if err := h.runner.RunLowOverheadsInto(&result, payloads, post); err != nil {
		t.Fatalf("decode (workers=%d): %v", workers, err)
	}
	return sum.Sum(nil)
}

// newWavefrontHarness builds a stream-runner harness backed by a worker pool of
// the requested size. It mirrors newDecodeBenchmarkHarness but is driven from a
// test and parameterized by the worker count so the same payloads can be
// decoded with one or many lanes.
func newWavefrontHarness(t testing.TB, payloads [][]byte, workers int) *decodeBenchmarkHarness {
	t.Helper()

	workerPool, err := threading.NewPool(workers)
	if err != nil {
		t.Fatal(err)
	}

	var probeStream internaldecoder.Stream
	probeEvents := make([]internaldecoder.Event, benchMaxProbeEvents(payloads))
	probeSpans := make([]internalparser.TileSpan, internalparser.MaxTiles)
	probeJobs := make([]internaltile.Job, internalparser.MaxTiles)
	probeBatches := make([]threading.Batch, internalparser.MaxTiles)

	plan, err := decoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, payloads, workers, probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}

	format, err := frameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		workerPool.Close()
		t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}
	const surfaceCount = internalparser.RefFrames + 1

	_, backingSize, err := framePoolRequiredSize(format, surfaceCount)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}
	frames := make([]Frame, surfaceCount)
	free := make([]int, surfaceCount)
	used := make([]bool, surfaceCount)
	pool, err := internalframe.BindPool(make([]byte, backingSize), format, frames, free, used)
	if err != nil {
		workerPool.Close()
		t.Fatal(err)
	}

	h := &decodeBenchmarkHarness{
		pool:       pool,
		workerPool: workerPool,
		refSurface: make([]int, internalparser.InterRefsPerFrame),
		refFrames:  make([]*Frame, internalparser.InterRefsPerFrame),
		releases:   make([]int, internalparser.RefFrames),
		payloads:   payloads,
	}
	h.scratch = newBenchStreamScratch(plan.Size)

	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &h.stream, decoderFrameWorkResidualEventRuntime{
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

func hashFramePlanes(sum interface{ Write([]byte) (int, error) }, f *Frame) {
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
	defer threading.OverrideWavefrontMinSBRowsPerWorkerForTest(1)()
	want := decodeWavefrontFrames(t, 1)
	for _, workers := range []int{2, 3, 4} {
		got := decodeWavefrontFrames(t, workers)
		if string(got) != string(want) {
			t.Fatalf("workers=%d output differs from workers=1 (wavefront not byte-identical)", workers)
		}
	}
}

// BenchmarkWavefrontDecode decodes the single-tile clip end to end with one
// worker and with four. The wavefront parallelizes only reconstruction; entropy
// decoding stays serial, so the end-to-end gain on this small clip is modest.
// The reconstruction phase alone is BenchmarkReconWavefrontReplay in the
// threading package.
func BenchmarkWavefrontDecode(b *testing.B) {
	defer threading.OverrideWavefrontMinSBRowsPerWorkerForTest(1)()
	for _, bc := range []struct {
		name    string
		workers int
	}{{"workers1", 1}, {"workers4", 4}} {
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				decodeWavefrontFrames(b, bc.workers)
			}
		})
	}
}
