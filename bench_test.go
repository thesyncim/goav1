package goav1_test

// End-to-end decoder throughput benchmarks. These exercise the public AV1
// decoder API against the bundled libaom quantizer_00 IVF and report the
// metrics production users care about: nanoseconds per frame, frames per
// second, and decoded-bitstream throughput (MB/s).
//
// The benchmarks deliberately use the same public stream-runner path that
// production callers wire up, so the numbers reflect actual library
// performance rather than an internal helper. All scratch buffers are
// allocated once during setup; the per-iteration hot path is expected to
// stay at zero allocations.

import (
	"os"
	"path/filepath"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

// benchVectorPath is the libaom 8-bit quantizer-00 IVF shipped in the
// repository test data. It is the smallest publicly-decoded vector the
// library exercises end-to-end and is small enough to fit comfortably in
// the L2 cache used during benchmarking.
const benchVectorPath = "internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf"

const postFilterBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-8bit-cdef-restoration-160x128.ivf"
const superResInterBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-8bit-superres-inter-160x128.ivf"

var decodeBenchmarkSink int

// BenchmarkDecodeFullVector decodes every frame of the bundled libaom
// quantizer_00 IVF using the public residual stream runner and reports
// throughput as ns/op, frames/op, fps, and MB/s of decoded bitstream.
//
// b.SetBytes reports the IVF bitstream byte count so `go test -bench`
// surfaces the standard "MB/s" column automatically. frames/op and fps are
// reported via b.ReportMetric for callers that want per-frame numbers.
func BenchmarkDecodeFullVector(b *testing.B) {
	ivfBytes := mustReadBenchVector(b)
	frames := mustCollectIVFFrames(b, ivfBytes)
	totalPayload := 0
	for _, f := range frames {
		totalPayload += len(f.Payload)
	}
	harness := newDecodeBenchmarkHarness(b, frames)
	defer harness.Close()

	b.SetBytes(int64(len(ivfBytes)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		completed, txbs, err := harness.runOnce()
		if err != nil {
			b.Fatal(err)
		}
		if completed != len(frames) {
			b.Fatalf("decoded %d frames want %d", completed, len(frames))
		}
		sum += txbs
	}
	decodeBenchmarkSink = sum

	b.ReportMetric(float64(len(frames)), "frames/op")
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(float64(len(frames)*b.N)/elapsed, "frames/s")
	}
}

// BenchmarkDecodePostFilteredProfileClip measures the high-level Decoder path
// on a small profile-1 clip that signals loop filter, CDEF, and loop
// restoration. This complements BenchmarkDecodeFullVector, whose fixture is
// primarily a residual/reconstruction throughput guard.
func BenchmarkDecodePostFilteredProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, postFilterBenchVectorPath, 4)
}

// BenchmarkDecodeSuperResInterProfileClip measures the high-level Decoder path
// on a profile-1 super-res inter stream. Later frames reference the upscaled
// external output surface, covering the output-pool publication path that plain
// reconstruction and all-key super-res benchmarks do not exercise.
func BenchmarkDecodeSuperResInterProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, superResInterBenchVectorPath, 8)
}

func benchmarkDecodeHighLevelProfileClip(b *testing.B, path string, frameCount int) {
	b.Helper()

	ivfBytes := mustReadBenchFile(b, path)
	dec, err := av1.NewDecoderFromIVF(ivfBytes)
	if err != nil {
		b.Fatalf("NewDecoderFromIVF: %v", err)
	}
	defer dec.Close()

	run := func() int {
		b.Helper()
		if err := dec.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		decoded := 0
		for {
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				b.Fatalf("DecodeNext: %v", err)
			}
			if !ok {
				break
			}
			decoded += len(frames)
		}
		if decoded != frameCount {
			b.Fatalf("decoded %d frames want %d", decoded, frameCount)
		}
		return decoded
	}

	// Prime the high-level path so any one-shot scratch growth does not pollute
	// the steady-state allocation and throughput measurements.
	run()

	b.SetBytes(int64(len(ivfBytes)))
	b.ReportAllocs()
	b.ResetTimer()

	visible := 0
	for i := 0; i < b.N; i++ {
		visible += run()
	}
	decodeBenchmarkSink = visible

	b.ReportMetric(float64(frameCount), "frames/op")
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(float64(frameCount*b.N)/elapsed, "frames/s")
	}
}

// BenchmarkDecodeFullVectorAllocs is a steady-state allocation guardrail
// for the end-to-end decode path. It runs the same harness as
// BenchmarkDecodeFullVector and confirms that decoding the libaom
// quantizer_00 IVF allocates zero bytes per frame once scratch is bound.
//
// The current decoder pipeline is zero-alloc by design: every scratch
// buffer and frame slot is caller-owned. This benchmark exists to alert
// future contributors when a change accidentally introduces a per-frame
// allocation. It is not a throughput benchmark and is intentionally short.
func BenchmarkDecodeFullVectorAllocs(b *testing.B) {
	ivfBytes := mustReadBenchVector(b)
	frames := mustCollectIVFFrames(b, ivfBytes)
	harness := newDecodeBenchmarkHarness(b, frames)
	defer harness.Close()

	// Prime the harness once so any one-shot lazy initialisation that
	// might happen on the very first decode does not pollute the
	// allocation counter we are about to read.
	if _, _, err := harness.runOnce(); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := harness.runOnce(); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDecodeFullVectorSteadyStateAllocs(t *testing.T) {
	ivfBytes := mustReadBenchVector(t)
	frames := mustCollectIVFFrames(t, ivfBytes)
	harness := newDecodeBenchmarkHarness(t, frames)
	defer harness.Close()

	run := func() {
		completed, _, err := harness.runOnce()
		if err != nil {
			t.Fatalf("decode full vector: %v", err)
		}
		if completed != len(frames) {
			t.Fatalf("decoded %d frames want %d", completed, len(frames))
		}
	}

	run()
	allocs := testing.AllocsPerRun(50, run)
	if allocs != 0 {
		t.Fatalf("steady-state decode allocated: %f", allocs)
	}
}

// BenchmarkDecodeFirstFrameOnly isolates the cost of decoding a single
// keyframe so callers can compare the first-frame latency profile against
// the steady-state per-frame cost reported by BenchmarkDecodeFullVector.
func BenchmarkDecodeFirstFrameOnly(b *testing.B) {
	ivfBytes := mustReadBenchVector(b)
	frames := mustCollectIVFFrames(b, ivfBytes)
	if len(frames) == 0 {
		b.Fatal("no frames in bench vector")
	}
	first := frames[:1]
	harness := newDecodeBenchmarkHarness(b, first)
	defer harness.Close()

	b.SetBytes(int64(len(first[0].Payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := harness.runOnce(); err != nil {
			b.Fatal(err)
		}
	}
}

// decodeBenchmarkHarness owns the scratch a residual stream runner needs
// to decode benchFrames end-to-end. It is allocated once and reused across
// benchmark iterations so per-iteration allocations are zero.
type decodeBenchmarkHarness struct {
	pool       av1.FramePool
	workerPool *av1.TileWorkerPool
	stream     av1.DecoderStream

	refs       av1.DecoderSurfaceReferences
	state      av1.DecoderFrameWorkState
	refSurface []int
	refFrames  []*av1.Frame
	releases   []int
	stats      av1.DecoderFrameWorkTileResidualStats
	sideData   av1.DecoderFrameWorkSideData
	batch      av1.DecoderFrameWorkBatchResidualRunner

	scratch av1.DecoderFrameWorkResidualStreamScratch
	runner  av1.DecoderFrameWorkResidualStreamRunner

	payloads [][]byte
}

func newDecodeBenchmarkHarness(b testing.TB, frames []av1.IVFFrame) *decodeBenchmarkHarness {
	b.Helper()
	payloads := make([][]byte, len(frames))
	for i, f := range frames {
		payloads[i] = f.Payload
	}

	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		b.Fatal(err)
	}

	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, benchMaxProbeEvents(payloads))
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)

	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, payloads, 1, probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		workerPool.Close()
		b.Fatal(err)
	}

	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		b.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}
	// Provide enough surfaces for the bounded reference set plus the in-flight
	// output. A pool of MaxFrameBufferCount slots matches the worst-case
	// requirement of the AV1 bitstream.
	const surfaceCount = av1.RefFrames + 1
	pool := bindBenchFramePool(b, format, surfaceCount)

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
		b.Fatal(err)
	}
	h.runner = runner
	return h
}

func (h *decodeBenchmarkHarness) Close() {
	if h.workerPool != nil {
		h.workerPool.Close()
		h.workerPool = nil
	}
}

func (h *decodeBenchmarkHarness) runOnce() (int, int, error) {
	h.pool.Reset()
	h.refs.Reset()
	h.state.Reset()
	h.stats = av1.DecoderFrameWorkTileResidualStats{}
	if err := h.runner.Reset(); err != nil {
		return 0, 0, err
	}
	var result av1.DecoderFrameWorkResidualStreamResult
	if err := h.runner.RunLowOverheadsInto(&result, h.payloads, nil); err != nil {
		return 0, 0, err
	}
	return result.Run.CompletedFrames, h.stats.TXBs, nil
}

func bindBenchFramePool(b testing.TB, format av1.FrameFormat, count int) av1.FramePool {
	b.Helper()
	_, backingSize, err := av1.FramePoolRequiredSize(format, count)
	if err != nil {
		b.Fatal(err)
	}
	frames := make([]av1.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := av1.BindFramePool(make([]byte, backingSize), format, frames, free, used)
	if err != nil {
		b.Fatal(err)
	}
	return pool
}

func newBenchStreamScratch(size av1.DecoderFrameWorkResidualStreamScratchSize) av1.DecoderFrameWorkResidualStreamScratch {
	return av1.DecoderFrameWorkResidualStreamScratch{
		Events:    make([]av1.DecoderEvent, size.Events),
		Event:     newBenchEventScratch(size.Event),
		SideData:  newBenchSideDataScratch(size.Event.SideData),
		Outputs:   make([]*av1.Frame, size.Event.Outputs),
		RTPBuffer: make([]byte, size.RTPBuffer),
		RTPSpans:  make([]av1.RTPObuSpan, size.RTPSpans),
	}
}

func newBenchEventScratch(size av1.DecoderFrameWorkResidualEventScratchSize) av1.DecoderFrameWorkResidualEventScratch {
	return av1.DecoderFrameWorkResidualEventScratch{
		Runner:   newBenchBatchRunnerScratch(size.Runner),
		SideData: newBenchSideDataScratch(size.SideData),
		Spans:    make([]av1.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]av1.TileJob, size.Plan.JobCount),
		Batches:  make([]av1.TileBatch, size.Plan.BatchCount),
	}
}

func newBenchBatchRunnerScratch(size av1.DecoderFrameWorkBatchResidualRunnerScratchSize) av1.DecoderFrameWorkBatchResidualRunnerScratch {
	return av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newBenchSideDataScratch(size av1.DecoderFrameWorkSideDataScratchSize) av1.DecoderFrameWorkSideDataScratch {
	return av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap),
		CDEFReadMap:              make([]bool, size.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow),
	}
}

func mustReadBenchVector(b testing.TB) []byte {
	b.Helper()
	// GOAV1_BENCH_VECTOR overrides the default lossless quantizer-00 clip so a
	// profiling run can target a more representative bitstream (e.g. a higher
	// quantizer that exercises the entropy and reconstruction paths more
	// heavily). It accepts either an absolute path or a bare libaom vector
	// name resolved under the bundled test-data directory.
	path := benchVectorPath
	if override := os.Getenv("GOAV1_BENCH_VECTOR"); override != "" {
		path = override
		if !filepath.IsAbs(override) && filepath.Dir(override) == "." {
			path = filepath.Join("internal/av1/testdata/libaom", override)
		}
	}
	return mustReadBenchFile(b, path)
}

func mustReadBenchFile(b testing.TB, path string) []byte {
	b.Helper()
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		b.Fatalf("read bench vector %q: %v", path, err)
	}
	return raw
}

func mustCollectIVFFrames(b testing.TB, ivf []byte) []av1.IVFFrame {
	b.Helper()
	it, err := av1.NewIVFIterator(ivf)
	if err != nil {
		b.Fatalf("NewIVFIterator: %v", err)
	}
	frames := make([]av1.IVFFrame, 0, it.Header().FrameCount)
	for {
		frame, ok, err := it.Next()
		if err != nil {
			b.Fatalf("ivf next: %v", err)
		}
		if !ok {
			break
		}
		frames = append(frames, frame)
	}
	if len(frames) == 0 {
		b.Fatal("no frames decoded from bench vector")
	}
	return frames
}

func benchMaxProbeEvents(payloads [][]byte) int {
	// Each frame typically expands to a handful of events. Reserve a
	// generous bound so the streams planner never returns
	// ErrDecoderEventBufferTooSmall during sizing.
	const eventsPerFrame = 16
	n := max(len(payloads)*eventsPerFrame, 64)
	return n
}
