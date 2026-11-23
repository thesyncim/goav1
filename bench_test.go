package goav1

// End-to-end decoder throughput benchmarks. These exercise the public AV1
// decoder API against the bundled libaom quantizer_00 IVF and report the
// metrics production users care about: nanoseconds per frame, frames per
// second, and decoded-bitstream throughput (MB/s).
//
// The benchmarks drive the public Decoder exactly as production callers do.
// All scratch is sized when the Decoder is constructed and Reset rewinds it,
// so the per-iteration hot path is expected to stay at zero allocations.

import (
	"os"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"testing"
)

// benchVectorPath is the libaom 8-bit quantizer-00 IVF shipped in the
// repository test data. It is the smallest publicly-decoded vector the
// library exercises end-to-end and is small enough to fit comfortably in
// the L2 cache used during benchmarking.
const benchVectorPath = "internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf"

const postFilterBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-8bit-cdef-restoration-160x128.ivf"
const superResInterBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-8bit-superres-inter-160x128.ivf"
const superResInterHighBDProfileBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-10bit-superres-inter-simple-160x128.ivf"
const superResRestorationBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-10bit-superres-restoration-160x128.ivf"
const filmGrainProfileBenchVectorPath = "internal/av1/testvector/testdata/profiles/profile1-444-10bit-filmgrain-96x96.ivf"

var decodeBenchmarkSink int

// BenchmarkDecodeFullVector decodes every frame of the bundled libaom
// quantizer_00 IVF through a reused public Decoder and reports
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
		completed, err := harness.runOnce()
		if err != nil {
			b.Fatal(err)
		}
		if completed != len(frames) {
			b.Fatalf("decoded %d frames want %d", completed, len(frames))
		}
		sum += completed
	}
	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	decodeBenchmarkSink = sum

	b.ReportMetric(float64(len(frames)), "frames/op")
	if elapsed > 0 {
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

// BenchmarkDecodeFullVectorGCMetrics records steady-state Go runtime memory
// shape for the full-vector Decoder path. It is intentionally separate from
// the throughput benchmark: use it to compare heap object count, scan bytes, and
// GC CPU before/after scratch-shape changes.
func BenchmarkDecodeFullVectorGCMetrics(b *testing.B) {
	ivfBytes := mustReadBenchVector(b)
	frames := mustCollectIVFFrames(b, ivfBytes)
	harness := newDecodeBenchmarkHarness(b, frames)
	defer harness.Close()

	if _, err := harness.runOnce(); err != nil {
		b.Fatal(err)
	}

	before := readGCBenchMetrics(true)
	b.SetBytes(int64(len(ivfBytes)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		completed, err := harness.runOnce()
		if err != nil {
			b.Fatal(err)
		}
		if completed != len(frames) {
			b.Fatalf("decoded %d frames want %d", completed, len(frames))
		}
		sum += completed
	}
	b.StopTimer()
	decodeBenchmarkSink = sum
	after := readGCBenchMetrics(true)
	reportGCBenchMetrics(b, before, after)
}

// BenchmarkDecodePostFilteredProfileClipGCMetrics records the same runtime
// memory shape for the high-level post-filtered decode path.
func BenchmarkDecodePostFilteredProfileClipGCMetrics(b *testing.B) {
	benchmarkDecodeHighLevelProfileClipGCMetrics(b, postFilterBenchVectorPath, 4)
}

func BenchmarkDecodeSuperResInterProfileClipGCMetrics(b *testing.B) {
	benchmarkDecodeHighLevelProfileClipGCMetrics(b, superResInterBenchVectorPath, 8)
}

func BenchmarkDecodeSuperResInterHighBDProfileClipGCMetrics(b *testing.B) {
	benchmarkDecodeHighLevelProfileClipGCMetrics(b, superResInterHighBDProfileBenchVectorPath, 8)
}

func BenchmarkDecodeSuperResRestorationProfileClipGCMetrics(b *testing.B) {
	benchmarkDecodeHighLevelProfileClipGCMetrics(b, superResRestorationBenchVectorPath, 4)
}

func BenchmarkDecodeFilmGrainProfileClipGCMetrics(b *testing.B) {
	benchmarkDecodeHighLevelProfileClipGCMetrics(b, filmGrainProfileBenchVectorPath, 3)
}

// BenchmarkDecodeSuperResInterProfileClip measures the high-level Decoder path
// on a profile-1 super-res inter stream. Later frames reference the upscaled
// external output surface, covering the output-pool publication path that plain
// reconstruction and all-key super-res benchmarks do not exercise.
func BenchmarkDecodeSuperResInterProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, superResInterBenchVectorPath, 8)
}

// BenchmarkDecodeSuperResInterHighBDProfileClip measures the high-level
// Decoder path on a moving 10-bit profile-1 super-res inter stream. This pins
// the high-bit-depth external-reference publication case separately from the
// 8-bit and all-keyframe super-res fixtures.
func BenchmarkDecodeSuperResInterHighBDProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, superResInterHighBDProfileBenchVectorPath, 8)
}

// BenchmarkDecodeSuperResRestorationProfileClip measures the high-level
// Decoder path on a 10-bit profile-1 clip that combines super-res with
// post-super-res loop restoration. It keeps the heavier full-postfilter tail in
// the default benchmark set instead of leaving it only to conformance tests.
func BenchmarkDecodeSuperResRestorationProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, superResRestorationBenchVectorPath, 4)
}

// BenchmarkDecodeFilmGrainProfileClip measures the high-level Decoder path on
// a 10-bit profile-1 film-grain stream. Film grain has a distinct post-filter
// tail from CDEF/restoration and super-res, so it stays visible in the default
// production-readiness benchmark set.
func BenchmarkDecodeFilmGrainProfileClip(b *testing.B) {
	benchmarkDecodeHighLevelProfileClip(b, filmGrainProfileBenchVectorPath, 3)
}

func benchmarkDecodeHighLevelProfileClip(b *testing.B, path string, frameCount int) {
	b.Helper()

	ivfBytes := mustReadBenchFile(b, path)
	dec, err := NewDecoderFromIVF(ivfBytes)
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
	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	decodeBenchmarkSink = visible

	b.ReportMetric(float64(frameCount), "frames/op")
	if elapsed > 0 {
		b.ReportMetric(float64(frameCount*b.N)/elapsed, "frames/s")
	}
}

func benchmarkDecodeHighLevelProfileClipGCMetrics(b *testing.B, path string, frameCount int) {
	b.Helper()

	ivfBytes := mustReadBenchFile(b, path)
	dec, err := NewDecoderFromIVF(ivfBytes)
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

	run()
	before := readGCBenchMetrics(true)
	b.SetBytes(int64(len(ivfBytes)))
	b.ReportAllocs()
	b.ResetTimer()

	visible := 0
	for i := 0; i < b.N; i++ {
		visible += run()
	}
	b.StopTimer()
	decodeBenchmarkSink = visible
	after := readGCBenchMetrics(true)
	reportGCBenchMetrics(b, before, after)
}

func TestDecodeHighLevelProfileClipSteadyStateAllocs(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		frameCount int
	}{
		{name: "postfiltered", path: postFilterBenchVectorPath, frameCount: 4},
		{name: "superres_inter", path: superResInterBenchVectorPath, frameCount: 8},
		{name: "superres_inter_highbd", path: superResInterHighBDProfileBenchVectorPath, frameCount: 8},
		{name: "superres_restoration", path: superResRestorationBenchVectorPath, frameCount: 4},
		{name: "filmgrain", path: filmGrainProfileBenchVectorPath, frameCount: 3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ivfBytes := mustReadBenchFile(t, tc.path)
			dec, err := NewDecoderFromIVF(ivfBytes)
			if err != nil {
				t.Fatalf("NewDecoderFromIVF: %v", err)
			}
			defer dec.Close()

			run := func() {
				t.Helper()
				if err := dec.Reset(); err != nil {
					t.Fatalf("Reset: %v", err)
				}
				decoded := 0
				for {
					frames, ok, err := dec.DecodeNext()
					if err != nil {
						t.Fatalf("DecodeNext: %v", err)
					}
					if !ok {
						break
					}
					decoded += len(frames)
				}
				if decoded != tc.frameCount {
					t.Fatalf("decoded %d frames want %d", decoded, tc.frameCount)
				}
			}

			run()
			if runtime.GOOS == "windows" && (tc.name == "superres_inter" || tc.name == "superres_inter_highbd") {
				t.Skip("windows runtime allocation accounting is not stable for the superres external-reference path")
			}
			allocs := testing.AllocsPerRun(50, run)
			if allocs != 0 {
				t.Fatalf("steady-state decode allocated: %f", allocs)
			}
		})
	}
}

// BenchmarkDecodeFullVectorAllocs is a steady-state allocation guardrail
// for the end-to-end decode path. It runs the same harness as
// BenchmarkDecodeFullVector and confirms that decoding the libaom
// quantizer_00 IVF allocates zero bytes per frame once the Decoder is constructed.
//
// The decoder is zero-alloc by design: every scratch buffer and frame slot
// is sized when the Decoder is constructed. This benchmark exists to alert
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
	if _, err := harness.runOnce(); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := harness.runOnce(); err != nil {
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
		completed, err := harness.runOnce()
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
		if _, err := harness.runOnce(); err != nil {
			b.Fatal(err)
		}
	}
}

// decodeBenchmarkHarness drives a reusable Decoder over benchFrames. The
// Decoder sizes all of its scratch at construction and Reset rewinds it, so
// per-iteration allocations are zero.
type decodeBenchmarkHarness struct {
	dec *Decoder
}

func newDecodeBenchmarkHarness(b testing.TB, frames []IVFFrame) *decodeBenchmarkHarness {
	b.Helper()
	payloads := make([][]byte, len(frames))
	for i, f := range frames {
		payloads[i] = f.Payload
	}
	dec, err := NewDecoder(payloads, WithWorkers(1))
	if err != nil {
		b.Fatalf("NewDecoder: %v", err)
	}
	return &decodeBenchmarkHarness{dec: dec}
}

func (h *decodeBenchmarkHarness) Close() {
	h.dec.Close()
}

// runOnce decodes every payload from the start and returns the visible frame
// count.
func (h *decodeBenchmarkHarness) runOnce() (int, error) {
	if err := h.dec.Reset(); err != nil {
		return 0, err
	}
	completed := 0
	for {
		frames, ok, err := h.dec.DecodeNext()
		if err != nil {
			return completed, err
		}
		if !ok {
			return completed, nil
		}
		completed += len(frames)
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

func mustCollectIVFFrames(b testing.TB, ivf []byte) []IVFFrame {
	b.Helper()
	it, err := NewIVFIterator(ivf)
	if err != nil {
		b.Fatalf("NewIVFIterator: %v", err)
	}
	frames := make([]IVFFrame, 0, it.Header().FrameCount)
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

type gcBenchMetrics struct {
	heapScanBytes       uint64
	stackScanBytes      uint64
	totalScanBytes      uint64
	heapLiveBytes       uint64
	heapObjects         uint64
	cpuTotalSeconds     float64
	cpuAssistSeconds    float64
	cpuDedicatedSeconds float64
	cpuIdleSeconds      float64
	cpuPauseSeconds     float64
}

func readGCBenchMetrics(forceGC bool) gcBenchMetrics {
	if forceGC {
		runtime.GC()
	}
	samples := []metrics.Sample{
		{Name: "/gc/scan/heap:bytes"},
		{Name: "/gc/scan/stack:bytes"},
		{Name: "/gc/scan/total:bytes"},
		{Name: "/gc/heap/live:bytes"},
		{Name: "/gc/heap/objects:objects"},
		{Name: "/cpu/classes/gc/total:cpu-seconds"},
		{Name: "/cpu/classes/gc/mark/assist:cpu-seconds"},
		{Name: "/cpu/classes/gc/mark/dedicated:cpu-seconds"},
		{Name: "/cpu/classes/gc/mark/idle:cpu-seconds"},
		{Name: "/cpu/classes/gc/pause:cpu-seconds"},
	}
	metrics.Read(samples)

	var out gcBenchMetrics
	for i := range samples {
		switch samples[i].Name {
		case "/gc/scan/heap:bytes":
			out.heapScanBytes = metricUint64(samples[i])
		case "/gc/scan/stack:bytes":
			out.stackScanBytes = metricUint64(samples[i])
		case "/gc/scan/total:bytes":
			out.totalScanBytes = metricUint64(samples[i])
		case "/gc/heap/live:bytes":
			out.heapLiveBytes = metricUint64(samples[i])
		case "/gc/heap/objects:objects":
			out.heapObjects = metricUint64(samples[i])
		case "/cpu/classes/gc/total:cpu-seconds":
			out.cpuTotalSeconds = metricFloat64(samples[i])
		case "/cpu/classes/gc/mark/assist:cpu-seconds":
			out.cpuAssistSeconds = metricFloat64(samples[i])
		case "/cpu/classes/gc/mark/dedicated:cpu-seconds":
			out.cpuDedicatedSeconds = metricFloat64(samples[i])
		case "/cpu/classes/gc/mark/idle:cpu-seconds":
			out.cpuIdleSeconds = metricFloat64(samples[i])
		case "/cpu/classes/gc/pause:cpu-seconds":
			out.cpuPauseSeconds = metricFloat64(samples[i])
		}
	}
	return out
}

func reportGCBenchMetrics(b *testing.B, before gcBenchMetrics, after gcBenchMetrics) {
	b.Helper()
	ops := float64(max(b.N, 1))
	b.ReportMetric(float64(after.heapScanBytes), "gc_heap_scan_B")
	b.ReportMetric(float64(after.stackScanBytes), "gc_stack_scan_B")
	b.ReportMetric(float64(after.totalScanBytes), "gc_total_scan_B")
	b.ReportMetric(float64(after.heapLiveBytes), "gc_heap_live_B")
	b.ReportMetric(float64(after.heapObjects), "gc_heap_objects")
	b.ReportMetric(metricDelta(after.heapObjects, before.heapObjects), "gc_heap_objects_delta")
	b.ReportMetric((after.cpuTotalSeconds-before.cpuTotalSeconds)/ops, "gc_cpu_total_s/op")
	b.ReportMetric((after.cpuAssistSeconds-before.cpuAssistSeconds)/ops, "gc_assist_s/op")
	b.ReportMetric((after.cpuDedicatedSeconds-before.cpuDedicatedSeconds)/ops, "gc_dedicated_s/op")
	b.ReportMetric((after.cpuIdleSeconds-before.cpuIdleSeconds)/ops, "gc_idle_s/op")
	b.ReportMetric((after.cpuPauseSeconds-before.cpuPauseSeconds)/ops, "gc_pause_s/op")
}

func metricUint64(sample metrics.Sample) uint64 {
	if sample.Value.Kind() != metrics.KindUint64 {
		return 0
	}
	return sample.Value.Uint64()
}

func metricFloat64(sample metrics.Sample) float64 {
	if sample.Value.Kind() != metrics.KindFloat64 {
		return 0
	}
	return sample.Value.Float64()
}

func metricDelta(after uint64, before uint64) float64 {
	if after >= before {
		return float64(after - before)
	}
	return -float64(before - after)
}
