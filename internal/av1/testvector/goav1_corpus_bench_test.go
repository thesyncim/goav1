//go:build goav1_oracle

package testvector

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

var corpusBenchmarkFramesSink int

const envCorpusBenchmarkDir = "GOAV1_BENCH_CORPUS_DIR"
const envCorpusBenchmarkClip = "GOAV1_BENCH_CORPUS_CLIP"

// TestGoav1CorpusBenchmarkClip is the narrow correctness gate for experiments
// measured by the long-corpus benchmarks below. It decodes only the selected
// clip and checks every visible output byte against its committed MD5 sidecar.
func TestGoav1CorpusBenchmarkClip(t *testing.T) {
	name := strings.TrimSuffix(strings.TrimSpace(os.Getenv(envCorpusBenchmarkClip)), ".ivf")
	if name == "" {
		t.Skipf("set %s to a corpus clip name", envCorpusBenchmarkClip)
	}
	if filepath.Base(name) != name {
		t.Fatalf("%s must be a clip name, got %q", envCorpusBenchmarkClip, name)
	}
	dir, ok := corpusDir(t)
	if !ok {
		t.Fatalf("no corpus clips in %s", dir)
	}
	clips, failed := loadCorpusClipCandidates(t, []corpusClipCandidate{{
		name:    name,
		ivfPath: filepath.Join(dir, name+".ivf"),
	}})
	for _, failure := range failed {
		t.Errorf("%s: %s", failure.name, failure.reason)
	}
	if len(failed) != 0 {
		t.FailNow()
	}
	if len(clips) != 1 {
		t.Fatalf("loaded %d clips, want 1", len(clips))
	}
	t.Logf("%s: %d frames %dx%d %s", clips[0].name, clips[0].frames, clips[0].width, clips[0].height, corpusClipOracleLog(clips[0]))
}

// BenchmarkGoav1CorpusDecode measures a complete single-worker decode over
// each long corpus clip. A sub-benchmark can be selected by name, for example:
//
//	go test -tags goav1_oracle ./internal/av1/testvector \
//	  -bench '^BenchmarkGoav1CorpusDecode/p720_inter_q32$' -run '^$'
//
// Input is retained in memory while every per-decode pool and scratch lifetime
// remains inside the timed operation, matching the in-process side of the
// cross-decoder corpus contract without running the full matrix.
func BenchmarkGoav1CorpusDecode(b *testing.B) {
	dir := os.Getenv(envCorpusBenchmarkDir)
	if dir == "" {
		_, filename, _, ok := runtime.Caller(0)
		if !ok {
			b.Fatal("resolve corpus benchmark source")
		}
		dir = filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", "testdata", "benchcorpus"))
	}
	clips, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		b.Fatal(err)
	}
	if len(clips) == 0 {
		b.Skipf("no corpus clips in %s", dir)
	}
	for _, path := range clips {
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		b.Run(name, func(b *testing.B) {
			data, err := os.ReadFile(path)
			if err != nil {
				b.Fatal(err)
			}
			warm, err := decodeCorpusClipDiscard(data)
			if err != nil {
				b.Fatal(err)
			}
			if warm.frames == 0 {
				b.Fatal("decoded no visible frames")
			}

			b.SetBytes(int64(len(data)))
			b.ReportAllocs()
			b.ResetTimer()
			frames := 0
			for i := 0; i < b.N; i++ {
				result, err := decodeCorpusClipDiscard(data)
				if err != nil {
					b.Fatal(err)
				}
				if result.frames != warm.frames {
					b.Fatalf("decoded %d visible frames, want %d", result.frames, warm.frames)
				}
				frames += result.frames
			}
			b.StopTimer()
			corpusBenchmarkFramesSink = frames
			b.ReportMetric(float64(warm.frames), "frames/op")
			if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
				b.ReportMetric(float64(frames)/elapsed, "frames/s")
			}
		})
	}
}

// BenchmarkGoav1CorpusDecodeSteadyState measures the reusable public decoder
// over the same long clips. It excludes construction and teardown but includes
// Reset plus every DecodeNext call, making single-thread hot-path changes and
// the zero-allocation contract visible without conflating them with pool churn.
func BenchmarkGoav1CorpusDecodeSteadyState(b *testing.B) {
	dir := os.Getenv(envCorpusBenchmarkDir)
	if dir == "" {
		_, filename, _, ok := runtime.Caller(0)
		if !ok {
			b.Fatal("resolve corpus benchmark source")
		}
		dir = filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", "testdata", "benchcorpus"))
	}
	clips, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		b.Fatal(err)
	}
	if len(clips) == 0 {
		b.Skipf("no corpus clips in %s", dir)
	}
	for _, path := range clips {
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		b.Run(name, func(b *testing.B) {
			data, err := os.ReadFile(path)
			if err != nil {
				b.Fatal(err)
			}
			dec, err := av1.NewDecoderFromIVF(data, av1.WithWorkers(1))
			if err != nil {
				b.Fatal(err)
			}
			defer dec.Close()

			run := func() int {
				if err := dec.Reset(); err != nil {
					b.Fatal(err)
				}
				frames := 0
				for {
					decoded, ok, err := dec.DecodeNext()
					if err != nil {
						b.Fatal(err)
					}
					if !ok {
						return frames
					}
					frames += len(decoded)
				}
			}
			wantFrames := run()
			if wantFrames == 0 {
				b.Fatal("decoded no visible frames")
			}

			b.SetBytes(int64(len(data)))
			b.ReportAllocs()
			b.ResetTimer()
			frames := 0
			for i := 0; i < b.N; i++ {
				decoded := run()
				if decoded != wantFrames {
					b.Fatalf("decoded %d visible frames, want %d", decoded, wantFrames)
				}
				frames += decoded
			}
			b.StopTimer()
			corpusBenchmarkFramesSink = frames
			b.ReportMetric(float64(wantFrames), "frames/op")
			if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
				b.ReportMetric(float64(frames)/elapsed, "frames/s")
			}
		})
	}
}
