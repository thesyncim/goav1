//go:build goav1_oracle

package testvector

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var corpusBenchmarkFramesSink int

const envCorpusBenchmarkDir = "GOAV1_BENCH_CORPUS_DIR"

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
