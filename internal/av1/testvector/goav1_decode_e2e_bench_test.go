//go:build goav1_oracle

package testvector

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// BenchmarkGoav1DecodeE2E measures full in-process goav1 decode (parse + tile
// decode + post-filter) over the bench corpus clips. Run under GOEXPERIMENT=simd
// vs production to see the SIMD pipeline's real decode impact.
func BenchmarkGoav1DecodeE2E(b *testing.B) {
	_, filename, _, _ := runtime.Caller(0)
	dir := filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", "testdata", "benchcorpus"))
	clips, _ := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if len(clips) == 0 {
		b.Skip("no corpus clips")
	}
	for _, clip := range clips {
		data, err := os.ReadFile(clip)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(filepath.Base(clip), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := decodeCorpusClipDiscard(data); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
