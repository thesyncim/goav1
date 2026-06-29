package encoder_test

import (
	"os"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// benchCorpus measures steady-state back-to-back encoding of one local corpus
// segment. It is a goav1 hot-path profiler, not a publishable external
// comparison; use cmd/qualitybench -publish for SVT-AV1/libaom tables.
func benchCorpus(b *testing.B, path string, fps, bitrate int) {
	raw, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	if len(raw)%fl != 0 {
		b.Fatalf("%s size=%d is not an exact number of 1080p I420 frames", path, len(raw))
	}
	nf := len(raw) / fl
	if nf < 2 {
		b.Fatalf("%s has %d frame(s), want at least 2", path, nf)
	}
	frames := make([]encoder.SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = encoder.SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: bitrate, FramesPerSecond: fps, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		_ = enc.Close()
	})
	if err := enc.SetTemporalLayers(2); err != nil {
		b.Fatal(err)
	}
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(frames[0], true); err != nil {
		b.Fatal(err)
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := enc.Encode(frames[1+i%(nf-1)], false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVideoEncoderRealC1080p is the complex-motion segment where search
// and residual coding dominate.
func BenchmarkVideoEncoderRealC1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/realC.yuv", 30, 5_000_000)
}

// BenchmarkVideoEncoderRealA1080p is the moderate camera segment.
func BenchmarkVideoEncoderRealA1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/realA.yuv", 30, 5_000_000)
}

// BenchmarkVideoEncoderScreen1080p is the screen-content segment.
func BenchmarkVideoEncoderScreen1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/screen.yuv", 60, 1_330_000)
}
