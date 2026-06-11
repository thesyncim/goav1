package encoder_test

import (
	"os"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// benchCorpus measures steady-state back-to-back encoding of one corpus
// segment - the honest throughput comparison against external encoders
// (no harness work between Encode calls, so the background filter pass
// cannot hide in an untimed gap).
func benchCorpus(b *testing.B, path string, fps, bitrate int) {
	raw, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
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
	if err := enc.SetTemporalLayers(2); err != nil {
		b.Fatal(err)
	}
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(frames[0], true); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
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
