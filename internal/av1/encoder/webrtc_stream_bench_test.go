package encoder_test

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

var webRTCStreamBenchmarkSink int

func BenchmarkWebRTCStreamEncodePictureModes(b *testing.B) {
	modes := []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeL2T3,
		encoder.ScalabilityModeL2T3_KEY,
		encoder.ScalabilityModeL2T3_KEY_SHIFT,
		encoder.ScalabilityModeS2T3,
		encoder.ScalabilityModeL3T3,
		encoder.ScalabilityModeL3T3_KEY,
		encoder.ScalabilityModeL3T3_KEY_SHIFT,
		encoder.ScalabilityModeS3T3h,
	}
	for _, mode := range modes {
		b.Run(mode.String(), func(b *testing.B) {
			cfg := webRTCDecodeMatrixConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				b.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			b.Cleanup(func() { _ = stream.Close() })
			frames := make([]encoder.SourceFrame420, 8)
			for i := range frames {
				frames[i] = webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), i)
			}
			warmWebRTCStreamBenchmark(b, stream, frames, mode)

			b.SetBytes(webRTCBenchmarkPictureBytes(stream.Config()))
			b.ReportAllocs()
			b.ResetTimer()
			sum := 0
			for i := 0; b.Loop(); i++ {
				picture, err := stream.EncodePicture(frames[1+i%7], false)
				if err != nil {
					b.Fatalf("EncodePicture(%s): %v", mode, err)
				}
				for frame := uint8(0); frame < picture.FrameNum; frame++ {
					sum += len(picture.Frames[frame].TU)
				}
			}
			webRTCStreamBenchmarkSink += sum
		})
	}
}

func BenchmarkWebRTCStreamEncodePicture1080p(b *testing.B) {
	for _, mode := range []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeS3T3,
	} {
		b.Run(mode.String(), func(b *testing.B) {
			cfg := webRTC1080pAllocConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				b.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			b.Cleanup(func() { _ = stream.Close() })
			frames := []encoder.SourceFrame420{
				makeEncoder1080pFrame(0),
				makeEncoder1080pFrame(1),
				makeEncoder1080pFrame(2),
				makeEncoder1080pFrame(3),
			}
			warmWebRTCStreamBenchmark(b, stream, frames, mode)

			b.SetBytes(webRTCBenchmarkPictureBytes(stream.Config()))
			b.ReportAllocs()
			b.ResetTimer()
			sum := 0
			for i := 0; b.Loop(); i++ {
				picture, err := stream.EncodePicture(frames[i&3], false)
				if err != nil {
					b.Fatalf("EncodePicture(%s): %v", mode, err)
				}
				for frame := uint8(0); frame < picture.FrameNum; frame++ {
					sum += len(picture.Frames[frame].TU)
				}
			}
			webRTCStreamBenchmarkSink += sum
		})
	}
}

func warmWebRTCStreamBenchmark(b *testing.B, stream *encoder.WebRTCStream, frames []encoder.SourceFrame420, mode encoder.ScalabilityMode) {
	b.Helper()
	if err := stream.Prewarm(); err != nil {
		b.Fatalf("Prewarm(%s): %v", mode, err)
	}
	for i := 0; i < len(frames)*2; i++ {
		forceKey := i == 0 || i == len(frames)
		picture, err := stream.EncodePicture(frames[i%len(frames)], forceKey)
		if err != nil {
			b.Fatalf("warm EncodePicture(%s, %d): %v", mode, i, err)
		}
		if picture.FrameNum == 0 {
			b.Fatalf("warm EncodePicture(%s, %d) emitted no frames", mode, i)
		}
	}
}

func webRTCBenchmarkPictureBytes(config encoder.Config) int64 {
	var bytes int64
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		resolution := config.SpatialLayers[i].Resolution
		bytes += int64(resolution.Width) * int64(resolution.Height) * 3 / 2
	}
	return bytes
}
