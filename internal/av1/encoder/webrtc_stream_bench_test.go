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
			frames := make([]encoder.SourceFrame420, 8)
			for i := range frames {
				frames[i] = webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), i)
			}
			if _, err := stream.EncodePicture(frames[0], false); err != nil {
				b.Fatalf("key EncodePicture(%s): %v", mode, err)
			}

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

func webRTCBenchmarkPictureBytes(config encoder.Config) int64 {
	var bytes int64
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		resolution := config.SpatialLayers[i].Resolution
		bytes += int64(resolution.Width) * int64(resolution.Height) * 3 / 2
	}
	return bytes
}
