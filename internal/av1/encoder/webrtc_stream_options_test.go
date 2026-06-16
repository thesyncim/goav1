package encoder

import (
	"errors"
	"testing"
)

func TestWebRTCStreamEncoderOptions(t *testing.T) {
	stream, err := NewWebRTCStreamLayers(192, 128, RateControlConfig{
		TargetBitsPerSecond: 500_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	stream.SetGoldenInterval(0)
	if got := stream.encoders[0].goldenEvery; got != 0 {
		t.Fatalf("golden interval=%d want 0", got)
	}
	stream.SetTileColumns(4)
	if got := stream.encoders[0].tileColsLog2; got != 2 {
		t.Fatalf("tileColsLog2=%d want 2", got)
	}
	if err := stream.Prewarm(); err != nil {
		t.Fatalf("prewarm: %v", err)
	}
	if stream.state.NextFrameID != 0 || stream.state.DeltaPictureIndex != 0 || stream.encoders[0].haveKey {
		t.Fatalf("prewarm advanced stream state: frameID=%d deltaPictureIndex=%d haveKey=%v", stream.state.NextFrameID, stream.state.DeltaPictureIndex, stream.encoders[0].haveKey)
	}
}

func TestWebRTCStreamConfigRejectsDeltaInterLayerPixelModes(t *testing.T) {
	_, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	})
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("NewWebRTCStreamConfig L2T2 err=%v want %v", err, ErrUnsupported)
	}
}

func TestWebRTCStreamConfigAcceptsSimulcastPixelModes(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig S2T2: %v", err)
	}
	if stream.config.Scalability != ScalabilityModeS2T2 || stream.config.SpatialLayerCount != 2 || stream.config.TemporalLayerCount != 2 {
		t.Fatalf("config mode=%s spatial=%d temporal=%d", stream.config.Scalability, stream.config.SpatialLayerCount, stream.config.TemporalLayerCount)
	}
}

func TestWebRTCStreamConfigPixelScalabilityModeMatrix(t *testing.T) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		t.Run(mode.String(), func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(Config{
				Resolution:        Resolution{Width: 1296, Height: 720},
				MaxFramerate:      Rational{Num: 30, Den: 1},
				MinBitrateKbps:    100,
				MaxBitrateKbps:    1200,
				TargetBitrateKbps: 800,
				Scalability:       mode,
			})
			spatial, _, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("mode %d invalid in matrix", mode)
			}
			wantUnsupported := spatial > 1 && !mode.IsSimulcast() && !mode.UsesKeyFrameInterLayerDependency()
			if wantUnsupported {
				if !errors.Is(err, ErrUnsupported) {
					t.Fatalf("NewWebRTCStreamConfig(%s) err=%v want %v", mode, err, ErrUnsupported)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			if stream.config.Scalability != mode {
				t.Fatalf("NewWebRTCStreamConfig(%s) normalized to %s", mode, stream.config.Scalability)
			}
		})
	}
}
