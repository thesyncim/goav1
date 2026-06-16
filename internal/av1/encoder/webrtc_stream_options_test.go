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

func TestWebRTCStreamSetConfigReconfigure(t *testing.T) {
	const w, h = 640, 360
	cfg := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	src := testWebRTCStreamFrame(w, h)
	key, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	delta, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	if !key.Keyframe || delta.Keyframe || key.FrameNum != 1 || delta.FrameNum != 1 || stream.state.NextFrameID != 2 {
		t.Fatalf("warm key=%+v delta=%+v state=%+v", key, delta, stream.state)
	}

	controlChange := cfg
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1200
	controlChange.TargetBitrateKbps = 900
	if err := stream.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	beforeFrameID := stream.state.NextFrameID
	delta, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after control change: %v", err)
	}
	if delta.Keyframe || delta.FrameNum != 1 || delta.Frames[0].Info.FrameID != beforeFrameID ||
		stream.config.TargetBitrateKbps != 900 || stream.state.NextFrameID != beforeFrameID+1 {
		t.Fatalf("control change delta=%+v state=%+v config=%+v", delta, stream.state, stream.config)
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T2
	if err := stream.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	beforeFrameID = stream.state.NextFrameID
	picture, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key after structure change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeFrameID ||
		picture.Frames[1].Info.FrameID != beforeFrameID+1 ||
		stream.state.NextFrameID != beforeFrameID+2 {
		t.Fatalf("structure change picture=%+v state=%+v", picture, stream.state)
	}

	keepConfig := stream.config
	keepState := stream.state
	bad := structureChange
	bad.Scalability = ScalabilityModeL2T2
	if err := stream.SetConfig(bad); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("SetConfig unsupported err=%v want %v", err, ErrUnsupported)
	}
	if stream.config != keepConfig || stream.state != keepState {
		t.Fatalf("unsupported SetConfig mutated config/state config=%+v want=%+v state=%+v want=%+v", stream.config, keepConfig, stream.state, keepState)
	}
}

func testWebRTCStreamFrame(width int, height int) SourceFrame420 {
	cw, ch := width/2, height/2
	f := SourceFrame420{
		Y: make([]byte, width*height), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: width, ChromaStride: cw, Width: width, Height: height,
	}
	for i := range f.Y {
		f.Y[i] = uint8(40 + i%170)
	}
	for i := range f.U {
		f.U[i] = 121
		f.V[i] = 133
	}
	return f
}
