package encoder

import (
	"errors"
	"testing"

	internalrtp "github.com/thesyncim/goav1/internal/av1/rtp"
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

func TestWebRTCStreamTracksReferenceSurfaces(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	if _, err := stream.EncodePicture(testWebRTCStreamFrame(640, 360), false); err != nil {
		t.Fatalf("EncodePicture key: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		ref := stream.referenceFrames[i]
		enc := stream.encoders[i]
		if enc == nil {
			t.Fatalf("missing layer encoder %d", i)
		}
		if ref.Y == nil || ref.Width != enc.width || ref.Height != enc.height {
			t.Fatalf("reference buffer %d geometry=%dx%d valid=%v want %dx%d",
				i, ref.Width, ref.Height, ref.Y != nil, enc.width, enc.height)
		}
	}

	controlChange := stream.config
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.TargetBitrateKbps += 50
	if err := stream.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	if stream.referenceFrames[0].Y == nil || stream.referenceFrames[1].Y == nil {
		t.Fatal("control-only change cleared reference surfaces")
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T3
	if err := stream.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	for i, ref := range stream.referenceFrames {
		if ref.Y != nil {
			t.Fatalf("reference buffer %d survived structure reset", i)
		}
	}
}

func TestWebRTCStreamDescriptorStateMatrix(t *testing.T) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		t.Run(mode.String(), func(t *testing.T) {
			cfg, supported := webRTCStreamDescriptorMatrixConfig(mode)
			if !supported {
				t.Skip("pixel stream intentionally rejects delta inter-layer SVC")
			}
			stream, err := NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			src := testWebRTCStreamFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height))
			var receiver internalrtp.DependencyDescriptorState

			key, err := stream.EncodePicture(src, false)
			if err != nil {
				t.Fatalf("key EncodePicture: %v", err)
			}
			if !key.Keyframe {
				t.Fatalf("first picture is not key: %+v", key)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, key)

			controlChange := stream.Config()
			controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps += 10
			controlChange.MaxBitrateKbps += 200
			controlChange.TargetBitrateKbps = (controlChange.MinBitrateKbps + controlChange.MaxBitrateKbps) / 2
			beforeFrameID := stream.state.NextFrameID
			if err := stream.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig control change: %v", err)
			}
			delta, err := stream.EncodePicture(src, false)
			if err != nil {
				t.Fatalf("delta EncodePicture after control change: %v", err)
			}
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("control change forced key or skipped id: picture=%+v before=%d", delta, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)
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

func webRTCStreamDescriptorMatrixConfig(mode ScalabilityMode) (Config, bool) {
	spatial, _, _, ok := mode.Layers()
	if !ok {
		return Config{}, false
	}
	if spatial > 1 && !mode.IsSimulcast() && !mode.UsesKeyFrameInterLayerDependency() {
		return Config{}, false
	}
	cfg := Config{
		Resolution:        Resolution{Width: 192, Height: 128},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    80,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
		Scalability:       mode,
	}
	switch spatial {
	case 2:
		cfg.Resolution = Resolution{Width: 640, Height: 360}
		cfg.MinBitrateKbps = 100
		cfg.MaxBitrateKbps = 1000
		cfg.TargetBitrateKbps = 650
	case 3:
		cfg.Resolution = Resolution{Width: 1008, Height: 576}
		cfg.MinBitrateKbps = 150
		cfg.MaxBitrateKbps = 1800
		cfg.TargetBitrateKbps = 1100
	}
	return cfg, true
}

func assertWebRTCStreamPictureDependencyDescriptors(t *testing.T, receiver *internalrtp.DependencyDescriptorState, picture WebRTCEncodedPicture) {
	t.Helper()
	if picture.FrameNum == 0 || picture.FrameNum > WebRTCMaxSpatialLayers {
		t.Fatalf("invalid picture frame num=%d", picture.FrameNum)
	}
	for i := uint8(0); i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		parsed, consumed, err := receiver.Parse(frame.Descriptor)
		if err != nil {
			t.Fatalf("frame %d dependency descriptor parse: %v", i, err)
		}
		if consumed != len(frame.Descriptor) ||
			parsed.Mandatory.FrameNumber != uint16(frame.Info.FrameID) ||
			!parsed.Mandatory.FirstPacketInFrame ||
			!parsed.Mandatory.LastPacketInFrame {
			t.Fatalf("frame %d parsed mandatory=%+v consumed=%d len=%d info=%+v", i, parsed.Mandatory, consumed, len(frame.Descriptor), frame.Info)
		}
		if parsed.HasAttachedStructure != frame.AttachDependencyStructure {
			t.Fatalf("frame %d attached=%v want %v", i, parsed.HasAttachedStructure, frame.AttachDependencyStructure)
		}
		if frame.AttachDependencyStructure &&
			(!receiver.Valid ||
				parsed.AttachedStructure.NumDecodeTargets != frame.Structure.NumDecodeTargets ||
				parsed.AttachedStructure.NumChains != frame.Structure.NumChains ||
				parsed.AttachedStructure.TemplateNum != frame.Structure.TemplateNum ||
				parsed.AttachedStructure.ResolutionNum != frame.Structure.ResolutionNum) {
			t.Fatalf("frame %d attached structure parsed=%+v encoder=%+v receiver=%+v", i, parsed.AttachedStructure, frame.Structure, receiver)
		}
		deps := parsed.FrameDependencies
		if deps.SpatialID != frame.Info.SpatialID || deps.TemporalID != frame.Info.TemporalID ||
			deps.DTINum != frame.Info.DTINum || deps.FrameDiffNum != frame.Info.DependencyNum {
			t.Fatalf("frame %d deps=%+v info=%+v", i, deps, frame.Info)
		}
		for target := uint8(0); target < frame.Info.DTINum; target++ {
			if deps.DTIs[target] != internalrtp.DependencyDescriptorDecodeTargetIndication(frame.Info.DTIs[target]) {
				t.Fatalf("frame %d target %d dti=%v want %v", i, target, deps.DTIs[target], frame.Info.DTIs[target])
			}
		}
		for dep := uint8(0); dep < frame.Info.DependencyNum; dep++ {
			wantDiff := uint16(frame.Info.FrameID - frame.Info.Dependencies[dep])
			if deps.FrameDiffs[dep] != wantDiff {
				t.Fatalf("frame %d dep %d diff=%d want %d info=%+v", i, dep, deps.FrameDiffs[dep], wantDiff, frame.Info)
			}
		}
		if parsed.HasResolution {
			resolution := frame.Structure.Resolutions[frame.Info.SpatialID]
			if parsed.Resolution.Width != uint16(resolution.Width) ||
				parsed.Resolution.Height != uint16(resolution.Height) {
				t.Fatalf("frame %d resolution=%+v want %+v", i, parsed.Resolution, resolution)
			}
		}
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
