package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
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

func TestWebRTCStreamEncoderOptionsSurviveSetConfig(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	stream.SetGoldenInterval(0)
	stream.SetTileColumns(4)

	change := stream.Config()
	change.Scalability = ScalabilityModeS2T2
	change.MinBitrateKbps = 120
	change.MaxBitrateKbps = 1200
	change.TargetBitrateKbps = 760
	change.SpatialLayers[0].MinBitrateKbps = 120
	change.SpatialLayers[0].MaxBitrateKbps = 480
	change.SpatialLayers[0].TargetBitrateKbps = 260
	change.SpatialLayers[1].MinBitrateKbps = 300
	change.SpatialLayers[1].MaxBitrateKbps = 1200
	change.SpatialLayers[1].TargetBitrateKbps = 760
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig simulcast: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		if got := stream.encoders[i].goldenEvery; got != 0 {
			t.Fatalf("simulcast layer %d golden interval=%d want 0", i, got)
		}
		if got := stream.encoders[i].tileColsLog2; got != 2 {
			t.Fatalf("simulcast layer %d tileColsLog2=%d want 2", i, got)
		}
	}

	change = stream.Config()
	change.Scalability = ScalabilityModeL2T2
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig shared SVC: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		if got := stream.encoders[i].goldenEvery; got != 0 {
			t.Fatalf("shared SVC layer %d golden interval=%d want 0", i, got)
		}
		if got := stream.encoders[i].tileColsLog2; got != 2 {
			t.Fatalf("shared SVC layer %d tileColsLog2=%d want 2", i, got)
		}
	}
}

func TestWebRTCStreamConfigMaxThreadsControlsTileColumns(t *testing.T) {
	const w, h = 640, 360
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		MaxThreads:        1,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()
	if got := stream.encoders[0].tileColsLog2; got != 0 {
		t.Fatalf("initial tileColsLog2=%d want 0", got)
	}
	if !stream.encoders[0].singleThread {
		t.Fatal("initial MaxThreads=1 did not enable single-thread mode")
	}
	picture, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("initial EncodePicture: %v", err)
	}
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 1 {
		t.Fatalf("initial encoded tile columns=%d want 1", got)
	}

	change := stream.Config()
	change.MaxThreads = 4
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig MaxThreads=4: %v", err)
	}
	if got := stream.encoders[0].tileColsLog2; got != 2 {
		t.Fatalf("updated tileColsLog2=%d want 2", got)
	}
	if stream.encoders[0].singleThread {
		t.Fatal("updated MaxThreads=4 left single-thread mode enabled")
	}
	picture, err = stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("updated EncodePicture: %v", err)
	}
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 4 {
		t.Fatalf("updated encoded tile columns=%d want 4", got)
	}

	change = stream.Config()
	change.MaxThreads = 0
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig MaxThreads=0: %v", err)
	}
	if got, want := stream.encoders[0].tileColsLog2, defaultTileColsLog2(w); got != want {
		t.Fatalf("defaulted tileColsLog2=%d want %d", got, want)
	}
	if stream.encoders[0].singleThread {
		t.Fatal("defaulted MaxThreads=0 left single-thread mode enabled")
	}
	picture, err = stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("defaulted EncodePicture: %v", err)
	}
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 2 {
		t.Fatalf("defaulted encoded tile columns=%d want 2", got)
	}
}

func TestWebRTCStreamConfigMaxThreadsAppliesToPixelEncoders(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		MaxThreads:        4,
	}
	for _, tc := range []struct {
		name   string
		config func(Config) Config
		assert func(*testing.T, *WebRTCStream)
	}{
		{
			name: "i420-8",
			config: func(cfg Config) Config {
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.encoders[0] == nil {
					t.Fatal("missing i420 encoder")
				}
				if got := stream.encoders[0].tileColsLog2; got != 2 {
					t.Fatalf("i420 encoder tileColsLog2=%d want 2", got)
				}
			},
		},
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.monoEncoders[0] == nil {
					t.Fatal("missing mono encoder")
				}
				if got := stream.monoEncoders[0].tileColsLog2; got != 2 {
					t.Fatalf("mono encoder tileColsLog2=%d want 2", got)
				}
			},
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].tileColsLog2; got != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", got)
				}
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].tileColsLog2; got != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", got)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(tc.config(base))
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			tc.assert(t, stream)
		})
	}
}

func TestWebRTCStreamTileColumnsSurvivePixelFormatReplacement(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
	}

	for _, tc := range []struct {
		name      string
		config    func(Config) Config
		assert    func(*testing.T, *WebRTCStream)
		encode    func(*testing.T, *WebRTCStream) WebRTCEncodedPicture
		wantTiles int
	}{
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.monoEncoders[0]
				if enc == nil {
					t.Fatal("missing mono encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeMonochromePicture(testWebRTCStreamMonoFrame(w, h), false)
				if err != nil {
					t.Fatalf("EncodeMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.mono16Encoders[0]
				if enc == nil {
					t.Fatal("missing mono16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i400-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.mono16Encoders[0]
				if enc == nil {
					t.Fatal("missing mono16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 12), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.color16Encoders[0]
				if enc == nil {
					t.Fatal("missing color16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i420-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.color16Encoders[0]
				if enc == nil {
					t.Fatal("missing color16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 12), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(base)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			stream.SetTileColumns(4)

			if err := stream.SetConfig(tc.config(base)); err != nil {
				t.Fatalf("SetConfig: %v", err)
			}
			tc.assert(t, stream)
			picture := tc.encode(t, stream)
			if !picture.Keyframe || picture.FrameNum != 1 {
				t.Fatalf("picture key=%v frameNum=%d want key single frame", picture.Keyframe, picture.FrameNum)
			}
			if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != tc.wantTiles {
				t.Fatalf("encoded tile columns=%d want %d", got, tc.wantTiles)
			}
		})
	}
}

func TestWebRTCStreamConfigAcceptsFullSVCPixelModes(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig L2T2: %v", err)
	}
	if stream.config.Scalability != ScalabilityModeL2T2 || stream.config.SpatialLayerCount != 2 {
		t.Fatalf("config mode=%s spatial=%d", stream.config.Scalability, stream.config.SpatialLayerCount)
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

func TestWebRTCStreamEncodeRejectsMultiSpatialWithoutMutating(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig L2T2: %v", err)
	}
	if _, err := stream.Encode(testWebRTCStreamFrame(640, 360), false); err != ErrUnsupported {
		t.Fatalf("Encode multi-spatial err=%v want %v", err, ErrUnsupported)
	}
	if stream.state.NextFrameID != 0 || stream.state.DependencyStructureState.Valid {
		t.Fatalf("unsupported Encode mutated state: %+v", stream.state)
	}
	picture, err := stream.EncodePicture(testWebRTCStreamFrame(640, 360), false)
	if err != nil {
		t.Fatalf("EncodePicture after rejected Encode: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != 0 ||
		picture.Frames[1].Info.FrameID != 1 {
		t.Fatalf("picture after rejected Encode=%+v", picture)
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
			if _, _, _, ok := mode.Layers(); !ok {
				t.Fatalf("mode %d invalid in matrix", mode)
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
	wantPerFrameBits := int(controlChange.TargetBitrateKbps) * 1000 / webRTCStreamFramesPerSecond(controlChange.MaxFramerate)
	if got, want := stream.encoders[0].rcPerFrameBits, wantPerFrameBits; got != want {
		t.Fatalf("control change rcPerFrameBits=%d want %d", got, want)
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
	structureChange.SpatialLayers[0].MinBitrateKbps = 120
	structureChange.SpatialLayers[0].MaxBitrateKbps = 480
	structureChange.SpatialLayers[0].TargetBitrateKbps = 240
	structureChange.SpatialLayers[1].MinBitrateKbps = 300
	structureChange.SpatialLayers[1].MaxBitrateKbps = 1400
	structureChange.SpatialLayers[1].TargetBitrateKbps = 840
	if err := stream.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	for i, wantTarget := range [...]int32{240, 840} {
		if got := stream.config.SpatialLayers[i].TargetBitrateKbps; got != wantTarget {
			t.Fatalf("layer %d target bitrate=%d want %d", i, got, wantTarget)
		}
		wantBits := int(wantTarget) * 1000 / webRTCStreamFramesPerSecond(structureChange.MaxFramerate)
		if got := stream.encoders[i].rcPerFrameBits; got != wantBits {
			t.Fatalf("layer %d rcPerFrameBits=%d want %d", i, got, wantBits)
		}
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

	fullSVC := structureChange
	fullSVC.Scalability = ScalabilityModeL2T2
	if err := stream.SetConfig(fullSVC); err != nil {
		t.Fatalf("SetConfig full SVC: %v", err)
	}
	beforeFrameID = stream.state.NextFrameID
	picture, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key after full SVC change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeFrameID ||
		picture.Frames[1].Info.FrameID != beforeFrameID+1 ||
		stream.state.NextFrameID != beforeFrameID+2 {
		t.Fatalf("full SVC change picture=%+v state=%+v", picture, stream.state)
	}
}

func TestWebRTCStreamSetConfigRateControlTransitions(t *testing.T) {
	const w, h = 640, 360
	cfg := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()

	src := testWebRTCStreamFrame(w, h)
	if _, err := stream.EncodePicture(src, false); err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	if _, err := stream.EncodePicture(src, false); err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}

	toCQP := stream.Config()
	toCQP.RateControl = RateControlCQP
	toCQP.Quantizer = 37
	beforeState := stream.state
	if err := stream.SetConfig(toCQP); err != nil {
		t.Fatalf("SetConfig CQP: %v", err)
	}
	if stream.state != beforeState {
		t.Fatalf("CQP control transition reset state: before=%+v after=%+v", beforeState, stream.state)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		enc := stream.encoders[i]
		if enc == nil || enc.rcEnabled || enc.rcPerFrameBits != 0 || enc.qIndex != 37 {
			t.Fatalf("layer %d after CQP transition encoder=%+v", i, enc)
		}
	}
	picture, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after CQP transition: %v", err)
	}
	if picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeState.NextFrameID ||
		picture.Frames[1].Info.FrameID != beforeState.NextFrameID+1 {
		t.Fatalf("CQP transition picture=%+v before=%+v state=%+v", picture, beforeState, stream.state)
	}

	toCBR := stream.Config()
	toCBR.RateControl = RateControlCBR
	toCBR.Quantizer = 0
	toCBR.MaxFramerate = Rational{Num: 60, Den: 1}
	for i := uint8(0); i < toCBR.SpatialLayerCount; i++ {
		layer := &toCBR.SpatialLayers[i]
		layer.MinBitrateKbps = 120 + int32(i)*180
		layer.MaxBitrateKbps = 620 + int32(i)*420
		layer.TargetBitrateKbps = layer.MinBitrateKbps + (layer.MaxBitrateKbps-layer.MinBitrateKbps)/2
	}
	beforeState = stream.state
	if err := stream.SetConfig(toCBR); err != nil {
		t.Fatalf("SetConfig CBR: %v", err)
	}
	if stream.state != beforeState {
		t.Fatalf("CBR control transition reset state: before=%+v after=%+v", beforeState, stream.state)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		enc := stream.encoders[i]
		wantBits := int(stream.config.SpatialLayers[i].TargetBitrateKbps) * 1000 / webRTCStreamFramesPerSecond(stream.config.MaxFramerate)
		if enc == nil || !enc.rcEnabled || enc.rcPerFrameBits != wantBits {
			t.Fatalf("layer %d after CBR transition rc=%v bits=%d want %d", i, enc != nil && enc.rcEnabled, encoderPerFrameBits(enc), wantBits)
		}
	}
	picture, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after CBR transition: %v", err)
	}
	if picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeState.NextFrameID ||
		picture.Frames[1].Info.FrameID != beforeState.NextFrameID+1 {
		t.Fatalf("CBR transition picture=%+v before=%+v state=%+v", picture, beforeState, stream.state)
	}
}

func TestWebRTCStreamSetConfigRejectsInvalidWithoutMutation(t *testing.T) {
	const w, h = 640, 360
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()
	if _, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), false); err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}

	type layerSnapshot struct {
		rcEnabled      bool
		rcPerFrameBits int
		qIndex         uint8
		temporalLayers int
		haveKey        bool
	}
	snapshotLayers := func() [WebRTCMaxSpatialLayers]layerSnapshot {
		var out [WebRTCMaxSpatialLayers]layerSnapshot
		for i := uint8(0); i < WebRTCMaxSpatialLayers; i++ {
			enc := stream.encoders[i]
			if enc == nil {
				continue
			}
			out[i] = layerSnapshot{
				rcEnabled:      enc.rcEnabled,
				rcPerFrameBits: enc.rcPerFrameBits,
				qIndex:         enc.qIndex,
				temporalLayers: enc.temporalLayers,
				haveKey:        enc.haveKey,
			}
		}
		return out
	}

	keepConfig := stream.config
	keepState := stream.state
	keepLayers := snapshotLayers()
	for _, tc := range []struct {
		name string
		cfg  Config
		want error
	}{
		{
			name: "invalid-target",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.TargetBitrateKbps = cfg.MaxBitrateKbps + 1
				return cfg
			}(),
			want: ErrInvalidConfig,
		},
		{
			name: "invalid-fps",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.MaxFramerate = Rational{Num: 0, Den: 1}
				return cfg
			}(),
			want: ErrInvalidConfig,
		},
		{
			name: "unsupported-cqp-zero",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.RateControl = RateControlCQP
				cfg.Quantizer = 0
				return cfg
			}(),
			want: ErrUnsupported,
		},
	} {
		if err := stream.SetConfig(tc.cfg); !errors.Is(err, tc.want) {
			t.Fatalf("%s SetConfig err=%v want %v", tc.name, err, tc.want)
		}
		if stream.config != keepConfig || stream.state != keepState || snapshotLayers() != keepLayers {
			t.Fatalf("%s mutated stream config=%+v state=%+v layers=%+v", tc.name, stream.config, stream.state, snapshotLayers())
		}
	}
}

func encoderPerFrameBits(enc *VideoEncoder) int {
	if enc == nil {
		return 0
	}
	return enc.rcPerFrameBits
}

func webRTCStreamDescriptorMatrixConfig(mode ScalabilityMode) (Config, bool) {
	spatial, _, _, ok := mode.Layers()
	if !ok {
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

func testWebRTCStreamMonoFrame(width int, height int) SourceFrameMono {
	f := SourceFrameMono{
		Y:       make([]byte, width*height),
		YStride: width,
		Width:   width,
		Height:  height,
	}
	for i := range f.Y {
		f.Y[i] = uint8(40 + i%170)
	}
	return f
}

func testWebRTCStreamMono16Frame(width int, height int, bitDepth uint8) SourceFrameMono16 {
	f := SourceFrameMono16{
		Y:        make([]uint16, width*height),
		YStride:  width,
		Width:    width,
		Height:   height,
		BitDepth: bitDepth,
	}
	base := uint16(1 << max(bitDepth-4, 0))
	span := uint16((1 << bitDepth) - 1)
	for i := range f.Y {
		f.Y[i] = (base + uint16((i*17)%int(span/2))) & span
	}
	return f
}

func testWebRTCStream42016Frame(width int, height int, bitDepth uint8) SourceFrame42016 {
	cw, ch := width/2, height/2
	f := SourceFrame42016{
		Y:            make([]uint16, width*height),
		U:            make([]uint16, cw*ch),
		V:            make([]uint16, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
		BitDepth:     bitDepth,
	}
	maxSample := uint16((1 << bitDepth) - 1)
	for i := range f.Y {
		f.Y[i] = uint16((64 + i%512) & int(maxSample))
	}
	for i := range f.U {
		f.U[i] = maxSample / 3
		f.V[i] = maxSample * 2 / 3
	}
	return f
}

func parseWebRTCTileColumns(t *testing.T, tu []byte) int {
	t.Helper()
	var seq parser.SequenceHeader
	haveSeq := false
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			seq, err = parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			haveSeq = true
		case obu.TypeFrameHeader, obu.TypeFrame:
			if !haveSeq {
				t.Fatal("frame header before sequence header")
			}
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			var size parser.FrameSize
			if prefix.UsesIntraFrameSizePath() {
				size, err = parser.ParseIntraFrameSize(unit.Payload, seq, prefix, unit.Header.TemporalID, unit.Header.SpatialID)
			} else {
				size, err = parser.ParseFrameSize(unit.Payload, seq, prefix, nil, unit.Header.TemporalID, unit.Header.SpatialID)
			}
			if err != nil {
				t.Fatalf("ParseFrameSize: %v", err)
			}
			tiles, err := parser.ParseTileInfo(unit.Payload, seq, prefix, size)
			if err != nil {
				t.Fatalf("ParseTileInfo: %v", err)
			}
			return int(tiles.Cols)
		}
	}
	t.Fatal("missing frame header")
	return 0
}
