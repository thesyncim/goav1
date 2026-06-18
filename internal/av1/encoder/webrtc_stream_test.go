package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestWebRTCStreamPackagesDecodableFrames is the WebRTC integration gate: an
// L1T1 stream must emit, per frame, a decodable temporal unit plus coherent
// dependency metadata — the keyframe carrying the attached structure with no
// dependencies, every delta frame depending on the previous frame ID — and
// the temporal units must decode bit-identical to the encoder reconstruction.
func TestWebRTCStreamPackagesDecodableFrames(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 132
		}
		sx, sy := 12+t*4, 20+t*2
		for y := sy; y < sy+20 && y < h; y++ {
			for x := sx; x < sx+20 && x < w; x++ {
				f.Y[y*w+x] = 210
			}
		}
		return f
	}

	stream, err := encoder.NewWebRTCStream(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 150_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}

	const frames = 5
	tus := make([][]byte, 0, frames)
	for i := range frames {
		ef, err := stream.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != ef.Keyframe {
			t.Fatalf("frame %d keyframe=%v", i, ef.Keyframe)
		}
		if ef.Info.FrameID != uint64(i) {
			t.Fatalf("frame %d id=%d", i, ef.Info.FrameID)
		}
		if i == 0 {
			if ef.Info.DependencyNum != 0 {
				t.Fatalf("keyframe has %d dependencies", ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) < 8 {
				t.Fatalf("keyframe descriptor %d bytes: structure not attached?", len(ef.Descriptor))
			}
		} else {
			if ef.Info.DependencyNum != 1 || ef.Info.Dependencies[0] != uint64(i-1) {
				t.Fatalf("frame %d deps=%v num=%d", i, ef.Info.Dependencies, ef.Info.DependencyNum)
			}
			if len(ef.Descriptor) == 0 {
				t.Fatalf("frame %d missing descriptor", i)
			}
		}
		tus = append(tus, append([]byte(nil), ef.TU...))
	}

	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	n := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !ok {
			break
		}
		n += len(batch)
	}
	if n != frames {
		t.Fatalf("decoded %d frames, want %d", n, frames)
	}
	fmt.Println("webrtc stream ok")
}

func TestWebRTCStreamAcceptedScalabilityModesDecode(t *testing.T) {
	for _, mode := range acceptedWebRTCStreamPixelModes() {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := webRTCDecodeMatrixConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			stream.SetGoldenInterval(0)

			var tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte
			var wantBySpatial [encoder.WebRTCMaxSpatialLayers]int
			var tusInOrder [][]byte
			for i := 0; i < 2; i++ {
				picture, err := stream.EncodePicture(webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), i), false)
				if err != nil {
					t.Fatalf("EncodePicture(%d): %v", i, err)
				}
				if picture.FrameNum == 0 {
					t.Fatalf("EncodePicture(%d) emitted no frames", i)
				}
				collectWebRTCPictureTUs(t, picture, &tusBySpatial, &wantBySpatial, &tusInOrder)
			}

			decodeCollectedWebRTCPictureTUs(t, stream.Config(), tusInOrder, tusBySpatial, wantBySpatial)
		})
	}
}

func TestWebRTCStreamControlCombinationMatrixDecode(t *testing.T) {
	for _, mode := range acceptedWebRTCStreamPixelModes() {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := webRTCDecodeMatrixConfig(mode)
			cfg.MaxFramerate = encoder.Rational{Num: 24, Den: 1}
			cfg.KeyFrameInterval = 4
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			stream.SetGoldenInterval(0)

			var tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte
			var wantBySpatial [encoder.WebRTCMaxSpatialLayers]int
			var tusInOrder [][]byte
			var parsedBySpatial [encoder.WebRTCMaxSpatialLayers]webRTCParsedTUState
			var parsedShared webRTCParsedTUState
			encode := func(frameIndex int, forceKey bool) encoder.WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodePicture(webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), frameIndex), forceKey)
				if err != nil {
					t.Fatalf("EncodePicture(%d, force=%v): %v", frameIndex, forceKey, err)
				}
				verifyWebRTCPicturePayloadHeaders(t, picture, stream.Config(), &parsedBySpatial, &parsedShared)
				collectWebRTCPictureTUs(t, picture, &tusBySpatial, &wantBySpatial, &tusInOrder)
				return picture
			}

			key := encode(0, false)
			if !key.Keyframe {
				t.Fatalf("initial picture was not a keyframe: %+v", key)
			}

			controlChange := stream.Config()
			controlChange.MaxFramerate = encoder.Rational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps += 10
			controlChange.MaxBitrateKbps += 250
			controlChange.TargetBitrateKbps += 120
			beforeFrameID := key.Frames[0].Info.FrameID + uint64(key.FrameNum)
			if err := stream.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig fps/bitrate control change: %v", err)
			}
			delta := encode(1, false)
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("control change picture=%+v beforeFrameID=%d", delta, beforeFrameID)
			}

			intervalChange := stream.Config()
			intervalChange.MaxFramerate = encoder.Rational{Num: 30000, Den: 1001}
			intervalChange.MinBitrateKbps += 5
			intervalChange.MaxBitrateKbps += 100
			intervalChange.TargetBitrateKbps += 40
			intervalChange.KeyFrameInterval = 2
			if err := stream.SetConfig(intervalChange); err != nil {
				t.Fatalf("SetConfig key-interval control change: %v", err)
			}
			intervalKey := encode(2, false)
			if !intervalKey.Keyframe {
				t.Fatalf("key interval did not force a key picture: %+v", intervalKey)
			}

			forceConfig := stream.Config()
			forceConfig.MaxFramerate = encoder.Rational{Num: 15, Den: 1}
			forceConfig.TargetBitrateKbps = forceConfig.MinBitrateKbps + (forceConfig.MaxBitrateKbps-forceConfig.MinBitrateKbps)/2
			if err := stream.SetConfig(forceConfig); err != nil {
				t.Fatalf("SetConfig pre-force control change: %v", err)
			}
			forced := encode(3, true)
			if !forced.Keyframe {
				t.Fatalf("force key did not produce key picture: %+v", forced)
			}

			decodeCollectedWebRTCPictureTUs(t, stream.Config(), tusInOrder, tusBySpatial, wantBySpatial)
		})
	}
}

func acceptedWebRTCStreamPixelModes() []encoder.ScalabilityMode {
	return []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T1,
		encoder.ScalabilityModeL1T2,
		encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeL2T1,
		encoder.ScalabilityModeL2T1h,
		encoder.ScalabilityModeL2T1_KEY,
		encoder.ScalabilityModeL2T2,
		encoder.ScalabilityModeL2T2h,
		encoder.ScalabilityModeL2T2_KEY,
		encoder.ScalabilityModeL2T2_KEY_SHIFT,
		encoder.ScalabilityModeL2T3,
		encoder.ScalabilityModeL2T3h,
		encoder.ScalabilityModeL2T3_KEY,
		encoder.ScalabilityModeL2T3_KEY_SHIFT,
		encoder.ScalabilityModeL3T1,
		encoder.ScalabilityModeL3T1h,
		encoder.ScalabilityModeL3T1_KEY,
		encoder.ScalabilityModeL3T2,
		encoder.ScalabilityModeL3T2h,
		encoder.ScalabilityModeL3T2_KEY,
		encoder.ScalabilityModeL3T2_KEY_SHIFT,
		encoder.ScalabilityModeL3T3,
		encoder.ScalabilityModeL3T3h,
		encoder.ScalabilityModeL3T3_KEY,
		encoder.ScalabilityModeL3T3_KEY_SHIFT,
		encoder.ScalabilityModeS2T1,
		encoder.ScalabilityModeS2T1h,
		encoder.ScalabilityModeS2T2,
		encoder.ScalabilityModeS2T2h,
		encoder.ScalabilityModeS2T3,
		encoder.ScalabilityModeS2T3h,
		encoder.ScalabilityModeS3T1,
		encoder.ScalabilityModeS3T1h,
		encoder.ScalabilityModeS3T2,
		encoder.ScalabilityModeS3T2h,
		encoder.ScalabilityModeS3T3,
		encoder.ScalabilityModeS3T3h,
	}
}

func webRTCDecodeMatrixConfig(mode encoder.ScalabilityMode) encoder.Config {
	spatial, _, _, _ := mode.Layers()
	resolution := encoder.Resolution{Width: 192, Height: 128}
	switch spatial {
	case 2:
		resolution = encoder.Resolution{Width: 640, Height: 360}
	case 3:
		resolution = encoder.Resolution{Width: 1008, Height: 576}
	}
	return encoder.Config{
		Resolution:        resolution,
		MaxFramerate:      encoder.Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       mode,
	}
}

func collectWebRTCPictureTUs(t *testing.T, picture encoder.WebRTCEncodedPicture, tusBySpatial *[encoder.WebRTCMaxSpatialLayers][][]byte, wantBySpatial *[encoder.WebRTCMaxSpatialLayers]int, tusInOrder *[][]byte) {
	t.Helper()
	if picture.FrameNum == 0 {
		t.Fatal("picture emitted no frames")
	}
	for frame := uint8(0); frame < picture.FrameNum; frame++ {
		spatialID := picture.Frames[frame].Info.SpatialID
		if spatialID >= encoder.WebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", frame, spatialID)
		}
		tu := append([]byte(nil), picture.Frames[frame].TU...)
		tusBySpatial[spatialID] = append(tusBySpatial[spatialID], tu)
		wantBySpatial[spatialID]++
		*tusInOrder = append(*tusInOrder, tu)
	}
}

func decodeCollectedWebRTCPictureTUs(t *testing.T, config encoder.Config, tusInOrder [][]byte, tusBySpatial [encoder.WebRTCMaxSpatialLayers][][]byte, wantBySpatial [encoder.WebRTCMaxSpatialLayers]int) {
	t.Helper()
	if webRTCSharedReferenceSlotMode(config) {
		frames := decodeLayerPoolLowOverheads(t, tusInOrder...)
		want := 0
		for _, count := range wantBySpatial {
			want += count
		}
		if len(frames) != want {
			t.Fatalf("shared SVC decoded %d frames, want %d", len(frames), want)
		}
		return
	}
	for spatialID, tus := range tusBySpatial {
		if len(tus) == 0 {
			continue
		}
		dec, err := goav1.NewDecoder(tus)
		if err != nil {
			t.Fatalf("spatial %d NewDecoder: %v", spatialID, err)
		}
		gotFrames := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				dec.Close()
				t.Fatalf("spatial %d decode: %v", spatialID, err)
			}
			if !ok {
				break
			}
			gotFrames += len(batch)
		}
		dec.Close()
		if gotFrames != wantBySpatial[spatialID] {
			t.Fatalf("spatial %d decoded %d frames, want %d", spatialID, gotFrames, wantBySpatial[spatialID])
		}
	}
}

func webRTCSharedReferenceSlotMode(config encoder.Config) bool {
	return config.SpatialLayerCount > 1 &&
		!config.Scalability.IsSimulcast()
}

type webRTCParsedTUState struct {
	seq     parser.SequenceHeader
	haveSeq bool
	refs    parser.ReferenceState
}

func verifyWebRTCPicturePayloadHeaders(t *testing.T, picture encoder.WebRTCEncodedPicture, config encoder.Config, states *[encoder.WebRTCMaxSpatialLayers]webRTCParsedTUState, shared *webRTCParsedTUState) {
	t.Helper()
	for frame := uint8(0); frame < picture.FrameNum; frame++ {
		encoded := picture.Frames[frame]
		spatialID := encoded.Info.SpatialID
		if spatialID >= encoder.WebRTCMaxSpatialLayers {
			t.Fatalf("frame %d spatial id=%d", frame, spatialID)
		}
		want := config.SpatialLayers[spatialID].Resolution
		state := &states[spatialID]
		if webRTCSharedReferenceSlotMode(config) {
			state = shared
		}
		state.verify(t, encoded, want)
	}
}

func (s *webRTCParsedTUState) verify(t *testing.T, frame encoder.WebRTCEncodedFrame, want encoder.Resolution) {
	t.Helper()
	it := obu.NewLowOverheadIterator(frame.TU)
	seenFrameHeader := false
	seenTileGroup := false
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse OBU: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			seq, err := parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			s.seq = seq
			s.haveSeq = true
		case obu.TypeFrameHeader, obu.TypeFrame:
			if unit.Header.TemporalID != frame.Info.TemporalID || unit.Header.SpatialID != frame.Info.SpatialID {
				t.Fatalf("frame header extension T%d/S%d want T%d/S%d", unit.Header.TemporalID, unit.Header.SpatialID, frame.Info.TemporalID, frame.Info.SpatialID)
			}
			size := s.parseAndRefreshFrameHeader(t, unit.Payload, unit.Header.TemporalID, unit.Header.SpatialID)
			wantCodedWidth := (uint32(want.Width) + 7) &^ 7
			wantCodedHeight := (uint32(want.Height) + 7) &^ 7
			if size.UpscaledWidth != wantCodedWidth || size.Height != wantCodedHeight ||
				size.RenderWidth != uint32(want.Width) || size.RenderHeight != uint32(want.Height) {
				t.Fatalf("payload coded=%dx%d render=%dx%d want coded=%dx%d render=%dx%d",
					size.UpscaledWidth, size.Height, size.RenderWidth, size.RenderHeight,
					wantCodedWidth, wantCodedHeight, want.Width, want.Height)
			}
			seenFrameHeader = true
		case obu.TypeTileGroup:
			if unit.Header.TemporalID != frame.Info.TemporalID || unit.Header.SpatialID != frame.Info.SpatialID {
				t.Fatalf("tile group extension T%d/S%d want T%d/S%d", unit.Header.TemporalID, unit.Header.SpatialID, frame.Info.TemporalID, frame.Info.SpatialID)
			}
			seenTileGroup = true
		}
	}
	if !seenFrameHeader || !seenTileGroup {
		t.Fatalf("missing frame header=%v tile group=%v", seenFrameHeader, seenTileGroup)
	}
}

func (s *webRTCParsedTUState) parseAndRefreshFrameHeader(t *testing.T, payload []byte, temporalID uint8, spatialID uint8) parser.FrameSize {
	t.Helper()
	if !s.haveSeq {
		t.Fatal("frame header before sequence header")
	}
	prefix, err := parser.ParseFrameHeaderPrefix(payload, s.seq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	var size parser.FrameSize
	if prefix.UsesIntraFrameSizePath() {
		size, err = parser.ParseIntraFrameSize(payload, s.seq, prefix, temporalID, spatialID)
	} else {
		size, err = parser.ParseFrameSize(payload, s.seq, prefix, &s.refs, temporalID, spatialID)
	}
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	s.refreshReferences(prefix, size)
	return size
}

func (s *webRTCParsedTUState) refreshReferences(prefix parser.FrameHeaderPrefix, size parser.FrameSize) {
	if size.RefreshFrameFlags == 0 {
		return
	}
	ref := parser.ReferenceFrame{
		Valid:        true,
		OrderHint:    prefix.OrderHint,
		GlobalMotion: parser.DefaultGlobalMotionParams(),
		Size:         size,
	}
	for i := uint8(0); i < parser.RefFrames; i++ {
		if size.RefreshFrameFlags&(1<<i) != 0 {
			s.refs.Frames[i] = ref
		}
	}
}

func webRTCDecodeMatrixFrame(width int, height int, n int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := 0; y < height; y++ {
		row := y * width
		for x := 0; x < width; x++ {
			f.Y[row+x] = uint8(48 + (x+n*7)%80 + (y+n*3)%40)
		}
	}
	for i := range f.U {
		f.U[i] = uint8(116 + n)
		f.V[i] = uint8(132 - n)
	}
	return f
}
