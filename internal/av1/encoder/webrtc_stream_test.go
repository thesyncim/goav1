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

func TestWebRTCStreamAcceptedScalabilityModesCoverExportedModes(t *testing.T) {
	modes := acceptedWebRTCStreamPixelModes()
	if len(modes) != encoder.WebRTCSVCScalabilityModeCount() {
		t.Fatalf("accepted mode count=%d want exported SVC count=%d", len(modes), encoder.WebRTCSVCScalabilityModeCount())
	}
	seen := make(map[encoder.ScalabilityMode]bool, len(modes))
	for _, mode := range modes {
		if !mode.Valid() {
			t.Fatalf("accepted invalid mode %d", mode)
		}
		if seen[mode] {
			t.Fatalf("accepted duplicate mode %s", mode)
		}
		seen[mode] = true
		parsed, ok := encoder.ParseScalabilityMode(mode.String())
		if !ok || parsed != mode {
			t.Fatalf("mode %s parse round trip=(%s,%v)", mode, parsed, ok)
		}
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

func TestWebRTCStream1080pHotPathAllocs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		mode       encoder.ScalabilityMode
		speed      int8
		maxThreads int32
	}{
		{name: encoder.ScalabilityModeL1T3.String(), mode: encoder.ScalabilityModeL1T3},
		{name: encoder.ScalabilityModeS3T3.String(), mode: encoder.ScalabilityModeS3T3},
		{name: "L1T3-single-thread-min-effort", mode: encoder.ScalabilityModeL1T3, speed: encoder.WebRTCMinEffortLevel, maxThreads: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := webRTC1080pAllocConfig(tc.mode)
			cfg.Speed = tc.speed
			cfg.MaxThreads = tc.maxThreads
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", tc.name, err)
			}
			t.Cleanup(func() { _ = stream.Close() })

			frames := [4]encoder.SourceFrame420{
				makeEncoder1080pFrame(0),
				makeEncoder1080pFrame(1),
				makeEncoder1080pFrame(2),
				makeEncoder1080pFrame(3),
			}
			if err := stream.Prewarm(); err != nil {
				t.Fatalf("Prewarm(%s): %v", tc.name, err)
			}
			for i := 0; i < 12; i++ {
				forceKey := i == 0 || i == 6
				picture, err := stream.EncodePicture(frames[i&3], forceKey)
				if err != nil {
					t.Fatalf("warm EncodePicture(%s, %d): %v", tc.name, i, err)
				}
				if picture.FrameNum == 0 {
					t.Fatalf("warm EncodePicture(%s, %d) emitted no frames", tc.name, i)
				}
			}

			frameIndex := 0
			pAllocs := testing.AllocsPerRun(3, func() {
				picture, err := stream.EncodePicture(frames[frameIndex&3], false)
				frameIndex++
				if err != nil {
					t.Fatal(err)
				}
				if picture.Keyframe {
					t.Fatal("steady WebRTC picture was coded as keyframe")
				}
				if picture.FrameNum == 0 {
					t.Fatal("empty steady WebRTC picture")
				}
			})
			if pAllocs != 0 {
				t.Fatalf("1080p WebRTC %s steady picture allocations=%f want 0", tc.name, pAllocs)
			}

			if _, err := stream.EncodePicture(frames[0], true); err != nil {
				t.Fatalf("forced-key warm EncodePicture(%s): %v", tc.name, err)
			}
			keyAllocs := testing.AllocsPerRun(3, func() {
				picture, err := stream.EncodePicture(frames[1], true)
				if err != nil {
					t.Fatal(err)
				}
				if !picture.Keyframe {
					t.Fatal("forced WebRTC picture was not coded as keyframe")
				}
				if picture.FrameNum == 0 {
					t.Fatal("empty forced-key WebRTC picture")
				}
			})
			if keyAllocs != 0 {
				t.Fatalf("1080p WebRTC %s forced-key picture allocations=%f want 0", tc.name, keyAllocs)
			}
		})
	}
}

func acceptedWebRTCStreamPixelModes() []encoder.ScalabilityMode {
	return encoder.AppendWebRTCSVCScalabilityModes(make([]encoder.ScalabilityMode, 0, encoder.WebRTCSVCScalabilityModeCount()))
}

func webRTC1080pAllocConfig(mode encoder.ScalabilityMode) encoder.Config {
	return encoder.Config{
		Resolution:        encoder.Resolution{Width: 1920, Height: 1080},
		MaxFramerate:      encoder.Rational{Num: 60, Den: 1},
		MinBitrateKbps:    800,
		MaxBitrateKbps:    8_000,
		TargetBitrateKbps: 4_000,
		Scalability:       mode,
		RateControl:       encoder.RateControlCBR,
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
