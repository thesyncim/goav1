package goav1_test

import (
	"errors"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// TestPublicVideoEncoderRoundTrip drives the public encoding surface end to
// end: CBR with two temporal layers, a mid-stream forced keyframe, and decode
// through the public Decoder with every frame bit-exact against the encoder
// reconstruction.
func TestPublicVideoEncoderRoundTrip(t *testing.T) {
	const w, h = 320, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(5))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(80))
	}
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		for y := 20 + n*3; y < 52+n*3 && y < h; y++ {
			for x := 16 + n*5; x < 48+n*5 && x < w; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 400_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	const frames = 12
	var tus [][]byte
	type snap struct {
		y, u, v []byte
	}
	var recons []snap
	keyAt := map[int]bool{}
	for i := range frames {
		out, err := enc.Encode(makeFrame(i), i == 6) // force a mid-stream keyframe
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if out.Keyframe {
			keyAt[i] = true
		}
		if i >= 1 && i < 6 && i%2 == 1 && out.TemporalID != 1 {
			t.Fatalf("frame %d temporal id %d, want 1", i, out.TemporalID)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
		r := enc.Reconstruction()
		recons = append(recons, snap{
			y: append([]byte(nil), r.Y...),
			u: append([]byte(nil), r.U...),
			v: append([]byte(nil), r.V...),
		})
	}
	if !keyAt[0] || !keyAt[6] {
		t.Fatalf("keyframes at %v, want frames 0 and 6", keyAt)
	}
	if enc.QIndex() == 0 {
		t.Fatal("rate controller reported qindex 0")
	}

	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			for y := range h {
				for x := range w {
					if f.Y.Pix[y*f.Y.Stride+x] != recons[i].y[y*w+x] {
						t.Fatalf("frame %d Y mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			for y := range ch {
				for x := range cw {
					if f.U.Pix[y*f.U.Stride+x] != recons[i].u[y*cw+x] || f.V.Pix[y*f.V.Stride+x] != recons[i].v[y*cw+x] {
						t.Fatalf("frame %d chroma mismatch at (%d,%d)", i, x, y)
					}
				}
			}
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}
}

// TestPublicRTCEncoder checks the WebRTC surface: every frame carries a
// dependency descriptor and the temporal units decode.
func TestPublicRTCEncoder(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	for i := range 6 {
		f := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for j := range f.Y {
			f.Y[j] = uint8(40 + (j+i*7)%150)
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if len(out.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		if i == 0 && !out.Keyframe {
			t.Fatal("first frame not a keyframe")
		}
		tus = append(tus, append([]byte(nil), out.Data...))
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
	if n != 6 {
		t.Fatalf("decoded %d frames, want 6", n)
	}
}

func TestPublicRTCFrameAppendRTPPackets(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 250_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	src := goav1.I420Frame{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	for i := range src.Y {
		src.Y[i] = uint8(40 + i%170)
	}
	for i := range src.U {
		src.U[i] = 120
		src.V[i] = 130
	}
	frame, err := enc.Encode(src, false)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	limits := goav1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first: %v", err)
	}
	obuScratch := make([]goav1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full: %v", err)
	}
	if size.Packetizer.Packets <= 1 {
		t.Fatalf("packetizer packets=%d want fragmented frame", size.Packetizer.Packets)
	}
	packetScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]goav1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]goav1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, descriptors, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets: %v", err)
	}
	if packetCount != size.Packetizer.Packets {
		t.Fatalf("packet count=%d want %d", packetCount, size.Packetizer.Packets)
	}
	payloadSlices := make([][]byte, packetCount)
	for i := range packetCount {
		span := spans[i]
		payloadSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
		desc := descriptors[span.DescriptorOffset : span.DescriptorOffset+span.DescriptorLength]
		mandatory, _, err := goav1.ParseRTPDependencyDescriptorMandatory(desc)
		if err != nil {
			t.Fatalf("packet %d descriptor: %v", i, err)
		}
		if mandatory.FirstPacketInFrame != (i == 0) || mandatory.LastPacketInFrame != (i == packetCount-1) {
			t.Fatalf("packet %d descriptor flags=%+v packets=%d", i, mandatory, packetCount)
		}
		if span.Marker != (i == packetCount-1) {
			t.Fatalf("packet %d marker=%v", i, span.Marker)
		}
	}
	if spans[0].DescriptorLength <= goav1.RTPDependencyDescriptorMandatorySize {
		t.Fatal("first descriptor did not attach dependency structure")
	}
	assembledLen, obuCount, err := goav1.AssembleRTPFrameSize(payloadSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	assembledOBUs := make([]goav1.RTPFrameOBU, obuCount)
	wrote, gotOBUs, err := goav1.AssembleRTPFrame(assembled, payloadSlices, assembledOBUs)
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	var expected []byte
	it := goav1.NewLowOverheadIterator(frame.Data)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("source OBU iteration: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case goav1.OBUTemporalDelimiter, goav1.OBUTileList, goav1.OBUPadding:
			continue
		default:
			expected = append(expected, unit.Raw...)
		}
	}
	if wrote != len(expected) || gotOBUs != obuCount || string(assembled[:wrote]) != string(expected) {
		t.Fatalf("assembled len=%d obus=%d match=%v want len=%d obus=%d", wrote, gotOBUs, string(assembled[:wrote]) == string(expected), len(expected), obuCount)
	}
}

func TestPublicRTCEncoderEncodePictureKeyShiftSpatial(t *testing.T) {
	const w, h = 640, 360
	cw, ch := w/2, h/2
	enc, err := goav1.NewRTCEncoderWithConfig(goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: w, Height: h},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    200,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 600,
		Scalability:       goav1.EncoderScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatal(err)
	}
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(55 + (i/7+n*9)%160)
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 134
		}
		return f
	}

	key, err := enc.EncodePicture(makeFrame(0), false)
	if err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	if key.FrameNum != 2 || !key.Keyframe {
		t.Fatalf("key FrameNum=%d Keyframe=%v, want 2 true", key.FrameNum, key.Keyframe)
	}
	cloneRTCPictureData(&key)
	for i := 0; i < key.FrameNum; i++ {
		frame := key.Frames[i]
		if frame.SpatialID != uint8(i) || frame.TemporalID != 0 || frame.FrameID != uint64(i) || !frame.Keyframe {
			t.Fatalf("key frame %d metadata S%d T%d id=%d key=%v", i, frame.SpatialID, frame.TemporalID, frame.FrameID, frame.Keyframe)
		}
		if len(frame.DependencyDescriptor) == 0 || !rtpPayloadHasLayerOBU(t, frame.Data, 0, uint8(i)) {
			t.Fatalf("key frame %d missing descriptor or layer OBUs", i)
		}
	}

	delta0, err := enc.EncodePicture(makeFrame(1), false)
	if err != nil {
		t.Fatalf("delta0 EncodePicture: %v", err)
	}
	cloneRTCPictureData(&delta0)
	delta1, err := enc.EncodePicture(makeFrame(2), false)
	if err != nil {
		t.Fatalf("delta1 EncodePicture: %v", err)
	}
	cloneRTCPictureData(&delta1)
	want0 := [2]uint8{0, 1}
	want1 := [2]uint8{1, 0}
	for i := 0; i < 2; i++ {
		if delta0.Frames[i].TemporalID != want0[i] || delta0.Frames[i].SpatialID != uint8(i) {
			t.Fatalf("delta0 frame %d S%d T%d, want S%d T%d", i, delta0.Frames[i].SpatialID, delta0.Frames[i].TemporalID, i, want0[i])
		}
		if delta1.Frames[i].TemporalID != want1[i] || delta1.Frames[i].SpatialID != uint8(i) {
			t.Fatalf("delta1 frame %d S%d T%d, want S%d T%d", i, delta1.Frames[i].SpatialID, delta1.Frames[i].TemporalID, i, want1[i])
		}
		if !rtpPayloadHasLayerOBU(t, delta0.Frames[i].Data, want0[i], uint8(i)) ||
			!rtpPayloadHasLayerOBU(t, delta1.Frames[i].Data, want1[i], uint8(i)) {
			t.Fatalf("delta frame %d missing expected OBU layer IDs", i)
		}
	}

	for spatial := 0; spatial < 2; spatial++ {
		tus := [][]byte{
			append([]byte(nil), key.Frames[spatial].Data...),
			append([]byte(nil), delta0.Frames[spatial].Data...),
			append([]byte(nil), delta1.Frames[spatial].Data...),
		}
		dec, err := goav1.NewDecoder(tus)
		if err != nil {
			t.Fatalf("spatial %d decoder: %v", spatial, err)
		}
		n := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				dec.Close()
				t.Fatalf("spatial %d decode: %v", spatial, err)
			}
			if !ok {
				break
			}
			n += len(batch)
		}
		dec.Close()
		if n != len(tus) {
			t.Fatalf("spatial %d decoded %d frames, want %d", spatial, n, len(tus))
		}
	}
}

func TestPublicRTCEncoderRejectsDeltaInterLayerPixelMode(t *testing.T) {
	_, err := goav1.NewRTCEncoderWithConfig(goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       goav1.EncoderScalabilityModeL2T2,
	})
	if !errors.Is(err, goav1.ErrEncoderUnsupported) {
		t.Fatalf("NewRTCEncoderWithConfig L2T2 err=%v want %v", err, goav1.ErrEncoderUnsupported)
	}
}

func TestPublicRTCEncoderSetConfigReconfigure(t *testing.T) {
	const w, h = 640, 360
	cw, ch := w/2, h/2
	makeFrame := func(n int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.Y {
			f.Y[i] = uint8(45 + (i+n*11)%160)
		}
		for i := range f.U {
			f.U[i] = 119
			f.V[i] = 132
		}
		return f
	}
	cfg := goav1.EncoderConfig{
		Resolution:        goav1.EncoderResolution{Width: w, Height: h},
		MaxFramerate:      goav1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       goav1.EncoderScalabilityModeL1T2,
	}
	enc, err := goav1.NewRTCEncoderWithConfig(cfg)
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}
	key, err := enc.Encode(makeFrame(0), false)
	if err != nil {
		t.Fatalf("key Encode: %v", err)
	}
	delta, err := enc.Encode(makeFrame(1), false)
	if err != nil {
		t.Fatalf("delta Encode: %v", err)
	}
	if !key.Keyframe || delta.Keyframe || key.FrameID != 0 || delta.FrameID != 1 {
		t.Fatalf("warm key=%+v delta=%+v", key, delta)
	}

	controlChange := cfg
	controlChange.MaxFramerate = goav1.EncoderRational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1200
	controlChange.TargetBitrateKbps = 900
	if err := enc.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	delta, err = enc.Encode(makeFrame(2), false)
	if err != nil {
		t.Fatalf("delta after control change: %v", err)
	}
	if delta.Keyframe || delta.FrameID != 2 || enc.Config().TargetBitrateKbps != 900 {
		t.Fatalf("control change delta=%+v config=%+v", delta, enc.Config())
	}

	structureChange := controlChange
	structureChange.Scalability = goav1.EncoderScalabilityModeS2T2
	if err := enc.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	picture, err := enc.EncodePicture(makeFrame(3), false)
	if err != nil {
		t.Fatalf("key after structure change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].FrameID != 3 || picture.Frames[1].FrameID != 4 {
		t.Fatalf("structure change picture=%+v", picture)
	}

	keepConfig := enc.Config()
	bad := structureChange
	bad.Scalability = goav1.EncoderScalabilityModeL2T2
	if err := enc.SetConfig(bad); !errors.Is(err, goav1.ErrEncoderUnsupported) {
		t.Fatalf("SetConfig unsupported err=%v want %v", err, goav1.ErrEncoderUnsupported)
	}
	if enc.Config() != keepConfig {
		t.Fatalf("unsupported SetConfig mutated config=%+v want=%+v", enc.Config(), keepConfig)
	}
	picture, err = enc.EncodePicture(makeFrame(4), false)
	if err != nil {
		t.Fatalf("delta after unsupported config: %v", err)
	}
	if picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].FrameID != 5 || picture.Frames[1].FrameID != 6 {
		t.Fatalf("post-unsupported picture=%+v", picture)
	}
}

func cloneRTCPictureData(p *goav1.RTCPicture) {
	for i := 0; i < p.FrameNum; i++ {
		p.Frames[i].Data = append([]byte(nil), p.Frames[i].Data...)
		p.Frames[i].DependencyDescriptor = append([]byte(nil), p.Frames[i].DependencyDescriptor...)
	}
}

func rtpPayloadHasLayerOBU(t *testing.T, data []byte, temporalID uint8, spatialID uint8) bool {
	t.Helper()
	it := obu.NewLowOverheadIterator(data)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			return false
		}
		switch unit.Header.Type {
		case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup, obu.TypeRedundantFrameHeader:
			if unit.Header.TemporalID != temporalID || unit.Header.SpatialID != spatialID {
				t.Fatalf("OBU %v layer T%d S%d, want T%d S%d", unit.Header.Type, unit.Header.TemporalID, unit.Header.SpatialID, temporalID, spatialID)
			}
			return true
		}
	}
}

func TestPublicRTCEncoderL1T3(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 300_000, Framerate: 30,
		TemporalLayers: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	var tus [][]byte
	for i, wantTID := range wantTIDs {
		f := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for j := range f.Y {
			f.Y[j] = uint8(50 + (j/7+i*11)%140)
		}
		for j := range f.U {
			f.U[j] = 120
			f.V[j] = 130
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if out.TemporalID != wantTID {
			t.Fatalf("frame %d temporal id %d want %d", i, out.TemporalID, wantTID)
		}
		if len(out.DependencyDescriptor) == 0 {
			t.Fatalf("frame %d missing dependency descriptor", i)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
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
	if n != len(wantTIDs) {
		t.Fatalf("decoded %d frames, want %d", n, len(wantTIDs))
	}
}

func TestPublicRTCEncoderGoldenInterval(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	base := make([]byte, w*h)
	for i := range base {
		base[i] = uint8(50 + rng.Intn(170))
	}
	mk := func(boxX int) goav1.I420Frame {
		f := goav1.I420Frame{
			Y: append([]byte(nil), base...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		if boxX >= 0 {
			for y := 32; y < 96; y++ {
				for x := boxX; x < boxX+64 && x < w; x++ {
					f.Y[y*w+x] = 235
				}
			}
		}
		return f
	}
	frames := []goav1.I420Frame{
		mk(-1),
		mk(64),
		mk(-1),
	}
	encode := func(goldenInterval int) [][]byte {
		enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
			Width: w, Height: h,
			TargetBitrate: 2_000_000, Framerate: 30,
			MinQIndex:      40,
			MaxQIndex:      160,
			TemporalLayers: 1,
			GoldenInterval: goldenInterval,
		})
		if err != nil {
			t.Fatal(err)
		}
		var tus [][]byte
		for i, f := range frames {
			out, err := enc.Encode(f, false)
			if err != nil {
				t.Fatalf("encode %d: %v", i, err)
			}
			tus = append(tus, append([]byte(nil), out.Data...))
		}
		return tus
	}
	lastOnly := encode(-1)
	withGolden := encode(16)
	t.Logf("rtc reveal frame: last-only %dB, with golden %dB", len(lastOnly[2]), len(withGolden[2]))
	if len(withGolden[2])*5 >= len(lastOnly[2])*4 {
		t.Fatalf("golden interval did not materially reduce reveal frame: last-only %dB, golden %dB", len(lastOnly[2]), len(withGolden[2]))
	}
}

func TestPublicEncoderZeroValueGuards(t *testing.T) {
	var video goav1.VideoEncoder
	if _, err := video.Encode(goav1.I420Frame{}, false); err == nil {
		t.Fatal("zero VideoEncoder Encode returned nil error")
	}
	video.SetDecisionStatsEnabled(true)
	video.ResetDecisionStats()
	if got := video.DecisionStats(); got != (goav1.EncoderDecisionStats{}) {
		t.Fatalf("zero VideoEncoder stats=%+v", got)
	}
	if got := video.QIndex(); got != 0 {
		t.Fatalf("zero VideoEncoder QIndex=%d", got)
	}
	if got := video.Reconstruction(); got.Width != 0 || got.Height != 0 {
		t.Fatalf("zero VideoEncoder reconstruction=%+v", got)
	}

	var rtc goav1.RTCEncoder
	if _, err := rtc.Encode(goav1.I420Frame{}, false); err == nil {
		t.Fatal("zero RTCEncoder Encode returned nil error")
	}
}

func TestPublicEncoderRejectsInvalidI420Frames(t *testing.T) {
	const w, h = 64, 64
	invalid := goav1.I420Frame{
		Y:            make([]byte, w*h-1),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	video, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h, QIndex: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := video.Encode(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted a short Y plane")
	}
	invalid.Y = make([]byte, w*h)
	invalid.ChromaStride = w/2 - 1
	if _, err := video.Encode(invalid, false); err == nil {
		t.Fatal("VideoEncoder accepted an invalid chroma stride")
	}

	rtc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 100_000, Framerate: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	invalid.ChromaStride = w / 2
	invalid.U = invalid.U[:len(invalid.U)-1]
	if _, err := rtc.Encode(invalid, false); err == nil {
		t.Fatal("RTCEncoder accepted a short U plane")
	}
}
