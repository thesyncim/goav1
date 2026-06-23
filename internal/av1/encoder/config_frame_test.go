package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestIntraHeaderTemporalUnitForConfig(t *testing.T) {
	cfg := Config{
		Resolution:         Resolution{Width: 640, Height: 360},
		Scalability:        ScalabilityModeL2T2,
		MaxFramerate:       Rational{Num: 30, Den: 1},
		MinBitrateKbps:     100,
		MaxBitrateKbps:     800,
		TargetBitrateKbps:  500,
		DependencyMetadata: true,
		LowOverheadOBU:     true,
		RTPPacketization:   true,
	}
	unit, err := IntraHeaderTemporalUnitForConfig(cfg, 17)
	if err != nil {
		t.Fatalf("IntraHeaderTemporalUnitForConfig: %v", err)
	}
	if unit.Sequence.OperatingPointsCount != 4 || unit.Sequence.MaxFrameWidth != 640 || unit.Sequence.MaxFrameHeight != 360 {
		t.Fatalf("sequence=%+v", unit.Sequence)
	}
	if unit.Prefix.FrameType != FrameHeaderTypeKey || !unit.Prefix.ShowFrame ||
		!unit.Prefix.ErrorResilientMode || unit.Prefix.OrderHint != 17 ||
		unit.Prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
		t.Fatalf("prefix=%+v", unit.Prefix)
	}
	if unit.Size.UpscaledWidth != 640 || unit.Size.Height != 360 ||
		unit.Size.SuperResDenominator != 8 || unit.Size.RefreshFrameFlags != 0xff {
		t.Fatalf("size=%+v", unit.Size)
	}
}

func TestIntraHeaderTemporalUnitForScreenContent(t *testing.T) {
	cfg := Config{
		Resolution: Resolution{Width: 640, Height: 360},
		Content:    ContentScreen,
	}
	unit, err := IntraHeaderTemporalUnitForConfig(cfg, 3)
	if err != nil {
		t.Fatalf("IntraHeaderTemporalUnitForConfig: %v", err)
	}
	if !unit.Prefix.AllowScreenContentTools || !unit.Prefix.ForceIntegerMV {
		t.Fatalf("screen prefix=%+v", unit.Prefix)
	}
	var buf [64]byte
	payload, err := AppendFrameHeaderIntraPayload(buf[:0], unit.Sequence, unit.Prefix, unit.Size)
	if err != nil {
		t.Fatalf("AppendFrameHeaderIntraPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, unit.Sequence)
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	if !parsedPrefix.AllowScreenContentTools || !parsedPrefix.ForceIntegerMV {
		t.Fatalf("parsed screen prefix=%+v", parsedPrefix)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnitForConfig(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	size, wantUnit, err := LowOverheadIntraHeaderTemporalUnitForConfigSize(cfg, 9)
	if err != nil {
		t.Fatalf("LowOverheadIntraHeaderTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := AppendLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 9)
	if err != nil {
		t.Fatalf("AppendLowOverheadIntraHeaderTemporalUnitForConfig: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit {
		t.Fatalf("len=%d want %d gotUnit=%+v wantUnit=%+v", len(out), size, gotUnit, wantUnit)
	}

	it := obu.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter ok=%v err=%v unit=%+v payload=% x", ok, err, td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok || seqUnit.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v unit=%+v", ok, err, seqUnit.Header)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsedSeq.OperatingPointsCount != 4 || parsedSeq.MaxFrameWidth != 640 || parsedSeq.MaxFrameHeight != 360 {
		t.Fatalf("parsed sequence=%+v", parsedSeq)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok || frameUnit.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame ok=%v err=%v unit=%+v", ok, err, frameUnit.Header)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedPrefix.FrameType != parser.FrameTypeKey || !parsedPrefix.ShowFrame || parsedPrefix.OrderHint != 9 ||
		parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed prefix=%+v size=%+v", parsedPrefix, parsedSize)
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra ok=%v err=%v unit=%+v", ok, err, extra.Header)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnitForConfigRejectsInvalid(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}}
	if _, _, err := AppendLowOverheadIntraHeaderTemporalUnitForConfig(nil, cfg, 128); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("order hint overflow err=%v want %v", err, ErrInvalidFrame)
	}

	var buf [8]byte
	dst := buf[:1]
	dst[0] = 0xee
	if out, _, err := AppendLowOverheadIntraHeaderTemporalUnitForConfig(dst, cfg, 0); !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	} else if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendLowOverheadIntraHeaderTemporalUnitForConfigAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}}
	var buf [192]byte
	if _, _, err := AppendLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 0); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = AppendLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 0)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadIntraHeaderTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestWebRTCKeyFrameTemporalUnitForConfigLayered(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 7, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if unit.FrameNum != 2 || unit.Control.FrameNum != 2 || !unit.Control.HasDependencyStructure {
		t.Fatalf("unit/control frame counts = %+v control=%+v", unit, unit.Control)
	}
	if unit.Frames[0].Type != FrameTypeKey || unit.Frames[0].Resolution != (Resolution{Width: 320, Height: 180}) ||
		unit.Frames[0].UpdateBuffer != 0 || unit.Frames[0].ReferenceCount != 0 {
		t.Fatalf("base frame=%+v", unit.Frames[0])
	}
	if unit.Frames[1].Type != FrameTypeDelta || unit.Frames[1].Resolution != (Resolution{Width: 640, Height: 360}) ||
		unit.Frames[1].ReferenceCount != 1 || unit.Frames[1].ReferenceBuffers[0] != 0 ||
		unit.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("upper frame=%+v", unit.Frames[1])
	}
	if !unit.Control.Frames[0].AttachDependencyStructure || unit.Control.Frames[1].AttachDependencyStructure ||
		unit.Control.Frames[1].GenericFrameInfo.DependencyNum != 1 ||
		unit.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 100 {
		t.Fatalf("control=%+v", unit.Control)
	}
	if !unit.Control.ReferenceState.Valid[0] || !unit.Control.ReferenceState.Valid[1] ||
		unit.Control.FrameIDState.FrameIDs[0] != 100 || unit.Control.FrameIDState.FrameIDs[1] != 101 {
		t.Fatalf("states refs=%+v ids=%+v", unit.Control.ReferenceState, unit.Control.FrameIDState)
	}
}

func TestWebRTCKeyFrameTemporalUnitForConfigSimulcast(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeS2T1,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 40)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig simulcast: %v", err)
	}
	if unit.FrameNum != 2 || unit.Frames[0].Type != FrameTypeKey || unit.Frames[1].Type != FrameTypeStart ||
		unit.Frames[1].ReferenceCount != 0 {
		t.Fatalf("simulcast frames=%+v", unit.Frames)
	}
	if unit.Control.Frames[1].GenericFrameInfo.DependencyNum != 0 ||
		unit.Control.FrameIDState.FrameIDs[0] != 40 || unit.Control.FrameIDState.FrameIDs[1] != 41 {
		t.Fatalf("simulcast control=%+v", unit.Control)
	}
}

func TestAppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	size, wantUnit, err := LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(cfg, 5, 200)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 5, 200)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit {
		t.Fatalf("len=%d want=%d got=%+v want=%+v", len(out), size, gotUnit, wantUnit)
	}
	it := obu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || len(tu.Raw) != len(out) {
		t.Fatalf("temporal unit ok=%v err=%v len=%d want=%d", ok, err, len(tu.Raw), len(out))
	}
	low := obu.NewLowOverheadIterator(out)
	if td, ok, err := low.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := low.Next(); err != nil || !ok || seq.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("seq ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	assertWebRTCScalabilityMetadataOBU(t, &low, ScalabilityModeL2T2)
	if frame, ok, err := low.Next(); err != nil || !ok || frame.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
}

func TestAppendLowOverheadWebRTCKeyFrameTemporalUnitForConfigAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	var buf [192]byte
	if _, _, err := AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 0, 1); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 0, 1)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestAppendLowOverheadWebRTCKeyFrameTemporalUnitForState(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	state := WebRTCEncoderState{NextOrderHint: 17, NextFrameID: 600}
	size, wantUnit, wantState, err := LowOverheadWebRTCKeyFrameTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCKeyFrameTemporalUnitForStateSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, gotState, err := AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit || gotState != wantState ||
		gotUnit.Header.Prefix.OrderHint != 17 || gotUnit.Control.Frames[0].GenericFrameInfo.FrameID != 600 ||
		gotState.NextFrameID != 602 || gotState.DeltaPictureIndex != 1 || !gotState.DependencyStructureState.Valid {
		t.Fatalf("len=%d want=%d unit=%+v wantUnit=%+v state=%+v wantState=%+v", len(out), size, gotUnit, wantUnit, gotState, wantState)
	}
	it := obu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || len(tu.Raw) != len(out) {
		t.Fatalf("temporal unit ok=%v err=%v len=%d want=%d", ok, err, len(tu.Raw), len(out))
	}
	low := obu.NewLowOverheadIterator(out)
	if td, ok, err := low.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := low.Next(); err != nil || !ok || seq.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("seq ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	assertWebRTCScalabilityMetadataOBU(t, &low, ScalabilityModeL2T2)
	if frame, ok, err := low.Next(); err != nil || !ok || frame.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame ok=%v err=%v header=%+v", ok, err, frame.Header)
	}
	var tiny [1]byte
	if _, _, _, err := AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(tiny[:0], cfg, state); !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
}

func TestAppendLowOverheadWebRTCKeyFrameTemporalUnitForStateAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	state := WebRTCEncoderState{NextFrameID: 1}
	var buf [192]byte
	if _, _, _, err := AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _ = AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadWebRTCKeyFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigLayered(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	if delta.FrameNum != 2 || delta.Control.FrameNum != 2 || delta.Control.HasDependencyStructure {
		t.Fatalf("delta counts/control=%+v", delta)
	}
	if delta.Frames[0].TemporalID != 1 || delta.Frames[0].ReferenceCount != 1 ||
		delta.Frames[0].ReferenceBuffers[0] != 0 ||
		!delta.Frames[0].UpdateBufferSet || delta.Frames[0].UpdateBuffer != 2 {
		t.Fatalf("base delta=%+v", delta.Frames[0])
	}
	if delta.Frames[1].TemporalID != 1 || delta.Frames[1].ReferenceCount != 2 ||
		delta.Frames[1].ReferenceBuffers[0] != 1 || delta.Frames[1].ReferenceBuffers[1] != 2 ||
		delta.Frames[1].UpdateBufferSet {
		t.Fatalf("upper delta=%+v", delta.Frames[1])
	}
	if delta.Control.Frames[0].GenericFrameInfo.DependencyNum != 1 ||
		delta.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 2 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 200 {
		t.Fatalf("delta generic info=%+v %+v", delta.Control.Frames[0].GenericFrameInfo, delta.Control.Frames[1].GenericFrameInfo)
	}
	if delta.Control.FrameIDState.FrameIDs[0] != 100 || delta.Control.FrameIDState.FrameIDs[1] != 101 ||
		delta.Control.FrameIDState.FrameIDs[2] != 200 {
		t.Fatalf("delta frame id state=%+v", delta.Control.FrameIDState)
	}
	if delta.Headers[0].Prefix.FrameType != FrameHeaderTypeInter ||
		!delta.Headers[0].Prefix.ErrorResilientMode ||
		delta.Headers[0].Prefix.OrderHint != 0 ||
		delta.Headers[0].Size.UpscaledWidth != 320 ||
		delta.Headers[0].Size.Height != 180 ||
		delta.Headers[0].Size.RefreshFrameFlags != 0x04 ||
		delta.Headers[0].Size.RefFrameIdx[0] != 0 {
		t.Fatalf("base delta header=%+v", delta.Headers[0])
	}
	if delta.Headers[1].Size.UpscaledWidth != 640 ||
		delta.Headers[1].Size.Height != 360 ||
		delta.Headers[1].Size.RefreshFrameFlags != 0x00 ||
		delta.Headers[1].Size.RefFrameIdx[0] != 1 ||
		delta.Headers[1].Size.RefFrameIdx[1] != 2 {
		t.Fatalf("upper delta header=%+v", delta.Headers[1])
	}
	for i := uint8(0); i < delta.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta.Headers[i])
	}

	base, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, delta.Control.ReferenceState, delta.Control.FrameIDState, 0, 202)
	if err != nil {
		t.Fatalf("base delta: %v", err)
	}
	if base.Frames[0].TemporalID != 0 || base.Frames[0].ReferenceBuffers[0] != 0 ||
		!base.Frames[0].UpdateBufferSet || base.Frames[0].UpdateBuffer != 0 ||
		base.Frames[1].TemporalID != 0 || base.Frames[1].ReferenceBuffers[0] != 1 ||
		base.Frames[1].ReferenceBuffers[1] != 0 ||
		!base.Frames[1].UpdateBufferSet || base.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("base delta frames=%+v", base.Frames)
	}
	if base.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		base.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		base.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 202 {
		t.Fatalf("base delta generic info=%+v %+v", base.Control.Frames[0].GenericFrameInfo, base.Control.Frames[1].GenericFrameInfo)
	}
}

func TestWebRTCEncoderStateTemporalUnitsL2T2KeyShift(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2_KEY_SHIFT,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 100})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 2 || key.Frames[0].Type != FrameTypeKey ||
		key.Frames[1].ReferenceCount != 1 || key.Frames[1].ReferenceBuffers[0] != 0 ||
		state.NextFrameID != 102 || state.DeltaPictureIndex != 1 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}

	delta0, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta0: %v", err)
	}
	if delta0.FrameNum != 2 ||
		delta0.Frames[0].SpatialID != 0 || delta0.Frames[0].TemporalID != 0 ||
		delta0.Frames[0].ReferenceCount != 1 || delta0.Frames[0].ReferenceBuffers[0] != 0 ||
		!delta0.Frames[0].UpdateBufferSet || delta0.Frames[0].UpdateBuffer != 0 ||
		delta0.Frames[1].SpatialID != 1 || delta0.Frames[1].TemporalID != 1 ||
		delta0.Frames[1].ReferenceCount != 1 || delta0.Frames[1].ReferenceBuffers[0] != 1 ||
		delta0.Frames[1].UpdateBufferSet {
		t.Fatalf("delta0 frames=%+v", delta0.Frames)
	}
	if delta0.Headers[0].Size.RefreshFrameFlags != 0x01 || delta0.Headers[1].Size.RefreshFrameFlags != 0x00 {
		t.Fatalf("delta0 refresh flags=%02x,%02x", delta0.Headers[0].Size.RefreshFrameFlags, delta0.Headers[1].Size.RefreshFrameFlags)
	}
	if delta0.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		delta0.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		state.FrameIDState.FrameIDs[0] != 102 || state.FrameIDState.FrameIDs[1] != 101 {
		t.Fatalf("delta0 control=%+v state=%+v", delta0.Control, state)
	}
	for i := uint8(0); i < delta0.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta0.Headers[i])
	}

	delta1, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta1: %v", err)
	}
	if delta1.FrameNum != 2 ||
		delta1.Frames[0].SpatialID != 0 || delta1.Frames[0].TemporalID != 1 ||
		delta1.Frames[0].ReferenceCount != 1 || delta1.Frames[0].ReferenceBuffers[0] != 0 ||
		delta1.Frames[0].UpdateBufferSet ||
		delta1.Frames[1].SpatialID != 1 || delta1.Frames[1].TemporalID != 0 ||
		delta1.Frames[1].ReferenceCount != 1 || delta1.Frames[1].ReferenceBuffers[0] != 1 ||
		!delta1.Frames[1].UpdateBufferSet || delta1.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("delta1 frames=%+v", delta1.Frames)
	}
	if delta1.Headers[0].Size.RefreshFrameFlags != 0x00 || delta1.Headers[1].Size.RefreshFrameFlags != 0x02 {
		t.Fatalf("delta1 refresh flags=%02x,%02x", delta1.Headers[0].Size.RefreshFrameFlags, delta1.Headers[1].Size.RefreshFrameFlags)
	}
	if delta1.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 102 ||
		delta1.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		state.FrameIDState.FrameIDs[0] != 102 || state.FrameIDState.FrameIDs[1] != 105 {
		t.Fatalf("delta1 control=%+v state=%+v", delta1.Control, state)
	}
	for i := uint8(0); i < delta1.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta1.Headers[i])
	}

	delta2, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta2: %v", err)
	}
	if delta2.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 102 ||
		delta2.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 105 {
		t.Fatalf("delta2 control=%+v", delta2.Control)
	}
}

func TestWebRTCEncoderStateTemporalUnitsKeyShiftModes(t *testing.T) {
	for _, tc := range [...]struct {
		name string
		mode ScalabilityMode
		want [][]uint8
	}{
		{name: "L2T3", mode: ScalabilityModeL2T3_KEY_SHIFT, want: [][]uint8{
			{2, 2},
			{0, 1},
			{2, 2},
			{1, 0},
		}},
		{name: "L3T2", mode: ScalabilityModeL3T2_KEY_SHIFT, want: [][]uint8{
			{0, 0, 1},
			{1, 1, 0},
		}},
		{name: "L3T3", mode: ScalabilityModeL3T3_KEY_SHIFT, want: [][]uint8{
			{0, 2, 2},
			{2, 0, 1},
			{1, 2, 2},
			{2, 1, 0},
		}},
	} {
		cfg := Config{
			Resolution:        Resolution{Width: 1280, Height: 720},
			Scalability:       tc.mode,
			MaxFramerate:      Rational{Num: 30, Den: 1},
			MinBitrateKbps:    100,
			MaxBitrateKbps:    1200,
			TargetBitrateKbps: 800,
		}
		_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 10})
		if err != nil {
			t.Fatalf("%s key: %v", tc.name, err)
		}
		for i, want := range tc.want {
			delta, next, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
			if err != nil {
				t.Fatalf("%s delta %d: %v", tc.name, i, err)
			}
			if int(delta.FrameNum) != len(want) {
				t.Fatalf("%s delta %d frame num=%d want %d", tc.name, i, delta.FrameNum, len(want))
			}
			for spatialID, wantTemporalID := range want {
				frame := delta.Frames[spatialID]
				primaryBuffer := uint8(spatialID)
				middleBuffer := delta.FrameNum + uint8(spatialID)
				selfReference := frame.ReferenceBuffers[0] == primaryBuffer ||
					(frame.TemporalID == 2 && frame.ReferenceBuffers[0] == middleBuffer)
				if frame.SpatialID != uint8(spatialID) || frame.TemporalID != wantTemporalID ||
					frame.ReferenceCount != 1 || !selfReference {
					t.Fatalf("%s delta %d spatial %d frame=%+v want temporal %d single self ref", tc.name, i, spatialID, frame, wantTemporalID)
				}
				if frame.UpdateBufferSet && frame.UpdateBuffer != primaryBuffer && frame.UpdateBuffer != middleBuffer {
					t.Fatalf("%s delta %d spatial %d update buffer=%d", tc.name, i, spatialID, frame.UpdateBuffer)
				}
			}
			state = next
		}
	}
}

func TestWebRTCControllerSettingsMatrix(t *testing.T) {
	profiles := [...]struct {
		name               string
		fps                Rational
		rateControl        RateControlMode
		quantizer          uint8
		minKbps            int32
		maxKbps            int32
		targetKbps         int32
		content            ContentHint
		dependencyMetadata bool
		lowOverheadOBU     bool
		rtpPacketization   bool
	}{
		{
			name:               "camera-cbr-30fps-rtp",
			fps:                Rational{Num: 30, Den: 1},
			rateControl:        RateControlCBR,
			minKbps:            120,
			maxKbps:            1800,
			targetKbps:         900,
			content:            ContentCamera,
			dependencyMetadata: true,
			lowOverheadOBU:     true,
			rtpPacketization:   true,
		},
		{
			name:               "screen-cqp-ntsc-headers",
			fps:                Rational{Num: 30000, Den: 1001},
			rateControl:        RateControlCQP,
			quantizer:          37,
			minKbps:            80,
			maxKbps:            1200,
			targetKbps:         640,
			content:            ContentScreen,
			dependencyMetadata: true,
			lowOverheadOBU:     true,
			rtpPacketization:   false,
		},
	}

	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		for _, profile := range profiles {
			name := mode.String() + "/" + profile.name
			t.Run(name, func(t *testing.T) {
				cfg := Config{
					Resolution:         Resolution{Width: 1280, Height: 720},
					MaxFramerate:       profile.fps,
					RateControl:        profile.rateControl,
					Quantizer:          profile.quantizer,
					MinBitrateKbps:     profile.minKbps,
					MaxBitrateKbps:     profile.maxKbps,
					TargetBitrateKbps:  profile.targetKbps,
					Content:            profile.content,
					Scalability:        mode,
					DependencyMetadata: profile.dependencyMetadata,
					LowOverheadOBU:     profile.lowOverheadOBU,
					RTPPacketization:   profile.rtpPacketization,
				}
				assertWebRTCControllerReferenceModel(t, cfg)
			})
		}
	}
}

func TestWebRTCControllerSettingsMatrixKeyInterval(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1280, Height: 720},
		Scalability:       ScalabilityModeL2T3_KEY_SHIFT,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
		KeyFrameInterval:  2,
	}
	unit, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1}, false)
	if err != nil || !unit.Key || unit.Delta {
		t.Fatalf("initial key unit=%+v state=%+v err=%v", unit, state, err)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil || !unit.Delta || unit.Key || state.DeltaPictureIndex != 2 {
		t.Fatalf("first delta unit=%+v state=%+v err=%v", unit, state, err)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil || !unit.Key || unit.Delta || state.DeltaPictureIndex != 1 {
		t.Fatalf("interval key unit=%+v state=%+v err=%v", unit, state, err)
	}
}

func TestWebRTCEncoderStateReconfigureControls(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1280, Height: 720},
		Scalability:       ScalabilityModeL1T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 20})
	if err != nil {
		t.Fatalf("key: %v", err)
	}
	delta, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta: %v", err)
	}
	if delta.FrameNum != 1 || state.DeltaPictureIndex != 2 {
		t.Fatalf("warm state delta=%+v state=%+v", delta, state)
	}

	controlChange := cfg
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1800
	controlChange.TargetBitrateKbps = 1100
	unit, next, err := WebRTCNextTemporalUnitForState(controlChange, state, false)
	if err != nil {
		t.Fatalf("control reconfigure: %v", err)
	}
	if !unit.Delta || unit.Key || unit.DeltaUnit.FrameNum != 1 || next.DeltaPictureIndex != 3 {
		t.Fatalf("control reconfigure unit=%+v next=%+v", unit, next)
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T2
	unit, next, err = WebRTCNextTemporalUnitForState(structureChange, next, false)
	if err != nil {
		t.Fatalf("structure reconfigure: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		!unit.KeyUnit.Control.HasDependencyStructure || next.DeltaPictureIndex != 1 {
		t.Fatalf("structure reconfigure unit=%+v next=%+v", unit, next)
	}
}

func assertWebRTCControllerReferenceModel(t *testing.T, cfg Config) {
	t.Helper()
	normalized, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig: %v", err)
	}
	spatialLayers, temporalLayers, _, ok := normalized.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid mode %v", normalized.Scalability)
	}
	if normalized.SpatialLayerCount != spatialLayers || normalized.TemporalLayerCount != temporalLayers {
		t.Fatalf("normalized layers=%d,%d want %d,%d", normalized.SpatialLayerCount, normalized.TemporalLayerCount, spatialLayers, temporalLayers)
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if structure.NumDecodeTargets != spatialLayers*temporalLayers ||
		structure.NumChains != spatialLayers ||
		structure.ResolutionNum != spatialLayers {
		t.Fatalf("structure shape=%+v", structure)
	}

	var headerBuf [2048]byte
	if _, picture, _, err := AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(headerBuf[:0], normalized, WebRTCEncoderState{NextFrameID: 1}, false); err != nil {
		t.Fatalf("AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState key: %v", err)
	} else if !picture.Key || picture.Delta {
		t.Fatalf("first picture=%+v want key", picture)
	}

	key, state, err := WebRTCKeyFrameTemporalUnitForState(normalized, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != spatialLayers || !key.Control.HasDependencyStructure ||
		state.NextFrameID != 1+uint64(spatialLayers) || state.DeltaPictureIndex != 1 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}
	type frameLayer struct {
		spatial  uint8
		temporal uint8
	}
	history := make(map[uint64]frameLayer, 64)
	for i := uint8(0); i < key.Control.FrameNum; i++ {
		info := key.Control.Frames[i].GenericFrameInfo
		history[info.FrameID] = frameLayer{spatial: info.SpatialID, temporal: info.TemporalID}
	}

	steps := webRTCTestControllerMatrixSteps(temporalLayers)
	for deltaIndex := uint64(1); deltaIndex <= steps; deltaIndex++ {
		currentState := state
		delta, next, err := WebRTCDeltaFrameTemporalUnitForState(normalized, state)
		if err != nil {
			t.Fatalf("delta %d: %v", deltaIndex, err)
		}
		if delta.FrameNum != spatialLayers || delta.Control.HasDependencyStructure {
			t.Fatalf("delta %d unit/control=%+v", deltaIndex, delta)
		}
		if next.NextFrameID != state.NextFrameID+uint64(spatialLayers) ||
			next.DeltaPictureIndex != state.DeltaPictureIndex+1 {
			t.Fatalf("delta %d next=%+v state=%+v", deltaIndex, next, state)
		}
		for i := uint8(0); i < delta.FrameNum; i++ {
			frame := delta.Frames[i]
			control := delta.Control.Frames[i]
			info := control.GenericFrameInfo
			wantTemporal := webRTCTestExpectedTemporalID(normalized.Scalability, i, deltaIndex)
			if frame.SpatialID != i || info.SpatialID != i ||
				frame.TemporalID != wantTemporal || info.TemporalID != wantTemporal ||
				delta.Headers[i].TemporalID != wantTemporal || delta.Headers[i].SpatialID != i {
				t.Fatalf("delta %d frame %d frame=%+v info=%+v header=%+v want temporal=%d", deltaIndex, i, frame, info, delta.Headers[i], wantTemporal)
			}
			if frame.RateControl != normalized.RateControl ||
				(normalized.RateControl == RateControlCQP && frame.Quantizer != normalized.Quantizer) {
				t.Fatalf("delta %d frame %d rate control frame=%+v config=%+v", deltaIndex, i, frame, normalized)
			}
			if normalized.Content == ContentScreen &&
				(!delta.Headers[i].Prefix.AllowScreenContentTools || !delta.Headers[i].Prefix.ForceIntegerMV) {
				t.Fatalf("delta %d frame %d screen header=%+v", deltaIndex, i, delta.Headers[i])
			}
			wantRefresh := uint8(0)
			if frame.UpdateBufferSet {
				wantRefresh = 1 << frame.UpdateBuffer
			}
			if delta.Headers[i].Size.RefreshFrameFlags != wantRefresh {
				t.Fatalf("delta %d frame %d refresh=%02x want %02x frame=%+v", deltaIndex, i, delta.Headers[i].Size.RefreshFrameFlags, wantRefresh, frame)
			}
			if (normalized.Scalability.IsSimulcast() || normalized.Scalability.UsesKeyFrameInterLayerDependency()) && frame.ReferenceCount != 1 {
				t.Fatalf("delta %d frame %d key/simulcast refs=%+v", deltaIndex, i, frame)
			}
			if webRTCUsesDeltaInterLayerReference(normalized) && i > 0 && frame.ReferenceCount < 2 {
				t.Fatalf("delta %d frame %d full-svc refs=%+v", deltaIndex, i, frame)
			}
			for j := uint8(0); j < info.DependencyNum; j++ {
				dep, ok := history[info.Dependencies[j]]
				if !ok {
					t.Fatalf("delta %d frame %d missing dependency %d in history", deltaIndex, i, info.Dependencies[j])
				}
				if dep.temporal > info.TemporalID {
					t.Fatalf("delta %d frame %d temporal dependency %d has T%d > T%d", deltaIndex, i, info.Dependencies[j], dep.temporal, info.TemporalID)
				}
				if normalized.Scalability.IsSimulcast() && dep.spatial != info.SpatialID {
					t.Fatalf("delta %d frame %d simulcast dependency spatial=%d want %d", deltaIndex, i, dep.spatial, info.SpatialID)
				}
			}
			var descriptorBuf [256]byte
			if _, err := AppendWebRTCDependencyDescriptor(descriptorBuf[:0], currentState.DependencyStructureState.Structure, info, true, true, false); err != nil {
				t.Fatalf("delta %d frame %d dependency descriptor: %v", deltaIndex, i, err)
			}
			history[info.FrameID] = frameLayer{spatial: info.SpatialID, temporal: info.TemporalID}
		}
		state = next
	}
}

func webRTCTestControllerMatrixSteps(temporalLayers uint8) uint64 {
	switch temporalLayers {
	case 1:
		return 3
	case 2:
		return 4
	default:
		return 8
	}
}

func webRTCTestExpectedTemporalID(mode ScalabilityMode, spatialID uint8, deltaPictureIndex uint64) uint8 {
	if mode.UsesKeyFrameInterLayerDependencyShift() {
		switch mode {
		case ScalabilityModeL2T2_KEY_SHIFT:
			table := [2][2]uint8{{0, 1}, {1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case ScalabilityModeL2T3_KEY_SHIFT:
			table := [4][2]uint8{{2, 2}, {0, 1}, {2, 2}, {1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		case ScalabilityModeL3T2_KEY_SHIFT:
			table := [2][3]uint8{{0, 0, 1}, {1, 1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case ScalabilityModeL3T3_KEY_SHIFT:
			table := [4][3]uint8{{0, 2, 2}, {2, 0, 1}, {1, 2, 2}, {2, 1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		}
	}
	_, temporalLayers, _, ok := mode.Layers()
	if !ok || deltaPictureIndex == 0 {
		return 0
	}
	trailingZeroCount := uint8(0)
	for value := deltaPictureIndex; value&1 == 0 && trailingZeroCount < temporalLayers-1; value >>= 1 {
		trailingZeroCount++
	}
	return temporalLayers - 1 - trailingZeroCount
}

func TestWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 1)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2, 37)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if delta.Headers[0].Prefix.OrderHint != 37 {
		t.Fatalf("delta header order hint=%d want 37", delta.Headers[0].Prefix.OrderHint)
	}
	assertParsedDeltaHeader(t, delta.Headers[0])
}

func TestAppendLowOverheadWebRTCDeltaHeaderTemporalUnit(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 10})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 2 {
		t.Fatalf("key frame num=%d", key.FrameNum)
	}
	size, wantUnit, wantState, err := LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, gotState, err := AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit || gotState != wantState {
		t.Fatalf("len=%d want=%d unit=%+v want=%+v state=%+v wantState=%+v", len(out), size, gotUnit, wantUnit, gotState, wantState)
	}

	it := obu.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter ok=%v err=%v header=%+v payload=% x", ok, err, td.Header, td.Payload)
	}
	for i := uint8(0); i < gotUnit.FrameNum; i++ {
		frameUnit, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("frame header %d ok=%v err=%v", i, ok, err)
		}
		header := gotUnit.Headers[i]
		if frameUnit.Header.Type != obu.TypeFrameHeader ||
			frameUnit.Header.TemporalID != header.TemporalID ||
			frameUnit.Header.SpatialID != header.SpatialID {
			t.Fatalf("frame header %d obu=%+v header=%+v", i, frameUnit.Header, header)
		}
		parsedSeq := parseEncoderSequenceHeader(t, header.Sequence)
		parsedPrefix, err := parser.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
		if err != nil {
			t.Fatalf("ParseFrameHeaderPrefix %d: %v", i, err)
		}
		refs := parserReferenceState(header.Sequence, header.Size.RefFrameIdx[:])
		parsedSize, err := parser.ParseFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, &refs, header.TemporalID, header.SpatialID)
		if err != nil {
			t.Fatalf("ParseFrameSize %d: %v", i, err)
		}
		if parsedPrefix.OrderHint != header.Prefix.OrderHint ||
			parsedSize.RefreshFrameFlags != header.Size.RefreshFrameFlags ||
			parsedSize.UpscaledWidth != header.Size.UpscaledWidth ||
			parsedSize.Height != header.Size.Height {
			t.Fatalf("parsed %d prefix=%+v size=%+v header=%+v", i, parsedPrefix, parsedSize, header)
		}
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra ok=%v err=%v header=%+v", ok, err, extra.Header)
	}

	var tiny [1]byte
	if out, _, _, err := AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(tiny[:0], cfg, state); !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short buffer out=% x err=%v want %v", out, err, bitstream.ErrShortBuffer)
	}
}

func TestAppendLowOverheadWebRTCDeltaHeaderTemporalUnitAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	var buf [192]byte
	if _, _, _, err := AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(buf[:0], cfg, state); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _ = AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(buf[:0], cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState allocated: %f", allocs)
	}
}

func TestAppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       RateControlCQP,
		Quantizer:         37,
	}
	state := WebRTCEncoderState{NextOrderHint: 11, NextFrameID: 50}
	size, wantUnit, wantState, wantHeader, err := LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForStateSize: %v", err)
	}
	var buf [512]byte
	out, gotUnit, gotState, gotHeader, err := AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit || gotState != wantState || gotHeader != wantHeader {
		t.Fatalf("len=%d want=%d unit=%+v want=%+v state=%+v want=%+v header=%+v want=%+v", len(out), size, gotUnit, wantUnit, gotState, wantState, gotHeader, wantHeader)
	}
	if gotHeader.Quantization.BaseQIdx != 37 || gotHeader.Tile.Cols == 0 || gotHeader.Tile.Rows == 0 ||
		gotHeader.TransformRef.TransformMode != TransformModeLargest {
		t.Fatalf("complete key header=%+v", gotHeader)
	}

	it := obu.NewLowOverheadIterator(out)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok || seqUnit.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("seq ok=%v err=%v header=%+v", ok, err, seqUnit.Header)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	assertWebRTCScalabilityMetadataOBU(t, &it, ScalabilityModeL2T2)
	frameUnit, ok, err := it.Next()
	if err != nil || !ok || frameUnit.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame ok=%v err=%v header=%+v", ok, err, frameUnit.Header)
	}
	wantPayload, parsed := appendAndParseIntraFrameHeader(t, gotUnit.Header.Sequence, gotHeader)
	if string(frameUnit.Payload) != string(wantPayload) {
		t.Fatalf("frame payload=% x want % x", frameUnit.Payload, wantPayload)
	}
	if parsedSeq.MaxFrameWidth != parsed.Size.UpscaledWidth || parsed.Quant.BaseQIdx != 37 ||
		parsed.Tile.Cols != gotHeader.Tile.Cols || parsed.Tile.Rows != gotHeader.Tile.Rows {
		t.Fatalf("parsed seq=%+v parsed=%+v header=%+v", parsedSeq, parsed, gotHeader)
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra ok=%v err=%v header=%+v", ok, err, extra.Header)
	}

	var tiny [1]byte
	if out, _, _, _, err := AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(tiny[:0], cfg, state); !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short buffer out=% x err=%v want %v", out, err, bitstream.ErrShortBuffer)
	}
}

func TestAppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       RateControlCQP,
		Quantizer:         29,
	}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 80})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	size, wantUnit, wantState, err := LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForStateSize: %v", err)
	}
	var buf [512]byte
	out, gotUnit, gotState, err := AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState: %v", err)
	}
	if len(out) != size || gotUnit != wantUnit || gotState != wantState {
		t.Fatalf("len=%d want=%d unit=%+v want=%+v state=%+v want=%+v", len(out), size, gotUnit, wantUnit, gotState, wantState)
	}

	it := obu.NewLowOverheadIterator(out)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	for i := uint8(0); i < gotUnit.FrameNum; i++ {
		frameUnit, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("frame %d ok=%v err=%v", i, ok, err)
		}
		header := gotUnit.Headers[i]
		if frameUnit.Header.Type != obu.TypeFrameHeader ||
			frameUnit.Header.TemporalID != header.TemporalID ||
			frameUnit.Header.SpatialID != header.SpatialID {
			t.Fatalf("frame %d obu=%+v header=%+v", i, frameUnit.Header, header)
		}
		refs := completeHeaderReferenceStateForBuffers(header, gotUnit.Control.ReferenceState)
		fullHeader, err := completeInterFrameHeaderParams(header, gotUnit.Frames[i], &refs)
		if err != nil {
			t.Fatalf("completeInterFrameHeaderParams %d: %v", i, err)
		}
		if gotUnit.Frames[i].ReferenceCount > 1 {
			layerRef := gotUnit.Frames[i].ReferenceBuffers[0]
			layerSize := refs.Frames[layerRef].Size
			if !refs.Frames[layerRef].Valid || layerSize.UpscaledWidth != 640 || layerSize.Height != 360 {
				t.Fatalf("frame %d layer ref %d size=%+v valid=%v", i, layerRef, layerSize, refs.Frames[layerRef].Valid)
			}
			lowerRef := gotUnit.Frames[i].ReferenceBuffers[1]
			refSize := refs.Frames[lowerRef].Size
			if !refs.Frames[lowerRef].Valid || refSize.UpscaledWidth != 320 || refSize.Height != 180 {
				t.Fatalf("frame %d lower ref %d size=%+v valid=%v header=%+v", i, lowerRef, refSize, refs.Frames[lowerRef].Valid, header)
			}
		}
		wantPayload, parsed := appendAndParseInterFrameHeader(t, header.Sequence, fullHeader, &refs)
		if string(frameUnit.Payload) != string(wantPayload) {
			t.Fatalf("frame %d payload=% x want % x", i, frameUnit.Payload, wantPayload)
		}
		if parsed.Quant.BaseQIdx != 29 || parsed.Size.UpscaledWidth != header.Size.UpscaledWidth ||
			parsed.Tile.Cols != fullHeader.Tile.Cols || parsed.Tile.Rows != fullHeader.Tile.Rows {
			t.Fatalf("frame %d parsed=%+v full=%+v", i, parsed, fullHeader)
		}
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra ok=%v err=%v header=%+v", ok, err, extra.Header)
	}

	var tiny [1]byte
	if out, _, _, err := AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(tiny[:0], cfg, state); !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short buffer out=% x err=%v want %v", out, err, bitstream.ErrShortBuffer)
	}
}

func TestAppendLowOverheadWebRTCCompleteHeaderTemporalUnitsAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	var keyBuf [512]byte
	if _, _, _, _, err := AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(keyBuf[:0], cfg, WebRTCEncoderState{NextFrameID: 1}); err != nil {
		t.Fatalf("key preflight: %v", err)
	}
	var deltaBuf [512]byte
	if _, _, _, err := AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(deltaBuf[:0], cfg, state); err != nil {
		t.Fatalf("delta preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _, _ = AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(keyBuf[:0], cfg, WebRTCEncoderState{NextFrameID: 1})
		_, _, _, _ = AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(deltaBuf[:0], cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadWebRTCCompleteHeaderTemporalUnits allocated: %f", allocs)
	}
}

func TestAppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  3,
	}
	state := WebRTCEncoderState{NextFrameID: 20}

	keySize, wantKeyUnit, wantKeyState, err := LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(cfg, state, false)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize key: %v", err)
	}
	var keyBuf [192]byte
	keyOut, gotKeyUnit, gotKeyState, err := AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(keyBuf[:0], cfg, state, false)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState key: %v", err)
	}
	if len(keyOut) != keySize || gotKeyUnit != wantKeyUnit || gotKeyState != wantKeyState ||
		!gotKeyUnit.Key || gotKeyUnit.Delta || gotKeyUnit.KeyUnit.Control.Frames[0].GenericFrameInfo.FrameID != 20 {
		t.Fatalf("key len=%d want=%d unit=%+v want=%+v state=%+v wantState=%+v", len(keyOut), keySize, gotKeyUnit, wantKeyUnit, gotKeyState, wantKeyState)
	}
	keyIt := obu.NewLowOverheadIterator(keyOut)
	if td, ok, err := keyIt.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("key TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := keyIt.Next(); err != nil || !ok || seq.Header.Type != obu.TypeSequenceHeader {
		t.Fatalf("key seq ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	assertWebRTCScalabilityMetadataOBU(t, &keyIt, ScalabilityModeL2T2)
	if fh, ok, err := keyIt.Next(); err != nil || !ok || fh.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("key frame header ok=%v err=%v header=%+v", ok, err, fh.Header)
	}

	deltaSize, wantDeltaUnit, wantDeltaState, err := LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(cfg, gotKeyState, false)
	if err != nil {
		t.Fatalf("LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize delta: %v", err)
	}
	var deltaBuf [192]byte
	deltaOut, gotDeltaUnit, gotDeltaState, err := AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(deltaBuf[:0], cfg, gotKeyState, false)
	if err != nil {
		t.Fatalf("AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState delta: %v", err)
	}
	if len(deltaOut) != deltaSize || gotDeltaUnit != wantDeltaUnit || gotDeltaState != wantDeltaState ||
		!gotDeltaUnit.Delta || gotDeltaUnit.Key || gotDeltaUnit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 22 {
		t.Fatalf("delta len=%d want=%d unit=%+v want=%+v state=%+v wantState=%+v", len(deltaOut), deltaSize, gotDeltaUnit, wantDeltaUnit, gotDeltaState, wantDeltaState)
	}
	deltaIt := obu.NewLowOverheadIterator(deltaOut)
	if td, ok, err := deltaIt.Next(); err != nil || !ok || td.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("delta TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	for i := uint8(0); i < gotDeltaUnit.DeltaUnit.FrameNum; i++ {
		fh, ok, err := deltaIt.Next()
		if err != nil || !ok || fh.Header.Type != obu.TypeFrameHeader ||
			fh.Header.TemporalID != gotDeltaUnit.DeltaUnit.Headers[i].TemporalID ||
			fh.Header.SpatialID != gotDeltaUnit.DeltaUnit.Headers[i].SpatialID {
			t.Fatalf("delta frame header %d ok=%v err=%v header=%+v unit=%+v", i, ok, err, fh.Header, gotDeltaUnit.DeltaUnit.Headers[i])
		}
	}
	if extra, ok, err := deltaIt.Next(); err != nil || ok {
		t.Fatalf("delta extra ok=%v err=%v header=%+v", ok, err, extra.Header)
	}

	var tiny [1]byte
	if out, _, _, err := AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(tiny[:0], cfg, gotKeyState, false); !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short buffer out=% x err=%v want %v", out, err, bitstream.ErrShortBuffer)
	}
	forced, forcedUnit, _, err := LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(cfg, gotDeltaState, true)
	if err != nil {
		t.Fatalf("forced key size: %v", err)
	}
	if forced == 0 || !forcedUnit.Key || forcedUnit.Delta {
		t.Fatalf("forced key size=%d unit=%+v", forced, forcedUnit)
	}
}

func TestAppendLowOverheadWebRTCPictureHeaderTemporalUnitForStateAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	var buf [192]byte
	if _, _, _, err := AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(buf[:0], cfg, state, false); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _ = AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(buf[:0], cfg, state, false)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState allocated: %f", allocs)
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigSimulcast(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeS2T1,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 20)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig simulcast: %v", err)
	}
	if delta.FrameNum != 2 || delta.Frames[0].ReferenceCount != 1 || delta.Frames[1].ReferenceCount != 1 ||
		delta.Frames[0].ReferenceBuffers[0] != 0 || delta.Frames[1].ReferenceBuffers[0] != 1 {
		t.Fatalf("simulcast delta frames=%+v", delta.Frames)
	}
	if delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 1 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 11 {
		t.Fatalf("simulcast delta control=%+v", delta.Control.Frames[1].GenericFrameInfo)
	}
}

func TestWebRTCTemporalUnitsPropagateCQPQuantizer(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       RateControlCQP,
		Quantizer:         37,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.Frames[0].RateControl != RateControlCQP || key.Frames[0].Quantizer != 37 ||
		key.Frames[1].RateControl != RateControlCQP || key.Frames[1].Quantizer != 37 {
		t.Fatalf("key frames=%+v", key.Frames)
	}
	delta, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if delta.Frames[0].RateControl != RateControlCQP || delta.Frames[0].Quantizer != 37 ||
		delta.Frames[1].RateControl != RateControlCQP || delta.Frames[1].Quantizer != 37 {
		t.Fatalf("delta frames=%+v", delta.Frames)
	}
}

func TestWebRTCTemporalUnitsPropagateScreenContentTools(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Content:           ContentScreen,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if !key.Header.Prefix.AllowScreenContentTools || !key.Header.Prefix.ForceIntegerMV {
		t.Fatalf("key prefix=%+v", key.Header.Prefix)
	}
	delta, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if !delta.Headers[0].Prefix.AllowScreenContentTools || !delta.Headers[0].Prefix.ForceIntegerMV ||
		!delta.Headers[1].Prefix.AllowScreenContentTools || !delta.Headers[1].Prefix.ForceIntegerMV {
		t.Fatalf("delta headers=%+v", delta.Headers)
	}
	for i := uint8(0); i < delta.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta.Headers[i])
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigRejectsInvalidState(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	if _, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, ReferenceBufferState{}, FrameIDBufferState{}, 0, 1); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing reference state err=%v want %v", err, ErrInvalidFrame)
	}
	ref := ReferenceBufferState{}
	ids := FrameIDBufferState{}
	ref.Valid[0] = true
	ref.Resolutions[0] = Resolution{Width: 320, Height: 180}
	ids.Valid[0] = true
	ids.FrameIDs[0] = 1
	if _, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, ref, ids, 2, 2); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("bad temporal id err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 1)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if _, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestWebRTCEncoderStateTemporalUnits(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 960, Height: 540},
		Scalability:       ScalabilityModeL3T3,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 800,
	}
	state := WebRTCEncoderState{NextOrderHint: 126, NextFrameID: 1000}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 3 || key.Header.Prefix.OrderHint != 126 ||
		state.NextOrderHint != 127 || state.NextFrameID != 1003 ||
		state.DeltaPictureIndex != 1 || !state.DependencyStructureState.Valid {
		t.Fatalf("key=%+v state=%+v", key, state)
	}

	wantTemporal := [...]uint8{2, 1, 2, 0}
	wantFirstID := uint64(1003)
	for i, want := range wantTemporal {
		delta, next, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
		if err != nil {
			t.Fatalf("delta %d: %v", i, err)
		}
		if delta.FrameNum != 3 || delta.Frames[0].TemporalID != want ||
			delta.Control.Frames[0].GenericFrameInfo.FrameID != wantFirstID ||
			delta.Control.HasDependencyStructure ||
			delta.Headers[0].Prefix.OrderHint != state.NextOrderHint {
			t.Fatalf("delta %d unit=%+v", i, delta)
		}
		if next.NextFrameID != wantFirstID+3 || next.DeltaPictureIndex != state.DeltaPictureIndex+1 ||
			next.DependencyStructureState.Structure != state.DependencyStructureState.Structure {
			t.Fatalf("delta %d next=%+v state=%+v", i, next, state)
		}
		if i == 0 && next.NextOrderHint != 0 {
			t.Fatalf("delta %d order hint=%d want wrap to 0", i, next.NextOrderHint)
		}
		var descriptorBuf [32]byte
		if _, err := AppendWebRTCDependencyDescriptor(
			descriptorBuf[:0],
			next.DependencyStructureState.Structure,
			delta.Control.Frames[2].GenericFrameInfo,
			true,
			true,
			false,
		); err != nil {
			t.Fatalf("delta %d descriptor: %v", i, err)
		}
		state = next
		wantFirstID += 3
	}
}

func assertParsedDeltaHeader(t *testing.T, header InterHeaderFrame) {
	t.Helper()
	var buf [80]byte
	payload, err := AppendFrameHeaderInterPayload(buf[:0], header.Sequence, header.Prefix, header.Size)
	if err != nil {
		t.Fatalf("AppendFrameHeaderInterPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, header.Sequence)
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := parserReferenceState(header.Sequence, header.Size.RefFrameIdx[:])
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if parsedPrefix.FrameType != parser.FrameTypeInter ||
		parsedPrefix.OrderHint != header.Prefix.OrderHint ||
		parsedPrefix.AllowScreenContentTools != header.Prefix.AllowScreenContentTools ||
		parsedPrefix.ForceIntegerMV != header.Prefix.ForceIntegerMV ||
		parsedSize.RefreshFrameFlags != header.Size.RefreshFrameFlags ||
		parsedSize.UpscaledWidth != header.Size.UpscaledWidth ||
		parsedSize.Height != header.Size.Height {
		t.Fatalf("parsed prefix=%+v size=%+v header=%+v", parsedPrefix, parsedSize, header)
	}
}

func TestWebRTCNextTemporalUnitForStateKeyInterval(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  3,
	}
	state := WebRTCEncoderState{NextFrameID: 100}

	unit, state, err := WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("initial next temporal unit: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		state.NextFrameID != 102 || state.DeltaPictureIndex != 1 ||
		!state.DependencyStructureState.Valid {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}

	want := [...]struct {
		key      bool
		temporal uint8
		firstID  uint64
	}{
		{temporal: 1, firstID: 102},
		{temporal: 0, firstID: 104},
		{key: true, firstID: 106},
		{temporal: 1, firstID: 108},
	}
	for i, want := range want {
		unit, next, err := WebRTCNextTemporalUnitForState(cfg, state, false)
		if err != nil {
			t.Fatalf("next %d: %v", i, err)
		}
		if want.key {
			if !unit.Key || unit.Delta || unit.KeyUnit.Control.Frames[0].GenericFrameInfo.FrameID != want.firstID ||
				!unit.KeyUnit.Control.HasDependencyStructure || next.DeltaPictureIndex != 1 {
				t.Fatalf("next %d key unit=%+v next=%+v", i, unit, next)
			}
		} else {
			if !unit.Delta || unit.Key || unit.DeltaUnit.Frames[0].TemporalID != want.temporal ||
				unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != want.firstID ||
				unit.DeltaUnit.Control.HasDependencyStructure {
				t.Fatalf("next %d delta unit=%+v", i, unit)
			}
		}
		state = next
	}
}

func TestWebRTCNextTemporalUnitForStateForceKey(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	unit, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 9}, false)
	if err != nil {
		t.Fatalf("initial next: %v", err)
	}
	if !unit.Key || state.NextFrameID != 10 {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("delta next: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 10 {
		t.Fatalf("delta unit=%+v state=%+v", unit, state)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, true)
	if err != nil {
		t.Fatalf("forced key next: %v", err)
	}
	if !unit.Key || unit.KeyUnit.Control.Frames[0].GenericFrameInfo.FrameID != 11 ||
		!unit.KeyUnit.Control.HasDependencyStructure || state.DeltaPictureIndex != 1 {
		t.Fatalf("forced unit=%+v state=%+v", unit, state)
	}
}

func TestWebRTCTemporalIDForDeltaPicture(t *testing.T) {
	tests := [...]struct {
		layers uint8
		index  uint64
		want   uint8
	}{
		{layers: 1, index: 1, want: 0},
		{layers: 2, index: 1, want: 1},
		{layers: 2, index: 2, want: 0},
		{layers: 3, index: 1, want: 2},
		{layers: 3, index: 2, want: 1},
		{layers: 3, index: 3, want: 2},
		{layers: 3, index: 4, want: 0},
	}
	for _, tt := range tests {
		got, err := WebRTCTemporalIDForDeltaPicture(tt.layers, tt.index)
		if err != nil {
			t.Fatalf("layers=%d index=%d: %v", tt.layers, tt.index, err)
		}
		if got != tt.want {
			t.Fatalf("layers=%d index=%d got=%d want=%d", tt.layers, tt.index, got, tt.want)
		}
	}
	if _, err := WebRTCTemporalIDForDeltaPicture(3, 0); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("zero index err=%v want %v", err, ErrInvalidConfig)
	}
	if _, err := WebRTCTemporalIDForDeltaPicture(WebRTCMaxTemporalLayers+1, 1); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("too many layers err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestWebRTCEncoderStateRejectsDeltaBeforeKey(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	_, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, WebRTCEncoderState{DeltaPictureIndex: 1})
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("delta before key err=%v want %v", err, ErrInvalidFrame)
	}
}

func TestWebRTCEncoderStateTemporalUnitsAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 2 || state.NextFrameID != 3 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}
	if _, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state); err != nil {
		t.Fatalf("delta preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestWebRTCNextTemporalUnitForStateAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2, KeyFrameInterval: 10}
	_, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1}, false)
	if err != nil {
		t.Fatalf("initial next: %v", err)
	}
	if _, _, err := WebRTCNextTemporalUnitForState(cfg, state, false); err != nil {
		t.Fatalf("delta preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCNextTemporalUnitForState(cfg, state, false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCNextTemporalUnitForState allocated: %f", allocs)
	}
}

func assertWebRTCScalabilityMetadataOBU(t *testing.T, it *obu.LowOverheadIterator, mode ScalabilityMode) {
	t.Helper()
	unit, ok, err := it.Next()
	if err != nil || !ok || unit.Header.Type != obu.TypeMetadata {
		t.Fatalf("metadata ok=%v err=%v header=%+v", ok, err, unit.Header)
	}
	meta, err := obu.ParseMetadata(unit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadata: %v", err)
	}
	idc, ok := WebRTCScalabilityModeIDC(mode)
	if !ok || meta.Type != obu.MetadataTypeScalability || meta.Scalability.ModeIDC != idc || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v mode=%s idc=%d ok=%v", meta, mode, idc, ok)
	}
}
