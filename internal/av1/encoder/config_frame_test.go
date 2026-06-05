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
		delta.Frames[0].ReferenceBuffers[0] != 0 {
		t.Fatalf("base delta=%+v", delta.Frames[0])
	}
	if delta.Frames[1].TemporalID != 1 || delta.Frames[1].ReferenceCount != 2 ||
		delta.Frames[1].ReferenceBuffers[0] != 1 || delta.Frames[1].ReferenceBuffers[1] != 0 {
		t.Fatalf("upper delta=%+v", delta.Frames[1])
	}
	if delta.Control.Frames[0].GenericFrameInfo.DependencyNum != 1 ||
		delta.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 2 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 200 {
		t.Fatalf("delta generic info=%+v %+v", delta.Control.Frames[0].GenericFrameInfo, delta.Control.Frames[1].GenericFrameInfo)
	}
	if delta.Control.FrameIDState.FrameIDs[0] != 200 || delta.Control.FrameIDState.FrameIDs[1] != 201 {
		t.Fatalf("delta frame id state=%+v", delta.Control.FrameIDState)
	}
	if delta.Headers[0].Prefix.FrameType != FrameHeaderTypeInter ||
		!delta.Headers[0].Prefix.ErrorResilientMode ||
		delta.Headers[0].Prefix.OrderHint != 0 ||
		delta.Headers[0].Size.UpscaledWidth != 320 ||
		delta.Headers[0].Size.Height != 180 ||
		delta.Headers[0].Size.RefreshFrameFlags != 0x01 ||
		delta.Headers[0].Size.RefFrameIdx[0] != 0 {
		t.Fatalf("base delta header=%+v", delta.Headers[0])
	}
	if delta.Headers[1].Size.UpscaledWidth != 640 ||
		delta.Headers[1].Size.Height != 360 ||
		delta.Headers[1].Size.RefreshFrameFlags != 0x02 ||
		delta.Headers[1].Size.RefFrameIdx[0] != 1 ||
		delta.Headers[1].Size.RefFrameIdx[1] != 0 {
		t.Fatalf("upper delta header=%+v", delta.Headers[1])
	}
	for i := uint8(0); i < delta.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta.Headers[i])
	}
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
