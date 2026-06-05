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
