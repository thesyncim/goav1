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
