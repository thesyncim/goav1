package goav1_test

import (
	"bytes"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicEncoderControlSurface(t *testing.T) {
	mode, ok := av1.ParseEncoderScalabilityMode("L2T2_KEY")
	if !ok || mode != av1.EncoderScalabilityModeL2T2_KEY {
		t.Fatalf("ParseEncoderScalabilityMode = %v,%v", mode, ok)
	}

	cfg, err := av1.SetWebRTCEncoderSVCConfig(av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		Scalability:       mode,
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
	}, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCEncoderSVCConfig: %v", err)
	}
	if cfg.SpatialLayerCount != 2 || cfg.TemporalLayerCount != 2 {
		t.Fatalf("layers = %d,%d; want 2,2", cfg.SpatialLayerCount, cfg.TemporalLayerCount)
	}
	if cfg.SpatialLayers[0].Resolution != (av1.EncoderResolution{Width: 320, Height: 180}) {
		t.Fatalf("base layer resolution = %+v", cfg.SpatialLayers[0].Resolution)
	}

	state := av1.EncoderReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = av1.EncoderResolution{Width: 320, Height: 180}
	frames := [...]av1.EncoderFrameEncodeSettings{
		{
			Type:             av1.EncoderFrameTypeDelta,
			Resolution:       av1.EncoderResolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [av1.EncoderWebRTCMaxReferences]uint8{0},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := av1.ValidateEncoderTemporalUnitFrames(frames[:], state, av1.EncoderRateControlCBR); err != nil {
		t.Fatalf("ValidateEncoderTemporalUnitFrames: %v", err)
	}
}

func TestPublicEncoderLowOverheadOBU(t *testing.T) {
	unit := av1.EncoderOBU{
		Type:       av1.OBUFrame,
		TemporalID: 1,
		SpatialID:  1,
		Payload:    []byte{0xaa, 0xbb},
	}
	size, err := av1.EncoderLowOverheadOBUSize(unit)
	if err != nil {
		t.Fatalf("EncoderLowOverheadOBUSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderLowOverheadOBU(buf[:0], unit)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("encoded len=%d want %d", len(out), size)
	}
	parsed, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != av1.OBUFrame || parsed.Header.TemporalID != 1 || parsed.Header.SpatialID != 1 {
		t.Fatalf("parsed header=%+v consumed=%d", parsed.Header, consumed)
	}
	if !bytes.Equal(parsed.Payload, unit.Payload) {
		t.Fatalf("payload=% x want % x", parsed.Payload, unit.Payload)
	}
}

func TestPublicEncoderLowOverheadTemporalUnit(t *testing.T) {
	units := [...]av1.EncoderOBU{
		{Type: av1.OBUFrame, Payload: []byte{0xaa}},
	}
	size, err := av1.EncoderLowOverheadTemporalUnitSize(units[:])
	if err != nil {
		t.Fatalf("EncoderLowOverheadTemporalUnitSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderLowOverheadTemporalUnit(buf[:0], units[:])
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadTemporalUnit: %v", err)
	}
	if len(out) != size {
		t.Fatalf("temporal unit len=%d want %d", len(out), size)
	}
	it := av1.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil {
		t.Fatalf("TemporalUnitIterator.Next: %v", err)
	}
	if !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit parsed ok=%v raw=% x", ok, tu.Raw)
	}
}

func TestPublicEncoderSequenceHeaderOBU(t *testing.T) {
	seq := av1.EncoderSequenceHeader{
		Profile:               av1.EncoderProfile0,
		OperatingPointsCount:  1,
		MaxFrameWidth:         16,
		MaxFrameHeight:        9,
		EnableFilterIntra:     true,
		EnableIntraEdgeFilter: true,
		EnableOrderHint:       true,
		OrderHintBits:         7,
		EnableSuperRes:        true,
		EnableCDEF:            true,
		EnableRestoration:     true,
		ColorConfig: av1.EncoderSequenceColorConfig{
			BitDepth:                8,
			ColorDescriptionPresent: true,
			ColorPrimaries:          av1.EncoderSequenceColorPrimariesBT709,
			TransferCharacteristics: av1.EncoderSequenceTransferCharacteristicsSRGB,
			MatrixCoefficients:      av1.EncoderSequenceMatrixCoefficientsIdentity,
			ColorRange:              true,
		},
	}
	seq.OperatingPoints[0].SeqLevelIdx = 5

	size, err := av1.EncoderLowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		t.Fatalf("EncoderLowOverheadSequenceHeaderOBUSize: %v", err)
	}
	var buf [80]byte
	out, err := av1.AppendEncoderLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadSequenceHeaderOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("sequence obu len=%d want %d", len(out), size)
	}
	unit, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("parsed header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := av1.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}
