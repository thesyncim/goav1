package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestAV1RTCPFeedbackConstants(t *testing.T) {
	if av1.RTCPPSFBLayerRefreshRequestFMT != 10 {
		t.Fatalf("RTCPPSFBLayerRefreshRequestFMT = %d, want 10", av1.RTCPPSFBLayerRefreshRequestFMT)
	}
	if av1.AV1RTCPLayerRefreshLayerIndexSize != 2 {
		t.Fatalf("AV1RTCPLayerRefreshLayerIndexSize = %d, want 2", av1.AV1RTCPLayerRefreshLayerIndexSize)
	}
	if av1.AV1RTCPLayerRefreshRequestEntrySize != 12 {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntrySize = %d, want 12", av1.AV1RTCPLayerRefreshRequestEntrySize)
	}
	if av1.AV1RTCPLayerRefreshMaxTemporalID != 7 ||
		av1.AV1RTCPLayerRefreshMaxSpatialID != 3 {
		t.Fatalf("unexpected AV1 LRR layer index limits")
	}
	if av1.AV1SDPRTCPFeedbackNACK != "nack" ||
		av1.AV1SDPRTCPFeedbackPLI != "nack pli" ||
		av1.AV1SDPRTCPFeedbackFIR != "ccm fir" ||
		av1.AV1SDPRTCPFeedbackLRR != "ccm lrr" {
		t.Fatalf("unexpected AV1 rtcp-fb constants")
	}
}

func TestAV1RTCPLayerRefreshLayerIndexRoundTrip(t *testing.T) {
	var buf [2]byte
	index := av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 7, SpatialID: 3}
	n, err := av1.PutAV1RTCPLayerRefreshLayerIndex(buf[:], index)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || buf != [2]byte{0x07, 0x03} {
		t.Fatalf("encoded layer index n=%d bytes=%#v", n, buf)
	}
	got, n, err := av1.ParseAV1RTCPLayerRefreshLayerIndex(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || got != index {
		t.Fatalf("parsed layer index n=%d got=%+v want=%+v", n, got, index)
	}

	got, _, err = av1.ParseAV1RTCPLayerRefreshLayerIndex([]byte{0xfa, 0xf1})
	if err != nil {
		t.Fatalf("Parse with reserved bits returned error: %v", err)
	}
	if got != (av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1}) {
		t.Fatalf("reserved-bit parse = %+v", got)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshLayerIndex([]byte{0x00, 0x04}); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse high SID bit error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if _, err := av1.PutAV1RTCPLayerRefreshLayerIndex(buf[:1], index); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Put error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshLayerIndex(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Parse error = %v, want ErrRTCPShortBuffer", err)
	}
}

func TestAV1RTCPLayerRefreshRequestEntryRoundTrip(t *testing.T) {
	entry := av1.AV1RTCPLayerRefreshRequestEntry{
		SSRC:           0x11223344,
		SequenceNumber: 250,
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	var buf [12]byte
	n, err := av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:], entry)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	wantBytes := [12]byte{0x11, 0x22, 0x33, 0x44, 0xfa, 0xe2, 0x00, 0x00, 0x02, 0x01, 0x00, 0x00}
	if n != 12 || buf != wantBytes {
		t.Fatalf("encoded LRR entry n=%d bytes=%#v want=%#v", n, buf, wantBytes)
	}
	got, n, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	if n != 12 || got != entry {
		t.Fatalf("parsed LRR entry n=%d got=%+v want=%+v", n, got, entry)
	}

	noCurrent := entry
	noCurrent.CurrentPresent = false
	noCurrent.Current = av1.AV1RTCPLayerRefreshLayerIndex{}
	n, err = av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:], noCurrent)
	if err != nil {
		t.Fatalf("Put no-current LRR entry: %v", err)
	}
	if buf[5] != 98 || buf[10] != 0 || buf[11] != 0 {
		t.Fatalf("no-current encoded bytes=%#v", buf)
	}
	buf[10], buf[11] = 0x07, 0x03
	got, _, err = av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("Parse no-current with ignored current field: %v", err)
	}
	if got.CurrentPresent || got.Current != (av1.AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("no-current parse got=%+v", got)
	}
}

func TestAV1RTCPLayerRefreshRequestRejectsInvalid(t *testing.T) {
	valid := av1.AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	for _, entry := range []av1.AV1RTCPLayerRefreshRequestEntry{
		{PayloadType: 128, Target: valid.Target},
		{PayloadType: 98, Target: av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 8}},
		{PayloadType: 98, Target: av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 4}},
		{PayloadType: 98, Target: valid.Target, Current: av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1}},
		{PayloadType: 98, Target: valid.Current, CurrentPresent: true, Current: valid.Target},
		{PayloadType: 98, Target: valid.Target, CurrentPresent: true, Current: valid.Target},
	} {
		if err := entry.Validate(); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
			t.Fatalf("Validate(%+v) error = %v, want ErrRTCPInvalidLayerRefreshRequest", entry, err)
		}
	}

	var buf [12]byte
	if _, err := av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:1], valid); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Put entry error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Parse entry error = %v, want ErrRTCPShortBuffer", err)
	}
	buf = [12]byte{0, 0, 0, 1, 1, 0x80 | 98, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:]); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse no-upgrade entry error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
}

func TestEncoderWebRTCValidateLayerRefreshRequest(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(av1.EncoderScalabilityModeL2T2)
	valid := av1.AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, valid); err != nil {
		t.Fatalf("EncoderWebRTCValidateLayerRefreshRequest valid: %v", err)
	}

	badTemporal := valid
	badTemporal.Target.TemporalID = 2
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, badTemporal); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad temporal error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	badSpatial := valid
	badSpatial.Target.SpatialID = 2
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, badSpatial); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad spatial error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(av1.EncoderConfig{}, valid); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("invalid config error = %v, want ErrEncoderInvalidConfig", err)
	}
}

func testAV1RTCPEncoderConfig(mode av1.EncoderScalabilityMode) av1.EncoderConfig {
	return av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		Scalability:       mode,
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
		RateControl:       av1.EncoderRateControlCBR,
	}
}
