package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendSegmentationParamsPayloadDisabled(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	payload, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, SegmentationParams{}, nil)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.Enabled || parsed.UpdateMap || parsed.UpdateData || parsed.AllLossless {
		t.Fatalf("parsed segmentation=%+v", parsed)
	}
}

func TestAppendSegmentationParamsPayloadDisabledInactiveSentinel(t *testing.T) {
	// A caller bridging decoder SegmentationData carries the SEG_LVL_REF_FRAME
	// disabled sentinel (RefFrame == -1) on inactive segments. With segmentation
	// disabled the feature data is never written, so this must encode to the same
	// single 'false' bit as a zero-value SegmentationParams rather than being
	// rejected as active data.
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	seg := SegmentationParams{Data: inactiveEncoderSegmentationData()}
	payload, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, seg, nil)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.Enabled {
		t.Fatalf("parsed segmentation=%+v", parsed)
	}
}

func TestAppendSegmentationParamsPayloadPrimaryRefNoneUpdateData(t *testing.T) {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaQ = -8
	data.Segments[1].DeltaLFYV = 3
	data.Segments[2].RefFrame = 4
	data.Segments[3].Skip = true
	data.Segments[4].GlobalMV = true
	seg := SegmentationParams{
		Enabled:    true,
		UpdateMap:  true,
		UpdateData: true,
		Data:       data,
	}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	_, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, seg, nil)
	if !parsed.Enabled || !parsed.UpdateMap || !parsed.UpdateData {
		t.Fatalf("parsed segmentation flags=%+v", parsed)
	}
	if parsed.Data.Segments[0].DeltaQ != -8 || parsed.QIndex[0] != 12 {
		t.Fatalf("parsed segment0=%+v q=%d", parsed.Data.Segments[0], parsed.QIndex[0])
	}
	if parsed.Data.Segments[1].DeltaLFYV != 3 || parsed.Data.Segments[2].RefFrame != 4 ||
		!parsed.Data.Segments[3].Skip || !parsed.Data.Segments[4].GlobalMV {
		t.Fatalf("parsed data=%+v", parsed.Data)
	}
	if !parsed.Data.Preskip || parsed.Data.LastActiveID != 4 {
		t.Fatalf("parsed derived data=%+v", parsed.Data)
	}
}

func TestAppendSegmentationParamsPayloadCopiesPreviousData(t *testing.T) {
	previous := inactiveParserSegmentationData()
	previous.Segments[2].DeltaQ = 9
	previous.LastActiveID = 2
	seg := SegmentationParams{
		Enabled:    true,
		UpdateMap:  false,
		UpdateData: false,
		Data:       inactiveEncoderSegmentationData(),
	}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	parsedPrefix := parser.FrameHeaderPrefix{PrimaryRefFrame: 0}
	_, parsed := appendAndParseSegmentationParams(t, prefix, parsedPrefix, seg, &previous)
	if !parsed.Enabled || parsed.UpdateMap || parsed.UpdateData {
		t.Fatalf("parsed flags=%+v", parsed)
	}
	if parsed.Data.Segments[2].DeltaQ != 9 || parsed.Data.LastActiveID != 2 {
		t.Fatalf("parsed previous data=%+v", parsed.Data)
	}
}

func TestAppendSegmentationParamsPayloadRejectsInvalid(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	active := inactiveEncoderSegmentationData()
	active.Segments[0].RefFrame = 8
	cases := [...]SegmentationParams{
		{UpdateMap: true},
		{Enabled: true, UpdateMap: false, UpdateData: true},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: active},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: segmentationDataWithDeltaQ(256)},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: segmentationDataWithDeltaLF(64)},
	}
	var buf [64]byte
	for _, seg := range cases {
		if _, err := AppendSegmentationParamsPayload(buf[:0], prefix, seg); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("AppendSegmentationParamsPayload(%+v) err=%v want ErrInvalidFrame", seg, err)
		}
	}
	nonNone := FrameHeaderPrefix{PrimaryRefFrame: 0}
	if _, err := AppendSegmentationParamsPayload(buf[:0], nonNone, SegmentationParams{Enabled: true, TemporalUpdate: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("temporal without update map err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendSegmentationParamsPayloadShortBuffer(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendSegmentationParamsPayload(dst, prefix, SegmentationParams{})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendSegmentationParamsPayloadAllocs(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	seg := SegmentationParams{}
	var buf [2]byte
	if _, err := AppendSegmentationParamsPayload(buf[:0], prefix, seg); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = SegmentationParamsPayloadSize(prefix, seg)
		_, _ = AppendSegmentationParamsPayload(buf[:0], prefix, seg)
	})
	if allocs != 0 {
		t.Fatalf("AppendSegmentationParamsPayload allocated: %f", allocs)
	}
}

func appendAndParseSegmentationParams(t *testing.T, prefix FrameHeaderPrefix, parsedPrefix parser.FrameHeaderPrefix, seg SegmentationParams, previous *parser.SegmentationData) ([]byte, parser.SegmentationParams) {
	t.Helper()
	size, err := SegmentationParamsPayloadSize(prefix, seg)
	if err != nil {
		t.Fatalf("SegmentationParamsPayloadSize: %v", err)
	}
	var buf [96]byte
	out, err := AppendSegmentationParamsPayload(buf[:0], prefix, seg)
	if err != nil {
		t.Fatalf("AppendSegmentationParamsPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
	parsed, err := parser.ParseSegmentationParams(out, parsedPrefix, parser.QuantizationParams{BaseQIdx: 20}, previous)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	return out, parsed
}

func inactiveEncoderSegmentationData() SegmentationData {
	var data SegmentationData
	for i := uint8(0); i < MaxSegments; i++ {
		data.Segments[i].RefFrame = -1
	}
	return data
}

func inactiveParserSegmentationData() parser.SegmentationData {
	var data parser.SegmentationData
	data.LastActiveID = -1
	for i := uint8(0); i < parser.MaxSegments; i++ {
		data.Segments[i].RefFrame = -1
	}
	return data
}

func segmentationDataWithDeltaQ(delta int16) SegmentationData {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaQ = delta
	return data
}

func segmentationDataWithDeltaLF(delta int16) SegmentationData {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaLFYV = delta
	return data
}
