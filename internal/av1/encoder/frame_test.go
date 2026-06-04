package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendFrameHeaderPrefixPayloadShownKeyFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		DisableCDFUpdate:   true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	out, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if len(out) != 2 {
		t.Fatalf("prefix bytes=% x len=%d want 2", out, len(out))
	}
	if parsed.ShowExistingFrame || parsed.FrameType != parser.FrameTypeKey || !parsed.ShowFrame ||
		!parsed.ErrorResilientMode || !parsed.DisableCDFUpdate || parsed.FrameSizeOverride ||
		parsed.OrderHint != 5 || parsed.PrimaryRefFrame != parser.PrimaryRefNone {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadHiddenInterFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:         FrameHeaderTypeInter,
		ShowableFrame:     true,
		FrameSizeOverride: true,
		OrderHint:         9,
		PrimaryRefFrame:   3,
	}
	_, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if parsed.FrameType != parser.FrameTypeInter || parsed.ShowFrame || !parsed.ShowableFrame ||
		parsed.ErrorResilientMode || parsed.DisableCDFUpdate || !parsed.FrameSizeOverride ||
		parsed.OrderHint != 9 || parsed.PrimaryRefFrame != 3 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadShowExisting(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.FrameIDNumbersPresent = true
	seq.DeltaFrameIDLength = 4
	seq.AdditionalFrameIDLength = 2
	prefix := FrameHeaderPrefix{
		ShowExistingFrame: true,
		ExistingFrameIdx:  3,
		FrameID:           42,
		PrimaryRefFrame:   EncoderPrimaryRefNone,
	}
	_, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if !parsed.ShowExistingFrame || parsed.ExistingFrameIdx != 3 || parsed.FrameID != 42 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadReducedStillPicture(t *testing.T) {
	seq := reducedStillEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	out, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if len(out) != 1 {
		t.Fatalf("prefix bytes=% x len=%d want 1", out, len(out))
	}
	if parsed.ShowExistingFrame || parsed.FrameType != parser.FrameTypeKey || !parsed.ShowFrame ||
		!parsed.ErrorResilientMode || parsed.DisableCDFUpdate || parsed.BitsRead != 1 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadShortBuffer(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:         FrameHeaderTypeInter,
		ShowableFrame:     true,
		FrameSizeOverride: true,
		OrderHint:         9,
		PrimaryRefFrame:   3,
	}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendFrameHeaderPrefixPayload(dst, seq, prefix)
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, bitstream.ErrShortBuffer)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendFrameHeaderPrefixPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	tests := []struct {
		name   string
		prefix FrameHeaderPrefix
		want   error
	}{
		{
			name: "shown key without implicit error resilient",
			prefix: FrameHeaderPrefix{
				FrameType:       FrameHeaderTypeKey,
				ShowFrame:       true,
				ForceIntegerMV:  true,
				PrimaryRefFrame: EncoderPrimaryRefNone,
			},
			want: ErrInvalidFrame,
		},
		{
			name: "order hint overflow",
			prefix: FrameHeaderPrefix{
				FrameType:         FrameHeaderTypeInter,
				ShowableFrame:     true,
				FrameSizeOverride: true,
				OrderHint:         128,
				PrimaryRefFrame:   3,
			},
			want: ErrInvalidFrame,
		},
		{
			name: "unsupported presentation delay",
			prefix: FrameHeaderPrefix{
				ShowExistingFrame:      true,
				ExistingFrameIdx:       1,
				FramePresentationDelay: 1,
				PrimaryRefFrame:        EncoderPrimaryRefNone,
			},
			want: ErrUnsupported,
		},
	}
	var buf [8]byte
	for _, tt := range tests {
		if _, err := AppendFrameHeaderPrefixPayload(buf[:0], seq, tt.prefix); !errors.Is(err, tt.want) {
			t.Fatalf("%s err=%v want %v", tt.name, err, tt.want)
		}
	}
}

func TestAppendFrameHeaderPrefixPayloadAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:         FrameHeaderTypeInter,
		ShowableFrame:     true,
		FrameSizeOverride: true,
		OrderHint:         9,
		PrimaryRefFrame:   3,
	}
	var buf [8]byte
	if _, err := AppendFrameHeaderPrefixPayload(buf[:0], seq, prefix); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendFrameHeaderPrefixPayload(buf[:0], seq, prefix)
	})
	if allocs != 0 {
		t.Fatalf("AppendFrameHeaderPrefixPayload allocated: %f", allocs)
	}
}

func appendAndParseFrameHeaderPrefix(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix) ([]byte, parser.FrameHeaderPrefix) {
	t.Helper()
	var seqBuf [96]byte
	seqPayload, err := AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	var buf [16]byte
	out, err := AppendFrameHeaderPrefixPayload(buf[:0], seq, prefix)
	if err != nil {
		t.Fatalf("AppendFrameHeaderPrefixPayload: %v", err)
	}
	parsed, err := parser.ParseFrameHeaderPrefix(out, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	return out, parsed
}

func reducedStillEncoderSequenceHeader() SequenceHeader {
	seq := realtimeEncoderSequenceHeader()
	seq.StillPicture = true
	seq.ReducedStillPictureHeader = true
	seq.EnableInterIntraCompound = false
	seq.EnableMaskedCompound = false
	seq.EnableDualFilter = false
	seq.EnableOrderHint = false
	seq.EnableJNTComp = false
	seq.EnableRefFrameMVS = false
	seq.SeqForceScreenContentTools = 0
	seq.SeqForceIntegerMV = 0
	seq.OrderHintBits = 0
	seq.EnableSuperRes = false
	seq.EnableRestoration = false
	seq.FilmGrainParamsPresent = false
	return seq
}
