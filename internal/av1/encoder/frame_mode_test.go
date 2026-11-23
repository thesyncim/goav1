package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendFrameModeParamsPayloadIntraReducedOnly(t *testing.T) {
	seq := frameModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true}
	params := FrameModeParams{ReducedTxSet: true}
	payload, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.AllowWarpedMotion || !parsed.ReducedTxSet || parsed.BitsRead != 1 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadInterWarpedAndReduced(t *testing.T) {
	seq := frameModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true}
	payload, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.AllowWarpedMotion || !parsed.ReducedTxSet || parsed.BitsRead != 2 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadSkipsWarpedWhenDisabled(t *testing.T) {
	seq := frameModeSequenceHeader()
	seq.EnableWarpedMotion = false
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := FrameModeParams{ReducedTxSet: false}
	_, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if parsed.AllowWarpedMotion || parsed.ReducedTxSet || parsed.BitsRead != 1 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadRejectsInvalid(t *testing.T) {
	seq := frameModeSequenceHeader()
	cases := [...]struct {
		seq    SequenceHeader
		prefix FrameHeaderPrefix
		params FrameModeParams
	}{
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ErrorResilientMode: true},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    frameModeSequenceHeaderWarpDisabled(),
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9)},
			params: FrameModeParams{},
		},
	}
	var buf [1]byte
	for _, tc := range cases {
		if _, err := appendFrameModeParamsPayload(buf[:0], tc.seq, tc.prefix, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendFrameModeParamsPayload(%+v,%+v) err=%v want ErrInvalidFrame", tc.prefix, tc.params, err)
		}
	}
}

func appendAndParseFrameModeParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) ([]byte, parser.FrameModeParams) {
	t.Helper()
	var buf [1]byte
	payload, err := appendFrameModeParamsPayload(buf[:0], seq, prefix, params)
	if err != nil {
		t.Fatalf("appendFrameModeParamsPayload: %v", err)
	}
	parsed, err := parser.ParseFrameModeParams(
		payload,
		parser.SequenceHeader{EnableWarpedMotion: seq.EnableWarpedMotion},
		parser.FrameHeaderPrefix{
			FrameType:          parser.FrameType(prefix.FrameType),
			ErrorResilientMode: prefix.ErrorResilientMode,
		},
		parser.SkipModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	return payload, parsed
}

func frameModeSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:              64,
		MaxFrameHeight:             64,
		Use128x128Superblock:       true,
		EnableFilterIntra:          true,
		EnableIntraEdgeFilter:      true,
		EnableInterIntraCompound:   true,
		EnableMaskedCompound:       true,
		EnableWarpedMotion:         true,
		EnableDualFilter:           true,
		EnableOrderHint:            true,
		EnableJNTComp:              true,
		EnableRefFrameMVS:          true,
		SeqForceScreenContentTools: SequenceSelectScreenContentTools,
		SeqForceIntegerMV:          SequenceSelectIntegerMV,
		OrderHintBits:              8,
		EnableSuperRes:             true,
		EnableCDEF:                 true,
		EnableRestoration:          true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func frameModeSequenceHeaderWarpDisabled() SequenceHeader {
	seq := frameModeSequenceHeader()
	seq.EnableWarpedMotion = false
	return seq
}
