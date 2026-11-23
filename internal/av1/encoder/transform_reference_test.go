package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestAppendTransformReferenceParamsPayloadLosslessIntraReadsNoBits(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	params := TransformReferenceParams{TransformMode: TransformMode4x4Only, ReferenceMode: ReferenceModeSingle}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, true, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.TransformMode != parser.TransformMode4x4Only || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 0 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadLargestAndSwitchable(t *testing.T) {
	cases := [...]struct {
		name string
		mode TransformMode
		want parser.TransformMode
	}{
		{name: "largest", mode: TransformModeLargest, want: parser.TransformModeLargest},
		{name: "switchable", mode: TransformModeSwitchable, want: parser.TransformModeSwitchable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
			params := TransformReferenceParams{TransformMode: tc.mode, ReferenceMode: ReferenceModeSingle}
			payload, parsed := appendAndParseTransformReferenceParams(t, prefix, false, params)
			if len(payload) != 1 {
				t.Fatalf("payload len=%d want 1", len(payload))
			}
			if parsed.TransformMode != tc.want || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 1 {
				t.Fatalf("parsed transform/reference=%+v want mode=%d bits=1", parsed, tc.want)
			}
		})
	}
}

func TestAppendTransformReferenceParamsPayloadInterReferenceMode(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := TransformReferenceParams{TransformMode: TransformMode4x4Only, ReferenceMode: ReferenceModeSelect}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, true, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.TransformMode != parser.TransformMode4x4Only || parsed.ReferenceMode != parser.ReferenceModeSelect || parsed.BitsRead != 1 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadInterSwitchableSingleReference(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := TransformReferenceParams{TransformMode: TransformModeSwitchable, ReferenceMode: ReferenceModeSingle}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, false, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.TransformMode != parser.TransformModeSwitchable || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 2 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadRejectsInvalid(t *testing.T) {
	intra := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	inter := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	cases := [...]struct {
		prefix      FrameHeaderPrefix
		allLossless bool
		params      TransformReferenceParams
	}{
		{prefix: intra, allLossless: true, params: TransformReferenceParams{TransformMode: TransformModeLargest}},
		{prefix: intra, allLossless: false, params: TransformReferenceParams{TransformMode: TransformMode4x4Only}},
		{prefix: intra, allLossless: false, params: TransformReferenceParams{TransformMode: TransformModeLargest, ReferenceMode: ReferenceModeSelect}},
		{prefix: inter, allLossless: false, params: TransformReferenceParams{TransformMode: TransformModeSwitchable, ReferenceMode: ReferenceModeCompound}},
		{prefix: inter, allLossless: false, params: TransformReferenceParams{TransformMode: TransformMode(9), ReferenceMode: ReferenceModeSingle}},
	}
	var buf [2]byte
	for _, tc := range cases {
		if _, err := appendTransformReferenceParamsPayload(buf[:0], tc.prefix, tc.allLossless, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendTransformReferenceParamsPayload(%+v,%v,%+v) err=%v want ErrInvalidFrame", tc.prefix, tc.allLossless, tc.params, err)
		}
	}
}

func appendAndParseTransformReferenceParams(t *testing.T, prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) ([]byte, parser.TransformReferenceParams) {
	t.Helper()
	var buf [2]byte
	payload, err := appendTransformReferenceParamsPayload(buf[:0], prefix, allLossless, params)
	if err != nil {
		t.Fatalf("appendTransformReferenceParamsPayload: %v", err)
	}
	parsed, err := parser.ParseTransformReferenceParams(
		payload,
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType)},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.RestorationParams{},
	)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	return payload, parsed
}
