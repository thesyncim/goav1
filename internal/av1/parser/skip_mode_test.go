package parser

import "testing"

func TestParseSkipModeParamsDisabledForSingleReference(t *testing.T) {
	params, err := ParseSkipModeParams(nil,
		SequenceHeader{EnableOrderHint: true, OrderHintBits: 4},
		FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 8},
		FrameSize{},
		nil,
		TransformReferenceParams{ReferenceMode: ReferenceModeSingle},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.Allowed || params.Enabled || params.BitsRead != 0 {
		t.Fatalf("skip mode=%+v", params)
	}
}

func TestParseSkipModeParamsBeforeAfterRefs(t *testing.T) {
	var w testBitWriter
	w.writeBool(true) // skip_mode_present

	seq := SequenceHeader{EnableOrderHint: true, OrderHintBits: 5}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 16}
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint32{15, 17, 14, 18, 13, 19, 12})

	params, err := ParseSkipModeParams(w.bytes(), seq, prefix, size, &refs,
		TransformReferenceParams{ReferenceMode: ReferenceModeSelect},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.Allowed || !params.Enabled || params.RefFrameIdx != [2]uint8{0, 1} || params.BitsRead != 1 {
		t.Fatalf("skip mode=%+v", params)
	}
}

func TestParseSkipModeParamsSecondBeforeRef(t *testing.T) {
	var w testBitWriter
	w.writeBool(false) // skip_mode_present

	seq := SequenceHeader{EnableOrderHint: true, OrderHintBits: 5}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 16}
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint32{12, 14, 11, 10, 9, 8, 7})

	params, err := ParseSkipModeParams(w.bytes(), seq, prefix, size, &refs,
		TransformReferenceParams{ReferenceMode: ReferenceModeSelect},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.Allowed || params.Enabled || params.RefFrameIdx != [2]uint8{0, 1} || params.BitsRead != 1 {
		t.Fatalf("skip mode=%+v", params)
	}
}

func TestParseSkipModeParamsNoLegalPair(t *testing.T) {
	seq := SequenceHeader{EnableOrderHint: true, OrderHintBits: 5}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 16}
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint32{16, 16, 16, 16, 16, 16, 16})

	params, err := ParseSkipModeParams(nil, seq, prefix, size, &refs,
		TransformReferenceParams{ReferenceMode: ReferenceModeSelect},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.Allowed || params.Enabled || params.BitsRead != 0 {
		t.Fatalf("skip mode=%+v", params)
	}
}

func TestParseSkipModeParamsAllocs(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	payload := w.bytes()
	seq := SequenceHeader{EnableOrderHint: true, OrderHintBits: 5}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 16}
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint32{15, 17, 14, 18, 13, 19, 12})
	transformRef := TransformReferenceParams{ReferenceMode: ReferenceModeSelect, BitsRead: 6}

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseSkipModeParams allocated: %f", allocs)
	}
}

func BenchmarkParseSkipModeParams(b *testing.B) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	payload := w.bytes()
	seq := SequenceHeader{EnableOrderHint: true, OrderHintBits: 5}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, OrderHint: 16}
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint32{15, 17, 14, 18, 13, 19, 12})
	transformRef := TransformReferenceParams{ReferenceMode: ReferenceModeSelect, BitsRead: 6}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
	}
}

func skipModeRefs(seq SequenceHeader, orderHints [InterRefsPerFrame]uint32) (FrameSize, ReferenceState) {
	var size FrameSize
	var refs ReferenceState
	for i := range InterRefsPerFrame {
		size.RefFrameIdx[i] = uint8(i)
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: orderHints[i],
			Size: FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}
	return size, refs
}
