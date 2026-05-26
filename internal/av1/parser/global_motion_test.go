package parser

import "testing"

func TestParseGlobalMotionParamsIntraReadsNoBits(t *testing.T) {
	params, err := ParseGlobalMotionParams(nil,
		FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize{},
		TileInfo{},
		nil,
		FrameModeParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.BitsRead != 0 {
		t.Fatalf("global motion bits=%d", params.BitsRead)
	}
	for i := range InterRefsPerFrame {
		if params.Ref[i] != DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, params.Ref[i])
		}
	}
}

func TestParseGlobalMotionParamsInterIdentity(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	for range InterRefsPerFrame {
		w.writeBool(false)
	}

	size, refs := globalMotionRefs()
	params, err := ParseGlobalMotionParams(w.bytes(),
		FrameHeaderPrefix{FrameType: FrameTypeInter, PrimaryRefFrame: PrimaryRefNone},
		size,
		TileInfo{},
		&refs,
		FrameModeParams{BitsRead: 6},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", params.BitsRead, w.bit)
	}
	for i := range InterRefsPerFrame {
		if params.Ref[i].Type != GlobalMotionIdentity || params.Ref[i] != DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, params.Ref[i])
		}
	}
}

func TestParseGlobalMotionParamsTranslationConsumesParams(t *testing.T) {
	var w testBitWriter
	w.writeBool(true)  // is_global
	w.writeBool(false) // is_rot_zoom
	w.writeBool(true)  // is_translation
	for range 14 {
		w.writeBool(false)
	}
	for i := 1; i < InterRefsPerFrame; i++ {
		w.writeBool(false)
	}

	size, refs := globalMotionRefs()
	params, err := ParseGlobalMotionParams(w.bytes(),
		FrameHeaderPrefix{FrameType: FrameTypeInter, PrimaryRefFrame: PrimaryRefNone},
		size,
		TileInfo{AllowHighPrecisionMV: true},
		&refs,
		FrameModeParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.Ref[0].Type != GlobalMotionTranslation {
		t.Fatalf("ref[0]=%+v", params.Ref[0])
	}
	if params.BitsRead <= InterRefsPerFrame {
		t.Fatalf("BitsRead=%d", params.BitsRead)
	}
}

func TestParseGlobalMotionParamsAllocs(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	for range InterRefsPerFrame {
		w.writeBool(false)
	}
	payload := w.bytes()
	size, refs := globalMotionRefs()
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, PrimaryRefFrame: PrimaryRefNone}
	frameMode := FrameModeParams{BitsRead: 6}

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseGlobalMotionParams(payload, prefix, size, TileInfo{}, &refs, frameMode)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseGlobalMotionParams allocated: %f", allocs)
	}
}

func BenchmarkParseGlobalMotionParams(b *testing.B) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	for range InterRefsPerFrame {
		w.writeBool(false)
	}
	payload := w.bytes()
	size, refs := globalMotionRefs()
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter, PrimaryRefFrame: PrimaryRefNone}
	frameMode := FrameModeParams{BitsRead: 6}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseGlobalMotionParams(payload, prefix, size, TileInfo{}, &refs, frameMode)
	}
}

func globalMotionRefs() (FrameSize, ReferenceState) {
	var size FrameSize
	var refs ReferenceState
	defaultGlobal := DefaultGlobalMotionParams()
	for i := range InterRefsPerFrame {
		size.RefFrameIdx[i] = uint8(i)
		refs.Frames[i] = ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
		}
	}
	return size, refs
}
