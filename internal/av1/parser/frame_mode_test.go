package parser

import "testing"

func TestParseFrameModeParamsIntraReadsReducedTxSet(t *testing.T) {
	var w testBitWriter
	w.writeBool(true) // reduced_tx_set

	params, err := ParseFrameModeParams(w.bytes(),
		SequenceHeader{EnableWarpedMotion: true},
		FrameHeaderPrefix{FrameType: FrameTypeKey, ErrorResilientMode: true},
		SkipModeParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.AllowWarpedMotion || !params.ReducedTxSet || params.BitsRead != 1 {
		t.Fatalf("frame mode=%+v", params)
	}
}

func TestParseFrameModeParamsInterWarpedMotion(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)  // allow_warped_motion
	w.writeBool(false) // reduced_tx_set

	params, err := ParseFrameModeParams(w.bytes(),
		SequenceHeader{EnableWarpedMotion: true},
		FrameHeaderPrefix{FrameType: FrameTypeInter},
		SkipModeParams{BitsRead: 6},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.AllowWarpedMotion || params.ReducedTxSet || params.BitsRead != w.bit {
		t.Fatalf("frame mode=%+v want bits=%d", params, w.bit)
	}
}

func TestParseFrameModeParamsErrorResilientSkipsWarpedMotion(t *testing.T) {
	var w testBitWriter
	w.writeBool(true) // reduced_tx_set

	params, err := ParseFrameModeParams(w.bytes(),
		SequenceHeader{EnableWarpedMotion: true},
		FrameHeaderPrefix{FrameType: FrameTypeInter, ErrorResilientMode: true},
		SkipModeParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.AllowWarpedMotion || !params.ReducedTxSet || params.BitsRead != 1 {
		t.Fatalf("frame mode=%+v", params)
	}
}

func TestParseFrameModeParamsAllocs(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	w.writeBool(false)
	payload := w.bytes()
	seq := SequenceHeader{EnableWarpedMotion: true}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter}
	skip := SkipModeParams{BitsRead: 6}

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseFrameModeParams(payload, seq, prefix, skip)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameModeParams allocated: %f", allocs)
	}
}

func BenchmarkParseFrameModeParams(b *testing.B) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	w.writeBool(false)
	payload := w.bytes()
	seq := SequenceHeader{EnableWarpedMotion: true}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter}
	skip := SkipModeParams{BitsRead: 6}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseFrameModeParams(payload, seq, prefix, skip)
	}
}
