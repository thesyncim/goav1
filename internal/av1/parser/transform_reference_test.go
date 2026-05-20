package parser

import "testing"

func TestParseTransformReferenceParamsLosslessIntraReadsNoBits(t *testing.T) {
	params, err := ParseTransformReferenceParams(nil,
		FrameHeaderPrefix{FrameType: FrameTypeKey},
		SegmentationParams{AllLossless: true},
		RestorationParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.TransformMode != TransformMode4x4Only || params.ReferenceMode != ReferenceModeSingle || params.BitsRead != 0 {
		t.Fatalf("transform/reference=%+v", params)
	}
}

func TestParseTransformReferenceParamsLargestAndSwitchable(t *testing.T) {
	cases := []struct {
		name string
		bit  bool
		want TransformMode
	}{
		{name: "largest", bit: false, want: TransformModeLargest},
		{name: "switchable", bit: true, want: TransformModeSwitchable},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var w testBitWriter
			w.writeBool(tc.bit)

			params, err := ParseTransformReferenceParams(w.bytes(),
				FrameHeaderPrefix{FrameType: FrameTypeKey},
				SegmentationParams{AllLossless: false},
				RestorationParams{},
			)
			if err != nil {
				t.Fatal(err)
			}
			if params.TransformMode != tc.want || params.ReferenceMode != ReferenceModeSingle || params.BitsRead != 1 {
				t.Fatalf("transform/reference=%+v want mode=%d bits=1", params, tc.want)
			}
		})
	}
}

func TestParseTransformReferenceParamsInterLosslessReferenceMode(t *testing.T) {
	var w testBitWriter
	w.writeBool(true) // reference_select

	params, err := ParseTransformReferenceParams(w.bytes(),
		FrameHeaderPrefix{FrameType: FrameTypeInter},
		SegmentationParams{AllLossless: true},
		RestorationParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.TransformMode != TransformMode4x4Only || params.ReferenceMode != ReferenceModeSelect || params.BitsRead != 1 {
		t.Fatalf("transform/reference=%+v", params)
	}
}

func TestParseTransformReferenceParamsSkipsRestorationBits(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101, 3)
	w.writeBool(true) // tx_mode_select
	w.writeBool(true) // reference_select

	params, err := ParseTransformReferenceParams(w.bytes(),
		FrameHeaderPrefix{FrameType: FrameTypeInter},
		SegmentationParams{AllLossless: false},
		RestorationParams{BitsRead: 3},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.TransformMode != TransformModeSwitchable || params.ReferenceMode != ReferenceModeSelect || params.BitsRead != w.bit {
		t.Fatalf("transform/reference=%+v want bits=%d", params, w.bit)
	}
}

func TestParseTransformReferenceParamsAllocs(t *testing.T) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	w.writeBool(false)
	payload := w.bytes()
	restoration := RestorationParams{BitsRead: 6}
	seg := SegmentationParams{AllLossless: false}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter}

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTransformReferenceParams allocated: %f", allocs)
	}
}

func BenchmarkParseTransformReferenceParams(b *testing.B) {
	var w testBitWriter
	w.writeBits(0b101010, 6)
	w.writeBool(true)
	w.writeBool(false)
	payload := w.bytes()
	restoration := RestorationParams{BitsRead: 6}
	seg := SegmentationParams{AllLossless: false}
	prefix := FrameHeaderPrefix{FrameType: FrameTypeInter}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ParseTransformReferenceParams(payload, prefix, seg, restoration)
	}
}
