package parser

import (
	"errors"
	"testing"
)

func TestParseFilmGrainParamsSkippedBySequence(t *testing.T) {
	seq := SequenceHeader{ColorConfig: ColorConfig{BitDepth: 8}}
	params, err := ParseFilmGrainParams(nil,
		seq,
		FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true},
		FrameSize{},
		nil,
		GlobalMotionParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if params.ParamsPresent || params.Apply || params.BitsRead != 0 || params.BitDepth != 8 {
		t.Fatalf("film grain=%+v", params)
	}
}

func TestParseFilmGrainParamsApplyFalse(t *testing.T) {
	seq := filmGrainSequence()
	var w testBitWriter
	w.writeBits(0b10101, 5)
	w.writeBool(false)

	params, err := ParseFilmGrainParams(w.bytes(),
		seq,
		FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true},
		FrameSize{},
		nil,
		GlobalMotionParams{BitsRead: 5},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.ParamsPresent || params.Apply || params.BitsRead != 6 || params.BitDepth != 8 {
		t.Fatalf("film grain=%+v", params)
	}
}

func TestParseFilmGrainParamsUpdateMinimal(t *testing.T) {
	seq := filmGrainSequence()
	payload, bits := filmGrainMinimalUpdatePayload()

	params, err := ParseFilmGrainParams(payload,
		seq,
		FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true},
		FrameSize{},
		nil,
		GlobalMotionParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.Apply || !params.Update || params.Seed != 0x1234 || params.BitsRead != bits {
		t.Fatalf("film grain=%+v bits=%d", params, bits)
	}
	if params.NumYPoints != 1 || params.YPoints[0][0] != 10 || params.YPoints[0][1] != 20 {
		t.Fatalf("y points=%+v", params.YPoints[:params.NumYPoints])
	}
	if params.ChromaScalingFromLuma || params.NumCbPoints != 0 || params.NumCrPoints != 0 {
		t.Fatalf("chroma fields=%+v", params)
	}
	if params.ScalingShift != 9 || params.ARCoeffLag != 0 || params.ARCoeffShift != 8 || params.GrainScaleShift != 3 {
		t.Fatalf("shift fields=%+v", params)
	}
	if !params.Overlap || params.ClipToRestrictedRange {
		t.Fatalf("range flags=%+v", params)
	}
}

func TestParseFilmGrainParamsRejectsNonIncreasingPoints(t *testing.T) {
	seq := filmGrainSequence()
	var w testBitWriter
	w.writeBool(true)     // apply_grain
	w.writeBits(0x99, 16) // grain_seed
	w.writeBits(2, 4)     // num_y_points
	w.writeBits(10, 8)
	w.writeBits(20, 8)
	w.writeBits(10, 8)
	w.writeBits(30, 8)

	_, err := ParseFilmGrainParams(w.bytes(),
		seq,
		FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true},
		FrameSize{},
		nil,
		GlobalMotionParams{},
	)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseFilmGrainParams err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseFilmGrainParamsCopiesReference(t *testing.T) {
	seq := filmGrainSequence()
	var w testBitWriter
	w.writeBool(true)       // apply_grain
	w.writeBits(0x2244, 16) // grain_seed
	w.writeBool(false)      // update_grain
	w.writeBits(2, 3)       // film_grain_params_ref_idx

	var size FrameSize
	size.RefFrameIdx[0] = 2
	var refs ReferenceState
	refs.Frames[2] = ReferenceFrame{
		Valid: true,
		FilmGrain: FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Update:        true,
			Seed:          0x1111,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [MaxFilmGrainYPoints][2]uint8{{40, 50}},
			ScalingShift:  9,
		},
	}

	params, err := ParseFilmGrainParams(w.bytes(),
		seq,
		FrameHeaderPrefix{FrameType: FrameTypeInter, ShowFrame: true},
		size,
		&refs,
		GlobalMotionParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !params.Apply || params.Update || params.RefIdx != 2 || params.Seed != 0x2244 {
		t.Fatalf("reference copy flags=%+v", params)
	}
	if params.NumYPoints != 1 || params.YPoints[0][0] != 40 || params.YPoints[0][1] != 50 {
		t.Fatalf("copied y points=%+v", params)
	}
	if params.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", params.BitsRead, w.bit)
	}
}

func TestParseFilmGrainParamsAllocs(t *testing.T) {
	seq := filmGrainSequence()
	payload, _ := filmGrainMinimalUpdatePayload()
	prefix := FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true}

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseFilmGrainParams(payload, seq, prefix, FrameSize{}, nil, GlobalMotionParams{})
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFilmGrainParams allocated: %f", allocs)
	}
}

func BenchmarkParseFilmGrainParams(b *testing.B) {
	seq := filmGrainSequence()
	payload, _ := filmGrainMinimalUpdatePayload()
	prefix := FrameHeaderPrefix{FrameType: FrameTypeKey, ShowFrame: true}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseFilmGrainParams(payload, seq, prefix, FrameSize{}, nil, GlobalMotionParams{})
	}
}

func filmGrainSequence() SequenceHeader {
	return SequenceHeader{
		FilmGrainParamsPresent: true,
		ColorConfig: ColorConfig{
			BitDepth: 8,
		},
	}
}

func filmGrainMinimalUpdatePayload() ([]byte, int) {
	var w testBitWriter
	w.writeBool(true)       // apply_grain
	w.writeBits(0x1234, 16) // grain_seed
	w.writeBits(1, 4)       // num_y_points
	w.writeBits(10, 8)
	w.writeBits(20, 8)
	w.writeBool(false) // chroma_scaling_from_luma
	w.writeBits(0, 4)  // num_cb_points
	w.writeBits(0, 4)  // num_cr_points
	w.writeBits(1, 2)  // grain_scaling_minus_8
	w.writeBits(0, 2)  // ar_coeff_lag
	w.writeBits(2, 2)  // ar_coeff_shift minus 6
	w.writeBits(3, 2)  // grain_scale_shift
	w.writeBool(true)  // overlap_flag
	w.writeBool(false) // clip_to_restricted_range
	return w.bytes(), w.bit
}
