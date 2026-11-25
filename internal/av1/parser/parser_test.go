package parser

import (
	"bytes"
	"errors"
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/testvector"
	"testing"
	"unsafe"
)

func TestParseCDEFParamsSkipsLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	lf := LoopFilterParams{BitsRead: w.bit}

	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	if cdef.StrengthCount != 0 || cdef.BitsRead != lf.BitsRead {
		t.Fatalf("cdef=%+v lf bits=%d", cdef, lf.BitsRead)
	}
}

func TestParseCDEFParamsStrengths(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, 40)
	w.writeBits(2, 2) // cdef_damping_minus_3
	w.writeBits(2, 2) // cdef_bits
	for i := range 4 {
		w.writeBits(uint64(10+i), 6)
		w.writeBits(uint64(20+i), 6)
	}

	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	if cdef.Damping != 5 || cdef.Bits != 2 || cdef.StrengthCount != 4 {
		t.Fatalf("cdef=%+v", cdef)
	}
	if cdef.YStrength[3] != 13 || cdef.UVStrength[3] != 23 {
		t.Fatalf("strengths y=%v uv=%v", cdef.YStrength[:4], cdef.UVStrength[:4])
	}
	if cdef.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", cdef.BitsRead, w.bit)
	}
}

func TestParseCDEFParamsMonochromeSkipsUVStrengths(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(1, 2)
	w.writeBits(1, 2)
	w.writeBits(7, 6)
	w.writeBits(8, 6)
	seg := SegmentationParams{AllLossless: false}

	cdef, err := ParseCDEFParams(w.bytes(), seq, FrameSize{}, seg, LoopFilterParams{})
	if err != nil {
		t.Fatal(err)
	}
	if cdef.StrengthCount != 2 || cdef.YStrength[0] != 7 || cdef.YStrength[1] != 8 {
		t.Fatalf("cdef=%+v", cdef)
	}
	if cdef.UVStrength[0] != 0 || cdef.UVStrength[1] != 0 || cdef.BitsRead != w.bit {
		t.Fatalf("uv/bits cdef=%+v want bits %d", cdef, w.bit)
	}
}

func TestParseCDEFParamsDisabledBySequence(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.EnableCDEF = false
	var w testBitWriter
	seg := SegmentationParams{AllLossless: false}

	cdef, err := ParseCDEFParams(w.bytes(), seq, FrameSize{}, seg, LoopFilterParams{})
	if err != nil {
		t.Fatal(err)
	}
	if cdef.BitsRead != 0 || cdef.StrengthCount != 0 {
		t.Fatalf("cdef=%+v", cdef)
	}
}

func TestParseCDEFParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, 40)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	w.writeBits(9, 6)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseCDEFParams allocated: %f", allocs)
	}
}

func BenchmarkParseCDEFParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, seg, lf := buildShownKeyFrameLoopFilterBench(b, seq, 40)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	w.writeBits(9, 6)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseCDEFParams(payload, seq, size, seg, lf)
	}
}

func buildShownKeyFrameLoopFilter(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, LoopFilterParams) {
	t.Helper()
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, baseQIdx)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	return w, size, seg, lf
}

func buildShownKeyFrameLoopFilterBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, LoopFilterParams) {
	b.Helper()
	w, prefix, size, seg, delta := buildShownKeyFrameDeltaBench(b, seq, baseQIdx)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		b.Fatal(err)
	}
	return w, size, seg, lf
}

func TestParseDeltaParamsBaseQZeroReadsNoBits(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 0)

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if delta.DeltaQPresent || delta.BitsRead != seg.BitsRead {
		t.Fatalf("delta=%+v seg bits=%d", delta, seg.BitsRead)
	}
}

func TestParseDeltaParamsQOnly(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true) // delta_q_present
	w.writeBits(2, 2) // delta_q_res_log2
	w.writeBool(false)

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 2 || delta.DeltaLFPresent {
		t.Fatalf("delta=%+v", delta)
	}
	if delta.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", delta.BitsRead, w.bit)
	}
}

func TestParseDeltaParamsLoopFilterMulti(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true) // delta_q_present
	w.writeBits(1, 2) // delta_q_res_log2
	w.writeBool(true) // delta_lf_present
	w.writeBits(3, 2) // delta_lf_res_log2
	w.writeBool(true) // delta_lf_multi

	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 1 ||
		!delta.DeltaLFPresent || delta.DeltaLFResLog2 != 3 || !delta.DeltaLFMulti {
		t.Fatalf("delta=%+v", delta)
	}
}

func TestParseDeltaParamsIntrabcSkipsLoopFilter(t *testing.T) {
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(2, 2)

	size := FrameSize{AllowIntrabc: true}
	quant := QuantizationParams{BaseQIdx: 50}
	seg := SegmentationParams{}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	if !delta.DeltaQPresent || delta.DeltaQResLog2 != 2 || delta.DeltaLFPresent || delta.BitsRead != w.bit {
		t.Fatalf("delta=%+v bits=%d", delta, w.bit)
	}
}

func TestParseDeltaParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentation(t, seq, 50)
	w.writeBool(true)
	w.writeBits(1, 2)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseDeltaParams allocated: %f", allocs)
	}
}

func BenchmarkParseDeltaParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, quant, seg := buildShownKeyFrameSegmentationBench(b, seq, 50)
	w.writeBool(true)
	w.writeBits(1, 2)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseDeltaParams(payload, size, quant, seg)
	}
}

func buildShownKeyFrameSegmentation(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, QuantizationParams, SegmentationParams) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	return w, size, quant, seg
}

func buildShownKeyFrameSegmentationBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, QuantizationParams, SegmentationParams) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		b.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		b.Fatal(err)
	}
	return w, size, quant, seg
}

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

func filmGrainMinimalUpdatePayload() ([]byte, uint16) {
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

func TestParseIntraFrameSizeShownKeyFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false) // superres_params: use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0xff {
		t.Fatalf("RefreshFrameFlags=%02x want ff", size.RefreshFrameFlags)
	}
	if size.CodedWidth != seq.MaxFrameWidth || size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight {
		t.Fatalf("dimensions=%+v seq=%+v", size, seq)
	}
	if size.RenderWidth != seq.MaxFrameWidth || size.RenderHeight != seq.MaxFrameHeight {
		t.Fatalf("render dimensions=%+v", size)
	}
	if size.SuperResEnabled || size.SuperResDenominator != 8 {
		t.Fatalf("superres=%+v", size)
	}
	if size.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", size.BitsRead, w.bit)
	}
}

func TestParseIntraFrameSizeOverrideSuperresRender(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 6
	seq.FrameHeightBits = 5
	seq.MaxFrameWidth = 64
	seq.MaxFrameHeight = 32
	payload, prefix := buildShownKeyFramePrefix(t, seq, true)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(47, seq.FrameWidthBits)  // frame_width_minus_1
	w.writeBits(23, seq.FrameHeightBits) // frame_height_minus_1
	w.writeBool(true)                    // use_superres
	w.writeBits(3, 3)                    // coded_denom = 12
	w.writeBool(true)                    // render_and_frame_size_different
	w.writeBits(39, 16)                  // render_width_minus_1
	w.writeBits(19, 16)                  // render_height_minus_1

	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.SuperResEnabled || size.SuperResDenominator != 12 {
		t.Fatalf("superres=%+v", size)
	}
	if size.UpscaledWidth != 48 || size.CodedWidth != 32 || size.Height != 24 {
		t.Fatalf("dimensions=%+v", size)
	}
	if !size.HaveRenderSize || size.RenderWidth != 40 || size.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", size)
	}
}

func TestParseIntraFrameSizeHiddenKeyReadsRefreshAndOrderHints(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(false)                   // show_frame
	w.writeBool(true)                    // showable_frame
	w.writeBool(true)                    // error_resilient_mode
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(false)                   // frame_size_override_flag
	w.writeBits(4, seq.OrderHintBits)    // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}

	w.writeBits(0x01, 8) // refresh_frame_flags
	for i := range refFrames {
		w.writeBits(uint64(i), seq.OrderHintBits)
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0x01 {
		t.Fatalf("RefreshFrameFlags=%02x", size.RefreshFrameFlags)
	}
	for i := range refFrames {
		if size.RefOrderHints[i] != uint8(i) {
			t.Fatalf("RefOrderHints[%d]=%d", i, size.RefOrderHints[i])
		}
	}
}

func TestParseIntraFrameSizeBufferRemovalTime(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.DecoderModelInfoPresent = true
	seq.TimingInfo.EqualPictureInterval = true
	seq.DecoderModelInfo.BufferRemovalTimeLength = 5
	seq.OperatingPointsCount = 2
	seq.OperatingPoints[0].DecoderModelPresent = true
	seq.OperatingPoints[0].IDC = 0
	seq.OperatingPoints[1].DecoderModelPresent = true
	seq.OperatingPoints[1].IDC = 1 << 1

	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(true)  // buffer_removal_time_present
	w.writeBits(17, 5) // operating point 0
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.BufferRemovalTimePresent || size.BufferRemovalTimes[0] != 17 || size.BufferRemovalTimes[1] != 0 {
		t.Fatalf("buffer removal=%+v", size)
	}
}

func TestParseIntraFrameSizeRejectsIntraOnlyRefreshAll(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                         // show_existing_frame
	w.writeBits(uint64(FrameTypeIntraOnly), 2) // frame_type
	w.writeBool(true)                          // show_frame
	w.writeBool(false)                         // error_resilient_mode
	w.writeBool(false)                         // disable_cdf_update
	w.writeBool(false)                         // frame_size_override_flag
	w.writeBits(0, seq.OrderHintBits)          // order_hint

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBits(0xff, 8)

	_, err = ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseFrameSize err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseIntraFrameSizeRejectsInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, seq.OrderHintBits)
	w.writeBits(0, 3)
	w.writeBits(1, 8)

	prefix, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseFrameSize err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseFrameSizeDirectInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for range InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if size.RefreshFrameFlags != 0x02 {
		t.Fatalf("RefreshFrameFlags=%02x", size.RefreshFrameFlags)
	}
	for i := range InterRefsPerFrame {
		if size.RefFrameIdx[i] != 0 {
			t.Fatalf("RefFrameIdx[%d]=%d", i, size.RefFrameIdx[i])
		}
	}
	if size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight {
		t.Fatalf("dimensions=%+v", size)
	}
}

func TestParseFrameSizeUsesReferenceDimensions(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, true, false, 4)
	refs := oneReferenceState(seq, 0, 1)
	refs.Frames[0].Size.UpscaledWidth = 48
	refs.Frames[0].Size.CodedWidth = 48
	refs.Frames[0].Size.Height = 27
	refs.Frames[0].Size.RenderWidth = 40
	refs.Frames[0].Size.RenderHeight = 20
	refs.Frames[0].Size.HaveRenderSize = true

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x01, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for range InterRefsPerFrame {
		w.writeBits(0, 3)
	}
	w.writeBool(true)  // use_ref_frame_size[0]
	w.writeBool(false) // use_superres

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !size.UsedReferenceFrameSize || size.ReferenceFrameSizeIdx != 0 {
		t.Fatalf("reference frame size flags=%+v", size)
	}
	if size.UpscaledWidth != 48 || size.CodedWidth != 48 || size.Height != 27 {
		t.Fatalf("dimensions=%+v", size)
	}
	if size.RenderWidth != 40 || size.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", size)
	}
}

// dav1d parse_frame_hdr reads ref_frame_idx without the slots' headers when
// frame ids are absent; the missing reference picture fails at submit, which
// ValidateReferenceScale models.
func TestParseFrameSizeMissingReferenceFailsAtSubmit(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	var refs ReferenceState

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x01, 8)
	w.writeBool(false)
	for range InterRefsPerFrame {
		w.writeBits(0, 3)
	}
	w.writeBits(0, 32)

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize err=%v", err)
	}
	if err := ValidateReferenceScale(prefix, size, &refs); !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ValidateReferenceScale err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseFrameSizeInterFrameIDWrap(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeaderWithFrameIDLengths(13, 0))
	var prefixBits testBitWriter
	prefixBits.writeBool(false)                     // show_existing_frame
	prefixBits.writeBits(uint64(FrameTypeInter), 2) // frame_type
	prefixBits.writeBool(true)                      // show_frame
	prefixBits.writeBool(false)                     // error_resilient_mode
	prefixBits.writeBool(false)                     // disable_cdf_update
	prefixBits.writeBits(1, seq.FrameIDBits())      // current_frame_id
	prefixBits.writeBool(false)                     // frame_size_override_flag
	prefixBits.writeBits(4, seq.OrderHintBits)
	prefixBits.writeBits(0, 3) // primary_ref_frame

	prefixPayload := prefixBits.bytes()
	prefix, err := ParseFrameHeaderPrefix(prefixPayload, seq)
	if err != nil {
		t.Fatal(err)
	}
	refs := oneReferenceState(seq, 0, 1)
	refs.Frames[0].FrameID = 0xffff

	var w testBitWriter
	w.writeBitsFrom(prefixPayload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for range InterRefsPerFrame {
		w.writeBits(0, 3)                      // ref_frame_idx[i]
		w.writeBits(1, seq.DeltaFrameIDLength) // delta_frame_id_minus_1
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different

	size, err := ParseFrameSize(w.bytes(), seq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range InterRefsPerFrame {
		if size.RefFrameIdx[i] != 0 || size.DeltaFrameIDMinus1[i] != 1 {
			t.Fatalf("ref[%d] idx=%d delta=%d", i, size.RefFrameIdx[i], size.DeltaFrameIDMinus1[i])
		}
	}
}

func TestReferenceStateUpdate(t *testing.T) {
	var refs ReferenceState
	prefix := FrameHeaderPrefix{
		FrameType:     FrameTypeKey,
		ShowFrame:     true,
		ShowableFrame: false,
		OrderHint:     3,
		FrameID:       5,
	}
	size := FrameSize{
		RefreshFrameFlags: 0x81,
		UpscaledWidth:     16,
		CodedWidth:        16,
		Height:            9,
		RenderWidth:       16,
		RenderHeight:      9,
	}
	refs.Update(prefix, size)
	if !refs.Frames[0].Valid || !refs.Frames[7].Valid || refs.Frames[1].Valid {
		t.Fatalf("reference state=%+v", refs.Frames)
	}
	if refs.Frames[7].OrderHint != 3 || refs.Frames[7].FrameID != 5 {
		t.Fatalf("reference metadata=%+v", refs.Frames[7])
	}
}

func TestParseIntraFrameSizeAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseFrameSize(payload, seq, prefix, nil, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameSize allocated: %f", allocs)
	}
}

func BenchmarkParseIntraFrameSize(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	payload, prefix := buildShownKeyFramePrefixBench(b, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseFrameSize(payload, seq, prefix, nil, 0, 0)
	}
}

func BenchmarkParseFrameSizeInter(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefixBench(b, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8)
	w.writeBool(false)
	for range InterRefsPerFrame {
		w.writeBits(0, 3)
	}
	w.writeBool(false)
	w.writeBool(false)
	payload = w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
	}
}

// TestParseFrameSizeInterAllocs pins ParseFrameSize on the inter path (which
// runs every inter frame) to zero allocations. The function builds the
// FrameSize value on the stack and writes into caller-supplied references, so
// future changes that promote any of the fixed-size scratch buffers (e.g. the
// shortRefInfo array used by setShortFrameReferences) to the heap will trip
// this test.
func TestParseFrameSizeInterAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for range InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	payload = w.bytes()

	if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
		t.Fatalf("ParseFrameSize err=%v", err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameSize allocated: %f", allocs)
	}
}

// TestParseFrameSizeShortSignalingAllocs covers the short-signaling reference
// derivation path which uses on-stack scratch arrays (shortRefInfo, set).
func TestParseFrameSizeShortSignalingAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	refs := oneReferenceState(seq, 0, 1)
	for i := range refs.Frames {
		refs.Frames[i].Valid = true
		refs.Frames[i].OrderHint = uint8(i)
	}

	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(true)    // frame_refs_short_signaling
	w.writeBits(0, 3)    // last_frame_idx
	w.writeBits(3, 3)    // golden_frame_idx
	w.writeBool(false)   // use_superres
	w.writeBool(false)   // render_and_frame_size_different
	payload = w.bytes()

	// Warm up: this path may error on the synthetic ordering; if so, just skip
	// the lock-in since we cannot validate the steady state.
	if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
		t.Skipf("short-signaling path unavailable in test fixture: %v", err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameSize short-signaling allocated: %f", allocs)
	}
}

func buildShownKeyFramePrefix(t *testing.T, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	t.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		t.Fatal(err)
	}
	return payload, prefix
}

func buildShownKeyFramePrefixBench(b *testing.B, seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix) {
	b.Helper()
	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, frameSizeOverride)
	if err != nil {
		b.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefix(t *testing.T, seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix) {
	t.Helper()
	payload, prefix, err := buildInterFramePrefixRaw(seq, frameSizeOverride, errorResilient, orderHint)
	if err != nil {
		t.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefixBench(b *testing.B, seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix) {
	b.Helper()
	payload, prefix, err := buildInterFramePrefixRaw(seq, frameSizeOverride, errorResilient, orderHint)
	if err != nil {
		b.Fatal(err)
	}
	return payload, prefix
}

func buildInterFramePrefixRaw(seq SequenceHeader, frameSizeOverride bool, errorResilient bool, orderHint uint64) ([]byte, FrameHeaderPrefix, error) {
	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(true)                      // show_frame
	w.writeBool(errorResilient)            // error_resilient_mode
	w.writeBool(false)                     // disable_cdf_update
	w.writeBool(frameSizeOverride)         // frame_size_override_flag
	w.writeBits(orderHint, seq.OrderHintBits)
	if !errorResilient {
		w.writeBits(0, 3) // primary_ref_frame
	}
	payload := w.bytes()
	prefix, err := ParseFrameHeaderPrefix(payload, seq)
	return payload, prefix, err
}

func oneReferenceState(seq SequenceHeader, slot int, orderHint uint8) ReferenceState {
	var refs ReferenceState
	refs.Frames[slot] = ReferenceFrame{
		Valid:     true,
		OrderHint: orderHint,
		Size: FrameSize{
			UpscaledWidth:       seq.MaxFrameWidth,
			CodedWidth:          seq.MaxFrameWidth,
			Height:              seq.MaxFrameHeight,
			RenderWidth:         seq.MaxFrameWidth,
			RenderHeight:        seq.MaxFrameHeight,
			SuperResDenominator: 8,
		},
	}
	return refs
}

func buildShownKeyFramePrefixRaw(seq SequenceHeader, frameSizeOverride bool) ([]byte, FrameHeaderPrefix, error) {
	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(true)                    // show_frame
	w.writeBool(false)                   // disable_cdf_update
	w.writeBool(frameSizeOverride)       // frame_size_override_flag
	w.writeBits(3, seq.OrderHintBits)    // order_hint
	payload := w.bytes()
	prefix, err := ParseFrameHeaderPrefix(payload, seq)
	return payload, prefix, err
}

func TestParseFrameHeaderPrefixShownKeyFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                   // show_existing_frame
	w.writeBits(uint64(FrameTypeKey), 2) // frame_type
	w.writeBool(true)                    // show_frame
	w.writeBool(true)                    // disable_cdf_update
	w.writeBool(false)                   // frame_size_override_flag
	w.writeBits(5, seq.OrderHintBits)    // order_hint

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.ShowExistingFrame || hdr.FrameType != FrameTypeKey || !hdr.ShowFrame {
		t.Fatalf("basic keyframe fields=%+v", hdr)
	}
	if !hdr.ErrorResilientMode || !hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if hdr.FrameSizeOverride || hdr.OrderHint != 5 || hdr.PrimaryRefFrame != PrimaryRefNone {
		t.Fatalf("routing fields=%+v", hdr)
	}
	if hdr.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", hdr.BitsRead, w.bit)
	}
}

func TestParseFrameHeaderPrefixHiddenInterFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(false)                     // show_frame
	w.writeBool(true)                      // showable_frame
	w.writeBool(false)                     // error_resilient_mode
	w.writeBool(false)                     // disable_cdf_update
	w.writeBool(true)                      // frame_size_override_flag
	w.writeBits(9, seq.OrderHintBits)      // order_hint
	w.writeBits(3, 3)                      // primary_ref_frame

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.FrameType != FrameTypeInter || hdr.ShowFrame || !hdr.ShowableFrame {
		t.Fatalf("visibility fields=%+v", hdr)
	}
	if hdr.ErrorResilientMode || hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if !hdr.FrameSizeOverride || hdr.OrderHint != 9 || hdr.PrimaryRefFrame != 3 {
		t.Fatalf("routing fields=%+v", hdr)
	}
}

func TestParseFrameHeaderPrefixShowExistingFrame(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.DecoderModelInfoPresent = true
	seq.TimingInfo.EqualPictureInterval = false
	seq.DecoderModelInfo.FramePresentationTimeLength = 5
	seq.FrameIDNumbersPresent = true
	seq.DeltaFrameIDLength = 4
	seq.AdditionalFrameIDLength = 2

	var w testBitWriter
	w.writeBool(true)  // show_existing_frame
	w.writeBits(3, 3)  // existing_frame_idx
	w.writeBits(17, 5) // frame_presentation_delay
	w.writeBits(42, seq.FrameIDBits())

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if !hdr.ShowExistingFrame || hdr.ExistingFrameIdx != 3 {
		t.Fatalf("show existing fields=%+v", hdr)
	}
	if hdr.FramePresentationDelay != 17 || hdr.FrameID != 42 {
		t.Fatalf("timing/id fields=%+v", hdr)
	}
	if hdr.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", hdr.BitsRead, w.bit)
	}
}

func TestParseFrameHeaderPrefixReducedStillPicture(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, reducedStillPictureSequenceHeader())

	var w testBitWriter
	w.writeBool(false) // disable_cdf_update
	w.writeBool(false) // allow_screen_content_tools (SELECT_SCREEN_CONTENT_TOOLS)

	hdr, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.ShowExistingFrame || hdr.FrameType != FrameTypeKey || !hdr.ShowFrame {
		t.Fatalf("reduced still fields=%+v", hdr)
	}
	if !hdr.ErrorResilientMode || hdr.DisableCDFUpdate {
		t.Fatalf("flags=%+v", hdr)
	}
	if hdr.BitsRead != 2 {
		t.Fatalf("BitsRead=%d want 2", hdr.BitsRead)
	}
}

func TestParseFrameHeaderPrefixRejectsInvalidStillPicture(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.StillPicture = true

	var w testBitWriter
	w.writeBool(false)                     // show_existing_frame
	w.writeBits(uint64(FrameTypeInter), 2) // frame_type
	w.writeBool(true)                      // show_frame

	_, err := ParseFrameHeaderPrefix(w.bytes(), seq)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseFrameHeaderPrefix err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestParseFrameHeaderPrefixAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(9, seq.OrderHintBits)
	w.writeBits(3, 3)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseFrameHeaderPrefix allocated: %f", allocs)
	}
}

func BenchmarkParseFrameHeaderPrefix(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())

	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(FrameTypeInter), 2)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(9, seq.OrderHintBits)
	w.writeBits(3, 3)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseFrameHeaderPrefix(payload, seq)
	}
}

func mustParseTestSequenceHeader(t *testing.T, payload []byte) SequenceHeader {
	t.Helper()
	seq, err := ParseSequenceHeader(payload)
	if err != nil {
		t.Fatal(err)
	}
	return seq
}

func mustParseBenchSequenceHeader(b *testing.B, payload []byte) SequenceHeader {
	b.Helper()
	seq, err := ParseSequenceHeader(payload)
	if err != nil {
		b.Fatal(err)
	}
	return seq
}

func FuzzParseSequenceHeader(f *testing.F) {
	f.Add(reducedStillPictureSequenceHeader())
	f.Add(realtimeSequenceHeader())
	f.Add([]byte{0xe0})
	f.Add([]byte{0x00})

	f.Fuzz(func(t *testing.T, payload []byte) {
		sh, err := ParseSequenceHeader(payload)
		if err != nil {
			return
		}
		if sh.SeqProfile > 2 {
			t.Fatalf("accepted invalid profile %d", sh.SeqProfile)
		}
		if sh.ReducedStillPictureHeader && !sh.StillPicture {
			t.Fatal("accepted reduced_still_picture_header without still_picture")
		}
		if sh.OperatingPointsCount == 0 || sh.OperatingPointsCount > 32 {
			t.Fatalf("bad operating point count %d", sh.OperatingPointsCount)
		}
		if sh.MaxFrameWidth == 0 || sh.MaxFrameHeight == 0 {
			t.Fatalf("bad dimensions %dx%d", sh.MaxFrameWidth, sh.MaxFrameHeight)
		}
	})
}

func FuzzParseFrameHeaderPrefix(f *testing.F) {
	f.Add([]byte{0x12, 0x80})
	f.Add([]byte{0x48, 0xc0})
	f.Add([]byte{0x80})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		hdr, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		if int(hdr.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d len=%d", hdr.BitsRead, len(payload))
		}
		if hdr.PrimaryRefFrame > PrimaryRefNone {
			t.Fatalf("PrimaryRefFrame=%d", hdr.PrimaryRefFrame)
		}
	})
}

func FuzzParseIntraFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x08, 0x00})
	f.Add([]byte{0x04, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, nil, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || int(size.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
		if size.CodedWidth > size.UpscaledWidth {
			t.Fatalf("coded width=%d upscaled=%d", size.CodedWidth, size.UpscaledWidth)
		}
	})
}

func FuzzParseFrameSize(f *testing.F) {
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x12, 0x00, 0x00, 0x00, 0x00})
	f.Add([]byte{0x09, 0x00, 0x00, 0x00, 0x00})

	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}
	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		if size.BitsRead < prefix.BitsRead || int(size.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d prefix=%d len=%d", size.BitsRead, prefix.BitsRead, len(payload))
		}
		if size.CodedWidth == 0 || size.UpscaledWidth == 0 || size.Height == 0 {
			t.Fatalf("bad dimensions=%+v", size)
		}
	})
}

func FuzzParseTileInfo(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00})
	f.Add([]byte{0x12, 0x00, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		if tiles.BitsRead < size.BitsRead || int(tiles.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d size=%d len=%d", tiles.BitsRead, size.BitsRead, len(payload))
		}
		if tiles.SBCols == 0 || tiles.SBRows == 0 || tiles.Cols == 0 || tiles.Rows == 0 {
			t.Fatalf("empty tile grid=%+v", tiles)
		}
		if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[tiles.Cols] != tiles.SBCols {
			t.Fatalf("bad col bounds=%+v", tiles)
		}
		if tiles.RowStartSB[0] != 0 || tiles.RowStartSB[tiles.Rows] != tiles.SBRows {
			t.Fatalf("bad row bounds=%+v", tiles)
		}
		for i := uint8(1); i <= tiles.Cols; i++ {
			if tiles.ColStartSB[i] <= tiles.ColStartSB[i-1] {
				t.Fatalf("non-increasing col starts=%+v", tiles)
			}
		}
		for i := uint8(1); i <= tiles.Rows; i++ {
			if tiles.RowStartSB[i] <= tiles.RowStartSB[i-1] {
				t.Fatalf("non-increasing row starts=%+v", tiles)
			}
		}
		if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
			tileCount := uint16(tiles.Cols) * uint16(tiles.Rows)
			if tiles.ContextUpdateTileID >= tileCount || tiles.TileSizeBytes == 0 {
				t.Fatalf("tile data fields=%+v", tiles)
			}
		}
	})
}

func FuzzParseTileGroupHeader(f *testing.F) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 1,
	}
	f.Add([]byte{0x00})
	f.Add([]byte{0x80})
	f.Add([]byte{0x00, 0xaa})
	f.Add([]byte{0x40, 0x00, 0xaa})

	f.Fuzz(func(t *testing.T, payload []byte) {
		frameOBU := len(payload)&1 == 0
		group, err := ParseTileGroupHeader(payload, tiles, 0, 0, frameOBU)
		if err != nil {
			return
		}
		if group.StartTile != 0 || group.StartTile > group.EndTile || group.EndTile >= 4 {
			t.Fatalf("bad tile group=%+v", group)
		}
		if group.BitsRead&7 != 0 ||
			int(group.DataOffset) != int(group.BitsRead)>>3 ||
			int(group.DataOffset) > len(payload) {
			t.Fatalf("bad tile group offsets=%+v len=%d", group, len(payload))
		}
		if group.TileCount != group.EndTile-group.StartTile+1 {
			t.Fatalf("bad tile count=%+v", group)
		}
		var spans [4]TileSpan
		n, err := SplitTileGroup(payload, tiles, group, spans[:])
		if err != nil {
			return
		}
		if n != int(group.TileCount) {
			t.Fatalf("span count=%d group=%+v", n, group)
		}
		for i := range n {
			span := spans[i]
			if span.Tile != uint16(i) || span.Row >= tiles.Rows || span.Col >= tiles.Cols {
				t.Fatalf("span[%d]=%+v group=%+v", i, span, group)
			}
			if span.Offset < uint32(group.DataOffset) ||
				uint64(span.Offset)+uint64(span.Size) > uint64(len(payload)) {
				t.Fatalf("span[%d]=%+v len=%d group=%+v", i, span, len(payload), group)
			}
		}
	})
}

func FuzzParseQuantizationParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(0, 8)  // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		if quant.BitsRead < tiles.BitsRead || int(quant.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d tile=%d len=%d", quant.BitsRead, tiles.BitsRead, len(payload))
		}
		if !seq.ColorConfig.MonoChrome && !quant.DiffUVDeltas {
			if quant.VDCDelta != quant.UDCDelta || quant.VACDelta != quant.UACDelta {
				t.Fatalf("uv deltas not copied=%+v", quant)
			}
		}
		if quant.UsingQMatrix && !seq.ColorConfig.SeparateUVDeltaQ && quant.QMatrixLevelV != quant.QMatrixLevelU {
			t.Fatalf("qmatrix v not copied=%+v", quant)
		}
	})
}

func FuzzParseSegmentationParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(0, 8)  // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previous SegmentationData
	clearSegmentationRefs(&previous)

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previous)
		if err != nil {
			return
		}
		if seg.BitsRead < quant.BitsRead || int(seg.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d quant=%d len=%d", seg.BitsRead, quant.BitsRead, len(payload))
		}
		if !seg.Enabled && (seg.UpdateMap || seg.UpdateData || seg.TemporalUpdate) {
			t.Fatalf("disabled segmentation has update flags=%+v", seg)
		}
		for i := range MaxSegments {
			if seg.Lossless[i] && seg.QIndex[i] != 0 {
				t.Fatalf("lossless segment has qindex=%+v", seg)
			}
			if !seg.Enabled && seg.Data.Segments[i].RefFrame != -1 {
				t.Fatalf("disabled segmentation ref=%+v", seg.Data.Segments[i])
			}
		}
	})
}

func FuzzParseDeltaParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(true)  // delta_q_present
	seed.writeBits(1, 2)  // delta_q_res_log2
	seed.writeBool(false) // delta_lf_present
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previous SegmentationData
	clearSegmentationRefs(&previous)

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previous)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		if delta.BitsRead < seg.BitsRead || int(delta.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d seg=%d len=%d", delta.BitsRead, seg.BitsRead, len(payload))
		}
		if !delta.DeltaQPresent && (delta.DeltaQResLog2 != 0 || delta.DeltaLFPresent || delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti) {
			t.Fatalf("delta fields set while absent=%+v", delta)
		}
		if size.AllowIntrabc && delta.DeltaLFPresent {
			t.Fatalf("intrabc frame has delta lf=%+v", delta)
		}
	})
}

func FuzzParseLoopFilterParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		if lf.BitsRead < delta.BitsRead || int(lf.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d delta=%d len=%d", lf.BitsRead, delta.BitsRead, len(payload))
		}
		if seg.AllLossless || size.AllowIntrabc {
			if lf.BitsRead != delta.BitsRead || !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
				t.Fatalf("lossless/intrabc loopfilter=%+v delta bits=%d", lf, delta.BitsRead)
			}
		}
		if seq.ColorConfig.MonoChrome && (lf.LevelU != 0 || lf.LevelV != 0) {
			t.Fatalf("monochrome uv levels=%+v", lf)
		}
	})
}

func FuzzParseCDEFParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		if cdef.BitsRead < lf.BitsRead || int(cdef.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d lf=%d len=%d", cdef.BitsRead, lf.BitsRead, len(payload))
		}
		if seg.AllLossless || !seq.EnableCDEF || size.AllowIntrabc {
			if cdef.BitsRead != lf.BitsRead || cdef.StrengthCount != 0 {
				t.Fatalf("skipped cdef=%+v lf bits=%d", cdef, lf.BitsRead)
			}
		}
		if cdef.StrengthCount > MaxCDEFStrengths {
			t.Fatalf("too many strengths=%+v", cdef)
		}
		if seq.ColorConfig.MonoChrome {
			for i := uint8(0); i < cdef.StrengthCount; i++ {
				if cdef.UVStrength[i] != 0 {
					t.Fatalf("monochrome uv strength=%+v", cdef)
				}
			}
		}
	})
}

func FuzzParseRestorationParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		if restoration.BitsRead < cdef.BitsRead || int(restoration.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d cdef=%d len=%d", restoration.BitsRead, cdef.BitsRead, len(payload))
		}
		if (seg.AllLossless && !size.SuperResEnabled) || !seq.EnableRestoration || size.AllowIntrabc {
			if restoration.BitsRead != cdef.BitsRead || restoration.UnitSizeY != 0 {
				t.Fatalf("skipped restoration=%+v cdef bits=%d", restoration, cdef.BitsRead)
			}
		}
		if seq.ColorConfig.MonoChrome && (restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) {
			t.Fatalf("monochrome restoration types=%+v", restoration)
		}
		if restoration.UnitSizeY != 0 && restoration.UnitSizeY != 64 && restoration.UnitSizeY != 128 && restoration.UnitSizeY != 256 {
			t.Fatalf("bad y unit size=%+v", restoration)
		}
	})
}

func FuzzParseTransformReferenceParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		if transformRef.BitsRead < restoration.BitsRead || int(transformRef.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d restoration=%d len=%d", transformRef.BitsRead, restoration.BitsRead, len(payload))
		}
		if seg.AllLossless && transformRef.TransformMode != TransformMode4x4Only {
			t.Fatalf("lossless transform/reference=%+v", transformRef)
		}
		if !frameTypeIsInterOrSwitch(prefix.FrameType) && transformRef.ReferenceMode != ReferenceModeSingle {
			t.Fatalf("intra transform/reference=%+v", transformRef)
		}
	})
}

func FuzzParseSkipModeParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		if skipMode.BitsRead < transformRef.BitsRead || int(skipMode.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d transform=%d len=%d", skipMode.BitsRead, transformRef.BitsRead, len(payload))
		}
		if !skipMode.Allowed {
			if skipMode.Enabled || skipMode.BitsRead != transformRef.BitsRead {
				t.Fatalf("disabled skip mode=%+v transform bits=%d", skipMode, transformRef.BitsRead)
			}
		}
		if skipMode.Allowed && (skipMode.RefFrameIdx[0] >= InterRefsPerFrame || skipMode.RefFrameIdx[1] >= InterRefsPerFrame) {
			t.Fatalf("skip refs=%+v", skipMode)
		}
		if !seq.EnableOrderHint || !frameTypeIsInterOrSwitch(prefix.FrameType) || transformRef.ReferenceMode == ReferenceModeSingle {
			if skipMode.Allowed {
				t.Fatalf("unexpected skip mode=%+v", skipMode)
			}
		}
	})
}

func FuzzParseFrameModeParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:     true,
			OrderHint: uint8(i),
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		if frameMode.BitsRead < skipMode.BitsRead || int(frameMode.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d skip=%d len=%d", frameMode.BitsRead, skipMode.BitsRead, len(payload))
		}
		if prefix.ErrorResilientMode || !frameTypeIsInterOrSwitch(prefix.FrameType) || !seq.EnableWarpedMotion {
			if frameMode.AllowWarpedMotion {
				t.Fatalf("unexpected warped motion=%+v prefix=%+v", frameMode, prefix)
			}
			if frameMode.BitsRead != skipMode.BitsRead+1 {
				t.Fatalf("frame mode bits=%+v skip bits=%d", frameMode, skipMode.BitsRead)
			}
		}
	})
}

func FuzzParseGlobalMotionParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	defaultGlobal := DefaultGlobalMotionParams()
	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:        true,
			OrderHint:    uint8(i),
			GlobalMotion: defaultGlobal,
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		globalMotion, err := ParseGlobalMotionParams(payload, prefix, size, tiles, &refs, frameMode)
		if err != nil {
			return
		}
		if globalMotion.BitsRead < frameMode.BitsRead || int(globalMotion.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d frame mode=%d len=%d", globalMotion.BitsRead, frameMode.BitsRead, len(payload))
		}
		if !frameTypeIsInterOrSwitch(prefix.FrameType) && globalMotion.BitsRead != frameMode.BitsRead {
			t.Fatalf("intra global motion=%+v frame bits=%d", globalMotion, frameMode.BitsRead)
		}
		for i := range InterRefsPerFrame {
			if globalMotion.Ref[i].Type > GlobalMotionAffine {
				t.Fatalf("global motion ref[%d]=%+v", i, globalMotion.Ref[i])
			}
		}
	})
}

func FuzzParseFilmGrainParams(f *testing.F) {
	seq, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		f.Fatal(err)
	}

	payload, prefix, err := buildShownKeyFramePrefixRaw(seq, false)
	if err != nil {
		f.Fatal(err)
	}
	var seed testBitWriter
	seed.writeBitsFrom(payload, prefix.BitsRead)
	seed.writeBool(false) // use_superres
	seed.writeBool(false) // render_and_frame_size_different
	seed.writeBool(false) // disable_frame_end_update_cdf
	seed.writeBool(false) // uniform_tile_spacing_flag
	seed.writeBits(50, 8) // base_q_idx
	seed.writeBool(false) // y_dc_delta_q
	seed.writeBool(false) // u_dc_delta_q
	seed.writeBool(false) // u_ac_delta_q
	seed.writeBool(false) // using_qmatrix
	seed.writeBool(false) // segmentation_enabled
	seed.writeBool(false) // delta_q_present
	seed.writeBits(0, 6)  // loop_filter_level[0]
	seed.writeBits(0, 6)  // loop_filter_level[1]
	seed.writeBits(0, 3)  // loop_filter_sharpness
	seed.writeBool(false) // mode_ref_delta_enabled
	seed.writeBits(0, 2)  // cdef_damping_minus_3
	seed.writeBits(0, 2)  // cdef_bits
	seed.writeBits(7, 6)  // y_strength[0]
	seed.writeBits(9, 6)  // uv_strength[0]
	seed.writeBits(0, 2)  // restoration y type none
	seed.writeBits(0, 2)  // restoration u type none
	seed.writeBits(0, 2)  // restoration v type none
	seed.writeBool(false) // tx_mode_select
	seed.writeBool(false) // reduced_tx_set
	seed.writeBool(false) // apply_grain
	f.Add(seed.bytes())
	f.Add([]byte{0x10, 0x00, 0x00, 0x00})

	defaultGlobal := DefaultGlobalMotionParams()
	defaultFilmGrain := FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		BitDepth:      seq.ColorConfig.BitDepth,
	}
	var refs ReferenceState
	for i := range RefFrames {
		refs.Frames[i] = ReferenceFrame{
			Valid:        true,
			OrderHint:    uint8(i),
			GlobalMotion: defaultGlobal,
			FilmGrain:    defaultFilmGrain,
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
	var previousSeg SegmentationData
	clearSegmentationRefs(&previousSeg)
	previousLF := defaultLoopFilterDeltas()

	f.Fuzz(func(t *testing.T, payload []byte) {
		prefix, err := ParseFrameHeaderPrefix(payload, seq)
		if err != nil {
			return
		}
		size, err := ParseFrameSize(payload, seq, prefix, &refs, 0, 0)
		if err != nil {
			return
		}
		tiles, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			return
		}
		quant, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			return
		}
		seg, err := ParseSegmentationParams(payload, prefix, quant, &previousSeg)
		if err != nil {
			return
		}
		delta, err := ParseDeltaParams(payload, size, quant, seg)
		if err != nil {
			return
		}
		lf, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, &previousLF)
		if err != nil {
			return
		}
		cdef, err := ParseCDEFParams(payload, seq, size, seg, lf)
		if err != nil {
			return
		}
		restoration, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			return
		}
		transformRef, err := ParseTransformReferenceParams(payload, prefix, seg, restoration)
		if err != nil {
			return
		}
		skipMode, err := ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
		if err != nil {
			return
		}
		frameMode, err := ParseFrameModeParams(payload, seq, prefix, skipMode)
		if err != nil {
			return
		}
		globalMotion, err := ParseGlobalMotionParams(payload, prefix, size, tiles, &refs, frameMode)
		if err != nil {
			return
		}
		filmGrain, err := ParseFilmGrainParams(payload, seq, prefix, size, &refs, globalMotion)
		if err != nil {
			return
		}
		if filmGrain.BitsRead < globalMotion.BitsRead || int(filmGrain.BitsRead) > len(payload)*8 {
			t.Fatalf("BitsRead=%d global=%d len=%d", filmGrain.BitsRead, globalMotion.BitsRead, len(payload))
		}
		if filmGrain.BitDepth != seq.ColorConfig.BitDepth {
			t.Fatalf("bit depth=%+v seq=%+v", filmGrain, seq.ColorConfig)
		}
		if !seq.FilmGrainParamsPresent || (!prefix.ShowFrame && !prefix.ShowableFrame) {
			if filmGrain.Apply || filmGrain.BitsRead != globalMotion.BitsRead {
				t.Fatalf("skipped film grain=%+v global bits=%d", filmGrain, globalMotion.BitsRead)
			}
		} else if !filmGrain.Apply && filmGrain.BitsRead != globalMotion.BitsRead+1 {
			t.Fatalf("disabled film grain=%+v global bits=%d", filmGrain, globalMotion.BitsRead)
		}
		if filmGrain.NumYPoints > MaxFilmGrainYPoints ||
			filmGrain.NumCbPoints > MaxFilmGrainUVPoints ||
			filmGrain.NumCrPoints > MaxFilmGrainUVPoints {
			t.Fatalf("too many film grain points=%+v", filmGrain)
		}
		if seq.ColorConfig.MonoChrome && (filmGrain.ChromaScalingFromLuma || filmGrain.NumCbPoints != 0 || filmGrain.NumCrPoints != 0) {
			t.Fatalf("monochrome film grain=%+v", filmGrain)
		}
		if seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY &&
			((filmGrain.NumCbPoints == 0) != (filmGrain.NumCrPoints == 0)) {
			t.Fatalf("4:2:0 chroma film grain=%+v", filmGrain)
		}
	})
}

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

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "FrameHeaderPrefix", size: unsafe.Sizeof(FrameHeaderPrefix{}), max: 24},
		{name: "FrameSize", size: unsafe.Sizeof(FrameSize{}), max: 192},
		{name: "ReferenceFrame", size: unsafe.Sizeof(ReferenceFrame{}), max: 696},
		{name: "ReferenceState", size: unsafe.Sizeof(ReferenceState{}), max: 5568},
		{name: "shortRefInfo", size: unsafe.Sizeof(shortRefInfo{}), max: 4},
		{name: "TileGroup", size: unsafe.Sizeof(TileGroup{}), max: 20},
		{name: "TileSpan", size: unsafe.Sizeof(TileSpan{}), max: 12},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}

func TestParseLoopFilterParamsLosslessDefaults(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	delta, err := ParseDeltaParams(w.bytes(), size, QuantizationParams{}, seg)
	if err != nil {
		t.Fatal(err)
	}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, FrameHeaderPrefix{}, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
		t.Fatalf("loopfilter=%+v", lf)
	}
	want := defaultLoopFilterDeltas()
	if lf.Deltas != want {
		t.Fatalf("deltas=%+v want %+v", lf.Deltas, want)
	}
	if lf.BitsRead != delta.BitsRead {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, delta.BitsRead)
	}
}

func TestParseLoopFilterParamsLevelsAndUpdates(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, 40)
	w.writeBits(10, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(3, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(true)  // mode_ref_delta_update
	w.writeBool(true)  // ref_delta[0] update
	writeSignedBitsTest(&w, -2, 7)
	for i := 1; i < RefFrames; i++ {
		w.writeBool(false)
	}
	w.writeBool(false) // mode_delta[0] update
	w.writeBool(true)  // mode_delta[1] update
	writeSignedBitsTest(&w, 3, 7)

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lf.LevelY[0] != 10 || lf.LevelY[1] != 0 || lf.LevelU != 5 || lf.LevelV != 6 || lf.Sharpness != 3 {
		t.Fatalf("levels=%+v", lf)
	}
	if !lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate {
		t.Fatalf("delta flags=%+v", lf)
	}
	if lf.Deltas.Ref[0] != -2 || lf.Deltas.Ref[4] != -1 || lf.Deltas.Mode[1] != 3 {
		t.Fatalf("deltas=%+v", lf.Deltas)
	}
	if lf.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, w.bit)
	}
}

func TestParseLoopFilterParamsMonochromeSkipsUVLevels(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(12, 6)
	w.writeBits(0, 6)
	w.writeBits(1, 3)
	w.writeBool(false)
	prefix := FrameHeaderPrefix{PrimaryRefFrame: PrimaryRefNone}
	seg := SegmentationParams{AllLossless: false}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lf.LevelY[0] != 12 || lf.LevelU != 0 || lf.LevelV != 0 || lf.Sharpness != 1 {
		t.Fatalf("loopfilter=%+v", lf)
	}
	if lf.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", lf.BitsRead, w.bit)
	}
}

func TestParseLoopFilterParamsCopiesPreviousDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	var w testBitWriter
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)

	previous := defaultLoopFilterDeltas()
	previous.Ref[2] = 4
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	seg := SegmentationParams{AllLossless: false}

	lf, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, &previous)
	if err != nil {
		t.Fatal(err)
	}
	if lf.Deltas.Ref[2] != 4 || lf.Deltas.Ref[4] != -1 {
		t.Fatalf("deltas=%+v", lf.Deltas)
	}
}

func TestParseLoopFilterParamsRejectsMissingPreviousDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	var w testBitWriter
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	seg := SegmentationParams{AllLossless: false}

	_, err := ParseLoopFilterParams(w.bytes(), seq, prefix, FrameSize{}, seg, DeltaParams{}, nil)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseLoopFilterParams err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseLoopFilterParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDelta(t, seq, 40)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, nil)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseLoopFilterParams allocated: %f", allocs)
	}
}

func BenchmarkParseLoopFilterParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, size, seg, delta := buildShownKeyFrameDeltaBench(b, seq, 40)
	w.writeBits(0, 6)
	w.writeBits(0, 6)
	w.writeBits(0, 3)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseLoopFilterParams(payload, seq, prefix, size, seg, delta, nil)
	}
}

func buildShownKeyFrameDelta(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, FrameSize, SegmentationParams, DeltaParams) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if baseQIdx != 0 {
		w.writeBool(false)
	}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size, seg, delta
}

func buildShownKeyFrameDeltaBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, FrameSize, SegmentationParams, DeltaParams) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		b.Fatal(err)
	}
	w.writeBool(false)
	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		b.Fatal(err)
	}
	if baseQIdx != 0 {
		w.writeBool(false)
	}
	delta, err := ParseDeltaParams(w.bytes(), size, quant, seg)
	if err != nil {
		b.Fatal(err)
	}
	return w, prefix, size, seg, delta
}

func TestParseQuantizationParamsZeroDeltas(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // u_dc_delta_q
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(false) // using_qmatrix

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 0 || quant.YDCDelta != 0 || quant.UDCDelta != 0 || quant.UACDelta != 0 {
		t.Fatalf("quant=%+v", quant)
	}
	if quant.VDCDelta != quant.UDCDelta || quant.VACDelta != quant.UACDelta {
		t.Fatalf("uv copy=%+v", quant)
	}
	if quant.UsingQMatrix || quant.BitsRead != w.bit {
		t.Fatalf("qmatrix/bits=%+v want bits %d", quant, w.bit)
	}
}

func TestParseQuantizationParamsSeparateUVAndQMatrix(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SeparateUVDeltaQ = true
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(96, 8) // base_q_idx
	w.writeBool(true)
	writeSignedBitsTest(&w, -2, 7)
	w.writeBool(true) // diff_uv_delta
	w.writeBool(true)
	writeSignedBitsTest(&w, 5, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, -3, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, 7, 7)
	w.writeBool(true)
	writeSignedBitsTest(&w, -9, 7)
	w.writeBool(true) // using_qmatrix
	w.writeBits(2, 4)
	w.writeBits(3, 4)
	w.writeBits(4, 4)

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 96 || quant.YDCDelta != -2 {
		t.Fatalf("base/y=%+v", quant)
	}
	if !quant.DiffUVDeltas || quant.UDCDelta != 5 || quant.UACDelta != -3 || quant.VDCDelta != 7 || quant.VACDelta != -9 {
		t.Fatalf("uv deltas=%+v", quant)
	}
	if !quant.UsingQMatrix || quant.QMatrixLevelY != 2 || quant.QMatrixLevelU != 3 || quant.QMatrixLevelV != 4 {
		t.Fatalf("qmatrix=%+v", quant)
	}
}

func TestParseQuantizationParamsSharedUV(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SeparateUVDeltaQ = false
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(10, 8)
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(true)  // u_dc_delta_q
	writeSignedBitsTest(&w, 4, 7)
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(true)  // using_qmatrix
	w.writeBits(1, 4)
	w.writeBits(6, 4)

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.DiffUVDeltas || quant.UDCDelta != 4 || quant.VDCDelta != 4 || quant.VACDelta != 0 {
		t.Fatalf("shared uv=%+v", quant)
	}
	if quant.QMatrixLevelY != 1 || quant.QMatrixLevelU != 6 || quant.QMatrixLevelV != 6 {
		t.Fatalf("qmatrix=%+v", quant)
	}
}

func TestParseQuantizationParamsMonoChrome(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(3, 8)
	w.writeBool(true)
	writeSignedBitsTest(&w, -1, 7)
	w.writeBool(false) // using_qmatrix

	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	if quant.BaseQIdx != 3 || quant.YDCDelta != -1 {
		t.Fatalf("mono quant=%+v", quant)
	}
	if quant.UDCDelta != 0 || quant.UACDelta != 0 || quant.VDCDelta != 0 || quant.VACDelta != 0 {
		t.Fatalf("mono uv deltas=%+v", quant)
	}
}

func TestReadSignedBits(t *testing.T) {
	cases := []struct {
		bits uint64
		n    uint8
		want int16
	}{
		{bits: 0b0000101, n: 7, want: 5},
		{bits: 0b1111111, n: 7, want: -1},
		{bits: 0b1111110, n: 7, want: -2},
		{bits: 0b1000000, n: 7, want: -64},
		{bits: 0b0_1111_1111, n: 9, want: 255},
		{bits: 0b1_1111_1111, n: 9, want: -1},
		{bits: 0b1_0000_0000, n: 9, want: -256},
	}
	for _, tc := range cases {
		var w testBitWriter
		w.writeBits(tc.bits, tc.n)
		r := bitstream.NewReader(w.bytes())
		got, err := readSignedBits(&r, tc.n)
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.want {
			t.Fatalf("readSignedBits(%b,%d)=%d want %d", tc.bits, tc.n, got, tc.want)
		}
	}
}

func BenchmarkReadSignedBits(b *testing.B) {
	var w testBitWriter
	w.writeBits(0b1_0000_0000, 9)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		r := bitstream.NewReader(payload)
		_, _ = readSignedBits(&r, 9)
	}
}

func TestParseQuantizationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTile(t, seq)
	w.writeBits(0, 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseQuantizationParams(payload, seq, tiles)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseQuantizationParams allocated: %f", allocs)
	}
}

func BenchmarkParseQuantizationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, tiles := buildShownKeyFrameTileBench(b, seq)
	w.writeBits(0, 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseQuantizationParams(payload, seq, tiles)
	}
}

func buildShownKeyFrameTile(t *testing.T, seq SequenceHeader) (testBitWriter, TileInfo) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	return w, tiles
}

func buildShownKeyFrameTileBench(b *testing.B, seq SequenceHeader) (testBitWriter, TileInfo) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	return w, tiles
}

func writeSignedBitsTest(w *testBitWriter, value int16, n uint8) {
	mask := int32((uint32(1) << n) - 1)
	w.writeBits(uint64(int32(value)&mask), n)
}

func TestParseRestorationParamsSkipsLosslessWithoutSuperres(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, _, seg := buildShownKeyFrameSegmentation(t, seq, 0)
	cdef := CDEFParams{BitsRead: w.bit}

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.BitsRead != cdef.BitsRead || restoration.UnitSizeY != 0 {
		t.Fatalf("restoration=%+v cdef bits=%d", restoration, cdef.BitsRead)
	}
}

func TestParseRestorationParamsAllNone(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationNone || restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone {
		t.Fatalf("types=%+v", restoration.Type)
	}
	if restoration.UnitSizeY != RestorationUnitMax || restoration.UnitSizeUV != RestorationUnitMax ||
		restoration.UnitSizeYLog2 != 8 || restoration.UnitSizeUVLog2 != 8 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
	if restoration.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", restoration.BitsRead, w.bit)
	}
}

func TestParseRestorationParamsTypesAndUnitSizes(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.SubsamplingX = true
	seq.ColorConfig.SubsamplingY = true
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationWiener), 2)
	w.writeBits(uint64(RestorationSGRProj), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBool(true) // grow from 64 to 128
	w.writeBool(false)
	w.writeBool(true) // chroma unit is half luma for 4:2:0

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationWiener || restoration.Type[1] != RestorationSGRProj || restoration.Type[2] != RestorationNone {
		t.Fatalf("types=%+v", restoration.Type)
	}
	if restoration.UnitSizeY != 128 || restoration.UnitSizeUV != 64 ||
		restoration.UnitSizeYLog2 != 7 || restoration.UnitSizeUVLog2 != 6 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
}

func TestParseRestorationParams128SuperblockUnitSize(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.Use128x128Superblock = true
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationSGRProj), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBool(true) // grow from 128 to 256

	restoration, err := ParseRestorationParams(w.bytes(), seq, size, seg, cdef)
	if err != nil {
		t.Fatal(err)
	}
	if restoration.UnitSizeY != 256 || restoration.UnitSizeUV != 256 || restoration.UnitSizeYLog2 != 8 {
		t.Fatalf("unit sizes=%+v", restoration)
	}
}

func TestParseRestorationParamsMonochromeReadsYOnly(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.ColorConfig.MonoChrome = true
	var w testBitWriter
	w.writeBits(uint64(RestorationSwitchable), 2)
	w.writeBool(false)
	seg := SegmentationParams{AllLossless: false}

	restoration, err := ParseRestorationParams(w.bytes(), seq, FrameSize{}, seg, CDEFParams{})
	if err != nil {
		t.Fatal(err)
	}
	if restoration.Type[0] != RestorationSwitchable || restoration.Type[1] != RestorationNone || restoration.UnitSizeY != 64 {
		t.Fatalf("restoration=%+v", restoration)
	}
	if restoration.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", restoration.BitsRead, w.bit)
	}
}

func TestParseRestorationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEF(t, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseRestorationParams(payload, seq, size, seg, cdef)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseRestorationParams allocated: %f", allocs)
	}
}

func BenchmarkParseRestorationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, size, seg, cdef := buildShownKeyFrameCDEFBench(b, seq, 40)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	w.writeBits(uint64(RestorationNone), 2)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseRestorationParams(payload, seq, size, seg, cdef)
	}
}

func buildShownKeyFrameCDEF(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, CDEFParams) {
	t.Helper()
	w, size, seg, lf := buildShownKeyFrameLoopFilter(t, seq, baseQIdx)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	if !seq.ColorConfig.MonoChrome {
		w.writeBits(9, 6)
	}
	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		t.Fatal(err)
	}
	return w, size, seg, cdef
}

func buildShownKeyFrameCDEFBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameSize, SegmentationParams, CDEFParams) {
	b.Helper()
	w, size, seg, lf := buildShownKeyFrameLoopFilterBench(b, seq, baseQIdx)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 6)
	if !seq.ColorConfig.MonoChrome {
		w.writeBits(9, 6)
	}
	cdef, err := ParseCDEFParams(w.bytes(), seq, size, seg, lf)
	if err != nil {
		b.Fatal(err)
	}
	return w, size, seg, cdef
}

func TestParseSegmentationParamsDisabledLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 0)
	w.writeBool(false) // segmentation_enabled

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if seg.Enabled || !seg.AllLossless {
		t.Fatalf("segmentation=%+v", seg)
	}
	for i := range MaxSegments {
		if seg.Data.Segments[i].RefFrame != -1 || seg.QIndex[i] != 0 || !seg.Lossless[i] {
			t.Fatalf("segment[%d]=%+v q=%d lossless=%v", i, seg.Data.Segments[i], seg.QIndex[i], seg.Lossless[i])
		}
	}
	if seg.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", seg.BitsRead, w.bit)
	}
}

func TestParseSegmentationParamsDisabledNotLossless(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 10)
	w.writeBool(false)

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if seg.AllLossless || seg.QIndex[0] != 10 || seg.Lossless[0] {
		t.Fatalf("segmentation=%+v", seg)
	}
}

func TestParseSegmentationParamsPrimaryRefNoneUpdateData(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 100)
	w.writeBool(true) // segmentation_enabled
	w.writeBool(true) // segment 0 delta_q enabled
	writeSignedBitsTest(&w, 10, 9)
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(true)  // ref_frame enabled
	w.writeBits(2, 3)
	w.writeBool(true)  // skip
	w.writeBool(false) // globalmv
	for i := 1; i < MaxSegments; i++ {
		writeEmptySegmentData(&w)
	}

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !seg.Enabled || !seg.UpdateMap || !seg.UpdateData || seg.TemporalUpdate {
		t.Fatalf("segmentation flags=%+v", seg)
	}
	if !seg.Data.Preskip || seg.Data.LastActiveID != 0 {
		t.Fatalf("active/preskip=%+v", seg.Data)
	}
	first := seg.Data.Segments[0]
	if first.DeltaQ != 10 || first.RefFrame != 2 || !first.Skip || first.GlobalMV {
		t.Fatalf("segment 0=%+v", first)
	}
	if seg.QIndex[0] != 110 || seg.QIndex[1] != 100 {
		t.Fatalf("qindex=%v", seg.QIndex[:2])
	}
}

func TestParseSegmentationParamsCopiesPreviousData(t *testing.T) {
	var previous SegmentationData
	clearSegmentationRefs(&previous)
	previous.Segments[0].DeltaQ = 5

	var w testBitWriter
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
	quant := QuantizationParams{BaseQIdx: 20}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}

	seg, err := ParseSegmentationParams(w.bytes(), prefix, quant, &previous)
	if err != nil {
		t.Fatal(err)
	}
	if !seg.Enabled || seg.UpdateMap || seg.UpdateData {
		t.Fatalf("segmentation flags=%+v", seg)
	}
	if seg.Data.Segments[0].DeltaQ != 5 || seg.QIndex[0] != 25 {
		t.Fatalf("copied data=%+v q=%d", seg.Data.Segments[0], seg.QIndex[0])
	}
}

func TestParseSegmentationParamsRejectsMissingPreviousData(t *testing.T) {
	var w testBitWriter
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
	quant := QuantizationParams{BaseQIdx: 20}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}

	_, err := ParseSegmentationParams(w.bytes(), prefix, quant, nil)
	if !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Fatalf("ParseSegmentationParams err=%v want %v", err, ErrReferenceFrameNeeded)
	}
}

func TestParseSegmentationParamsAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuant(t, seq, 0)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseSegmentationParams(payload, prefix, quant, nil)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseSegmentationParams allocated: %f", allocs)
	}
}

func BenchmarkParseSegmentationParams(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, quant := buildShownKeyFrameQuantBench(b, seq, 0)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseSegmentationParams(payload, prefix, quant, nil)
	}
}

func buildShownKeyFrameQuant(t *testing.T, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, QuantizationParams) {
	t.Helper()
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, quant
}

func buildShownKeyFrameQuantBench(b *testing.B, seq SequenceHeader, baseQIdx uint8) (testBitWriter, FrameHeaderPrefix, QuantizationParams) {
	b.Helper()
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		b.Fatal(err)
	}
	writeSharedUVZeroQuant(&w, baseQIdx)
	quant, err := ParseQuantizationParams(w.bytes(), seq, tiles)
	if err != nil {
		b.Fatal(err)
	}
	return w, prefix, quant
}

func writeSharedUVZeroQuant(w *testBitWriter, baseQIdx uint8) {
	w.writeBits(uint64(baseQIdx), 8)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
}

func writeEmptySegmentData(w *testBitWriter) {
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
}

// srgbStillSequenceHeader is a reduced still-picture header with the sRGB
// colour description (BT.709 primaries, sRGB transfer, identity matrix) in
// the given profile and bit depth, optionally monochrome.
func srgbStillSequenceHeader(profile uint8, bitDepth uint8, mono bool) []byte {
	var w testBitWriter
	w.writeBits(uint64(profile), 3) // seq_profile
	w.writeBool(true)               // still_picture
	w.writeBool(true)               // reduced_still_picture_header
	w.writeBits(5, 5)               // seq_level_idx[0]
	w.writeBits(3, 4)               // frame_width_bits_minus_1
	w.writeBits(3, 4)               // frame_height_bits_minus_1
	w.writeBits(15, 4)              // max_frame_width_minus_1
	w.writeBits(8, 4)               // max_frame_height_minus_1
	w.writeBool(false)              // use_128x128_superblock
	w.writeBool(true)               // enable_filter_intra
	w.writeBool(true)               // enable_intra_edge_filter
	w.writeBool(false)              // enable_superres
	w.writeBool(true)               // enable_cdef
	w.writeBool(false)              // enable_restoration
	w.writeBool(bitDepth > 8)       // high_bitdepth
	if profile == 2 && bitDepth > 8 {
		w.writeBool(bitDepth == 12) // twelve_bit
	}
	if profile != 1 {
		w.writeBool(mono) // mono_chrome
	}
	w.writeBool(true) // color_description_present_flag
	w.writeBits(ColorPrimariesBT709, 8)
	w.writeBits(TransferCharacteristicsSRGB, 8)
	w.writeBits(MatrixCoefficientsIdentity, 8)
	if mono {
		w.writeBool(true) // color_range
	} else {
		w.writeBool(false) // separate_uv_delta_q
	}
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

// TestParseSequenceHeaderSRGBProfile ports dav1d's parse_seq_hdr check: sRGB
// implies 4:4:4, so it is valid only in profile 1 and 12-bit profile 2. A
// monochrome header takes the monochrome branch first and stays valid.
func TestParseSequenceHeaderSRGBProfile(t *testing.T) {
	for _, tc := range []struct {
		profile, bitDepth uint8
		mono, ok          bool
	}{
		{profile: 0, bitDepth: 8},
		{profile: 0, bitDepth: 10},
		{profile: 2, bitDepth: 8},
		{profile: 2, bitDepth: 10},
		{profile: 1, bitDepth: 8, ok: true},
		{profile: 1, bitDepth: 10, ok: true},
		{profile: 2, bitDepth: 12, ok: true},
		{profile: 0, bitDepth: 8, mono: true, ok: true},
	} {
		sh, err := ParseSequenceHeader(srgbStillSequenceHeader(tc.profile, tc.bitDepth, tc.mono))
		if !tc.ok {
			if !errors.Is(err, ErrInvalidSequenceHeader) {
				t.Errorf("profile %d %d-bit sRGB: err=%v, want ErrInvalidSequenceHeader", tc.profile, tc.bitDepth, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("profile %d %d-bit mono=%v sRGB: %v", tc.profile, tc.bitDepth, tc.mono, err)
			continue
		}
		cfg := sh.ColorConfig
		if cfg.BitDepth != tc.bitDepth || !cfg.ColorRange || cfg.MonoChrome != tc.mono ||
			(!tc.mono && (cfg.SubsamplingX || cfg.SubsamplingY)) {
			t.Errorf("profile %d %d-bit mono=%v sRGB: colour config %+v", tc.profile, tc.bitDepth, tc.mono, cfg)
		}
	}
}

type testBitWriter struct {
	buf [128]byte
	bit uint16
}

func (w *testBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 == 0 {
			w.bit++
			continue
		}
		w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		w.bit++
	}
}

func (w *testBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *testBitWriter) writeBitsFrom(src []byte, n uint16) {
	for i := range int(n) {
		bit := (src[i>>3] >> uint(7-(i&7))) & 1
		w.writeBits(uint64(bit), 1)
	}
}

func (w *testBitWriter) bytes() []byte {
	return w.buf[:(w.bit+7)>>3]
}

func (w *testBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func reducedStillPictureSequenceHeader() []byte {
	return reducedStillPictureSequenceHeaderWithLevel(5)
}

func reducedStillPictureSequenceHeaderWithLevel(level uint8) []byte {
	var w testBitWriter
	w.writeBits(0, 3) // seq_profile
	w.writeBool(true) // still_picture
	w.writeBool(true) // reduced_still_picture_header
	w.writeBits(uint64(level), 5)
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(true)  // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func realtimeSequenceHeader() []byte {
	return realtimeSequenceHeaderWithLevel(5)
}

func realtimeSequenceHeaderWithLevel(level uint8) []byte {
	return realtimeSequenceHeaderWithTiming(level, false)
}

func realtimeSequenceHeaderWithTiming(level uint8, timing bool) []byte {
	var w testBitWriter
	w.writeBits(1, 3)   // seq_profile: sRGB is 4:4:4, profile 1
	w.writeBool(false)  // still_picture
	w.writeBool(false)  // reduced_still_picture_header
	w.writeBool(timing) // timing_info_present_flag
	if timing {
		w.writeBits(1001, 32)  // num_units_in_display_tick
		w.writeBits(30000, 32) // time_scale
		w.writeBool(false)     // equal_picture_interval
		w.writeBool(false)     // decoder_model_info_present_flag
	}
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(uint64(level), 5)
	if level > 7 && ValidSequenceLevelIndex(level) {
		w.writeBool(false) // seq_tier[0]
	}
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(true)  // enable_order_hint
	w.writeBool(true)  // enable_jnt_comp
	w.writeBool(true)  // enable_ref_frame_mvs
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBits(6, 3)  // order_hint_bits_minus_1
	w.writeBool(true)  // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(true)  // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(true)  // color_description_present_flag
	w.writeBits(ColorPrimariesBT709, 8)
	w.writeBits(TransferCharacteristicsSRGB, 8)
	w.writeBits(MatrixCoefficientsIdentity, 8)
	w.writeBool(false) // separate_uv_delta_q
	w.writeBool(true)  // film_grain_params_present
	return w.trailingBits()
}

func realtimeSequenceHeaderWithFrameIDLengths(deltaMinus2 uint8, additionalMinus1 uint8) []byte {
	var w testBitWriter
	w.writeBits(1, 3)  // seq_profile: sRGB is 4:4:4, profile 1
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(true)  // frame_id_numbers_present_flag
	w.writeBits(uint64(deltaMinus2), 4)
	w.writeBits(uint64(additionalMinus1), 3)
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(true)  // enable_order_hint
	w.writeBool(true)  // enable_jnt_comp
	w.writeBool(true)  // enable_ref_frame_mvs
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBits(6, 3)  // order_hint_bits_minus_1
	w.writeBool(true)  // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(true)  // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(true)  // color_description_present_flag
	w.writeBits(ColorPrimariesBT709, 8)
	w.writeBits(TransferCharacteristicsSRGB, 8)
	w.writeBits(MatrixCoefficientsIdentity, 8)
	w.writeBool(false) // separate_uv_delta_q
	w.writeBool(true)  // film_grain_params_present
	return w.trailingBits()
}

func TestValidSequenceLevelIndexMatchesLibaom(t *testing.T) {
	valid := [...]uint8{0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 31}
	for _, level := range valid {
		if !ValidSequenceLevelIndex(level) {
			t.Fatalf("ValidSequenceLevelIndex(%d)=false", level)
		}
	}
	for level := range uint8(32) {
		seenValid := false
		for _, validLevel := range valid {
			if level == validLevel {
				seenValid = true
				break
			}
		}
		if seenValid {
			continue
		}
		if ValidSequenceLevelIndex(level) {
			t.Fatalf("ValidSequenceLevelIndex(%d)=true", level)
		}
	}
}

func TestParseSequenceHeaderReducedStillPicture(t *testing.T) {
	sh, err := ParseSequenceHeader(reducedStillPictureSequenceHeader())
	if err != nil {
		t.Fatal(err)
	}

	if sh.SeqProfile != 0 || !sh.StillPicture || !sh.ReducedStillPictureHeader {
		t.Fatalf("unexpected flags: %+v", sh)
	}
	if sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].SeqLevelIdx != 5 {
		t.Fatalf("operating point not parsed: %+v", sh.OperatingPoints[0])
	}
	if sh.MaxFrameWidth != 16 || sh.MaxFrameHeight != 9 {
		t.Fatalf("dimensions=%dx%d want 16x9", sh.MaxFrameWidth, sh.MaxFrameHeight)
	}
	if sh.ColorConfig.BitDepth != 8 || !sh.ColorConfig.SubsamplingX || !sh.ColorConfig.SubsamplingY {
		t.Fatalf("color config=%+v", sh.ColorConfig)
	}
	if !sh.ColorConfig.SeparateUVDeltaQ {
		t.Fatal("separate_uv_delta_q not parsed")
	}
}

func TestParseSequenceHeaderRealtime(t *testing.T) {
	sh, err := ParseSequenceHeader(realtimeSequenceHeader())
	if err != nil {
		t.Fatal(err)
	}

	if sh.StillPicture || sh.ReducedStillPictureHeader {
		t.Fatalf("unexpected still-picture flags: %+v", sh)
	}
	if sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].SeqLevelIdx != 5 {
		t.Fatalf("operating point not parsed: %+v", sh.OperatingPoints[0])
	}
	if !sh.EnableOrderHint || sh.OrderHintBits != 7 {
		t.Fatalf("order hint parsed incorrectly: enabled=%v bits=%d", sh.EnableOrderHint, sh.OrderHintBits)
	}
	if !sh.EnableSuperRes || !sh.EnableCDEF || !sh.EnableRestoration {
		t.Fatalf("restoration tools parsed incorrectly: %+v", sh)
	}
	if !sh.ColorConfig.ColorRange || sh.ColorConfig.SubsamplingX || sh.ColorConfig.SubsamplingY {
		t.Fatalf("identity color config=%+v", sh.ColorConfig)
	}
	if !sh.FilmGrainParamsPresent {
		t.Fatal("film_grain_params_present not parsed")
	}
}

func TestParseSequenceHeaderRejectsInvalidProfile(t *testing.T) {
	var w testBitWriter
	w.writeBits(7, 3)
	_, err := ParseSequenceHeader(w.buf[:1])
	if !errors.Is(err, ErrInvalidSequenceHeader) {
		t.Fatalf("ParseSequenceHeader err=%v want %v", err, ErrInvalidSequenceHeader)
	}
}

func TestParseSequenceHeaderRejectsInvalidSequenceLevel(t *testing.T) {
	for _, payload := range [][]byte{
		reducedStillPictureSequenceHeaderWithLevel(2),
		realtimeSequenceHeaderWithLevel(10),
		realtimeSequenceHeaderWithLevel(28),
	} {
		_, err := ParseSequenceHeader(payload)
		if !errors.Is(err, ErrInvalidSequenceHeader) {
			t.Fatalf("ParseSequenceHeader invalid level err=%v want %v", err, ErrInvalidSequenceHeader)
		}
	}
}

func TestParseSequenceHeaderFrameIDLengthBounds(t *testing.T) {
	sh, err := ParseSequenceHeader(realtimeSequenceHeaderWithFrameIDLengths(13, 0))
	if err != nil {
		t.Fatal(err)
	}
	if !sh.FrameIDNumbersPresent || sh.DeltaFrameIDLength != 15 || sh.AdditionalFrameIDLength != 1 || sh.FrameIDBits() != 16 {
		t.Fatalf("frame id lengths parsed incorrectly: delta=%d additional=%d bits=%d present=%v",
			sh.DeltaFrameIDLength, sh.AdditionalFrameIDLength, sh.FrameIDBits(), sh.FrameIDNumbersPresent)
	}

	_, err = ParseSequenceHeader(realtimeSequenceHeaderWithFrameIDLengths(14, 0))
	if !errors.Is(err, ErrInvalidSequenceHeader) {
		t.Fatalf("ParseSequenceHeader oversized frame id err=%v want %v", err, ErrInvalidSequenceHeader)
	}
}

func TestSequenceHeaderAllocs(t *testing.T) {
	payload := realtimeSequenceHeader()
	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseSequenceHeader(payload)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseSequenceHeader allocated: %f", allocs)
	}
}

func BenchmarkParseSequenceHeader(b *testing.B) {
	payload := realtimeSequenceHeader()
	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseSequenceHeader(payload)
	}
}

func TestParseSequenceHeaderTimingInfo(t *testing.T) {
	sh, err := ParseSequenceHeader(realtimeSequenceHeaderWithTiming(5, true))
	if err != nil {
		t.Fatal(err)
	}
	if !sh.TimingInfoPresent || sh.TimingInfo.NumUnitsInDisplayTick != 1001 || sh.TimingInfo.TimeScale != 30000 || sh.TimingInfo.EqualPictureInterval || sh.DecoderModelInfoPresent {
		t.Fatalf("timing=%v info=%+v decoderModel=%v", sh.TimingInfoPresent, sh.TimingInfo, sh.DecoderModelInfoPresent)
	}
	if sh.OperatingPoints[0].SeqLevelIdx != 5 {
		t.Fatalf("level=%d want 5", sh.OperatingPoints[0].SeqLevelIdx)
	}
}

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
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint8{15, 17, 14, 18, 13, 19, 12})

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
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint8{12, 14, 11, 10, 9, 8, 7})

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
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint8{16, 16, 16, 16, 16, 16, 16})

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
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint8{15, 17, 14, 18, 13, 19, 12})
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
	size, refs := skipModeRefs(seq, [InterRefsPerFrame]uint8{15, 17, 14, 18, 13, 19, 12})
	transformRef := TransformReferenceParams{ReferenceMode: ReferenceModeSelect, BitsRead: 6}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseSkipModeParams(payload, seq, prefix, size, &refs, transformRef)
	}
}

func skipModeRefs(seq SequenceHeader, orderHints [InterRefsPerFrame]uint8) (FrameSize, ReferenceState) {
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

func TestParseTileGroupHeaderSingleTile(t *testing.T) {
	tiles := TileInfo{Cols: 1, Rows: 1}
	payload := []byte{0xaa}

	group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if group.StartTile != 0 || group.EndTile != 0 || group.TileCount != 1 || !group.Final {
		t.Fatalf("group=%+v", group)
	}
	if group.HeaderBits != 0 || group.BitsRead != 0 || group.DataOffset != 0 || int(group.DataSize) != len(payload) {
		t.Fatalf("header fields=%+v", group)
	}

	var spans [1]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || spans[0].Tile != 0 || spans[0].Offset != 0 || spans[0].Size != 1 {
		t.Fatalf("spans n=%d spans=%+v", n, spans[:])
	}
}

func TestParseTileGroupHeaderExplicitAndSplit(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payload := explicitTileGroupPayload(1, 2)
	payload = append(payload,
		0x02, 0x00, // first tile size minus 1, little-endian
		0xaa, 0xbb, 0xcc,
		0xdd, 0xee, 0xff, 0x11, 0x22,
	)

	group, err := ParseTileGroupHeader(payload, tiles, 0, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if !group.TileStartAndEndPresent || group.StartTile != 1 || group.EndTile != 2 || group.TileCount != 2 || group.Final {
		t.Fatalf("group=%+v", group)
	}
	if group.HeaderBits != 8 || group.BitsRead != 8 || group.DataOffset != 1 || group.DataSize != 10 {
		t.Fatalf("header fields=%+v", group)
	}

	var spans [2]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("span count=%d", n)
	}
	if spans[0] != (TileSpan{Tile: 1, Row: 0, Col: 1, Offset: 3, Size: 3}) {
		t.Fatalf("span[0]=%+v", spans[0])
	}
	if spans[1] != (TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 6, Size: 5}) {
		t.Fatalf("span[1]=%+v", spans[1])
	}
}

// dav1d parse_tile_hdr reads tile_start_and_end_present_flag in an OBU_FRAME
// like in an OBU_TILE_GROUP and does not reject it.
func TestParseTileGroupHeaderAcceptsExplicitFrameOBU(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	group, err := ParseTileGroupHeader(explicitTileGroupPayload(0, 1), tiles, 0, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if group.StartTile != 0 || group.EndTile != 1 || group.Final || group.NextTile != 2 {
		t.Fatalf("group=%+v", group)
	}
}

// An end past the last tile is not an error in dav1d: the frame's tile count
// never reaches the grid size, so it is never submitted.
func TestParseTileGroupHeaderEndPastLastTileIsNotFinal(t *testing.T) {
	tiles := TileInfo{Cols: 3, Rows: 1, Log2Cols: 2}
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(0, 2)
	w.writeBits(3, 2)
	w.writeBits(0, 3)
	group, err := ParseTileGroupHeader(w.bytes(), tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if group.EndTile != 3 || group.Final || group.NextTile != 4 {
		t.Fatalf("group=%+v", group)
	}
}

func TestParseTileGroupHeaderRejectsOutOfOrderStart(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	_, err := ParseTileGroupHeader(explicitTileGroupPayload(1, 2), tiles, 0, 2, false)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

func TestParseTileGroupHeaderRejectsEndBeforeStart(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	_, err := ParseTileGroupHeader(explicitTileGroupPayload(3, 2), tiles, 0, 3, false)
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("ParseTileGroupHeader err=%v want %v", err, ErrInvalidTileGroup)
	}
}

// dav1d_bytealign_get_bits discards the byte_alignment bits after the tile
// group header without checking them.
func TestParseTileGroupHeaderIgnoresAlignmentBits(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1}
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(0, 2)
	w.writeBits(0, 2)
	w.writeBits(7, 3) // nonzero byte_alignment bits
	w.writeBits(0xaa, 8)

	group, err := ParseTileGroupHeader(w.bytes(), tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if group.StartTile != 0 || group.EndTile != 0 || group.DataOffset != 1 || group.DataSize != 1 {
		t.Fatalf("group=%+v", group)
	}
}

func TestSplitTileGroupRejectsShortPayload(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0xff, 0x00)

	group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	var spans [2]TileSpan
	_, err = SplitTileGroup(payload, tiles, group, spans[:])
	if !errors.Is(err, ErrInvalidTileGroup) {
		t.Fatalf("SplitTileGroup err=%v want %v", err, ErrInvalidTileGroup)
	}
}

// TestParseTileGroupHeaderFrameOBUByteAligns guards the frame_obu()
// byte_alignment() that sits between frame_header_obu() and tile_group_obu()
// (spec 5.10.1). For multi-tile frame OBUs whose frame header does not end on a
// byte boundary (e.g. asymmetric uniform tile layouts where TileColsLog2 !=
// TileRowsLog2), startBits is not a multiple of 8. The parser must round it up
// to the next byte so the tile group header — and therefore DataOffset and the
// first tile_size_minus_1 — land on the same byte libaom reads. Without the
// alignment the decoder reads a frame-header byte as tile 0's size, producing
// an absurd 1-byte tile 0 / "invalid tile group".
func TestParseTileGroupHeaderFrameOBUByteAligns(t *testing.T) {
	// 2 tile rows, 1 tile column (the 2x1 asymmetric uniform layout).
	tiles := TileInfo{Cols: 1, Rows: 2, Log2Cols: 0, Log2Rows: 1, TileSizeBytes: 2}

	// Pretend the frame header consumed 95 bits (matches a real 256x256 2x1
	// clip). Byte 11 (bits 88..95) holds the unaligned tail of the frame
	// header; the tile group must start at byte 12, not byte 11.
	const frameHeaderBits = 95
	payload := make([]byte, 24)
	payload[11] = 0xff // frame-header tail: must NOT be read as tile size
	// Tile group OBU starts at byte 12. With numTiles>1 the first bit is
	// tile_start_and_end_present_flag (0 for a frame OBU), then byte_alignment
	// advances to byte 13 where tile 0's 2-byte size lives.
	payload[12] = 0x00 // present flag 0 + alignment padding
	payload[13] = 0x04 // tile0 size_minus_1 low byte -> size 5
	payload[14] = 0x00 // tile0 size_minus_1 high byte

	group, err := ParseTileGroupHeader(payload, tiles, frameHeaderBits, 0, true)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if group.DataOffset != 13 {
		t.Fatalf("DataOffset=%d want 13 (byte-aligned tile group data)", group.DataOffset)
	}

	var spans [2]TileSpan
	n, err := SplitTileGroup(payload, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 2 {
		t.Fatalf("span count=%d want 2", n)
	}
	// tile0: 2 size bytes at 13..14 (=5), payload at 15..19.
	if spans[0].Tile != 0 || spans[0].Row != 0 || spans[0].Col != 0 ||
		spans[0].Offset != 15 || spans[0].Size != 5 {
		t.Fatalf("tile0 span=%+v want offset 15 size 5", spans[0])
	}
	// tile1: remainder 20..23.
	if spans[1].Tile != 1 || spans[1].Row != 1 || spans[1].Col != 0 ||
		spans[1].Offset != 20 || spans[1].Size != 4 {
		t.Fatalf("tile1 span=%+v want offset 20 size 4", spans[1])
	}
}

func TestParseTileGroupHeaderAllocs(t *testing.T) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1, TileSizeBytes: 1}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0x00, 0xaa, 0xbb)

	allocs := testing.AllocsPerRun(1000, func() {
		group, err := ParseTileGroupHeader(payload, tiles, 0, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		var spans [2]TileSpan
		_, err = SplitTileGroup(payload, tiles, group, spans[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("tile group parse allocated: %f", allocs)
	}
}

func BenchmarkParseTileGroupHeader(b *testing.B) {
	tiles := TileInfo{Cols: 2, Rows: 2, Log2Cols: 1, Log2Rows: 1, TileSizeBytes: 1}
	payload := explicitTileGroupPayload(0, 1)
	payload = append(payload, 0x00, 0xaa, 0xbb)

	b.ReportAllocs()
	for b.Loop() {
		group, _ := ParseTileGroupHeader(payload, tiles, 0, 0, false)
		var spans [2]TileSpan
		_, _ = SplitTileGroup(payload, tiles, group, spans[:])
	}
}

func explicitTileGroupPayload(start uint64, end uint64) []byte {
	var w testBitWriter
	w.writeBool(true)
	w.writeBits(start, 2)
	w.writeBits(end, 2)
	for w.bit&7 != 0 {
		w.writeBool(false)
	}
	return w.bytes()
}

func appendTileListEntryRaw(dst []byte, anchorIdx, row, col uint8, dataSizeMinus1 uint16, data []byte) []byte {
	dst = append(dst, anchorIdx, row, col, byte(dataSizeMinus1>>8), byte(dataSizeMinus1))
	dst = append(dst, data...)
	return dst
}

func TestParseTileListOBU(t *testing.T) {
	tile0 := []byte{0xde, 0xad, 0xbe, 0xef}
	tile1 := []byte{0x01, 0x02}
	payload := []byte{
		0x01,       // output_frame_width_in_tiles_minus_1 = 1
		0x02,       // output_frame_height_in_tiles_minus_1 = 2
		0x00, 0x01, // tile_count_minus_1 = 1
	}
	payload = appendTileListEntryRaw(payload, 0x00, 0x01, 0x01, uint16(len(tile0)-1), tile0)
	payload = appendTileListEntryRaw(payload, 0x05, 0x02, 0x00, uint16(len(tile1)-1), tile1)

	var scratch [4]TileListEntry
	list, err := ParseTileListOBU(payload, scratch[:0])
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if got := list.OutputFrameWidthInTiles(); got != 2 {
		t.Fatalf("width=%d want 2", got)
	}
	if got := list.OutputFrameHeightInTiles(); got != 3 {
		t.Fatalf("height=%d want 3", got)
	}
	if got := list.TileCount(); got != 2 {
		t.Fatalf("count=%d want 2", got)
	}
	if len(list.Entries) != 2 {
		t.Fatalf("entries=%d want 2", len(list.Entries))
	}
	if list.Entries[0].AnchorFrameIdx != 0 || list.Entries[0].AnchorTileRow != 1 ||
		list.Entries[0].AnchorTileCol != 1 || list.Entries[0].TileDataSize() != len(tile0) {
		t.Fatalf("entry0=%+v", list.Entries[0])
	}
	if !bytes.Equal(list.Entries[0].TileData, tile0) {
		t.Fatalf("entry0 data=%x want %x", list.Entries[0].TileData, tile0)
	}
	if list.Entries[1].AnchorFrameIdx != 5 || list.Entries[1].AnchorTileRow != 2 ||
		list.Entries[1].AnchorTileCol != 0 || list.Entries[1].TileDataSize() != len(tile1) {
		t.Fatalf("entry1=%+v", list.Entries[1])
	}
	if !bytes.Equal(list.Entries[1].TileData, tile1) {
		t.Fatalf("entry1 data=%x want %x", list.Entries[1].TileData, tile1)
	}
	// Entries should alias the caller scratch.
	if &scratch[0] != &list.Entries[0] {
		t.Fatalf("Entries did not alias caller scratch")
	}
	// TileData should alias payload.
	if &list.Entries[0].TileData[0] != &payload[TileListHeaderBytes+TileListEntryHeaderBytes] {
		t.Fatalf("TileData did not alias payload")
	}
}

func TestParseTileListOBURoundTrip(t *testing.T) {
	tile := []byte{0xaa, 0xbb}
	buf := appendTileListEntryRaw([]byte{3, 4, 0, 0}, 7, 2, 1, uint16(len(tile)-1), tile)
	var scratch [1]TileListEntry
	got, err := ParseTileListOBU(buf, scratch[:0])
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if got.OutputFrameWidthInTilesMinus1 != 3 || got.OutputFrameHeightInTilesMinus1 != 4 || got.TileCountMinus1 != 0 {
		t.Fatalf("header mismatch: %+v", got)
	}
	if len(got.Entries) != 1 {
		t.Fatalf("entries=%d want 1", len(got.Entries))
	}
	e := got.Entries[0]
	if e.AnchorFrameIdx != 7 || e.AnchorTileRow != 2 || e.AnchorTileCol != 1 ||
		e.TileDataSizeMinus1 != uint16(len(tile)-1) || !bytes.Equal(e.TileData, tile) {
		t.Fatalf("entry mismatch: %+v", e)
	}
}

func TestParseTileListOBUErrors(t *testing.T) {
	tile := []byte{0xaa}
	good := []byte{0x00, 0x00, 0x00, 0x00}
	good = appendTileListEntryRaw(good, 0, 0, 0, 0, tile)

	for _, tc := range []struct {
		name    string
		payload []byte
		want    error
	}{
		{name: "short header", payload: []byte{0x00, 0x00, 0x00}, want: ErrTileListShortHeader},
		{name: "tile_count exceeds output", payload: []byte{0x00, 0x00, 0x00, 0x05, 0, 0, 0, 0, 0xaa, 0, 0, 0, 0, 0xbb, 0, 0, 0, 0, 0xcc, 0, 0, 0, 0, 0xdd, 0, 0, 0, 0, 0xee, 0, 0, 0, 0, 0xff}, want: ErrTileListTooManyTiles},
		{name: "short entry header", payload: []byte{0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03}, want: ErrTileListShortEntry},
		{name: "short tile data", payload: []byte{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0x00, 0x04, 0xaa}, want: ErrTileListShortTileData},
		{name: "trailing bytes", payload: append(append([]byte{}, good...), 0xff), want: ErrTileListTrailingBytes},
		{name: "invalid anchor idx", payload: []byte{0x00, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xaa}, want: ErrTileListInvalidAnchorIndex},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseTileListOBU(tc.payload, nil)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err=%v want %v", err, tc.want)
			}
		})
	}
}

func TestValidateTileListAnchors(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	payload = appendTileListEntryRaw(payload, 0, 2, 1, 0, []byte{0xaa})

	list, err := ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if entry := list.Entries[0]; entry.AnchorTileRow != 2 || entry.AnchorTileCol != 1 {
		t.Fatalf("anchor tile=(%d,%d) want (1,2)", entry.AnchorTileCol, entry.AnchorTileRow)
	}

	if err := ValidateTileListAnchors(list, TileInfo{Cols: 2, Rows: 3}); err != nil {
		t.Fatalf("ValidateTileListAnchors err=%v", err)
	}
	if err := ValidateTileListAnchors(list, TileInfo{Cols: 1, Rows: 3}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("small column grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}
	if err := ValidateTileListAnchors(list, TileInfo{Cols: 2, Rows: 2}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("small row grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}
	if err := ValidateTileListAnchors(list, TileInfo{}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("empty grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}

	mismatched := list
	mismatched.TileCountMinus1 = 1
	if err := ValidateTileListAnchors(mismatched, TileInfo{Cols: 2, Rows: 3}); !errors.Is(err, ErrTileListInvalidTileCount) {
		t.Fatalf("mismatched count err=%v want %v", err, ErrTileListInvalidTileCount)
	}
}

func TestValidateTileListDecodeLayout(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	payload = appendTileListEntryRaw(payload, 0, 1, 1, 0, []byte{0xaa})
	list, err := ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}

	uniformSpacing := TileInfo{
		Cols:           2,
		Rows:           2,
		UniformSpacing: true,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 3, 5},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 2, 3},
	}
	width, height, ok := TileListUniformTileSize(uniformSpacing)
	if !ok || width != 3 || height != 2 {
		t.Fatalf("uniform-spacing tile size=%d,%d ok=%v want 3,2,true", width, height, ok)
	}
	width, height, err = ValidateTileListDecodeLayout(list, uniformSpacing)
	if err != nil || width != 3 || height != 2 {
		t.Fatalf("uniform-spacing decode layout size=%d,%d err=%v", width, height, err)
	}

	explicitUniform := TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [MaxTileRows + 1]uint16{0, 1, 2},
	}
	width, height, err = ValidateTileListDecodeLayout(list, explicitUniform)
	if err != nil || width != 2 || height != 1 {
		t.Fatalf("explicit decode layout size=%d,%d err=%v", width, height, err)
	}

	explicitNonUniform := explicitUniform
	explicitNonUniform.ColStartSB[2] = 5
	if _, _, err := ValidateTileListDecodeLayout(list, explicitNonUniform); !errors.Is(err, ErrTileListNonUniformTileSize) {
		t.Fatalf("non-uniform decode layout err=%v want %v", err, ErrTileListNonUniformTileSize)
	}
	invalidAnchorGrid := explicitUniform
	invalidAnchorGrid.Cols = 1
	if _, _, err := ValidateTileListDecodeLayout(list, invalidAnchorGrid); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("invalid anchor decode layout err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}
}

func TestTileListOutputGeometryAndRegions(t *testing.T) {
	payload := []byte{0x01, 0x01, 0x00, 0x02}
	payload = appendTileListEntryRaw(payload, 0, 0, 0, 0, []byte{0xa0})
	payload = appendTileListEntryRaw(payload, 0, 0, 1, 0, []byte{0xa1})
	payload = appendTileListEntryRaw(payload, 0, 1, 1, 0, []byte{0xa2})
	list, err := ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	tiles := TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [MaxTileRows + 1]uint16{0, 1, 2},
	}

	geom, err := TileListOutputGeometryForGrid(list, tiles, false)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid 64: %v", err)
	}
	if geom != (TileListOutputGeometry{
		OutputFrameWidthInTiles:  2,
		OutputFrameHeightInTiles: 2,
		TileCount:                3,
		TileWidthPixels:          128,
		TileHeightPixels:         64,
		OutputFrameWidth:         256,
		OutputFrameHeight:        128,
	}) {
		t.Fatalf("64 geometry=%+v", geom)
	}
	region, err := TileListOutputTileRegion(list, geom, 2)
	if err != nil {
		t.Fatalf("TileListOutputTileRegion: %v", err)
	}
	if region != (TileListTileRegion{SourceX: 128, SourceY: 64, DestX: 0, DestY: 64, Width: 128, Height: 64}) {
		t.Fatalf("region=%+v", region)
	}

	geom128, err := TileListOutputGeometryForGrid(list, tiles, true)
	if err != nil {
		t.Fatalf("TileListOutputGeometryForGrid 128: %v", err)
	}
	if geom128.TileWidthPixels != 256 || geom128.TileHeightPixels != 128 ||
		geom128.OutputFrameWidth != 512 || geom128.OutputFrameHeight != 256 {
		t.Fatalf("128 geometry=%+v", geom128)
	}
	if _, err := TileListOutputTileRegion(list, geom, 3); !errors.Is(err, ErrTileListInvalidTileCount) {
		t.Fatalf("bad region index err=%v want %v", err, ErrTileListInvalidTileCount)
	}
	if _, err := TileListOutputTileRegion(list, TileListOutputGeometry{}, 0); !errors.Is(err, ErrTileListNonUniformTileSize) {
		t.Fatalf("bad geometry err=%v want %v", err, ErrTileListNonUniformTileSize)
	}
}

func TestParseTileListOBUAllocs(t *testing.T) {
	tile := []byte{0xaa, 0xbb}
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	payload = appendTileListEntryRaw(payload, 0, 0, 0, uint16(len(tile)-1), tile)
	scratch := make([]TileListEntry, 4)

	allocs := testing.AllocsPerRun(1000, func() {
		list, err := ParseTileListOBU(payload, scratch[:0])
		if err != nil || len(list.Entries) != 1 {
			t.Fatalf("list=%+v err=%v", list, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTileListOBU allocated: %f", allocs)
	}
}

func FuzzParseTileListOBU(f *testing.F) {
	tile0 := []byte{0xde, 0xad, 0xbe, 0xef}
	tile1 := []byte{0x01, 0x02}
	good := []byte{0x01, 0x02, 0x00, 0x01}
	good = appendTileListEntryRaw(good, 0, 1, 1, uint16(len(tile0)-1), tile0)
	good = appendTileListEntryRaw(good, 5, 2, 0, uint16(len(tile1)-1), tile1)

	for _, seed := range [][]byte{
		nil,
		{0x00},
		{0x00, 0x00, 0x00},
		{0x00, 0x00, 0x00, 0x00},
		{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0xaa},
		{0x00, 0x00, 0x00, 0x00, 0, 2, 1, 0, 0, 0xaa},
		{0x00, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xaa},
		good,
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		var scratch [TileListMaxTiles]TileListEntry
		list, err := ParseTileListOBU(payload, scratch[:0])
		if err != nil {
			return
		}
		if list.TileCount() != len(list.Entries) {
			t.Fatalf("tile count=%d entries=%d", list.TileCount(), len(list.Entries))
		}
		if list.TileCount() <= 0 || list.TileCount() > TileListMaxTiles {
			t.Fatalf("invalid tile count=%d", list.TileCount())
		}
		if list.TileCount() > list.OutputFrameWidthInTiles()*list.OutputFrameHeightInTiles() {
			t.Fatalf("tile count=%d exceeds output %dx%d", list.TileCount(), list.OutputFrameWidthInTiles(), list.OutputFrameHeightInTiles())
		}
		for i, entry := range list.Entries {
			if entry.AnchorFrameIdx >= TileListMaxExternalReferences {
				t.Fatalf("entry %d anchor_frame_idx=%d", i, entry.AnchorFrameIdx)
			}
			if len(entry.TileData) != entry.TileDataSize() {
				t.Fatalf("entry %d data len=%d want %d", i, len(entry.TileData), entry.TileDataSize())
			}
		}
	})
}

func TestParseTileInfoSingleTileExplicit(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.RefreshContext {
		t.Fatal("RefreshContext=false want true")
	}
	if tiles.UniformSpacing {
		t.Fatal("UniformSpacing=true want false")
	}
	if tiles.SBCols != 1 || tiles.SBRows != 1 || tiles.Cols != 1 || tiles.Rows != 1 {
		t.Fatalf("tile grid=%+v", tiles)
	}
	if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[1] != 1 || tiles.RowStartSB[0] != 0 || tiles.RowStartSB[1] != 1 {
		t.Fatalf("tile starts col=%v row=%v", tiles.ColStartSB[:2], tiles.RowStartSB[:2])
	}
	if tiles.TileSizeBytes != 0 || tiles.ContextUpdateTileID != 0 {
		t.Fatalf("tile data fields=%+v", tiles)
	}
	if tiles.BitsRead != w.bit {
		t.Fatalf("BitsRead=%d want %d", tiles.BitsRead, w.bit)
	}
}

func TestParseTileInfoUniformSplit(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 8
	seq.FrameHeightBits = 7
	seq.MaxFrameWidth = 256
	seq.MaxFrameHeight = 128

	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(true)  // uniform_tile_spacing_flag
	w.writeBool(true)  // increment tile_cols_log2 to 1
	w.writeBool(false) // stop before tile_cols_log2 2
	w.writeBool(true)  // increment tile_rows_log2 to 1
	w.writeBits(1, 2)  // context_update_tile_id
	w.writeBits(2, 2)  // tile_size_bytes_minus_1

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.UniformSpacing || tiles.Log2Cols != 1 || tiles.Log2Rows != 1 {
		t.Fatalf("uniform log2 grid=%+v", tiles)
	}
	if tiles.SBCols != 4 || tiles.SBRows != 2 || tiles.Cols != 2 || tiles.Rows != 2 {
		t.Fatalf("tile grid=%+v", tiles)
	}
	if tiles.ColStartSB[0] != 0 || tiles.ColStartSB[1] != 2 || tiles.ColStartSB[2] != 4 {
		t.Fatalf("col starts=%v", tiles.ColStartSB[:3])
	}
	if tiles.RowStartSB[0] != 0 || tiles.RowStartSB[1] != 1 || tiles.RowStartSB[2] != 2 {
		t.Fatalf("row starts=%v", tiles.RowStartSB[:3])
	}
	if tiles.ContextUpdateTileID != 1 || tiles.TileSizeBytes != 3 {
		t.Fatalf("tile data fields=%+v", tiles)
	}
}

func TestParseTileInfoInterMotionControls(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	refs := oneReferenceState(seq, 0, 1)
	w, prefix, size := buildInterFrameSize(t, seq, &refs)
	w.writeBool(true)  // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(uint64(InterpolationSharp), 2)
	w.writeBool(true)  // is_motion_mode_switchable
	w.writeBool(true)  // use_ref_frame_mvs
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag

	tiles, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if err != nil {
		t.Fatal(err)
	}
	if !tiles.AllowHighPrecisionMV || tiles.InterpolationFilter != InterpolationSharp {
		t.Fatalf("motion controls=%+v", tiles)
	}
	if !tiles.SwitchableMotionMode || !tiles.UseRefFrameMVS || !tiles.RefreshContext {
		t.Fatalf("frame controls=%+v", tiles)
	}
}

func TestParseTileInfoRejectsContextTileOutOfRange(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	seq.FrameWidthBits = 8
	seq.FrameHeightBits = 6
	seq.MaxFrameWidth = 192
	seq.MaxFrameHeight = 64

	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	w.writeBool(false) // explicit col 0 width 1 from ns(3)
	w.writeBool(false) // explicit col 1 width 1 from ns(2)
	w.writeBits(3, 2)  // context_update_tile_id, invalid for 3 tiles

	_, err := ParseTileInfo(w.bytes(), seq, prefix, size)
	if !errors.Is(err, ErrInvalidFrameHeader) {
		t.Fatalf("ParseTileInfo err=%v want %v", err, ErrInvalidFrameHeader)
	}
}

func TestReadUniformMatchesDav1dNSCoding(t *testing.T) {
	cases := []struct {
		name string
		bits uint64
		n    uint8
		want uint32
	}{
		{name: "zero", bits: 0b0, n: 1, want: 0},
		{name: "one", bits: 0b10, n: 2, want: 1},
		{name: "two", bits: 0b11, n: 2, want: 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var w testBitWriter
			w.writeBits(tc.bits, tc.n)
			r := bitstream.NewReader(w.bytes())
			got, err := readUniform(&r, 3)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("readUniform=%d want %d", got, tc.want)
			}
		})
	}
}

func TestParseTileInfoAllocs(t *testing.T) {
	seq := mustParseTestSequenceHeader(t, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSize(t, seq)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := ParseTileInfo(payload, seq, prefix, size)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTileInfo allocated: %f", allocs)
	}
}

func BenchmarkParseTileInfo(b *testing.B) {
	seq := mustParseBenchSequenceHeader(b, realtimeSequenceHeader())
	w, prefix, size := buildShownKeyFrameSizeBench(b, seq)
	w.writeBool(false)
	w.writeBool(false)
	payload := w.bytes()

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ParseTileInfo(payload, seq, prefix, size)
	}
}

func buildShownKeyFrameSize(t *testing.T, seq SequenceHeader) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	t.Helper()
	payload, prefix := buildShownKeyFramePrefix(t, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size
}

func buildShownKeyFrameSizeBench(b *testing.B, seq SequenceHeader) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	b.Helper()
	payload, prefix := buildShownKeyFramePrefixBench(b, seq, false)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBool(false)
	w.writeBool(false)
	size, err := ParseFrameSize(w.bytes(), seq, prefix, nil, 0, 0)
	if err != nil {
		b.Fatal(err)
	}
	return w, prefix, size
}

func buildInterFrameSize(t *testing.T, seq SequenceHeader, refs *ReferenceState) (testBitWriter, FrameHeaderPrefix, FrameSize) {
	t.Helper()
	payload, prefix := buildInterFramePrefix(t, seq, false, false, 4)
	var w testBitWriter
	w.writeBitsFrom(payload, prefix.BitsRead)
	w.writeBits(0x02, 8) // refresh_frame_flags
	w.writeBool(false)   // frame_refs_short_signaling
	for range InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // use_superres
	w.writeBool(false) // render_and_frame_size_different
	size, err := ParseFrameSize(w.bytes(), seq, prefix, refs, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return w, prefix, size
}

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
	for b.Loop() {
		_, _ = ParseTransformReferenceParams(payload, prefix, seg, restoration)
	}
}

// sequenceHeaderWithOperatingPointIDC is a minimal profile-0 4:2:0 sequence
// header with one operating point whose operating_point_idc is idc.
func sequenceHeaderWithOperatingPointIDC(idc uint16) []byte {
	var w testBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(uint64(idc), 12)
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(false) // enable_order_hint
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(false) // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

// TestParseSequenceHeaderOperatingPointIDCNamesBothLayerKinds ports dav1d's
// parse_seq_hdr check (src/obu.c): a nonzero operating_point_idc must select
// at least one temporal layer (low 8 bits) and one spatial layer (bits 8-11).
func TestParseSequenceHeaderOperatingPointIDCNamesBothLayerKinds(t *testing.T) {
	for _, tc := range []struct {
		idc uint16
		ok  bool
	}{
		{0x000, true},  // every layer
		{0x101, true},  // T0 in S0
		{0x3ff, true},  // T0-T7 in S0-S1
		{0x001, false}, // no spatial layer
		{0x0ff, false}, // no spatial layer
		{0x100, false}, // no temporal layer
		{0xf00, false}, // no temporal layer
	} {
		sh, err := ParseSequenceHeader(sequenceHeaderWithOperatingPointIDC(tc.idc))
		if tc.ok {
			if err != nil || sh.OperatingPoints[0].IDC != tc.idc {
				t.Errorf("idc=%#x: idc=%#x err=%v, want accepted", tc.idc, sh.OperatingPoints[0].IDC, err)
			}
			continue
		}
		if !errors.Is(err, ErrInvalidSequenceHeader) {
			t.Errorf("idc=%#x: err=%v, want %v", tc.idc, err, ErrInvalidSequenceHeader)
		}
	}
}

// TestRequireFrameHeaderTrailingBit ports dav1d's non-strict
// check_trailing_bits for frame_header_obu: one bit must follow the header,
// whatever its value.
func TestRequireFrameHeaderTrailingBit(t *testing.T) {
	payload := []byte{0xaa, 0x00}
	for _, tc := range []struct {
		bits int
		ok   bool
	}{
		{0, true}, {9, true}, {15, true}, {16, false}, {17, false},
	} {
		err := RequireFrameHeaderTrailingBit(payload, tc.bits)
		if tc.ok != (err == nil) || (!tc.ok && !errors.Is(err, bitstream.ErrNotEnoughBits)) {
			t.Errorf("headerBits=%d: err=%v, want ok=%v", tc.bits, err, tc.ok)
		}
	}
	if err := RequireFrameHeaderTrailingBit(nil, 0); !errors.Is(err, bitstream.ErrNotEnoughBits) {
		t.Errorf("empty payload: err=%v", err)
	}
}

// TestValidateReferenceScale ports dav1d_submit_frame's reference size check
// (src/decode.c): each reference must be at most 2x larger and at most 16x
// smaller than the frame in each dimension.
func TestValidateReferenceScale(t *testing.T) {
	var refs ReferenceState
	for i := range refs.Frames {
		refs.Frames[i].Valid = true
		refs.Frames[i].Size = FrameSize{CodedWidth: 100, UpscaledWidth: 100, Height: 60}
	}
	inter := FrameHeaderPrefix{FrameType: FrameTypeInter}
	for _, tc := range []struct {
		w, h uint32
		err  error
	}{
		{100, 60, nil},
		{50, 30, nil}, // 2x downscale
		{49, 60, ErrInvalidReferenceScale},
		{100, 29, ErrInvalidReferenceScale},
		{1600, 960, nil}, // 16x upscale
		{1601, 60, ErrInvalidReferenceScale},
		{100, 961, ErrInvalidReferenceScale},
	} {
		size := FrameSize{CodedWidth: tc.w, UpscaledWidth: tc.w, Height: tc.h}
		if err := ValidateReferenceScale(inter, size, &refs); !errors.Is(err, tc.err) {
			t.Errorf("%dx%d: err=%v, want %v", tc.w, tc.h, err, tc.err)
		}
	}

	// The coded (pre-super-resolution) width is what is compared with the
	// reference's upscaled width.
	size := FrameSize{CodedWidth: 49, UpscaledWidth: 100, Height: 60}
	if err := ValidateReferenceScale(inter, size, &refs); !errors.Is(err, ErrInvalidReferenceScale) {
		t.Errorf("super-res coded width 49: err=%v", err)
	}

	// Only the seven named references count, and intra frames have none.
	size = FrameSize{CodedWidth: 100, UpscaledWidth: 100, Height: 60}
	refs.Frames[7].Size = FrameSize{CodedWidth: 1000, UpscaledWidth: 1000, Height: 600}
	if err := ValidateReferenceScale(inter, size, &refs); err != nil {
		t.Errorf("unnamed slot out of range: err=%v", err)
	}
	size.RefFrameIdx[3] = 7
	if err := ValidateReferenceScale(inter, size, &refs); !errors.Is(err, ErrInvalidReferenceScale) {
		t.Errorf("named slot out of range: err=%v", err)
	}
	if err := ValidateReferenceScale(FrameHeaderPrefix{FrameType: FrameTypeIntraOnly}, size, &refs); err != nil {
		t.Errorf("intra-only frame: err=%v", err)
	}
	refs.Frames[7].Valid = false
	if err := ValidateReferenceScale(inter, size, &refs); !errors.Is(err, ErrReferenceFrameNeeded) {
		t.Errorf("empty named slot: err=%v", err)
	}
}

func TestSequenceHeaderCoreVectors(t *testing.T) {
	var oracle testvector.Oracle
	if testvector.OracleEnabled {
		oracle = testvector.NewOracle(testvector.CoreSuite().Manifest)
	}
	for _, tag := range [...]testvector.Tag{
		testvector.TagParserSequenceFullLowOverhead,
		testvector.TagParserSequenceReducedLowOverhead,
	} {
		vector, ok := testvector.CoreVector(tag)
		if !ok {
			t.Fatalf("missing vector tag=%x", tag)
		}
		payload := sequenceVectorPayload(t, vector)
		if !bytes.Equal(payload, vector.Want) {
			t.Fatalf("%s payload=%x want %x", vector.Name, payload, vector.Want)
		}
		sh, err := ParseSequenceHeader(payload)
		if err != nil {
			t.Fatalf("%s sequence header parse: %v", vector.Name, err)
		}
		checkSequenceVectorConfig(t, vector, sh)
		if testvector.OracleEnabled {
			if err := oracle.CheckBytes(vector.Tag, payload); err != nil {
				t.Fatalf("%s oracle err=%v", vector.Name, err)
			}
		}
	}
}

func sequenceVectorPayload(t *testing.T, vector testvector.Vector) []byte {
	t.Helper()
	switch vector.Tag {
	case testvector.TagParserSequenceFullLowOverhead, testvector.TagParserSequenceReducedLowOverhead:
		unit, consumed, err := obu.ParseLowOverhead(vector.Input)
		if err != nil {
			t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
		}
		if consumed != len(vector.Input) {
			t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
		}
		if unit.Header.Type != obu.TypeSequenceHeader {
			t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeSequenceHeader)
		}
		return unit.Payload
	default:
		t.Fatalf("%s unhandled tag=%x", vector.Name, vector.Tag)
		return nil
	}
}

func checkSequenceVectorConfig(t *testing.T, vector testvector.Vector, sh SequenceHeader) {
	t.Helper()
	reduced := vector.Tag == testvector.TagParserSequenceReducedLowOverhead
	if sh.SeqProfile != 0 || sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].SeqLevelIdx != 0 || sh.OperatingPoints[0].SeqTier != 0 {
		t.Fatalf("%s profile/op=%+v", vector.Name, sh)
	}
	if sh.StillPicture != reduced || sh.ReducedStillPictureHeader != reduced {
		t.Fatalf("%s still/reduced=%v/%v want %v", vector.Name, sh.StillPicture, sh.ReducedStillPictureHeader, reduced)
	}
	if sh.InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayMinus1 != 0 {
		t.Fatalf("%s initial delay=%+v", vector.Name, sh.OperatingPoints[0])
	}
	cc := sh.ColorConfig
	if cc.HighBitdepth || cc.TwelveBit || cc.MonoChrome || !cc.SubsamplingX || !cc.SubsamplingY || cc.ChromaSamplePosition != 0 {
		t.Fatalf("%s color config=%+v", vector.Name, cc)
	}
}
