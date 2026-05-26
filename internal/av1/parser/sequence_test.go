package parser

import (
	"errors"
	"testing"
)

type testBitWriter struct {
	buf [128]byte
	bit int
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

func (w *testBitWriter) writeBitsFrom(src []byte, n int) {
	for i := range n {
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
	var w testBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
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
	w.writeBool(false) // mono_chrome
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
