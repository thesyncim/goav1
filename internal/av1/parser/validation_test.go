package parser

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

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
