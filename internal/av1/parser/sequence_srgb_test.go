package parser

import (
	"errors"
	"testing"
)

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
