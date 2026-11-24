package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestSequenceHeaderForConfigSRGBProfile keeps the encoder from writing the
// sRGB colour description (BT.709 / sRGB / identity matrix) where dav1d and
// libaom reject it: the triple implies 4:4:4, which only profile 1 and
// 12-bit profile 2 carry. Every accepted header must parse back.
func TestSequenceHeaderForConfigSRGBProfile(t *testing.T) {
	srgb := func(bitDepth uint8) SequenceColorConfig {
		return SequenceColorConfig{
			BitDepth:                bitDepth,
			ColorDescriptionPresent: true,
			ColorPrimaries:          SequenceColorPrimariesBT709,
			TransferCharacteristics: SequenceTransferCharacteristicsSRGB,
			MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
			ColorRange:              true,
		}
	}
	mono := srgb(8)
	mono.MonoChrome = true
	identity420 := SequenceColorConfig{
		BitDepth:                8,
		ColorDescriptionPresent: true,
		ColorPrimaries:          SequenceColorPrimariesBT709,
		TransferCharacteristics: 1,
		MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
		SubsamplingX:            true,
		SubsamplingY:            true,
	}
	for _, tc := range []struct {
		name    string
		profile Profile
		color   SequenceColorConfig
		ok      bool
	}{
		{"profile0-8bit", Profile0, srgb(8), false},
		{"profile0-10bit", Profile0, srgb(10), false},
		{"profile2-8bit", Profile2, srgb(8), false},
		{"profile2-10bit", Profile2, srgb(10), false},
		{"profile1-8bit", Profile1, srgb(8), true},
		{"profile1-10bit", Profile1, srgb(10), true},
		{"profile2-12bit", Profile2, srgb(12), true},
		{"profile0-mono", Profile0, mono, true},
		{"profile0-identity-420", Profile0, identity420, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seq, err := SequenceHeaderForConfig(Config{
				Resolution:     Resolution{Width: 64, Height: 64},
				Profile:        tc.profile,
				BitDepth:       tc.color.BitDepth,
				ColorConfigSet: true,
				ColorConfig:    tc.color,
			})
			if !tc.ok {
				if !errors.Is(err, ErrInvalidConfig) {
					t.Fatalf("SequenceHeaderForConfig err=%v want %v", err, ErrInvalidConfig)
				}
				return
			}
			if err != nil {
				t.Fatalf("SequenceHeaderForConfig: %v", err)
			}
			var buf [160]byte
			out, err := appendSequenceHeaderPayload(buf[:0], seq)
			if err != nil {
				t.Fatalf("appendSequenceHeaderPayload: %v", err)
			}
			if _, err := parser.ParseSequenceHeader(out); err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
		})
	}
}
