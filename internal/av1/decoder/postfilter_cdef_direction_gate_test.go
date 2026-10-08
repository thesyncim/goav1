package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestCDEFPrimaryDirectionGateUsesRawStrengths(t *testing.T) {
	for _, tc := range []struct {
		name       string
		yStrength  uint8
		uvStrength uint8
		plane      int
		forceLuma  bool
		want       bool
	}{
		{name: "luma secondary only", yStrength: 2, forceLuma: true, want: false},
		{name: "chroma primary needs luma", uvStrength: 3 << 2, forceLuma: true, want: true},
		{name: "uv secondary only", uvStrength: 2, forceLuma: true, want: false},
		{name: "luma primary", yStrength: 1 << 2, forceLuma: true, want: true},
		{name: "luma primary when variance-adjusted strength may be zero", yStrength: 1 << 2, want: true},
		{name: "monochrome ignores uv primary", uvStrength: 1 << 2, want: false},
		{name: "u chroma primary", uvStrength: 1 << 2, plane: 1, want: true},
		{name: "u chroma secondary only", uvStrength: 2, plane: 1, want: false},
		{name: "v chroma primary", uvStrength: 1 << 2, plane: 2, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := parser.CDEFParams{
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{tc.yStrength},
				UVStrength:    [parser.MaxCDEFStrengths]uint8{tc.uvStrength},
			}
			if got := frameWorkCDEFPlaneNeedsPrimaryDirection(params, 0, tc.plane, tc.forceLuma); got != tc.want {
				t.Fatalf("needs direction=%v want %v (Y=%d UV=%d plane=%d force=%v)", got, tc.want, tc.yStrength, tc.uvStrength, tc.plane, tc.forceLuma)
			}
		})
	}
}
