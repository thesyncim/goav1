package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestLossyEncodeStateScansCoverThinChromaMaxTransformSizes(t *testing.T) {
	var st lossyEncodeState
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	cases := []struct {
		name  string
		block tile.BlockSize
		color parser.ColorConfig
		want  tile.TransformSize
	}{
		{name: "420_8x32", block: tile.BlockSize8x32, color: color420, want: tile.TransformSize4x16},
		{name: "420_32x8", block: tile.BlockSize32x8, color: color420, want: tile.TransformSize16x4},
		{name: "420_16x64", block: tile.BlockSize16x64, color: color420, want: tile.TransformSize8x32},
		{name: "420_64x16", block: tile.BlockSize64x16, color: color420, want: tile.TransformSize32x8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tile.MaxTransformSize(tc.block, tc.color, 1)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("MaxTransformSize=%d want %d", got, tc.want)
			}
			dims, ok := got.Dimensions()
			if !ok {
				t.Fatalf("invalid transform size %d", got)
			}
			scan, ok := st.scanForTransformSize(got)
			if !ok {
				scan, ok = st.scanForThinTransformSize(got)
			}
			if !ok {
				t.Fatalf("missing scan for transform %d", got)
			}
			wantLen := int(dims.W4) * 4 * int(dims.H4) * 4
			if len(scan) != wantLen {
				t.Fatalf("scan len=%d want %d", len(scan), wantLen)
			}
		})
	}
}

func TestForwardDCTBlockSupportsThinRectangles(t *testing.T) {
	cases := []struct {
		name string
		w    int
		h    int
	}{
		{name: "4x16", w: 4, h: 16},
		{name: "16x4", w: 16, h: 4},
		{name: "8x32", w: 8, h: 32},
		{name: "32x8", w: 32, h: 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.w * tc.h
			residual := make([]int16, n)
			for i := range residual {
				residual[i] = int16((i*17)%511 - 255)
			}
			coeff := make([]int32, n)
			if err := forwardDCTBlock(coeff, residual, tc.w, tc.h); err != nil {
				t.Fatalf("forwardDCTBlock(%dx%d): %v", tc.w, tc.h, err)
			}
			if coeff[0] == 0 {
				t.Fatalf("forwardDCTBlock(%dx%d) produced zero DC", tc.w, tc.h)
			}
		})
	}
}
