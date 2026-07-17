package loopfilter

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestTrustedFilterEdgesMatchCheckedEntries(t *testing.T) {
	type checkedFilter func(frame.Plane, int, uint8, Edge, int32, int32, int32, Thresholds) error
	type trustedFilter func(frame.Plane, int, uint8, Edge, int, int, int, Thresholds)
	tests := []struct {
		name    string
		checked checkedFilter
		trusted trustedFilter
	}{
		{name: "filter4", checked: Filter4Edge, trusted: Filter4EdgeTrusted},
		{name: "filter6", checked: Filter6Edge, trusted: Filter6EdgeTrusted},
		{name: "filter8", checked: Filter8Edge, trusted: Filter8EdgeTrusted},
		{name: "filter14", checked: Filter14Edge, trusted: Filter14EdgeTrusted},
	}
	thresholds := Thresholds{Limit: 37, BlockLimit: 71, HighEdgeVariance: 11}

	for _, tt := range tests {
		for _, bitDepth := range []uint8{8, 10, 12} {
			bytesPerSample := 1
			if bitDepth > 8 {
				bytesPerSample = 2
			}
			for _, edge := range []Edge{EdgeHorizontal, EdgeVertical} {
				width, height := 48, 48
				plane := testPlane(width, height, bytesPerSample, width*bytesPerSample+8)
				maxSample := (1 << bitDepth) - 1
				for y := range height {
					for x := range width {
						value := uint16((x*73 + y*151 + x*y*7) & maxSample)
						setSample(plane, bytesPerSample, x, y, value)
					}
				}

				want := plane
				want.Pix = append([]byte(nil), plane.Pix...)
				got := plane
				got.Pix = append([]byte(nil), plane.Pix...)
				const x, y, length = 8, 8, 32
				if err := tt.checked(want, bytesPerSample, bitDepth, edge, x, y, length, thresholds); err != nil {
					t.Fatalf("%s bit_depth=%d edge=%d checked entry: %v", tt.name, bitDepth, edge, err)
				}
				tt.trusted(got, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("%s bit_depth=%d edge=%d trusted output differs", tt.name, bitDepth, edge)
				}
			}
		}
	}
}
