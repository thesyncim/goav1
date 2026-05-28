package prediction

import (
	"slices"
	"testing"
)

// Pinned outputs from libaom v3.13.1's av1_filter_intra_predictor_c, captured
// via a standalone harness linked against /tmp/libaom-v3.13.1/out/libaom.a.
// Edge samples below were fed verbatim into both libaom and goav1; the
// expected[] vectors are the bytes libaom wrote into its destination buffer.
// This pins all five filter-intra modes (DC, V, H, D157, PAETH) for two block
// sizes (4x4 and 8x8) at 8-bit depth.

type filterIntraGoldenCase struct {
	name      string
	mode      FilterIntraMode
	width     int
	height    int
	aboveLeft uint16
	above     []uint16
	left      []uint16
	expected  []uint16
}

var filterIntraLibaomGolden = []filterIntraGoldenCase{
	{
		name:  "DC_4x4",
		mode:  FilterIntraModeDC,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			120, 144, 58, 67,
			139, 145, 97, 95,
			55, 80, 64, 72,
			40, 63, 60, 61,
		},
	},
	{
		name:  "DC_8x8",
		mode:  FilterIntraModeDC,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			120, 144, 58, 67, 99, 112, 119, 111,
			139, 145, 97, 95, 108, 111, 120, 113,
			55, 80, 64, 72, 86, 94, 102, 104,
			40, 63, 60, 61, 72, 81, 90, 93,
			82, 86, 78, 75, 78, 83, 87, 91,
			49, 57, 59, 65, 70, 74, 79, 84,
			177, 151, 130, 115, 106, 99, 97, 95,
			103, 98, 94, 103, 102, 97, 95, 97,
		},
	},
	{
		name:  "V_4x4",
		mode:  FilterIntraModeVertical,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			132, 163, 26, 50,
			153, 175, 34, 54,
			84, 134, 7, 40,
			84, 133, 6, 40,
		},
	},
	{
		name:  "V_8x8",
		mode:  FilterIntraModeVertical,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			132, 163, 26, 50, 109, 121, 136, 108,
			153, 175, 34, 54, 111, 123, 137, 109,
			84, 134, 7, 40, 102, 118, 134, 107,
			84, 133, 6, 40, 102, 118, 134, 107,
			124, 157, 22, 48, 107, 121, 136, 108,
			93, 139, 10, 42, 103, 119, 135, 107,
			203, 205, 54, 64, 117, 127, 141, 110,
			125, 158, 23, 48, 107, 121, 137, 108,
		},
	},
	{
		name:  "H_4x4",
		mode:  FilterIntraModeHorizontal,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			131, 155, 91, 108,
			149, 162, 130, 138,
			32, 39, 23, 27,
			28, 31, 23, 25,
		},
	},
	{
		name:  "H_8x8",
		mode:  FilterIntraModeHorizontal,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			131, 155, 91, 108, 139, 147, 155, 141,
			149, 162, 130, 138, 154, 157, 161, 155,
			32, 39, 23, 27, 35, 37, 39, 36,
			28, 31, 23, 25, 29, 30, 31, 29,
			90, 92, 88, 89, 91, 92, 92, 91,
			40, 41, 39, 39, 40, 40, 41, 40,
			216, 216, 215, 215, 216, 216, 216, 216,
			90, 91, 90, 90, 90, 90, 91, 90,
		},
	},
	{
		name:  "D157_4x4",
		mode:  FilterIntraModeD157,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			104, 121, 64, 57,
			123, 125, 96, 82,
			60, 87, 85, 85,
			42, 61, 67, 70,
		},
	},
	{
		name:  "D157_8x8",
		mode:  FilterIntraModeD157,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			104, 121, 64, 57, 82, 97, 114, 107,
			123, 125, 96, 82, 85, 91, 101, 103,
			60, 87, 85, 85, 85, 88, 94, 98,
			42, 61, 67, 70, 76, 81, 87, 91,
			73, 70, 71, 69, 72, 76, 81, 85,
			50, 58, 64, 65, 68, 72, 76, 79,
			155, 117, 100, 82, 77, 75, 76, 77,
			107, 109, 107, 97, 89, 84, 82, 81,
		},
	},
	{
		name:  "PAETH_4x4",
		mode:  FilterIntraModePaeth,
		width: 4, height: 4,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41},
		left:      []uint16{102, 135, 25, 24},
		expected: []uint16{
			143, 177, 60, 85,
			165, 192, 90, 100,
			65, 102, 20, 36,
			60, 96, 18, 37,
		},
	},
	{
		name:  "PAETH_8x8",
		mode:  FilterIntraModePaeth,
		width: 8, height: 8,
		aboveLeft: 30,
		above:     []uint16{87, 136, 8, 41, 103, 118, 134, 107},
		left:      []uint16{102, 135, 25, 24, 88, 39, 215, 90},
		expected: []uint16{
			143, 177, 60, 85, 134, 141, 153, 126,
			165, 192, 90, 100, 139, 147, 154, 132,
			65, 102, 20, 36, 78, 93, 103, 88,
			60, 96, 18, 37, 74, 90, 93, 88,
			112, 135, 63, 75, 103, 112, 112, 105,
			64, 92, 34, 46, 73, 86, 88, 86,
			215, 217, 156, 155, 165, 163, 158, 149,
			102, 119, 81, 84, 100, 106, 110, 107,
		},
	},
}

// TestFilterIntraPredictorMatchesLibaomGolden pins every filter-intra mode
// against vectors captured directly from libaom v3.13.1's
// av1_filter_intra_predictor_c. These vectors are independent of the goav1
// implementation, so they catch regressions in the rounding, clamping, tap
// table, and edge-sample ordering paths simultaneously.
func TestFilterIntraPredictorMatchesLibaomGolden(t *testing.T) {
	for _, tc := range filterIntraLibaomGolden {
		t.Run(tc.name, func(t *testing.T) {
			edges := IntraEdges{
				Above:              tc.above,
				Left:               tc.left,
				AboveLeft:          tc.aboveLeft,
				AboveAvailable:     true,
				LeftAvailable:      true,
				AboveLeftAvailable: true,
			}
			plane, _ := testPlane(tc.width, tc.height, 1, tc.width)
			if err := PredictFilterIntraPlaneBlock(plane, 1, 8, 0, 0, tc.width, tc.height, tc.mode, edges); err != nil {
				t.Fatalf("PredictFilterIntraPlaneBlock err=%v", err)
			}
			got := collectPlaneSamples(plane, 1, tc.width, tc.height)
			if !slices.Equal(got, tc.expected) {
				t.Fatalf("mode=%d %dx%d\n got=%v\nwant=%v", tc.mode, tc.width, tc.height, got, tc.expected)
			}
		})
	}
}
