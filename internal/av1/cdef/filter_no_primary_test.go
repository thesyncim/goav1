package cdef

import "testing"

func TestTrustedNoPrimaryDirectionMatchesFilterAndLeavesGridsUntouched(t *testing.T) {
	for _, tc := range []struct {
		name   string
		plane  Plane
		xDec   uint8
		yDec   uint8
		shift  uint8
		width  int
		height int
	}{
		{name: "luma_8bit", plane: PlaneY, width: 8, height: 8},
		{name: "luma_10bit", plane: PlaneY, shift: 2, width: 8, height: 8},
		{name: "chroma_422_12bit", plane: PlaneU, xDec: 1, shift: 4, width: 4, height: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := FrameFilterParams{
				XDec: tc.xDec, YDec: tc.yDec, Plane: tc.plane,
				Level: 0, SecondaryStrength: 2, Damping: 4, CoeffShift: tc.shift,
			}
			input := make([]uint16, InputBufferSize)
			for i := range input {
				input[i] = uint16((i*37 + i/11*13) & ((1 << (8 + tc.shift)) - 1))
			}
			blocks := []BlockPosition{{BY: 0, BX: 0}}
			stride := tc.width + 3
			want := make([]uint16, stride*tc.height)
			got := make([]uint16, stride*tc.height)
			for i := range want {
				want[i] = uint16(i*19 + 7)
				got[i] = want[i]
			}
			var wantDirections, gotDirections DirectionGrid
			var wantVariances, gotVariances VarianceGrid
			for by := range NBlocks {
				for bx := range NBlocks {
					wantDirections[by][bx] = 5
					gotDirections[by][bx] = 5
					wantVariances[by][bx] = 0x12345
					gotVariances[by][bx] = 0x12345
				}
			}
			if err := FilterFrameBlocksTrusted(want, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &wantDirections, &wantVariances, params); err != nil {
				t.Fatalf("trusted filter: %v", err)
			}
			if err := FilterFrameBlocksTrustedNoPrimaryDirection(got, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &gotDirections, &gotVariances, params); err != nil {
				t.Fatalf("no-direction filter: %v", err)
			}
			for row := range tc.height {
				for col := range tc.width {
					i := row*stride + col
					if got[i] != want[i] {
						t.Fatalf("pixel (%d,%d): got=%d want=%d", col, row, got[i], want[i])
					}
				}
			}
			for by := range NBlocks {
				for bx := range NBlocks {
					if gotDirections[by][bx] != 5 || gotVariances[by][bx] != 0x12345 {
						t.Fatalf("grid changed at (%d,%d): dir=%d var=%d", bx, by, gotDirections[by][bx], gotVariances[by][bx])
					}
				}
			}
		})
	}
}

func TestTrustedU8NoPrimaryDirectionMatchesFilterAndLeavesGridsUntouched(t *testing.T) {
	for _, tc := range []struct {
		name   string
		plane  Plane
		xDec   uint8
		yDec   uint8
		width  int
		height int
	}{
		{name: "luma", plane: PlaneY, width: 8, height: 8},
		{name: "chroma_422", plane: PlaneU, xDec: 1, width: 4, height: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := FrameFilterParams{
				XDec: tc.xDec, YDec: tc.yDec, Plane: tc.plane,
				Level: 0, SecondaryStrength: 2, Damping: 4,
			}
			input := make([]uint16, InputBufferSize)
			for i := range input {
				input[i] = uint16((i*29 + i/7*17) & 255)
			}
			blocks := []BlockPosition{{BY: 0, BX: 0}}
			stride := tc.width + 3
			want := make([]byte, stride*tc.height)
			got := make([]byte, stride*tc.height)
			for i := range want {
				want[i] = byte(i*23 + 9)
				got[i] = want[i]
			}
			var wantDirections, gotDirections DirectionGrid
			var wantVariances, gotVariances VarianceGrid
			for by := range NBlocks {
				for bx := range NBlocks {
					wantDirections[by][bx] = 5
					gotDirections[by][bx] = 5
					wantVariances[by][bx] = 0x12345
					gotVariances[by][bx] = 0x12345
				}
			}
			if err := FilterFrameBlocksU8Trusted(want, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &wantDirections, &wantVariances, params); err != nil {
				t.Fatalf("trusted filter: %v", err)
			}
			if err := FilterFrameBlocksU8TrustedNoPrimaryDirection(got, stride, input, VerticalBorder*BStride+HorizontalBorder, blocks, &gotDirections, &gotVariances, params); err != nil {
				t.Fatalf("no-direction filter: %v", err)
			}
			for row := range tc.height {
				for col := range tc.width {
					i := row*stride + col
					if got[i] != want[i] {
						t.Fatalf("pixel (%d,%d): got=%d want=%d", col, row, got[i], want[i])
					}
				}
			}
			for by := range NBlocks {
				for bx := range NBlocks {
					if gotDirections[by][bx] != 5 || gotVariances[by][bx] != 0x12345 {
						t.Fatalf("grid changed at (%d,%d): dir=%d var=%d", bx, by, gotDirections[by][bx], gotVariances[by][bx])
					}
				}
			}
		})
	}
}
