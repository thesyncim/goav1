package superres

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// libaomSuperResAlignedRow0 captures the exact luma source row 0 (the
// MI-aligned 112-column span) and the resulting upscaled output row 0 that
// libaom v3.14.0's av1_upscale_normative_rows produces for a coded-width-107,
// upscaled-width-160 super-res frame (denom 12, restoration+CDEF disabled,
// 160x128 input). The source span's columns [107, 112) hold the genuine
// MI-aligned reconstructed pixels (219, 217, 217, 217, 217) carried in the
// coded frame buffer, not a clamp of coded column 106 (218). This pins
// UpscalePlaneCoded against libaom's aligned-width source semantics.
func TestUpscalePlaneCodedMatchesLibaomAlignedRow(t *testing.T) {
	src := []uint16{2, 6, 12, 16, 20, 24, 29, 33, 38, 43, 48, 52, 55, 59, 65, 69, 73, 79, 82, 86, 91, 95, 100, 105, 108, 112, 116, 121, 126, 131, 136, 142, 146, 151, 156, 161, 164, 168, 173, 178, 182, 187, 192, 196, 200, 205, 210, 214, 216, 220, 226, 232, 234, 237, 242, 249, 255, 130, 0, 8, 14, 18, 23, 30, 36, 41, 44, 47, 51, 56, 59, 63, 68, 72, 76, 81, 85, 90, 94, 101, 107, 112, 115, 118, 122, 127, 131, 135, 139, 144, 149, 153, 158, 163, 167, 171, 174, 179, 186, 192, 195, 198, 204, 210, 215, 216, 218, 219, 217, 217, 217, 217}
	want := []uint16{2, 3, 7, 11, 14, 17, 19, 22, 25, 28, 31, 34, 37, 41, 44, 47, 50, 53, 55, 57, 60, 64, 67, 70, 72, 76, 80, 82, 84, 87, 90, 93, 96, 99, 103, 106, 108, 110, 113, 116, 119, 122, 126, 129, 132, 136, 140, 143, 146, 149, 152, 156, 159, 162, 164, 166, 169, 173, 176, 179, 182, 185, 189, 192, 195, 197, 200, 203, 207, 210, 213, 215, 216, 218, 222, 226, 230, 233, 234, 236, 239, 242, 243, 255, 255, 183, 72, 0, 0, 15, 14, 16, 20, 23, 28, 32, 36, 40, 42, 44, 46, 48, 51, 55, 57, 59, 62, 65, 68, 71, 73, 76, 80, 83, 85, 89, 91, 94, 99, 104, 107, 111, 113, 115, 117, 120, 123, 126, 129, 131, 134, 137, 140, 143, 146, 149, 152, 155, 159, 162, 165, 167, 170, 172, 175, 178, 182, 187, 191, 194, 195, 197, 201, 205, 209, 213, 215, 216, 217, 218}

	const codedWidth = 107
	const alignedWidth = 112
	const upscaledWidth = 160
	if len(src) != alignedWidth {
		t.Fatalf("src len = %d, want %d", len(src), alignedWidth)
	}
	if len(want) != upscaledWidth {
		t.Fatalf("want len = %d, want %d", len(want), upscaledWidth)
	}

	srcPlane := frame.SamplePlane{Pix: src, Width: alignedWidth, Height: 1, Stride: alignedWidth}
	dst := make([]uint16, upscaledWidth)
	dstPlane := frame.SamplePlane{Pix: dst, Width: upscaledWidth, Height: 1, Stride: upscaledWidth}

	if err := UpscalePlaneCoded(srcPlane, dstPlane, codedWidth, 8); err != nil {
		t.Fatalf("UpscalePlaneCoded: %v", err)
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("aligned-source upscale mismatch at col %d: got %d want %d", i, dst[i], want[i])
		}
	}

	// The aligned span columns [107, 112) carry genuine reconstructed pixels
	// (219, 217, 217, 217, 217) distinct from coded column 106 (218), confirming
	// the source extent is the MI-aligned width and not a clamp of the last
	// coded column.
	if src[codedWidth] == src[codedWidth-1] {
		t.Fatalf("aligned column %d (%d) unexpectedly equals last coded column %d (%d)",
			codedWidth, src[codedWidth], codedWidth-1, src[codedWidth-1])
	}
}
