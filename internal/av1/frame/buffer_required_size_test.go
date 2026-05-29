package frame

import "testing"

// TestRequiredSizeMIAlignment verifies the MI-aligned byte-buffer extent that
// RequiredSize computes (mirroring libaom aom_realloc_frame_buffer in
// aom_scale/generic/yv12config.c: aligned_width = ROUND_POWER_OF_TWO(width, 3)
// << 3, i.e. round up to a multiple of 8). The reported plane Width/Height stay
// at the cropped (visible) extent while the buffer spans the aligned extent so
// the bottom/right partial superblock can be reconstructed into the padding.
//
// Expected values are computed by hand: alignedW = (W+7)&^7,
// alignedH = (H+7)&^7, YStride = align(alignedW*bps, Align),
// ySize = YStride*alignedH; chroma mirrors with subsampling applied to the
// aligned extent for the buffer span and to the visible extent for ChromaW/H.
func TestRequiredSizeMIAlignment(t *testing.T) {
	cases := []struct {
		name                     string
		format                   Format
		wantYStride, wantCStride int
		wantChromaWidth, wantCH  int
		wantSize                 int
	}{
		{
			// 34x34 4:2:0 8-bit (libaom conformance vector size). Both dims
			// round 34 -> 40. Chroma visible = (34+1)>>1 = 17; chroma aligned
			// extent = 40>>1 = 20 -> stride align(20,32)=32.
			name:        "34x34-420-8bit",
			format:      Format{Width: 34, Height: 34, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			wantYStride: 64, wantCStride: 32,
			wantChromaWidth: 17, wantCH: 17,
			wantSize: 64*40 + 32*20*2,
		},
		{
			// 66x66 4:2:0 8-bit. Both dims round 66 -> 72. YStride =
			// align(72,32)=96; chroma aligned 36 -> align(36,32)=64.
			name:        "66x66-420-8bit",
			format:      Format{Width: 66, Height: 66, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			wantYStride: 96, wantCStride: 64,
			wantChromaWidth: 33, wantCH: 33,
			wantSize: 96*72 + 64*36*2,
		},
		{
			// 64x64 4:4:4 8-bit (profile 1). No subsampling: chroma planes are
			// full-size, so CStride == YStride and chroma extent == luma extent.
			name:        "64x64-444-8bit",
			format:      Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: false, SubsamplingY: false, Align: 32},
			wantYStride: 64, wantCStride: 64,
			wantChromaWidth: 64, wantCH: 64,
			wantSize: 64*64 + 64*64*2,
		},
		{
			// 64x64 4:2:2 8-bit (profile 2). Horizontal-only subsampling:
			// chroma width halves (32) but chroma height stays 64.
			name:        "64x64-422-8bit",
			format:      Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: false, Align: 32},
			wantYStride: 64, wantCStride: 32,
			wantChromaWidth: 32, wantCH: 64,
			wantSize: 64*64 + 32*64*2,
		},
		{
			// 64x64 4:2:0 12-bit (profile 2). 2 bytes/sample doubles strides:
			// YStride = align(64*2,32)=128, CStride = align(32*2,32)=64.
			name:        "64x64-420-12bit",
			format:      Format{Width: 64, Height: 64, BitDepth: 12, SubsamplingX: true, SubsamplingY: true, Align: 32},
			wantYStride: 128, wantCStride: 64,
			wantChromaWidth: 32, wantCH: 32,
			wantSize: 128*64 + 64*32*2,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			layout, err := RequiredSize(tc.format)
			if err != nil {
				t.Fatalf("RequiredSize: %v", err)
			}
			if layout.YStride != tc.wantYStride {
				t.Errorf("YStride=%d want %d", layout.YStride, tc.wantYStride)
			}
			if layout.UStride != tc.wantCStride || layout.VStride != tc.wantCStride {
				t.Errorf("CStride=(%d,%d) want %d", layout.UStride, layout.VStride, tc.wantCStride)
			}
			if layout.ChromaWidth != tc.wantChromaWidth || layout.ChromaHeight != tc.wantCH {
				t.Errorf("chroma=%dx%d want %dx%d", layout.ChromaWidth, layout.ChromaHeight, tc.wantChromaWidth, tc.wantCH)
			}
			if layout.Size != tc.wantSize {
				t.Errorf("Size=%d want %d", layout.Size, tc.wantSize)
			}
			// The U/V plane offsets must partition the buffer exactly:
			// YOffset=0, UOffset=ySize, VOffset=ySize+uSize, Size=ySize+2*uSize.
			ySize := tc.wantYStride * ((tc.format.Height + 7) &^ 7)
			if layout.UOffset != ySize {
				t.Errorf("UOffset=%d want %d (MI-aligned ySize)", layout.UOffset, ySize)
			}
			uSize := layout.VOffset - layout.UOffset
			if layout.Size != layout.VOffset+uSize {
				t.Errorf("VOffset=%d uSize=%d do not partition Size=%d", layout.VOffset, uSize, layout.Size)
			}

			// Bind must accept a buffer of exactly Size and the planes must
			// fully cover it with cropped (visible) Width/Height.
			buf := make([]byte, layout.Size)
			f, err := Bind(buf, tc.format)
			if err != nil {
				t.Fatalf("Bind: %v", err)
			}
			if f.Y.Width != tc.format.Width || f.Y.Height != tc.format.Height {
				t.Errorf("Y plane=%dx%d want %dx%d (visible extent)", f.Y.Width, f.Y.Height, tc.format.Width, tc.format.Height)
			}
			if f.U.Width != tc.wantChromaWidth || f.U.Height != tc.wantCH {
				t.Errorf("U plane=%dx%d want %dx%d", f.U.Width, f.U.Height, tc.wantChromaWidth, tc.wantCH)
			}
		})
	}
}
