package frame

import "testing"

// TestRequiredSizeMIAlignment verifies the superblock-aligned byte-buffer
// extent that RequiredSize computes. libaom allocates the YV12 buffer at the
// MI-aligned dimensions plus a pixel border (aom_realloc_frame_buffer in
// aom_scale/generic/yv12config.c); the bottom/right partial superblock then
// reconstructs the FULL transform of every in-grid block, whose trailing
// rows/cols spill into that border (av1_inverse_transform_block writes the
// whole tx_size; cfl_store_tx subsamples the full reconstructed transform). We
// mirror that by allocating the byte buffer at the superblock-aligned extent so
// any in-grid transform's full extent fits inside the allocation. The reported
// plane Width/Height stay at the cropped (visible) extent.
//
// Expected values are computed by hand: sb = 1<<SBSizeLog2 (default 64),
// alignedW = (W+sb-1)&^(sb-1), alignedH = (H+sb-1)&^(sb-1),
// YStride = align(alignedW*bps, Align), ySize = YStride*alignedH; chroma
// mirrors with subsampling applied to the aligned extent for the buffer span
// and to the visible extent for ChromaW/H.
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
			// round 34 -> 64 (SB64). Chroma visible = (34+1)>>1 = 17; chroma
			// aligned extent = 64>>1 = 32 -> stride align(32,32)=32.
			name:        "34x34-420-8bit",
			format:      Format{Width: 34, Height: 34, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			wantYStride: 64, wantCStride: 32,
			wantChromaWidth: 17, wantCH: 17,
			wantSize: 64*64 + 32*32*2,
		},
		{
			// 66x66 4:2:0 8-bit. Both dims round 66 -> 128 (SB64). YStride =
			// align(128,32)=128; chroma aligned 64 -> align(64,32)=64.
			name:        "66x66-420-8bit",
			format:      Format{Width: 66, Height: 66, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			wantYStride: 128, wantCStride: 64,
			wantChromaWidth: 33, wantCH: 33,
			wantSize: 128*128 + 64*64*2,
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
			sb := 1 << 6
			ySize := tc.wantYStride * ((tc.format.Height + sb - 1) &^ (sb - 1))
			if layout.UOffset != ySize {
				t.Errorf("UOffset=%d want %d (SB-aligned ySize)", layout.UOffset, ySize)
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
