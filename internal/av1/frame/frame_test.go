package frame

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
)

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

func TestRequiredSize420Aligned(t *testing.T) {
	layout, err := RequiredSize(Format{
		Width:        16,
		Height:       9,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	if err != nil {
		t.Fatal(err)
	}

	if layout.YStride != 64 || layout.UStride != 64 || layout.VStride != 64 {
		t.Fatalf("strides: %+v", layout)
	}
	if layout.ChromaWidth != 8 || layout.ChromaHeight != 5 {
		t.Fatalf("chroma: %dx%d", layout.ChromaWidth, layout.ChromaHeight)
	}
	// The byte buffer spans the superblock-aligned extent (width 16 -> 64,
	// height 9 -> 64; chroma 32x32) while ChromaWidth/Height report the cropped
	// extent.
	if layout.Size != 64*64+64*32*2 {
		t.Fatalf("size=%d", layout.Size)
	}
}

func TestBindFrame(t *testing.T) {
	format := Format{
		Width:        17,
		Height:       9,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	frame, err := Bind(buffer, format)
	if err != nil {
		t.Fatal(err)
	}
	if frame.Layout.Size != layout.Size || len(frame.Y.Pix)+len(frame.U.Pix)+len(frame.V.Pix) != layout.Size {
		t.Fatalf("frame=%+v layout=%+v", frame, layout)
	}
	if frame.Y.Width != 17 || frame.U.Width != 9 || frame.V.Height != 5 {
		t.Fatalf("planes: Y=%+v U=%+v V=%+v", frame.Y, frame.U, frame.V)
	}
}

func TestRequiredSizeMonochrome(t *testing.T) {
	layout, err := RequiredSize(Format{
		Width:      16,
		Height:     9,
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	})
	if err != nil {
		t.Fatal(err)
	}
	if layout.YStride != 64 || layout.UStride != 0 || layout.VStride != 0 {
		t.Fatalf("strides: %+v", layout)
	}
	if layout.ChromaWidth != 0 || layout.ChromaHeight != 0 {
		t.Fatalf("chroma: %dx%d", layout.ChromaWidth, layout.ChromaHeight)
	}
	// Superblock-aligned allocation rounds width 16 -> 64 and height 9 -> 64.
	if layout.UOffset != 64*64 || layout.VOffset != 64*64 || layout.Size != 64*64 {
		t.Fatalf("layout=%+v", layout)
	}

	buffer := make([]byte, layout.Size)
	frame, err := Bind(buffer, Format{Width: 16, Height: 9, BitDepth: 8, MonoChrome: true, Align: 64})
	if err != nil {
		t.Fatal(err)
	}
	if len(frame.U.Pix) != 0 || len(frame.V.Pix) != 0 || frame.U.Stride != 0 || frame.V.Stride != 0 {
		t.Fatalf("monochrome planes: U=%+v V=%+v", frame.U, frame.V)
	}
}

func TestBindRejectsShortBuffer(t *testing.T) {
	_, err := Bind(make([]byte, 1), Format{Width: 16, Height: 16, BitDepth: 8})
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("Bind err=%v want %v", err, ErrShortBuffer)
	}
}

func TestSamplePlaneLoadStore8Bit(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 5, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte(10*y + x)
		}
	}
	need, err := SamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if need != 24 {
		t.Fatalf("need=%d", need)
	}
	scratch := make([]uint16, need)
	samples, err := LoadSamplePlane(scratch, plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Stride != 8 || samples.Width != 5 || samples.Height != 3 {
		t.Fatalf("samples=%+v", samples)
	}
	if got := samples.Pix[2*samples.Stride+4]; got != 24 {
		t.Fatalf("sample=%d want 24", got)
	}

	samples.Pix[1*samples.Stride+3] = 201
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := StoreSamplePlane(dst, 1, samples); err != nil {
		t.Fatal(err)
	}
	if dst.Pix[1*dst.Stride+3] != 201 || dst.Pix[2*dst.Stride+4] != 24 {
		t.Fatalf("dst=%v", dst.Pix)
	}
}

func TestSamplePlaneLoadStoreHighBitDepth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 12*2), Stride: 12, Width: 4, Height: 2}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setTestPlaneSample(plane, 2, x, y, uint16(1000+y*10+x))
		}
	}
	need, err := SamplePlaneLen(plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	if need != 12 {
		t.Fatalf("need=%d", need)
	}
	samples, err := LoadSamplePlane(make([]uint16, need), plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Stride != 6 || samples.Pix[1*samples.Stride+2] != 1012 {
		t.Fatalf("samples=%+v", samples)
	}
	samples.Pix[0*samples.Stride+1] = 4095
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := StoreSamplePlane(dst, 2, samples); err != nil {
		t.Fatal(err)
	}
	if got := getTestPlaneSample(dst, 2, 1, 0); got != 4095 {
		t.Fatalf("stored=%d want 4095", got)
	}
	if got := getTestPlaneSample(dst, 2, 2, 1); got != 1012 {
		t.Fatalf("stored=%d want 1012", got)
	}
}

func TestBindSamplePlane(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 5, Height: 3}
	need, err := SamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, need+3)
	samples, err := BindSamplePlane(scratch, plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Stride != 8 || samples.Width != 5 || samples.Height != 3 || len(samples.Pix) != need || cap(samples.Pix) != cap(scratch) {
		t.Fatalf("samples=%+v len/cap=%d/%d want len %d cap %d", samples, len(samples.Pix), cap(samples.Pix), need, cap(scratch))
	}
	if _, err := BindSamplePlane(scratch[:need-1], plane, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short bind err=%v want %v", err, ErrShortBuffer)
	}
}

func TestBorderedSamplePlaneLoadStore8Bit(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 5, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte(10*y + x)
		}
	}

	layout, err := BorderedSamplePlaneLen(plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if want := (BorderedSamplePlaneLayout{Stride: 16, Origin: 35, Rows: 7, Len: 112}); layout != want {
		t.Fatalf("layout=%+v want %+v", layout, want)
	}

	scratch := make([]uint16, layout.Len)
	for i := range scratch {
		scratch[i] = 0xeeee
	}
	samples, err := LoadBorderedSamplePlane(scratch, plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Stride != 16 || samples.Origin != 35 || samples.Width != 5 || samples.Height != 3 {
		t.Fatalf("samples=%+v", samples)
	}
	if samples.Pix[0] != 0xeeee || samples.Pix[samples.Origin-1] != 0xeeee {
		t.Fatalf("border samples were overwritten")
	}
	if got := samples.Pix[samples.Origin+2*samples.Stride+4]; got != 24 {
		t.Fatalf("sample=%d want 24", got)
	}

	samples.Pix[samples.Origin+1*samples.Stride+3] = 201
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := StoreBorderedSamplePlaneRectTrusted(dst, 1, samples, 0, 0, samples.Width, samples.Height); err != nil {
		t.Fatal(err)
	}
	if dst.Pix[1*dst.Stride+3] != 201 || dst.Pix[2*dst.Stride+4] != 24 {
		t.Fatalf("dst=%v", dst.Pix)
	}
}

func TestBorderedSamplePlaneLoadStoreHighBitDepth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 12*2), Stride: 12, Width: 4, Height: 2}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setTestPlaneSample(plane, 2, x, y, uint16(1000+y*10+x))
		}
	}

	layout, err := BorderedSamplePlaneLen(plane, 2, 5, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if want := (BorderedSamplePlaneLayout{Stride: 16, Origin: 21, Rows: 4, Len: 64}); layout != want {
		t.Fatalf("layout=%+v want %+v", layout, want)
	}

	samples, err := LoadBorderedSamplePlane(make([]uint16, layout.Len), plane, 2, 5, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if got := samples.Pix[samples.Origin+1*samples.Stride+2]; got != 1012 {
		t.Fatalf("sample=%d want 1012", got)
	}
	samples.Pix[samples.Origin+1] = 4095
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := StoreBorderedSamplePlaneRectTrusted(dst, 2, samples, 0, 0, samples.Width, samples.Height); err != nil {
		t.Fatal(err)
	}
	if got := getTestPlaneSample(dst, 2, 1, 0); got != 4095 {
		t.Fatalf("stored=%d want 4095", got)
	}
	if got := getTestPlaneSample(dst, 2, 2, 1); got != 1012 {
		t.Fatalf("stored=%d want 1012", got)
	}
}

// TestStoreBorderedSamplePlaneRectTrustedMatchesFullStore proves that tiling a
// plane with rect stores reproduces the full trusted store byte-for-byte, and
// that a partial rect store leaves every destination byte outside the rect
// untouched.
func TestStoreBorderedSamplePlaneRectTrustedMatchesFullStore(t *testing.T) {
	for _, bytesPerSample := range [...]int{1, 2} {
		plane := Plane{Pix: make([]byte, 9*bytesPerSample*6), Stride: 9 * bytesPerSample, Width: 7, Height: 6}
		maxSample := uint16(0xff)
		if bytesPerSample == 2 {
			maxSample = 0xfff
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setTestPlaneSample(plane, bytesPerSample, x, y, uint16(31*y+7*x)&maxSample)
			}
		}
		layout, err := BorderedSamplePlaneLen(plane, bytesPerSample, 3, 2, 8)
		if err != nil {
			t.Fatal(err)
		}
		samples, err := LoadBorderedSamplePlane(make([]uint16, layout.Len), plane, bytesPerSample, 3, 2, 8)
		if err != nil {
			t.Fatal(err)
		}

		full := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
		if err := StoreBorderedSamplePlaneRectTrusted(full, bytesPerSample, samples, 0, 0, samples.Width, samples.Height); err != nil {
			t.Fatal(err)
		}
		tiled := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
		rects := [][4]int{{0, 0, 4, 3}, {4, 0, 3, 3}, {0, 3, 4, 3}, {4, 3, 3, 3}}
		for _, r := range rects {
			if err := StoreBorderedSamplePlaneRectTrusted(tiled, bytesPerSample, samples, r[0], r[1], r[2], r[3]); err != nil {
				t.Fatal(err)
			}
		}
		if string(tiled.Pix) != string(full.Pix) {
			t.Fatalf("bytesPerSample=%d tiled=% x full=% x", bytesPerSample, tiled.Pix, full.Pix)
		}

		partial := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
		for i := range partial.Pix {
			partial.Pix[i] = 0xa5
		}
		if err := StoreBorderedSamplePlaneRectTrusted(partial, bytesPerSample, samples, 2, 1, 3, 2); err != nil {
			t.Fatal(err)
		}
		for y := 0; y < partial.Height; y++ {
			for x := 0; x < partial.Width; x++ {
				inside := x >= 2 && x < 5 && y >= 1 && y < 3
				got := getTestPlaneSample(partial, bytesPerSample, x, y)
				if inside {
					want := getTestPlaneSample(full, bytesPerSample, x, y)
					if got != want {
						t.Fatalf("bytesPerSample=%d inside (%d,%d)=%d want %d", bytesPerSample, x, y, got, want)
					}
					continue
				}
				stale := uint16(0xa5)
				if bytesPerSample == 2 {
					stale = 0xa5a5
				}
				if got != stale {
					t.Fatalf("bytesPerSample=%d outside (%d,%d)=%d want stale %d", bytesPerSample, x, y, got, stale)
				}
			}
		}

		if err := StoreBorderedSamplePlaneRectTrusted(partial, bytesPerSample, samples, 0, 0, 0, 0); err != nil {
			t.Fatalf("empty rect err=%v", err)
		}
		for _, bad := range [][4]int{{-1, 0, 2, 2}, {0, -1, 2, 2}, {6, 0, 2, 2}, {0, 5, 2, 2}, {0, 0, 8, 1}, {0, 0, 1, 7}} {
			if err := StoreBorderedSamplePlaneRectTrusted(partial, bytesPerSample, samples, bad[0], bad[1], bad[2], bad[3]); !errors.Is(err, ErrInvalidPlane) {
				t.Fatalf("rect %v err=%v want %v", bad, err, ErrInvalidPlane)
			}
		}
	}
}

func TestStoreBorderedSamplePlaneTrustedRejectsInvalidGeometry(t *testing.T) {
	src := BorderedSamplePlane{Pix: make([]uint16, 10), Stride: 4, Origin: 3, Width: 2, Height: 1}
	dst := Plane{Pix: make([]byte, 2), Stride: 2, Width: 2, Height: 1}
	if err := StoreBorderedSamplePlaneRectTrusted(dst, 1, src, 0, 0, src.Width, src.Height); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("cross-stride trusted store err=%v want %v", err, ErrInvalidPlane)
	}
	mismatch := BorderedSamplePlane{Pix: make([]uint16, 4), Stride: 2, Origin: 0, Width: 1, Height: 1}
	if err := StoreBorderedSamplePlaneRectTrusted(dst, 1, mismatch, 0, 0, mismatch.Width, mismatch.Height); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("mismatch trusted store err=%v want %v", err, ErrInvalidPlane)
	}
}

func TestBindBorderedSamplePlane(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 5, Height: 3}
	layout, err := BorderedSamplePlaneLen(plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, layout.Len)
	for i := range scratch {
		scratch[i] = 0xeeee
	}

	samples, err := BindBorderedSamplePlane(scratch, plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Stride != layout.Stride || samples.Origin != layout.Origin ||
		samples.Width != plane.Width || samples.Height != plane.Height {
		t.Fatalf("samples=%+v layout=%+v", samples, layout)
	}
	if samples.Pix[0] != 0xeeee || samples.Pix[samples.Origin] != 0xeeee {
		t.Fatalf("bind unexpectedly initialized samples")
	}
}

func TestSamplePlaneRejectsInvalidInputs(t *testing.T) {
	plane := Plane{Pix: make([]byte, 16), Stride: 4, Width: 4, Height: 4}
	if _, err := SamplePlaneLen(plane, 3); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("bad bps err=%v want %v", err, ErrInvalidPlane)
	}
	if _, err := SamplePlaneLen(Plane{Pix: make([]byte, 7), Stride: 7, Width: 3, Height: 1}, 2); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("misaligned stride err=%v want %v", err, ErrInvalidPlane)
	}
	if need, err := SamplePlaneLen(Plane{Stride: 4, Width: 4, Height: 4}, 1); err != nil || need != 16 {
		t.Fatalf("geometry-only len=%d err=%v want 16,nil", need, err)
	}
	if _, err := LoadSamplePlane(make([]uint16, 3), plane, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, ErrShortBuffer)
	}
	short := Plane{Pix: plane.Pix[:15], Stride: 4, Width: 4, Height: 4}
	if _, err := LoadSamplePlane(make([]uint16, 16), short, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("short plane err=%v want %v", err, ErrInvalidPlane)
	}
	samples := SamplePlane{Pix: []uint16{256}, Stride: 1, Width: 1, Height: 1}
	if err := StoreSamplePlane(Plane{Pix: make([]byte, 1), Stride: 1, Width: 1, Height: 1}, 1, samples); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("overflow store err=%v want %v", err, ErrInvalidPlane)
	}
	if err := StoreSamplePlane(Plane{Pix: make([]byte, 1), Stride: 1, Width: 1, Height: 1}, 1, SamplePlane{Pix: make([]uint16, 2), Stride: 2, Width: 2, Height: 1}); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("mismatch store err=%v want %v", err, ErrInvalidPlane)
	}
}

func TestBorderedSamplePlaneRejectsInvalidInputs(t *testing.T) {
	plane := Plane{Pix: make([]byte, 16), Stride: 4, Width: 4, Height: 4}
	if _, err := BorderedSamplePlaneLen(plane, 1, -1, 0, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("negative border err=%v want %v", err, ErrInvalidPlane)
	}
	if _, err := BorderedSamplePlaneLen(plane, 1, 1, 1, 3); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("bad align err=%v want %v", err, ErrInvalidPlane)
	}
	if _, err := BorderedSamplePlaneLen(Plane{}, 1, 1, 0, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("empty bordered plane err=%v want %v", err, ErrInvalidPlane)
	}
	if layout, err := BorderedSamplePlaneLen(Plane{Stride: 4, Width: 4, Height: 4}, 1, 1, 1, 1); err != nil || layout.Len == 0 {
		t.Fatalf("geometry-only bordered layout=%+v err=%v", layout, err)
	}
	if _, err := LoadBorderedSamplePlane(make([]uint16, 3), plane, 1, 1, 1, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, ErrShortBuffer)
	}
	if _, err := BindBorderedSamplePlane(make([]uint16, 3), plane, 1, 1, 1, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short bind err=%v want %v", err, ErrShortBuffer)
	}
	crossesStride := BorderedSamplePlane{Pix: make([]uint16, 10), Stride: 4, Origin: 3, Width: 2, Height: 1}
	if err := StoreBorderedSamplePlaneRectTrusted(Plane{Pix: make([]byte, 2), Stride: 2, Width: 2, Height: 1}, 1, crossesStride, 0, 0, 2, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("cross-stride store err=%v want %v", err, ErrInvalidPlane)
	}
}

func TestFrameBindAllocs(t *testing.T) {
	format := Format{Width: 128, Height: 72, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := Bind(buffer, format)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Bind allocated: %f", allocs)
	}
}

func TestSamplePlaneAllocs(t *testing.T) {
	plane := Plane{Pix: make([]byte, 32*16), Stride: 32, Width: 16, Height: 16}
	scratch := make([]uint16, 32*16)
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: 32, Width: 16, Height: 16}
	allocs := testing.AllocsPerRun(1000, func() {
		samples, err := LoadSamplePlane(scratch, plane, 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := StoreSamplePlane(dst, 1, samples); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("sample plane helpers allocated: %f", allocs)
	}
}

func TestBindSamplePlaneAllocs(t *testing.T) {
	plane := Plane{Pix: make([]byte, 32*16), Stride: 32, Width: 16, Height: 16}
	scratch := make([]uint16, 32*16)
	allocs := testing.AllocsPerRun(1000, func() {
		samples, err := BindSamplePlane(scratch, plane, 1)
		if err != nil {
			t.Fatal(err)
		}
		if samples.Stride != 32 || samples.Width != 16 || samples.Height != 16 {
			t.Fatalf("samples=%+v", samples)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindSamplePlane allocated: %f", allocs)
	}
}

func TestBorderedSamplePlaneAllocs(t *testing.T) {
	plane := Plane{Pix: make([]byte, 32*16), Stride: 32, Width: 16, Height: 16}
	layout, err := BorderedSamplePlaneLen(plane, 1, 4, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, layout.Len)
	dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: 32, Width: 16, Height: 16}
	allocs := testing.AllocsPerRun(1000, func() {
		samples, err := LoadBorderedSamplePlane(scratch, plane, 1, 4, 2, 16)
		if err != nil {
			t.Fatal(err)
		}
		if err := StoreBorderedSamplePlaneRectTrusted(dst, 1, samples, 0, 0, samples.Width, samples.Height); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("bordered sample plane helpers allocated: %f", allocs)
	}
}

func FuzzSamplePlaneRoundTrip(f *testing.F) {
	f.Add(uint8(5), uint8(3), uint8(1), uint8(2), uint32(0x12345678))
	f.Add(uint8(4), uint8(2), uint8(2), uint8(3), uint32(0x90abcdef))
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBPS uint8, rawPad uint8, seed uint32) {
		width := int(rawW%64) + 1
		height := int(rawH%32) + 1
		bytesPerSample := int(rawBPS&1) + 1
		strideSamples := width + int(rawPad%8)
		stride := strideSamples * bytesPerSample
		plane := Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
		state := seed
		for y := range height {
			for x := range width {
				state = state*1664525 + 1013904223
				max := uint16(0xff)
				if bytesPerSample == 2 {
					max = 0x0fff
				}
				setTestPlaneSample(plane, bytesPerSample, x, y, uint16(state)&max)
			}
		}
		need, err := SamplePlaneLen(plane, bytesPerSample)
		if err != nil {
			t.Fatalf("SamplePlaneLen err=%v", err)
		}
		samples, err := LoadSamplePlane(make([]uint16, need), plane, bytesPerSample)
		if err != nil {
			t.Fatalf("LoadSamplePlane err=%v", err)
		}
		dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: stride, Width: width, Height: height}
		if err := StoreSamplePlane(dst, bytesPerSample, samples); err != nil {
			t.Fatalf("StoreSamplePlane err=%v", err)
		}
		for y := range height {
			for x := range width {
				got := getTestPlaneSample(dst, bytesPerSample, x, y)
				want := getTestPlaneSample(plane, bytesPerSample, x, y)
				if got != want {
					t.Fatalf("sample x=%d y=%d got=%d want=%d", x, y, got, want)
				}
			}
		}
	})
}

func FuzzBorderedSamplePlaneRoundTrip(f *testing.F) {
	f.Add(uint8(5), uint8(3), uint8(1), uint8(2), uint8(3), uint8(2), uint8(3), uint32(0x12345678))
	f.Add(uint8(4), uint8(2), uint8(2), uint8(3), uint8(5), uint8(1), uint8(4), uint32(0x90abcdef))
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBPS uint8, rawPad uint8, rawBorderHorz uint8, rawBorderVert uint8, rawAlign uint8, seed uint32) {
		width := int(rawW%64) + 1
		height := int(rawH%32) + 1
		bytesPerSample := int(rawBPS&1) + 1
		strideSamples := width + int(rawPad%8)
		stride := strideSamples * bytesPerSample
		borderHorz := int(rawBorderHorz % 16)
		borderVert := int(rawBorderVert % 8)
		align := 1 << (rawAlign & 3)
		plane := Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
		state := seed
		for y := range height {
			for x := range width {
				state = state*1664525 + 1013904223
				max := uint16(0xff)
				if bytesPerSample == 2 {
					max = 0x0fff
				}
				setTestPlaneSample(plane, bytesPerSample, x, y, uint16(state)&max)
			}
		}
		layout, err := BorderedSamplePlaneLen(plane, bytesPerSample, borderHorz, borderVert, align)
		if err != nil {
			t.Fatalf("BorderedSamplePlaneLen err=%v", err)
		}
		scratch := make([]uint16, layout.Len)
		for i := range scratch {
			scratch[i] = 0xeeee
		}
		samples, err := LoadBorderedSamplePlane(scratch, plane, bytesPerSample, borderHorz, borderVert, align)
		if err != nil {
			t.Fatalf("LoadBorderedSamplePlane err=%v", err)
		}
		if borderVert > 0 && samples.Pix[0] != 0xeeee {
			t.Fatalf("top border overwritten")
		}
		if borderHorz > 0 && samples.Pix[samples.Origin-1] != 0xeeee {
			t.Fatalf("left border overwritten")
		}
		dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: stride, Width: width, Height: height}
		if err := StoreBorderedSamplePlaneRectTrusted(dst, bytesPerSample, samples, 0, 0, samples.Width, samples.Height); err != nil {
			t.Fatalf("StoreBorderedSamplePlane err=%v", err)
		}
		for y := range height {
			for x := range width {
				got := getTestPlaneSample(dst, bytesPerSample, x, y)
				want := getTestPlaneSample(plane, bytesPerSample, x, y)
				if got != want {
					t.Fatalf("sample x=%d y=%d got=%d want=%d", x, y, got, want)
				}
			}
		}
	})
}

func BenchmarkBindFrame(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	buffer := make([]byte, layout.Size)

	b.ReportAllocs()
	for b.Loop() {
		_, _ = Bind(buffer, format)
	}
}

func BenchmarkSamplePlaneLoadStore(b *testing.B) {
	for _, bytesPerSample := range [...]int{1, 2} {
		b.Run(samplePlaneBenchmarkName(bytesPerSample), func(b *testing.B) {
			plane := Plane{Pix: make([]byte, 1920*bytesPerSample*1080), Stride: 1920 * bytesPerSample, Width: 1920, Height: 1080}
			scratch := make([]uint16, 1920*1080)
			dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}

			b.ResetTimer()
			b.ReportAllocs()
			for b.Loop() {
				samples, _ := LoadSamplePlane(scratch, plane, bytesPerSample)
				_ = StoreSamplePlane(dst, bytesPerSample, samples)
			}
		})
	}
}

func BenchmarkBorderedSamplePlaneLoadStore(b *testing.B) {
	for _, bytesPerSample := range [...]int{1, 2} {
		b.Run(samplePlaneBenchmarkName(bytesPerSample), func(b *testing.B) {
			plane := Plane{Pix: make([]byte, 1920*bytesPerSample*1080), Stride: 1920 * bytesPerSample, Width: 1920, Height: 1080}
			layout, err := BorderedSamplePlaneLen(plane, bytesPerSample, 32, 32, 64)
			if err != nil {
				b.Fatal(err)
			}
			scratch := make([]uint16, layout.Len)
			dst := Plane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}

			b.ResetTimer()
			b.ReportAllocs()
			for b.Loop() {
				samples, _ := LoadBorderedSamplePlane(scratch, plane, bytesPerSample, 32, 32, 64)
				_ = StoreBorderedSamplePlaneRectTrusted(dst, bytesPerSample, samples, 0, 0, samples.Width, samples.Height)
			}
		})
	}
}

func samplePlaneBenchmarkName(bytesPerSample int) string {
	if bytesPerSample == 1 {
		return "8bit"
	}
	return "16bit"
}

func getTestPlaneSample(plane Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setTestPlaneSample(plane Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func TestExtendBordersReplicatesEdges(t *testing.T) {
	for _, tc := range []struct {
		name           string
		bitDepth       uint8
		bytesPerSample int
	}{
		{"8bit", 8, 1},
		{"10bit", 10, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Width/Height 17x9 with 4:2:0 and SB size 8 (via Align only sets
			// stride; SBSizeLog2 defaults to 6 -> 64). Use a small crop inside a
			// 64-aligned allocation so the padding region is large.
			format := Format{
				Width:        17,
				Height:       9,
				BitDepth:     tc.bitDepth,
				SubsamplingX: true,
				SubsamplingY: true,
				Align:        1,
			}
			layout, err := RequiredSize(format)
			if err != nil {
				t.Fatal(err)
			}
			buf := make([]byte, layout.Size)
			f, err := Bind(buf, format)
			if err != nil {
				t.Fatal(err)
			}

			// Fill each plane's cropped region with a distinguishable gradient.
			fill := func(p Plane) {
				for y := 0; y < p.Height; y++ {
					for x := 0; x < p.Width; x++ {
						setTestPlaneSample(p, tc.bytesPerSample, x, y, uint16((x*7+y*3)&0x3ff))
					}
				}
			}
			fill(f.Y)
			fill(f.U)
			fill(f.V)

			f.ExtendBorders()

			check := func(p Plane) {
				allocWidth := p.Stride / tc.bytesPerSample
				allocHeight := len(p.Pix) / p.Stride
				// Right edge: padding columns equal the last valid column.
				for y := 0; y < p.Height; y++ {
					edge := getTestPlaneSample(p, tc.bytesPerSample, p.Width-1, y)
					for x := p.Width; x < allocWidth; x++ {
						if got := getTestPlaneSample(p, tc.bytesPerSample, x, y); got != edge {
							t.Fatalf("right pad (%d,%d)=%d want %d", x, y, got, edge)
						}
					}
				}
				// Bottom edge (incl. corner): padding rows equal the last valid
				// row, including its right-extended columns.
				for y := p.Height; y < allocHeight; y++ {
					for x := range allocWidth {
						want := getTestPlaneSample(p, tc.bytesPerSample, x, p.Height-1)
						if got := getTestPlaneSample(p, tc.bytesPerSample, x, y); got != want {
							t.Fatalf("bottom pad (%d,%d)=%d want %d", x, y, got, want)
						}
					}
				}
			}
			check(f.Y)
			check(f.U)
			check(f.V)
		})
	}
}

func TestExtendBordersNoopWhenFilled(t *testing.T) {
	// A frame whose cropped extent equals its allocation must be untouched.
	format := Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 1}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, layout.Size)
	for i := range buf {
		buf[i] = byte(i * 31)
	}
	want := append([]byte(nil), buf...)
	f, err := Bind(buf, format)
	if err != nil {
		t.Fatal(err)
	}
	f.ExtendBorders()
	for i := range buf {
		if buf[i] != want[i] {
			t.Fatalf("byte %d changed: got %d want %d", i, buf[i], want[i])
		}
	}
}

func BenchmarkExtendBorders8BitPartialSuperblock(b *testing.B) {
	format := Format{Width: 160, Height: 128, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, layout.Size)
	f, err := Bind(buf, format)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		f.ExtendBorders()
	}
}

func BenchmarkExtendBorders10BitPartialSuperblock(b *testing.B) {
	format := Format{Width: 160, Height: 128, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, layout.Size)
	f, err := Bind(buf, format)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		f.ExtendBorders()
	}
}

// TestExtendBorderRowsThenFrom: filling the right padding of the top rows
// first (ExtendBorderRows) and the rest later (ExtendBordersFrom) gives the
// bytes ExtendBorders gives, and the second pass writes nothing in the rows
// the first one finished, which other goroutines may already be reading.
func TestExtendBorderRowsThenFrom(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bitDepth uint8
		ssX, ssY bool
		mono     bool
	}{
		{"420-8bit", 8, true, true, false},
		{"422-8bit", 8, true, false, false},
		{"444-8bit", 8, false, false, false},
		{"mono-8bit", 8, true, true, true},
		{"420-10bit", 10, true, true, false},
		{"444-12bit", 12, false, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			format := Format{Width: 70, Height: 75, BitDepth: tc.bitDepth, SubsamplingX: tc.ssX, SubsamplingY: tc.ssY, MonoChrome: tc.mono, Align: 1}
			layout, err := RequiredSize(format)
			if err != nil {
				t.Fatal(err)
			}
			base := make([]byte, layout.Size)
			for i := range base {
				base[i] = byte(i*13 + i>>7)
			}
			want := append([]byte(nil), base...)
			wf, err := Bind(want, format)
			if err != nil {
				t.Fatal(err)
			}
			wf.ExtendBorders()

			for _, k := range []int{0, 32, 64, 74} {
				buf := append([]byte(nil), base...)
				f, err := Bind(buf, format)
				if err != nil {
					t.Fatal(err)
				}
				f.ExtendBorderRows(0, k)
				// Poison the finished rows: the second pass must not touch them.
				planes := []*Plane{&f.Y, &f.U, &f.V}
				if tc.mono {
					planes = planes[:1]
				}
				rowsOf := func(i int) int {
					if i > 0 && tc.ssY {
						return k >> 1
					}
					return k
				}
				saved := make([][]byte, len(planes))
				for i, p := range planes {
					n := rowsOf(i) * p.Stride
					saved[i] = append([]byte(nil), p.Pix[:n]...)
					for j := range n {
						p.Pix[j] ^= 0xFF
					}
				}
				f.ExtendBordersFrom(k)
				for i, p := range planes {
					n := rowsOf(i) * p.Stride
					for j := range n {
						if p.Pix[j] != saved[i][j]^0xFF {
							t.Fatalf("k=%d plane %d: finished row byte %d rewritten", k, i, j)
						}
					}
					copy(p.Pix[:n], saved[i])
				}
				for i := range buf {
					if buf[i] != want[i] {
						t.Fatalf("k=%d: byte %d = %d, ExtendBorders gives %d", k, i, buf[i], want[i])
					}
				}
			}
		})
	}
}

// TestFillBytes: every length from 0 to 100 at every alignment and the
// bytes around the run are what the scalar fill leaves.
func TestFillBytes(t *testing.T) {
	const guard = 8
	for n := 0; n <= 100; n++ {
		for off := 0; off < 4; off++ {
			for _, v := range []byte{0, 1, 0x7f, 0x80, 0xff} {
				got := make([]byte, guard+off+n+guard)
				want := make([]byte, len(got))
				for i := range got {
					got[i] = byte(0xa5 ^ i)
					want[i] = got[i]
				}
				FillBytes(got[guard+off:guard+off+n], v)
				fillBytesScalar(want[guard+off:guard+off+n], v)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n=%d off=%d v=%#x: byte %d got %#x want %#x", n, off, v, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkFillBytes80(b *testing.B) {
	buf := make([]byte, 720*1440)
	b.SetBytes(720 * 160)
	for i := 0; i < b.N; i++ {
		for r := 0; r < 720; r++ {
			row := buf[r*1440 : r*1440+1440]
			FillBytes(row[:80], byte(r))
			FillBytes(row[1360:], byte(r+1))
		}
	}
}

func BenchmarkFillBytes80Scalar(b *testing.B) {
	buf := make([]byte, 720*1440)
	b.SetBytes(720 * 160)
	for i := 0; i < b.N; i++ {
		for r := 0; r < 720; r++ {
			row := buf[r*1440 : r*1440+1440]
			fillBytesScalar(row[:80], byte(r))
			fillBytesScalar(row[1360:], byte(r+1))
		}
	}
}

// layerPoolFixture owns caller-side storage for a small LayerPool and its
// on-demand sub-pools. backings retains the per-format byte slices so that GC
// cannot reclaim them while a sub-pool is in use.
type layerPoolFixture struct {
	subPools     []Pool
	subFormats   []Format
	subBound     []bool
	backings     [][]byte
	framesArenas [][]Frame
	freeArenas   [][]int
	usedArenas   [][]bool
	allocs       int
}

func (f *layerPoolFixture) factory(format Format) (Pool, error) {
	layout, err := RequiredSize(format)
	if err != nil {
		return Pool{}, err
	}
	count := 4
	backing := make([]byte, layout.Size*count)
	frames := make([]Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := BindPool(backing, format, frames, free, used)
	if err != nil {
		return Pool{}, err
	}
	f.backings = append(f.backings, backing)
	f.framesArenas = append(f.framesArenas, frames)
	f.freeArenas = append(f.freeArenas, free)
	f.usedArenas = append(f.usedArenas, used)
	f.allocs++
	return pool, nil
}

func newLayerPoolFixture(capacity int) (*layerPoolFixture, LayerPool, error) {
	f := &layerPoolFixture{
		subPools:   make([]Pool, capacity),
		subFormats: make([]Format, capacity),
		subBound:   make([]bool, capacity),
	}
	pool, err := BindLayerPool(f.subPools, f.subFormats, f.subBound, 256, LayerFactoryFunc(f.factory))
	if err != nil {
		return nil, LayerPool{}, err
	}
	return f, pool, nil
}

func TestLayerPoolSingleResolution(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	format := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	id0, frame0, err := pool.Acquire(format)
	if err != nil {
		t.Fatalf("Acquire layer-0 #0: %v", err)
	}
	if id0 < 0 || frame0 == nil {
		t.Fatalf("id=%d frame=%p", id0, frame0)
	}
	if pool.Len() != 1 || fixture.allocs != 1 {
		t.Fatalf("len=%d allocs=%d after first Acquire", pool.Len(), fixture.allocs)
	}

	id1, frame1, err := pool.Acquire(format)
	if err != nil {
		t.Fatalf("Acquire layer-0 #1: %v", err)
	}
	if id1 == id0 || frame1 == nil || frame1 == frame0 {
		t.Fatalf("id1=%d id0=%d frame1=%p frame0=%p", id1, id0, frame1, frame0)
	}
	if pool.Len() != 1 || fixture.allocs != 1 {
		t.Fatalf("second Acquire bound a new sub-pool: len=%d allocs=%d", pool.Len(), fixture.allocs)
	}

	got0, err := pool.Frame(id0)
	if err != nil || got0 != frame0 {
		t.Fatalf("Frame(id0)=%p err=%v want %p", got0, err, frame0)
	}
	got1, err := pool.Frame(id1)
	if err != nil || got1 != frame1 {
		t.Fatalf("Frame(id1)=%p err=%v want %p", got1, err, frame1)
	}

	if err := pool.Release(id0); err != nil {
		t.Fatalf("Release id0: %v", err)
	}
	if err := pool.Release(id1); err != nil {
		t.Fatalf("Release id1: %v", err)
	}
	if _, err := pool.Frame(id0); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(released id0) err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestLayerPoolTwoSpatialLayers(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	// SBSizeLog2 defaults to 6 after normalization; set it here so the stored
	// (normalized) format compares equal below.
	layer0 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, SBSizeLog2: 6, Align: 64}
	layer1 := Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, SBSizeLog2: 6, Align: 64}

	id0, frame0, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatalf("Acquire layer-0: %v", err)
	}
	if pool.Len() != 1 {
		t.Fatalf("Len after layer-0 Acquire=%d want 1", pool.Len())
	}

	id1, frame1, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatalf("Acquire layer-1: %v", err)
	}
	if pool.Len() != 2 || fixture.allocs != 2 {
		t.Fatalf("Len=%d allocs=%d after layer-1 Acquire", pool.Len(), fixture.allocs)
	}
	if frame0.Format.Width == frame1.Format.Width {
		t.Fatalf("layer-0 and layer-1 share width; got %d / %d", frame0.Format.Width, frame1.Format.Width)
	}

	// Global IDs must encode disjoint sub-pool slots so cross-pool resolution
	// works.
	got0, err := pool.Frame(id0)
	if err != nil || got0 != frame0 {
		t.Fatalf("Frame(layer-0 id)=%p err=%v want %p", got0, err, frame0)
	}
	got1, err := pool.Frame(id1)
	if err != nil || got1 != frame1 {
		t.Fatalf("Frame(layer-1 id)=%p err=%v want %p", got1, err, frame1)
	}
	if got0.Format != layer0 || got1.Format != layer1 {
		t.Fatalf("formats: layer-0=%+v layer-1=%+v", got0.Format, got1.Format)
	}

	// Repeated Acquire of layer-0 must reuse the same sub-pool.
	id0Again, frame0Again, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatalf("Acquire layer-0 again: %v", err)
	}
	if id0Again == id0 || id0Again == id1 {
		t.Fatalf("id0Again=%d collides with prior ids", id0Again)
	}
	if frame0Again == nil || frame0Again == frame0 {
		t.Fatalf("frame0Again did not advance: %p / %p", frame0Again, frame0)
	}
	if pool.Len() != 2 || fixture.allocs != 2 {
		t.Fatalf("layer-0 reuse bound a new sub-pool: len=%d allocs=%d", pool.Len(), fixture.allocs)
	}

	if err := pool.ReleaseMany([]int{id0, id1, id0Again}); err != nil {
		t.Fatalf("ReleaseMany: %v", err)
	}
}

func TestLayerPoolAcquireReleasePingPong(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	layer0 := Format{Width: 320, Height: 180, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	// Warm both sub-pools so that the steady-state acquire/release loop does
	// not call the factory.
	first0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	first1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(first0); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(first1); err != nil {
		t.Fatal(err)
	}
	warmAllocs := fixture.allocs

	// 64 ping-pong iterations: each acquires from both layers, swaps which is
	// released first, and verifies cross-pool resolution holds throughout.
	for iter := range 64 {
		id0, f0, err := pool.Acquire(layer0)
		if err != nil {
			t.Fatalf("iter=%d Acquire layer-0: %v", iter, err)
		}
		id1, f1, err := pool.Acquire(layer1)
		if err != nil {
			t.Fatalf("iter=%d Acquire layer-1: %v", iter, err)
		}
		if got, _ := pool.Frame(id0); got != f0 {
			t.Fatalf("iter=%d Frame(id0)=%p want %p", iter, got, f0)
		}
		if got, _ := pool.Frame(id1); got != f1 {
			t.Fatalf("iter=%d Frame(id1)=%p want %p", iter, got, f1)
		}
		if iter&1 == 0 {
			if err := pool.Release(id0); err != nil {
				t.Fatalf("iter=%d Release id0: %v", iter, err)
			}
			if err := pool.Release(id1); err != nil {
				t.Fatalf("iter=%d Release id1: %v", iter, err)
			}
		} else {
			if err := pool.Release(id1); err != nil {
				t.Fatalf("iter=%d Release id1: %v", iter, err)
			}
			if err := pool.Release(id0); err != nil {
				t.Fatalf("iter=%d Release id0: %v", iter, err)
			}
		}
	}
	if fixture.allocs != warmAllocs {
		t.Fatalf("steady-state allocs grew: warm=%d final=%d", warmAllocs, fixture.allocs)
	}
}

func TestLayerPoolAcquireZeroAllocSteadyState(t *testing.T) {
	fixture, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	layer0 := Format{Width: 320, Height: 180, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	// Warm both sub-pools.
	id0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(id0); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(id1); err != nil {
		t.Fatal(err)
	}
	if fixture.allocs != 2 {
		t.Fatalf("expected 2 factory calls during warm-up, got %d", fixture.allocs)
	}

	allocs := testing.AllocsPerRun(200, func() {
		id, _, err := pool.Acquire(layer0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Frame(id); err != nil {
			t.Fatal(err)
		}
		if err := pool.Release(id); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("steady-state Acquire/Release allocated: %f", allocs)
	}
}

func TestLayerPoolReleaseManyRejectsAtomically(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	id0, _, err := pool.Acquire(format)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(format)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.ReleaseMany([]int{id0, id0}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("duplicate err=%v want %v", err, ErrInvalidSlot)
	}
	// Both IDs must remain acquired after atomic rejection.
	if _, err := pool.Frame(id0); err != nil {
		t.Fatalf("id0 lost after duplicate rejection: %v", err)
	}
	if _, err := pool.Frame(id1); err != nil {
		t.Fatalf("id1 lost after duplicate rejection: %v", err)
	}

	if err := pool.ReleaseMany([]int{id0, -1}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("negative id err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(id0); err != nil {
		t.Fatalf("id0 lost after invalid rejection: %v", err)
	}
	if err := pool.ReleaseMany([]int{id0, id1}); err != nil {
		t.Fatalf("valid ReleaseMany: %v", err)
	}
}

func TestLayerPoolRejectsInvalidGlobalID(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Frame(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(-1) err=%v want %v", err, ErrInvalidSlot)
	}
	if err := pool.Release(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Release(-1) err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(1024); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("Frame(unbound sub-pool) err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestLayerPoolFactoryFormatMismatch(t *testing.T) {
	wrong := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	subPools := make([]Pool, 1)
	subFormats := make([]Format, 1)
	subBound := make([]bool, 1)
	pool, err := BindLayerPool(subPools, subFormats, subBound, 256, LayerFactoryFunc(func(format Format) (Pool, error) {
		layout, err := RequiredSize(wrong)
		if err != nil {
			return Pool{}, err
		}
		backing := make([]byte, layout.Size*1)
		frames := make([]Frame, 1)
		free := make([]int, 1)
		used := make([]bool, 1)
		return BindPool(backing, wrong, frames, free, used)
	}))
	if err != nil {
		t.Fatal(err)
	}
	requested := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(requested); !errors.Is(err, ErrInvalidLayerFactory) {
		t.Fatalf("Acquire mismatched factory err=%v want %v", err, ErrInvalidLayerFactory)
	}
}

func TestLayerPoolFull(t *testing.T) {
	_, pool, err := newLayerPoolFixture(1)
	if err != nil {
		t.Fatal(err)
	}
	a := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	b := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pool.Acquire(b); !errors.Is(err, ErrLayerPoolFull) {
		t.Fatalf("second distinct format err=%v want %v", err, ErrLayerPoolFull)
	}
}

func TestLayerPoolReset(t *testing.T) {
	_, pool, err := newLayerPoolFixture(2)
	if err != nil {
		t.Fatal(err)
	}
	a := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	b := Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	if _, _, err := pool.Acquire(a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pool.Acquire(b); err != nil {
		t.Fatal(err)
	}
	pool.Reset()
	sub, err := pool.SubPool(a)
	if err != nil {
		t.Fatal(err)
	}
	if sub.Available() != sub.Cap() {
		t.Fatalf("layer-a not reset: avail=%d cap=%d", sub.Available(), sub.Cap())
	}
}

func TestBindLayerPoolRejectsBadStorage(t *testing.T) {
	pools := make([]Pool, 2)
	formats := make([]Format, 1)
	bound := make([]bool, 2)
	factory := LayerFactoryFunc(func(format Format) (Pool, error) { return Pool{}, nil })
	if _, err := BindLayerPool(pools, formats, bound, 256, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("mismatched lens err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool(nil, nil, nil, 256, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("empty storage err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool([]Pool{{}}, []Format{{}}, []bool{false}, 0, factory); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("zero stride err=%v want %v", err, ErrInvalidLayerPool)
	}
	if _, err := BindLayerPool([]Pool{{}}, []Format{{}}, []bool{false}, 256, nil); !errors.Is(err, ErrInvalidLayerPool) {
		t.Fatalf("nil factory err=%v want %v", err, ErrInvalidLayerPool)
	}
}

func TestSlicesOverlap(t *testing.T) {
	buf := make([]uint16, 64)
	other := make([]uint16, 64)
	for _, tc := range []struct {
		name string
		a, b []uint16
		want bool
	}{
		{"same", buf, buf, true},
		{"disjoint-adjacent", buf[:10], buf[10:], false},
		{"touching", buf[:11], buf[10:20], true},
		{"inside", buf[5:50], buf[20:30], true},
		{"inside-rev", buf[20:30], buf[5:50], true},
		{"gap", buf[0:10], buf[30:40], false},
		{"other-array", buf, other, false},
		{"empty", buf[:0], buf, false},
		{"end", buf[63:], buf[60:], true},
	} {
		if got := SlicesOverlap(tc.a, tc.b); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}

func TestBindPoolAcquireRelease(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	if pool.Cap() != 2 || pool.Available() != 2 {
		t.Fatalf("pool cap=%d available=%d", pool.Cap(), pool.Available())
	}

	index0, frame0, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index0 != 0 || frame0 == nil {
		t.Fatalf("first acquire index=%d frame=%p", index0, frame0)
	}
	frame0.Y.Pix[0] = 0x33
	if backing[0] != 0x33 {
		t.Fatalf("frame 0 did not bind backing")
	}

	index1, frame1, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index1 != 1 || frame1 == nil {
		t.Fatalf("second acquire index=%d frame=%p", index1, frame1)
	}
	frame1.Y.Pix[0] = 0x44
	if backing[layout.Size] != 0x44 {
		t.Fatalf("frame 1 did not bind backing")
	}

	_, _, err = pool.Acquire()
	if !errors.Is(err, ErrPoolEmpty) {
		t.Fatalf("third acquire err=%v want %v", err, ErrPoolEmpty)
	}

	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}
	index0Again, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index0Again != index0 {
		t.Fatalf("reacquire index=%d want %d", index0Again, index0)
	}
	if err := pool.Release(index0Again); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0Again); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("double release err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestFramePoolFrame(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Frame(0); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("free Frame err=%v want %v", err, ErrInvalidSlot)
	}

	index, acquired, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	acquired.Y.Pix[0] = 0x55

	retained, err := pool.Frame(index)
	if err != nil {
		t.Fatal(err)
	}
	if retained != acquired || retained.Y.Pix[0] != 0x55 {
		t.Fatalf("retained=%p acquired=%p value=%x", retained, acquired, retained.Y.Pix[0])
	}

	if err := pool.Release(index); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Frame(index); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("released Frame err=%v want %v", err, ErrInvalidSlot)
	}
	if _, err := pool.Frame(-1); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("negative Frame err=%v want %v", err, ErrInvalidSlot)
	}
}

func TestFramePoolFormatAndLayout(t *testing.T) {
	// SBSizeLog2 defaults to 6 (64x64) after normalization; the pool stores the
	// normalized format, so the comparison below uses the normalized value.
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, SBSizeLog2: 6, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	poolFormat, err := pool.Format()
	if err != nil {
		t.Fatal(err)
	}
	if poolFormat != format {
		t.Fatalf("format=%+v want %+v", poolFormat, format)
	}
	poolLayout, err := pool.Layout()
	if err != nil {
		t.Fatal(err)
	}
	if poolLayout != layout {
		t.Fatalf("layout=%+v want %+v", poolLayout, layout)
	}
}

func TestFramePoolAcquireFormat(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	index, acquired, err := pool.AcquireFormat(format)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || acquired == nil {
		t.Fatalf("index=%d frame=%p", index, acquired)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestFramePoolAcquireFormatRejectsMismatch(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = pool.AcquireFormat(Format{Width: 80, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	if !errors.Is(err, ErrInvalidFormat) {
		t.Fatalf("AcquireFormat err=%v want %v", err, ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("mismatch consumed a slot, available=%d", pool.Available())
	}
}

func TestFramePoolReleaseMany(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*3)
	var frames [3]Frame
	var free [3]int
	var used [3]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index2, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	if err := pool.ReleaseMany([]int{index0, index2}); err != nil {
		t.Fatal(err)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	reacquired2, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if reacquired2 != index2 {
		t.Fatalf("first reacquire=%d want %d", reacquired2, index2)
	}
	reacquired0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if reacquired0 != index0 {
		t.Fatalf("second reacquire=%d want %d", reacquired0, index0)
	}

	if err := pool.ReleaseMany([]int{reacquired2, reacquired0, index1}); err != nil {
		t.Fatal(err)
	}
	if pool.Available() != 3 {
		t.Fatalf("available=%d want 3", pool.Available())
	}
	if err := pool.ReleaseMany(nil); err != nil {
		t.Fatal(err)
	}
}

func TestFramePoolReleaseManyRejectsAtomically(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool

	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	if err := pool.ReleaseMany([]int{index0, index0}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("duplicate release err=%v want %v", err, ErrInvalidSlot)
	}
	if pool.Available() != 0 {
		t.Fatalf("duplicate changed available=%d", pool.Available())
	}
	if _, err := pool.Frame(index0); err != nil {
		t.Fatalf("duplicate released index0: %v", err)
	}

	if err := pool.ReleaseMany([]int{index0, 99}); !errors.Is(err, ErrInvalidSlot) {
		t.Fatalf("invalid release err=%v want %v", err, ErrInvalidSlot)
	}
	if pool.Available() != 0 {
		t.Fatalf("invalid changed available=%d", pool.Available())
	}
	if _, err := pool.Frame(index1); err != nil {
		t.Fatalf("invalid released index1: %v", err)
	}

	if err := pool.ReleaseMany([]int{index0, index1}); err != nil {
		t.Fatal(err)
	}
}

func TestBindPoolRejectsBadStorage(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [1]int
	var used [2]bool

	_, err = BindPool(backing, format, frames[:], free[:], used[:])
	if !errors.Is(err, ErrInvalidPool) {
		t.Fatalf("BindPool short free err=%v want %v", err, ErrInvalidPool)
	}

	var fullFree [2]int
	_, err = BindPool(backing[:layout.Size*2-1], format, frames[:], fullFree[:], used[:])
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("BindPool short backing err=%v want %v", err, ErrShortBuffer)
	}
}

func TestFramePoolReset(t *testing.T) {
	format := Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	pool.Reset()
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	index, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 {
		t.Fatalf("index=%d want 0", index)
	}
}

func TestFramePoolAllocs(t *testing.T) {
	format := Format{Width: 128, Height: 72, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		index0, _, err := pool.AcquireFormat(format)
		if err != nil {
			t.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Frame(index0); err != nil {
			t.Fatal(err)
		}
		var batch [2]int
		batch[0] = index0
		batch[1] = index1
		if err := pool.ReleaseMany(batch[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("pool acquire/release allocated: %f", allocs)
	}
}

func BenchmarkFramePoolAcquireRelease(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size*4)
	var frames [4]Frame
	var free [4]int
	var used [4]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		index, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFramePoolAcquireFormatRelease(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size*4)
	var frames [4]Frame
	var free [4]int
	var used [4]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		index, _, err := pool.AcquireFormat(format)
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFramePoolReleaseMany(b *testing.B) {
	format := Format{Width: 1920, Height: 1080, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	layout, err := RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size*4)
	var frames [4]Frame
	var free [4]int
	var used [4]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		b.Fatal(err)
	}

	var batch [2]int
	b.ReportAllocs()
	for b.Loop() {
		index0, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		index1, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		batch[0] = index0
		batch[1] = index1
		if err := pool.ReleaseMany(batch[:]); err != nil {
			b.Fatal(err)
		}
	}
}

// TestFramePoolAcquireFormatBindsSmallerFrames checks that a slot sized for
// the pool format takes any frame size that fits, and that the slot is
// rebound to the pool format on a plain Acquire.
func TestFramePoolAcquireFormatBindsSmallerFrames(t *testing.T) {
	format := Format{Width: 192, Height: 108, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, 2*layout.Size)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	small := Format{Width: 140, Height: 79, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	index, f, err := pool.AcquireFormat(small)
	if err != nil {
		t.Fatal(err)
	}
	if f.Format.Width != 140 || f.Format.Height != 79 || f.U.Width != 70 || f.U.Height != 40 {
		t.Fatalf("bound frame %+v", f.Format)
	}
	if got, err := pool.Format(); err != nil || got != pool.frames[1-index].Format {
		t.Fatalf("pool format %+v err=%v", got, err)
	}
	if err := pool.Release(index); err != nil {
		t.Fatal(err)
	}
	index2, f2, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if index2 != index || f2.Format.Width != 192 || f2.Format.Height != 108 {
		t.Fatalf("re-acquired slot %d format %+v", index2, f2.Format)
	}
	if _, _, err := pool.AcquireFormat(Format{Width: 140, Height: 79, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 32}); !errors.Is(err, ErrInvalidFormat) {
		t.Fatalf("10-bit frame larger than the slot: err=%v", err)
	}
}

// TestFramePoolAcquireFormatBindsOtherSampleFormats checks that a slot takes a
// frame of another bit depth, chroma layout or superblock size when its layout
// fits, as a new coded video sequence may change all three.
func TestFramePoolAcquireFormatBindsOtherSampleFormats(t *testing.T) {
	format := Format{Width: 192, Height: 108, BitDepth: 10, Align: 32}
	layout, err := RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	var frames [1]Frame
	var free [1]int
	var used [1]bool
	pool, err := BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []Format{
		{Width: 192, Height: 108, BitDepth: 8, Align: 32},
		{Width: 192, Height: 108, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, SBSizeLog2: 7, Align: 32},
		{Width: 100, Height: 60, BitDepth: 12, SubsamplingX: true, Align: 32},
		{Width: 192, Height: 108, BitDepth: 10, MonoChrome: true, Align: 32},
	} {
		index, got, err := pool.AcquireFormat(f)
		if err != nil {
			t.Fatalf("AcquireFormat(%+v): %v", f, err)
		}
		want, err := normalizeFormat(f)
		if err != nil {
			t.Fatal(err)
		}
		if got.Format != want {
			t.Fatalf("bound %+v, want %+v", got.Format, want)
		}
		if err := pool.Release(index); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := pool.AcquireFormat(Format{Width: 192, Height: 108, BitDepth: 10, Align: 64}); !errors.Is(err, ErrInvalidFormat) {
		t.Fatalf("other stride alignment: err=%v", err)
	}
	if pool.Available() != 1 {
		t.Fatalf("rejected format consumed a slot, available=%d", pool.Available())
	}
}

// refLoadSampleRows16 is the byte-exact scalar reference for the 16-bit sample
// staging load, independent of the build-tag-selected loadSampleRows16 (the
// little-endian per-row memmove on samples_le.go arches, the scalar loop on
// samples_generic.go arches). The differential tests below assert the active
// implementation is bit-identical to this reference.
func refLoadSampleRows16(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	srcOff := 0
	dstOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		srcLine := src[srcOff : srcOff+rowBytes : srcOff+rowBytes]
		dstLine := dst[dstOff : dstOff+width : dstOff+width]
		for x := 0; x < width; x++ {
			off := x * 2
			dstLine[x] = uint16(srcLine[off]) | uint16(srcLine[off+1])<<8
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}

// refStoreSampleRows16 is the byte-exact scalar reference for the 16-bit sample
// staging store.
func refStoreSampleRows16(dst []byte, dstStride int, src []uint16, srcStride int, width int, height int) {
	dstOff := 0
	srcOff := 0
	rowBytes := width * 2
	for y := 0; y < height; y++ {
		dstLine := dst[dstOff : dstOff+rowBytes : dstOff+rowBytes]
		srcLine := src[srcOff : srcOff+width : srcOff+width]
		for x, sample := range srcLine {
			off := x * 2
			dstLine[off] = byte(sample)
			dstLine[off+1] = byte(sample >> 8)
		}
		dstOff += dstStride
		srcOff += srcStride
	}
}

// TestLoadSampleRows16MatchesScalar fuzzes the active loadSampleRows16 against
// the scalar reference over randomized widths, heights, and strides, including
// non-contiguous rows whose stride exceeds the visible width so the untouched
// stride padding is exercised. The whole destination buffer (padding included)
// must stay byte-identical to the reference run.
func TestLoadSampleRows16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x10adbeef))
	const sentinel = uint16(0xa5c3)
	for iter := 0; iter < 4000; iter++ {
		width := rng.Intn(65)         // 0..64 visible samples
		height := rng.Intn(49)        // 0..48 rows
		dstPad := rng.Intn(9)         // extra dst samples of stride padding
		srcPad := rng.Intn(17)        // extra src bytes of stride padding
		dstStride := width + dstPad   // samples
		srcStride := width*2 + srcPad // bytes

		srcLen := height*srcStride + width*2 + 8
		src := make([]byte, srcLen)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}

		dstLen := height*dstStride + width + 8
		got := make([]uint16, dstLen)
		want := make([]uint16, dstLen)
		for i := range got {
			got[i] = sentinel
			want[i] = sentinel
		}

		loadSampleRows16(got, dstStride, src, srcStride, width, height)
		refLoadSampleRows16(want, dstStride, src, srcStride, width, height)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("iter=%d w=%d h=%d dstStride=%d srcStride=%d: dst[%d]=%#x want %#x",
					iter, width, height, dstStride, srcStride, i, got[i], want[i])
			}
		}
	}
}

// TestStoreSampleRows16MatchesScalar mirrors the load test for the store path,
// spanning the full 16-bit sample range so both serialized bytes are exercised.
func TestStoreSampleRows16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5701e12e))
	const sentinel = byte(0x5a)
	for iter := 0; iter < 4000; iter++ {
		width := rng.Intn(65)
		height := rng.Intn(49)
		srcPad := rng.Intn(9)         // extra src samples of stride padding
		dstPad := rng.Intn(17)        // extra dst bytes of stride padding
		srcStride := width + srcPad   // samples
		dstStride := width*2 + dstPad // bytes

		srcLen := height*srcStride + width + 8
		src := make([]uint16, srcLen)
		for i := range src {
			src[i] = uint16(rng.Intn(65536))
		}

		dstLen := height*dstStride + width*2 + 8
		got := make([]byte, dstLen)
		want := make([]byte, dstLen)
		for i := range got {
			got[i] = sentinel
			want[i] = sentinel
		}

		storeSampleRows16(got, dstStride, src, srcStride, width, height)
		refStoreSampleRows16(want, dstStride, src, srcStride, width, height)

		if !bytes.Equal(got, want) {
			t.Fatalf("iter=%d w=%d h=%d dstStride=%d srcStride=%d: store mismatch",
				iter, width, height, dstStride, srcStride)
		}
	}
}

// benchSampleRows16 sizes a representative 1080p luma plane with a little
// stride padding so the per-row copy path (not one big contiguous copy) is
// exercised, matching how the decoder stages one plane at a time.
const (
	benchW      = 1920
	benchH      = 1080
	benchDstPad = 8
)

func BenchmarkLoadSampleRows16Fast(b *testing.B) {
	dstStride := benchW + benchDstPad
	srcStride := benchW * 2
	src := make([]byte, benchH*srcStride)
	dst := make([]uint16, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

func BenchmarkStoreSampleRows16Fast(b *testing.B) {
	srcStride := benchW + benchDstPad
	dstStride := benchW * 2
	src := make([]uint16, benchH*srcStride)
	dst := make([]byte, benchH*dstStride)
	b.SetBytes(int64(benchW * benchH * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSampleRows16(dst, dstStride, src, srcStride, benchW, benchH)
	}
}

// TestSampleRows16ZeroAlloc pins that neither staging helper allocates: the
// unsafe row view on little-endian arches must not force dst/src to escape.
func TestSampleRows16ZeroAlloc(t *testing.T) {
	const width, height = 64, 48
	dstStride := width + 4
	srcStride := width*2 + 6
	src8 := make([]byte, height*srcStride+width*2)
	dst16 := make([]uint16, height*dstStride+width)
	if n := testing.AllocsPerRun(50, func() {
		loadSampleRows16(dst16, dstStride, src8, srcStride, width, height)
	}); n != 0 {
		t.Fatalf("loadSampleRows16 allocated %.0f times, want 0", n)
	}

	srcStrideS := width + 4
	dstStrideB := width*2 + 6
	src16 := make([]uint16, height*srcStrideS+width)
	dst8 := make([]byte, height*dstStrideB+width*2)
	if n := testing.AllocsPerRun(50, func() {
		storeSampleRows16(dst8, dstStrideB, src16, srcStrideS, width, height)
	}); n != 0 {
		t.Fatalf("storeSampleRows16 allocated %.0f times, want 0", n)
	}
}

// TestLoadSamplePlaneRowsMatchesFull checks that loading a plane in bands
// fills the same snapshot as LoadSamplePlaneFull.
func TestLoadSamplePlaneRowsMatchesFull(t *testing.T) {
	for _, bps := range []int{1, 2} {
		const w, h = 37, 23
		stride := (w + 5) * bps
		src := Plane{Pix: make([]byte, stride*h), Stride: stride, Width: w, Height: h}
		for i := range src.Pix {
			src.Pix[i] = byte(i*7 + i>>3)
		}
		if bps == 2 {
			for i := 1; i < len(src.Pix); i += 2 {
				src.Pix[i] &= 3
			}
		}
		n, err := SamplePlaneLen(src, bps)
		if err != nil {
			t.Fatal(err)
		}
		full := make([]uint16, n)
		if _, _, err := LoadSamplePlaneFull(full, src, bps); err != nil {
			t.Fatal(err)
		}
		banded := make([]uint16, n)
		dst, err := BindSamplePlane(banded, src, bps)
		if err != nil {
			t.Fatal(err)
		}
		for _, y := range [][2]int{{0, 5}, {5, 6}, {6, 17}, {17, h}} {
			if err := LoadSamplePlaneRows(dst, src, bps, y[0], y[1]); err != nil {
				t.Fatal(err)
			}
		}
		for i := range full {
			if full[i] != banded[i] {
				t.Fatalf("bps=%d: sample %d = %d, want %d", bps, i, banded[i], full[i])
			}
		}
	}
}

// TestLoadSamplePlaneFullLoadsPastVisibleColumns covers the new variant
// introduced for postfilter passes (CDEF) whose superblock-edge reads extend
// past the visible plane width into the row-stride padding columns written by
// the reconstruction stage. The default LoadSamplePlane leaves those columns at
// whatever value dst already held; LoadSamplePlaneFull must instead populate
// every column up to Stride/bytesPerSample with the actual byte values.
func TestLoadSamplePlaneFullLoadsPastVisibleColumns8Bit(t *testing.T) {
	// Width=6 visible, Stride=8 bytes → 2 past-visible columns per row.
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 6, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Stride; x++ {
			// Encode (row, col) so we can verify past-visible columns weren't
			// confused with visible ones.
			plane.Pix[y*plane.Stride+x] = byte(10*y + x)
		}
	}

	need, err := SamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, need)
	for i := range scratch {
		scratch[i] = 0xefef
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(scratch, plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 8 {
		t.Fatalf("loadedWidth=%d want 8", loadedWidth)
	}
	if samples.Stride != 8 || samples.Width != 6 || samples.Height != 3 {
		t.Fatalf("samples=%+v", samples)
	}
	for y := 0; y < samples.Height; y++ {
		for x := range loadedWidth {
			if got, want := samples.Pix[y*samples.Stride+x], uint16(10*y+x); got != want {
				t.Fatalf("row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullLoadsPastVisibleColumns16Bit verifies the same
// extension for little-endian 16-bit byte planes (10-bit / 12-bit content).
func TestLoadSamplePlaneFullLoadsPastVisibleColumns16Bit(t *testing.T) {
	// Width=3 visible samples, Stride=12 bytes → strideSamples=6, 3 past-visible
	// samples per row.
	plane := Plane{Pix: make([]byte, 12*2), Stride: 12, Width: 3, Height: 2}
	for y := 0; y < plane.Height; y++ {
		for s := 0; s < plane.Stride/2; s++ {
			sample := uint16(1000 + y*10 + s)
			plane.Pix[y*plane.Stride+s*2] = byte(sample)
			plane.Pix[y*plane.Stride+s*2+1] = byte(sample >> 8)
		}
	}

	need, err := SamplePlaneLen(plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, need), plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 6 {
		t.Fatalf("loadedWidth=%d want 6", loadedWidth)
	}
	if samples.Stride != 6 || samples.Width != 3 || samples.Height != 2 {
		t.Fatalf("samples=%+v", samples)
	}
	for y := 0; y < samples.Height; y++ {
		for s := range loadedWidth {
			if got, want := samples.Pix[y*samples.Stride+s], uint16(1000+y*10+s); got != want {
				t.Fatalf("row=%d sample=%d got=%d want=%d", y, s, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullPreservesWidth pins the contract that Width remains
// the visible plane width even though the loaded data extends past it.
// Callers (such as the CDEF postfilter) rely on Width reporting the visible
// extent so the filtered output never writes past it.
func TestLoadSamplePlaneFullPreservesWidth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 16*2), Stride: 16, Width: 10, Height: 2}
	for i := range plane.Pix {
		plane.Pix[i] = byte(i)
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, 16*2), plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Width != 10 {
		t.Fatalf("Width=%d want 10 (visible)", samples.Width)
	}
	if loadedWidth != 16 {
		t.Fatalf("loadedWidth=%d want 16 (stride samples)", loadedWidth)
	}
}

// TestLoadSamplePlaneFullStrideEqualsWidth covers the degenerate layout where
// the plane has no row-stride padding columns: loadedWidth must equal Width and
// every visible sample must still be loaded correctly.
func TestLoadSamplePlaneFullStrideEqualsWidth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 5*3), Stride: 5, Width: 5, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte(7*y + x)
		}
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, 15), plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 5 || samples.Stride != 5 || samples.Width != 5 {
		t.Fatalf("loadedWidth=%d samples=%+v", loadedWidth, samples)
	}
	for y := 0; y < samples.Height; y++ {
		for x := 0; x < samples.Width; x++ {
			if got, want := samples.Pix[y*samples.Stride+x], uint16(7*y+x); got != want {
				t.Fatalf("row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullEmptyPlane keeps the empty-plane contract aligned
// with LoadSamplePlane: zero-sized planes yield an empty samples view and
// loadedWidth=0 without scanning Pix.
func TestLoadSamplePlaneFullEmptyPlane(t *testing.T) {
	samples, loadedWidth, err := LoadSamplePlaneFull(nil, Plane{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 0 || samples.Width != 0 || samples.Height != 0 || samples.Stride != 0 {
		t.Fatalf("loadedWidth=%d samples=%+v", loadedWidth, samples)
	}
}

// TestLoadSamplePlaneFullShortBuffer rejects scratch shorter than the
// caller-owned layout reported by SamplePlaneLen.
func TestLoadSamplePlaneFullShortBuffer(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*2), Stride: 8, Width: 4, Height: 2}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 3), plane, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("err=%v want ErrShortBuffer", err)
	}
}

// TestLoadSamplePlaneFullShortPixForFinalRow rejects planes whose Pix slice is
// shorter than Height*Stride, which would otherwise let the past-visible read
// of the final row index out of bounds.
func TestLoadSamplePlaneFullShortPixForFinalRow(t *testing.T) {
	// Width=4, Stride=8, Height=2 needs 16 bytes; the truncated buffer holds
	// only the bytes through the last visible sample (1*8 + 4 = 12), which is
	// what samplePlaneLayout allows for LoadSamplePlane but is too short for
	// LoadSamplePlaneFull's past-visible read on row 1.
	plane := Plane{Pix: make([]byte, 12), Stride: 8, Width: 4, Height: 2}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 16), plane, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("err=%v want ErrInvalidPlane", err)
	}
}

// TestLoadSamplePlaneFullRejectsInvalidBytesPerSample mirrors LoadSamplePlane's
// validation so callers can substitute the variants without changing error
// semantics.
func TestLoadSamplePlaneFullRejectsInvalidBytesPerSample(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8), Stride: 8, Width: 4, Height: 1}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 8), plane, 3); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("err=%v want ErrInvalidPlane", err)
	}
}
