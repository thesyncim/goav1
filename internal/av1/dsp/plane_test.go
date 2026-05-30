package dsp

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestFillPlaneBlock8Bit(t *testing.T) {
	plane, _ := testPlane(7, 5, 1, 12)
	for i := range plane.Pix {
		plane.Pix[i] = 0xee
	}
	if err := FillPlaneBlock(plane, 1, 2, 1, 3, 2, 0x2a); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			got := plane.Pix[y*plane.Stride+x]
			want := byte(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = 0x2a
			}
			if got != want {
				t.Fatalf("sample(%d,%d)=%x want %x", x, y, got, want)
			}
		}
	}
}

func TestCopyPlaneBlockHighBitDepth(t *testing.T) {
	src, _ := testPlane(6, 5, 2, 16)
	dst, _ := testPlane(7, 4, 2, 16)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	if err := CopyPlaneBlock(dst, src, 2, 2, 1, 1, 2, 3, 2); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getSample(dst, 2, 2+x, 1+y)
			want := uint16(1000 + (2+y)*10 + 1 + x)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestAddResidualPlaneBlock8BitClips(t *testing.T) {
	plane, _ := testPlane(4, 2, 1, 6)
	if err := FillPlaneBlock(plane, 1, 0, 0, 4, 2, 100); err != nil {
		t.Fatal(err)
	}
	residual := []int16{
		-200, -1, 0, 200,
		20, -20, 155, 156,
	}
	if err := AddResidualPlaneBlock(plane, 1, 8, 0, 0, 4, 2, residual, 4); err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 99, 100, 255, 120, 80, 255, 255}
	for y := range 2 {
		for x := range 4 {
			got := getSample(plane, 1, x, y)
			if got != want[y*4+x] {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want[y*4+x])
			}
		}
	}
}

func TestAddResidualPlaneBlockHighBitDepthClips(t *testing.T) {
	plane, _ := testPlane(3, 2, 2, 8)
	if err := FillPlaneBlock(plane, 2, 0, 0, 3, 2, 900); err != nil {
		t.Fatal(err)
	}
	residual := []int16{
		-1000, -1, 0,
		100, 123, 2000,
	}
	if err := AddResidualPlaneBlock(plane, 2, 10, 0, 0, 3, 2, residual, 3); err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 899, 900, 1000, 1023, 1023}
	for y := range 2 {
		for x := range 3 {
			got := getSample(plane, 2, x, y)
			if got != want[y*3+x] {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want[y*3+x])
			}
		}
	}
}

// TestAddResidualPlaneBlock10BitHighQRegression pins the 10-bit residual
// addition path to libaom's highbd_clip_pixel_add semantics for q63-style
// inputs: a DC predictor at the 10-bit midpoint (512) combined with the
// large signed residuals produced by high-QP dequant + IDCT for 10-bit
// content. Specifically:
//   - predictor=512 (1<<(bd-1)) is the AV1 fallback for first-block intra DC
//   - residuals span the full int16 range; the result must clip to [0,1023]
//     not [0,255], and must preserve the high byte of the LE uint16 sample
//
// This guards against historical "samples ~4-5x too low" regressions where
// the high BD residual write path accidentally clipped at 0xff or stored
// only the low byte of the LE sample.
func TestAddResidualPlaneBlock10BitHighQRegression(t *testing.T) {
	const (
		width   = 8
		height  = 8
		bd      = 10
		predDC  = 1 << (bd - 1) // 512: AV1 default DC predictor for 10-bit
		bdClip  = (1 << bd) - 1 // 1023
		strideB = width * 2
	)
	plane, _ := testPlane(width, height, 2, strideB)
	if err := FillPlaneBlock(plane, 2, 0, 0, width, height, predDC); err != nil {
		t.Fatal(err)
	}
	// Residuals chosen to span: small positive, small negative, large positive
	// (must clip), large negative (must clip), and exact in-range values.
	// At q63 10-bit, real IDCT residuals can hit ~+/-2048 after the bd+6
	// column clamp and 4-bit round-shift.
	residual := make([]int16, width*height)
	for i := range residual {
		switch i % 8 {
		case 0:
			residual[i] = 0 // identity case
		case 1:
			residual[i] = 1 // smallest positive
		case 2:
			residual[i] = -1 // smallest negative
		case 3:
			residual[i] = 200 // small positive: 512+200=712, in-range
		case 4:
			residual[i] = -300 // 512-300=212, in-range
		case 5:
			residual[i] = 1000 // 512+1000=1512 -> must clip to 1023
		case 6:
			residual[i] = -1000 // 512-1000=-488 -> must clip to 0
		case 7:
			residual[i] = 2047 // largest q63-realistic positive -> clip to 1023
		}
	}
	if err := AddResidualPlaneBlock(plane, 2, bd, 0, 0, width, height, residual, width); err != nil {
		t.Fatal(err)
	}
	want := []uint16{predDC, predDC + 1, predDC - 1, predDC + 200, predDC - 300, bdClip, 0, bdClip}
	for y := range height {
		for x := range width {
			got := getSample(plane, 2, x, y)
			expected := want[x%8]
			if got != expected {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, expected)
			}
		}
	}
	// Verify the high byte of every sample is the correct upper bits of the
	// 10-bit value (catches a "store only low byte" regression).
	for y := range height {
		for x := range width {
			off := y*plane.Stride + x*2
			lo := plane.Pix[off]
			hi := plane.Pix[off+1]
			expected := want[x%8]
			if uint16(lo)|uint16(hi)<<8 != expected {
				t.Fatalf("sample(%d,%d) LE bytes=(lo=%d,hi=%d) reconstruct=%d want %d",
					x, y, lo, hi, uint16(lo)|uint16(hi)<<8, expected)
			}
		}
	}
}

// TestAddResidualPlaneBlock10BitClipBoundary pins that AddResidualPlaneBlock
// applies the bit-depth-aware clip (1023 for bd=10) and not the byte clip
// (255). A predictor right at 255 + a residual of 100 must produce 355 in
// 10-bit, not 255.
func TestAddResidualPlaneBlock10BitClipBoundary(t *testing.T) {
	plane, _ := testPlane(4, 1, 2, 8)
	if err := FillPlaneBlock(plane, 2, 0, 0, 4, 1, 255); err != nil {
		t.Fatal(err)
	}
	residual := []int16{100, 200, 768, 769}
	if err := AddResidualPlaneBlock(plane, 2, 10, 0, 0, 4, 1, residual, 4); err != nil {
		t.Fatal(err)
	}
	// 255+100=355, 255+200=455, 255+768=1023, 255+769=1024 -> clip to 1023.
	want := []uint16{355, 455, 1023, 1023}
	for x := range 4 {
		got := getSample(plane, 2, x, 0)
		if got != want[x] {
			t.Fatalf("sample(%d,0)=%d want %d", x, got, want[x])
		}
	}
}

func TestPlaneBlockRejectsInvalidInputs(t *testing.T) {
	plane, _ := testPlane(4, 4, 1, 4)
	if err := FillPlaneBlock(plane, 3, 0, 0, 1, 1, 0); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("invalid sample width err=%v want %v", err, ErrInvalidBlock)
	}
	if err := FillPlaneBlock(plane, 1, 0, 0, 1, 1, 256); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("invalid value err=%v want %v", err, ErrInvalidBlock)
	}
	if err := FillPlaneBlock(plane, 1, -1, 0, 1, 1, 0); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("negative x err=%v want %v", err, ErrInvalidBlock)
	}
	if err := CopyPlaneBlock(plane, plane, 1, 3, 0, 0, 0, 2, 1); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("outside copy err=%v want %v", err, ErrInvalidBlock)
	}
	if err := AddResidualPlaneBlock(plane, 1, 10, 0, 0, 1, 1, []int16{0}, 1); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("bitdepth mismatch err=%v want %v", err, ErrInvalidBlock)
	}
	if err := AddResidualPlaneBlock(plane, 1, 8, 0, 0, 2, 2, []int16{0, 1, 2}, 2); !errors.Is(err, ErrInvalidBlock) {
		t.Fatalf("short residual err=%v want %v", err, ErrInvalidBlock)
	}
}

func TestPlaneBlockAllocs(t *testing.T) {
	src, _ := testPlane(16, 16, 1, 32)
	dst, _ := testPlane(16, 16, 1, 32)
	residual := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := FillPlaneBlock(src, 1, 0, 0, 16, 16, 91); err != nil {
			t.Fatal(err)
		}
		if err := CopyPlaneBlock(dst, src, 1, 0, 0, 0, 0, 16, 16); err != nil {
			t.Fatal(err)
		}
		if err := AddResidualPlaneBlock(dst, 1, 8, 0, 0, 16, 16, residual, 16); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("plane block operations allocated: %f", allocs)
	}
}

func FuzzPlaneBlockOps(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(1), uint8(0), uint8(0), uint8(4), uint8(4), int16(0))
	f.Add(uint8(17), uint8(9), uint8(2), uint8(3), uint8(2), uint8(8), uint8(4), int16(200))
	f.Add(uint8(5), uint8(5), uint8(1), uint8(4), uint8(4), uint8(1), uint8(1), int16(-200))

	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawBPS uint8, rawX uint8, rawY uint8, rawBW uint8, rawBH uint8, delta int16) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bytesPerSample := int(rawBPS%2) + 1
		bitDepth := uint8(8)
		fill := uint16(91)
		if bytesPerSample == 2 {
			bitDepth = 10
			fill = 512
		}
		stride := (width + 7) * bytesPerSample
		src, _ := testPlane(width, height, bytesPerSample, stride)
		dst, _ := testPlane(width, height, bytesPerSample, stride)

		x := int(rawX) % width
		y := int(rawY) % height
		blockW := int(rawBW)%width + 1
		blockH := int(rawBH)%height + 1
		if blockW > width-x {
			blockW = width - x
		}
		if blockH > height-y {
			blockH = height - y
		}
		residual := make([]int16, blockW*blockH)
		for i := range residual {
			residual[i] = delta
		}

		if err := FillPlaneBlock(src, bytesPerSample, x, y, blockW, blockH, fill); err != nil {
			t.Fatalf("FillPlaneBlock err=%v", err)
		}
		if err := CopyPlaneBlock(dst, src, bytesPerSample, x, y, x, y, blockW, blockH); err != nil {
			t.Fatalf("CopyPlaneBlock err=%v", err)
		}
		if err := AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, blockW, blockH, residual, blockW); err != nil {
			t.Fatalf("AddResidualPlaneBlock err=%v", err)
		}

		max := uint16(0xff)
		if bytesPerSample == 2 {
			max = 0x3ff
		}
		want := clipSample(int(fill)+int(delta), max)
		for row := 0; row < blockH; row++ {
			for col := 0; col < blockW; col++ {
				got := getSample(dst, bytesPerSample, x+col, y+row)
				if got != want {
					t.Fatalf("sample=%d want %d", got, want)
				}
			}
		}
	})
}

func BenchmarkFillPlaneBlock(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	b.ReportAllocs()
	for b.Loop() {
		_ = FillPlaneBlock(plane, 1, 0, 0, 64, 64, 128)
	}
}

func BenchmarkCopyPlaneBlock(b *testing.B) {
	src, _ := testPlane(64, 64, 1, 64)
	dst, _ := testPlane(64, 64, 1, 64)
	b.ReportAllocs()
	for b.Loop() {
		_ = CopyPlaneBlock(dst, src, 1, 0, 0, 0, 0, 64, 64)
	}
}

func BenchmarkAddResidualPlaneBlock(b *testing.B) {
	plane, _ := testPlane(64, 64, 1, 64)
	residual := make([]int16, 64*64)
	b.ReportAllocs()
	for b.Loop() {
		_ = AddResidualPlaneBlock(plane, 1, 8, 0, 0, 64, 64, residual, 64)
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) (frame.Plane, []byte) {
	buf := make([]byte, stride*height)
	return frame.Plane{
		Pix:    buf,
		Stride: stride,
		Width:  width,
		Height: height,
	}, buf
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}
