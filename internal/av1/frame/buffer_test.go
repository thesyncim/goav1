package frame

import (
	"errors"
	"testing"
)

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
	if err := StoreBorderedSamplePlane(dst, 1, samples); err != nil {
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
	if err := StoreBorderedSamplePlane(dst, 2, samples); err != nil {
		t.Fatal(err)
	}
	if got := getTestPlaneSample(dst, 2, 1, 0); got != 4095 {
		t.Fatalf("stored=%d want 4095", got)
	}
	if got := getTestPlaneSample(dst, 2, 2, 1); got != 1012 {
		t.Fatalf("stored=%d want 1012", got)
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
	overflow := BorderedSamplePlane{Pix: []uint16{256}, Stride: 1, Origin: 0, Width: 1, Height: 1}
	if err := StoreBorderedSamplePlane(Plane{Pix: make([]byte, 1), Stride: 1, Width: 1, Height: 1}, 1, overflow); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("overflow store err=%v want %v", err, ErrInvalidPlane)
	}
	crossesStride := BorderedSamplePlane{Pix: make([]uint16, 10), Stride: 4, Origin: 3, Width: 2, Height: 1}
	if err := StoreBorderedSamplePlane(Plane{Pix: make([]byte, 2), Stride: 2, Width: 2, Height: 1}, 1, crossesStride); !errors.Is(err, ErrInvalidPlane) {
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
		if err := StoreBorderedSamplePlane(dst, 1, samples); err != nil {
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
		if err := StoreBorderedSamplePlane(dst, bytesPerSample, samples); err != nil {
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
				_ = StoreBorderedSamplePlane(dst, bytesPerSample, samples)
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
