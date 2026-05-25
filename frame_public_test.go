package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicFramePoolRequiredSizeAndBind(t *testing.T) {
	format := av1.FrameFormat{
		Width:        16,
		Height:       16,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, backingSize, err := av1.FramePoolRequiredSize(format, 3)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Size <= 0 || backingSize != layout.Size*3 {
		t.Fatalf("layout=%+v backingSize=%d", layout, backingSize)
	}

	backing := make([]byte, backingSize)
	var frames [3]av1.Frame
	var free [3]int
	var used [3]bool
	pool, err := av1.BindFramePool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		index, frame, err := pool.AcquireFormat(format)
		if err != nil {
			t.Fatal(err)
		}
		if index != i || frame == nil {
			t.Fatalf("acquire %d index=%d frame=%p", i, index, frame)
		}
		frame.Y.Pix[0] = byte(0x20 + i)
		if backing[i*layout.Size] != byte(0x20+i) {
			t.Fatalf("frame %d did not bind expected backing slot", i)
		}
	}
	if _, _, err := pool.Acquire(); !errors.Is(err, av1.ErrFramePoolEmpty) {
		t.Fatalf("empty acquire err=%v want %v", err, av1.ErrFramePoolEmpty)
	}
	if err := pool.ReleaseMany([]int{0, 2}); err != nil {
		t.Fatal(err)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
}

func TestPublicFramePoolRequiredSizeRejectsInvalid(t *testing.T) {
	format := av1.FrameFormat{Width: 16, Height: 16, BitDepth: 8}
	if _, _, err := av1.FramePoolRequiredSize(format, 0); !errors.Is(err, av1.ErrFrameInvalidPool) {
		t.Fatalf("count err=%v want %v", err, av1.ErrFrameInvalidPool)
	}

	layout, err := av1.FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	overflowCount := int(^uint(0)>>1)/layout.Size + 1
	if _, _, err := av1.FramePoolRequiredSize(format, overflowCount); !errors.Is(err, av1.ErrFrameInvalidFormat) {
		t.Fatalf("overflow err=%v want %v", err, av1.ErrFrameInvalidFormat)
	}
}

func TestPublicFrameSamplePlaneRoundTrip8Bit(t *testing.T) {
	plane := av1.FramePlane{Pix: make([]byte, 8*3), Stride: 8, Width: 5, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte(10*y + x)
		}
	}

	need, err := av1.FrameSamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if need != 24 {
		t.Fatalf("need=%d want 24", need)
	}
	samples, err := av1.LoadFrameSamplePlane(make([]uint16, need), plane, 1)
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
	dst := av1.FramePlane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := av1.StoreFrameSamplePlane(dst, 1, samples); err != nil {
		t.Fatal(err)
	}
	if dst.Pix[1*dst.Stride+3] != 201 || dst.Pix[2*dst.Stride+4] != 24 {
		t.Fatalf("dst=%v", dst.Pix)
	}
}

func TestPublicFrameBorderedSamplePlaneRoundTripHighBitDepth(t *testing.T) {
	plane := av1.FramePlane{Pix: make([]byte, 12*2), Stride: 12, Width: 4, Height: 2}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, 2, x, y, uint16(1000+y*10+x))
		}
	}

	layout, err := av1.FrameBorderedSamplePlaneLen(plane, 2, 5, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if want := (av1.FrameBorderedSamplePlaneLayout{Stride: 16, Origin: 21, Rows: 4, Len: 64}); layout != want {
		t.Fatalf("layout=%+v want %+v", layout, want)
	}
	scratch := make([]uint16, layout.Len)
	for i := range scratch {
		scratch[i] = 0xeeee
	}
	samples, err := av1.LoadFrameBorderedSamplePlane(scratch, plane, 2, 5, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Pix[0] != 0xeeee || samples.Pix[samples.Origin-1] != 0xeeee {
		t.Fatalf("border samples were overwritten")
	}
	if got := samples.Pix[samples.Origin+1*samples.Stride+2]; got != 1012 {
		t.Fatalf("sample=%d want 1012", got)
	}

	samples.Pix[samples.Origin+1] = 4095
	dst := av1.FramePlane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	if err := av1.StoreFrameBorderedSamplePlane(dst, 2, samples); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(dst, 2, 1, 0); got != 4095 {
		t.Fatalf("stored=%d want 4095", got)
	}
	if got := getPublicFrameSample(dst, 2, 2, 1); got != 1012 {
		t.Fatalf("stored=%d want 1012", got)
	}

	bound, err := av1.BindFrameBorderedSamplePlane(scratch, plane, 2, 5, 1, 16)
	if err != nil {
		t.Fatal(err)
	}
	if bound.Stride != layout.Stride || bound.Origin != layout.Origin || &bound.Pix[0] != &scratch[0] {
		t.Fatalf("bound=%+v layout=%+v", bound, layout)
	}
}

func TestPublicFrameSamplePlaneRejectsInvalid(t *testing.T) {
	plane := av1.FramePlane{Pix: make([]byte, 8), Stride: 4, Width: 3, Height: 2}
	if _, err := av1.FrameSamplePlaneLen(plane, 3); !errors.Is(err, av1.ErrFrameInvalidPlane) {
		t.Fatalf("bytesPerSample err=%v want %v", err, av1.ErrFrameInvalidPlane)
	}
	if need, err := av1.FrameSamplePlaneLen(av1.FramePlane{Stride: 4, Width: 4, Height: 2}, 1); err != nil || need != 8 {
		t.Fatalf("geometry-only len=%d err=%v want 8,nil", need, err)
	}
	if _, err := av1.LoadFrameSamplePlane(make([]uint16, 3), plane, 1); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short load err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	overflow := av1.FrameSamplePlane{Pix: []uint16{256}, Stride: 1, Width: 1, Height: 1}
	dst := av1.FramePlane{Pix: make([]byte, 1), Stride: 1, Width: 1, Height: 1}
	if err := av1.StoreFrameSamplePlane(dst, 1, overflow); !errors.Is(err, av1.ErrFrameInvalidPlane) {
		t.Fatalf("overflow store err=%v want %v", err, av1.ErrFrameInvalidPlane)
	}
	if _, err := av1.FrameBorderedSamplePlaneLen(plane, 1, 1, 1, 3); !errors.Is(err, av1.ErrFrameInvalidPlane) {
		t.Fatalf("border align err=%v want %v", err, av1.ErrFrameInvalidPlane)
	}
}

func TestPublicFrameSamplePlaneAllocs(t *testing.T) {
	plane := av1.FramePlane{Pix: make([]byte, 16*16), Stride: 16, Width: 16, Height: 16}
	dst := av1.FramePlane{Pix: make([]byte, len(plane.Pix)), Stride: plane.Stride, Width: plane.Width, Height: plane.Height}
	need, err := av1.FrameSamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, need)
	allocs := testing.AllocsPerRun(1000, func() {
		samples, err := av1.LoadFrameSamplePlane(scratch, plane, 1)
		if err != nil {
			t.Fatalf("LoadFrameSamplePlane err=%v", err)
		}
		if err := av1.StoreFrameSamplePlane(dst, 1, samples); err != nil {
			t.Fatalf("StoreFrameSamplePlane err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}

	layout, err := av1.FrameBorderedSamplePlaneLen(plane, 1, 4, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	borderedScratch := make([]uint16, layout.Len)
	allocs = testing.AllocsPerRun(1000, func() {
		samples, err := av1.LoadFrameBorderedSamplePlane(borderedScratch, plane, 1, 4, 2, 16)
		if err != nil {
			t.Fatalf("LoadFrameBorderedSamplePlane err=%v", err)
		}
		if err := av1.StoreFrameBorderedSamplePlane(dst, 1, samples); err != nil {
			t.Fatalf("StoreFrameBorderedSamplePlane err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("bordered allocs=%v want 0", allocs)
	}
}

func setPublicFrameSample(plane av1.FramePlane, bytesPerSample int, x int, y int, sample uint16) {
	off := y*plane.Stride + x*bytesPerSample
	plane.Pix[off] = byte(sample)
	if bytesPerSample == 2 {
		plane.Pix[off+1] = byte(sample >> 8)
	}
}

func getPublicFrameSample(plane av1.FramePlane, bytesPerSample int, x int, y int) uint16 {
	off := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[off])
	}
	return uint16(plane.Pix[off]) | uint16(plane.Pix[off+1])<<8
}
