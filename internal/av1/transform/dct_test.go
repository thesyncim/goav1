package transform

import (
	"errors"
	"testing"
)

func TestScratchLen(t *testing.T) {
	got, err := ScratchLen(Size{Width: 16, Height: 32})
	if err != nil {
		t.Fatal(err)
	}
	if got != 512 {
		t.Fatalf("ScratchLen=%d want 512", got)
	}
	if _, err := ScratchLen(Size{Width: 4, Height: 12}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid ScratchLen err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestInverseDCT4(t *testing.T) {
	values := []int32{100, -20, 30, 7}
	inverseDCT4(values, 1, minInt16, maxInt16)
	want := []int32{76, 35, 63, 108}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCT4Strided(t *testing.T) {
	values := []int32{100, 99, -20, 99, 30, 99, 7}
	inverseDCT4(values, 2, minInt16, maxInt16)
	want := []int32{76, 99, 35, 99, 63, 99, 108}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCTBlock4x4DC(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	if err := InverseDCTBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseDCTBlock4x4Mixed(t *testing.T) {
	coeff := []int32{
		100, -20, 30, 7,
		5, -9, 13, -17,
		0, 4, -8, 12,
		-16, 20, -24, 28,
	}
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	if err := InverseDCTBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		3, 1, 3, 4,
		3, 3, 0, 9,
		3, 1, 4, 1,
		4, 1, 4, 4,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseDCTBlock4x4Strides(t *testing.T) {
	coeff := make([]int32, 7*4)
	dst := make([]int16, 6*4)
	scratch := make([]int32, 20)
	coeff[1] = 16
	if err := InverseDCTBlock(dst, 6, coeff, 7, scratch, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 4; row++ {
		for col := 4; col < 6; col++ {
			if got := dst[row*6+col]; got != 0 {
				t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
			}
		}
	}
	for i := 16; i < len(scratch); i++ {
		if scratch[i] != 0 {
			t.Fatalf("scratch padding[%d]=%d want 0", i, scratch[i])
		}
	}
}

func TestInverseDCTBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	scratch := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		scratch     []int32
		size        Size
	}{
		{name: "zero size", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{}},
		{name: "unsupported size", dst: make([]int16, 8*8), dstStride: 8, coeff: make([]int32, 8*8), coeffStride: 8, scratch: make([]int32, 8*8), size: Size{Width: 8, Height: 8}},
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short scratch", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch[:15], size: Size{Width: 4, Height: 4}},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		err := InverseDCTBlock(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.scratch, tt.size)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseDCTBlockAllocs(t *testing.T) {
	coeff := make([]int32, 4*4)
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseDCTBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseDCTBlock allocated: %f", allocs)
	}
}

func FuzzInverseDCTBlock(f *testing.F) {
	f.Add(int16(0), int16(0), int16(0), int16(0))
	f.Add(int16(16), int16(0), int16(0), int16(0))
	f.Add(int16(-128), int16(64), int16(32), int16(-16))
	f.Add(int16(1024), int16(-512), int16(255), int16(-255))

	f.Fuzz(func(t *testing.T, dc int16, c1 int16, c2 int16, c3 int16) {
		const (
			coeffStride = 7
			dstStride   = 6
		)
		coeff := make([]int32, coeffStride*4)
		dst := make([]int16, dstStride*4)
		scratch := make([]int32, 20)
		for row := 0; row < 4; row++ {
			for col := 0; col < 4; col++ {
				base := int32(dc) + int32(row*3-col*2)
				switch (row + col) & 3 {
				case 1:
					base += int32(c1)
				case 2:
					base += int32(c2)
				case 3:
					base += int32(c3)
				}
				coeff[row*coeffStride+col] = base
			}
		}
		if err := InverseDCTBlock(dst, dstStride, coeff, coeffStride, scratch, Size{Width: 4, Height: 4}); err != nil {
			t.Fatalf("InverseDCTBlock err=%v", err)
		}
		for row := 0; row < 4; row++ {
			for col := 4; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := 16; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("scratch padding[%d]=%d want 0", i, scratch[i])
			}
		}
	})
}

func BenchmarkInverseDCTBlock4x4(b *testing.B) {
	coeff := make([]int32, 4*4)
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	for i := range coeff {
		coeff[i] = int32(i%9) - 4
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = InverseDCTBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4})
	}
}
