package transform

import (
	"errors"
	"math"
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

func TestInverseDCT8(t *testing.T) {
	values := []int32{100, -20, 30, 7, 11, -13, 17, -19}
	inverseDCT8(values, 1, minInt16, maxInt16)
	want := []int32{88, 65, 30, 44, 44, 104, 53, 136}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCT1DMatchesLibaomReference(t *testing.T) {
	tests := []struct {
		length   int
		maxError int32
	}{
		{length: 4, maxError: 6},
		{length: 8, maxError: 10},
		{length: 16, maxError: 19},
		{length: 32, maxError: 31},
		{length: 64, maxError: 40},
	}
	for _, tt := range tests {
		input := make([]int32, tt.length)
		for i := range input {
			input[i] = int32(((i*37+11)%31)-15) * 7
		}
		if tt.length == 64 {
			for i := 32; i < len(input); i++ {
				input[i] = 0
			}
		}
		got := append([]int32(nil), input...)
		inverseDCT1D(got, 1, tt.length, minInt16, maxInt16)
		want := referenceIDCT1DInt(input)
		for i := range got {
			if diff := abs32(got[i] - want[i]); diff > tt.maxError {
				t.Fatalf("length=%d coeff=%d got=%d want=%d diff=%d max=%d", tt.length, i, got[i], want[i], diff, tt.maxError)
			}
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
	coeff := av1CoeffOrder([]int32{
		100, -20, 30, 7,
		5, -9, 13, -17,
		0, 4, -8, 12,
		-16, 20, -24, 28,
	}, 4, 4)
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

func TestInverseDCTBlock8x8DC(t *testing.T) {
	coeff := make([]int32, 8*8)
	coeff[0] = 32
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	if err := InverseDCTBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseDCTBlock8x8Mixed(t *testing.T) {
	coeff := av1CoeffOrder([]int32{
		-20, 14, 7, 0, -7, -14, 20, 13,
		-9, -15, 20, 14, 8, -3, -9, -15,
		2, -3, -8, -18, 18, 8, 3, -2,
		13, 9, 0, -4, -13, 19, 15, 6,
		-17, -20, 13, 5, -3, -11, -14, 19,
		-6, -13, -20, 14, 7, 0, -7, -14,
		5, -1, -7, -13, -19, 11, 5, -1,
		16, 11, 6, -4, -9, -19, 17, 12,
	}, 8, 8)
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	if err := InverseDCTBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		0, -1, 0, -1, -1, 0, 0, 1,
		2, 0, -1, -1, -1, -1, -2, 1,
		0, 1, 1, -1, 0, 1, -2, 0,
		-1, -1, -5, -2, 0, 0, 0, -1,
		1, 1, 3, -5, 0, 0, 0, 0,
		1, -1, 4, 1, -2, 1, 0, -2,
		1, 0, 2, 1, 2, 1, -3, -1,
		-1, -1, 2, -1, 1, -2, -1, 0,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseDCTBlockSupportedSizes(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
		{Width: 8, Height: 32},
		{Width: 16, Height: 16},
		{Width: 32, Height: 8},
		{Width: 32, Height: 32},
		{Width: 16, Height: 64},
		{Width: 64, Height: 16},
		{Width: 32, Height: 64},
		{Width: 64, Height: 32},
		{Width: 64, Height: 64},
	}
	for _, size := range sizes {
		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 1
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = int32(((row*13+col*7+5)%29)-14) * 3
			}
		}
		dst := make([]int16, dstStride*height)
		scratchLen := width * height
		scratch := make([]int32, scratchLen+3)
		if err := InverseDCTBlock(dst, dstStride, coeff, coeffStride, scratch, size); err != nil {
			t.Fatalf("%dx%d InverseDCTBlock err=%v", size.Width, size.Height, err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("%dx%d dst padding row=%d col=%d overwritten with %d", size.Width, size.Height, row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("%dx%d scratch padding[%d]=%d want 0", size.Width, size.Height, i, scratch[i])
			}
		}
	}
}

func TestInverseDCTBlock4x4Strides(t *testing.T) {
	coeff := make([]int32, 6*4)
	dst := make([]int16, 6*4)
	scratch := make([]int32, 20)
	coeff[1] = 16
	if err := InverseDCTBlock(dst, 6, coeff, 6, scratch, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	for row := range 4 {
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
		{name: "unsupported size", dst: make([]int16, 64*8), dstStride: 64, coeff: make([]int32, 64*8), coeffStride: 8, scratch: make([]int32, 64*8), size: Size{Width: 64, Height: 8}},
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
	coeff4 := make([]int32, 4*4)
	dst4 := make([]int16, 4*4)
	scratch4 := make([]int32, 4*4)
	coeff8 := make([]int32, 8*8)
	dst8 := make([]int16, 8*8)
	scratch8 := make([]int32, 8*8)
	coeff16 := make([]int32, 16*16)
	dst16 := make([]int16, 16*16)
	scratch16 := make([]int32, 16*16)
	coeff32 := make([]int32, 32*32)
	dst32 := make([]int16, 32*32)
	scratch32 := make([]int32, 32*32)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseDCTBlock(dst4, 4, coeff4, 4, scratch4, Size{Width: 4, Height: 4}); err != nil {
			t.Fatal(err)
		}
		if err := InverseDCTBlock(dst8, 8, coeff8, 8, scratch8, Size{Width: 8, Height: 8}); err != nil {
			t.Fatal(err)
		}
		if err := InverseDCTBlock(dst16, 16, coeff16, 16, scratch16, Size{Width: 16, Height: 16}); err != nil {
			t.Fatal(err)
		}
		if err := InverseDCTBlock(dst32, 32, coeff32, 32, scratch32, Size{Width: 32, Height: 32}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseDCTBlock allocated: %f", allocs)
	}
}

func FuzzInverseDCTBlock(f *testing.F) {
	f.Add(uint8(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(0), int16(16), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(32), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(-128), int16(64), int16(32), int16(-16))
	f.Add(uint8(2), int16(64), int16(-21), int16(17), int16(-9))
	f.Add(uint8(3), int16(128), int16(31), int16(-19), int16(11))
	f.Add(uint8(4), int16(256), int16(-33), int16(23), int16(-13))
	f.Add(uint8(1), int16(1024), int16(-512), int16(255), int16(-255))

	f.Fuzz(func(t *testing.T, rawSize uint8, dc int16, c1 int16, c2 int16, c3 int16) {
		size := Size{Width: 4, Height: 4}
		switch rawSize & 7 {
		case 1:
			size = Size{Width: 8, Height: 8}
		case 2:
			size = Size{Width: 16, Height: 16}
		case 3:
			size = Size{Width: 32, Height: 32}
		case 4:
			size = Size{Width: 64, Height: 64}
		case 5:
			size = Size{Width: 64, Height: 32}
		case 6:
			size = Size{Width: 32, Height: 64}
		}
		width := int(size.Width)
		height := int(size.Height)
		blockLen := width * height
		coeffStride := height + 3
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		scratch := make([]int32, blockLen+4)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				base := int32(dc) + int32(row*3-col*2)
				switch (row + col) & 3 {
				case 1:
					base += int32(c1)
				case 2:
					base += int32(c2)
				case 3:
					base += int32(c3)
				}
				coeff[col*coeffStride+row] = base
			}
		}
		if err := InverseDCTBlock(dst, dstStride, coeff, coeffStride, scratch, size); err != nil {
			t.Fatalf("InverseDCTBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := blockLen; i < len(scratch); i++ {
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
	for b.Loop() {
		_ = InverseDCTBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4})
	}
}

func BenchmarkInverseDCTBlock8x8(b *testing.B) {
	coeff := make([]int32, 8*8)
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	for i := range coeff {
		coeff[i] = int32(i%17) - 8
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = InverseDCTBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8})
	}
}

func referenceIDCT1DInt(input []int32) []int32 {
	size := len(input)
	out := make([]int32, size)
	invSqrt2 := 1 / math.Sqrt2
	for k := range size {
		sum := 0.0
		for n := range size {
			scale := 1.0
			if n == 0 {
				scale = invSqrt2
			}
			angle := math.Pi * float64((2*k+1)*n) / float64(2*size)
			sum += scale * float64(input[n]) * math.Cos(angle)
		}
		out[k] = int32(math.Round(sum))
	}
	return out
}

func av1CoeffOrder(rowMajor []int32, width int, height int) []int32 {
	out := make([]int32, width*height)
	for row := range height {
		for col := range width {
			out[col*height+row] = rowMajor[row*width+col]
		}
	}
	return out
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}
