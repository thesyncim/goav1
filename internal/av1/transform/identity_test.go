package transform

import (
	"errors"
	"testing"
)

func TestSizeShift(t *testing.T) {
	tests := []struct {
		size      Size
		wantShift int
		wantRect2 bool
	}{
		{size: Size{Width: 4, Height: 4}, wantShift: 0},
		{size: Size{Width: 4, Height: 8}, wantShift: 0, wantRect2: true},
		{size: Size{Width: 8, Height: 4}, wantShift: 0, wantRect2: true},
		{size: Size{Width: 4, Height: 16}, wantShift: 1},
		{size: Size{Width: 8, Height: 8}, wantShift: 1},
		{size: Size{Width: 8, Height: 16}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 8, Height: 32}, wantShift: 2},
		{size: Size{Width: 16, Height: 4}, wantShift: 1},
		{size: Size{Width: 16, Height: 8}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 16, Height: 16}, wantShift: 2},
		{size: Size{Width: 16, Height: 32}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 16, Height: 64}, wantShift: 2},
		{size: Size{Width: 32, Height: 8}, wantShift: 2},
		{size: Size{Width: 32, Height: 16}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 32, Height: 32}, wantShift: 2},
		{size: Size{Width: 32, Height: 64}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 64, Height: 16}, wantShift: 2},
		{size: Size{Width: 64, Height: 32}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 64, Height: 64}, wantShift: 2},
	}
	for _, tt := range tests {
		shift, err := tt.size.Shift()
		if err != nil {
			t.Fatalf("%+v Shift err=%v", tt.size, err)
		}
		if shift != tt.wantShift {
			t.Fatalf("%+v shift=%d want %d", tt.size, shift, tt.wantShift)
		}
		if !tt.size.Valid() {
			t.Fatalf("%+v not valid", tt.size)
		}
		if got := tt.size.IsRect2(); got != tt.wantRect2 {
			t.Fatalf("%+v IsRect2=%t want %t", tt.size, got, tt.wantRect2)
		}
	}
}

func TestSizeRejectsInvalid(t *testing.T) {
	invalid := []Size{
		{},
		{Width: 4, Height: 12},
		{Width: 12, Height: 4},
		{Width: 64, Height: 8},
		{Width: 128, Height: 128},
	}
	for _, size := range invalid {
		if size.Valid() {
			t.Fatalf("%+v unexpectedly valid", size)
		}
		if _, err := size.Shift(); !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%+v Shift err=%v want %v", size, err, ErrInvalidTransform)
		}
	}
}

func TestRoundShift(t *testing.T) {
	tests := []struct {
		value int64
		bits  uint8
		want  int32
	}{
		{value: 15, bits: 1, want: 8},
		{value: -15, bits: 1, want: -7},
		{value: 31, bits: 4, want: 2},
		{value: -31, bits: 4, want: -2},
	}
	for _, tt := range tests {
		got, err := RoundShift(tt.value, tt.bits)
		if err != nil {
			t.Fatalf("RoundShift(%d,%d) err=%v", tt.value, tt.bits, err)
		}
		if got != tt.want {
			t.Fatalf("RoundShift(%d,%d)=%d want %d", tt.value, tt.bits, got, tt.want)
		}
	}
	if _, err := RoundShift(1, 0); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("zero bit shift err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := RoundShift(1<<40, 1); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("overflow shift err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := RoundShift(maxInt64, 1); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("bias overflow shift err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestRoundShiftArrayMatchesLibaomBitSweep(t *testing.T) {
	widths := [...]int{4, 8, 16, 32, 64}
	bits := [...]int{-4, -3, -2, -1, 0, 1, 2, 3, 4}
	for _, width := range widths {
		for _, bit := range bits {
			var input [64]int32
			fillRoundShiftInput(input[:])
			got := input
			want := input
			if err := RoundShiftArray(got[:width], bit); err != nil {
				t.Fatalf("RoundShiftArray width=%d bit=%d: %v", width, bit, err)
			}
			referenceRoundShiftArray(want[:width], bit)
			for i := 0; i < width; i++ {
				if got[i] != want[i] {
					t.Fatalf("RoundShiftArray width=%d bit=%d index=%d got=%d want %d", width, bit, i, got[i], want[i])
				}
			}
		}
	}
}

func TestRoundShiftArrayClampsAndRejectsInvalidBits(t *testing.T) {
	values := []int32{maxInt32, minInt32, 1, -1, 0}
	if err := RoundShiftArray(values, -1); err != nil {
		t.Fatal(err)
	}
	if values[0] != maxInt32 || values[1] != minInt32 || values[2] != 2 || values[3] != -2 || values[4] != 0 {
		t.Fatalf("values=%v", values)
	}
	if err := RoundShiftArray(values, 63); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("positive invalid bit err=%v want %v", err, ErrInvalidTransform)
	}
	if err := RoundShiftArray(values, -63); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("negative invalid bit err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestRoundShiftArrayAllocs(t *testing.T) {
	var values [64]int32
	fillRoundShiftInput(values[:])
	allocs := testing.AllocsPerRun(1000, func() {
		if err := RoundShiftArray(values[:], -4); err != nil {
			t.Fatal(err)
		}
		if err := RoundShiftArray(values[:], 4); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("RoundShiftArray allocated: %f", allocs)
	}
}

func TestInverseIdentity1DValue(t *testing.T) {
	tests := []struct {
		length int
		want   int32
	}{
		{length: 4, want: 141},
		{length: 8, want: 200},
		{length: 16, want: 283},
		{length: 32, want: 400},
	}
	for _, tt := range tests {
		got, err := InverseIdentity1DValue(100, tt.length)
		if err != nil {
			t.Fatalf("InverseIdentity1DValue length %d err=%v", tt.length, err)
		}
		if got != tt.want {
			t.Fatalf("InverseIdentity1DValue length %d=%d want %d", tt.length, got, tt.want)
		}
	}
	if _, err := InverseIdentity1DValue(100, 64); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid identity length err=%v want %v", err, ErrInvalidTransform)
	}
}

func fillRoundShiftInput(dst []int32) {
	x := uint32(0x243f6a88)
	for i := range dst {
		x = x*1664525 + 1013904223
		dst[i] = int32(x>>3) - 0x10000000
	}
}

func referenceRoundShiftArray(values []int32, bit int) {
	if bit == 0 {
		return
	}
	if bit > 0 {
		for i, v := range values {
			values[i] = clipInt32((int64(v) + (int64(1) << (bit - 1))) >> bit)
		}
		return
	}
	shift := -bit
	for i, v := range values {
		if v == 0 {
			values[i] = 0
			continue
		}
		if shift >= 31 {
			if v < 0 {
				values[i] = minInt32
			} else {
				values[i] = maxInt32
			}
			continue
		}
		values[i] = clipInt32(int64(v) << uint(shift))
	}
}

func TestInverseIdentityBlock4x4(t *testing.T) {
	coeff := []int32{
		16, -16, 0, 0,
		0, 32, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	dst := make([]int16, 4*4)
	if err := InverseIdentityBlock(dst, 4, coeff, 4, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		2, -2, 0, 0,
		0, 4, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseIdentityBlockRect2Scaling(t *testing.T) {
	coeff := make([]int32, 4*8)
	coeff[0] = 64
	dst := make([]int16, 4*8)
	if err := InverseIdentityBlock(dst, 4, coeff, 4, Size{Width: 4, Height: 8}); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 8 {
		t.Fatalf("rect2 dst[0]=%d want 8", dst[0])
	}
}

func TestInverseIdentityBlockStrides(t *testing.T) {
	coeff := make([]int32, 6*4)
	dst := make([]int16, 7*4)
	coeff[1*6+2] = 16
	if err := InverseIdentityBlock(dst, 7, coeff, 6, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	if got := dst[1*7+2]; got != 2 {
		t.Fatalf("strided residual=%d want 2", got)
	}
	if dst[1*7+4] != 0 {
		t.Fatalf("stride padding overwritten: %d", dst[1*7+4])
	}
}

func TestInverseIdentityBlockClips(t *testing.T) {
	coeff := []int32{maxInt32}
	dst := []int16{0}
	if err := InverseIdentityBlock(dst, 1, coeff, 1, Size{Width: 4, Height: 4}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidTransform)
	}

	coeff = make([]int32, 4*4)
	dst = make([]int16, 4*4)
	coeff[0] = maxInt32
	if err := InverseIdentityBlock(dst, 4, coeff, 4, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	if dst[0] != maxInt16 {
		t.Fatalf("clipped residual=%d want %d", dst[0], maxInt16)
	}
}

func TestInverseIdentityBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		size        Size
	}{
		{name: "zero size", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, size: Size{}},
		{name: "64 idtx", dst: make([]int16, 64*64), dstStride: 64, coeff: make([]int32, 64*64), coeffStride: 64, size: Size{Width: 64, Height: 64}},
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, size: Size{Width: 4, Height: 4}},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, size: Size{Width: 4, Height: 4}},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, size: Size{Width: 4, Height: 4}},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		err := InverseIdentityBlock(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.size)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseIdentityBlockAllocs(t *testing.T) {
	coeff := make([]int32, 16*16)
	dst := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseIdentityBlock(dst, 16, coeff, 16, Size{Width: 16, Height: 16}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseIdentityBlock allocated: %f", allocs)
	}
}

func FuzzInverseIdentityBlock(f *testing.F) {
	f.Add(uint8(0), int16(0))
	f.Add(uint8(1), int16(64))
	f.Add(uint8(5), int16(-128))
	f.Add(uint8(9), int16(1024))

	sizes := [...]Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 8, Height: 32},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 16},
		{Width: 16, Height: 32},
		{Width: 32, Height: 8},
		{Width: 32, Height: 16},
		{Width: 32, Height: 32},
	}

	f.Fuzz(func(t *testing.T, rawSize uint8, coeffValue int16) {
		size := sizes[int(rawSize)%len(sizes)]
		coeffStride := size.Width + 3
		dstStride := size.Width + 5
		coeff := make([]int32, coeffStride*size.Height)
		dst := make([]int16, dstStride*size.Height)
		for row := 0; row < size.Height; row++ {
			for col := 0; col < size.Width; col++ {
				coeff[row*coeffStride+col] = int32(coeffValue) + int32(row-col)
			}
		}

		if err := InverseIdentityBlock(dst, dstStride, coeff, coeffStride, size); err != nil {
			t.Fatalf("InverseIdentityBlock err=%v", err)
		}
		for row := 0; row < size.Height; row++ {
			padding := dst[row*dstStride+size.Width : row*dstStride+dstStride]
			for i, got := range padding {
				if got != 0 {
					t.Fatalf("padding row=%d col=%d overwritten with %d", row, size.Width+i, got)
				}
			}
		}
	})
}

func BenchmarkInverseIdentityBlock(b *testing.B) {
	coeff := make([]int32, 32*32)
	dst := make([]int16, 32*32)
	for i := range coeff {
		coeff[i] = int32(i%17) - 8
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = InverseIdentityBlock(dst, 32, coeff, 32, Size{Width: 32, Height: 32})
	}
}
