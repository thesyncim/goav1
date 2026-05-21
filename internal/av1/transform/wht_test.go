package transform

import (
	"errors"
	"testing"
)

func TestInverseWHT4x4BlockEOB1(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	dst := make([]int16, 4*4)
	if err := InverseWHT4x4Block(dst, 4, coeff, 4, 1); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseWHT4x4BlockEOB16(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	coeff[1] = 8
	dst := make([]int16, 6*4)
	if err := InverseWHT4x4Block(dst, 6, coeff, 4, 2); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		2, 2, 2, 2,
		1, 1, 1, 1,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			if got := dst[row*6+col]; got != want[row*4+col] {
				t.Fatalf("dst(%d,%d)=%d want %d", col, row, got, want[row*4+col])
			}
		}
		if got := dst[row*6+4]; got != 0 {
			t.Fatalf("dst padding row=%d overwritten with %d", row, got)
		}
	}
}

func TestInverseWHT4x4BlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		eob         int
	}{
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, eob: 1},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, eob: 1},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, eob: 1},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, eob: 1},
		{name: "negative eob", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, eob: -1},
	}
	for _, tt := range tests {
		err := InverseWHT4x4Block(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.eob)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseWHT4x4BlockAllocs(t *testing.T) {
	coeff := make([]int32, 4*4)
	dst := make([]int16, 4*4)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseWHT4x4Block(dst, 4, coeff, 4, 2); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseWHT4x4Block allocated: %f", allocs)
	}
}
