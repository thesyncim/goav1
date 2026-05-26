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
	for row := range 4 {
		for col := range 4 {
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

// FuzzInverseWHT4x4Block stresses the 4x4 inverse Walsh-Hadamard with random
// coefficient values, strides, and EOB choices. The transform must complete
// without panicking and must not write outside the requested block.
func FuzzInverseWHT4x4Block(f *testing.F) {
	f.Add(uint8(0), int16(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(16), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(2), int16(64), int16(-32), int16(16), int16(-8), int16(4))
	f.Add(uint8(3), int16(-256), int16(128), int16(-64), int16(32), int16(-16))
	f.Add(uint8(0xff), int16(32767), int16(-32768), int16(32767), int16(-32768), int16(0))
	f.Add(uint8(7), int16(1024), int16(-1024), int16(0), int16(255), int16(-255))

	f.Fuzz(func(t *testing.T, rawMode uint8, c0 int16, c1 int16, c2 int16, c3 int16, c4 int16) {
		coeffStride := 4 + int(rawMode&3)
		dstStride := 4 + int((rawMode>>2)&3)
		eob := 1
		if rawMode&0x80 != 0 {
			eob = 16
		} else if rawMode&0x40 != 0 {
			eob = 2
		}

		coeff := make([]int32, coeffStride*4+1)
		dst := make([]int16, dstStride*4+1)
		seeds := [5]int16{c0, c1, c2, c3, c4}
		for col := range 4 {
			for row := range 4 {
				idx := col*coeffStride + row
				coeff[idx] = int32(seeds[(col+row)%5]) + int32(col)*3 - int32(row)
			}
		}

		const sentinel = int16(0x5a5a)
		dst[len(dst)-1] = sentinel
		for row := range 4 {
			for col := 4; col < dstStride; col++ {
				dst[row*dstStride+col] = sentinel
			}
		}

		if err := InverseWHT4x4Block(dst, dstStride, coeff, coeffStride, eob); err != nil {
			t.Fatalf("InverseWHT4x4Block err=%v", err)
		}
		if got := dst[len(dst)-1]; got != sentinel {
			t.Fatalf("tail sentinel overwritten: got %d want %d", got, sentinel)
		}
		for row := range 4 {
			for col := 4; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != sentinel {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
	})
}
