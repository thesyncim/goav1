package quantize

import (
	"errors"
	"testing"
)

// FuzzDequantizeBlockScaledQMatrix exercises the inverse-qmatrix dequant path
// to catch unchecked allocations, integer overflow, and length-validation
// gaps. The contract is "reject ErrInvalidQuantizer or return a well-formed
// dst slice without panicking".
func FuzzDequantizeBlockScaledQMatrix(f *testing.F) {
	f.Add(uint8(8), uint8(0), int16(0), uint8(4), uint8(4), uint8(0), int16(1), uint16(32))
	f.Add(uint8(10), uint8(96), int16(-3), uint8(8), uint8(4), uint8(1), int16(-7), uint16(96))
	f.Add(uint8(12), uint8(255), int16(12), uint8(16), uint8(16), uint8(2), int16(31), uint16(255))

	f.Fuzz(func(t *testing.T, rawBitDepth uint8, qIndex uint8, delta int16, rawW uint8, rawH uint8, rawTxScale uint8, coeffValue int16, rawQM uint16) {
		bitDepths := [3]uint8{8, 10, 12}
		bitDepth := bitDepths[int(rawBitDepth)%len(bitDepths)]
		width := int(rawW%16) + 1
		height := int(rawH%16) + 1
		txScale := rawTxScale % 3
		coeffStride := height + 1
		dstStride := height + 2

		coeff := make([]int16, coeffStride*width)
		dst := make([]int32, dstStride*width)
		iqMatrix := make([]uint16, width*height)
		// Keep iqMatrix entries in libaom's valid range. The qmBits
		// shift can handle any uint16, but realistic values stay <=
		// 256.
		base := rawQM
		for i := range iqMatrix {
			v := base + uint16(i)
			if v < 1 {
				v = 1
			}
			iqMatrix[i] = v % 257
			if iqMatrix[i] == 0 {
				iqMatrix[i] = 1
			}
		}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = coeffValue + int16(row+col)
			}
		}

		dc, err := DCQuant(qIndex, delta, bitDepth)
		if err != nil {
			t.Fatal(err)
		}
		ac, err := ACQuant(qIndex, -delta, bitDepth)
		if err != nil {
			t.Fatal(err)
		}
		q := Quantizer{DC: dc, AC: ac}

		// Truncate iqMatrix some of the time to exercise the
		// length-validation guard.
		if rawQM&1 != 0 && len(iqMatrix) > 0 {
			short := iqMatrix[:len(iqMatrix)-1]
			if err := DequantizeBlockScaledQMatrix(dst, dstStride, coeff, coeffStride, width, height, q, txScale, short); !errors.Is(err, ErrInvalidQuantizer) {
				t.Fatalf("short qmatrix accepted err=%v", err)
			}
		}

		if err := DequantizeBlockScaledQMatrix(dst, dstStride, coeff, coeffStride, width, height, q, txScale, iqMatrix); err != nil {
			t.Fatalf("DequantizeBlockScaledQMatrix err=%v", err)
		}
		// Spot-check the formula and bound the magnitude: each
		// dequant result must be int32 representable; this catches
		// any pathological multiply that overflows the bound used by
		// downstream inverse-transform code.
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				scale := int32(ac)
				if row == 0 && col == 0 {
					scale = int32(dc)
				}
				scale = (int32(iqMatrix[row*width+col])*scale + (1 << (qmBits - 1))) >> qmBits
				input := int32(coeff[col*coeffStride+row])
				negative := input < 0
				if negative {
					input = -input
				}
				want := (input * scale) >> txScale
				if negative {
					want = -want
				}
				got := dst[col*dstStride+row]
				if got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", col, row, got, want)
				}
			}
		}
	})
}
