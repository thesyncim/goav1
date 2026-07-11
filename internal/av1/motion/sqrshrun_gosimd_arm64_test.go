//go:build goexperiment.simd

package motion

import (
	"simd/archsimd"
	"testing"
)

// refSqrshrun is the scalar reference for SQRSHRUN Vd.8B, Vn.8H, #shift:
// signed rounding-shift-right of each int16 lane by shift, then unsigned
// saturation to [0,255], narrowed to uint8.
func refSqrshrun(v int16, shift uint) uint8 {
	r := (int32(v) + (1 << (shift - 1))) >> shift
	if r < 0 {
		r = 0
	}
	if r > 255 {
		r = 255
	}
	return uint8(r)
}

func TestShiftRightRoundNarrowUint8ByteExact(t *testing.T) {
	shifts := []uint{1, 2, 4, 6, 8}
	// Cover the full int16 range plus boundaries around saturation and rounding.
	inputs := []int16{
		0, 1, -1, 2, -2, 31, 32, 33, 63, 64, 65,
		255, 256, 257, -255, -256,
		16383, 16384, -16383, -16384,
		32767, -32768, 100, -100, 8191, 8192, 12345, -12345,
	}
	for _, sh := range shifts {
		for base := 0; base+8 <= len(inputs); base++ {
			var in [8]int16
			copy(in[:], inputs[base:base+8])
			v := archsimd.LoadInt16x8Array(&in)
			var got [16]uint8
			switch sh {
			case 1:
				v.ShiftRightRoundNarrowUint8(1).StoreArray(&got)
			case 2:
				v.ShiftRightRoundNarrowUint8(2).StoreArray(&got)
			case 4:
				v.ShiftRightRoundNarrowUint8(4).StoreArray(&got)
			case 6:
				v.ShiftRightRoundNarrowUint8(6).StoreArray(&got)
			case 8:
				v.ShiftRightRoundNarrowUint8(8).StoreArray(&got)
			}
			for i := 0; i < 8; i++ {
				want := refSqrshrun(in[i], sh)
				if got[i] != want {
					t.Fatalf("shift=%d in=%d: got %d want %d", sh, in[i], got[i], want)
				}
			}
		}
	}
}
