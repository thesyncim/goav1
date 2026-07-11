package transform

import "testing"

// TestForwardBlock8x8ADSTImplExtremeResidual hardens the SIMD-vs-PureGo
// differential gate at the residual extremes (all +255 / -255 and alternating
// checkerboards), where the int16 SIMD kernel's intermediates are largest, to
// prove the saturating int16 adds never clamp inside the valid 8-bit residual
// domain (i.e. still byte-identical to the int32 scalar reference).
func TestForwardBlock8x8ADSTImplExtremeResidual(t *testing.T) {
	const resStride, coeffStride = 19, 13
	patterns := []func(r, c int) int16{
		func(r, c int) int16 { return 255 },
		func(r, c int) int16 { return -255 },
		func(r, c int) int16 {
			if (r+c)&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if r&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if c&1 == 0 {
				return 255
			}
			return -255
		},
	}
	cases := []struct {
		name string
		impl func([]int32, int, []int16, int, []int32)
		pure func([]int32, int, []int16, int, []int32)
	}{
		{"ADST_DCT", forwardBlock8x8ADSTDCTImpl, forwardBlock8x8ADSTDCTPureGo},
		{"DCT_ADST", forwardBlock8x8DCTADSTImpl, forwardBlock8x8DCTADSTPureGo},
		{"ADST_ADST", forwardBlock8x8ADSTADSTImpl, forwardBlock8x8ADSTADSTPureGo},
		{"IDTX", forwardBlock8x8IDTXImpl, forwardBlock8x8IDTXPureGo},
	}
	for _, tc := range cases {
		for pi, pat := range patterns {
			residual := make([]int16, resStride*8)
			for r := range 8 {
				for c := range 8 {
					residual[r*resStride+c] = pat(r, c)
				}
			}
			var gs, ws [64]int32
			got := make([]int32, coeffStride*8)
			want := make([]int32, coeffStride*8)
			tc.impl(got, coeffStride, residual, resStride, gs[:])
			tc.pure(want, coeffStride, residual, resStride, ws[:])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s pattern %d: coeff[%d] impl %d want %d", tc.name, pi, i, got[i], want[i])
				}
			}
		}
	}
}
