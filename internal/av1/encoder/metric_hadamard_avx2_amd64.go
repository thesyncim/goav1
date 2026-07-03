//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// metric_hadamard_avx2_amd64.go closes the amd64 gap for the SATD Hadamard
// producers (hadamard4x4/8x8/16x16/32x32), mirroring the NEON kernels bound on
// arm64. They default to pure-Go on amd64.
//
// The 4x4 and 8x8 primitives run the SVT/AOM Walsh-Hadamard butterfly in int16
// lanes (VPADDW/VPSUBW wrap exactly like the portable int16 arithmetic) with a
// full SSE lane transpose between the two passes, producing the transposed
// coefficient order that svt_aom_hadamard_{4x4,8x8}_neon emits. The SATD
// consumer is order-invariant, and the encoder's parity tests explicitly accept
// this NEON order (hadamard8x8TransposeReference), so the result is bit-exact
// with the accepted reference.
//
// The 16x16 and 32x32 producers reuse the AVX2 8x8 primitive for the four
// quadrants, then apply the SVT quadrant combine (the same signed halving
// add/sub and scatter order as svt_aom_hadamard_16x16/32x32_neon). The combine
// is a light in-place int32 pass, so it stays in Go.
//
// The differential test calls the kernels directly (ungated on CPUID) so hosts
// that hide AVX2 (Rosetta 2) still prove bit-exactness.

// hadamardAVX2Ctx carries the low-bitdepth Hadamard producer arguments. Field
// offsets are mirrored by #define in metric_hadamard_avx2_amd64.s.
type hadamardAVX2Ctx struct {
	Src       unsafe.Pointer
	SrcStride int64
	Coeff     unsafe.Pointer
}

//go:noescape
func hadamard4x4AVX2Asm(ctx *hadamardAVX2Ctx)

//go:noescape
func hadamard8x8AVX2Asm(ctx *hadamardAVX2Ctx)

func hadamard4x4AVX2(src []int16, srcStride int, coeff []int32) {
	_ = src[3*srcStride+3]
	_ = coeff[15]
	ctx := hadamardAVX2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Coeff:     unsafe.Pointer(&coeff[0]),
	}
	hadamard4x4AVX2Asm(&ctx)
}

func hadamard8x8AVX2(src []int16, srcStride int, coeff []int32) {
	_ = src[7*srcStride+7]
	_ = coeff[63]
	ctx := hadamardAVX2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Coeff:     unsafe.Pointer(&coeff[0]),
	}
	hadamard8x8AVX2Asm(&ctx)
}

// hadamard16x16AVX2 mirrors svt_aom_hadamard_16x16_neon: four transposed-order
// 8x8 quadrants followed by the signed halving quadrant combine with the SVT
// scatter order.
func hadamard16x16AVX2(src []int16, srcStride int, coeff []int32) {
	_ = src[15*srcStride+15]
	_ = coeff[255]
	hadamard8x8AVX2(src, srcStride, coeff)
	hadamard8x8AVX2(src[8:], srcStride, coeff[64:])
	hadamard8x8AVX2(src[8*srcStride:], srcStride, coeff[128:])
	hadamard8x8AVX2(src[8*srcStride+8:], srcStride, coeff[192:])
	hadamard16x16CombineAVX2(coeff)
}

func hadamard16x16CombineAVX2(coeff []int32) {
	for base := 0; base < 64; base += 16 {
		var c [4][4][4]int32
		for g, off := range []int{0, 4, 8, 12} {
			for lane := range 4 {
				idx := base + off + lane
				a0 := coeff[idx]
				a1 := coeff[64+idx]
				a2 := coeff[128+idx]
				a3 := coeff[192+idx]

				b0 := (a0 + a1) >> 1
				b1 := (a0 - a1) >> 1
				b2 := (a2 + a3) >> 1
				b3 := (a2 - a3) >> 1

				c[g][0][lane] = b0 + b2
				c[g][1][lane] = b1 + b3
				c[g][2][lane] = b0 - b2
				c[g][3][lane] = b1 - b3
			}
		}
		for lane := range 4 {
			coeff[base+0+lane] = c[0][0][lane]
			coeff[base+4+lane] = c[2][0][lane]
			coeff[base+8+lane] = c[1][0][lane]
			coeff[base+12+lane] = c[3][0][lane]

			coeff[64+base+0+lane] = c[0][1][lane]
			coeff[64+base+4+lane] = c[2][1][lane]
			coeff[64+base+8+lane] = c[1][1][lane]
			coeff[64+base+12+lane] = c[3][1][lane]

			coeff[128+base+0+lane] = c[0][2][lane]
			coeff[128+base+4+lane] = c[2][2][lane]
			coeff[128+base+8+lane] = c[1][2][lane]
			coeff[128+base+12+lane] = c[3][2][lane]

			coeff[192+base+0+lane] = c[0][3][lane]
			coeff[192+base+4+lane] = c[2][3][lane]
			coeff[192+base+8+lane] = c[1][3][lane]
			coeff[192+base+12+lane] = c[3][3][lane]
		}
	}
}

// hadamard32x32AVX2 mirrors svt_aom_hadamard_32x32_neon: four 16x16 producers
// followed by the >>2 quadrant combine.
func hadamard32x32AVX2(src []int16, srcStride int, coeff []int32) {
	_ = src[31*srcStride+31]
	_ = coeff[1023]
	hadamard16x16AVX2(src, srcStride, coeff)
	hadamard16x16AVX2(src[16:], srcStride, coeff[256:])
	hadamard16x16AVX2(src[16*srcStride:], srcStride, coeff[512:])
	hadamard16x16AVX2(src[16*srcStride+16:], srcStride, coeff[768:])

	for base := 0; base < 256; base += 4 {
		for lane := range 4 {
			idx := base + lane
			a0 := coeff[idx]
			a1 := coeff[256+idx]
			a2 := coeff[512+idx]
			a3 := coeff[768+idx]

			b0 := (a0 + a1) >> 2
			b1 := (a0 - a1) >> 2
			b2 := (a2 + a3) >> 2
			b3 := (a2 - a3) >> 2

			coeff[idx] = b0 + b2
			coeff[256+idx] = b1 + b3
			coeff[512+idx] = b0 - b2
			coeff[768+idx] = b1 - b3
		}
	}
}

func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	hadamard4x4Impl = hadamard4x4AVX2
	hadamard8x8Impl = hadamard8x8AVX2
	hadamard16x16Impl = hadamard16x16AVX2
	hadamard32x32Impl = hadamard32x32AVX2
}
