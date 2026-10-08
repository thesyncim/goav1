//go:build amd64 && !purego

package transform

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// fdctBigAMD64Ctx carries the arguments for the 16x16 and 32x32 forward DCT
// kernels; offsets are mirrored by #define in the generated .s files. Strides
// are in elements. Buf points at a caller-owned NxN int32 inter-pass buffer
// and Bank at two int32 butterfly banks (2*N*8 int32) the ping-pong network
// stages through.
type fdctBigAMD64Ctx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
	Buf       unsafe.Pointer
	Bank      unsafe.Pointer
}

//go:noescape
func fdct16x16AVX2Asm(ctx *fdctBigAMD64Ctx)

//go:noescape
func fdct32x32AVX2Asm(ctx *fdctBigAMD64Ctx)

func forwardDCT16x16AVX2(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 16, 16, 255) {
		forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	var buf [256]int32  // 16x16 inter-pass
	var bank [256]int32 // two 16-vector butterfly banks (2*16*8)
	ctx := fdctBigAMD64Ctx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
		Bank:      unsafe.Pointer(&bank[0]),
	}
	fdct16x16AVX2Asm(&ctx)
}

func forwardDCT32x32AVX2(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 32, 32, 255) {
		forwardDCT32x32PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	var buf [1024]int32 // 32x32 inter-pass
	var bank [512]int32 // two 32-vector butterfly banks (2*32*8)
	ctx := fdctBigAMD64Ctx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
		Bank:      unsafe.Pointer(&bank[0]),
	}
	fdct32x32AVX2Asm(&ctx)
}

// init routes the 16x16 and 32x32 forward DCT through AVX2 when available.
// These kernels are exact for 8-bit residuals; their wrappers preserve
// full-int16 behavior by falling back to the portable implementation outside
// that range.
func init() {
	if cpu.Detected.AVX2 {
		forwardDCT16x16Impl = forwardDCT16x16AVX2
		forwardDCT32x32Impl = forwardDCT32x32AVX2
	}
}
