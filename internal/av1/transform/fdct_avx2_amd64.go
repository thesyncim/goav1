//go:build amd64 && !purego

package transform

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// fdctAMD64Ctx carries the kernel arguments; offsets are mirrored by #define
// in fdct_avx2_amd64.s. Strides are in elements.
type fdctAMD64Ctx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
}

//go:noescape
func fdct8x8AVX2Asm(ctx *fdctAMD64Ctx)

//go:noescape
func fdct4x4AVX2Asm(ctx *fdctAMD64Ctx)

// fdct4Pass is the shared av1_fdct4 vector pass used by fdct4x4AVX2Asm; it
// operates entirely on registers set up by its caller.
func fdct4Pass()

func forwardDCT4x4AVX2(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdctAMD64Ctx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct4x4AVX2Asm(&ctx)
}

func forwardDCT8x8AVX2(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdctAMD64Ctx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct8x8AVX2Asm(&ctx)
}

// init routes the 8x8 forward DCT through the AVX2 kernel when the CPU
// advertises it; otherwise the dispatch keeps the portable reference. The
// kernel is bit-exact with the reference for 8-bit residual ranges
// (TestForwardDCT8x8AVX2MatchesPureGo calls it directly regardless of the
// CPUID report, so Rosetta hosts still verify it).
func init() {
	if cpu.Detected.AVX2 {
		forwardDCT8x8Impl = forwardDCT8x8AVX2
		forwardDCT4x4Impl = forwardDCT4x4AVX2
	}
}
