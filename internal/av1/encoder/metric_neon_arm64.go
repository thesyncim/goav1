//go:build arm64 && !purego

package encoder

import "unsafe"

// pixelStatsNEONCtx carries one width-multiple-of-eight metric block. Field
// offsets are mirrored by #define in metric_neon_arm64.s.
type pixelStatsNEONCtx struct {
	Src       unsafe.Pointer
	SrcStride int64
	Ref       unsafe.Pointer
	RefStride int64
	W         int64
	H         int64
	SSE       int64
	Sum       int64
}

//go:noescape
func pixelStatsNEONAsm(ctx *pixelStatsNEONCtx)

// satdCoeffsNEONCtx carries SVT's int32 coefficient SATD reducer arguments.
// Field offsets are mirrored by #define in metric_neon_arm64.s.
type satdCoeffsNEONCtx struct {
	Coeff unsafe.Pointer
	Count int64
	Sum   int64
}

//go:noescape
func satdCoeffsNEONAsm(ctx *satdCoeffsNEONCtx)

// hadamard8x8NEONCtx carries SVT's low-bitdepth 8x8 Hadamard producer
// arguments. Field offsets are mirrored by #define in metric_neon_arm64.s.
type hadamard8x8NEONCtx struct {
	Src       unsafe.Pointer
	SrcStride int64
	Coeff     unsafe.Pointer
}

//go:noescape
func hadamard8x8NEONAsm(ctx *hadamard8x8NEONCtx)

//go:noescape
func hadamard16x16CombineNEONAsm(coeff unsafe.Pointer)

//go:noescape
func hadamard32x32CombineNEONAsm(coeff unsafe.Pointer)

func pixelStatsNEON(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	ctx := pixelStatsNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         int64(w),
		H:         int64(h),
	}
	pixelStatsNEONAsm(&ctx)
	return uint32(ctx.SSE), int32(ctx.Sum)
}

func pixelStats8x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 8)
}

func pixelStats16x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 32)
}

func satdCoeffsNEON(coeff []int32, count int) int {
	ctx := satdCoeffsNEONCtx{
		Coeff: unsafe.Pointer(&coeff[0]),
		Count: int64(count),
	}
	satdCoeffsNEONAsm(&ctx)
	return int(ctx.Sum)
}

func hadamard8x8NEON(src []int16, srcStride int, coeff []int32) {
	ctx := hadamard8x8NEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Coeff:     unsafe.Pointer(&coeff[0]),
	}
	hadamard8x8NEONAsm(&ctx)
}

func hadamard16x16NEON(src []int16, srcStride int, coeff []int32) {
	_ = src[15*srcStride+15]
	_ = coeff[255]
	hadamard8x8NEON(src, srcStride, coeff)
	hadamard8x8NEON(src[8:], srcStride, coeff[64:])
	hadamard8x8NEON(src[8*srcStride:], srcStride, coeff[128:])
	hadamard8x8NEON(src[8*srcStride+8:], srcStride, coeff[192:])
	hadamard16x16CombineNEONAsm(unsafe.Pointer(&coeff[0]))
}

func hadamard32x32NEON(src []int16, srcStride int, coeff []int32) {
	_ = src[31*srcStride+31]
	_ = coeff[1023]
	hadamard16x16NEON(src, srcStride, coeff)
	hadamard16x16NEON(src[16:], srcStride, coeff[256:])
	hadamard16x16NEON(src[16*srcStride:], srcStride, coeff[512:])
	hadamard16x16NEON(src[16*srcStride+16:], srcStride, coeff[768:])
	hadamard32x32CombineNEONAsm(unsafe.Pointer(&coeff[0]))
}

func init() {
	pixelStats8x8Impl = pixelStats8x8NEON
	pixelStats16x16Impl = pixelStats16x16NEON
	pixelStats32x32Impl = pixelStats32x32NEON
	satdCoeffsImpl = satdCoeffsNEON
	hadamard8x8Impl = hadamard8x8NEON
	hadamard16x16Impl = hadamard16x16NEON
	hadamard32x32Impl = hadamard32x32NEON
}
