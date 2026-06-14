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

//go:noescape
func pixelStats4NEONAsm(ctx *pixelStatsNEONCtx)

// satdCoeffsNEONCtx carries SVT's int32 coefficient SATD reducer arguments.
// Field offsets are mirrored by #define in metric_neon_arm64.s.
type satdCoeffsNEONCtx struct {
	Coeff unsafe.Pointer
	Count int64
	Sum   int64
}

//go:noescape
func satdCoeffsNEONAsm(ctx *satdCoeffsNEONCtx)

//go:noescape
func hadamard4x4NEONAsm(ctx *hadamard8x8NEONCtx)

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

func pixelStats4NEON(src []byte, srcStride int, ref []byte, refStride int, h int) (sse uint32, sum int32) {
	ctx := pixelStatsNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         4,
		H:         int64(h),
	}
	pixelStats4NEONAsm(&ctx)
	return uint32(ctx.SSE), int32(ctx.Sum)
}

func pixelStats8x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 8)
}

func pixelStats4x4NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStats4NEON(src, srcStride, ref, refStride, 4)
}

func pixelStats8x4NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 4)
}

func pixelStats4x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStats4NEON(src, srcStride, ref, refStride, 8)
}

func pixelStats16x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats8x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats16x4NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 4)
}

func pixelStats4x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStats4NEON(src, srcStride, ref, refStride, 16)
}

func pixelStats16x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 8)
}

func pixelStats8x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 32)
}

func pixelStats32x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats16x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats32x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats64x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 64, 16)
}

func pixelStats16x64NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 64)
}

func pixelStats64x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 64, 32)
}

func pixelStats32x64NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 64)
}

func satdCoeffsNEON(coeff []int32, count int) int {
	ctx := satdCoeffsNEONCtx{
		Coeff: unsafe.Pointer(&coeff[0]),
		Count: int64(count),
	}
	satdCoeffsNEONAsm(&ctx)
	return int(ctx.Sum)
}

func hadamard4x4NEON(src []int16, srcStride int, coeff []int32) {
	ctx := hadamard8x8NEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Coeff:     unsafe.Pointer(&coeff[0]),
	}
	hadamard4x4NEONAsm(&ctx)
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
	pixelStats4x4Impl = pixelStats4x4NEON
	pixelStats8x4Impl = pixelStats8x4NEON
	pixelStats4x8Impl = pixelStats4x8NEON
	pixelStats16x8Impl = pixelStats16x8NEON
	pixelStats8x16Impl = pixelStats8x16NEON
	pixelStats16x4Impl = pixelStats16x4NEON
	pixelStats4x16Impl = pixelStats4x16NEON
	pixelStats16x16Impl = pixelStats16x16NEON
	pixelStats32x8Impl = pixelStats32x8NEON
	pixelStats8x32Impl = pixelStats8x32NEON
	pixelStats32x16Impl = pixelStats32x16NEON
	pixelStats16x32Impl = pixelStats16x32NEON
	pixelStats32x32Impl = pixelStats32x32NEON
	pixelStats64x16Impl = pixelStats64x16NEON
	pixelStats16x64Impl = pixelStats16x64NEON
	pixelStats64x32Impl = pixelStats64x32NEON
	pixelStats32x64Impl = pixelStats32x64NEON
	satdCoeffsImpl = satdCoeffsNEON
	hadamard4x4Impl = hadamard4x4NEON
	hadamard8x8Impl = hadamard8x8NEON
	hadamard16x16Impl = hadamard16x16NEON
	hadamard32x32Impl = hadamard32x32NEON
}
