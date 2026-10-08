//go:build arm64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

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

//go:noescape
func pixelStatsDotProdAsm(ctx *pixelStatsNEONCtx)

var useDotProdPixelStats = cpu.Detected.DOTPROD

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

func pixelStatsDotProd(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	ctx := pixelStatsNEONCtx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         int64(w),
		H:         int64(h),
	}
	pixelStatsDotProdAsm(&ctx)
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

func pixelStats8x4DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 8, 4)
}

func pixelStats4x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStats4NEON(src, srcStride, ref, refStride, 8)
}

func pixelStats16x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats16x8DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats8x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats8x16DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats16x4NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 4)
}

func pixelStats16x4DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 16, 4)
}

func pixelStats4x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStats4NEON(src, srcStride, ref, refStride, 16)
}

func pixelStats16x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats16x16DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x8NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 8)
}

func pixelStats32x8DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 32, 8)
}

func pixelStats8x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 8, 32)
}

func pixelStats8x32DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 8, 32)
}

func pixelStats32x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats32x16DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats16x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats16x32DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats32x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats32x32DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats64x16NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 64, 16)
}

func pixelStats64x16DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 64, 16)
}

func pixelStats16x64NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 16, 64)
}

func pixelStats16x64DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 16, 64)
}

func pixelStats64x32NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 64, 32)
}

func pixelStats64x32DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 64, 32)
}

func pixelStats32x64NEON(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsNEON(src, srcStride, ref, refStride, 32, 64)
}

func pixelStats32x64DotProd(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsDotProd(src, srcStride, ref, refStride, 32, 64)
}

// bindPixelStatsNEON binds every pixel-domain statistics kernel to its NEON
// (or DOTPROD, when detected) implementation. It is shared by the NEON-only
// dispatch init and the goexperiment.simd dispatch init. SATD and Hadamard
// use Go SIMD when enabled and scalar implementations otherwise.
func bindPixelStatsNEON() {
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
	if useDotProdPixelStats {
		pixelStats16x4Impl = pixelStats16x4DotProd
		pixelStats16x8Impl = pixelStats16x8DotProd
		pixelStats8x16Impl = pixelStats8x16DotProd
		pixelStats16x16Impl = pixelStats16x16DotProd
		pixelStats32x8Impl = pixelStats32x8DotProd
		pixelStats8x32Impl = pixelStats8x32DotProd
		pixelStats32x16Impl = pixelStats32x16DotProd
		pixelStats16x32Impl = pixelStats16x32DotProd
		pixelStats32x32Impl = pixelStats32x32DotProd
		pixelStats64x16Impl = pixelStats64x16DotProd
		pixelStats16x64Impl = pixelStats16x64DotProd
		pixelStats64x32Impl = pixelStats64x32DotProd
		pixelStats32x64Impl = pixelStats32x64DotProd
	}
}
