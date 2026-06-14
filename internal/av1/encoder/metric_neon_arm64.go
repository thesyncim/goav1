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

func init() {
	pixelStats8x8Impl = pixelStats8x8NEON
	pixelStats16x16Impl = pixelStats16x16NEON
	pixelStats32x32Impl = pixelStats32x32NEON
	satdCoeffsImpl = satdCoeffsNEON
}
