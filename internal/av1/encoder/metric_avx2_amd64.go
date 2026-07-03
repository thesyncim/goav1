//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// metric_avx2_amd64.go closes the amd64 gap for the encoder's pixel-domain
// distortion statistics (pixelStats*, the 20 active block shapes) and the
// SATD coefficient reducer (satdCoeffs). These mirror the NEON surface bound on
// arm64 and default to pure-Go on amd64.
//
// pixelStats returns (sse, sum) where d = src - ref per sample: sum = Σd (signed)
// and sse = Σd². The AVX2 kernels zero-extend bytes to int16 lanes, subtract to
// get the exact signed diff (|d| ≤ 255 fits int16), then fold with VPMADDWD:
// madd(diff, ones) yields the per-pair signed diff sum and madd(diff, diff)
// yields the per-pair squared sum — both exact integer arithmetic, so the result
// is bit-exact with pixelStatsPureGo by construction. This mirrors libaom's
// aom_variance / get_var AVX2 layout (aom_dsp/x86/variance_avx2.c).
//
// The differential tests call these functions directly, so hosts whose CPUID
// hides AVX2 (Rosetta 2) still exercise them.

// pixelStatsAVX2Ctx carries one metric block. Field offsets are mirrored by
// #define in metric_stats_avx2_amd64.s.
type pixelStatsAVX2Ctx struct {
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
func pixelStatsWideAVX2Asm(ctx *pixelStatsAVX2Ctx)

//go:noescape
func pixelStats8AVX2Asm(ctx *pixelStatsAVX2Ctx)

//go:noescape
func pixelStats4AVX2Asm(ctx *pixelStatsAVX2Ctx)

func pixelStatsWideAVX2(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	ctx := pixelStatsAVX2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         int64(w),
		H:         int64(h),
	}
	pixelStatsWideAVX2Asm(&ctx)
	return uint32(ctx.SSE), int32(ctx.Sum)
}

func pixelStats8AVX2(src []byte, srcStride int, ref []byte, refStride int, h int) (sse uint32, sum int32) {
	ctx := pixelStatsAVX2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         8,
		H:         int64(h),
	}
	pixelStats8AVX2Asm(&ctx)
	return uint32(ctx.SSE), int32(ctx.Sum)
}

func pixelStats4AVX2(src []byte, srcStride int, ref []byte, refStride int, h int) (sse uint32, sum int32) {
	ctx := pixelStatsAVX2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		SrcStride: int64(srcStride),
		Ref:       unsafe.Pointer(&ref[0]),
		RefStride: int64(refStride),
		W:         4,
		H:         int64(h),
	}
	pixelStats4AVX2Asm(&ctx)
	return uint32(ctx.SSE), int32(ctx.Sum)
}

func pixelStats8x8AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8AVX2(src, srcStride, ref, refStride, 8)
}

func pixelStats4x4AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4AVX2(src, srcStride, ref, refStride, 4)
}

func pixelStats8x4AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8AVX2(src, srcStride, ref, refStride, 4)
}

func pixelStats4x8AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4AVX2(src, srcStride, ref, refStride, 8)
}

func pixelStats16x8AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats8x16AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8AVX2(src, srcStride, ref, refStride, 16)
}

func pixelStats16x4AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 16, 4)
}

func pixelStats4x16AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4AVX2(src, srcStride, ref, refStride, 16)
}

func pixelStats16x16AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x8AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 32, 8)
}

func pixelStats8x32AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8AVX2(src, srcStride, ref, refStride, 32)
}

func pixelStats32x16AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats16x32AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats32x32AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats64x16AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 64, 16)
}

func pixelStats16x64AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 16, 64)
}

func pixelStats64x32AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 64, 32)
}

func pixelStats32x64AVX2(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStatsWideAVX2(src, srcStride, ref, refStride, 32, 64)
}

// satdCoeffsAVX2Ctx carries the int32 coefficient SATD reducer arguments. Field
// offsets are mirrored by #define in metric_stats_avx2_amd64.s.
type satdCoeffsAVX2Ctx struct {
	Coeff unsafe.Pointer
	Count int64
	Sum   int64
}

//go:noescape
func satdCoeffsAVX2Asm(ctx *satdCoeffsAVX2Ctx)

func satdCoeffsAVX2(coeff []int32, count int) int {
	ctx := satdCoeffsAVX2Ctx{
		Coeff: unsafe.Pointer(&coeff[0]),
		Count: int64(count),
	}
	satdCoeffsAVX2Asm(&ctx)
	return int(ctx.Sum)
}

func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	pixelStats8x8Impl = pixelStats8x8AVX2
	pixelStats4x4Impl = pixelStats4x4AVX2
	pixelStats8x4Impl = pixelStats8x4AVX2
	pixelStats4x8Impl = pixelStats4x8AVX2
	pixelStats16x8Impl = pixelStats16x8AVX2
	pixelStats8x16Impl = pixelStats8x16AVX2
	pixelStats16x4Impl = pixelStats16x4AVX2
	pixelStats4x16Impl = pixelStats4x16AVX2
	pixelStats16x16Impl = pixelStats16x16AVX2
	pixelStats32x8Impl = pixelStats32x8AVX2
	pixelStats8x32Impl = pixelStats8x32AVX2
	pixelStats32x16Impl = pixelStats32x16AVX2
	pixelStats16x32Impl = pixelStats16x32AVX2
	pixelStats32x32Impl = pixelStats32x32AVX2
	pixelStats64x16Impl = pixelStats64x16AVX2
	pixelStats16x64Impl = pixelStats16x64AVX2
	pixelStats64x32Impl = pixelStats64x32AVX2
	pixelStats32x64Impl = pixelStats32x64AVX2
	satdCoeffsImpl = satdCoeffsAVX2
}
