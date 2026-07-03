//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// sad_avx2_amd64.go binds the AVX2 SAD kernels that close the amd64 gap versus
// the arm64/NEON encoder surface. SSE2 already covers the single 8x8/16x16 and
// the 8x8 dual leaf; this file adds the motion-search x4 fan-out (sadNxNx4d and
// the raster step-4 variant), the wider single/dual leaves, and the compound
// average precheck. Every kernel is a straight VPSADBW accumulation, so it is
// bit-exact with the portable reference by construction (SAD is exact integer
// arithmetic). The differential tests call these functions directly, so hosts
// whose CPUID hides AVX2 (Rosetta 2) still exercise them.

// sadX4AMD64Ctx carries the four-reference kernel arguments; the assembly writes
// the four block sums into Sum0..Sum3. Field offsets are mirrored by #define in
// sad_avx2_amd64.s.
type sadX4AMD64Ctx struct {
	Src    unsafe.Pointer
	Ref0   unsafe.Pointer
	Ref1   unsafe.Pointer
	Ref2   unsafe.Pointer
	Ref3   unsafe.Pointer
	Stride int64
	Sum0   int64
	Sum1   int64
	Sum2   int64
	Sum3   int64
}

// sadStep4AMD64Ctx carries the raster step-4 kernel arguments: a single source
// and a single reference base, with the four candidates read at ref+0/+4/+8/+12.
type sadStep4AMD64Ctx struct {
	Src    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Sum0   int64
	Sum1   int64
	Sum2   int64
	Sum3   int64
}

//go:noescape
func sad8x8x4AVX2Asm(ctx *sadX4AMD64Ctx)

//go:noescape
func sad16x16x4AVX2Asm(ctx *sadX4AMD64Ctx)

//go:noescape
func sad32x32x4AVX2Asm(ctx *sadX4AMD64Ctx)

//go:noescape
func sad8x8x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)

//go:noescape
func sad16x16x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)

//go:noescape
func sad32x32x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)

func sad8x8x4AVX2(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sadX4AMD64Ctx{
		Src: unsafe.Pointer(&src[0]), Ref0: unsafe.Pointer(&ref0[0]),
		Ref1: unsafe.Pointer(&ref1[0]), Ref2: unsafe.Pointer(&ref2[0]),
		Ref3: unsafe.Pointer(&ref3[0]), Stride: int64(stride),
	}
	sad8x8x4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4AVX2(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sadX4AMD64Ctx{
		Src: unsafe.Pointer(&src[0]), Ref0: unsafe.Pointer(&ref0[0]),
		Ref1: unsafe.Pointer(&ref1[0]), Ref2: unsafe.Pointer(&ref2[0]),
		Ref3: unsafe.Pointer(&ref3[0]), Stride: int64(stride),
	}
	sad16x16x4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4AVX2(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	ctx := sadX4AMD64Ctx{
		Src: unsafe.Pointer(&src[0]), Ref0: unsafe.Pointer(&ref0[0]),
		Ref1: unsafe.Pointer(&ref1[0]), Ref2: unsafe.Pointer(&ref2[0]),
		Ref3: unsafe.Pointer(&ref3[0]), Stride: int64(stride),
	}
	sad32x32x4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad8x8x4Step4AVX2(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sadStep4AMD64Ctx{Src: unsafe.Pointer(&src[0]), Ref: unsafe.Pointer(&ref[0]), Stride: int64(stride)}
	sad8x8x4Step4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad16x16x4Step4AVX2(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sadStep4AMD64Ctx{Src: unsafe.Pointer(&src[0]), Ref: unsafe.Pointer(&ref[0]), Stride: int64(stride)}
	sad16x16x4Step4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

func sad32x32x4Step4AVX2(src, ref []byte, stride int) (int, int, int, int) {
	ctx := sadStep4AMD64Ctx{Src: unsafe.Pointer(&src[0]), Ref: unsafe.Pointer(&ref[0]), Stride: int64(stride)}
	sad32x32x4Step4AVX2Asm(&ctx)
	return int(ctx.Sum0), int(ctx.Sum1), int(ctx.Sum2), int(ctx.Sum3)
}

// initAVX2SAD upgrades the motion-search x4 fan-out slots to AVX2 when the CPU
// advertises it. The SSE2 init already ran (both files' init run at package
// load); this overrides only the slots AVX2 improves and leaves the SSE2 8x8/
// 16x16/8x8-dual leaves in place where they are already vector-optimal.
func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	sad8x8x4Impl = sad8x8x4AVX2
	sad16x16x4Impl = sad16x16x4AVX2
	sad32x32x4Impl = sad32x32x4AVX2
	sad8x8x4Step4Impl = sad8x8x4Step4AVX2
	sad16x16x4Step4Impl = sad16x16x4Step4AVX2
	sad32x32x4Step4Impl = sad32x32x4Step4AVX2
}
