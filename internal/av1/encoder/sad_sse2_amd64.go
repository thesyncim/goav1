//go:build amd64 && !purego

package encoder

import "unsafe"

// sad8x8SSE2Ctx carries the kernel arguments; the assembly writes the sum
// into Sum. Field offsets are mirrored by #define in sad_sse2_amd64.s.
type sad8x8SSE2Ctx struct {
	Src    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Sum    int64
}

//go:noescape
func sad8x8SSE2Asm(ctx *sad8x8SSE2Ctx)

// sad8x8SSE2 computes the full 8x8 SAD with PSADBW row pairs. SSE2 is part of
// the amd64 baseline, so no feature detection is needed. The limit hint is
// ignored: the vector kernel finishes the block faster than the scalar early
// exit can help.
func sad8x8SSE2(src, ref []byte, stride int, limit int) int {
	ctx := sad8x8SSE2Ctx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad8x8SSE2Asm(&ctx)
	return int(ctx.Sum)
}

//go:noescape
func sad16x16SSE2Asm(ctx *sad8x8SSE2Ctx)

// sad16x16SSE2 computes the full 16x16 SAD with one PSADBW per 16-byte row.
func sad16x16SSE2(src, ref []byte, stride int) int {
	ctx := sad8x8SSE2Ctx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad16x16SSE2Asm(&ctx)
	return int(ctx.Sum)
}

// sad8x8DualSSE2Ctx carries the two-stride kernel arguments.
type sad8x8DualSSE2Ctx struct {
	Src       unsafe.Pointer
	Ref       unsafe.Pointer
	SrcStride int64
	RefStride int64
	Sum       int64
}

//go:noescape
func sad8x8DualSSE2Asm(ctx *sad8x8DualSSE2Ctx)

// sad8x8DualSSE2 is the SSE2 8x8 SAD with independent source and reference
// strides.
func sad8x8DualSSE2(src []byte, srcStride int, ref []byte, refStride int) int {
	ctx := sad8x8DualSSE2Ctx{
		Src:       unsafe.Pointer(&src[0]),
		Ref:       unsafe.Pointer(&ref[0]),
		SrcStride: int64(srcStride),
		RefStride: int64(refStride),
	}
	sad8x8DualSSE2Asm(&ctx)
	return int(ctx.Sum)
}

func init() {
	sad8x8Impl = sad8x8SSE2
	sad16x16Impl = sad16x16SSE2
	sad8x8DualImpl = sad8x8DualSSE2
}
