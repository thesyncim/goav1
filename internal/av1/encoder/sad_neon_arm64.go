//go:build arm64 && !purego

package encoder

import "unsafe"

// sad8x8NEONCtx carries the kernel arguments; the assembly writes the sum into
// Sum. Field offsets are mirrored by #define in sad_neon_arm64.s.
type sad8x8NEONCtx struct {
	Src    unsafe.Pointer
	Ref    unsafe.Pointer
	Stride int64
	Sum    int64
}

//go:noescape
func sad8x8NEONAsm(ctx *sad8x8NEONCtx)

// sad8x8NEON computes the full 8x8 SAD with NEON widening absolute-difference
// accumulation. The limit hint is ignored: the vector kernel finishes the
// block faster than the scalar early exit can help.
func sad8x8NEON(src, ref []byte, stride int, limit int) int {
	ctx := sad8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Ref:    unsafe.Pointer(&ref[0]),
		Stride: int64(stride),
	}
	sad8x8NEONAsm(&ctx)
	return int(ctx.Sum)
}

func init() {
	sad8x8Impl = sad8x8NEON
}
