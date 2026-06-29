//go:build arm64 && !purego

package encoder

import "unsafe"

// realtimeAvg8x8NEONCtx carries libaom's aom_avg_8x8_neon arguments. Field
// offsets are mirrored by #define in pframe_avg_arm64.s.
type realtimeAvg8x8NEONCtx struct {
	Src    unsafe.Pointer
	Stride int64
	Avg    int64
}

//go:noescape
func realtimeAvg8x8NEONAsm(ctx *realtimeAvg8x8NEONCtx)

func realtimeAvg8x8NEON(src []byte, stride int) int {
	_ = src[7*stride+7]
	ctx := realtimeAvg8x8NEONCtx{
		Src:    unsafe.Pointer(&src[0]),
		Stride: int64(stride),
	}
	realtimeAvg8x8NEONAsm(&ctx)
	return int(ctx.Avg)
}

func realtimeAvg8x8QuadNEON(src []byte, stride int) (int, int, int, int) {
	_ = src[15*stride+15]
	return realtimeAvg8x8NEON(src, stride),
		realtimeAvg8x8NEON(src[8:], stride),
		realtimeAvg8x8NEON(src[8*stride:], stride),
		realtimeAvg8x8NEON(src[8*stride+8:], stride)
}

func init() {
	realtimeAvg8x8Impl = realtimeAvg8x8NEON
	realtimeAvg8x8QuadImpl = realtimeAvg8x8QuadNEON
}
