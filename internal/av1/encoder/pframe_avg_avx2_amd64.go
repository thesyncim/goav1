//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// pframe_avg_avx2_amd64.go closes the amd64 gap for the realtime partitioning
// 8x8 average (realtimeAvg8x8 and its 2x2 quad), mirroring the NEON kernels
// bound on arm64. Each average is Σbytes over an 8x8 block rounded as
// (sum + 32) >> 6. Byte sums are exact via (V)PSADBW against zero, so the result
// is bit-exact with realtimeAvg8x8PureGo.
//
// The differential tests call the kernels directly (ungated on CPUID) so hosts
// that hide AVX2 (Rosetta 2) still prove bit-exactness.

// realtimeAvgAVX2Ctx carries the average kernel arguments; the single kernel
// writes Out0, the quad kernel writes Out0..Out3. Field offsets are mirrored by
// #define in pframe_avg_avx2_amd64.s.
type realtimeAvgAVX2Ctx struct {
	Src    unsafe.Pointer
	Stride int64
	Out0   int64
	Out1   int64
	Out2   int64
	Out3   int64
}

//go:noescape
func realtimeAvg8x8AVX2Asm(ctx *realtimeAvgAVX2Ctx)

//go:noescape
func realtimeAvg8x8QuadAVX2Asm(ctx *realtimeAvgAVX2Ctx)

func realtimeAvg8x8AVX2(src []byte, stride int) int {
	ctx := realtimeAvgAVX2Ctx{Src: unsafe.Pointer(&src[0]), Stride: int64(stride)}
	realtimeAvg8x8AVX2Asm(&ctx)
	return int(ctx.Out0)
}

func realtimeAvg8x8QuadAVX2(src []byte, stride int) (int, int, int, int) {
	ctx := realtimeAvgAVX2Ctx{Src: unsafe.Pointer(&src[0]), Stride: int64(stride)}
	realtimeAvg8x8QuadAVX2Asm(&ctx)
	return int(ctx.Out0), int(ctx.Out1), int(ctx.Out2), int(ctx.Out3)
}

func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	realtimeAvg8x8Impl = realtimeAvg8x8AVX2
	realtimeAvg8x8QuadImpl = realtimeAvg8x8QuadAVX2
}
