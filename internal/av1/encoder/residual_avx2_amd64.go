//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// residual_avx2_amd64.go closes the amd64 gap for the encoder's prediction
// residual extraction (residualBlock), mirroring the NEON kernel bound on arm64.
// dst[r*w+c] = int16(src) - int16(pred); bytes are zero-extended to int16 lanes
// and subtracted (|diff| ≤ 255 fits int16), so the packed int16 result is
// bit-exact with residualBlockPureGo. The kernel's narrowest row form is eight
// columns; narrower widths fall back to the portable path exactly as NEON does.
//
// The differential test calls the kernel directly (ungated on CPUID) so hosts
// that hide AVX2 (Rosetta 2) still prove bit-exactness.

// residualAVX2Ctx carries the residual-extraction kernel arguments. Field
// offsets are mirrored by #define in residual_avx2_amd64.s.
type residualAVX2Ctx struct {
	Src        unsafe.Pointer
	SrcStride  int64
	Pred       unsafe.Pointer
	PredStride int64
	Dst        unsafe.Pointer
	W          int64
	H          int64
}

//go:noescape
func residualBlockAVX2Asm(ctx *residualAVX2Ctx)

func residualBlockAVX2(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	if w < 8 {
		residualBlockPureGo(dst, srcPlane, srcOff, stride, pred, predStride, w, h)
		return
	}
	ctx := residualAVX2Ctx{
		Src:        unsafe.Pointer(&srcPlane[srcOff]),
		SrcStride:  int64(stride),
		Pred:       unsafe.Pointer(&pred[0]),
		PredStride: int64(predStride),
		Dst:        unsafe.Pointer(&dst[0]),
		W:          int64(w),
		H:          int64(h),
	}
	residualBlockAVX2Asm(&ctx)
}

func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	residualBlockImpl = residualBlockAVX2
}
