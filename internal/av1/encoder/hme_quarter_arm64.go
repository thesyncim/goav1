//go:build arm64 && !purego

package encoder

import "unsafe"

// buildQuarterPlaneNEONCtx carries the 4:1 HME downsample kernel arguments.
// Field offsets are mirrored by hme_quarter_arm64.s.
type buildQuarterPlaneNEONCtx struct {
	Dst       unsafe.Pointer
	Src       unsafe.Pointer
	SrcStride int64
	DstStride int64
	Width     int64
	Height    int64
}

//go:noescape
func buildQuarterPlaneNEONAsm(ctx *buildQuarterPlaneNEONCtx)

func buildQuarterPlaneArch(dst []byte, src []byte, stride, qw, qh int) {
	neonCols := qw &^ 15
	if neonCols > 0 && qh > 0 {
		_ = dst[(qh-1)*qw+neonCols-1]
		_ = src[(qh*4-1)*stride+neonCols*4-1]
		ctx := buildQuarterPlaneNEONCtx{
			Dst:       unsafe.Pointer(&dst[0]),
			Src:       unsafe.Pointer(&src[0]),
			SrcStride: int64(stride),
			DstStride: int64(qw),
			Width:     int64(neonCols),
			Height:    int64(qh),
		}
		buildQuarterPlaneNEONAsm(&ctx)
	}
	if neonCols != qw {
		for qy := 0; qy < qh; qy++ {
			buildQuarterPlanePureGo(
				dst[qy*qw+neonCols:],
				src[qy*4*stride+neonCols*4:],
				stride,
				qw-neonCols,
				1,
			)
		}
	}
}
