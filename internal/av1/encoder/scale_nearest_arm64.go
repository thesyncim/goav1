//go:build arm64 && !purego

package encoder

import "unsafe"

type scaleNearestRowCtx struct {
	Dst    unsafe.Pointer
	Src    unsafe.Pointer
	Groups int64
}

//go:noescape
func scaleNearestRowDown2NEON(ctx *scaleNearestRowCtx)

//go:noescape
func scaleNearestRowDown4NEON(ctx *scaleNearestRowCtx)

func init() {
	scalePlaneNearestImpl = scalePlaneNearestARM64
}

func scalePlaneNearestARM64(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	switch {
	case srcWidth == dstWidth*2 && srcHeight == dstHeight*2:
		scalePlaneNearestDown2ARM64(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	case srcWidth == dstWidth*4 && srcHeight == dstHeight*4:
		scalePlaneNearestDown4ARM64(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	default:
		scalePlaneNearestPureGo(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
	}
}

func scalePlaneNearestDown2ARM64(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*2)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*2-1]
			ctx := scaleNearestRowCtx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 16),
			}
			scaleNearestRowDown2NEON(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*2]
		}
	}
}

func scalePlaneNearestDown4ARM64(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*4)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*4-1]
			ctx := scaleNearestRowCtx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 16),
			}
			scaleNearestRowDown4NEON(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*4]
		}
	}
}
