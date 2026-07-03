//go:build amd64 && !purego

package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// scale_nearest_avx2_amd64.go closes the amd64 gap for nearest-neighbour plane
// scaling (scalePlaneNearest / scalePlaneNearest16), mirroring the NEON kernels
// bound on arm64. As on arm64 only the exact 2x and 4x downscales are
// vectorized (every 2nd / 4th sample, the realtime scaler's hot ratios); any
// other ratio falls back to the portable path. The AVX2 row kernels pick the
// wanted lanes with in-range PAND masks and PACKUSDW/PACKUSWB packs (all values
// stay within range so the saturating packs are lossless), so they are bit-exact
// with scalePlaneNearestPureGo.
//
// The differential tests call the kernels directly (ungated on CPUID) so hosts
// that hide AVX2 (Rosetta 2) still prove bit-exactness.

// scaleNearestRowAVX2Ctx carries one row's pick arguments. Field offsets are
// mirrored by #define in scale_nearest_avx2_amd64.s.
type scaleNearestRowAVX2Ctx struct {
	Dst    unsafe.Pointer
	Src    unsafe.Pointer
	Groups int64
}

//go:noescape
func scaleNearestRowDown2AVX2(ctx *scaleNearestRowAVX2Ctx)

//go:noescape
func scaleNearestRowDown4AVX2(ctx *scaleNearestRowAVX2Ctx)

//go:noescape
func scaleNearestRow16Down2AVX2(ctx *scaleNearestRowAVX2Ctx)

//go:noescape
func scaleNearestRow16Down4AVX2(ctx *scaleNearestRowAVX2Ctx)

func scalePlaneNearestAVX2(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	switch {
	case srcWidth == dstWidth*2 && srcHeight == dstHeight*2:
		scalePlaneNearestDown2AVX2(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	case srcWidth == dstWidth*4 && srcHeight == dstHeight*4:
		scalePlaneNearestDown4AVX2(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	default:
		scalePlaneNearestPureGo(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
	}
}

func scalePlaneNearestDown2AVX2(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*2)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*2-1]
			ctx := scaleNearestRowAVX2Ctx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 16),
			}
			scaleNearestRowDown2AVX2(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*2]
		}
	}
}

func scalePlaneNearestDown4AVX2(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*4)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*4-1]
			ctx := scaleNearestRowAVX2Ctx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 16),
			}
			scaleNearestRowDown4AVX2(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*4]
		}
	}
}

func scalePlaneNearest16AVX2(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride, srcWidth, srcHeight int) {
	switch {
	case srcWidth == dstWidth*2 && srcHeight == dstHeight*2:
		scalePlaneNearest16Down2AVX2(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	case srcWidth == dstWidth*4 && srcHeight == dstHeight*4:
		scalePlaneNearest16Down4AVX2(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	default:
		scalePlaneNearest16PureGo(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
	}
}

func scalePlaneNearest16Down2AVX2(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride int) {
	vectorWidth := dstWidth &^ 7
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*2)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*2-1]
			ctx := scaleNearestRowAVX2Ctx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 8),
			}
			scaleNearestRow16Down2AVX2(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*2]
		}
	}
}

func scalePlaneNearest16Down4AVX2(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride int) {
	vectorWidth := dstWidth &^ 7
	for y := 0; y < dstHeight; y++ {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*4)*srcStride:]
		if vectorWidth > 0 {
			_ = drow[vectorWidth-1]
			_ = srow[vectorWidth*4-1]
			ctx := scaleNearestRowAVX2Ctx{
				Dst:    unsafe.Pointer(&drow[0]),
				Src:    unsafe.Pointer(&srow[0]),
				Groups: int64(vectorWidth / 8),
			}
			scaleNearestRow16Down4AVX2(&ctx)
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*4]
		}
	}
}

func init() {
	if !cpu.Detected.AVX2 {
		return
	}
	scalePlaneNearestImpl = scalePlaneNearestAVX2
	scalePlaneNearest16Impl = scalePlaneNearest16AVX2
}
