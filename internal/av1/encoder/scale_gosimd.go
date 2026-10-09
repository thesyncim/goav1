//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

// scale_gosimd.go hosts the Go-native SIMD nearest-neighbour plane scalers.
// As in the retired NEON and AVX2 kernels, only the exact 2x and 4x downscales
// are vectorized (every 2nd / 4th sample); any other ratio falls back to the
// scalar reference. The per-row picks are arch-specific (scale_gosimd_arm64.go
// uses ConcatEven, scale_gosimd_amd64.go uses PermuteOrZero).

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	scalePlaneNearestImpl = scalePlaneNearestSIMD
	scalePlaneNearest16Impl = scalePlaneNearest16SIMD
}

func scalePlaneNearestSIMD(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	switch {
	case srcWidth == dstWidth*2 && srcHeight == dstHeight*2:
		scaleNearestDown2SIMD(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	case srcWidth == dstWidth*4 && srcHeight == dstHeight*4:
		scaleNearestDown4SIMD(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	default:
		scalePlaneNearestPureGo(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
	}
}

func scaleNearestDown2SIMD(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 31
	for y := range dstHeight {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*2)*srcStride:]
		if vectorWidth > 0 {
			scaleRow8Down2(drow[:vectorWidth], srow[:2*vectorWidth])
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*2]
		}
	}
}

func scaleNearestDown4SIMD(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := range dstHeight {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*4)*srcStride:]
		if vectorWidth > 0 {
			scaleRow8Down4(drow[:vectorWidth], srow[:4*vectorWidth])
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*4]
		}
	}
}

func scalePlaneNearest16SIMD(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride, srcWidth, srcHeight int) {
	switch {
	case srcWidth == dstWidth*2 && srcHeight == dstHeight*2:
		scaleNearest16Down2SIMD(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	case srcWidth == dstWidth*4 && srcHeight == dstHeight*4:
		scaleNearest16Down4SIMD(dst, dstStride, dstWidth, dstHeight, src, srcStride)
	default:
		scalePlaneNearest16PureGo(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
	}
}

func scaleNearest16Down2SIMD(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride int) {
	vectorWidth := dstWidth &^ 15
	for y := range dstHeight {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*2)*srcStride:]
		if vectorWidth > 0 {
			scaleRow16Down2(drow[:vectorWidth], srow[:2*vectorWidth])
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*2]
		}
	}
}

func scaleNearest16Down4SIMD(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride int) {
	vectorWidth := dstWidth &^ 7
	for y := range dstHeight {
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[(y*4)*srcStride:]
		if vectorWidth > 0 {
			scaleRow16Down4(drow[:vectorWidth], srow[:4*vectorWidth])
		}
		for x := vectorWidth; x < dstWidth; x++ {
			drow[x] = srow[x*4]
		}
	}
}
