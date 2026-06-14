package encoder

// metric.go hosts pixel-domain distortion helpers mirroring the active square
// sizes used by the inter encoder. They are kept as statistics kernels
// (SSE plus signed residual sum) so variance follows the AV1/AOM formula:
// SSE - (sum*sum >> log2(width*height)).

var pixelStats8x8Impl = pixelStats8x8PureGo
var pixelStats16x16Impl = pixelStats16x16PureGo
var pixelStats32x32Impl = pixelStats32x32PureGo
var satdCoeffsImpl = satdCoeffsPureGo

func pixelStats8x8PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 8, 8)
}

func pixelStats16x16PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x32PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 32, 32)
}

func pixelStatsPureGo(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	total := 0
	sumInt := 0
	for r := range h {
		srow := r * srcStride
		rrow := r * refStride
		for c := range w {
			d := int(src[srow+c]) - int(ref[rrow+c])
			sumInt += d
			total += d * d
		}
	}
	return uint32(total), int32(sumInt)
}

func sseVariance8x8(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats8x8(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 6)
}

func sseVariance16x16(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats16x16(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 8)
}

func sseVariance32x32(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats32x32(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 10)
}

func sseVariance64x64(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	s0, m0 := pixelStats32x32(src, srcStride, ref, refStride)
	s1, m1 := pixelStats32x32(src[32:], srcStride, ref[32:], refStride)
	s2, m2 := pixelStats32x32(src[32*srcStride:], srcStride, ref[32*refStride:], refStride)
	s3, m3 := pixelStats32x32(src[32*srcStride+32:], srcStride, ref[32*refStride+32:], refStride)
	sse = s0 + s1 + s2 + s3
	sum := m0 + m1 + m2 + m3
	return sse, varianceFromStats(sse, sum, 12)
}

func varianceFromStats(sse uint32, sum int32, shift uint) uint32 {
	return sse - uint32((int64(sum)*int64(sum))>>shift)
}

// satdCoeffsPureGo mirrors SVT's svt_aom_satd_c reducer. The caller supplies
// the active coefficient count; SVT's SIMD reducer is specialized for counts
// that are multiples of 16, which covers the AV1 {16,64,256,1024} TX sizes.
func satdCoeffsPureGo(coeff []int32, count int) int {
	total := 0
	for i := 0; i < count; i++ {
		v := coeff[i]
		if v < 0 {
			v = -v
		}
		total += int(v)
	}
	return total
}
