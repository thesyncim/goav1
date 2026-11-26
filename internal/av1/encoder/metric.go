package encoder

// metric.go hosts pixel-domain distortion helpers mirroring the active square
// sizes used by the inter encoder. They are kept as statistics kernels
// (SSE plus signed residual sum) so variance follows the AV1/AOM formula:
// SSE - (sum*sum >> log2(width*height)).

var pixelStats8x8Impl = pixelStats8x8PureGo
var pixelStats4x4Impl = pixelStats4x4PureGo
var pixelStats16x8Impl = pixelStats16x8PureGo
var pixelStats8x16Impl = pixelStats8x16PureGo
var pixelStats16x16Impl = pixelStats16x16PureGo
var pixelStats32x16Impl = pixelStats32x16PureGo
var pixelStats16x32Impl = pixelStats16x32PureGo
var pixelStats32x32Impl = pixelStats32x32PureGo
var pixelStats64x32Impl = pixelStats64x32PureGo
var pixelStats32x64Impl = pixelStats32x64PureGo

func pixelStats8x8PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 8, 8)
}

func pixelStats4x4PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 4, 4)
}

func pixelStats16x8PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 16, 8)
}

func pixelStats8x16PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 8, 16)
}

func pixelStats16x16PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 16, 16)
}

func pixelStats32x16PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 32, 16)
}

func pixelStats16x32PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 16, 32)
}

func pixelStats32x32PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 32, 32)
}

func pixelStats64x32PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 64, 32)
}

func pixelStats32x64PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 32, 64)
}

func pixelStatsPureGo(src []byte, srcStride int, ref []byte, refStride int, w, h int) (sse uint32, sum int32) {
	total := 0
	sumInt := 0
	for r := range h {
		a := src[r*srcStride : r*srcStride+w]
		b := ref[r*refStride : r*refStride+w]
		c := 0
		for ; c+3 < w; c += 4 {
			d0 := int(a[c]) - int(b[c])
			d1 := int(a[c+1]) - int(b[c+1])
			d2 := int(a[c+2]) - int(b[c+2])
			d3 := int(a[c+3]) - int(b[c+3])
			sumInt += d0 + d1 + d2 + d3
			total += d0*d0 + d1*d1 + d2*d2 + d3*d3
		}
		for ; c < w; c++ {
			d := int(a[c]) - int(b[c])
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

func sseVariance4x4(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats4x4(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 4)
}

func sseVariance16x8(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats16x8(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 7)
}

func sseVariance8x16(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats8x16(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 7)
}

func sseVariance16x16(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats16x16(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 8)
}

func sseVariance32x16(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats32x16(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 9)
}

func sseVariance16x32(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, variance uint32) {
	sse, sum := pixelStats16x32(src, srcStride, ref, refStride)
	return sse, varianceFromStats(sse, sum, 9)
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
