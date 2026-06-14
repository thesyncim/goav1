package encoder

// metric.go hosts pixel-domain distortion helpers mirroring the active square
// sizes used by the inter encoder. They are kept as statistics kernels
// (SSE plus signed residual sum) so variance follows the AV1/AOM formula:
// SSE - (sum*sum >> log2(width*height)).

var pixelStats8x8Impl = pixelStats8x8PureGo
var pixelStats16x8Impl = pixelStats16x8PureGo
var pixelStats8x16Impl = pixelStats8x16PureGo
var pixelStats16x16Impl = pixelStats16x16PureGo
var pixelStats32x16Impl = pixelStats32x16PureGo
var pixelStats16x32Impl = pixelStats16x32PureGo
var pixelStats32x32Impl = pixelStats32x32PureGo
var satdCoeffsImpl = satdCoeffsPureGo
var hadamard4x4Impl = hadamard4x4PureGo
var hadamard8x8Impl = hadamard8x8PureGo
var hadamard16x16Impl = hadamard16x16PureGo
var hadamard32x32Impl = hadamard32x32PureGo

func pixelStats8x8PureGo(src []byte, srcStride int, ref []byte, refStride int) (sse uint32, sum int32) {
	return pixelStatsPureGo(src, srcStride, ref, refStride, 8, 8)
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

// hadamard4x4PureGo mirrors SVT's svt_aom_hadamard_4x4_c low-bitdepth
// producer, including the final transpose used to match the SSE2 coefficient
// order.
func hadamard4x4PureGo(src []int16, srcStride int, coeff []int32) {
	_ = src[3*srcStride+3]
	_ = coeff[15]
	var buffer [16]int16
	var buffer2 [16]int16
	for idx := range 4 {
		hadamardCol4(src[idx:], srcStride, buffer[idx*4:])
	}
	for idx := range 4 {
		hadamardCol4(buffer[idx:], 4, buffer2[idx*4:])
	}
	for i := range 4 {
		for j := range 4 {
			coeff[i*4+j] = int32(buffer2[j*4+i])
		}
	}
}

func hadamardCol4(src []int16, srcStride int, coeff []int16) {
	_ = src[3*srcStride]
	_ = coeff[3]
	b0 := (src[0*srcStride] + src[1*srcStride]) >> 1
	b1 := (src[0*srcStride] - src[1*srcStride]) >> 1
	b2 := (src[2*srcStride] + src[3*srcStride]) >> 1
	b3 := (src[2*srcStride] - src[3*srcStride]) >> 1

	coeff[0] = b0 + b2
	coeff[1] = b1 + b3
	coeff[2] = b0 - b2
	coeff[3] = b1 - b3
}

// hadamard8x8PureGo mirrors SVT's svt_aom_hadamard_8x8_c low-bitdepth
// producer. SVT allows SIMD implementations to use a transposed coefficient
// order because the consumer is SATD, which is order-invariant.
func hadamard8x8PureGo(src []int16, srcStride int, coeff []int32) {
	_ = src[7*srcStride+7]
	_ = coeff[63]
	var buffer [64]int16
	var buffer2 [64]int16
	for idx := range 8 {
		hadamardCol8(src[idx:], srcStride, buffer[idx*8:])
	}
	for idx := range 8 {
		hadamardCol8(buffer[idx:], 8, buffer2[idx*8:])
	}
	for idx := range 64 {
		coeff[idx] = int32(buffer2[idx])
	}
}

func hadamardCol8(src []int16, srcStride int, coeff []int16) {
	_ = src[7*srcStride]
	_ = coeff[7]
	b0 := src[0*srcStride] + src[1*srcStride]
	b1 := src[0*srcStride] - src[1*srcStride]
	b2 := src[2*srcStride] + src[3*srcStride]
	b3 := src[2*srcStride] - src[3*srcStride]
	b4 := src[4*srcStride] + src[5*srcStride]
	b5 := src[4*srcStride] - src[5*srcStride]
	b6 := src[6*srcStride] + src[7*srcStride]
	b7 := src[6*srcStride] - src[7*srcStride]

	c0 := b0 + b2
	c1 := b1 + b3
	c2 := b0 - b2
	c3 := b1 - b3
	c4 := b4 + b6
	c5 := b5 + b7
	c6 := b4 - b6
	c7 := b5 - b7

	coeff[0] = c0 + c4
	coeff[7] = c1 + c5
	coeff[3] = c2 + c6
	coeff[4] = c3 + c7
	coeff[2] = c0 - c4
	coeff[6] = c1 - c5
	coeff[1] = c2 - c6
	coeff[5] = c3 - c7
}

// hadamard16x16PureGo mirrors SVT's svt_aom_hadamard_16x16_c: four 8x8
// producers followed by the source-shaped quadrant combine.
func hadamard16x16PureGo(src []int16, srcStride int, coeff []int32) {
	_ = src[15*srcStride+15]
	_ = coeff[255]
	hadamard8x8PureGo(src, srcStride, coeff)
	hadamard8x8PureGo(src[8:], srcStride, coeff[64:])
	hadamard8x8PureGo(src[8*srcStride:], srcStride, coeff[128:])
	hadamard8x8PureGo(src[8*srcStride+8:], srcStride, coeff[192:])

	for idx := range 64 {
		a0 := coeff[idx]
		a1 := coeff[64+idx]
		a2 := coeff[128+idx]
		a3 := coeff[192+idx]

		b0 := (a0 + a1) >> 1
		b1 := (a0 - a1) >> 1
		b2 := (a2 + a3) >> 1
		b3 := (a2 - a3) >> 1

		coeff[idx] = b0 + b2
		coeff[64+idx] = b1 + b3
		coeff[128+idx] = b0 - b2
		coeff[192+idx] = b1 - b3
	}
}

// hadamard32x32PureGo mirrors SVT's svt_aom_hadamard_32x32_c: four 16x16
// producers followed by the source-shaped quadrant combine.
func hadamard32x32PureGo(src []int16, srcStride int, coeff []int32) {
	_ = src[31*srcStride+31]
	_ = coeff[1023]
	hadamard16x16PureGo(src, srcStride, coeff)
	hadamard16x16PureGo(src[16:], srcStride, coeff[256:])
	hadamard16x16PureGo(src[16*srcStride:], srcStride, coeff[512:])
	hadamard16x16PureGo(src[16*srcStride+16:], srcStride, coeff[768:])

	for idx := range 256 {
		a0 := coeff[idx]
		a1 := coeff[256+idx]
		a2 := coeff[512+idx]
		a3 := coeff[768+idx]

		b0 := (a0 + a1) >> 2
		b1 := (a0 - a1) >> 2
		b2 := (a2 + a3) >> 2
		b3 := (a2 - a3) >> 2

		coeff[idx] = b0 + b2
		coeff[256+idx] = b1 + b3
		coeff[512+idx] = b0 - b2
		coeff[768+idx] = b1 - b3
	}
}
