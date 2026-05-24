package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	warpedModelPrecBits           = 16
	warpedModelOne                = 1 << warpedModelPrecBits
	warpedModelTransClamp         = 128 << warpedModelPrecBits
	warpedModelNonDiagAffineClamp = 1 << (warpedModelPrecBits - 3)
	warpParamReduceBits           = 6
	warpDivLUTPrecBits            = 14
	warpDivLUTBits                = 8
)

// WarpedMotionModel carries a decoded local WARPED_CAUSAL affine model and the
// reduced shear parameters required by AV1's warp filter.
type WarpedMotionModel struct {
	Params       parser.WarpedMotionParams
	Alpha, Beta  int16
	Gamma, Delta int16
}

func (s OverlappableNeighborSet) WarpProjection(block BlockVisit, ref ReferenceFrame, mv motion.Vector) (WarpedMotionModel, bool, error) {
	var samples [maxWarpSamples]warpSample
	count, err := s.warpSamples(block, ref, &samples)
	if err != nil {
		return WarpedMotionModel{}, false, err
	}
	if count == 0 {
		return WarpedMotionModel{}, true, nil
	}
	if count > 1 {
		count = selectWarpSamples(samples[:count], block.Size, mv)
	}
	return findWarpProjection(samples[:count], block, mv)
}

func findWarpProjection(samples []warpSample, block BlockVisit, mv motion.Vector) (WarpedMotionModel, bool, error) {
	model, invalid, err := findWarpAffineInt(samples, block, mv)
	if invalid || err != nil {
		return model, invalid, err
	}
	model, ok := warpShearParams(model)
	return model, !ok, nil
}

func findWarpAffineInt(samples []warpSample, block BlockVisit, mv motion.Vector) (WarpedMotionModel, bool, error) {
	dims, ok := block.Size.Dimensions()
	if !ok {
		return WarpedMotionModel{}, false, ErrInvalidDecodeState
	}
	bw := int(dims.W4) * 4
	bh := int(dims.H4) * 4
	rsuy := bh/2 - 1
	rsux := bw/2 - 1
	suy := rsuy * motion.SubpelScale
	sux := rsux * motion.SubpelScale
	duy := suy + int(mv.Row)
	dux := sux + int(mv.Col)
	var a00, a01, a11 int64
	var bx0, bx1, by0, by1 int64
	for i := range samples {
		dx := samples[i].RefX - dux
		dy := samples[i].RefY - duy
		sx := samples[i].X - sux
		sy := samples[i].Y - suy
		if absInt(sx-dx) < warpLeastSquaresMVMax && absInt(sy-dy) < warpLeastSquaresMVMax {
			a00 += int64(warpLeastSquaresSquare(sx))
			a01 += int64(warpLeastSquaresProduct1(sx, sy))
			a11 += int64(warpLeastSquaresSquare(sy))
			bx0 += int64(warpLeastSquaresProduct2(sx, dx))
			bx1 += int64(warpLeastSquaresProduct1(sy, dx))
			by0 += int64(warpLeastSquaresProduct1(sx, dy))
			by1 += int64(warpLeastSquaresProduct2(sy, dy))
		}
	}

	det := a00*a11 - a01*a01
	if det == 0 {
		return WarpedMotionModel{}, true, nil
	}

	reciprocal, shift := warpResolveDivisor64(absInt64AsUint(det))
	iDet := int64(reciprocal)
	if det < 0 {
		iDet = -iDet
	}
	shift -= warpedModelPrecBits
	if shift < 0 {
		iDet <<= -shift
		shift = 0
	}

	px0 := a11*bx0 - a01*bx1
	px1 := -a01*bx0 + a00*bx1
	py0 := a11*by0 - a01*by1
	py1 := -a01*by0 + a00*by1

	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[2] = warpMultShiftDiag(px0, iDet, shift)
	params.Matrix[3] = warpMultShiftNonDiag(px1, iDet, shift)
	params.Matrix[4] = warpMultShiftNonDiag(py0, iDet, shift)
	params.Matrix[5] = warpMultShiftDiag(py1, iDet, shift)

	isuy := int(block.MIRow)*4 + rsuy
	isux := int(block.MICol)*4 + rsux
	vx := int64(mv.Col)*(1<<(warpedModelPrecBits-motion.SubpelBits)) -
		(int64(isux)*(int64(params.Matrix[2])-warpedModelOne) + int64(isuy)*int64(params.Matrix[3]))
	vy := int64(mv.Row)*(1<<(warpedModelPrecBits-motion.SubpelBits)) -
		(int64(isux)*int64(params.Matrix[4]) + int64(isuy)*(int64(params.Matrix[5])-warpedModelOne))
	params.Matrix[0] = int32(clampInt64(vx, -warpedModelTransClamp, warpedModelTransClamp-1))
	params.Matrix[1] = int32(clampInt64(vy, -warpedModelTransClamp, warpedModelTransClamp-1))
	return WarpedMotionModel{Params: params}, false, nil
}

func warpShearParams(model WarpedMotionModel) (WarpedMotionModel, bool) {
	mat := model.Params.Matrix
	if mat[2] <= 0 {
		return model, false
	}
	alpha := clampInt64(int64(mat[2])-warpedModelOne, -1<<15, 1<<15-1)
	beta := clampInt64(int64(mat[3]), -1<<15, 1<<15-1)
	reciprocal, shift := warpResolveDivisor32(uint32(absInt(int(mat[2]))))
	y := int64(reciprocal)
	if mat[2] < 0 {
		y = -y
	}
	v := int64(mat[4]) * warpedModelOne * y
	gamma := clampInt64(warpRoundPowerOfTwoSigned64(v, shift), -1<<15, 1<<15-1)
	v = int64(mat[3]) * int64(mat[4]) * y
	delta := clampInt64(int64(mat[5])-warpRoundPowerOfTwoSigned64(v, shift)-warpedModelOne, -1<<15, 1<<15-1)

	model.Alpha = int16(warpRoundPowerOfTwoSigned64(alpha, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Beta = int16(warpRoundPowerOfTwoSigned64(beta, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Gamma = int16(warpRoundPowerOfTwoSigned64(gamma, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Delta = int16(warpRoundPowerOfTwoSigned64(delta, warpParamReduceBits) * (1 << warpParamReduceBits))
	return model, warpAffineShearAllowed(model.Alpha, model.Beta, model.Gamma, model.Delta)
}

func warpAffineShearAllowed(alpha int16, beta int16, gamma int16, delta int16) bool {
	return 4*absInt(int(alpha))+7*absInt(int(beta)) < warpedModelOne &&
		4*absInt(int(gamma))+4*absInt(int(delta)) < warpedModelOne
}

func warpMultShiftNonDiag(px int64, iDet int64, shift int) int32 {
	v := px * iDet
	return int32(clampInt64(
		warpRoundPowerOfTwoSigned64(v, shift),
		-warpedModelNonDiagAffineClamp+1,
		warpedModelNonDiagAffineClamp-1,
	))
}

func warpMultShiftDiag(px int64, iDet int64, shift int) int32 {
	v := px * iDet
	return int32(clampInt64(
		warpRoundPowerOfTwoSigned64(v, shift),
		warpedModelOne-warpedModelNonDiagAffineClamp+1,
		warpedModelOne+warpedModelNonDiagAffineClamp-1,
	))
}

func warpLeastSquaresProduct2(a int, b int) int {
	return (a*b*4 + (a+b)*2*motion.SubpelScale + motion.SubpelScale*motion.SubpelScale*2) >> 4
}

func warpResolveDivisor64(d uint64) (int16, int) {
	shift := bits.Len64(d) - 1
	e := d - (uint64(1) << shift)
	var f uint64
	if shift > warpDivLUTBits {
		f = warpRoundPowerOfTwoUnsigned64(e, shift-warpDivLUTBits)
	} else {
		f = e << (warpDivLUTBits - shift)
	}
	return int16(warpDivLUT[f]), shift + warpDivLUTPrecBits
}

func warpResolveDivisor32(d uint32) (int16, int) {
	shift := bits.Len32(d) - 1
	e := d - (uint32(1) << shift)
	var f uint32
	if shift > warpDivLUTBits {
		f = uint32(warpRoundPowerOfTwoUnsigned64(uint64(e), shift-warpDivLUTBits))
	} else {
		f = e << (warpDivLUTBits - shift)
	}
	return int16(warpDivLUT[f]), shift + warpDivLUTPrecBits
}

func warpRoundPowerOfTwoUnsigned64(value uint64, bits int) uint64 {
	if bits <= 0 {
		return value
	}
	return (value + (uint64(1) << (bits - 1))) >> bits
}

func warpRoundPowerOfTwoSigned64(value int64, bits int) int64 {
	if bits <= 0 {
		return value
	}
	if value < 0 {
		return -int64(warpRoundPowerOfTwoUnsigned64(uint64(-value), bits))
	}
	return int64(warpRoundPowerOfTwoUnsigned64(uint64(value), bits))
}

func absInt64AsUint(v int64) uint64 {
	if v < 0 {
		return uint64(-v)
	}
	return uint64(v)
}

var warpDivLUT = [257]uint16{
	16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768,
	15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142,
	15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564,
	14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028,
	13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662, 13618, 13574, 13530,
	13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107, 13066,
	13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633,
	12596, 12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228,
	12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848,
	11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
	11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155,
	11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
	10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
	10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255,
	10230, 10205, 10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,
	9963, 9939, 9916, 9892, 9869, 9846, 9823, 9800, 9777, 9754, 9732,
	9709, 9687, 9664, 9642, 9620, 9598, 9576, 9554, 9533, 9511, 9489,
	9468, 9447, 9425, 9404, 9383, 9362, 9341, 9321, 9300, 9279, 9259,
	9239, 9218, 9198, 9178, 9158, 9138, 9118, 9098, 9079, 9059, 9039,
	9020, 9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867, 8849, 8830,
	8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684, 8666, 8648, 8630,
	8613, 8595, 8577, 8560, 8542, 8525, 8508, 8490, 8473, 8456, 8439,
	8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306, 8289, 8273, 8257,
	8240, 8224, 8208, 8192,
}
