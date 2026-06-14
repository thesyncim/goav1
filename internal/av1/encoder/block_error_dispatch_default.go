//go:build purego || !arm64

package encoder

func blockError(coeff []int32, dqcoeff []int32, count int) (err int64, ssz int64) {
	return blockErrorPureGo(coeff, dqcoeff, count)
}
