package encoder

// blockErrorPureGo mirrors SVT's NEON source shape, including the
// load_tran_low_to_s16q narrowing of TranLow int32 lanes before the 16-bit
// subtract and square accumulation.
func blockErrorPureGo(coeff []int32, dqcoeff []int32, count int) (err int64, ssz int64) {
	if count <= 0 {
		return 0, 0
	}
	_ = coeff[count-1]
	_ = dqcoeff[count-1]
	for i := 0; i < count; i++ {
		c := int16(coeff[i])
		d := int16(dqcoeff[i])
		diff := int64(int16(c - d))
		err += diff * diff
		cv := int64(c)
		ssz += cv * cv
	}
	return err, ssz
}
