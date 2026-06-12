package encoder

// sad.go hosts the motion-search SAD kernel dispatch. sad8x8Impl computes the
// sum of absolute differences of one 8x8 block; implementations may ignore
// limit (it is an early-exit hint, not a contract), so callers must compare
// the returned total against their own threshold. Every architecture variant
// must be bit-exact with sad8x8PureGo.
var sad8x8Impl = sad8x8PureGo

// sad16x16Impl computes the full 16x16 SAD (no early exit); used by the
// merge-tier searches where 16x16 and 32x32 SADs dominate the decider.
var sad16x16Impl = sad16x16PureGo

// sad32x32Impl computes the full 32x32 SAD; the full-pel search probes this
// shape often enough that avoiding four separate 16x16 dispatches matters.
var sad32x32Impl = sad32x32PureGo

// sad8x8DualImpl computes the 8x8 SAD between planes with different strides
// (the subpel verifier compares the source plane against the n-stride
// prediction scratch).
var sad8x8DualImpl = sad8x8DualPureGo

// sad8x8CompoundAvgBlockImpl computes SAD(src, round((ref0+ref1)/2)) for an
// 8x8 block. It is the compound-reference precheck counterpart to sad8x8Dual.
var sad8x8CompoundAvgBlockImpl = sad8x8CompoundAvgBlockPureGo

// sad8x8PureGo is the portable reference with a row-granular early exit.
func sad8x8PureGo(src, ref []byte, stride int, limit int) int {
	total := 0
	for r := range 8 {
		row := r * stride
		for c := range 8 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
		if total >= limit {
			return total
		}
	}
	return total
}

// sad16x16PureGo is the portable 16x16 reference.
func sad16x16PureGo(src, ref []byte, stride int) int {
	total := 0
	for r := range 16 {
		row := r * stride
		for c := range 16 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

// sad32x32PureGo is the portable 32x32 reference.
func sad32x32PureGo(src, ref []byte, stride int) int {
	total := 0
	for r := range 32 {
		row := r * stride
		for c := range 32 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

// sad8x8DualPureGo is the portable two-stride 8x8 reference.
func sad8x8DualPureGo(src []byte, srcStride int, ref []byte, refStride int) int {
	total := 0
	for r := range 8 {
		srow := r * srcStride
		rrow := r * refStride
		for c := range 8 {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sad8x8CompoundAvgBlockPureGo(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	total := 0
	for r := range 8 {
		srow := r * srcStride
		r0row := r * ref0Stride
		r1row := r * ref1Stride
		for c := range 8 {
			pred := (int(ref0[r0row+c]) + int(ref1[r1row+c]) + 1) >> 1
			d := int(src[srow+c]) - pred
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}
