package motion

import "sync"

const scaledIMMaxSamples = scaledIMMaxHeight * maxBlockSize

type scaledHighBDIM [scaledIMMaxSamples]int32
type scaledIM [scaledIMMaxSamples]int16

// ScaledConvolveScratch carries reusable temporary storage for scaled
// interpolation. It keeps the large intermediate blocks out of hot call frames
// on framework decode paths.
type ScaledConvolveScratch struct {
	im       scaledIM
	highBD   scaledHighBDIM
	compound CompoundConvolveScratch
	conv     ConvolveScratch
}

// Conv exposes the embedded translational 2D convolve scratch so callers on
// the unscaled fast path reuse it instead of zero-filling a fresh stack
// intermediate block on every call.
func (s *ScaledConvolveScratch) Conv() *ConvolveScratch {
	if s == nil {
		return nil
	}
	return &s.conv
}

var scaledHighBDIMPool = sync.Pool{
	New: func() any {
		return new(scaledHighBDIM)
	},
}

func scaledHighBDIMForScratch(scratch *ScaledConvolveScratch) (*scaledHighBDIM, bool) {
	if scratch != nil {
		return &scratch.highBD, false
	}
	return scaledHighBDIMPool.Get().(*scaledHighBDIM), true
}

func putScaledHighBDIM(im *scaledHighBDIM, pooled bool) {
	if pooled {
		scaledHighBDIMPool.Put(im)
	}
}
