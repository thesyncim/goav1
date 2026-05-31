package motion

import "sync"

const scaledIMMaxSamples = scaledIMMaxHeight * maxBlockSize

type scaledHighBDIM [scaledIMMaxSamples]int32

// ScaledConvolveScratch carries reusable temporary storage for scaled
// high-bit-depth interpolation. It keeps the large intermediate block out of
// the heap on framework decode hot paths.
type ScaledConvolveScratch struct {
	highBD scaledHighBDIM
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
