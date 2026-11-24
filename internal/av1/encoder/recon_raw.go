package encoder

import "github.com/thesyncim/goav1/internal/av1/dsp"

var addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockPureGo

// addRawResidualToPredReconBlock writes the prediction plus the rounded raw
// inverse transform in one pass. Its caller supplies positive block dimensions,
// fitting slices and strides at least w, with raw in separate scratch storage.
// pred is either dst itself (same start and stride: each sample is read before
// it is written) or disjoint from it; every encoder caller predicts into
// scratch separate from the reconstruction plane.
func addRawResidualToPredReconBlock(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	addRawResidualToPredReconBlockImpl(dst, stride, pred, predStride, raw, w, h)
}

func addRawResidualToPredReconBlockCopied(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	copyPredToReconBlock(dst, pred, predStride, stride, 0, 0, w, h)
	dsp.AddRawTransformPlaneBlockTrusted(dst, stride, 1, 0xff, w, h, raw, w)
}

func addRawResidualToPredReconBlockPureGo(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	if h <= 0 {
		return
	}
	for y := range h {
		for x := range w {
			// Widen before rounding so every int32 input keeps the original
			// DSP result, including the positive overflow boundary.
			r := (int64(raw[y*w+x]) + 8) >> 4
			r = max(-32768, min(32767, r))
			dst[y*stride+x] = uint8(max(0, min(255, int64(pred[y*predStride+x])+r)))
		}
	}
}
