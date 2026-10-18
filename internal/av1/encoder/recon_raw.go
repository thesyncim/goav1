package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp"
)

var addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockPureGo

// addRawResidualToPredReconBlock writes the prediction plus the rounded raw
// inverse transform in one pass. Its caller supplies positive block dimensions,
// fitting slices and strides at least w, with raw in separate scratch storage.
func addRawResidualToPredReconBlock(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	addRawResidualToPredReconBlockImpl(dst, stride, pred, predStride, raw, w, h)
}

// Exact in-place prediction is safe. Partial overlap must retain the original
// complete prediction-copy pass before any residual is added: a row can still
// read prediction bytes written by an earlier row of that copy.
func rawPredReconNeedsCopy(dst []byte, stride int, pred []byte, predStride, w, h int) bool {
	d := uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
	p := uintptr(unsafe.Pointer(unsafe.SliceData(pred)))
	return (d != p || stride != predStride) &&
		d < p+uintptr((h-1)*predStride+w) && p < d+uintptr((h-1)*stride+w)
}

func addRawResidualToPredReconBlockCopied(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	copyPredToReconBlock(dst, pred, predStride, stride, 0, 0, w, h)
	dsp.AddRawTransformPlaneBlockTrusted(dst, stride, 1, 0xff, w, h, raw, w)
}

func addRawResidualToPredReconBlockPureGo(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	if h <= 0 {
		return
	}
	if rawPredReconNeedsCopy(dst, stride, pred, predStride, w, h) {
		addRawResidualToPredReconBlockCopied(dst, stride, pred, predStride, raw, w, h)
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
