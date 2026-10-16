package encoder

// The reconstruction copies of the inter TX path: the prediction becomes the
// recon block (all-zero transform blocks) or the prediction plus one constant
// residual (DC-only DCT_DCT blocks). Scalar references unless the Go SIMD
// kernels bind them in recon_gosimd.go.
var (
	copyPredToReconBlockImpl                = copyPredToReconBlockPureGo
	addConstantResidualToPredReconBlockImpl = addConstantResidualToPredReconBlockPureGo
)

func copyPredToReconBlock(reconPlane, pred []byte, predStride, stride, px, py, w, h int) {
	copyPredToReconBlockImpl(reconPlane, pred, predStride, stride, px, py, w, h)
}

func addConstantResidualToPredReconBlock(reconPlane, pred []byte, predStride, stride, px, py, w, h int, residual int16) {
	addConstantResidualToPredReconBlockImpl(reconPlane, pred, predStride, stride, px, py, w, h, residual)
}
