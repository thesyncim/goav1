//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func svtBilinearPredictArch(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
}

func svtSubpelVarianceArch(scratch *svtSubpelScratch, ref []byte, refStride int, src []byte, srcStride, bw, bh, xoff, yoff int) uint32 {
	svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
	_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
	return v
}
