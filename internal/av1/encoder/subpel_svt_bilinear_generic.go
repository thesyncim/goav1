//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func svtBilinearPredictArch(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	svtBilinearPredict(scratch, ref, refStride, bw, bh, xoff, yoff)
}
