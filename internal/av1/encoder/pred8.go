package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// pred8.go is the encoder's fast path for 8-bit luma inter prediction at
// blocks 8 or more samples wide and tall: the libaom convolve_x_sr /
// convolve_y_sr / convolve_2d_sr arithmetic (the decoder's predictor, see
// motion.convolve*8PureGo) on blocks whose whole 8-tap footprint lies inside
// the reference plane, so no edge clamping is needed. The kernel can store
// the prediction or, for motion search, return its SAD against the source
// without materializing it. Blocks near the plane edge, narrower shapes and
// builds without the vector kernel keep the motion package predictor.

// pred8IMLen is the 2-D intermediate scratch: up to 64+7 rows of 64 columns.
const pred8IMLen = (64 + 7) * 64

// pred8Impl, when non-nil, predicts (dst != nil) or SADs against src
// (src != nil) one w x h block whose reference origin ref is the full-pel
// sample (refX, refY); subX/subY are the Q4 fractions and xk/yk the 8-tap
// kernels. Store mode needs w a multiple of 8, SAD mode a multiple of 16.
var pred8Impl func(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int,
	ref unsafe.Pointer, refStride, w, h, subX, subY int, xk, yk *[8]int16, im *[pred8IMLen]int16) int

// pred8IMImpl and pred8VIMImpl are the two 2-D passes on their own (rows of
// the first-pass intermediate at any int16 row stride in bytes, and the
// second pass over an intermediate whose first row is the block's row -3),
// for callers that share one intermediate across several vectors.
var (
	pred8IMImpl  func(im unsafe.Pointer, imStride int, ref unsafe.Pointer, refStride, w, rows int, k *[8]int16)
	pred8VIMImpl func(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, im unsafe.Pointer, imStride, w, h int, k *[8]int16) int
)

// pred8Kernels are the 8-tap tables by filter (regular, smooth, sharp).
var pred8Kernels [3]motion.SubpelKernelTable

// pred8KernelsOK reports whether every tap is even, which the vector kernels'
// halved-tap 16-bit accumulation relies on.
var pred8KernelsOK = func() bool {
	for f := motion.InterpEightTapRegular; f <= motion.InterpMultiTapSharp; f++ {
		t, err := motion.SubpelKernelTableFor(f, 8)
		if err != nil {
			return false
		}
		for _, k := range t {
			for _, v := range k {
				if v&1 != 0 {
					return false
				}
			}
		}
		pred8Kernels[f] = *t
	}
	return true
}()

// pred8Fits reports whether the 8-tap footprint of a w x h block at the
// full-pel origin (refX, refY), plus the kernels' 16-byte row over-read, lies
// inside the plane.
func pred8Fits(ref *frame.Plane, refX, refY, w, h int) bool {
	return refX >= 3 && refY >= 3 && refX+w+4 <= ref.Width && refY+h+4 <= ref.Height &&
		(refY+h+4)*ref.Stride+refX+w+16 <= len(ref.Pix)
}

// pred8Predict writes the w x h prediction at dst (stride dstStride) for the
// full-pel origin (refX, refY) and Q4 fractions, reporting false (nothing
// written) when the fast path does not apply.
func pred8Predict(dst []byte, dstStride int, ref *frame.Plane, refX, refY, w, h, subX, subY int, filters motion.InterpFilters, im *[pred8IMLen]int16) bool {
	if pred8Impl == nil || !pred8KernelsOK || w&7 != 0 || h < 8 || w > 64 || h > 64 ||
		filters.X > motion.InterpMultiTapSharp || filters.Y > motion.InterpMultiTapSharp ||
		!pred8Fits(ref, refX, refY, w, h) || len(dst) < (h-1)*dstStride+w {
		return false
	}
	pred8Impl(unsafe.Pointer(&dst[0]), dstStride, nil, 0,
		unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX]), ref.Stride, w, h, subX, subY,
		&pred8Kernels[filters.X][subX], &pred8Kernels[filters.Y][subY], im)
	return true
}

// predictLuma is predictIntoFilters for an unscaled luma block (dst at stride
// bw): the fast predictor for interior blocks, the motion package predictor
// otherwise.
func (st *lossyEncodeState) predictLuma(dst []byte, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector, filters motion.InterpFilters) error {
	return st.predictPlane(dst, refPlane, stride, width, height, px, py, bw, bh, mv, false, false, filters)
}

// predictPlane is predictLuma for any unscaled plane: chroma blocks 8 or more
// samples wide and tall use the same 8-tap kernels at their subsampled
// positions.
func (st *lossyEncodeState) predictPlane(dst []byte, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool, filters motion.InterpFilters) error {
	if pred8Impl != nil && bw >= 8 && bh >= 8 {
		refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, ssX, ssY)
		if err == nil {
			ref := frame.Plane{Pix: refPlane, Stride: stride, Width: width, Height: height}
			if st.subpelCache == nil {
				st.subpelCache = new(subpelProbeCache)
			}
			if pred8Predict(dst, bw, &ref, refX, refY, bw, bh, subX, subY, filters, &st.subpelCache.scratch) {
				return nil
			}
		}
	}
	return predictIntoFilters(dst, refPlane, stride, width, height, px, py, bw, bh, mv, ssX, ssY, filters, st.scaledScratch.Conv())
}

// pred8SAD returns the SAD between the regular-filter prediction of a w x h
// block and src (stride srcStride), reporting false when the fast path does
// not apply.
func pred8SAD(src []byte, srcStride int, ref *frame.Plane, refX, refY, w, h, subX, subY int, im *[pred8IMLen]int16) (int, bool) {
	if pred8Impl == nil || !pred8KernelsOK || w&15 != 0 || h < 8 || w > 64 || h > 64 ||
		!pred8Fits(ref, refX, refY, w, h) || len(src) < (h-1)*srcStride+w+16 {
		return 0, false
	}
	k := &pred8Kernels[motion.InterpEightTapRegular]
	return pred8Impl(nil, 0, unsafe.Pointer(&src[0]), srcStride,
		unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX]), ref.Stride, w, h, subX, subY,
		&k[subX], &k[subY], im), true
}
