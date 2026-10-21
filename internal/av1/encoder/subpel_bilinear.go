package encoder

// subpel_bilinear.go ports the sub-pixel error kernel libaom's realtime
// motion search scores its subpel tree with: use_accurate_subpel_search =
// USE_2_TAPS (av1/encoder/speed_features.c
// set_rt_speed_features_framesize_independent), under which
// av1_find_best_sub_pixel_tree takes the check_better_fast path and calls
// vfp->svf, the two-pass bilinear sub-pixel kernel
// (aom_dsp/variance.c var_filter_block2d_bil_first_pass_c /
// var_filter_block2d_bil_second_pass_c with bilinear_filters_2t). Each pass
// rounds to FILTER_BITS, so the vertical pass reads the horizontal pass's
// rounded rows; the kernels below fuse the two passes row by row.

// bilinearFilters2T is aom_dsp/variance.c bilinear_filters_2t, indexed by the
// eighth-pel offset.
var bilinearFilters2T = [8][2]uint16{
	{128, 0}, {112, 16}, {96, 32}, {80, 48},
	{64, 64}, {48, 80}, {32, 96}, {16, 112},
}

// bilinearSubpelSAD scores the n x n block of src against ref at the
// full-pel origin (refX, refY) plus the eighth-pel offset (fx, fy) with the
// two-pass bilinear prediction. ok=false when the (n+1) x (n+1) source
// window leaves the plane.
func bilinearSubpelSAD(src []byte, srcStride int, ref []byte, refStride, refW, refH, refX, refY, n, fx, fy int) (int, bool) {
	if n <= 0 || n > 64 || refX < 0 || refY < 0 || refX+n >= refW || refY+n >= refH ||
		(refY+n)*refStride+refX+n+1 > len(ref) || (n-1)*srcStride+n > len(src) {
		return 0, false
	}
	block := ref[refY*refStride+refX:]
	if (refY+n)*refStride+refX+n+16 > len(ref) || (n-1)*srcStride+n+16 > len(src) {
		// The vector kernels over-read 16 bytes per chunk.
		return bilinearSubpelSADScalar(src, srcStride, block, refStride, n, fx, fy), true
	}
	return bilinearSubpelSADImpl(src, srcStride, block, refStride, n, fx, fy), true
}

// bilinearSubpelSADScalar is the reference form of the fused two-pass
// kernel.
func bilinearSubpelSADScalar(src []byte, srcStride int, ref []byte, refStride, n, fx, fy int) int {
	hf := bilinearFilters2T[fx]
	vf := bilinearFilters2T[fy]
	var rows [2][64]uint16
	hpass := func(r int, out *[64]uint16) {
		row := ref[r*refStride : r*refStride+n+1]
		for c := 0; c < n; c++ {
			out[c] = (uint16(row[c])*hf[0] + uint16(row[c+1])*hf[1] + 64) >> 7
		}
	}
	hpass(0, &rows[0])
	sad := 0
	for r := 0; r < n; r++ {
		prev, cur := &rows[r&1], &rows[(r+1)&1]
		hpass(r+1, cur)
		s := src[r*srcStride : r*srcStride+n]
		for c := 0; c < n; c++ {
			p := (prev[c]*vf[0] + cur[c]*vf[1] + 64) >> 7
			d := int(s[c]) - int(p)
			if d < 0 {
				d = -d
			}
			sad += d
		}
	}
	return sad
}
