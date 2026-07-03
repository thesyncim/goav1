// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

// AVX2-accelerated CfL luma subsampling and the directional Z1/Z2/Z3 edge
// interpolation helpers. The .s file implements the inner loops for the common
// 8-bit shapes; the Go wrappers below resolve base pointers and route the
// shapes the asm does not special-case (width 4, high bit depth, the clamp
// boundary of a directional row, the sub-8 tail of a run) to the pure-Go
// references, keeping the asm single-path and the byte-exactness contract easy
// to audit. These mirror the arm64 NEON kernels in intra_kernels_neon_arm64.s
// and, like them, are byte-exact with the *PureGo references in
// intra_kernels.go.
//
// Bit-exactness with the *PureGo references:
//   - subsampleLuma8AVX2 reproduces the (a+b+c+d)<<1, (a+b)<<2 and x<<3
//     reductions with VPMADDUBSW (pairwise byte adds against a vector of ones)
//     and a left shift, matching the libaom lbd cfl_luma_subsampling_*_c paths.
//   - dirInterp8AVX2Asm computes roundPowerOfTwo(p0*(32-shift)+p1*shift,5) in
//     u32 lanes via VPMOVZXWD/VPMULLD + (x+16)>>5, matching the reference; it is
//     only invoked for runs fully inside the interpolation region (no clamp).
//
// dav1d reference: third_party/upstream/dav1d/src/x86/ipred_avx2.asm
// (ipred_cfl_ac_420/422/444 for the luma subsampling reductions; ipred_z1/z2/z3
// for the fractional edge interpolation).

//go:noescape
func subsampleLuma8AVX2Asm(out *uint16, top *uint8, bot *uint8, mode uint64)

//go:noescape
func dirInterp8AVX2Asm(dst *byte, src *uint16, weight32 uint64, weightShift uint64, n uint64)

//go:noescape
func dirLeftCol8AVX2Asm(dst *byte, stride uint64, ref *uint16, weight32 uint64, weightShift uint64, n uint64)

func subsampleLuma8AVX2(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		// out width = width/2; process 8 output samples (16 input cols) per call.
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < height; row += 2 {
			outRow := row >> 1
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				subsampleLuma8AVX2Asm(
					&outputQ3[outRow*CFLBufLine+oc],
					&input[row*inputStride+ic],
					&input[(row+1)*inputStride+ic],
					2)
			}
		}
	case subX:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				subsampleLuma8AVX2Asm(
					&outputQ3[row*CFLBufLine+oc],
					&input[row*inputStride+ic],
					nil,
					1)
			}
		}
	default:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			for oc := 0; oc < outW; oc += 8 {
				subsampleLuma8AVX2Asm(
					&outputQ3[row*CFLBufLine+oc],
					&input[row*inputStride+oc],
					nil,
					0)
			}
		}
	}
}

func dirRowInterp8AVX2(dst []byte, above []uint16, base int, shift int, maxBase int, width int) {
	// The asm handles only fully-interpolated rows (every column inside
	// [0,maxBase)) with width a multiple of 8; anything touching the clamp or a
	// width-4 block falls back to the scalar reference.
	if width%8 != 0 || base < 0 || base+width > maxBase {
		dirRowInterp8PureGo(dst, above, base, shift, maxBase, width)
		return
	}
	dirInterp8AVX2Asm(&dst[0], &above[base], uint64(32-shift), uint64(shift), uint64(width))
}

func dirAboveRun8AVX2(dst []byte, ref []uint16, shift int, count int) {
	// Process 8 contiguous outputs per asm call; the (0..7) tail uses the scalar
	// reference. ref[0..count] are all valid (pre-validated span), so the asm's
	// ref[i]/ref[i+1] loads never read past the slice.
	chunks := count &^ 7
	if chunks > 0 {
		dirInterp8AVX2Asm(&dst[0], &ref[0], uint64(32-shift), uint64(shift), uint64(chunks))
	}
	if chunks < count {
		dirAboveRun8PureGo(dst[chunks:], ref[chunks:], shift, count-chunks)
	}
}

func dirLeftCol8AVX2(dst []byte, stride int, ref []uint16, shift int, count int) {
	// Process 8 rows per asm call (strided byte stores), tail rows via scalar.
	chunks := count &^ 7
	if chunks > 0 {
		dirLeftCol8AVX2Asm(&dst[0], uint64(stride), &ref[0], uint64(32-shift), uint64(shift), uint64(chunks))
	}
	if chunks < count {
		dirLeftCol8PureGo(dst[chunks*stride:], stride, ref[chunks:], shift, count-chunks)
	}
}
