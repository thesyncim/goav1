// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// NEON-accelerated DC sum, CfL subsample/apply and directional Z1 row
// interpolation. The .s file implements the 8-lane inner loops for the common
// 8-bit shapes; the Go wrappers below resolve base pointers and route the
// shapes the asm does not handle (width 4, high bit depth, the clamp boundary
// of a directional row) to the pure-Go references, keeping the asm single-path
// and the byte-exactness contract easy to audit.
//
// Bit-exactness with the *PureGo references in intra_kernels.go:
//   - sumSamplesNEON folds uint16 lanes through uaddlp into u64 accumulators;
//     the running 64-bit total matches the scalar add.
//   - subsampleLuma8NEON reproduces the (a+b+c+d)<<1, (a+b)<<2 and x<<3
//     reductions with uaddlp / uxtl and a left shift.
//   - applyCFLNEON computes alpha*ac in s32 lanes, rounds with
//     sign(ac*alpha) * ((|ac*alpha|+32)>>6) (abs + urshr #6 + conditional
//     negate, identical to roundPowerOfTwoSigned), adds the widened chroma DC
//     and saturates to [0,255] with sqxtun.
//   - dirRowInterp8NEON computes roundPowerOfTwo(p0*(32-shift)+p1*shift,5) in
//     u32 lanes via umull/umlal + urshr #5, matching the reference; it is only
//     invoked for rows fully inside the interpolation region (no clamp).

//go:noescape
func sumSamplesNEONAsm(samples *uint16, n uintptr) uint64

//go:noescape
func subsampleLuma8NEONAsm(out *uint16, top *uint8, bot *uint8, mode uintptr, n uintptr)

//go:noescape
func applyCFLRowNEONAsm(dst *byte, ac *int16, alpha uintptr, n uintptr)

//go:noescape
func dirRowInterp8NEONAsm(dst *byte, above *uint16, weight32 uintptr, weightShift uintptr, n uintptr)

//go:noescape
func dirAboveRun8NEONAsm(dst *byte, ref *uint16, weight32 uintptr, weightShift uintptr, n uintptr)

//go:noescape
func dirLeftCol8NEONAsm(dst *byte, stride uintptr, ref *uint16, weight32 uintptr, weightShift uintptr, n uintptr)

func sumSamplesNEON(samples []uint16) int {
	n := len(samples)
	chunks := n &^ 7
	total := 0
	if chunks > 0 {
		total = int(sumSamplesNEONAsm(&samples[0], uintptr(chunks)))
	}
	for i := chunks; i < n; i++ {
		total += int(samples[i])
	}
	return total
}

func subsampleLuma8NEON(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
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
				subsampleLuma8NEONAsm(
					&outputQ3[outRow*CFLBufLine+oc],
					&input[row*inputStride+ic],
					&input[(row+1)*inputStride+ic],
					2, 0)
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
				subsampleLuma8NEONAsm(
					&outputQ3[row*CFLBufLine+oc],
					&input[row*inputStride+ic],
					nil,
					1, 0)
			}
		}
	default:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			for oc := 0; oc < outW; oc += 8 {
				subsampleLuma8NEONAsm(
					&outputQ3[row*CFLBufLine+oc],
					&input[row*inputStride+oc],
					nil,
					0, 0)
			}
		}
	}
}

func applyCFLNEON(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	if bytesPerSample != 1 || visibleWidth%8 != 0 || max != 0xff {
		applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
		return
	}
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride:]
		ac := acQ3[row*CFLBufLine:]
		for col := 0; col < visibleWidth; col += 8 {
			applyCFLRowNEONAsm(&line[col], &ac[col], uintptr(int16(alphaQ3)), 0)
		}
	}
}

func dirRowInterp8NEON(dst []byte, above []uint16, base int, shift int, maxBase int, width int) {
	// The asm handles only fully-interpolated rows (every column inside
	// [0,maxBase)) with width a multiple of 8; anything touching the clamp or a
	// width-4 block falls back to the scalar reference.
	if width%8 != 0 || base < 0 || base+width > maxBase {
		dirRowInterp8PureGo(dst, above, base, shift, maxBase, width)
		return
	}
	dirRowInterp8NEONAsm(&dst[0], &above[base], uintptr(32-shift), uintptr(shift), uintptr(width))
}

func dirAboveRun8NEON(dst []byte, ref []uint16, shift int, count int) {
	// Process 8 contiguous outputs per asm call; the (0..7) tail uses the
	// scalar reference. ref[0..count] are all valid (pre-validated span), so the
	// asm's ref[i]/ref[i+1] loads never read past the slice.
	chunks := count &^ 7
	if chunks > 0 {
		dirAboveRun8NEONAsm(&dst[0], &ref[0], uintptr(32-shift), uintptr(shift), uintptr(chunks))
	}
	if chunks < count {
		dirAboveRun8PureGo(dst[chunks:], ref[chunks:], shift, count-chunks)
	}
}

func dirLeftCol8NEON(dst []byte, stride int, ref []uint16, shift int, count int) {
	// Process 8 rows per asm call (strided byte stores), tail rows via scalar.
	chunks := count &^ 7
	if chunks > 0 {
		dirLeftCol8NEONAsm(&dst[0], uintptr(stride), &ref[0], uintptr(32-shift), uintptr(shift), uintptr(chunks))
	}
	if chunks < count {
		dirLeftCol8PureGo(dst[chunks*stride:], stride, ref[chunks:], shift, count-chunks)
	}
}
