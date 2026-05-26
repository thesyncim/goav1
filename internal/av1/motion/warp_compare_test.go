package motion

import (
	"math/rand"
	"testing"
)

// libaomWarpAffineC is a hand-translated, intentionally faithful port of
// av1/common/warped_motion.c:av1_warp_affine_c (single-reference, 8-bit, not
// compound). It is used purely as a parity oracle to discover any drift in our
// production warpAffine8 implementation.
func libaomWarpAffineC(pred []byte, predStride int, ref []byte, refStride, refWidth, refHeight int,
	pCol, pRow, pWidth, pHeight int, mat [6]int32, alpha, beta, gamma, delta int16,
	subsamplingX, subsamplingY int,
) {
	const (
		WARPEDMODEL_PREC_BITS = 16
		WARP_PARAM_REDUCE = 6
		WARPEDDIFF_PREC_BITS = WARPEDMODEL_PREC_BITS - 6 // = 10
		WARPEDPIXEL_PREC_SHIFTS = 64
		FILTER_BITS = 7
		bd = 8
		reduceBitsHoriz = 3
		offsetBitsHoriz = bd + FILTER_BITS - 1 // 14
		reduceBitsVert = 2*FILTER_BITS - reduceBitsHoriz // 11
		offsetBitsVert = bd + 2*FILTER_BITS - reduceBitsHoriz // 19
	)

	var tmp [15 * 8]int32

	round := func(v int, n int) int {
		bias := 1 << (n - 1)
		// C arithmetic shift semantics: (v+bias) >> n on signed int.
		return (v + bias) >> n
	}
	clamp := func(v, lo, hi int) int {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
	clipPix := func(v int) byte {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return byte(v)
	}

	for i := pRow; i < pRow+pHeight; i += 8 {
		for j := pCol; j < pCol+pWidth; j += 8 {
			srcX := int64(j+4) << uint(subsamplingX)
			srcY := int64(i+4) << uint(subsamplingY)
			dstX := int64(mat[2])*srcX + int64(mat[3])*srcY + int64(mat[0])
			dstY := int64(mat[4])*srcX + int64(mat[5])*srcY + int64(mat[1])
			x4 := dstX >> uint(subsamplingX)
			y4 := dstY >> uint(subsamplingY)

			ix4 := int32(x4 >> WARPEDMODEL_PREC_BITS)
			sx4 := int32(x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))
			iy4 := int32(y4 >> WARPEDMODEL_PREC_BITS)
			sy4 := int32(y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))

			sx4 += int32(alpha)*-4 + int32(beta)*-4
			sy4 += int32(gamma)*-4 + int32(delta)*-4

			sx4 &^= (1 << WARP_PARAM_REDUCE) - 1
			sy4 &^= (1 << WARP_PARAM_REDUCE) - 1

			// Horizontal filter
			for k := -7; k < 8; k++ {
				iy := clamp(int(iy4)+k, 0, refHeight-1)
				sx := int(sx4) + int(beta)*(k+4)
				for l := -4; l < 4; l++ {
					ix := int(ix4) + l - 3
					offs := round(sx, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsHoriz
					for m := 0; m < 8; m++ {
						sampleX := clamp(ix+m, 0, refWidth-1)
						sum += int(ref[iy*refStride+sampleX]) * int(coeffs[m])
					}
					tmp[(k+7)*8+(l+4)] = int32(round(sum, reduceBitsHoriz))
					sx += int(alpha)
				}
			}

			// Vertical filter
			rowEnd := pRow + pHeight - i - 4
			if rowEnd > 4 {
				rowEnd = 4
			}
			colEnd := pCol + pWidth - j - 4
			if colEnd > 4 {
				colEnd = 4
			}
			for k := -4; k < rowEnd; k++ {
				sy := int(sy4) + int(delta)*(k+4)
				for l := -4; l < colEnd; l++ {
					offs := round(sy, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := 0; m < 8; m++ {
						sum += int(tmp[(k+m+4)*8+(l+4)]) * int(coeffs[m])
					}
					sum = round(sum, reduceBitsVert)
					row := i - pRow + k + 4
					col := j - pCol + l + 4
					pred[row*predStride+col] = clipPix(sum - (1 << (bd - 1)) - (1 << bd))
					sy += int(gamma)
				}
			}
		}
	}
}

// TestWarpAffineMatchesLibaomReference compares the production warpAffine8
// against the hand-rolled libaom-faithful oracle on a battery of randomized
// inputs that mirror the bitstream contract for local-warp shear parameters.
func TestWarpAffineMatchesLibaomReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA0BEEF))

	const refW = 96
	const refH = 96
	ref := make([]byte, refW*refH)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}

	// Block sizes that occur in practice (powers of 2 × pairs).
	cases := []struct {
		w, h int
	}{
		{8, 8}, {16, 16}, {8, 16}, {16, 8}, {32, 8}, {8, 32}, {32, 16}, {16, 32},
	}

	for caseIdx := 0; caseIdx < 64; caseIdx++ {
		for _, sz := range cases {
			pCol := rng.Intn(refW-sz.w+1) &^ 7
			pRow := rng.Intn(refH-sz.h+1) &^ 7
			// Build a representative warp matrix: small perturbation around identity.
			matrix := [6]int32{
				int32(rng.Intn(1 << 16)) - (1 << 15), // mat[0]: trans X (Q0 << 16)
				int32(rng.Intn(1 << 16)) - (1 << 15), // mat[1]: trans Y
				1<<16 + int32(rng.Intn(8192)) - 4096, // mat[2] ≈ 1 + small
				int32(rng.Intn(8192)) - 4096,         // mat[3]
				int32(rng.Intn(8192)) - 4096,         // mat[4]
				1<<16 + int32(rng.Intn(8192)) - 4096, // mat[5]
			}
			// Pick alpha/beta/gamma/delta consistent with WARP_PARAM_REDUCE.
			roundReduce := func(v int) int16 {
				if v < 0 {
					return int16(-((-v + 32) >> 6) * 64)
				}
				return int16(((v + 32) >> 6) * 64)
			}
			alpha := roundReduce(rng.Intn(1024) - 512)
			beta := roundReduce(rng.Intn(1024) - 512)
			gamma := roundReduce(rng.Intn(1024) - 512)
			delta := roundReduce(rng.Intn(1024) - 512)

			// Filter the "interesting" cases: must satisfy is_affine_shear_allowed.
			abs := func(x int16) int {
				if x < 0 {
					return -int(x)
				}
				return int(x)
			}
			if 4*abs(alpha)+7*abs(beta) >= 1<<16 || 4*abs(gamma)+4*abs(delta) >= 1<<16 {
				continue
			}

			expected := make([]byte, sz.w*sz.h)
			actual := make([]byte, sz.w*sz.h)

			libaomWarpAffineC(expected, sz.w, ref, refW, refW, refH,
				pCol, pRow, sz.w, sz.h, matrix, alpha, beta, gamma, delta, 0, 0)

			// Wrap our destination into a frame.Plane covering the block region exactly.
			dst, _ := testPlane(sz.w, sz.h, 1, sz.w)
			refPlane, _ := testPlane(refW, refH, 1, refW)
			copy(refPlane.Pix, ref)

			if err := PredictWarpedPlaneBlockBitDepth(dst, refPlane, 1, 8, 0, 0, sz.w, sz.h, matrix, alpha, beta, gamma, delta, false, false); err != nil {
				t.Fatalf("case %d sz=%dx%d: %v", caseIdx, sz.w, sz.h, err)
			}

			// Because the production function uses absolute coordinates, recompute
			// expected with pCol=0, pRow=0.
			expected = make([]byte, sz.w*sz.h)
			libaomWarpAffineC(expected, sz.w, ref, refW, refW, refH,
				0, 0, sz.w, sz.h, matrix, alpha, beta, gamma, delta, 0, 0)
			for i := range expected {
				actual[i] = dst.Pix[i]
			}
			for y := 0; y < sz.h; y++ {
				for x := 0; x < sz.w; x++ {
					if expected[y*sz.w+x] != actual[y*sz.w+x] {
						t.Fatalf("case %d sz=%dx%d mismatch at (%d,%d): expected=%d actual=%d matrix=%v alpha=%d beta=%d gamma=%d delta=%d",
							caseIdx, sz.w, sz.h, x, y, expected[y*sz.w+x], actual[y*sz.w+x], matrix, alpha, beta, gamma, delta)
					}
				}
			}
		}
	}
}
