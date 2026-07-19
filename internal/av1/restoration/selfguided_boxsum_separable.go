// Ported from libaom:
//   av1/common/restoration.c (boxsum1 / boxsum2)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package restoration

// boxsumSeparable computes the same clamped box sums as boxsum, but with
// libaom's separable running-sum decomposition (boxsum1 for r==1, boxsum2 for
// r==2 in av1/common/restoration.c). Instead of re-summing the whole
// (2r+1)x(2r+1) window per output pixel, it runs a vertical pass over the
// (2r+1)-row window followed by an in-place horizontal pass over the
// (2r+1)-column window, so each output is O(1) amortized.
//
// The vertical pass keeps a per-column running sum: as the window slides down
// one row it adds the incoming bottom row and subtracts the outgoing top row
// (dst[i] = dst[i-1] + in - out), matching the set of rows boxsum sums for
// output row i (window [max(0,i-r), min(height-1,i+r)]). The horizontal pass is
// libaom's shift-register reduction, which reads two/four columns ahead and
// writes behind, so it is safe in place.
//
// int32 addition wraps modulo 2^32 and is associative and commutative, so
// accumulating the identical set of (squared) samples as a running sum yields a
// bit-identical int32 total to boxsum for every output position; the values in
// dst are therefore sample-for-sample equal to the brute-force reference.
func boxsumSeparable(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	// Only r in {1,2} occur in AV1 SGR. Degenerate blocks narrower/shorter than
	// the window (never produced by the decoder, where widthExt,heightExt >= 7)
	// keep the brute-force reference so the semantics stay exact.
	if (radius != 1 && radius != 2) || width < 2*radius+1 || height < 2*radius+1 {
		boxsum(src, srcOrigin, width, height, srcStride, radius, squared, dst, dstStride)
		return
	}
	r := radius

	// ---- Vertical pass: dst[i][j] = sum_{y in [max(0,i-r), min(h-1,i+r)]} f(src[y][j]) ----
	// Row 0's window is [0, r] (h > 2r+1 guarantees r <= h-1).
	row0 := dst[:width:width]
	for y := 0; y <= r; y++ {
		base := srcOrigin + y*srcStride
		s := src[base : base+width : base+width]
		if squared {
			if y == 0 {
				for j, v := range s {
					row0[j] = v * v
				}
			} else {
				for j, v := range s {
					row0[j] += v * v
				}
			}
		} else {
			if y == 0 {
				for j, v := range s {
					row0[j] = v
				}
			} else {
				for j, v := range s {
					row0[j] += v
				}
			}
		}
	}
	for i := 1; i < height; i++ {
		addRow := i + r     // incoming bottom row, present when addRow <= height-1
		remRow := i - r - 1 // outgoing top row, present when remRow >= 0
		prev := dst[(i-1)*dstStride : (i-1)*dstStride+width : (i-1)*dstStride+width]
		cur := dst[i*dstStride : i*dstStride+width : i*dstStride+width]
		switch {
		case addRow < height && remRow >= 0:
			ab := srcOrigin + addRow*srcStride
			rb := srcOrigin + remRow*srcStride
			as := src[ab : ab+width : ab+width]
			rs := src[rb : rb+width : rb+width]
			if squared {
				for j := range cur {
					cur[j] = prev[j] + as[j]*as[j] - rs[j]*rs[j]
				}
			} else {
				for j := range cur {
					cur[j] = prev[j] + as[j] - rs[j]
				}
			}
		case addRow < height:
			ab := srcOrigin + addRow*srcStride
			as := src[ab : ab+width : ab+width]
			if squared {
				for j := range cur {
					cur[j] = prev[j] + as[j]*as[j]
				}
			} else {
				for j := range cur {
					cur[j] = prev[j] + as[j]
				}
			}
		case remRow >= 0:
			rb := srcOrigin + remRow*srcStride
			rs := src[rb : rb+width : rb+width]
			if squared {
				for j := range cur {
					cur[j] = prev[j] - rs[j]*rs[j]
				}
			} else {
				for j := range cur {
					cur[j] = prev[j] - rs[j]
				}
			}
		default:
			copy(cur, prev)
		}
	}

	// ---- Horizontal pass over the vertical sums, in place (libaom shift register) ----
	if r == 1 {
		for i := 0; i < height; i++ {
			row := dst[i*dstStride : i*dstStride+width : i*dstStride+width]
			a := row[0]
			b := row[1]
			c := row[2]
			row[0] = a + b
			j := 1
			for ; j < width-2; j++ {
				row[j] = a + b + c
				a = b
				b = c
				c = row[j+2]
			}
			row[j] = a + b + c
			row[j+1] = b + c
		}
		return
	}
	// r == 2
	for i := 0; i < height; i++ {
		row := dst[i*dstStride : i*dstStride+width : i*dstStride+width]
		a := row[0]
		b := row[1]
		c := row[2]
		d := row[3]
		e := row[4]
		row[0] = a + b + c
		row[1] = a + b + c + d
		j := 2
		for ; j < width-3; j++ {
			row[j] = a + b + c + d + e
			a = b
			b = c
			c = d
			d = e
			e = row[j+3]
		}
		row[j] = a + b + c + d + e
		row[j+1] = b + c + d + e
		row[j+2] = c + d + e
	}
}
