// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"encoding/binary"
	"math/bits"
)

// PredictIntra8 is the decoder's 8-bit DC, V, H, Paeth and smooth entry, the
// ipred call dav1d's recon_b_intra makes after dav1d_prepare_intra_edges. pix
// holds the w x h block from its top-left sample with rows stride bytes apart.
// The edges are pixels, as dav1d's edge buffer: top is the row above the block,
// read in place, and left holds the h samples of the column to its left; DC
// takes a nil edge as unavailable, the other modes need the edges they read.
// aboveLeft is read by Paeth only. The smooth and Paeth kernels read the edges
// widened into wide, at least w+h samples of caller scratch.
func PredictIntra8(pix []byte, stride int, w int, h int, mode IntraMode, top []byte, left []byte, aboveLeft byte, wide []uint16) error {
	if w < 4 || w > 64 || h < 4 || h > 64 || w&(w-1) != 0 || h&(h-1) != 0 ||
		stride < w || len(pix) < (h-1)*stride+w {
		return ErrInvalidPrediction
	}
	block := planeBlock{pix: pix, stride: stride, width: w, height: h, rowBytes: w}
	switch mode {
	case IntraModeDC:
		if top != nil && len(top) < w || left != nil && len(left) < h {
			return ErrInvalidPrediction
		}
		fillBlock8(block, dc8(w, h, top, left))
		return nil
	case IntraModeVertical:
		if len(top) < w {
			return ErrInvalidPrediction
		}
		predictVertical8Bytes(block, top[:w])
		return nil
	case IntraModeHorizontal:
		if len(left) < h {
			return ErrInvalidPrediction
		}
		predictHorizontal8Bytes(block, left[:h])
		return nil
	}
	if len(top) < w || len(left) < h || len(wide) < w+h {
		return ErrInvalidPrediction
	}
	above, leftWide := wide[:w], wide[w:w+h]
	widenEdge8(above, top)
	widenEdge8(leftWide, left)
	weightsW := smoothWeights[w-4 : w-4+w]
	weightsH := smoothWeights[h-4 : h-4+h]
	switch mode {
	case IntraModePaeth:
		predictPaethImpl(block, 1, above, leftWide, uint16(aboveLeft))
	case IntraModeSmooth:
		predictSmoothImpl(block, 1, weightsW, weightsH, above, leftWide, leftWide[h-1], above[w-1])
	case IntraModeSmoothVertical:
		predictSmoothVerticalImpl(block, 1, weightsH, above, leftWide[h-1])
	case IntraModeSmoothHorizontal:
		predictSmoothHorizontalImpl(block, 1, weightsW, leftWide, above[w-1])
	default:
		return ErrInvalidPrediction
	}
	return nil
}

// dc8 is dav1d's DC_PRED, TOP_DC_PRED, LEFT_DC_PRED and DC_128_PRED value
// for 8-bit samples, selected by which edges are present.
func dc8(w int, h int, top []byte, left []byte) byte {
	switch {
	case top != nil && left != nil:
		sum := sumBytes(top[:w]) + sumBytes(left[:h])
		if w == h {
			return byte((sum + w) >> bits.TrailingZeros(uint(w+h)))
		}
		if v, ok := dcRectPredictionValue(w, h, 0xff, sum); ok {
			return byte(v)
		}
		return byte((sum + (w+h)>>1) / (w + h))
	case top != nil:
		return byte((sumBytes(top[:w]) + w>>1) >> bits.TrailingZeros(uint(w)))
	case left != nil:
		return byte((sumBytes(left[:h]) + h>>1) >> bits.TrailingZeros(uint(h)))
	default:
		return 128
	}
}

// sumBytes is the sum of a power-of-two run of at least four bytes, eight
// bytes per word: pairwise byte sums into 16-bit lanes, then one multiply
// folds the four lanes into the top lane.
func sumBytes(b []byte) int {
	const lanes = 0x00ff00ff00ff00ff
	if len(b) == 4 {
		x := uint64(binary.LittleEndian.Uint32(b))
		x = x&lanes + x>>8&lanes
		return int((x * 0x0001000100010001) >> 48)
	}
	sum := 0
	for i := 0; i+8 <= len(b); i += 8 {
		x := binary.LittleEndian.Uint64(b[i:])
		x = x&lanes + x>>8&lanes
		sum += int((x * 0x0001000100010001) >> 48)
	}
	return sum
}
