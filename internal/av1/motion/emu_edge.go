// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// emuEdgeStride is one row of the emulated-edge scratch window: the widest
// inter block (128) plus the full 8-tap halo (the horizontal kernels read
// width+filterTaps-1 samples per row, and the width-4 and slide-vector
// variants read one byte more).
const emuEdgeStride = maxBlockSize + filterTaps

// emuEdgeRows bounds the tallest emulated-edge window: the tallest inter
// block (128) plus the vertical 8-tap halo.
const emuEdgeRows = maxBlockSize + filterTaps - 1

// emuEdge8 ports dav1d's emu_edge_c (src/mc_tmpl.c) for 8-bit planes: it
// materializes the bw x bh source window whose top-left corner is (x, y) in
// ref coordinates — possibly overhanging the plane on any side — into dst
// with edge-clamp replication. dav1d's reconstruction (src/recon_tmpl.c mc())
// runs emu_edge for out-of-bounds motion vectors and then applies the plain
// put/prep 8tap kernel to the materialized window; the samples match what a
// per-tap clamped convolution would load, so the results are bit-identical.
func emuEdge8(bw int, bh int, ref frame.Plane, x int, y int, dst []byte, dstStride int) {
	iw := ref.Width
	ih := ref.Height

	// find offset in reference of visible block to copy
	refOff := clampInt(y, 0, ih-1)*ref.Stride + clampInt(x, 0, iw-1)

	// number of pixels to extend (left, right, top, bottom)
	leftExt := clampInt(-x, 0, bw-1)
	rightExt := clampInt(x+bw-iw, 0, bw-1)
	topExt := clampInt(-y, 0, bh-1)
	bottomExt := clampInt(y+bh-ih, 0, bh-1)

	// copy visible portion first
	centerW := bw - leftExt - rightExt
	centerH := bh - topExt - bottomExt
	blk := dst[topExt*dstStride:]
	for yy := 0; yy < centerH; yy++ {
		row := blk[yy*dstStride : yy*dstStride+bw]
		src := ref.Pix[refOff+yy*ref.Stride:]
		copy(row[leftExt:leftExt+centerW], src[:centerW])
		// extend left edge for this line
		if leftExt > 0 {
			fill := row[leftExt]
			left := row[:leftExt]
			for i := range left {
				left[i] = fill
			}
		}
		// extend right edge for this line
		if rightExt > 0 {
			fill := row[leftExt+centerW-1]
			right := row[leftExt+centerW:]
			for i := range right {
				right[i] = fill
			}
		}
	}
	// copy top/bottom edges by replicating the first/last visible row
	topRow := blk[:bw]
	for yy := 0; yy < topExt; yy++ {
		copy(dst[yy*dstStride:yy*dstStride+bw], topRow)
	}
	bottomRow := blk[(centerH-1)*dstStride : (centerH-1)*dstStride+bw]
	for yy := 0; yy < bottomExt; yy++ {
		off := (topExt + centerH + yy) * dstStride
		copy(dst[off:off+bw], bottomRow)
	}
}

// emuEdgeWindow materializes the full 8-tap halo window of a width x height
// block at (refX, refY) into edge (an emuEdgeStride-strided buffer) and
// returns a plane over it plus the block's origin inside that plane. The
// window spans filterTaps extra columns and filterTaps-1 extra rows so every
// horizontal/vertical/2D 8-bit kernel shape (including the width-4 and
// slide-vector over-reads) stays resident.
func emuEdgeWindow(ref frame.Plane, refX int, refY int, width int, height int, edge *[emuEdgeStride * emuEdgeRows]byte) (frame.Plane, int, int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	haloW := width + filterTaps
	haloH := height + filterTaps - 1
	emuEdge8(haloW, haloH, ref, refX-foX, refY-foY, edge[:], emuEdgeStride)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: emuEdgeStride,
		Width:  haloW,
		Height: haloH,
	}
	return emu, foX, foY
}

// emuEdgeWindow8X materializes only the horizontal halo of an 8-bit width x
// height block at (refX, refY). This mirrors dav1d's reconstruction edge path
// (src/recon_tmpl.c mc(): bw = w + !!mx*7, bh = h + !!my*7), with one extra
// byte of horizontal padding because goav1's NEON/DOTPROD/I8MM slide kernels
// intentionally over-read by one byte in their last vector group. Returns a
// plane over the resident window plus the block's x origin inside it; y origin
// is 0.
func emuEdgeWindow8X(ref frame.Plane, refX int, refY int, width int, height int, edge *[emuEdgeStride * emuEdgeRows]byte) (frame.Plane, int) {
	foX := filterTaps/2 - 1
	haloW := width + filterTaps
	emuEdge8(haloW, height, ref, refX-foX, refY, edge[:], emuEdgeStride)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: emuEdgeStride,
		Width:  haloW,
		Height: height,
	}
	return emu, foX
}

// emuEdgeWindow8Y is the vertical-only sibling of emuEdgeWindow8X. It
// materializes the width x (height+7) tap window and returns the block's y
// origin inside the resident plane; x origin is 0.
func emuEdgeWindow8Y(ref frame.Plane, refX int, refY int, width int, height int, edge *[emuEdgeStride * emuEdgeRows]byte) (frame.Plane, int) {
	foY := filterTaps/2 - 1
	haloH := height + filterTaps - 1
	emuEdge8(width, haloH, ref, refX, refY-foY, edge[:], emuEdgeStride)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: emuEdgeStride,
		Width:  width,
		Height: haloH,
	}
	return emu, foY
}
