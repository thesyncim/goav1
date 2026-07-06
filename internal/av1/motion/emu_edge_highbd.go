// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// High-bit-depth (10/12-bit) emulated-edge window. This is the 16bpc sibling of
// emuEdge8 / emuEdgeWindow (emu_edge.go): dav1d's reconstruction
// (src/recon_tmpl.c mc(), src/mc_tmpl.c emu_edge_c) materializes the clamped
// tap-window halo into a small scratch buffer and then runs the plain
// resident put/prep 8tap kernel over it. The samples fed to the kernel are
// exactly what a per-tap edge-clamped convolution would load, so the result is
// bit-identical to the pure-Go clamped reference.
//
// Samples are little-endian uint16 (2 bytes/sample) and the plane Stride is in
// bytes, matching frame.Plane / loadHighBDSample.

// emuEdge16Stride is one row of the 16bpc emulated-edge scratch window, in
// samples. The widest inter block is maxBlockSize; the resident NEON horizontal
// convolves load 8 samples at [cursor] and another 8 at [cursor+16 bytes], so
// the final width-4 column group over-reads up to 5 samples past the
// width+filterTaps-1 tap span. The extra filterTaps of slack columns absorb
// that unused tail so the last-row load never crosses the buffer end.
const emuEdge16Stride = maxBlockSize + 2*filterTaps

// emuEdge16Rows bounds the tallest 16bpc emulated-edge window: the tallest
// inter block plus the vertical 8-tap halo (height+filterTaps-1), with one
// extra slack row so the horizontal pass's trailing column over-read on the
// last processed row stays inside the allocation.
const emuEdge16Rows = maxBlockSize + filterTaps

// emuEdge16Buf is the byte-backed scratch for a full 16bpc halo window
// (2 bytes/sample).
type emuEdge16Buf [emuEdge16Stride * emuEdge16Rows * 2]byte

// emuEdge16 ports emu_edge_c (src/mc_tmpl.c) for 16bpc planes: it materializes
// the bw x bh source window whose top-left corner is (x, y) in ref coordinates
// — possibly overhanging the plane on any side — into dst (a dstStrideBytes-
// strided byte buffer, 2 bytes/sample) with edge-clamp replication. It mirrors
// emuEdge8 sample-for-sample, scaled to 2-byte samples.
func emuEdge16(bw int, bh int, ref frame.Plane, x int, y int, dst []byte, dstStrideBytes int) {
	iw := ref.Width
	ih := ref.Height

	// find byte offset in reference of visible block to copy
	refOff := clampInt(y, 0, ih-1)*ref.Stride + clampInt(x, 0, iw-1)*2

	// number of pixels to extend (left, right, top, bottom)
	leftExt := clampInt(-x, 0, bw-1)
	rightExt := clampInt(x+bw-iw, 0, bw-1)
	topExt := clampInt(-y, 0, bh-1)
	bottomExt := clampInt(y+bh-ih, 0, bh-1)

	centerW := bw - leftExt - rightExt
	centerH := bh - topExt - bottomExt
	blk := dst[topExt*dstStrideBytes:]
	for yy := 0; yy < centerH; yy++ {
		row := blk[yy*dstStrideBytes : yy*dstStrideBytes+bw*2]
		src := ref.Pix[refOff+yy*ref.Stride:]
		copy(row[leftExt*2:(leftExt+centerW)*2], src[:centerW*2])
		// extend left edge for this line
		if leftExt > 0 {
			f0, f1 := row[leftExt*2], row[leftExt*2+1]
			for i := 0; i < leftExt; i++ {
				row[i*2], row[i*2+1] = f0, f1
			}
		}
		// extend right edge for this line
		if rightExt > 0 {
			base := (leftExt + centerW - 1) * 2
			f0, f1 := row[base], row[base+1]
			for i := leftExt + centerW; i < bw; i++ {
				row[i*2], row[i*2+1] = f0, f1
			}
		}
	}
	// copy top/bottom edges by replicating the first/last visible row
	topRow := blk[:bw*2]
	for yy := 0; yy < topExt; yy++ {
		copy(dst[yy*dstStrideBytes:yy*dstStrideBytes+bw*2], topRow)
	}
	bottomRow := blk[(centerH-1)*dstStrideBytes : (centerH-1)*dstStrideBytes+bw*2]
	for yy := 0; yy < bottomExt; yy++ {
		off := (topExt + centerH + yy) * dstStrideBytes
		copy(dst[off:off+bw*2], bottomRow)
	}
}

// emuEdgeWindow16 materializes the full 8-tap halo window of a width x height
// 16bpc block at (refX, refY) into edge and returns a plane over it plus the
// block's origin inside that plane. The window spans filterTaps extra columns
// and filterTaps-1 extra rows so every horizontal/vertical/2D HBD kernel shape
// stays resident.
func emuEdgeWindow16(ref frame.Plane, refX int, refY int, width int, height int, edge *emuEdge16Buf) (frame.Plane, int, int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	haloW := width + filterTaps
	haloH := height + filterTaps - 1
	const strideBytes = emuEdge16Stride * 2
	emuEdge16(haloW, haloH, ref, refX-foX, refY-foY, edge[:], strideBytes)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: strideBytes,
		Width:  haloW,
		Height: haloH,
	}
	return emu, foX, foY
}

// emuEdgeWindow16X materializes only the horizontal halo of a width x height
// 16bpc block at (refX, refY): dav1d's reconstruction sizes the emu_edge
// window per filtered direction (src/recon_tmpl.c mc(): bw = w + !!mx * 7,
// bh = h + !!my * 7), so a horizontal-only convolve copies just height rows.
// Returns a plane over the window plus the block's x origin inside it (its y
// origin is 0).
func emuEdgeWindow16X(ref frame.Plane, refX int, refY int, width int, height int, edge *emuEdge16Buf) (frame.Plane, int) {
	foX := filterTaps/2 - 1
	haloW := width + filterTaps
	const strideBytes = emuEdge16Stride * 2
	emuEdge16(haloW, height, ref, refX-foX, refY, edge[:], strideBytes)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: strideBytes,
		Width:  haloW,
		Height: height,
	}
	return emu, foX
}

// emuEdgeWindow16Y materializes only the vertical halo of a width x height
// 16bpc block at (refX, refY): dav1d's reconstruction sizes the emu_edge
// window per filtered direction (src/recon_tmpl.c mc(): bw = w + !!mx * 7,
// bh = h + !!my * 7), so a vertical-only convolve copies just width columns.
// Returns a plane over the window plus the block's y origin inside it (its x
// origin is 0).
func emuEdgeWindow16Y(ref frame.Plane, refX int, refY int, width int, height int, edge *emuEdge16Buf) (frame.Plane, int) {
	foY := filterTaps/2 - 1
	haloH := height + filterTaps - 1
	const strideBytes = emuEdge16Stride * 2
	emuEdge16(width, haloH, ref, refX, refY-foY, edge[:], strideBytes)
	emu := frame.Plane{
		Pix:    edge[:],
		Stride: strideBytes,
		Width:  width,
		Height: haloH,
	}
	return emu, foY
}
