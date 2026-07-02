// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package frame

type loadSampleRows8NEONCtx struct {
	dst    *uint16
	src    *byte
	dstStr uintptr // dst stride in uint16 elements
	srcStr uintptr // src stride in bytes
	width  uintptr
	height uintptr
}

//go:noescape
func loadSampleRows8NEONAsm(ctx *loadSampleRows8NEONCtx)

// loadSampleRows8 binds the NEON widen on arm64. NEON (AdvSIMD) is mandatory
// on arm64 so this is a static binding; see the dispatch note on
// loadSampleRows8PureGo for why it is not a func variable.
func loadSampleRows8(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	loadSampleRows8NEON(dst, dstStride, src, srcStride, width, height)
}

// loadSampleRows8NEON widens 8-bit sample rows into uint16 staging storage
// with vector loads (ld1 16b -> ushll/ushll2 -> st1 2x8h). Rows whose width
// is not a vector multiple finish with one overlapping 8-wide group aligned
// to the row end, which rewrites up to seven already-written columns with
// identical values; widths below one vector fall back to the pure-Go
// reference. Output is bit-identical to loadSampleRows8PureGo.
func loadSampleRows8NEON(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width < 8 || height <= 0 {
		loadSampleRows8PureGo(dst, dstStride, src, srcStride, width, height)
		return
	}
	// Bounds contract identical to the pure-Go reference slices: the last
	// row's [0, width) region must be resident in both buffers.
	_ = src[(height-1)*srcStride+width-1]
	_ = dst[(height-1)*dstStride+width-1]
	ctx := loadSampleRows8NEONCtx{
		dst:    &dst[0],
		src:    &src[0],
		dstStr: uintptr(dstStride),
		srcStr: uintptr(srcStride),
		width:  uintptr(width),
		height: uintptr(height),
	}
	loadSampleRows8NEONAsm(&ctx)
}
