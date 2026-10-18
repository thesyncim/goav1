package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Bounds are pixel coordinates inside one 64x64 luma (or subsampled) unit.
// The entire bounding rectangle is staged: the mixed-strength SIMD route
// also scans the gaps between selected blocks when proving its tap bounds.
type frameWorkCDEFStageRect struct {
	x0, y0, x1, y1 uint8
	active         bool
}

func frameWorkCDEFSparseStageRect(blocks []cdef.BlockPosition, w, h, bw, bh int) frameWorkCDEFStageRect {
	if len(blocks) == 0 || w <= 0 || h <= 0 || w > 64 || h > 64 ||
		(bw != 4 && bw != 8) || (bh != 4 && bh != 8) || w%bw != 0 || h%bh != 0 {
		return frameWorkCDEFStageRect{}
	}
	x0, y0, x1, y1 := w, h, 0, 0
	for _, block := range blocks {
		x, y := int(block.BX)*bw, int(block.BY)*bh
		x0, y0 = min(x0, x), min(y0, y)
		x1, y1 = max(x1, x+bw), max(y1, y+bh)
	}
	if x1 > w || y1 > h || (x0 == 0 && y0 == 0 && x1 == w && y1 == h) {
		return frameWorkCDEFStageRect{}
	}
	return frameWorkCDEFStageRect{uint8(x0), uint8(y0), uint8(x1), uint8(y1), true}
}

// Preserve the original copy order for any caller-supplied source/scratch
// overlap. Addresses are only compared; no pointer is reconstructed.
func frameWorkCDEFSparseScratchIndependent(input []uint16, pix, lines, top, bottom []byte) bool {
	view, ok := cdefByteScratchView(input, 2*len(input))
	return ok && !frameWorkCDEFBytesOverlap(view, pix) &&
		!frameWorkCDEFBytesOverlap(view, lines) &&
		!frameWorkCDEFBytesOverlap(view, top) && !frameWorkCDEFBytesOverlap(view, bottom)
}

func frameWorkCDEFBytesOverlap(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap <= bp {
		return bp-ap < uintptr(len(a))
	}
	return ap-bp < uintptr(len(b))
}

// The byte route already requires an entirely real two-pixel unit halo.
// Leave its kernel, original validation, and fallback unchanged; copy only
// the selected blocks' complete rectangle plus the same two-pixel halo.
func frameWorkAssembleCDEFSparseByteInputU8(input []byte, dst frame.Plane, top, bottom, left []byte, leftBackup bool, ux, uy, uw, uh int, rect frameWorkCDEFStageRect) bool {
	const reach = 2
	width, stride := dst.Width, dst.Stride
	if !rect.active || rect.x0 >= rect.x1 || rect.y0 >= rect.y1 || int(rect.x1) > uw || int(rect.y1) > uh ||
		uw <= 0 || uh <= 0 || ux < reach || uy < reach || ux+uw+reach > width || uy+uh+reach > dst.Height ||
		uw+4 > cdef.BStride || stride < width || (dst.Height-1)*stride+width > len(dst.Pix) ||
		cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder+(uh+1)*cdef.BStride+uw+2 > len(input) ||
		len(top) < 2*width || bottom != nil && len(bottom) < 2*width || leftBackup && len(left) < cdef.HorizontalBorder*uh {
		return false
	}
	x0, x1 := int(rect.x0)-reach, int(rect.x1)+reach
	y0, y1 := int(rect.y0)-reach, int(rect.y1)+reach
	span := x1 - x0
	const origin = cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder
	for y := y0; y < y1; y++ {
		out := input[origin+y*cdef.BStride+x0:][:span]
		switch {
		case y < 0:
			copy(out, top[(y+2)*width+ux+x0:][:span])
		case y >= uh && bottom != nil:
			copy(out, bottom[(y-uh)*width+ux+x0:][:span])
		case y < uh && x0 < 0 && leftBackup:
			copy(out[:-x0], left[y*cdef.HorizontalBorder+cdef.HorizontalBorder+x0:][:-x0])
			copy(out[-x0:], dst.Pix[(uy+y)*stride+ux:][:x1])
		default:
			copy(out, dst.Pix[(uy+y)*stride+ux+x0:][:span])
		}
	}
	return true
}

// The widened route keeps the original eight-column/two-row halo, including
// frame-boundary sentinels and the aligned-view right/bottom replication.
// Every source region is copied before any filtering of this unit begins.
func frameWorkAssembleCDEFSparseInputU8(input []uint16, dst frame.Plane, top, bottom, left []byte, leftBackup bool, unitRow, ux, uy, uw, uh int, rect frameWorkCDEFStageRect) bool {
	width, height, stride := dst.Width, dst.Height, dst.Stride
	if !rect.active || rect.x0 >= rect.x1 || rect.y0 >= rect.y1 || int(rect.x1) > uw || int(rect.y1) > uh ||
		uw <= 0 || uh <= 0 || ux < 0 || uy < 0 || ux+uw > width || uy+uh > height ||
		uw+2*cdef.HorizontalBorder > cdef.BStride || !cdefInputRectFits(len(input), uw+2*cdef.HorizontalBorder, uh+2*cdef.VerticalBorder) ||
		stride < width || (height-1)*stride+width > len(dst.Pix) ||
		uy > 0 && (unitRow == 0 || len(top) < 2*width) || bottom != nil && len(bottom) < 2*width ||
		leftBackup && len(left) < cdef.HorizontalBorder*uh {
		return false
	}
	x0, x1 := int(rect.x0)-cdef.HorizontalBorder, int(rect.x1)+cdef.HorizontalBorder
	y0, y1 := int(rect.y0)-cdef.VerticalBorder, int(rect.y1)+cdef.VerticalBorder
	sx0, sx1 := max(ux+x0, 0), min(ux+x1, width)
	sy0, sy1 := max(uy+y0, 0), min(uy+y1, height)
	colExtend := ux+x1 > width && ux+uw < width
	rowExtend := uy+y1 > height && uy+uh < height
	coveredX1, coveredY1 := sx1-ux, sy1-uy
	if colExtend {
		coveredX1 = x1
	}
	if rowExtend {
		coveredY1 = y1
	}
	const origin = cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder
	for y := y0; y < y1; y++ {
		out := input[origin+y*cdef.BStride+x0:]
		if y < sy0-uy || y >= coveredY1 {
			frameWorkFillCDEFInputSentinelRow(out, x1-x0)
			continue
		}
		if n := sx0 - ux - x0; n > 0 {
			frameWorkFillCDEFInputSentinelRow(out, n)
		}
		if coveredX1 < x1 {
			frameWorkFillCDEFInputSentinelRow(out[coveredX1-x0:], x1-coveredX1)
		}
	}
	copyW := sx1 - sx0
	at := origin + (sy0-uy)*cdef.BStride + sx0 - ux
	if n := min(sy1, uy) - sy0; n > 0 {
		frame.LoadSampleRows8Trusted(input[at:], cdef.BStride, top[(sy0-uy+2)*width+sx0:], width, copyW, n)
	}
	my0, my1 := max(sy0, uy), min(sy1, uy+uh)
	mid := origin + (my0-uy)*cdef.BStride + sx0 - ux
	if n := ux - sx0; n > 0 {
		if leftBackup {
			frame.LoadSampleRows8Trusted(input[mid:], cdef.BStride, left[(my0-uy)*cdef.HorizontalBorder+cdef.HorizontalBorder-n:], cdef.HorizontalBorder, n, my1-my0)
		} else {
			frame.LoadSampleRows8Trusted(input[mid:], cdef.BStride, dst.Pix[my0*stride+sx0:], stride, n, my1-my0)
		}
	}
	mx0 := max(sx0, ux)
	frame.LoadSampleRows8Trusted(input[mid+mx0-sx0:], cdef.BStride, dst.Pix[my0*stride+mx0:], stride, sx1-mx0, my1-my0)
	by0 := max(sy0, uy+uh)
	if n := sy1 - by0; n > 0 {
		out := origin + (by0-uy)*cdef.BStride + sx0 - ux
		if bottom != nil {
			frame.LoadSampleRows8Trusted(input[out:], cdef.BStride, bottom[(by0-uy-uh)*width+sx0:], width, copyW, n)
		} else {
			frame.LoadSampleRows8Trusted(input[out:], cdef.BStride, dst.Pix[by0*stride+sx0:], stride, copyW, n)
		}
	}
	if rowExtend {
		last := origin + (sy1-uy-1)*cdef.BStride + sx0 - ux
		for y := sy1 - uy; y < y1; y++ {
			out := origin + y*cdef.BStride + sx0 - ux
			copy(input[out:out+copyW], input[last:last+copyW])
		}
	}
	if colExtend {
		for y := sy0 - uy; y < coveredY1; y++ {
			row := origin + y*cdef.BStride
			last := input[row+sx1-ux-1]
			for x := sx1 - ux; x < x1; x++ {
				input[row+x] = last
			}
		}
	}
	return true
}
