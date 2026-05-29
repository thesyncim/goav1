// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import "errors"

var (
	ErrInvalidFormat = errors.New("frame: invalid format")
	ErrInvalidPool   = errors.New("frame: invalid pool")
	ErrPoolEmpty     = errors.New("frame: pool empty")
	ErrInvalidSlot   = errors.New("frame: invalid pool slot")
	ErrInvalidPlane  = errors.New("frame: invalid plane")
	ErrShortBuffer   = errors.New("frame: short buffer")
)

type Format struct {
	Width  int
	Height int

	BitDepth uint8

	MonoChrome   bool
	SubsamplingX bool
	SubsamplingY bool

	Align int
}

type Layout struct {
	YOffset int
	UOffset int
	VOffset int

	YStride int
	UStride int
	VStride int

	ChromaWidth  int
	ChromaHeight int

	BytesPerSample int
	Size           int
}

type Plane struct {
	Pix    []byte
	Stride int
	Width  int
	Height int
}

type Frame struct {
	Format Format
	Layout Layout

	Y Plane
	U Plane
	V Plane
}

func RequiredSize(format Format) (Layout, error) {
	format, err := normalizeFormat(format)
	if err != nil {
		return Layout{}, err
	}

	bytesPerSample := 1
	if format.BitDepth > 8 {
		bytesPerSample = 2
	}

	// libaom allocates the YV12 buffer at the MI-aligned dimensions
	// (aligned_width = ROUND_POWER_OF_TWO(width, 3) << 3 etc., see
	// aom_realloc_frame_buffer in aom_scale/generic/yv12config.c) so the
	// bottom/right partial superblock can be reconstructed into the padding
	// rows/cols and used as intra-prediction neighbor context and CDEF
	// bottom/right-edge input (cdef_prepare_fb vsize = nvb << mi_high_l2).
	// We mirror that: the byte buffer spans the MI-aligned extent while the
	// reported plane Width/Height stay at the cropped (visible) extent, which
	// is the surface MD5/output consumers hash. Downstream reconstruction
	// derives the aligned writable extent from Stride and the Pix length.
	alignedWidth := (format.Width + 7) &^ 7
	alignedHeight := (format.Height + 7) &^ 7

	yStride, ok := checkedAlign(alignedWidth*bytesPerSample, format.Align)
	if !ok {
		return Layout{}, ErrInvalidFormat
	}

	ySize, ok := checkedMul(yStride, alignedHeight)
	if !ok {
		return Layout{}, ErrInvalidFormat
	}

	chromaWidth := 0
	chromaHeight := 0
	cStride := 0
	chromaAlignedHeight := 0
	if !format.MonoChrome {
		chromaWidth = format.Width
		chromaHeight = format.Height
		alignedChromaWidth := alignedWidth
		chromaAlignedHeight = alignedHeight
		if format.SubsamplingX {
			chromaWidth = (chromaWidth + 1) >> 1
			alignedChromaWidth >>= 1
		}
		if format.SubsamplingY {
			chromaHeight = (chromaHeight + 1) >> 1
			chromaAlignedHeight >>= 1
		}
		cStride, ok = checkedAlign(alignedChromaWidth*bytesPerSample, format.Align)
		if !ok {
			return Layout{}, ErrInvalidFormat
		}
	}
	uSize, ok := checkedMul(cStride, chromaAlignedHeight)
	if !ok {
		return Layout{}, ErrInvalidFormat
	}
	vOffset, ok := checkedAdd(ySize, uSize)
	if !ok {
		return Layout{}, ErrInvalidFormat
	}
	total, ok := checkedAdd(vOffset, uSize)
	if !ok {
		return Layout{}, ErrInvalidFormat
	}

	return Layout{
		YOffset:        0,
		UOffset:        ySize,
		VOffset:        vOffset,
		YStride:        yStride,
		UStride:        cStride,
		VStride:        cStride,
		ChromaWidth:    chromaWidth,
		ChromaHeight:   chromaHeight,
		BytesPerSample: bytesPerSample,
		Size:           total,
	}, nil
}

func Bind(buffer []byte, format Format) (Frame, error) {
	normalized, err := normalizeFormat(format)
	if err != nil {
		return Frame{}, err
	}
	layout, err := RequiredSize(normalized)
	if err != nil {
		return Frame{}, err
	}
	if len(buffer) < layout.Size {
		return Frame{}, ErrShortBuffer
	}

	ySize := layout.UOffset - layout.YOffset
	uSize := layout.VOffset - layout.UOffset
	return Frame{
		Format: normalized,
		Layout: layout,
		Y: Plane{
			Pix:    buffer[layout.YOffset : layout.YOffset+ySize],
			Stride: layout.YStride,
			Width:  normalized.Width,
			Height: normalized.Height,
		},
		U: Plane{
			Pix:    buffer[layout.UOffset : layout.UOffset+uSize],
			Stride: layout.UStride,
			Width:  layout.ChromaWidth,
			Height: layout.ChromaHeight,
		},
		V: Plane{
			Pix:    buffer[layout.VOffset:layout.Size],
			Stride: layout.VStride,
			Width:  layout.ChromaWidth,
			Height: layout.ChromaHeight,
		},
	}, nil
}

func normalizeFormat(format Format) (Format, error) {
	if format.Width <= 0 || format.Height <= 0 {
		return Format{}, ErrInvalidFormat
	}
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	if format.BitDepth != 8 && format.BitDepth != 10 && format.BitDepth != 12 {
		return Format{}, ErrInvalidFormat
	}
	if format.Align <= 0 {
		format.Align = 1
	}
	if format.Align&(format.Align-1) != 0 {
		return Format{}, ErrInvalidFormat
	}
	return format, nil
}

func checkedAlign(value int, align int) (int, bool) {
	add := align - 1
	value, ok := checkedAdd(value, add)
	if !ok {
		return 0, false
	}
	return value &^ add, true
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
