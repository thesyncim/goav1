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

	// SBSizeLog2 is the superblock size in log2 luma pixels (6 for 64x64, 7
	// for 128x128). Zero defaults to 6. The byte buffer is allocated to the
	// superblock-aligned extent so the bottom/right partial superblock can
	// reconstruct the full transform of any in-grid block into the padding,
	// mirroring libaom writing whole transform blocks into the YV12 border
	// (see RequiredSize).
	SBSizeLog2 uint8

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
	// aom_realloc_frame_buffer in aom_scale/generic/yv12config.c) plus a pixel
	// border (AOM_BORDER_IN_PIXELS). The bottom/right partial superblock
	// reconstructs the FULL transform of every in-grid block (libaom's
	// av1_inverse_transform_block writes the whole tx_size and only skips a
	// transform when its blk_row/blk_col origin reaches max_blocks_high/wide,
	// the MI grid edge); a transform whose origin is inside the MI grid but
	// whose extent crosses the cropped edge spills its trailing rows/cols into
	// that border. cfl_store_tx then subsamples the full reconstructed
	// transform with no boundary clamp, and right/bottom-edge chroma reads
	// those genuinely-reconstructed past-crop samples.
	//
	// We mirror that by allocating the byte buffer to the SUPERBLOCK-aligned
	// extent: any transform whose origin is inside the MI grid lies within a
	// superblock, so its full extent fits inside the superblock-aligned
	// allocation. The reported plane Width/Height stay at the cropped (visible)
	// extent, which is the surface MD5/output consumers hash. Downstream
	// reconstruction derives the writable extent from Stride and the Pix
	// length.
	sbSizeLog2 := uint(format.SBSizeLog2)
	if sbSizeLog2 == 0 {
		sbSizeLog2 = 6
	}
	sbSize := 1 << sbSizeLog2
	alignedWidth := (format.Width + sbSize - 1) &^ (sbSize - 1)
	alignedHeight := (format.Height + sbSize - 1) &^ (sbSize - 1)

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

// ExtendBorders edge-replicates each plane's cropped edge outward into the
// allocated superblock-aligned padding, mirroring libaom's
// aom_extend_frame_borders / aom_yv12_extend_frame_borders (which replicate the
// last valid row/column of each plane outward, plus the bottom-right corner,
// after a frame is decoded and before it can be used as a reference).
//
// The reconstruction foundation (RequiredSize) allocates the byte buffer to the
// superblock-aligned extent but leaves the region past the cropped Width/Height
// zero-filled. A bottom/right partial-superblock inter block whose
// motion-compensated read lands in that past-crop region would otherwise read
// zeros where libaom reads the replicated edge pixel. Extending the borders here
// makes those reads match libaom whether they go through the convolve's
// per-sample clamp or read the buffer directly. It is a no-op for any sample
// that lies inside the cropped extent, so frames with no off-edge reads are
// unaffected.
func (f *Frame) ExtendBorders() {
	bytesPerSample := f.Layout.BytesPerSample
	if bytesPerSample <= 0 {
		bytesPerSample = 1
	}
	f.Y.extendBorders(bytesPerSample)
	if !f.Format.MonoChrome {
		f.U.extendBorders(bytesPerSample)
		f.V.extendBorders(bytesPerSample)
	}
}

// extendBorders edge-replicates this plane's last valid column into the columns
// [Width, allocWidth) of every valid row, then replicates the last valid row
// (now including its extended columns) into rows [Height, allocHeight). The
// bottom-right corner is filled by the second pass copying the fully extended
// last row. bytesPerSample is 1 for 8-bit content and 2 for 10/12-bit content.
func (p *Plane) extendBorders(bytesPerSample int) {
	if p.Stride <= 0 || p.Width <= 0 || p.Height <= 0 || bytesPerSample <= 0 {
		return
	}
	allocWidth := p.Stride / bytesPerSample
	if allocWidth <= 0 {
		return
	}
	allocHeight := len(p.Pix) / p.Stride
	if allocHeight <= 0 {
		return
	}
	width := min(p.Width, allocWidth)
	height := min(p.Height, allocHeight)

	// Right edge: replicate the last valid sample of each valid row across the
	// padding columns [width, allocWidth).
	if width < allocWidth {
		lastCol := (width - 1) * bytesPerSample
		switch bytesPerSample {
		case 1:
			for y := 0; y < height; y++ {
				row := p.Pix[y*p.Stride : y*p.Stride+p.Stride]
				edge := row[lastCol]
				for x := width; x < allocWidth; x++ {
					row[x] = edge
				}
			}
		case 2:
			for y := 0; y < height; y++ {
				row := p.Pix[y*p.Stride : y*p.Stride+p.Stride]
				lo := row[lastCol]
				hi := row[lastCol+1]
				for x := width; x < allocWidth; x++ {
					dst := x * 2
					row[dst] = lo
					row[dst+1] = hi
				}
			}
		default:
			for y := 0; y < height; y++ {
				row := p.Pix[y*p.Stride : y*p.Stride+p.Stride]
				edge := row[lastCol : lastCol+bytesPerSample]
				for x := width; x < allocWidth; x++ {
					dst := x * bytesPerSample
					copy(row[dst:dst+bytesPerSample], edge)
				}
			}
		}
	}

	// Bottom edge: replicate the last valid row (already right-extended) into
	// the padding rows [height, allocHeight). This also fills the bottom-right
	// corner.
	if height < allocHeight {
		lastRow := p.Pix[(height-1)*p.Stride : (height-1)*p.Stride+p.Stride]
		for y := height; y < allocHeight; y++ {
			dst := p.Pix[y*p.Stride : y*p.Stride+p.Stride]
			copy(dst, lastRow)
		}
	}
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
	if format.SBSizeLog2 == 0 {
		format.SBSizeLog2 = 6
	}
	if format.SBSizeLog2 != 6 && format.SBSizeLog2 != 7 {
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
