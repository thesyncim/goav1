package goav1

// CopyTileListEntryToOutputFrame copies one already-decoded tile-list entry
// from src into dst using the source/destination rectangles derived from list
// and geometry. It is the pixel-copy stage of libaom's
// copy_decoded_tile_to_tile_list_buffer(): luma coordinates are copied as-is,
// and chroma coordinates are shifted by the frame subsampling factors.
//
// This helper does not decode entry.TileData. Callers use it after decoding the
// tile payload against the entry's anchor frame, then copy the decoded tile
// rectangle into the tile-list output mosaic.
func CopyTileListEntryToOutputFrame(dst *Frame, src *Frame, list TileList, geometry TileListOutputGeometry, entryIndex int) error {
	if dst == nil || src == nil {
		return ErrFrameInvalidFormat
	}
	if dst.Layout.BytesPerSample != src.Layout.BytesPerSample ||
		dst.Format.BitDepth != src.Format.BitDepth ||
		dst.Format.MonoChrome != src.Format.MonoChrome ||
		dst.Format.SubsamplingX != src.Format.SubsamplingX ||
		dst.Format.SubsamplingY != src.Format.SubsamplingY {
		return ErrFrameInvalidFormat
	}
	if dst.Format.Width < geometry.OutputFrameWidth || dst.Format.Height < geometry.OutputFrameHeight {
		return ErrFrameInvalidFormat
	}
	region, err := TileListOutputTileRegion(list, geometry, entryIndex)
	if err != nil {
		return err
	}
	bps := dst.Layout.BytesPerSample
	if bps != 1 && bps != 2 {
		return ErrFrameInvalidFormat
	}
	if !tileListPlaneRegionFits(dst.Y, bps, region.DestX, region.DestY, region.Width, region.Height) ||
		!tileListPlaneRegionFits(src.Y, bps, region.SourceX, region.SourceY, region.Width, region.Height) {
		return ErrFrameInvalidFormat
	}
	if !dst.Format.MonoChrome {
		shiftX, shiftY := 0, 0
		if dst.Format.SubsamplingX {
			shiftX = 1
		}
		if dst.Format.SubsamplingY {
			shiftY = 1
		}
		chroma := TileListTileRegion{
			SourceX: region.SourceX >> shiftX,
			SourceY: region.SourceY >> shiftY,
			DestX:   region.DestX >> shiftX,
			DestY:   region.DestY >> shiftY,
			Width:   region.Width >> shiftX,
			Height:  region.Height >> shiftY,
		}
		if !tileListPlaneRegionFits(dst.U, bps, chroma.DestX, chroma.DestY, chroma.Width, chroma.Height) ||
			!tileListPlaneRegionFits(src.U, bps, chroma.SourceX, chroma.SourceY, chroma.Width, chroma.Height) ||
			!tileListPlaneRegionFits(dst.V, bps, chroma.DestX, chroma.DestY, chroma.Width, chroma.Height) ||
			!tileListPlaneRegionFits(src.V, bps, chroma.SourceX, chroma.SourceY, chroma.Width, chroma.Height) {
			return ErrFrameInvalidFormat
		}
	}
	if err := CopyPlaneBlock(dst.Y, src.Y, bps, region.DestX, region.DestY, region.SourceX, region.SourceY, region.Width, region.Height); err != nil {
		return err
	}
	if dst.Format.MonoChrome {
		return nil
	}

	shiftX, shiftY := 0, 0
	if dst.Format.SubsamplingX {
		shiftX = 1
	}
	if dst.Format.SubsamplingY {
		shiftY = 1
	}
	chroma := TileListTileRegion{
		SourceX: region.SourceX >> shiftX,
		SourceY: region.SourceY >> shiftY,
		DestX:   region.DestX >> shiftX,
		DestY:   region.DestY >> shiftY,
		Width:   region.Width >> shiftX,
		Height:  region.Height >> shiftY,
	}
	if err := CopyPlaneBlock(dst.U, src.U, bps, chroma.DestX, chroma.DestY, chroma.SourceX, chroma.SourceY, chroma.Width, chroma.Height); err != nil {
		return err
	}
	return CopyPlaneBlock(dst.V, src.V, bps, chroma.DestX, chroma.DestY, chroma.SourceX, chroma.SourceY, chroma.Width, chroma.Height)
}

func tileListPlaneRegionFits(plane FramePlane, bytesPerSample int, x int, y int, width int, height int) bool {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return false
	}
	if plane.Stride <= 0 ||
		plane.Width <= 0 ||
		plane.Height <= 0 ||
		x < 0 ||
		y < 0 ||
		width <= 0 ||
		height <= 0 ||
		x > plane.Width-width ||
		y > plane.Height-height {
		return false
	}
	rowBytes, ok := tileListCheckedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
		return false
	}
	rowOffset, ok := tileListCheckedMul(y, plane.Stride)
	if !ok {
		return false
	}
	colOffset, ok := tileListCheckedMul(x, bytesPerSample)
	if !ok {
		return false
	}
	offset, ok := tileListCheckedAdd(rowOffset, colOffset)
	if !ok {
		return false
	}
	lastRowOffset, ok := tileListCheckedMul(height-1, plane.Stride)
	if !ok {
		return false
	}
	windowLen, ok := tileListCheckedAdd(lastRowOffset, rowBytes)
	if !ok {
		return false
	}
	end, ok := tileListCheckedAdd(offset, windowLen)
	return ok && offset >= 0 && end <= len(plane.Pix)
}

func tileListCheckedAdd(a int, b int) (int, bool) {
	c := a + b
	if (b > 0 && c < a) || (b < 0 && c > a) {
		return 0, false
	}
	return c, true
}

func tileListCheckedMul(a int, b int) (int, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	c := a * b
	if c/b != a {
		return 0, false
	}
	return c, true
}
