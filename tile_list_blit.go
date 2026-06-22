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
