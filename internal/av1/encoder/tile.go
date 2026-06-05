package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	MaxTileRows = parser.MaxTileRows
	MaxTileCols = parser.MaxTileCols

	maxTileWidthPixels = 4096
	maxTileAreaPixels  = 4096 * 2304
)

type InterpolationFilter = parser.InterpolationFilter
type TileInfo = parser.TileInfo

const (
	InterpolationEightTap   = parser.InterpolationEightTap
	InterpolationSmooth     = parser.InterpolationSmooth
	InterpolationSharp      = parser.InterpolationSharp
	InterpolationBilinear   = parser.InterpolationBilinear
	InterpolationSwitchable = parser.InterpolationSwitchable
)

func TileInfoPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) (int, error) {
	w := newSizingBitWriter()
	if err := writeTileInfoPayload(&w, seq, prefix, codedWidth, height, tiles); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendTileInfoPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) ([]byte, error) {
	payloadSize, err := TileInfoPayloadSize(seq, prefix, codedWidth, height, tiles)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeTileInfoPayload(&w, seq, prefix, codedWidth, height, tiles); err != nil {
		return dst, err
	}
	return out, nil
}

func writeTileInfoPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) error {
	derived, err := deriveEncoderTileInfo(seq, codedWidth, height, tiles)
	if err != nil {
		return err
	}
	if err := validateTileInfo(seq, prefix, tiles, derived); err != nil {
		return err
	}
	if prefix.FrameType.interOrSwitch() {
		if !prefix.ForceIntegerMV {
			if err := w.writeBool(tiles.AllowHighPrecisionMV); err != nil {
				return err
			}
		}
		if tiles.InterpolationFilter == InterpolationSwitchable {
			if err := w.writeBool(true); err != nil {
				return err
			}
		} else {
			if err := w.writeBool(false); err != nil {
				return err
			}
			if err := w.writeBits(uint64(tiles.InterpolationFilter), 2); err != nil {
				return err
			}
		}
		if err := w.writeBool(tiles.SwitchableMotionMode); err != nil {
			return err
		}
		if !prefix.ErrorResilientMode && seq.EnableRefFrameMVS && seq.EnableOrderHint {
			if err := w.writeBool(tiles.UseRefFrameMVS); err != nil {
				return err
			}
		}
	}
	if !seq.ReducedStillPictureHeader && !prefix.DisableCDFUpdate {
		if err := w.writeBool(!tiles.RefreshContext); err != nil {
			return err
		}
	}
	if err := w.writeBool(tiles.UniformSpacing); err != nil {
		return err
	}
	if tiles.UniformSpacing {
		if err := writeUniformTileLayout(w, derived, tiles); err != nil {
			return err
		}
	} else if err := writeExplicitTileLayout(w, derived, tiles); err != nil {
		return err
	}
	if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
		if err := w.writeBits(uint64(tiles.ContextUpdateTileID), tiles.Log2Cols+tiles.Log2Rows); err != nil {
			return err
		}
		return w.writeBits(uint64(tiles.TileSizeBytes-1), 2)
	}
	return nil
}

type encoderTileDerived struct {
	SBCols           uint16
	SBRows           uint16
	MinLog2Cols      uint8
	MaxLog2Cols      uint8
	MinLog2Rows      uint8
	MaxLog2Rows      uint8
	MinLog2Tiles     uint8
	MaxTileWidthSB   uint16
	MaxTileAreaSB    uint32
	ExplicitMaxRowSB uint16
}

func deriveEncoderTileInfo(seq SequenceHeader, codedWidth uint32, height uint32, tiles TileInfo) (encoderTileDerived, error) {
	if err := validateSequenceHeader(seq); err != nil {
		return encoderTileDerived{}, err
	}
	if codedWidth == 0 || height == 0 {
		return encoderTileDerived{}, ErrInvalidFrame
	}
	sbSizeLog2 := uint8(6)
	if seq.Use128x128Superblock {
		sbSizeLog2 = 7
	}
	sbCols := ceilShift32Encoder(codedWidth, sbSizeLog2)
	sbRows := ceilShift32Encoder(height, sbSizeLog2)
	if sbCols == 0 || sbRows == 0 || sbCols > 65535 || sbRows > 65535 {
		return encoderTileDerived{}, ErrInvalidFrame
	}
	maxTileWidthSB := maxTileWidthPixels >> sbSizeLog2
	maxTileAreaSB := maxTileAreaPixels >> (2 * sbSizeLog2)
	minLog2Cols := uint8(tileLog2Encoder(maxTileWidthSB, int(sbCols)))
	maxLog2Cols := uint8(tileLog2Encoder(1, minInt(int(sbCols), MaxTileCols)))
	maxLog2Rows := uint8(tileLog2Encoder(1, minInt(int(sbRows), MaxTileRows)))
	minLog2Tiles := uint8(maxInt(tileLog2Encoder(maxTileAreaSB, int(sbCols*sbRows)), int(minLog2Cols)))
	explicitMaxArea := maxTileAreaSB
	if minLog2Tiles != 0 {
		explicitMaxArea >>= minLog2Tiles + 1
	}
	widest := widestTileSB(tiles)
	if widest == 0 {
		widest = 1
	}
	return encoderTileDerived{
		SBCols:           uint16(sbCols),
		SBRows:           uint16(sbRows),
		MinLog2Cols:      minLog2Cols,
		MaxLog2Cols:      maxLog2Cols,
		MaxLog2Rows:      maxLog2Rows,
		MinLog2Tiles:     minLog2Tiles,
		MaxTileWidthSB:   uint16(maxTileWidthSB),
		MaxTileAreaSB:    uint32(maxTileAreaSB),
		ExplicitMaxRowSB: uint16(maxInt(explicitMaxArea/widest, 1)),
	}, nil
}

func validateTileInfo(seq SequenceHeader, prefix FrameHeaderPrefix, tiles TileInfo, derived encoderTileDerived) error {
	if !prefix.FrameType.valid() || prefix.ShowExistingFrame {
		return ErrInvalidFrame
	}
	if !prefix.FrameType.interOrSwitch() {
		if tiles.AllowHighPrecisionMV || tiles.InterpolationFilter != 0 || tiles.SwitchableMotionMode || tiles.UseRefFrameMVS {
			return ErrInvalidFrame
		}
	} else {
		if prefix.ForceIntegerMV && tiles.AllowHighPrecisionMV {
			return ErrInvalidFrame
		}
		if tiles.InterpolationFilter > InterpolationSwitchable {
			return ErrInvalidFrame
		}
		if (prefix.ErrorResilientMode || !seq.EnableRefFrameMVS || !seq.EnableOrderHint) && tiles.UseRefFrameMVS {
			return ErrInvalidFrame
		}
	}
	if (seq.ReducedStillPictureHeader || prefix.DisableCDFUpdate) && tiles.RefreshContext {
		return ErrInvalidFrame
	}
	if tiles.SBCols != derived.SBCols || tiles.SBRows != derived.SBRows ||
		tiles.MinLog2Cols != derived.MinLog2Cols || tiles.MaxLog2Cols != derived.MaxLog2Cols ||
		tiles.MaxLog2Rows != derived.MaxLog2Rows || tiles.MinLog2Tiles != derived.MinLog2Tiles {
		return ErrInvalidFrame
	}
	if tiles.Cols == 0 || tiles.Rows == 0 || tiles.Cols > MaxTileCols || tiles.Rows > MaxTileRows {
		return ErrInvalidFrame
	}
	if tiles.ColStartSB[0] != 0 || tiles.RowStartSB[0] != 0 ||
		tiles.ColStartSB[tiles.Cols] != derived.SBCols || tiles.RowStartSB[tiles.Rows] != derived.SBRows {
		return ErrInvalidFrame
	}
	if tiles.Log2Cols != uint8(tileLog2Encoder(1, int(tiles.Cols))) || tiles.Log2Rows != uint8(tileLog2Encoder(1, int(tiles.Rows))) {
		return ErrInvalidFrame
	}
	if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
		if tiles.ContextUpdateTileID >= uint16(tiles.Cols)*uint16(tiles.Rows) || tiles.TileSizeBytes == 0 || tiles.TileSizeBytes > 4 {
			return ErrInvalidFrame
		}
	} else if tiles.ContextUpdateTileID != 0 || tiles.TileSizeBytes != 0 {
		return ErrInvalidFrame
	}
	if tiles.BitsRead != 0 {
		return ErrInvalidFrame
	}
	return validateTileStarts(tiles)
}

func validateTileStarts(tiles TileInfo) error {
	for i := uint8(0); i < tiles.Cols; i++ {
		if tiles.ColStartSB[i] >= tiles.ColStartSB[i+1] {
			return ErrInvalidFrame
		}
	}
	for i := uint8(0); i < tiles.Rows; i++ {
		if tiles.RowStartSB[i] >= tiles.RowStartSB[i+1] {
			return ErrInvalidFrame
		}
	}
	return nil
}

func writeUniformTileLayout(w *bitWriter, derived encoderTileDerived, tiles TileInfo) error {
	if tiles.Log2Cols < derived.MinLog2Cols || tiles.Log2Cols > derived.MaxLog2Cols {
		return ErrInvalidFrame
	}
	for log2Cols := derived.MinLog2Cols; log2Cols < derived.MaxLog2Cols; log2Cols++ {
		more := log2Cols < tiles.Log2Cols
		if err := w.writeBool(more); err != nil {
			return err
		}
		if !more {
			break
		}
	}
	minRows := maxInt(int(derived.MinLog2Tiles)-int(tiles.Log2Cols), 0)
	if tiles.MinLog2Rows != uint8(minRows) || tiles.Log2Rows < uint8(minRows) || tiles.Log2Rows > derived.MaxLog2Rows {
		return ErrInvalidFrame
	}
	for log2Rows := uint8(minRows); log2Rows < derived.MaxLog2Rows; log2Rows++ {
		more := log2Rows < tiles.Log2Rows
		if err := w.writeBool(more); err != nil {
			return err
		}
		if !more {
			break
		}
	}
	return validateUniformTileStarts(derived, tiles)
}

func validateUniformTileStarts(derived encoderTileDerived, tiles TileInfo) error {
	tileW := uint16(1 + ((uint32(derived.SBCols) - 1) >> tiles.Log2Cols))
	cols := uint8(0)
	for sbx := uint16(0); sbx < derived.SBCols; sbx += tileW {
		if cols >= MaxTileCols || tiles.ColStartSB[cols] != sbx {
			return ErrInvalidFrame
		}
		cols++
	}
	if cols != tiles.Cols {
		return ErrInvalidFrame
	}
	tileH := uint16(1 + ((uint32(derived.SBRows) - 1) >> tiles.Log2Rows))
	rows := uint8(0)
	for sby := uint16(0); sby < derived.SBRows; sby += tileH {
		if rows >= MaxTileRows || tiles.RowStartSB[rows] != sby {
			return ErrInvalidFrame
		}
		rows++
	}
	if rows != tiles.Rows {
		return ErrInvalidFrame
	}
	return nil
}

func writeExplicitTileLayout(w *bitWriter, derived encoderTileDerived, tiles TileInfo) error {
	for col := uint8(0); col < tiles.Cols; col++ {
		width := tiles.ColStartSB[col+1] - tiles.ColStartSB[col]
		remaining := derived.SBCols - tiles.ColStartSB[col]
		tileWidthLimit := minInt(int(remaining), int(derived.MaxTileWidthSB))
		if int(width) > tileWidthLimit {
			return ErrInvalidFrame
		}
		if tileWidthLimit > 1 {
			if err := writeUniform(w, uint32(tileWidthLimit), uint32(width-1)); err != nil {
				return err
			}
		}
	}
	for row := uint8(0); row < tiles.Rows; row++ {
		height := tiles.RowStartSB[row+1] - tiles.RowStartSB[row]
		remaining := derived.SBRows - tiles.RowStartSB[row]
		tileHeightLimit := minInt(int(remaining), int(derived.ExplicitMaxRowSB))
		if int(height) > tileHeightLimit {
			return ErrInvalidFrame
		}
		if tileHeightLimit > 1 {
			if err := writeUniform(w, uint32(tileHeightLimit), uint32(height-1)); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeUniform(w *bitWriter, max uint32, v uint32) error {
	if max <= 1 {
		if v != 0 {
			return ErrInvalidFrame
		}
		return nil
	}
	if v >= max {
		return ErrInvalidFrame
	}
	l := uint8(0)
	for (uint32(1) << (l + 1)) <= max {
		l++
	}
	l++
	m := (uint32(1) << l) - max
	if v < m {
		return w.writeBits(uint64(v), l-1)
	}
	value := v + m
	if err := w.writeBits(uint64(value>>1), l-1); err != nil {
		return err
	}
	return w.writeBit(uint8(value & 1))
}

func codedWidthFromIntraFrameSize(size IntraFrameSize) uint32 {
	if size.SuperResDenominator != 8 {
		return superResCodedWidthEncoder(size.UpscaledWidth, size.SuperResDenominator)
	}
	return size.UpscaledWidth
}

func codedWidthFromInterFrameSize(size InterFrameSize) uint32 {
	if size.SuperResDenominator != 8 {
		return superResCodedWidthEncoder(size.UpscaledWidth, size.SuperResDenominator)
	}
	return size.UpscaledWidth
}

func superResCodedWidthEncoder(upscaledWidth uint32, denominator uint8) uint32 {
	w := (upscaledWidth*8 + uint32(denominator>>1)) / uint32(denominator)
	minWidth := min32(upscaledWidth, 16)
	if w < minWidth {
		return minWidth
	}
	return w
}

func ceilShift32Encoder(v uint32, bits uint8) uint32 {
	return (v + (uint32(1) << bits) - 1) >> bits
}

func tileLog2Encoder(sz int, target int) int {
	k := 0
	for (sz << k) < target {
		k++
	}
	return k
}

func widestTileSB(tiles TileInfo) int {
	widest := 0
	for i := uint8(0); i < tiles.Cols; i++ {
		width := int(tiles.ColStartSB[i+1] - tiles.ColStartSB[i])
		if width > widest {
			widest = width
		}
	}
	return widest
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func min32(a uint32, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}
