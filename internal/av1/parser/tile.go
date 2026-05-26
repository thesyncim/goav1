// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	MaxTileRows = 64
	MaxTileCols = 64

	maxTileWidthPixels = 4096
	maxTileAreaPixels  = 4096 * 2304
)

type InterpolationFilter uint8

const (
	InterpolationEightTap InterpolationFilter = iota
	InterpolationSmooth
	InterpolationSharp
	InterpolationBilinear
	InterpolationSwitchable
)

// TileInfo is the tile layout and the immediately preceding post-size frame
// controls from uncompressed_header().
type TileInfo struct {
	AllowHighPrecisionMV bool
	InterpolationFilter  InterpolationFilter
	SwitchableMotionMode bool
	UseRefFrameMVS       bool
	RefreshContext       bool
	ContextUpdateTileID  uint16
	TileSizeBytes        uint8
	UniformSpacing       bool
	SBCols               uint16
	SBRows               uint16
	MinLog2Cols          uint8
	MaxLog2Cols          uint8
	MinLog2Rows          uint8
	MaxLog2Rows          uint8
	MinLog2Tiles         uint8
	Log2Cols             uint8
	Log2Rows             uint8
	Cols                 uint8
	Rows                 uint8
	ColStartSB           [MaxTileCols + 1]uint16
	RowStartSB           [MaxTileRows + 1]uint16
	BitsRead             int
}

// ParseTileInfo parses post-size inter controls, refresh-context state, and
// tile_info() from a frame header payload.
func ParseTileInfo(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size FrameSize) (TileInfo, error) {
	if prefix.ShowExistingFrame {
		return TileInfo{}, ErrInvalidFrameHeader
	}
	if size.CodedWidth == 0 || size.Height == 0 {
		return TileInfo{}, ErrInvalidFrameHeader
	}

	r := bitstream.NewReader(payload)
	if err := r.SkipBits(size.BitsRead); err != nil {
		return TileInfo{}, err
	}

	var tiles TileInfo
	if err := parsePostSizeFrameControls(&r, seq, prefix, &tiles); err != nil {
		return TileInfo{}, err
	}
	if !seq.ReducedStillPictureHeader && !prefix.DisableCDFUpdate {
		disableRefresh, err := r.ReadBool()
		if err != nil {
			return TileInfo{}, err
		}
		tiles.RefreshContext = !disableRefresh
	}
	if err := parseTileLayout(&r, seq, size, &tiles); err != nil {
		return TileInfo{}, err
	}
	tiles.BitsRead = r.BitsRead()
	return tiles, nil
}

func parsePostSizeFrameControls(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, tiles *TileInfo) error {
	if !frameTypeIsInterOrSwitch(prefix.FrameType) {
		return nil
	}
	var err error
	if !prefix.ForceIntegerMV {
		if tiles.AllowHighPrecisionMV, err = r.ReadBool(); err != nil {
			return err
		}
	}
	switchable, err := r.ReadBool()
	if err != nil {
		return err
	}
	if switchable {
		tiles.InterpolationFilter = InterpolationSwitchable
	} else {
		v, err := r.ReadBits(2)
		if err != nil {
			return err
		}
		tiles.InterpolationFilter = InterpolationFilter(v)
	}
	if tiles.SwitchableMotionMode, err = r.ReadBool(); err != nil {
		return err
	}
	if !prefix.ErrorResilientMode && seq.EnableRefFrameMVS && seq.EnableOrderHint {
		tiles.UseRefFrameMVS, err = r.ReadBool()
	}
	return err
}

func parseTileLayout(r *bitstream.Reader, seq SequenceHeader, size FrameSize, tiles *TileInfo) error {
	sbSizeLog2 := uint8(6)
	if seq.Use128x128Superblock {
		sbSizeLog2 = 7
	}
	sbCols := ceilShift32(size.CodedWidth, sbSizeLog2)
	sbRows := ceilShift32(size.Height, sbSizeLog2)
	if sbCols == 0 || sbRows == 0 || sbCols > 65535 || sbRows > 65535 {
		return ErrInvalidFrameHeader
	}

	maxTileWidthSB := maxTileWidthPixels >> sbSizeLog2
	maxTileAreaSB := maxTileAreaPixels >> (2 * sbSizeLog2)
	tiles.SBCols = uint16(sbCols)
	tiles.SBRows = uint16(sbRows)
	tiles.MinLog2Cols = uint8(tileLog2(maxTileWidthSB, int(sbCols)))
	tiles.MaxLog2Cols = uint8(tileLog2(1, minInt(int(sbCols), MaxTileCols)))
	tiles.MaxLog2Rows = uint8(tileLog2(1, minInt(int(sbRows), MaxTileRows)))
	tiles.MinLog2Tiles = uint8(maxInt(tileLog2(maxTileAreaSB, int(sbCols*sbRows)), int(tiles.MinLog2Cols)))

	uniform, err := r.ReadBool()
	if err != nil {
		return err
	}
	tiles.UniformSpacing = uniform
	if uniform {
		if err := parseUniformTileLayout(r, int(sbCols), int(sbRows), tiles); err != nil {
			return err
		}
	} else if err := parseExplicitTileLayout(r, int(sbCols), int(sbRows), maxTileWidthSB, maxTileAreaSB, tiles); err != nil {
		return err
	}

	if tiles.Cols == 0 || tiles.Rows == 0 {
		return ErrInvalidFrameHeader
	}
	if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
		v, err := r.ReadBits(tiles.Log2Cols + tiles.Log2Rows)
		if err != nil {
			return err
		}
		if v >= uint64(tiles.Cols)*uint64(tiles.Rows) {
			return ErrInvalidFrameHeader
		}
		tiles.ContextUpdateTileID = uint16(v)
		v, err = r.ReadBits(2)
		if err != nil {
			return err
		}
		tiles.TileSizeBytes = uint8(v + 1)
	}
	return nil
}

func parseUniformTileLayout(r *bitstream.Reader, sbCols int, sbRows int, tiles *TileInfo) error {
	log2Cols := tiles.MinLog2Cols
	for log2Cols < tiles.MaxLog2Cols {
		more, err := r.ReadBool()
		if err != nil {
			return err
		}
		if !more {
			break
		}
		log2Cols++
	}
	tiles.Log2Cols = log2Cols
	tileW := 1 + ((sbCols - 1) >> log2Cols)
	cols := 0
	for sbx := 0; sbx < sbCols; sbx += tileW {
		if cols >= MaxTileCols {
			return ErrInvalidFrameHeader
		}
		tiles.ColStartSB[cols] = uint16(sbx)
		cols++
	}
	tiles.Cols = uint8(cols)
	tiles.ColStartSB[cols] = uint16(sbCols)

	minRows := max(int(tiles.MinLog2Tiles)-int(tiles.Log2Cols), 0)
	tiles.MinLog2Rows = uint8(minRows)
	log2Rows := tiles.MinLog2Rows
	for log2Rows < tiles.MaxLog2Rows {
		more, err := r.ReadBool()
		if err != nil {
			return err
		}
		if !more {
			break
		}
		log2Rows++
	}
	tiles.Log2Rows = log2Rows
	tileH := 1 + ((sbRows - 1) >> log2Rows)
	rows := 0
	for sby := 0; sby < sbRows; sby += tileH {
		if rows >= MaxTileRows {
			return ErrInvalidFrameHeader
		}
		tiles.RowStartSB[rows] = uint16(sby)
		rows++
	}
	tiles.Rows = uint8(rows)
	tiles.RowStartSB[rows] = uint16(sbRows)
	return nil
}

func parseExplicitTileLayout(r *bitstream.Reader, sbCols int, sbRows int, maxTileWidthSB int, maxTileAreaSB int, tiles *TileInfo) error {
	cols := 0
	widestTile := 0
	sbx := 0
	for sbx < sbCols && cols < MaxTileCols {
		tileWidthLimit := minInt(sbCols-sbx, maxTileWidthSB)
		tileW := 1
		if tileWidthLimit > 1 {
			v, err := readUniform(r, uint32(tileWidthLimit))
			if err != nil {
				return err
			}
			tileW = int(v) + 1
		}
		tiles.ColStartSB[cols] = uint16(sbx)
		sbx += tileW
		if tileW > widestTile {
			widestTile = tileW
		}
		cols++
	}
	if sbx < sbCols {
		return ErrInvalidFrameHeader
	}
	tiles.Cols = uint8(cols)
	tiles.ColStartSB[cols] = uint16(sbCols)
	tiles.Log2Cols = uint8(tileLog2(1, cols))

	if tiles.MinLog2Tiles != 0 {
		maxTileAreaSB >>= tiles.MinLog2Tiles + 1
	}
	maxTileHeightSB := maxInt(maxTileAreaSB/widestTile, 1)
	rows := 0
	sby := 0
	for sby < sbRows && rows < MaxTileRows {
		tileHeightLimit := minInt(sbRows-sby, maxTileHeightSB)
		tileH := 1
		if tileHeightLimit > 1 {
			v, err := readUniform(r, uint32(tileHeightLimit))
			if err != nil {
				return err
			}
			tileH = int(v) + 1
		}
		tiles.RowStartSB[rows] = uint16(sby)
		sby += tileH
		rows++
	}
	if sby < sbRows {
		return ErrInvalidFrameHeader
	}
	tiles.Rows = uint8(rows)
	tiles.RowStartSB[rows] = uint16(sbRows)
	tiles.Log2Rows = uint8(tileLog2(1, rows))
	return nil
}

func readUniform(r *bitstream.Reader, max uint32) (uint32, error) {
	if max <= 1 {
		return 0, nil
	}
	l := uint8(0)
	for (uint32(1) << (l + 1)) <= max {
		l++
	}
	l++
	m := (uint32(1) << l) - max
	v, err := r.ReadBits(l - 1)
	if err != nil {
		return 0, err
	}
	if uint32(v) < m {
		return uint32(v), nil
	}
	bit, err := r.ReadBit()
	if err != nil {
		return 0, err
	}
	return (uint32(v) << 1) - m + uint32(bit), nil
}

func ceilShift32(v uint32, bits uint8) uint32 {
	return (v + (uint32(1) << bits) - 1) >> bits
}

func tileLog2(sz int, target int) int {
	k := 0
	for (sz << k) < target {
		k++
	}
	return k
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
