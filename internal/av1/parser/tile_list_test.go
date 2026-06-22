// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package parser

import (
	"bytes"
	"errors"
	"testing"
)

func appendTileListEntryRaw(dst []byte, anchorIdx, row, col uint8, dataSizeMinus1 uint16, data []byte) []byte {
	dst = append(dst, anchorIdx, row, col, byte(dataSizeMinus1>>8), byte(dataSizeMinus1))
	dst = append(dst, data...)
	return dst
}

func TestParseTileListOBU(t *testing.T) {
	tile0 := []byte{0xde, 0xad, 0xbe, 0xef}
	tile1 := []byte{0x01, 0x02}
	payload := []byte{
		0x01,       // output_frame_width_in_tiles_minus_1 = 1
		0x02,       // output_frame_height_in_tiles_minus_1 = 2
		0x00, 0x01, // tile_count_minus_1 = 1
	}
	payload = appendTileListEntryRaw(payload, 0x00, 0x01, 0x01, uint16(len(tile0)-1), tile0)
	payload = appendTileListEntryRaw(payload, 0x05, 0x02, 0x00, uint16(len(tile1)-1), tile1)

	var scratch [4]TileListEntry
	list, err := ParseTileListOBU(payload, scratch[:0])
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if got := list.OutputFrameWidthInTiles(); got != 2 {
		t.Fatalf("width=%d want 2", got)
	}
	if got := list.OutputFrameHeightInTiles(); got != 3 {
		t.Fatalf("height=%d want 3", got)
	}
	if got := list.TileCount(); got != 2 {
		t.Fatalf("count=%d want 2", got)
	}
	if len(list.Entries) != 2 {
		t.Fatalf("entries=%d want 2", len(list.Entries))
	}
	if list.Entries[0].AnchorFrameIdx != 0 || list.Entries[0].AnchorTileRow != 1 ||
		list.Entries[0].AnchorTileCol != 1 || list.Entries[0].TileDataSize() != len(tile0) {
		t.Fatalf("entry0=%+v", list.Entries[0])
	}
	if !bytes.Equal(list.Entries[0].TileData, tile0) {
		t.Fatalf("entry0 data=%x want %x", list.Entries[0].TileData, tile0)
	}
	if list.Entries[1].AnchorFrameIdx != 5 || list.Entries[1].AnchorTileRow != 2 ||
		list.Entries[1].AnchorTileCol != 0 || list.Entries[1].TileDataSize() != len(tile1) {
		t.Fatalf("entry1=%+v", list.Entries[1])
	}
	if !bytes.Equal(list.Entries[1].TileData, tile1) {
		t.Fatalf("entry1 data=%x want %x", list.Entries[1].TileData, tile1)
	}
	// Entries should alias the caller scratch.
	if &scratch[0] != &list.Entries[0] {
		t.Fatalf("Entries did not alias caller scratch")
	}
	// TileData should alias payload.
	if &list.Entries[0].TileData[0] != &payload[TileListHeaderBytes+TileListEntryHeaderBytes] {
		t.Fatalf("TileData did not alias payload")
	}
}

func TestParseTileListOBURoundTrip(t *testing.T) {
	tile := []byte{0xaa, 0xbb}
	list := TileList{
		OutputFrameWidthInTilesMinus1:  3,
		OutputFrameHeightInTilesMinus1: 4,
		TileCountMinus1:                0,
		Entries: []TileListEntry{
			{
				AnchorFrameIdx:     7,
				AnchorTileRow:      2,
				AnchorTileCol:      1,
				TileDataSizeMinus1: uint16(len(tile) - 1),
				TileData:           tile,
			},
		},
	}
	buf := AppendTileListOBU(nil, list)
	var scratch [1]TileListEntry
	got, err := ParseTileListOBU(buf, scratch[:0])
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if got.OutputFrameWidthInTilesMinus1 != list.OutputFrameWidthInTilesMinus1 ||
		got.OutputFrameHeightInTilesMinus1 != list.OutputFrameHeightInTilesMinus1 ||
		got.TileCountMinus1 != list.TileCountMinus1 {
		t.Fatalf("header mismatch: %+v", got)
	}
	if len(got.Entries) != 1 {
		t.Fatalf("entries=%d want 1", len(got.Entries))
	}
	e := got.Entries[0]
	if e.AnchorFrameIdx != 7 || e.AnchorTileRow != 2 || e.AnchorTileCol != 1 ||
		e.TileDataSizeMinus1 != uint16(len(tile)-1) || !bytes.Equal(e.TileData, tile) {
		t.Fatalf("entry mismatch: %+v", e)
	}
}

func TestParseTileListOBUErrors(t *testing.T) {
	tile := []byte{0xaa}
	good := []byte{0x00, 0x00, 0x00, 0x00}
	good = appendTileListEntryRaw(good, 0, 0, 0, 0, tile)

	for _, tc := range []struct {
		name    string
		payload []byte
		want    error
	}{
		{name: "short header", payload: []byte{0x00, 0x00, 0x00}, want: ErrTileListShortHeader},
		{name: "tile_count exceeds output", payload: []byte{0x00, 0x00, 0x00, 0x05, 0, 0, 0, 0, 0xaa, 0, 0, 0, 0, 0xbb, 0, 0, 0, 0, 0xcc, 0, 0, 0, 0, 0xdd, 0, 0, 0, 0, 0xee, 0, 0, 0, 0, 0xff}, want: ErrTileListTooManyTiles},
		{name: "short entry header", payload: []byte{0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03}, want: ErrTileListShortEntry},
		{name: "short tile data", payload: []byte{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0x00, 0x04, 0xaa}, want: ErrTileListShortTileData},
		{name: "trailing bytes", payload: append(append([]byte{}, good...), 0xff), want: ErrTileListTrailingBytes},
		{name: "invalid anchor idx", payload: []byte{0x00, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xaa}, want: ErrTileListInvalidAnchorIndex},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseTileListOBU(tc.payload, nil)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err=%v want %v", err, tc.want)
			}
		})
	}
}

func TestValidateTileListAnchors(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	payload = appendTileListEntryRaw(payload, 0, 2, 1, 0, []byte{0xaa})

	list, err := ParseTileListOBU(payload, nil)
	if err != nil {
		t.Fatalf("ParseTileListOBU err=%v", err)
	}
	if entry := list.Entries[0]; entry.AnchorTileRow != 2 || entry.AnchorTileCol != 1 {
		t.Fatalf("anchor tile=(%d,%d) want (1,2)", entry.AnchorTileCol, entry.AnchorTileRow)
	}

	if err := ValidateTileListAnchors(list, TileInfo{Cols: 2, Rows: 3}); err != nil {
		t.Fatalf("ValidateTileListAnchors err=%v", err)
	}
	if err := ValidateTileListAnchors(list, TileInfo{Cols: 1, Rows: 3}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("small column grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}
	if err := ValidateTileListAnchors(list, TileInfo{Cols: 2, Rows: 2}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("small row grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}
	if err := ValidateTileListAnchors(list, TileInfo{}); !errors.Is(err, ErrTileListInvalidAnchorTile) {
		t.Fatalf("empty grid err=%v want %v", err, ErrTileListInvalidAnchorTile)
	}

	mismatched := list
	mismatched.TileCountMinus1 = 1
	if err := ValidateTileListAnchors(mismatched, TileInfo{Cols: 2, Rows: 3}); !errors.Is(err, ErrTileListInvalidTileCount) {
		t.Fatalf("mismatched count err=%v want %v", err, ErrTileListInvalidTileCount)
	}
}

func TestParseTileListOBUAllocs(t *testing.T) {
	tile := []byte{0xaa, 0xbb}
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	payload = appendTileListEntryRaw(payload, 0, 0, 0, uint16(len(tile)-1), tile)
	scratch := make([]TileListEntry, 4)

	allocs := testing.AllocsPerRun(1000, func() {
		list, err := ParseTileListOBU(payload, scratch[:0])
		if err != nil || len(list.Entries) != 1 {
			t.Fatalf("list=%+v err=%v", list, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ParseTileListOBU allocated: %f", allocs)
	}
}

func FuzzParseTileListOBU(f *testing.F) {
	tile0 := []byte{0xde, 0xad, 0xbe, 0xef}
	tile1 := []byte{0x01, 0x02}
	good := []byte{0x01, 0x02, 0x00, 0x01}
	good = appendTileListEntryRaw(good, 0, 1, 1, uint16(len(tile0)-1), tile0)
	good = appendTileListEntryRaw(good, 5, 2, 0, uint16(len(tile1)-1), tile1)

	for _, seed := range [][]byte{
		nil,
		{0x00},
		{0x00, 0x00, 0x00},
		{0x00, 0x00, 0x00, 0x00},
		{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0xaa},
		{0x00, 0x00, 0x00, 0x00, 0, 2, 1, 0, 0, 0xaa},
		{0x00, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xaa},
		good,
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		var scratch [TileListMaxTiles]TileListEntry
		list, err := ParseTileListOBU(payload, scratch[:0])
		if err != nil {
			return
		}
		if list.TileCount() != len(list.Entries) {
			t.Fatalf("tile count=%d entries=%d", list.TileCount(), len(list.Entries))
		}
		if list.TileCount() <= 0 || list.TileCount() > TileListMaxTiles {
			t.Fatalf("invalid tile count=%d", list.TileCount())
		}
		if list.TileCount() > list.OutputFrameWidthInTiles()*list.OutputFrameHeightInTiles() {
			t.Fatalf("tile count=%d exceeds output %dx%d", list.TileCount(), list.OutputFrameWidthInTiles(), list.OutputFrameHeightInTiles())
		}
		for i, entry := range list.Entries {
			if entry.AnchorFrameIdx >= TileListMaxExternalReferences {
				t.Fatalf("entry %d anchor_frame_idx=%d", i, entry.AnchorFrameIdx)
			}
			if len(entry.TileData) != entry.TileDataSize() {
				t.Fatalf("entry %d data len=%d want %d", i, len(entry.TileData), entry.TileDataSize())
			}
		}
		if got := AppendTileListOBU(nil, list); !bytes.Equal(got, payload) {
			t.Fatalf("round trip mismatch\n got=%x\nwant=%x", got, payload)
		}
	})
}
