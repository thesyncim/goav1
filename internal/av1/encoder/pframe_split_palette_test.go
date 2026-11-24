package encoder

import (
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// TestRecordPaletteCoversPaletteModeResult pins recordPalette to every value
// field of tile.PaletteModeResult, so a field added there cannot be silently
// dropped by the pointer-free split record.
func TestRecordPaletteCoversPaletteModeResult(t *testing.T) {
	want := map[string]bool{"YSize": true, "UVSize": true, "YColors": true, "UColors": true, "VColors": true, "YMapSlot": true, "UVMapSlot": true}
	typ := reflect.TypeFor[tile.PaletteModeResult]()
	for i := range typ.NumField() {
		if name := typ.Field(i).Name; !want[name] {
			t.Errorf("tile.PaletteModeResult gained field %s; carry it in recordPalette", name)
		}
	}
	if typ.NumField() != len(want) {
		t.Errorf("tile.PaletteModeResult has %d fields, recordPalette covers %d", typ.NumField(), len(want))
	}
	var p tile.PaletteModeResult
	p.YSize, p.UVSize = 5, 3
	for i := range p.YColors {
		p.YColors[i] = uint16(10 + i)
		p.UColors[i] = uint16(20 + i)
		p.VColors[i] = uint16(30 + i)
	}
	rec := recordPaletteOf(p)
	if got := rec.result(); got != p {
		t.Fatalf("round trip %+v, want %+v", got, p)
	}
}
