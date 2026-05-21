package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

func TestDefaultRestorationReferencesAndCDFs(t *testing.T) {
	refs := DefaultRestorationReferences()
	if refs.Wiener != av1restoration.DefaultWienerInfo() {
		t.Fatalf("Wiener ref=%+v", refs.Wiener)
	}
	if refs.SGRProj.ParamsIndex != 0 || refs.SGRProj.XQD != [2]int{-32, 31} {
		t.Fatalf("SGR ref=%+v", refs.SGRProj)
	}

	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}
	if err := InitDefaultRestorationCDFs(cdfs); err != nil {
		t.Fatal(err)
	}
	assertEntropyCDFValues(t, switchable.Values(), []uint16{23355, 10187, 0, 0})
	assertEntropyCDFValues(t, wiener.Values(), []uint16{21198, 0, 0})
	assertEntropyCDFValues(t, sgr.Values(), []uint16{15913, 0, 0})
}

func TestReadRestorationUnitNoneSkipsTileBits(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	before := state.Reader.BitsRead()
	unit, err := state.ReadRestorationUnit(parser.RestorationNone, 0, nil, RestorationCDFs{})
	if err != nil {
		t.Fatal(err)
	}
	if unit.Type != parser.RestorationNone {
		t.Fatalf("unit=%+v", unit)
	}
	if after := state.Reader.BitsRead(); after != before {
		t.Fatalf("BitsRead=%d want %d", after, before)
	}
}

func TestReadRestorationUnitWienerDisabled(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()
	wantRefs := refs

	unit, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, cdfs)
	if err != nil {
		t.Fatal(err)
	}
	if unit.Type != parser.RestorationNone {
		t.Fatalf("unit=%+v", unit)
	}
	if refs != wantRefs {
		t.Fatalf("refs changed %+v want %+v", refs, wantRefs)
	}
	assertEntropyCDFValues(t, cdfs.Wiener.Values(), []uint16{19874, 0, 1})
}

func TestReadRestorationUnitWienerEnabled(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff, 0xff, 0xff, 0xff}, Job{Offset: 0, Size: 4}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()

	unit, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, cdfs)
	if err != nil {
		t.Fatal(err)
	}
	if unit.Type != parser.RestorationWiener {
		t.Fatalf("unit=%+v", unit)
	}
	if !validDecodedWienerInfo(unit.Wiener) {
		t.Fatalf("invalid Wiener=%+v", unit.Wiener)
	}
	if refs.Wiener != unit.Wiener {
		t.Fatalf("refs=%+v want %+v", refs.Wiener, unit.Wiener)
	}
	assertEntropyCDFValues(t, cdfs.Wiener.Values(), []uint16{21921, 0, 1})
}

func TestReadWienerFilterMatchesLibaomDefaults(t *testing.T) {
	refs := DefaultRestorationReferences()
	reader := entropy.NewReader([]byte{0x00, 0x00, 0x00, 0x00})
	info, err := readWienerFilter(&reader, false, &refs)
	if err != nil {
		t.Fatal(err)
	}
	if info != av1restoration.DefaultWienerInfo() {
		t.Fatalf("luma Wiener=%+v", info)
	}
	if refs.Wiener != info {
		t.Fatalf("refs=%+v want %+v", refs.Wiener, info)
	}

	refs = DefaultRestorationReferences()
	reader = entropy.NewReader([]byte{0x00, 0x00, 0x00, 0x00})
	info, err = readWienerFilter(&reader, true, &refs)
	if err != nil {
		t.Fatal(err)
	}
	wantChroma := av1restoration.WienerInfo{
		VFilter: av1restoration.NewWienerFilter(0, -7, 15),
		HFilter: av1restoration.NewWienerFilter(0, -7, 15),
	}
	if info != wantChroma {
		t.Fatalf("chroma Wiener=%+v want %+v", info, wantChroma)
	}
}

func TestReadSGRProjFilterMatchesLibaomDefaults(t *testing.T) {
	refs := DefaultRestorationReferences()
	reader := entropy.NewReader([]byte{0x00, 0x00, 0x00, 0x00})
	info, err := readSGRProjFilter(&reader, &refs)
	if err != nil {
		t.Fatal(err)
	}
	if info.ParamsIndex != 0 || info.XQD != [2]int{-32, 31} {
		t.Fatalf("SGR=%+v", info)
	}
	if refs.SGRProj != info {
		t.Fatalf("refs=%+v want %+v", refs.SGRProj, info)
	}
}

func TestReadRestorationUnitRejectsInvalidInputs(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()

	var nilState *DecodeState
	if _, err := nilState.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, cdfs); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v", err)
	}
	if _, err := state.ReadRestorationUnit(parser.RestorationType(9), 0, &refs, cdfs); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad frame type err=%v", err)
	}
	if _, err := state.ReadRestorationUnit(parser.RestorationWiener, 3, &refs, cdfs); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad plane err=%v", err)
	}
	badRefs := refs
	badRefs.Wiener.HFilter[7] = 1
	if _, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &badRefs, cdfs); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad refs err=%v", err)
	}
	if _, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, RestorationCDFs{}); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad cdf err=%v", err)
	}
	if err := InitDefaultRestorationCDFs(RestorationCDFs{}); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad init cdfs err=%v", err)
	}
}

func TestReadRestorationUnitAllocs(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	var state DecodeState
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if err := InitDefaultRestorationCDFs(cdfs); err != nil {
			t.Fatal(err)
		}
		refs := DefaultRestorationReferences()
		if _, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, cdfs); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ReadRestorationUnit allocated: %f", allocs)
	}
}

func FuzzDecodeStateRestorationUnit(f *testing.F) {
	f.Add([]byte{0x00}, uint8(0), uint8(0))
	f.Add([]byte{0xff, 0xff, 0xff, 0xff}, uint8(2), uint8(0))
	f.Add([]byte{0xa5, 0x5a, 0xff, 0x00}, uint8(3), uint8(2))
	f.Fuzz(func(t *testing.T, payload []byte, rawType uint8, rawPlane uint8) {
		if len(payload) > 64 {
			return
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		cdfs := initRestorationCDFs(t)
		refs := DefaultRestorationReferences()
		frameTypes := [...]parser.RestorationType{
			parser.RestorationNone,
			parser.RestorationSwitchable,
			parser.RestorationWiener,
			parser.RestorationSGRProj,
		}
		frameType := frameTypes[rawType%uint8(len(frameTypes))]
		unit, err := state.ReadRestorationUnit(frameType, int(rawPlane%3), &refs, cdfs)
		if err != nil {
			t.Fatalf("ReadRestorationUnit err=%v frameType=%d payloadLen=%d", err, frameType, len(payload))
		}
		switch unit.Type {
		case parser.RestorationNone:
		case parser.RestorationWiener:
			if !validDecodedWienerInfo(unit.Wiener) {
				t.Fatalf("invalid Wiener=%+v", unit.Wiener)
			}
		case parser.RestorationSGRProj:
			if unit.SGRProj.ParamsIndex >= av1restoration.SGRProjParams ||
				unit.SGRProj.XQD[0] < av1restoration.SGRProjPrjMin0 ||
				unit.SGRProj.XQD[0] > av1restoration.SGRProjPrjMax0 ||
				unit.SGRProj.XQD[1] < av1restoration.SGRProjPrjMin1 ||
				unit.SGRProj.XQD[1] > av1restoration.SGRProjPrjMax1 {
				t.Fatalf("invalid SGR=%+v", unit.SGRProj)
			}
		default:
			t.Fatalf("bad unit type=%d", unit.Type)
		}
	})
}

func BenchmarkReadRestorationUnit(b *testing.B) {
	payload := []byte{0x00, 0x00, 0x00, 0x00}
	var state DecodeState
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		if err := InitDefaultRestorationCDFs(cdfs); err != nil {
			b.Fatal(err)
		}
		refs := DefaultRestorationReferences()
		if _, err := state.ReadRestorationUnit(parser.RestorationWiener, 0, &refs, cdfs); err != nil {
			b.Fatal(err)
		}
	}
}

func initRestorationCDFs(tb testing.TB) RestorationCDFs {
	tb.Helper()
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}
	if err := InitDefaultRestorationCDFs(cdfs); err != nil {
		tb.Fatal(err)
	}
	return cdfs
}
