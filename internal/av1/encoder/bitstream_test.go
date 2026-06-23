package encoder

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

var bitWriterBenchSink byte

func TestBitWriterWriteBitsMatchesReference(t *testing.T) {
	fields := [...]struct {
		value uint64
		bits  uint8
	}{
		{value: 0x1, bits: 1},
		{value: 0x2, bits: 2},
		{value: 0x15, bits: 5},
		{value: 0x0, bits: 3},
		{value: 0xff, bits: 8},
		{value: 0x155, bits: 9},
		{value: 0xabc, bits: 12},
		{value: 0x12345678, bits: 32},
		{value: 0xffffffffffffffff, bits: 64},
	}
	var got [32]byte
	var want [32]byte
	w := newBitWriter(got[:])
	refBit := 0
	for _, field := range fields {
		if err := w.writeBits(field.value, field.bits); err != nil {
			t.Fatalf("writeBits(%x,%d): %v", field.value, field.bits, err)
		}
		writeBitsReference(want[:], &refBit, field.value, field.bits)
	}
	gotLen := w.bytesWritten()
	wantLen := (refBit + 7) >> 3
	if gotLen != wantLen || !bytes.Equal(got[:gotLen], want[:wantLen]) {
		t.Fatalf("writeBits bytes=% x/%d want % x/%d", got[:gotLen], gotLen, want[:wantLen], wantLen)
	}
}

func BenchmarkBitWriterWriteBits(b *testing.B) {
	fields := [...]struct {
		value uint64
		bits  uint8
	}{
		{value: 0x1, bits: 1},
		{value: 0x3, bits: 2},
		{value: 0x15, bits: 5},
		{value: 0xff, bits: 8},
		{value: 0x155, bits: 9},
		{value: 0xabc, bits: 12},
		{value: 0x12345678, bits: 32},
	}
	var buf [16]byte
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		w := newBitWriter(buf[:])
		for _, field := range fields {
			_ = w.writeBits(field.value, field.bits)
		}
		bitWriterBenchSink ^= buf[0]
	}
}

func TestAppendLowOverheadOBU(t *testing.T) {
	unit := OBU{
		Type:       obu.TypeFrame,
		TemporalID: 2,
		SpatialID:  1,
		Payload:    []byte{0xaa, 0xbb},
	}
	size, err := LowOverheadOBUSize(unit)
	if err != nil {
		t.Fatalf("LowOverheadOBUSize: %v", err)
	}
	if size != 5 {
		t.Fatalf("LowOverheadOBUSize = %d; want 5", size)
	}
	var buf [8]byte
	out, err := AppendLowOverheadOBU(buf[:0], unit)
	if err != nil {
		t.Fatalf("AppendLowOverheadOBU: %v", err)
	}
	want := []byte{0x36, 0x48, 0x02, 0xaa, 0xbb}
	if !bytes.Equal(out, want) {
		t.Fatalf("AppendLowOverheadOBU = % x; want % x", out, want)
	}
	parsed, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != unit.Type || parsed.Header.TemporalID != unit.TemporalID || parsed.Header.SpatialID != unit.SpatialID {
		t.Fatalf("parsed header=%+v consumed=%d", parsed.Header, consumed)
	}
	if !bytes.Equal(parsed.Payload, unit.Payload) {
		t.Fatalf("payload = % x; want % x", parsed.Payload, unit.Payload)
	}
}

func TestAppendLowOverheadOBUEmptyTemporalDelimiter(t *testing.T) {
	var buf [4]byte
	out, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		t.Fatalf("AppendLowOverheadOBU temporal delimiter: %v", err)
	}
	want := []byte{0x12, 0x00}
	if !bytes.Equal(out, want) {
		t.Fatalf("temporal delimiter = % x; want % x", out, want)
	}
}

func TestAppendLowOverheadOBURejectsInvalid(t *testing.T) {
	var buf [8]byte
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeReserved}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("reserved type err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.Type(16)}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("out-of-range type err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadOBU(buf[:0], OBU{Type: obu.TypeFrame, SpatialID: 4}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad spatial id err=%v; want ErrInvalidFrame", err)
	}
}

func TestAppendLowOverheadOBUShortBuffer(t *testing.T) {
	var buf [4]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeFrame, Payload: []byte{0xaa, 0xbb, 0xcc}})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v; want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output: % x", out)
	}
}

func TestAppendLowOverheadOBUAllocs(t *testing.T) {
	payload := [...]byte{0xaa, 0xbb, 0xcc}
	unit := OBU{Type: obu.TypeFrame, Payload: payload[:]}
	var buf [8]byte
	if _, err := AppendLowOverheadOBU(buf[:0], unit); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadOBU(buf[:0], unit)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadOBU allocated: %f", allocs)
	}
}

func TestAppendLowOverheadWebRTCScalabilityMetadataOBU(t *testing.T) {
	size, ok, err := LowOverheadWebRTCScalabilityMetadataOBUSize(ScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("LowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
	}
	var buf [8]byte
	out, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], ScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("AppendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
	}
	if len(out) != size {
		t.Fatalf("len=%d want %d", len(out), size)
	}
	unit, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != obu.TypeMetadata {
		t.Fatalf("metadata consumed=%d header=%+v", consumed, unit.Header)
	}
	meta, err := obu.ParseMetadata(unit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadata: %v", err)
	}
	if meta.Type != obu.MetadataTypeScalability || meta.Scalability.ModeIDC != obu.ScalabilityModeL2T2 || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v", meta)
	}

	prefix := []byte{0xee}
	noMetadata, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(prefix, ScalabilityModeL1T1)
	if err != nil || ok || !bytes.Equal(noMetadata, prefix) {
		t.Fatalf("L1T1 metadata out=% x ok=%v err=%v", noMetadata, ok, err)
	}
	if size, ok, err := LowOverheadWebRTCScalabilityMetadataOBUSize(ScalabilityModeL1T1); err != nil || ok || size != 0 {
		t.Fatalf("L1T1 size=%d ok=%v err=%v", size, ok, err)
	}

	var tiny [2]byte
	if out, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(tiny[:0], ScalabilityModeL2T2); !ok || !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short metadata out=% x ok=%v err=%v", out, ok, err)
	}
}

func TestAppendLowOverheadWebRTCScalabilityMetadataOBUExplicitSS(t *testing.T) {
	for _, tc := range []struct {
		mode      ScalabilityMode
		spatial   uint8
		temporal  uint8
		refIDs    []uint8
		groupTIDs []uint8
		groupDiff []uint8
	}{
		{
			mode:      ScalabilityModeL2T1_KEY,
			spatial:   2,
			temporal:  1,
			refIDs:    []uint8{0xff, 0},
			groupTIDs: []uint8{0},
			groupDiff: []uint8{1},
		},
		{
			mode:      ScalabilityModeL3T1h,
			spatial:   3,
			temporal:  1,
			refIDs:    []uint8{0xff, 0, 1},
			groupTIDs: []uint8{0},
			groupDiff: []uint8{1},
		},
		{
			mode:      ScalabilityModeS3T3h,
			spatial:   3,
			temporal:  3,
			refIDs:    []uint8{0xff, 0xff, 0xff},
			groupTIDs: []uint8{0, 2, 1, 2},
			groupDiff: []uint8{4, 1, 2, 1},
		},
	} {
		t.Run(tc.mode.String(), func(t *testing.T) {
			size, ok, err := LowOverheadWebRTCScalabilityMetadataOBUSize(tc.mode)
			if err != nil || !ok {
				t.Fatalf("LowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
			}
			var buf [64]byte
			out, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], tc.mode)
			if err != nil || !ok {
				t.Fatalf("AppendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
			}
			if len(out) != size {
				t.Fatalf("len=%d want %d", len(out), size)
			}
			unit, consumed, err := obu.ParseLowOverhead(out)
			if err != nil {
				t.Fatalf("ParseLowOverhead: %v", err)
			}
			if consumed != len(out) || unit.Header.Type != obu.TypeMetadata {
				t.Fatalf("metadata consumed=%d header=%+v", consumed, unit.Header)
			}
			meta, err := obu.ParseMetadata(unit.Payload)
			if err != nil {
				t.Fatalf("ParseMetadata: %v", err)
			}
			assertWebRTCScalabilityStructure(t, meta, tc.mode, tc.spatial, tc.temporal, false, tc.refIDs, tc.groupTIDs, tc.groupDiff)
		})
	}
}

func TestAppendLowOverheadWebRTCScalabilityMetadataOBUAllWebRTCModes(t *testing.T) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		t.Run(mode.String(), func(t *testing.T) {
			size, ok, err := LowOverheadWebRTCScalabilityMetadataOBUSize(mode)
			if err != nil {
				t.Fatalf("LowOverheadWebRTCScalabilityMetadataOBUSize: %v", err)
			}
			if mode == ScalabilityModeL1T1 {
				if ok || size != 0 {
					t.Fatalf("L1T1 metadata size=%d ok=%v", size, ok)
				}
				return
			}
			if !ok || size == 0 {
				t.Fatalf("%s metadata size=%d ok=%v", mode, size, ok)
			}
			var buf [96]byte
			out, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], mode)
			if err != nil || !ok {
				t.Fatalf("AppendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
			}
			unit, consumed, err := obu.ParseLowOverhead(out)
			if err != nil || consumed != len(out) || unit.Header.Type != obu.TypeMetadata {
				t.Fatalf("ParseLowOverhead consumed=%d len=%d header=%+v err=%v", consumed, len(out), unit.Header, err)
			}
			meta, err := obu.ParseMetadata(unit.Payload)
			if err != nil {
				t.Fatalf("ParseMetadata: %v", err)
			}
			if idc, hasIDC := WebRTCScalabilityModeIDC(mode); hasIDC {
				if meta.Type != obu.MetadataTypeScalability || meta.Scalability.ModeIDC != idc || meta.Scalability.HasStructure {
					t.Fatalf("predefined metadata=%+v idc=%d", meta, idc)
				}
				return
			}
			spatial, temporal, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid mode %s", mode)
			}
			refIDs := make([]uint8, spatial)
			for i := range refIDs {
				refIDs[i] = webRTCScalabilitySpatialLayerRefID(mode, uint8(i))
			}
			groupSize, ok := webRTCScalabilityTemporalGroupSize(temporal)
			if !ok {
				t.Fatalf("temporal group for %s", mode)
			}
			groupTIDs := make([]uint8, groupSize)
			groupDiffs := make([]uint8, groupSize)
			for i := uint8(0); i < groupSize; i++ {
				entry := webRTCScalabilityTemporalGroupEntry(temporal, i)
				groupTIDs[i] = entry.temporalID
				groupDiffs[i] = entry.refPicDiff
			}
			assertWebRTCScalabilityStructure(t, meta, mode, spatial, temporal, false, refIDs, groupTIDs, groupDiffs)
		})
	}
}

func TestAppendWebRTCScalabilityMetadataPayloadAllocs(t *testing.T) {
	var payload [4]byte
	if _, ok, err := AppendWebRTCScalabilityMetadataPayload(payload[:0], ScalabilityModeL2T2); err != nil || !ok {
		t.Fatalf("preflight payload ok=%v err=%v", ok, err)
	}
	var obuBuf [8]byte
	if _, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(obuBuf[:0], ScalabilityModeL2T2); err != nil || !ok {
		t.Fatalf("preflight obu ok=%v err=%v", ok, err)
	}
	var ssPayload [32]byte
	if _, ok, err := AppendWebRTCScalabilityMetadataPayload(ssPayload[:0], ScalabilityModeS3T3h); err != nil || !ok {
		t.Fatalf("preflight ss payload ok=%v err=%v", ok, err)
	}
	var ssOBU [40]byte
	if _, ok, err := AppendLowOverheadWebRTCScalabilityMetadataOBU(ssOBU[:0], ScalabilityModeS3T3h); err != nil || !ok {
		t.Fatalf("preflight ss obu ok=%v err=%v", ok, err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = AppendWebRTCScalabilityMetadataPayload(payload[:0], ScalabilityModeL2T2)
		_, _, _ = AppendLowOverheadWebRTCScalabilityMetadataOBU(obuBuf[:0], ScalabilityModeL2T2)
		_, _, _ = AppendWebRTCScalabilityMetadataPayload(ssPayload[:0], ScalabilityModeS3T3h)
		_, _, _ = AppendLowOverheadWebRTCScalabilityMetadataOBU(ssOBU[:0], ScalabilityModeS3T3h)
	})
	if allocs != 0 {
		t.Fatalf("metadata writers allocated: %f", allocs)
	}
}

func TestAppendLowOverheadTemporalUnit(t *testing.T) {
	units := [...]OBU{
		{Type: obu.TypeSequenceHeader, Payload: []byte{0x01, 0x02}},
		{Type: obu.TypeFrame, TemporalID: 1, SpatialID: 1, Payload: []byte{0xaa}},
	}
	size, err := LowOverheadTemporalUnitSize(units[:])
	if err != nil {
		t.Fatalf("LowOverheadTemporalUnitSize: %v", err)
	}
	var buf [16]byte
	out, err := AppendLowOverheadTemporalUnit(buf[:0], units[:])
	if err != nil {
		t.Fatalf("AppendLowOverheadTemporalUnit: %v", err)
	}
	if len(out) != size {
		t.Fatalf("temporal unit len=%d want %d", len(out), size)
	}
	want := []byte{
		0x12, 0x00,
		0x0a, 0x02, 0x01, 0x02,
		0x36, 0x28, 0x01, 0xaa,
	}
	if !bytes.Equal(out, want) {
		t.Fatalf("temporal unit = % x; want % x", out, want)
	}
	it := obu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil {
		t.Fatalf("TemporalUnitIterator.Next: %v", err)
	}
	if !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit parsed ok=%v raw=% x", ok, tu.Raw)
	}
	_, ok, err = it.Next()
	if err != nil || ok {
		t.Fatalf("TemporalUnitIterator second ok=%v err=%v", ok, err)
	}
}

func TestAppendLowOverheadTemporalUnitRejectsBeforeWrite(t *testing.T) {
	var buf [16]byte
	dst := buf[:1]
	dst[0] = 0xee
	if _, err := AppendLowOverheadTemporalUnit(dst, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("empty temporal unit err=%v; want ErrInvalidFrame", err)
	}
	if _, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeTemporalDelimiter}}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("nested temporal delimiter err=%v; want ErrInvalidFrame", err)
	}
	out, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeFrame}, {Type: obu.TypeReserved}})
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad temporal unit err=%v; want ErrInvalidFrame", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("invalid temporal unit mutated output: % x", out)
	}
}

func TestAppendLowOverheadTemporalUnitShortBuffer(t *testing.T) {
	var buf [5]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendLowOverheadTemporalUnit(dst, []OBU{{Type: obu.TypeFrame, Payload: []byte{0xaa, 0xbb}}})
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short temporal unit err=%v; want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short temporal unit mutated output: % x", out)
	}
}

func TestAppendLowOverheadTemporalUnitAllocs(t *testing.T) {
	payload := [...]byte{0xaa, 0xbb}
	units := [...]OBU{{Type: obu.TypeFrame, Payload: payload[:]}}
	var buf [8]byte
	if _, err := AppendLowOverheadTemporalUnit(buf[:0], units[:]); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadTemporalUnit(buf[:0], units[:])
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadTemporalUnit allocated: %f", allocs)
	}
}

func assertWebRTCScalabilityStructure(t *testing.T, meta obu.Metadata, mode ScalabilityMode, spatial uint8, temporal uint8, wantDimensions bool, refIDs []uint8, groupTIDs []uint8, groupDiffs []uint8) {
	t.Helper()
	if meta.Type != obu.MetadataTypeScalability || meta.Scalability.ModeIDC != obu.ScalabilityModeSS || !meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v want SS structure for %s", meta, mode)
	}
	structure := meta.Scalability.Structure
	if structure.SpatialLayersCountMinus1 != spatial-1 ||
		structure.SpatialLayerDimensionsPresent != wantDimensions ||
		!structure.SpatialLayerDescriptionPresent ||
		!structure.TemporalGroupDescriptionPresent {
		t.Fatalf("structure flags=%+v mode=%s spatial=%d dimensions=%v", structure, mode, spatial, wantDimensions)
	}
	for i, want := range refIDs {
		if structure.SpatialLayerRefID[i] != want {
			t.Fatalf("refID[%d]=%d want %d structure=%+v", i, structure.SpatialLayerRefID[i], want, structure)
		}
	}
	if structure.TemporalGroupSize != uint8(len(groupTIDs)) || len(groupTIDs) != len(groupDiffs) {
		t.Fatalf("temporal group size=%d tids=%d diffs=%d", structure.TemporalGroupSize, len(groupTIDs), len(groupDiffs))
	}
	for i, wantTID := range groupTIDs {
		entry := structure.TemporalGroup[i]
		if entry.TemporalID != wantTID || entry.RefCount != 1 || entry.RefPicDiff[0] != groupDiffs[i] {
			t.Fatalf("temporal group[%d]=%+v want tid=%d diff=%d", i, entry, wantTID, groupDiffs[i])
		}
		wantTemporalSwitchingUp := true
		if len(groupTIDs) == 4 && i == 2 {
			wantTemporalSwitchingUp = false
		}
		if entry.TemporalSwitchingUp != wantTemporalSwitchingUp || entry.SpatialSwitchingUp {
			t.Fatalf("temporal group[%d] switching=%+v want temporal=%v spatial=false", i, entry, wantTemporalSwitchingUp)
		}
		if i == 0 && entry.TemporalID != 0 {
			t.Fatalf("first temporal group id=%d want 0", entry.TemporalID)
		}
		if temporal == 1 && entry.TemporalID != 0 {
			t.Fatalf("single temporal layer entry=%+v", entry)
		}
	}
}

func writeBitsReference(dst []byte, bit *int, value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		byteIndex := *bit >> 3
		shift := uint(7 - (*bit & 7))
		if shift == 7 {
			dst[byteIndex] = 0
		}
		if (value>>uint(i))&1 != 0 {
			dst[byteIndex] |= 1 << shift
		} else {
			dst[byteIndex] &^= 1 << shift
		}
		(*bit)++
	}
}
