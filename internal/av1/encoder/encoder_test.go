package encoder

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	internalrtp "github.com/thesyncim/goav1/internal/av1/rtp"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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
	size, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSize(ScalabilityModeL2T2, webRTCScalabilityMetadataDimensions{})
	if err != nil || !ok {
		t.Fatalf("lowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
	}
	var buf [8]byte
	out, ok, err := appendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], ScalabilityModeL2T2, webRTCScalabilityMetadataDimensions{})
	if err != nil || !ok {
		t.Fatalf("appendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
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
	noMetadata, ok, err := appendLowOverheadWebRTCScalabilityMetadataOBU(prefix, ScalabilityModeL1T1, webRTCScalabilityMetadataDimensions{})
	if err != nil || ok || !bytes.Equal(noMetadata, prefix) {
		t.Fatalf("L1T1 metadata out=% x ok=%v err=%v", noMetadata, ok, err)
	}
	if size, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSize(ScalabilityModeL1T1, webRTCScalabilityMetadataDimensions{}); err != nil || ok || size != 0 {
		t.Fatalf("L1T1 size=%d ok=%v err=%v", size, ok, err)
	}

	var tiny [2]byte
	if out, ok, err := appendLowOverheadWebRTCScalabilityMetadataOBU(tiny[:0], ScalabilityModeL2T2, webRTCScalabilityMetadataDimensions{}); !ok || !errors.Is(err, bitstream.ErrShortBuffer) || len(out) != 0 {
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
			size, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSize(tc.mode, webRTCScalabilityMetadataDimensions{})
			if err != nil || !ok {
				t.Fatalf("lowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
			}
			var buf [64]byte
			out, ok, err := appendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], tc.mode, webRTCScalabilityMetadataDimensions{})
			if err != nil || !ok {
				t.Fatalf("appendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
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
			size, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSize(mode, webRTCScalabilityMetadataDimensions{})
			if err != nil {
				t.Fatalf("lowOverheadWebRTCScalabilityMetadataOBUSize: %v", err)
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
			out, ok, err := appendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], mode, webRTCScalabilityMetadataDimensions{})
			if err != nil || !ok {
				t.Fatalf("appendLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
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

func cdefDiffFillPlane(pix []byte, stride, w, h, salt int) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := (x*37 + y*53 + salt*91 + (x>>3)*29 + (y>>3)*17) & 255
			pix[y*stride+x] = byte(v)
		}
	}
}

func cdefDiffFrame(w, h int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	cdefDiffFillPlane(f.Y, f.YStride, w, h, 0)
	cdefDiffFillPlane(f.U, f.ChromaStride, cw, ch, 1)
	cdefDiffFillPlane(f.V, f.ChromaStride, cw, ch, 2)
	return f
}

func cdefDiffCopyInto(dst *SourceFrame420, src SourceFrame420) {
	copy(dst.Y, src.Y)
	copy(dst.U, src.U)
	copy(dst.V, src.V)
}

func cdefDiffMarkSkippedBlocks(m threading.FrameWorkLoopFilterMap) {
	stride := int(m.Stride)
	rows := int(m.Rows)
	for miRow := 2; miRow+1 < rows; miRow += 8 {
		for miCol := 2; miCol+1 < stride; miCol += 10 {
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					m.Records[(miRow+dy)*stride+miCol+dx].SkipTransform = true
				}
			}
		}
	}
}

func cdefDiffApplySerialSnapshot(t *testing.T, a *cdefApplier, recon *SourceFrame420, params parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap) {
	t.Helper()
	active, err := a.bindApplyContext(recon, params, lfMap, nil, true)
	if err != nil {
		t.Fatalf("bind snapshot: %v", err)
	}
	if !active {
		t.Fatal("snapshot route inactive")
	}
	req := a.jobReq
	req.InputScratch = a.bandIn[0]
	req.UnitDstScratch = a.bandUnit[0]
	if _, err := a.jobCtx.ApplyCDEFPostFilterUnitRows(req, 0, a.unitRows); err != nil {
		t.Fatalf("snapshot route: %v", err)
	}
}

func cdefDiffAssertEqual(t *testing.T, snapshot, inPlace SourceFrame420) {
	t.Helper()
	for _, p := range []struct {
		name         string
		a, b         []byte
		stride, w, h int
	}{
		{"Y", snapshot.Y, inPlace.Y, snapshot.YStride, snapshot.Width, snapshot.Height},
		{"U", snapshot.U, inPlace.U, snapshot.ChromaStride, snapshot.Width / 2, snapshot.Height / 2},
		{"V", snapshot.V, inPlace.V, snapshot.ChromaStride, snapshot.Width / 2, snapshot.Height / 2},
	} {
		for y := 0; y < p.h; y++ {
			for x := 0; x < p.w; x++ {
				i := y*p.stride + x
				if p.a[i] != p.b[i] {
					t.Fatalf("plane %s differs at x=%d y=%d: snapshot=%d inplace=%d", p.name, x, y, p.a[i], p.b[i])
				}
			}
		}
	}
}

func TestCDEFApplySerialU8InPlaceMatchesSnapshot(t *testing.T) {
	const width = 160
	const height = 96
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})
	cdefDiffMarkSkippedBlocks(lf.filtMap)

	src := cdefDiffFrame(width, height)
	snapshot := lfDiffCopy(src)
	inPlace := lfDiffCopy(src)

	var snapshotApply cdefApplier
	if err := snapshotApply.init(width, height, params); err != nil {
		t.Fatalf("snapshot cdef init: %v", err)
	}
	defer snapshotApply.close()
	cdefDiffApplySerialSnapshot(t, &snapshotApply, &snapshot, params, &lf.filtMap)

	var inPlaceApply cdefApplier
	if err := inPlaceApply.init(width, height, params); err != nil {
		t.Fatalf("in-place cdef init: %v", err)
	}
	defer inPlaceApply.close()
	if err := inPlaceApply.applySerial(&inPlace, params, &lf.filtMap, nil); err != nil {
		t.Fatalf("in-place route: %v", err)
	}

	changed := false
	for i := range src.Y {
		if inPlace.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("degenerate: CDEF changed no luma pixels")
	}
	cdefDiffAssertEqual(t, snapshot, inPlace)
}

func TestCDEFApplyBandedU8InPlaceMatchesSnapshot(t *testing.T) {
	const width = 192
	const height = 640
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 2,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63, 31},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47, 19},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})
	cdefDiffMarkSkippedBlocks(lf.filtMap)

	src := cdefDiffFrame(width, height)
	snapshot := lfDiffCopy(src)
	inPlace := lfDiffCopy(src)

	var snapshotApply cdefApplier
	if err := snapshotApply.init(width, height, params); err != nil {
		t.Fatalf("snapshot cdef init: %v", err)
	}
	defer snapshotApply.close()
	cdefDiffApplySerialSnapshot(t, &snapshotApply, &snapshot, params, &lf.filtMap)

	var inPlaceApply cdefApplier
	if err := inPlaceApply.init(width, height, params); err != nil {
		t.Fatalf("in-place cdef init: %v", err)
	}
	defer inPlaceApply.close()
	if err := inPlaceApply.apply(&inPlace, params, &lf.filtMap, nil); err != nil {
		t.Fatalf("banded in-place route: %v", err)
	}

	cdefDiffAssertEqual(t, snapshot, inPlace)
}

func TestCDEFApplySerialIsZeroAlloc(t *testing.T) {
	const width = 128
	const height = 128
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})

	var a cdefApplier
	if err := a.init(width, height, params); err != nil {
		t.Fatalf("cdef init: %v", err)
	}
	defer a.close()

	src := cdefDiffFrame(width, height)
	out := lfDiffCopy(src)
	allocs := testing.AllocsPerRun(16, func() {
		cdefDiffCopyInto(&out, src)
		if err := a.applySerial(&out, params, &lf.filtMap, nil); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("serial CDEF apply allocated: %v allocs/op", allocs)
	}
}

func TestCDEFApplyBandedIsZeroAlloc(t *testing.T) {
	const width = 192
	const height = 512
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})

	var a cdefApplier
	if err := a.init(width, height, params); err != nil {
		t.Fatalf("cdef init: %v", err)
	}
	defer a.close()

	src := cdefDiffFrame(width, height)
	out := lfDiffCopy(src)
	if err := a.apply(&out, params, &lf.filtMap, nil); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(16, func() {
		cdefDiffCopyInto(&out, src)
		if err := a.apply(&out, params, &lf.filtMap, nil); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("banded CDEF apply allocated: %v allocs/op", allocs)
	}
}

// TestCDEFDist8MatchesScalar: the dispatched cdefDist8 equals the scalar
// svt_aom_compute_cdef_dist_8bit_c port for every block shape and
// subsampling factor the search uses.
func TestCDEFDist8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	src := make([]byte, 80*70)
	flt := make([]byte, 64*64)
	for trial := 0; trial < 300; trial++ {
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		for i := range flt {
			flt[i] = byte(rng.Intn(256))
		}
		dec := rng.Intn(2)
		n := 8 >> dec
		var dlist []cdef.BlockPosition
		for by := 0; by < n; by++ {
			for bx := 0; bx < n; bx++ {
				if rng.Intn(3) != 0 {
					dlist = append(dlist, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		for _, ss := range []int{1, 2, 4} {
			bl := 3 - dec
			want := computeCDEFDist8(src, 80, flt, 64>>dec, dlist, bl, bl, ss)
			got := cdefDist8(src, 80, flt, 64>>dec, dlist, bl, bl, ss)
			if got != want {
				t.Fatalf("trial %d dec %d ss %d: %d != %d", trial, dec, ss, got, want)
			}
		}
	}
}

func TestAppendCDEFParamsPayloadSkipsLossless(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseCDEFParams(t, seq, size, true, CDEFParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.StrengthCount != 0 || parsed.BitsRead != 0 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
}

func TestAppendCDEFParamsPayloadStrengths(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 5, Bits: 2}
	for i := uint8(0); i < 4; i++ {
		cdef.YStrength[i] = 10 + i
		cdef.UVStrength[i] = 20 + i
	}
	_, parsed := appendAndParseCDEFParams(t, seq, size, false, cdef)
	if parsed.Damping != 5 || parsed.Bits != 2 || parsed.StrengthCount != 4 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
	if parsed.YStrength[3] != 13 || parsed.UVStrength[3] != 23 {
		t.Fatalf("parsed strengths y=%v uv=%v", parsed.YStrength[:4], parsed.UVStrength[:4])
	}
}

func TestAppendCDEFParamsPayloadMonochromeSkipsUVStrengths(t *testing.T) {
	seq := realtimeEncoderMonochromeSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	cdef := CDEFParams{Damping: 4, Bits: 1}
	cdef.YStrength[0] = 7
	cdef.YStrength[1] = 8
	_, parsed := appendAndParseCDEFParams(t, seq, size, false, cdef)
	if parsed.StrengthCount != 2 || parsed.YStrength[0] != 7 || parsed.YStrength[1] != 8 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
	if parsed.UVStrength[0] != 0 || parsed.UVStrength[1] != 0 {
		t.Fatalf("parsed uv strengths=%v", parsed.UVStrength[:2])
	}
}

func TestAppendCDEFParamsPayloadDisabledBySequence(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.EnableCDEF = false
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseCDEFParams(t, seq, size, false, CDEFParams{})
	if len(payload) != 0 || parsed.StrengthCount != 0 {
		t.Fatalf("payload=% x parsed=%+v", payload, parsed)
	}
}

func TestAppendCDEFParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	cdef := CDEFParams{Damping: 3, Bits: 0}
	cdef.YStrength[0] = 11
	cdef.UVStrength[0] = 12
	var buf [4]byte
	payload, err := appendCDEFParamsInterPayload(buf[:0], seq, size, false, cdef)
	if err != nil {
		t.Fatalf("appendCDEFParamsInterPayload: %v", err)
	}
	parsed, err := parser.ParseCDEFParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameSize{}, parser.SegmentationParams{}, parser.LoopFilterParams{})
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	if parsed.Damping != 3 || parsed.Bits != 0 || parsed.YStrength[0] != 11 || parsed.UVStrength[0] != 12 {
		t.Fatalf("parsed inter cdef=%+v", parsed)
	}
}

func TestAppendCDEFParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	nonzeroDisabled := CDEFParams{Damping: 3}
	unused := CDEFParams{Damping: 3, Bits: 0}
	unused.YStrength[1] = 1
	tooHigh := CDEFParams{Damping: 3, Bits: 0}
	tooHigh.YStrength[0] = 64
	cases := [...]CDEFParams{
		{Damping: 2},
		{Damping: 7},
		{Damping: 3, Bits: 4},
		unused,
		tooHigh,
	}
	var buf [8]byte
	for _, cdef := range cases {
		if _, err := appendCDEFParamsPayload(buf[:0], seq, size, false, cdef); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendCDEFParamsPayload(%+v) err=%v want ErrInvalidFrame", cdef, err)
		}
	}
	if _, err := appendCDEFParamsPayload(buf[:0], seq, size, true, nonzeroDisabled); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless nonzero cdef err=%v want ErrInvalidFrame", err)
	}
	mono := seq
	mono.ColorConfig.MonoChrome = true
	uv := CDEFParams{Damping: 3, Bits: 0}
	uv.UVStrength[0] = 1
	if _, err := appendCDEFParamsPayload(buf[:0], mono, size, false, uv); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono uv cdef err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseCDEFParams(t *testing.T, seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) ([]byte, parser.CDEFParams) {
	t.Helper()
	var buf [16]byte
	payload, err := appendCDEFParamsPayload(buf[:0], seq, size, allLossless, cdef)
	if err != nil {
		t.Fatalf("appendCDEFParamsPayload: %v", err)
	}
	parsed, err := parser.ParseCDEFParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.LoopFilterParams{},
	)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	return payload, parsed
}

func TestIntraHeaderTemporalUnitForConfig(t *testing.T) {
	cfg := Config{
		Resolution:         Resolution{Width: 640, Height: 360},
		Scalability:        ScalabilityModeL2T2,
		MaxFramerate:       Rational{Num: 30, Den: 1},
		MinBitrateKbps:     100,
		MaxBitrateKbps:     800,
		TargetBitrateKbps:  500,
		DependencyMetadata: true,
		LowOverheadOBU:     true,
		RTPPacketization:   true,
	}
	unit, err := IntraHeaderTemporalUnitForConfig(cfg, 17)
	if err != nil {
		t.Fatalf("IntraHeaderTemporalUnitForConfig: %v", err)
	}
	if unit.Sequence.OperatingPointsCount != 4 || unit.Sequence.MaxFrameWidth != 640 || unit.Sequence.MaxFrameHeight != 360 {
		t.Fatalf("sequence=%+v", unit.Sequence)
	}
	if unit.Prefix.FrameType != FrameHeaderTypeKey || !unit.Prefix.ShowFrame ||
		!unit.Prefix.ErrorResilientMode || unit.Prefix.OrderHint != 17 ||
		unit.Prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
		t.Fatalf("prefix=%+v", unit.Prefix)
	}
	if unit.Size.UpscaledWidth != 640 || unit.Size.Height != 360 ||
		unit.Size.SuperResDenominator != 8 || unit.Size.RefreshFrameFlags != 0xff {
		t.Fatalf("size=%+v", unit.Size)
	}
}

func TestIntraHeaderTemporalUnitForScreenContent(t *testing.T) {
	cfg := Config{
		Resolution: Resolution{Width: 640, Height: 360},
		Content:    ContentScreen,
	}
	unit, err := IntraHeaderTemporalUnitForConfig(cfg, 3)
	if err != nil {
		t.Fatalf("IntraHeaderTemporalUnitForConfig: %v", err)
	}
	if !unit.Prefix.AllowScreenContentTools || !unit.Prefix.ForceIntegerMV {
		t.Fatalf("screen prefix=%+v", unit.Prefix)
	}
	var buf [64]byte
	payload, err := appendFrameHeaderIntraPayload(buf[:0], unit.Sequence, unit.Prefix, unit.Size)
	if err != nil {
		t.Fatalf("appendFrameHeaderIntraPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, unit.Sequence)
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	if !parsedPrefix.AllowScreenContentTools || !parsedPrefix.ForceIntegerMV {
		t.Fatalf("parsed screen prefix=%+v", parsedPrefix)
	}
}

func TestWebRTCKeyFrameTemporalUnitForConfigLayered(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 7, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if unit.FrameNum != 2 || unit.Control.FrameNum != 2 || !unit.Control.HasDependencyStructure {
		t.Fatalf("unit/control frame counts = %+v control=%+v", unit, unit.Control)
	}
	if unit.Frames[0].Type != FrameTypeKey || unit.Frames[0].Resolution != (Resolution{Width: 320, Height: 180}) ||
		unit.Frames[0].UpdateBuffer != 0 || unit.Frames[0].ReferenceCount != 0 {
		t.Fatalf("base frame=%+v", unit.Frames[0])
	}
	if unit.Frames[1].Type != FrameTypeDelta || unit.Frames[1].Resolution != (Resolution{Width: 640, Height: 360}) ||
		unit.Frames[1].ReferenceCount != 1 || unit.Frames[1].ReferenceBuffers[0] != 0 ||
		unit.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("upper frame=%+v", unit.Frames[1])
	}
	if !unit.Control.Frames[0].AttachDependencyStructure || unit.Control.Frames[1].AttachDependencyStructure ||
		unit.Control.Frames[1].GenericFrameInfo.DependencyNum != 1 ||
		unit.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 100 {
		t.Fatalf("control=%+v", unit.Control)
	}
	if !unit.Control.ReferenceState.Valid[0] || !unit.Control.ReferenceState.Valid[1] ||
		unit.Control.FrameIDState.FrameIDs[0] != 100 || unit.Control.FrameIDState.FrameIDs[1] != 101 {
		t.Fatalf("states refs=%+v ids=%+v", unit.Control.ReferenceState, unit.Control.FrameIDState)
	}
}

func TestWebRTCKeyFrameTemporalUnitForConfigSimulcast(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeS2T1,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 40)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig simulcast: %v", err)
	}
	if unit.FrameNum != 2 || unit.Frames[0].Type != FrameTypeKey || unit.Frames[1].Type != FrameTypeStart ||
		unit.Frames[1].ReferenceCount != 0 {
		t.Fatalf("simulcast frames=%+v", unit.Frames)
	}
	if unit.Control.Frames[1].GenericFrameInfo.DependencyNum != 0 ||
		unit.Control.FrameIDState.FrameIDs[0] != 40 || unit.Control.FrameIDState.FrameIDs[1] != 41 {
		t.Fatalf("simulcast control=%+v", unit.Control)
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigLayered(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200, 0, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if delta.FrameNum != 2 || delta.Control.FrameNum != 2 || delta.Control.HasDependencyStructure {
		t.Fatalf("delta counts/control=%+v", delta)
	}
	if delta.Frames[0].TemporalID != 1 || delta.Frames[0].ReferenceCount != 1 ||
		delta.Frames[0].ReferenceBuffers[0] != 0 ||
		!delta.Frames[0].UpdateBufferSet || delta.Frames[0].UpdateBuffer != 2 {
		t.Fatalf("base delta=%+v", delta.Frames[0])
	}
	if delta.Frames[1].TemporalID != 1 || delta.Frames[1].ReferenceCount != 2 ||
		delta.Frames[1].ReferenceBuffers[0] != 1 || delta.Frames[1].ReferenceBuffers[1] != 2 ||
		delta.Frames[1].UpdateBufferSet {
		t.Fatalf("upper delta=%+v", delta.Frames[1])
	}
	if delta.Control.Frames[0].GenericFrameInfo.DependencyNum != 1 ||
		delta.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 2 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 200 {
		t.Fatalf("delta generic info=%+v %+v", delta.Control.Frames[0].GenericFrameInfo, delta.Control.Frames[1].GenericFrameInfo)
	}
	if delta.Control.FrameIDState.FrameIDs[0] != 100 || delta.Control.FrameIDState.FrameIDs[1] != 101 ||
		delta.Control.FrameIDState.FrameIDs[2] != 200 {
		t.Fatalf("delta frame id state=%+v", delta.Control.FrameIDState)
	}
	if delta.Headers[0].Prefix.FrameType != FrameHeaderTypeInter ||
		!delta.Headers[0].Prefix.ErrorResilientMode ||
		delta.Headers[0].Prefix.OrderHint != 0 ||
		delta.Headers[0].Size.UpscaledWidth != 320 ||
		delta.Headers[0].Size.Height != 180 ||
		delta.Headers[0].Size.RefreshFrameFlags != 0x04 ||
		delta.Headers[0].Size.RefFrameIdx[0] != 0 {
		t.Fatalf("base delta header=%+v", delta.Headers[0])
	}
	if delta.Headers[1].Size.UpscaledWidth != 640 ||
		delta.Headers[1].Size.Height != 360 ||
		delta.Headers[1].Size.RefreshFrameFlags != 0x00 ||
		delta.Headers[1].Size.RefFrameIdx[0] != 1 ||
		delta.Headers[1].Size.RefFrameIdx[1] != 2 {
		t.Fatalf("upper delta header=%+v", delta.Headers[1])
	}
	for i := uint8(0); i < delta.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta.Headers[i])
	}

	base, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, delta.Control.ReferenceState, delta.Control.FrameIDState, 0, 202, 0, 0, false)
	if err != nil {
		t.Fatalf("base delta: %v", err)
	}
	if base.Frames[0].TemporalID != 0 || base.Frames[0].ReferenceBuffers[0] != 0 ||
		!base.Frames[0].UpdateBufferSet || base.Frames[0].UpdateBuffer != 0 ||
		base.Frames[1].TemporalID != 0 || base.Frames[1].ReferenceBuffers[0] != 1 ||
		base.Frames[1].ReferenceBuffers[1] != 0 ||
		!base.Frames[1].UpdateBufferSet || base.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("base delta frames=%+v", base.Frames)
	}
	if base.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		base.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		base.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 202 {
		t.Fatalf("base delta generic info=%+v %+v", base.Control.Frames[0].GenericFrameInfo, base.Control.Frames[1].GenericFrameInfo)
	}
}

func TestWebRTCEncoderStateTemporalUnitsL2T2KeyShift(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2_KEY_SHIFT,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 100})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 2 || key.Frames[0].Type != FrameTypeKey ||
		key.Frames[1].ReferenceCount != 1 || key.Frames[1].ReferenceBuffers[0] != 0 ||
		state.NextFrameID != 102 || state.DeltaPictureIndex != 1 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}

	delta0, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta0: %v", err)
	}
	if delta0.FrameNum != 2 ||
		delta0.Frames[0].SpatialID != 0 || delta0.Frames[0].TemporalID != 0 ||
		delta0.Frames[0].ReferenceCount != 1 || delta0.Frames[0].ReferenceBuffers[0] != 0 ||
		!delta0.Frames[0].UpdateBufferSet || delta0.Frames[0].UpdateBuffer != 0 ||
		delta0.Frames[1].SpatialID != 1 || delta0.Frames[1].TemporalID != 1 ||
		delta0.Frames[1].ReferenceCount != 1 || delta0.Frames[1].ReferenceBuffers[0] != 1 ||
		delta0.Frames[1].UpdateBufferSet {
		t.Fatalf("delta0 frames=%+v", delta0.Frames)
	}
	if delta0.Headers[0].Size.RefreshFrameFlags != 0x01 || delta0.Headers[1].Size.RefreshFrameFlags != 0x00 {
		t.Fatalf("delta0 refresh flags=%02x,%02x", delta0.Headers[0].Size.RefreshFrameFlags, delta0.Headers[1].Size.RefreshFrameFlags)
	}
	if delta0.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 100 ||
		delta0.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		state.FrameIDState.FrameIDs[0] != 102 || state.FrameIDState.FrameIDs[1] != 101 {
		t.Fatalf("delta0 control=%+v state=%+v", delta0.Control, state)
	}
	for i := uint8(0); i < delta0.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta0.Headers[i])
	}

	delta1, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta1: %v", err)
	}
	if delta1.FrameNum != 2 ||
		delta1.Frames[0].SpatialID != 0 || delta1.Frames[0].TemporalID != 1 ||
		delta1.Frames[0].ReferenceCount != 1 || delta1.Frames[0].ReferenceBuffers[0] != 0 ||
		delta1.Frames[0].UpdateBufferSet ||
		delta1.Frames[1].SpatialID != 1 || delta1.Frames[1].TemporalID != 0 ||
		delta1.Frames[1].ReferenceCount != 1 || delta1.Frames[1].ReferenceBuffers[0] != 1 ||
		!delta1.Frames[1].UpdateBufferSet || delta1.Frames[1].UpdateBuffer != 1 {
		t.Fatalf("delta1 frames=%+v", delta1.Frames)
	}
	if delta1.Headers[0].Size.RefreshFrameFlags != 0x00 || delta1.Headers[1].Size.RefreshFrameFlags != 0x02 {
		t.Fatalf("delta1 refresh flags=%02x,%02x", delta1.Headers[0].Size.RefreshFrameFlags, delta1.Headers[1].Size.RefreshFrameFlags)
	}
	if delta1.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 102 ||
		delta1.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 101 ||
		state.FrameIDState.FrameIDs[0] != 102 || state.FrameIDState.FrameIDs[1] != 105 {
		t.Fatalf("delta1 control=%+v state=%+v", delta1.Control, state)
	}
	for i := uint8(0); i < delta1.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta1.Headers[i])
	}

	delta2, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta2: %v", err)
	}
	if delta2.Control.Frames[0].GenericFrameInfo.Dependencies[0] != 102 ||
		delta2.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 105 {
		t.Fatalf("delta2 control=%+v", delta2.Control)
	}
}

func TestWebRTCEncoderStateTemporalUnitsKeyShiftModes(t *testing.T) {
	for _, tc := range [...]struct {
		name string
		mode ScalabilityMode
		want [][]uint8
	}{
		{name: "L2T3", mode: ScalabilityModeL2T3_KEY_SHIFT, want: [][]uint8{
			{2, 2},
			{0, 1},
			{2, 2},
			{1, 0},
		}},
		{name: "L3T2", mode: ScalabilityModeL3T2_KEY_SHIFT, want: [][]uint8{
			{0, 0, 1},
			{1, 1, 0},
		}},
		{name: "L3T3", mode: ScalabilityModeL3T3_KEY_SHIFT, want: [][]uint8{
			{0, 2, 2},
			{2, 0, 1},
			{1, 2, 2},
			{2, 1, 0},
		}},
	} {
		cfg := Config{
			Resolution:        Resolution{Width: 1280, Height: 720},
			Scalability:       tc.mode,
			MaxFramerate:      Rational{Num: 30, Den: 1},
			MinBitrateKbps:    100,
			MaxBitrateKbps:    1200,
			TargetBitrateKbps: 800,
		}
		_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 10})
		if err != nil {
			t.Fatalf("%s key: %v", tc.name, err)
		}
		for i, want := range tc.want {
			delta, next, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
			if err != nil {
				t.Fatalf("%s delta %d: %v", tc.name, i, err)
			}
			if int(delta.FrameNum) != len(want) {
				t.Fatalf("%s delta %d frame num=%d want %d", tc.name, i, delta.FrameNum, len(want))
			}
			for spatialID, wantTemporalID := range want {
				frame := delta.Frames[spatialID]
				primaryBuffer := uint8(spatialID)
				middleBuffer := delta.FrameNum + uint8(spatialID)
				selfReference := frame.ReferenceBuffers[0] == primaryBuffer ||
					(frame.TemporalID == 2 && frame.ReferenceBuffers[0] == middleBuffer)
				if frame.SpatialID != uint8(spatialID) || frame.TemporalID != wantTemporalID ||
					frame.ReferenceCount != 1 || !selfReference {
					t.Fatalf("%s delta %d spatial %d frame=%+v want temporal %d single self ref", tc.name, i, spatialID, frame, wantTemporalID)
				}
				if frame.UpdateBufferSet && frame.UpdateBuffer != primaryBuffer && frame.UpdateBuffer != middleBuffer {
					t.Fatalf("%s delta %d spatial %d update buffer=%d", tc.name, i, spatialID, frame.UpdateBuffer)
				}
			}
			state = next
		}
	}
}

func TestWebRTCControllerSettingsMatrix(t *testing.T) {
	profiles := [...]struct {
		name               string
		fps                Rational
		rateControl        RateControlMode
		quantizer          uint8
		minKbps            int32
		maxKbps            int32
		targetKbps         int32
		content            ContentHint
		dependencyMetadata bool
		lowOverheadOBU     bool
		rtpPacketization   bool
	}{
		{
			name:               "camera-cbr-30fps-rtp",
			fps:                Rational{Num: 30, Den: 1},
			rateControl:        RateControlCBR,
			minKbps:            120,
			maxKbps:            1800,
			targetKbps:         900,
			content:            ContentCamera,
			dependencyMetadata: true,
			lowOverheadOBU:     true,
			rtpPacketization:   true,
		},
		{
			name:               "screen-cqp-ntsc-headers",
			fps:                Rational{Num: 30000, Den: 1001},
			rateControl:        RateControlCQP,
			quantizer:          37,
			minKbps:            80,
			maxKbps:            1200,
			targetKbps:         640,
			content:            ContentScreen,
			dependencyMetadata: true,
			lowOverheadOBU:     true,
			rtpPacketization:   false,
		},
	}

	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		for _, profile := range profiles {
			name := mode.String() + "/" + profile.name
			t.Run(name, func(t *testing.T) {
				cfg := Config{
					Resolution:         Resolution{Width: 1280, Height: 720},
					MaxFramerate:       profile.fps,
					RateControl:        profile.rateControl,
					Quantizer:          profile.quantizer,
					MinBitrateKbps:     profile.minKbps,
					MaxBitrateKbps:     profile.maxKbps,
					TargetBitrateKbps:  profile.targetKbps,
					Content:            profile.content,
					Scalability:        mode,
					DependencyMetadata: profile.dependencyMetadata,
					LowOverheadOBU:     profile.lowOverheadOBU,
					RTPPacketization:   profile.rtpPacketization,
				}
				assertWebRTCControllerReferenceModel(t, cfg)
			})
		}
	}
}

func TestWebRTCControllerSettingsMatrixKeyInterval(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1280, Height: 720},
		Scalability:       ScalabilityModeL2T3_KEY_SHIFT,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
		KeyFrameInterval:  2,
	}
	unit, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1}, false)
	if err != nil || !unit.Key || unit.Delta {
		t.Fatalf("initial key unit=%+v state=%+v err=%v", unit, state, err)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil || !unit.Delta || unit.Key || state.DeltaPictureIndex != 2 {
		t.Fatalf("first delta unit=%+v state=%+v err=%v", unit, state, err)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil || !unit.Key || unit.Delta || state.DeltaPictureIndex != 1 {
		t.Fatalf("interval key unit=%+v state=%+v err=%v", unit, state, err)
	}
}

func TestWebRTCEncoderStateReconfigureControls(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1280, Height: 720},
		Scalability:       ScalabilityModeL1T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	_, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 20})
	if err != nil {
		t.Fatalf("key: %v", err)
	}
	delta, state, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("delta: %v", err)
	}
	if delta.FrameNum != 1 || state.DeltaPictureIndex != 2 {
		t.Fatalf("warm state delta=%+v state=%+v", delta, state)
	}

	controlChange := cfg
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1800
	controlChange.TargetBitrateKbps = 1100
	unit, next, err := WebRTCNextTemporalUnitForState(controlChange, state, false)
	if err != nil {
		t.Fatalf("control reconfigure: %v", err)
	}
	if !unit.Delta || unit.Key || unit.DeltaUnit.FrameNum != 1 || next.DeltaPictureIndex != 3 {
		t.Fatalf("control reconfigure unit=%+v next=%+v", unit, next)
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T2
	unit, next, err = WebRTCNextTemporalUnitForState(structureChange, next, false)
	if err != nil {
		t.Fatalf("structure reconfigure: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		!unit.KeyUnit.Control.HasDependencyStructure || next.DeltaPictureIndex != 1 {
		t.Fatalf("structure reconfigure unit=%+v next=%+v", unit, next)
	}
}

func assertWebRTCControllerReferenceModel(t *testing.T, cfg Config) {
	t.Helper()
	normalized, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig: %v", err)
	}
	spatialLayers, temporalLayers, _, ok := normalized.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid mode %v", normalized.Scalability)
	}
	if normalized.SpatialLayerCount != spatialLayers || normalized.TemporalLayerCount != temporalLayers {
		t.Fatalf("normalized layers=%d,%d want %d,%d", normalized.SpatialLayerCount, normalized.TemporalLayerCount, spatialLayers, temporalLayers)
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if structure.NumDecodeTargets != spatialLayers*temporalLayers ||
		structure.NumChains != spatialLayers ||
		structure.ResolutionNum != spatialLayers {
		t.Fatalf("structure shape=%+v", structure)
	}

	if picture, _, err := WebRTCNextTemporalUnitForState(normalized, WebRTCEncoderState{NextFrameID: 1}, false); err != nil {
		t.Fatalf("WebRTCNextTemporalUnitForState key: %v", err)
	} else if !picture.Key || picture.Delta {
		t.Fatalf("first picture=%+v want key", picture)
	}

	key, state, err := WebRTCKeyFrameTemporalUnitForState(normalized, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != spatialLayers || !key.Control.HasDependencyStructure ||
		state.NextFrameID != 1+uint64(spatialLayers) || state.DeltaPictureIndex != 1 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}
	type frameLayer struct {
		spatial  uint8
		temporal uint8
	}
	history := make(map[uint64]frameLayer, 64)
	for i := uint8(0); i < key.Control.FrameNum; i++ {
		info := key.Control.Frames[i].GenericFrameInfo
		history[info.FrameID] = frameLayer{spatial: info.SpatialID, temporal: info.TemporalID}
	}

	steps := webRTCTestControllerMatrixSteps(temporalLayers)
	for deltaIndex := uint64(1); deltaIndex <= steps; deltaIndex++ {
		currentState := state
		delta, next, err := WebRTCDeltaFrameTemporalUnitForState(normalized, state)
		if err != nil {
			t.Fatalf("delta %d: %v", deltaIndex, err)
		}
		if delta.FrameNum != spatialLayers || delta.Control.HasDependencyStructure {
			t.Fatalf("delta %d unit/control=%+v", deltaIndex, delta)
		}
		if next.NextFrameID != state.NextFrameID+uint64(spatialLayers) ||
			next.DeltaPictureIndex != state.DeltaPictureIndex+1 {
			t.Fatalf("delta %d next=%+v state=%+v", deltaIndex, next, state)
		}
		for i := uint8(0); i < delta.FrameNum; i++ {
			frame := delta.Frames[i]
			control := delta.Control.Frames[i]
			info := control.GenericFrameInfo
			wantTemporal := webRTCTestExpectedTemporalID(normalized.Scalability, i, deltaIndex)
			if frame.SpatialID != i || info.SpatialID != i ||
				frame.TemporalID != wantTemporal || info.TemporalID != wantTemporal ||
				delta.Headers[i].TemporalID != wantTemporal || delta.Headers[i].SpatialID != i {
				t.Fatalf("delta %d frame %d frame=%+v info=%+v header=%+v want temporal=%d", deltaIndex, i, frame, info, delta.Headers[i], wantTemporal)
			}
			if frame.RateControl != normalized.RateControl ||
				(normalized.RateControl == RateControlCQP && frame.Quantizer != normalized.Quantizer) {
				t.Fatalf("delta %d frame %d rate control frame=%+v config=%+v", deltaIndex, i, frame, normalized)
			}
			// SVT-AV1 rtc: screen tools on I-slices only, force_integer_mv 0.
			if delta.Headers[i].Prefix.AllowScreenContentTools || delta.Headers[i].Prefix.ForceIntegerMV {
				t.Fatalf("delta %d frame %d screen header=%+v", deltaIndex, i, delta.Headers[i])
			}
			wantRefresh := uint8(0)
			if frame.UpdateBufferSet {
				wantRefresh = 1 << frame.UpdateBuffer
			}
			if delta.Headers[i].Size.RefreshFrameFlags != wantRefresh {
				t.Fatalf("delta %d frame %d refresh=%02x want %02x frame=%+v", deltaIndex, i, delta.Headers[i].Size.RefreshFrameFlags, wantRefresh, frame)
			}
			if (normalized.Scalability.IsSimulcast() || normalized.Scalability.UsesKeyFrameInterLayerDependency()) && frame.ReferenceCount != 1 {
				t.Fatalf("delta %d frame %d key/simulcast refs=%+v", deltaIndex, i, frame)
			}
			if webRTCUsesDeltaInterLayerReference(normalized) && i > 0 && frame.ReferenceCount < 2 {
				t.Fatalf("delta %d frame %d full-svc refs=%+v", deltaIndex, i, frame)
			}
			for j := uint8(0); j < info.DependencyNum; j++ {
				dep, ok := history[info.Dependencies[j]]
				if !ok {
					t.Fatalf("delta %d frame %d missing dependency %d in history", deltaIndex, i, info.Dependencies[j])
				}
				if dep.temporal > info.TemporalID {
					t.Fatalf("delta %d frame %d temporal dependency %d has T%d > T%d", deltaIndex, i, info.Dependencies[j], dep.temporal, info.TemporalID)
				}
				if normalized.Scalability.IsSimulcast() && dep.spatial != info.SpatialID {
					t.Fatalf("delta %d frame %d simulcast dependency spatial=%d want %d", deltaIndex, i, dep.spatial, info.SpatialID)
				}
			}
			var descriptorBuf [256]byte
			if _, err := AppendWebRTCDependencyDescriptor(descriptorBuf[:0], currentState.DependencyStructureState.Structure, info, true, true, false); err != nil {
				t.Fatalf("delta %d frame %d dependency descriptor: %v", deltaIndex, i, err)
			}
			history[info.FrameID] = frameLayer{spatial: info.SpatialID, temporal: info.TemporalID}
		}
		state = next
	}
}

func webRTCTestControllerMatrixSteps(temporalLayers uint8) uint64 {
	switch temporalLayers {
	case 1:
		return 3
	case 2:
		return 4
	default:
		return 8
	}
}

func webRTCTestExpectedTemporalID(mode ScalabilityMode, spatialID uint8, deltaPictureIndex uint64) uint8 {
	if mode.UsesKeyFrameInterLayerDependencyShift() {
		switch mode {
		case ScalabilityModeL2T2_KEY_SHIFT:
			table := [2][2]uint8{{0, 1}, {1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case ScalabilityModeL2T3_KEY_SHIFT:
			table := [4][2]uint8{{2, 2}, {0, 1}, {2, 2}, {1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		case ScalabilityModeL3T2_KEY_SHIFT:
			table := [2][3]uint8{{0, 0, 1}, {1, 1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case ScalabilityModeL3T3_KEY_SHIFT:
			table := [4][3]uint8{{0, 2, 2}, {2, 0, 1}, {1, 2, 2}, {2, 1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		}
	}
	_, temporalLayers, _, ok := mode.Layers()
	if !ok || deltaPictureIndex == 0 {
		return 0
	}
	trailingZeroCount := uint8(0)
	for value := deltaPictureIndex; value&1 == 0 && trailingZeroCount < temporalLayers-1; value >>= 1 {
		trailingZeroCount++
	}
	return temporalLayers - 1 - trailingZeroCount
}

func TestWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 1)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2, 37, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if delta.Headers[0].Prefix.OrderHint != 37 {
		t.Fatalf("delta header order hint=%d want 37", delta.Headers[0].Prefix.OrderHint)
	}
	assertParsedDeltaHeader(t, delta.Headers[0])
}

func TestWebRTCDeltaFrameTemporalUnitForConfigSimulcast(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeS2T1,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 20, 0, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint simulcast: %v", err)
	}
	if delta.FrameNum != 2 || delta.Frames[0].ReferenceCount != 1 || delta.Frames[1].ReferenceCount != 1 ||
		delta.Frames[0].ReferenceBuffers[0] != 0 || delta.Frames[1].ReferenceBuffers[0] != 1 {
		t.Fatalf("simulcast delta frames=%+v", delta.Frames)
	}
	if delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 1 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 11 {
		t.Fatalf("simulcast delta control=%+v", delta.Control.Frames[1].GenericFrameInfo)
	}
}

func TestWebRTCTemporalUnitsPropagateCQPQuantizer(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       RateControlCQP,
		Quantizer:         37,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.Frames[0].RateControl != RateControlCQP || key.Frames[0].Quantizer != 37 ||
		key.Frames[1].RateControl != RateControlCQP || key.Frames[1].Quantizer != 37 {
		t.Fatalf("key frames=%+v", key.Frames)
	}
	delta, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if delta.Frames[0].RateControl != RateControlCQP || delta.Frames[0].Quantizer != 37 ||
		delta.Frames[1].RateControl != RateControlCQP || delta.Frames[1].Quantizer != 37 {
		t.Fatalf("delta frames=%+v", delta.Frames)
	}
}

func TestWebRTCTemporalUnitsScreenContentToolsOnKeyOnly(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Content:           ContentScreen,
	}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if !key.Header.Prefix.AllowScreenContentTools || !key.Header.Prefix.ForceIntegerMV {
		t.Fatalf("key prefix=%+v", key.Header.Prefix)
	}
	delta, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	// SVT-AV1 rtc: screen tools on I-slices only, force_integer_mv 0.
	if delta.Headers[0].Prefix.AllowScreenContentTools || delta.Headers[0].Prefix.ForceIntegerMV ||
		delta.Headers[1].Prefix.AllowScreenContentTools || delta.Headers[1].Prefix.ForceIntegerMV {
		t.Fatalf("delta headers=%+v", delta.Headers)
	}
	for i := uint8(0); i < delta.FrameNum; i++ {
		assertParsedDeltaHeader(t, delta.Headers[i])
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigRejectsInvalidState(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	if _, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, ReferenceBufferState{}, FrameIDBufferState{}, 0, 1, 0, 0, false); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing reference state err=%v want %v", err, ErrInvalidFrame)
	}
	ref := ReferenceBufferState{}
	ids := FrameIDBufferState{}
	ref.Valid[0] = true
	ref.Resolutions[0] = Resolution{Width: 320, Height: 180}
	ids.Valid[0] = true
	ids.FrameIDs[0] = 1
	if _, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, ref, ids, 2, 2, 0, 0, false); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("bad temporal id err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestWebRTCDeltaFrameTemporalUnitForConfigAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 1)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if _, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2, 0, 0, false); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 2, 0, 0, false)
	})
	if allocs != 0 {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint allocated: %f", allocs)
	}
}

func TestWebRTCEncoderStateTemporalUnits(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 960, Height: 540},
		Scalability:       ScalabilityModeL3T3,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 800,
	}
	state := WebRTCEncoderState{NextOrderHint: 126, NextFrameID: 1000}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 3 || key.Header.Prefix.OrderHint != 126 ||
		state.NextOrderHint != 127 || state.NextFrameID != 1003 ||
		state.DeltaPictureIndex != 1 || !state.DependencyStructureState.Valid {
		t.Fatalf("key=%+v state=%+v", key, state)
	}

	wantTemporal := [...]uint8{2, 1, 2, 0}
	wantFirstID := uint64(1003)
	for i, want := range wantTemporal {
		delta, next, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state)
		if err != nil {
			t.Fatalf("delta %d: %v", i, err)
		}
		if delta.FrameNum != 3 || delta.Frames[0].TemporalID != want ||
			delta.Control.Frames[0].GenericFrameInfo.FrameID != wantFirstID ||
			delta.Control.HasDependencyStructure ||
			delta.Headers[0].Prefix.OrderHint != state.NextOrderHint {
			t.Fatalf("delta %d unit=%+v", i, delta)
		}
		if next.NextFrameID != wantFirstID+3 || next.DeltaPictureIndex != state.DeltaPictureIndex+1 ||
			next.DependencyStructureState.Structure != state.DependencyStructureState.Structure {
			t.Fatalf("delta %d next=%+v state=%+v", i, next, state)
		}
		if i == 0 && next.NextOrderHint != 0 {
			t.Fatalf("delta %d order hint=%d want wrap to 0", i, next.NextOrderHint)
		}
		var descriptorBuf [32]byte
		if _, err := AppendWebRTCDependencyDescriptor(
			descriptorBuf[:0],
			next.DependencyStructureState.Structure,
			delta.Control.Frames[2].GenericFrameInfo,
			true,
			true,
			false,
		); err != nil {
			t.Fatalf("delta %d descriptor: %v", i, err)
		}
		state = next
		wantFirstID += 3
	}
}

func assertParsedDeltaHeader(t *testing.T, header InterHeaderFrame) {
	t.Helper()
	var buf [80]byte
	payload, err := appendFrameHeaderInterPayload(buf[:0], header.Sequence, header.Prefix, header.Size)
	if err != nil {
		t.Fatalf("appendFrameHeaderInterPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, header.Sequence)
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := parserReferenceState(header.Sequence, header.Size.RefFrameIdx[:])
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if parsedPrefix.FrameType != parser.FrameTypeInter ||
		parsedPrefix.OrderHint != header.Prefix.OrderHint ||
		parsedPrefix.AllowScreenContentTools != header.Prefix.AllowScreenContentTools ||
		parsedPrefix.ForceIntegerMV != header.Prefix.ForceIntegerMV ||
		parsedSize.RefreshFrameFlags != header.Size.RefreshFrameFlags ||
		parsedSize.UpscaledWidth != header.Size.UpscaledWidth ||
		parsedSize.Height != header.Size.Height {
		t.Fatalf("parsed prefix=%+v size=%+v header=%+v", parsedPrefix, parsedSize, header)
	}
}

func TestWebRTCNextTemporalUnitForStateKeyInterval(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		Scalability:       ScalabilityModeL2T2,
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  3,
	}
	state := WebRTCEncoderState{NextFrameID: 100}

	unit, state, err := WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("initial next temporal unit: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		state.NextFrameID != 102 || state.DeltaPictureIndex != 1 ||
		!state.DependencyStructureState.Valid {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}

	want := [...]struct {
		key      bool
		temporal uint8
		firstID  uint64
	}{
		{temporal: 1, firstID: 102},
		{temporal: 0, firstID: 104},
		{key: true, firstID: 106},
		{temporal: 1, firstID: 108},
	}
	for i, want := range want {
		unit, next, err := WebRTCNextTemporalUnitForState(cfg, state, false)
		if err != nil {
			t.Fatalf("next %d: %v", i, err)
		}
		if want.key {
			if !unit.Key || unit.Delta || unit.KeyUnit.Control.Frames[0].GenericFrameInfo.FrameID != want.firstID ||
				!unit.KeyUnit.Control.HasDependencyStructure || next.DeltaPictureIndex != 1 {
				t.Fatalf("next %d key unit=%+v next=%+v", i, unit, next)
			}
		} else {
			if !unit.Delta || unit.Key || unit.DeltaUnit.Frames[0].TemporalID != want.temporal ||
				unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != want.firstID ||
				unit.DeltaUnit.Control.HasDependencyStructure {
				t.Fatalf("next %d delta unit=%+v", i, unit)
			}
		}
		state = next
	}
}

func TestWebRTCNextTemporalUnitForStateForceKey(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	unit, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 9}, false)
	if err != nil {
		t.Fatalf("initial next: %v", err)
	}
	if !unit.Key || state.NextFrameID != 10 {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("delta next: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 10 {
		t.Fatalf("delta unit=%+v state=%+v", unit, state)
	}
	unit, state, err = WebRTCNextTemporalUnitForState(cfg, state, true)
	if err != nil {
		t.Fatalf("forced key next: %v", err)
	}
	if !unit.Key || unit.KeyUnit.Control.Frames[0].GenericFrameInfo.FrameID != 11 ||
		!unit.KeyUnit.Control.HasDependencyStructure || state.DeltaPictureIndex != 1 {
		t.Fatalf("forced unit=%+v state=%+v", unit, state)
	}
}

func TestWebRTCTemporalIDForDeltaPicture(t *testing.T) {
	tests := [...]struct {
		layers uint8
		index  uint64
		want   uint8
	}{
		{layers: 1, index: 1, want: 0},
		{layers: 2, index: 1, want: 1},
		{layers: 2, index: 2, want: 0},
		{layers: 3, index: 1, want: 2},
		{layers: 3, index: 2, want: 1},
		{layers: 3, index: 3, want: 2},
		{layers: 3, index: 4, want: 0},
	}
	for _, tt := range tests {
		got, err := WebRTCTemporalIDForDeltaPicture(tt.layers, tt.index)
		if err != nil {
			t.Fatalf("layers=%d index=%d: %v", tt.layers, tt.index, err)
		}
		if got != tt.want {
			t.Fatalf("layers=%d index=%d got=%d want=%d", tt.layers, tt.index, got, tt.want)
		}
	}
	if _, err := WebRTCTemporalIDForDeltaPicture(3, 0); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("zero index err=%v want %v", err, ErrInvalidConfig)
	}
	if _, err := WebRTCTemporalIDForDeltaPicture(WebRTCMaxTemporalLayers+1, 1); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("too many layers err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestWebRTCEncoderStateRejectsDeltaBeforeKey(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL1T1}
	_, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, WebRTCEncoderState{DeltaPictureIndex: 1})
	if !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("delta before key err=%v want %v", err, ErrInvalidFrame)
	}
}

func TestWebRTCEncoderStateTemporalUnitsAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	key, state, err := WebRTCKeyFrameTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 2 || state.NextFrameID != 3 {
		t.Fatalf("key=%+v state=%+v", key, state)
	}
	if _, _, err := WebRTCDeltaFrameTemporalUnitForState(cfg, state); err != nil {
		t.Fatalf("delta preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestWebRTCNextTemporalUnitForStateAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2, KeyFrameInterval: 10}
	_, state, err := WebRTCNextTemporalUnitForState(cfg, WebRTCEncoderState{NextFrameID: 1}, false)
	if err != nil {
		t.Fatalf("initial next: %v", err)
	}
	if _, _, err := WebRTCNextTemporalUnitForState(cfg, state, false); err != nil {
		t.Fatalf("delta preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCNextTemporalUnitForState(cfg, state, false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCNextTemporalUnitForState allocated: %f", allocs)
	}
}

func TestProfileString(t *testing.T) {
	for _, tc := range []struct {
		in      string
		profile Profile
	}{
		{in: "0", profile: Profile0},
		{in: "1", profile: Profile1},
		{in: "2", profile: Profile2},
	} {
		if got := tc.profile.String(); got != tc.in {
			t.Fatalf("Profile(%q).String() = %q", tc.in, got)
		}
	}
}

func TestScalabilityModeLayers(t *testing.T) {
	for _, tc := range []struct {
		in       string
		mode     ScalabilityMode
		spatial  uint8
		temporal uint8
		key      bool
		small    bool
		sim      bool
	}{
		{in: "L1T1", mode: ScalabilityModeL1T1, spatial: 1, temporal: 1},
		{in: "L2T2_KEY_SHIFT", mode: ScalabilityModeL2T2_KEY_SHIFT, spatial: 2, temporal: 2, key: true},
		{in: "L2T3_KEY_SHIFT", mode: ScalabilityModeL2T3_KEY_SHIFT, spatial: 2, temporal: 3, key: true},
		{in: "L2T1h", mode: ScalabilityModeL2T1h, spatial: 2, temporal: 1, small: true},
		{in: "L3T2_KEY_SHIFT", mode: ScalabilityModeL3T2_KEY_SHIFT, spatial: 3, temporal: 2, key: true},
		{in: "L3T3_KEY", mode: ScalabilityModeL3T3_KEY, spatial: 3, temporal: 3, key: true},
		{in: "L3T3_KEY_SHIFT", mode: ScalabilityModeL3T3_KEY_SHIFT, spatial: 3, temporal: 3, key: true},
		{in: "S3T2h", mode: ScalabilityModeS3T2h, spatial: 3, temporal: 2, small: true, sim: true},
	} {
		got := tc.mode
		spatial, temporal, key, ok := got.Layers()
		if !ok || spatial != tc.spatial || temporal != tc.temporal || key != tc.key {
			t.Fatalf("%q Layers() = %d,%d,%v,%v", tc.in, spatial, temporal, key, ok)
		}
		wantShared := tc.spatial > 1 && !tc.sim
		if got.UsesSmallResolutionStep() != tc.small ||
			got.IsSimulcast() != tc.sim ||
			got.UsesSharedReferenceSlots() != wantShared {
			t.Fatalf("%q flags small=%v sim=%v shared=%v", tc.in, got.UsesSmallResolutionStep(), got.IsSimulcast(), got.UsesSharedReferenceSlots())
		}
		if got.UsesKeyFrameInterLayerDependency() != tc.key {
			t.Fatalf("%q key flag=%v", tc.in, got.UsesKeyFrameInterLayerDependency())
		}
		if got.String() != tc.in {
			t.Fatalf("%v.String() = %q; want %q", got, got.String(), tc.in)
		}
	}
}

func TestScalabilityModePinnedLibWebRTCSVCModes(t *testing.T) {
	for _, tc := range []struct {
		name      string
		spatial   uint8
		temporal  uint8
		key       bool
		keyShift  bool
		smallStep bool
		simulcast bool
	}{
		{name: "L1T1", spatial: 1, temporal: 1},
		{name: "L1T2", spatial: 1, temporal: 2},
		{name: "L1T3", spatial: 1, temporal: 3},
		{name: "L2T1", spatial: 2, temporal: 1},
		{name: "L2T1h", spatial: 2, temporal: 1, smallStep: true},
		{name: "L2T1_KEY", spatial: 2, temporal: 1, key: true},
		{name: "L2T2", spatial: 2, temporal: 2},
		{name: "L2T2h", spatial: 2, temporal: 2, smallStep: true},
		{name: "L2T2_KEY", spatial: 2, temporal: 2, key: true},
		{name: "L2T2_KEY_SHIFT", spatial: 2, temporal: 2, key: true, keyShift: true},
		{name: "L2T3", spatial: 2, temporal: 3},
		{name: "L2T3h", spatial: 2, temporal: 3, smallStep: true},
		{name: "L2T3_KEY", spatial: 2, temporal: 3, key: true},
		{name: "L3T1", spatial: 3, temporal: 1},
		{name: "L3T1h", spatial: 3, temporal: 1, smallStep: true},
		{name: "L3T1_KEY", spatial: 3, temporal: 1, key: true},
		{name: "L3T2", spatial: 3, temporal: 2},
		{name: "L3T2h", spatial: 3, temporal: 2, smallStep: true},
		{name: "L3T2_KEY", spatial: 3, temporal: 2, key: true},
		{name: "L3T3", spatial: 3, temporal: 3},
		{name: "L3T3h", spatial: 3, temporal: 3, smallStep: true},
		{name: "L3T3_KEY", spatial: 3, temporal: 3, key: true},
		{name: "S2T1", spatial: 2, temporal: 1, simulcast: true},
		{name: "S2T1h", spatial: 2, temporal: 1, smallStep: true, simulcast: true},
		{name: "S2T2", spatial: 2, temporal: 2, simulcast: true},
		{name: "S2T2h", spatial: 2, temporal: 2, smallStep: true, simulcast: true},
		{name: "S2T3", spatial: 2, temporal: 3, simulcast: true},
		{name: "S2T3h", spatial: 2, temporal: 3, smallStep: true, simulcast: true},
		{name: "S3T1", spatial: 3, temporal: 1, simulcast: true},
		{name: "S3T1h", spatial: 3, temporal: 1, smallStep: true, simulcast: true},
		{name: "S3T2", spatial: 3, temporal: 2, simulcast: true},
		{name: "S3T2h", spatial: 3, temporal: 2, smallStep: true, simulcast: true},
		{name: "S3T3", spatial: 3, temporal: 3, simulcast: true},
		{name: "S3T3h", spatial: 3, temporal: 3, smallStep: true, simulcast: true},
	} {
		mode, ok := scalabilityModeNamed(tc.name)
		if !ok {
			t.Fatalf("no scalability mode named %q", tc.name)
		}
		if got := mode.String(); got != tc.name {
			t.Fatalf("%q String()=%q", tc.name, got)
		}
		spatial, temporal, key, ok := mode.Layers()
		if !ok || spatial != tc.spatial || temporal != tc.temporal || key != tc.key {
			t.Fatalf("%q Layers()=%d,%d,%v,%v", tc.name, spatial, temporal, key, ok)
		}
		if mode.UsesSmallResolutionStep() != tc.smallStep ||
			mode.IsSimulcast() != tc.simulcast ||
			mode.UsesKeyFrameInterLayerDependency() != tc.key ||
			mode.UsesKeyFrameInterLayerDependencyShift() != tc.keyShift {
			t.Fatalf("%q flags small=%v sim=%v key=%v shift=%v", tc.name,
				mode.UsesSmallResolutionStep(),
				mode.IsSimulcast(),
				mode.UsesKeyFrameInterLayerDependency(),
				mode.UsesKeyFrameInterLayerDependencyShift())
		}
		cfg, err := SetWebRTCSVCConfig(Config{
			Resolution:  Resolution{Width: 1280, Height: 720},
			Scalability: mode,
		}, 0, 0)
		if err != nil {
			t.Fatalf("SetWebRTCSVCConfig(%q): %v", tc.name, err)
		}
		if cfg.SpatialLayerCount != tc.spatial || cfg.TemporalLayerCount != tc.temporal {
			t.Fatalf("%q normalized layers=%d,%d", tc.name, cfg.SpatialLayerCount, cfg.TemporalLayerCount)
		}
		structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
		if err != nil {
			t.Fatalf("WebRTCFrameDependencyStructureForConfig(%q): %v", tc.name, err)
		}
		wantTemplates := webRTCTestReferenceTemplateNum(mode, tc.spatial, tc.temporal)
		if structure.NumDecodeTargets != tc.spatial*tc.temporal ||
			structure.NumChains != tc.spatial ||
			structure.TemplateNum != wantTemplates ||
			structure.ResolutionNum != tc.spatial {
			t.Fatalf("%q structure shape=%+v", tc.name, structure)
		}
	}
}

func webRTCTestReferenceTemplateNum(mode ScalabilityMode, spatial uint8, temporal uint8) uint8 {
	switch mode {
	case ScalabilityModeL1T2:
		return 3
	case ScalabilityModeL1T3:
		return 5
	case ScalabilityModeL2T1, ScalabilityModeL2T1h, ScalabilityModeL2T1_KEY,
		ScalabilityModeS2T1, ScalabilityModeS2T1h:
		return 4
	case ScalabilityModeL2T2, ScalabilityModeL2T2h, ScalabilityModeL2T2_KEY,
		ScalabilityModeS2T2, ScalabilityModeS2T2h:
		return 6
	case ScalabilityModeL2T2_KEY_SHIFT:
		return 7
	case ScalabilityModeL2T3, ScalabilityModeL2T3h, ScalabilityModeL2T3_KEY,
		ScalabilityModeS2T3, ScalabilityModeS2T3h:
		return 10
	case ScalabilityModeL3T1, ScalabilityModeL3T1h, ScalabilityModeL3T1_KEY,
		ScalabilityModeS3T1, ScalabilityModeS3T1h:
		return 6
	case ScalabilityModeL3T2, ScalabilityModeL3T2h, ScalabilityModeL3T2_KEY,
		ScalabilityModeS3T2, ScalabilityModeS3T2h:
		return 9
	case ScalabilityModeL3T3, ScalabilityModeL3T3h, ScalabilityModeL3T3_KEY,
		ScalabilityModeS3T3, ScalabilityModeS3T3h:
		return 15
	default:
		return spatial * temporal
	}
}

func TestAppendWebRTCScalabilityModesMatchesPinnedLibWebRTC(t *testing.T) {
	want := pinnedLibWebRTCScalabilityModes(t)
	prefix := []ScalabilityMode{ScalabilityModeL3T3}
	modes := AppendWebRTCScalabilityModes(prefix)
	if WebRTCScalabilityModeCount() != len(want) {
		t.Fatalf("WebRTCScalabilityModeCount=%d want %d", WebRTCScalabilityModeCount(), len(want))
	}
	if len(modes) != len(want)+1 {
		t.Fatalf("len=%d want %d", len(modes), len(want)+1)
	}
	if modes[0] != ScalabilityModeL3T3 {
		t.Fatalf("prefix mutated to %s", modes[0])
	}
	seen := make(map[ScalabilityMode]bool, len(want))
	for i, wantMode := range want {
		mode := modes[i+1]
		if mode != wantMode {
			t.Fatalf("mode[%d]=%s want %s", i+1, mode, wantMode)
		}
		if seen[mode] {
			t.Fatalf("duplicate mode %s", mode)
		}
		seen[mode] = true
	}
}

// scalabilityModeNamed looks a mode up by its String name.
func scalabilityModeNamed(name string) (ScalabilityMode, bool) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		if mode.String() == name {
			return mode, true
		}
	}
	return 0, false
}

func pinnedLibWebRTCScalabilityModes(t *testing.T) []ScalabilityMode {
	t.Helper()
	path := filepath.Join("testdata", "webrtc", "scalability_mode.h")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read pinned libwebrtc scalability mode source: %v", err)
	}

	var modes []ScalabilityMode
	inCatalog := false
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.Contains(line, "kAllScalabilityModes[]") {
			inCatalog = true
			continue
		}
		if !inCatalog {
			continue
		}
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "};") {
			break
		}
		const token = "ScalabilityMode::k"
		idx := strings.Index(line, token)
		if idx < 0 {
			continue
		}
		name := line[idx+len(token):]
		if end := strings.IndexAny(name, ", \t\r/"); end >= 0 {
			name = name[:end]
		}
		mode, ok := scalabilityModeNamed(name)
		if !ok {
			t.Fatalf("pinned libwebrtc scalability mode %q is not implemented", name)
		}
		modes = append(modes, mode)
	}
	if len(modes) == 0 {
		t.Fatalf("no pinned libwebrtc scalability modes found in %s", path)
	}
	return modes
}

func TestWebRTCScalabilityModeIDC(t *testing.T) {
	seen := make(map[ScalabilityMode]bool, scalabilityModeCount)
	for _, tc := range []struct {
		mode ScalabilityMode
		idc  uint8
		ok   bool
	}{
		{mode: ScalabilityModeL1T1},
		{mode: ScalabilityModeL1T2, idc: obu.ScalabilityModeL1T2, ok: true},
		{mode: ScalabilityModeL1T3, idc: obu.ScalabilityModeL1T3, ok: true},
		{mode: ScalabilityModeL2T1, idc: obu.ScalabilityModeL2T1, ok: true},
		{mode: ScalabilityModeL2T1h, idc: obu.ScalabilityModeL2T1h, ok: true},
		{mode: ScalabilityModeL2T1_KEY},
		{mode: ScalabilityModeL2T2, idc: obu.ScalabilityModeL2T2, ok: true},
		{mode: ScalabilityModeL2T2h, idc: obu.ScalabilityModeL2T2h, ok: true},
		{mode: ScalabilityModeL2T2_KEY, idc: obu.ScalabilityModeL3T2_KEY, ok: true},
		{mode: ScalabilityModeL2T2_KEY_SHIFT, idc: obu.ScalabilityModeL3T2_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL2T3, idc: obu.ScalabilityModeL2T3, ok: true},
		{mode: ScalabilityModeL2T3h, idc: obu.ScalabilityModeL2T3h, ok: true},
		{mode: ScalabilityModeL2T3_KEY, idc: obu.ScalabilityModeL3T3_KEY, ok: true},
		{mode: ScalabilityModeL2T3_KEY_SHIFT, idc: obu.ScalabilityModeL3T3_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL3T1, idc: obu.ScalabilityModeL3T1, ok: true},
		{mode: ScalabilityModeL3T1h},
		{mode: ScalabilityModeL3T1_KEY},
		{mode: ScalabilityModeL3T2, idc: obu.ScalabilityModeL3T2, ok: true},
		{mode: ScalabilityModeL3T2h},
		{mode: ScalabilityModeL3T2_KEY, idc: obu.ScalabilityModeL4T5_KEY, ok: true},
		{mode: ScalabilityModeL3T2_KEY_SHIFT, idc: obu.ScalabilityModeL4T5_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL3T3, idc: obu.ScalabilityModeL3T3, ok: true},
		{mode: ScalabilityModeL3T3h},
		{mode: ScalabilityModeL3T3_KEY, idc: obu.ScalabilityModeL4T7_KEY, ok: true},
		{mode: ScalabilityModeL3T3_KEY_SHIFT, idc: obu.ScalabilityModeL4T7_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeS2T1, idc: obu.ScalabilityModeS2T1, ok: true},
		{mode: ScalabilityModeS2T1h, idc: obu.ScalabilityModeS2T1h, ok: true},
		{mode: ScalabilityModeS2T2, idc: obu.ScalabilityModeS2T2, ok: true},
		{mode: ScalabilityModeS2T2h, idc: obu.ScalabilityModeS2T2h, ok: true},
		{mode: ScalabilityModeS2T3, idc: obu.ScalabilityModeS2T3, ok: true},
		{mode: ScalabilityModeS2T3h, idc: obu.ScalabilityModeS2T3h, ok: true},
		{mode: ScalabilityModeS3T1, idc: obu.ScalabilityModeS3T1, ok: true},
		{mode: ScalabilityModeS3T1h},
		{mode: ScalabilityModeS3T2, idc: obu.ScalabilityModeS3T2, ok: true},
		{mode: ScalabilityModeS3T2h},
		{mode: ScalabilityModeS3T3, idc: obu.ScalabilityModeS3T3, ok: true},
		{mode: ScalabilityModeS3T3h},
	} {
		got, ok := WebRTCScalabilityModeIDC(tc.mode)
		if got != tc.idc || ok != tc.ok {
			t.Fatalf("WebRTCScalabilityModeIDC(%s)=%d,%v want %d,%v", tc.mode, got, ok, tc.idc, tc.ok)
		}
		got, ok = tc.mode.AV1ScalabilityModeIDC()
		if got != tc.idc || ok != tc.ok {
			t.Fatalf("%s.AV1ScalabilityModeIDC()=%d,%v want %d,%v", tc.mode, got, ok, tc.idc, tc.ok)
		}
		seen[tc.mode] = true
	}
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		if !seen[mode] {
			t.Fatalf("missing IDC test coverage for %s", mode)
		}
	}
	if got, ok := WebRTCScalabilityModeIDC(ScalabilityMode(scalabilityModeCount)); got != 0 || ok {
		t.Fatalf("invalid mode IDC=%d,%v want 0,false", got, ok)
	}
}

func TestDefaultScalabilityMode(t *testing.T) {
	mode, ok := DefaultScalabilityMode(3, 2)
	if !ok || mode != ScalabilityModeL2T3_KEY {
		t.Fatalf("DefaultScalabilityMode(3,2) = %v,%v; want L2T3_KEY,true", mode, ok)
	}
	mode, ok = DefaultScalabilityMode(1, 1)
	if !ok || mode != ScalabilityModeL1T1 {
		t.Fatalf("DefaultScalabilityMode(1,1) = %v,%v; want L1T1,true", mode, ok)
	}
	if _, ok := DefaultScalabilityMode(4, 2); ok {
		t.Fatal("DefaultScalabilityMode accepted unsupported temporal layer count")
	}
}

func TestLimitedSpatialLayers(t *testing.T) {
	for _, tc := range []struct {
		res  Resolution
		want uint8
	}{
		{res: Resolution{Width: 1280, Height: 720}, want: 3},
		{res: Resolution{Width: 640, Height: 360}, want: 2},
		{res: Resolution{Width: 256, Height: 144}, want: 1},
		{res: Resolution{Width: 720, Height: 1280}, want: 3},
	} {
		if got := LimitedSpatialLayers(tc.res); got != tc.want {
			t.Fatalf("LimitedSpatialLayers(%+v) = %d; want %d", tc.res, got, tc.want)
		}
	}
}

func TestSetWebRTCSVCConfig(t *testing.T) {
	base := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 350,
	}

	got, err := SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L1T1: %v", err)
	}
	if got.Scalability != ScalabilityModeL1T1 || got.SpatialLayerCount != 1 || got.TemporalLayerCount != 1 {
		t.Fatalf("L1T1 config mode=%v spatial=%d temporal=%d", got.Scalability, got.SpatialLayerCount, got.TemporalLayerCount)
	}
	if !got.SpatialLayers[0].Active || got.SpatialLayers[1].Active {
		t.Fatalf("unexpected active layers: %+v", got.SpatialLayers)
	}
	if got.SpatialLayers[0].MinBitrateKbps != 100 || got.SpatialLayers[0].MaxBitrateKbps != 500 || got.SpatialLayers[0].TargetBitrateKbps != 350 {
		t.Fatalf("single-layer bitrate not copied: %+v", got.SpatialLayers[0])
	}

	midpoint := base
	midpoint.TargetBitrateKbps = 0
	got, err = SetWebRTCSVCConfig(midpoint, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L1T1 midpoint: %v", err)
	}
	if got.SpatialLayers[0].TargetBitrateKbps != 300 {
		t.Fatalf("single-layer default target=%d want 300", got.SpatialLayers[0].TargetBitrateKbps)
	}

	base.Scalability = ScalabilityModeL2T2
	got, err = SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L2T2: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 320, Height: 180}) || got.SpatialLayers[1].Resolution != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("L2T2 resolutions = %+v %+v", got.SpatialLayers[0].Resolution, got.SpatialLayers[1].Resolution)
	}
	if got.SpatialLayers[0].MinBitrateKbps != 20 || got.SpatialLayers[0].MaxBitrateKbps != 142 {
		t.Fatalf("L2T2 base bitrate = %+v", got.SpatialLayers[0])
	}
	if got.SpatialLayers[1].MinBitrateKbps != 135 || got.SpatialLayers[1].MaxBitrateKbps != 418 {
		t.Fatalf("L2T2 top bitrate = %+v", got.SpatialLayers[1])
	}
}

func TestSetWebRTCSVCConfigExplicitColorBitDepthOverridesStaleTopLevel(t *testing.T) {
	base := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 350,
	}
	normalized, err := SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig base: %v", err)
	}
	if normalized.BitDepth != 8 {
		t.Fatalf("base normalized bit depth=%d want 8", normalized.BitDepth)
	}

	reconfigured := normalized
	reconfigured.Profile = Profile0
	reconfigured.ColorConfigSet = true
	reconfigured.ColorConfig = SequenceColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}
	got, err := SetWebRTCSVCConfig(reconfigured, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig high-bit-depth mono reconfigure: %v", err)
	}
	if got.BitDepth != 10 || got.ColorConfig.BitDepth != 10 || !got.ColorConfig.MonoChrome {
		t.Fatalf("reconfigured color=%+v top bitDepth=%d want 10-bit monochrome", got.ColorConfig, got.BitDepth)
	}
}

func TestSetWebRTCSVCConfigPreservesSpatialLayerBitrates(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1000,
		TargetBitrateKbps: 650,
		Scalability:       ScalabilityModeS2T3,
	}
	cfg.SpatialLayers[0].MinBitrateKbps = 120
	cfg.SpatialLayers[0].MaxBitrateKbps = 520
	cfg.SpatialLayers[0].TargetBitrateKbps = 260
	cfg.SpatialLayers[1].MinBitrateKbps = 340
	cfg.SpatialLayers[1].MaxBitrateKbps = 1500
	cfg.SpatialLayers[1].TargetBitrateKbps = 900

	got, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig explicit layers: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 320, Height: 180}) ||
		got.SpatialLayers[0].MinBitrateKbps != 120 ||
		got.SpatialLayers[0].MaxBitrateKbps != 520 ||
		got.SpatialLayers[0].TargetBitrateKbps != 260 {
		t.Fatalf("base layer = %+v", got.SpatialLayers[0])
	}
	if got.SpatialLayers[1].Resolution != (Resolution{Width: 640, Height: 360}) ||
		got.SpatialLayers[1].MinBitrateKbps != 340 ||
		got.SpatialLayers[1].MaxBitrateKbps != 1500 ||
		got.SpatialLayers[1].TargetBitrateKbps != 900 {
		t.Fatalf("top layer = %+v", got.SpatialLayers[1])
	}

	if _, err := SetWebRTCSVCConfig(got, 0, 0); err != nil {
		t.Fatalf("SetWebRTCSVCConfig normalized explicit layers: %v", err)
	}
}

func TestSetWebRTCSVCConfigSmallStepAndReduction(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 1500, Height: 900},
		Scalability: ScalabilityModeL2T1h,
	}
	got, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L2T1h: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 1000, Height: 600}) {
		t.Fatalf("L2T1h base resolution = %+v", got.SpatialLayers[0].Resolution)
	}

	cfg = Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL3T3,
	}
	got, err = SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L3T3 reduction: %v", err)
	}
	if got.Scalability != ScalabilityModeL2T3 {
		t.Fatalf("small input reduced to %v; want L2T3", got.Scalability)
	}

	cfg = Config{
		Resolution:  Resolution{Width: 256, Height: 144},
		Scalability: ScalabilityModeL3T3_KEY,
	}
	got, err = SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L3T3_KEY single-layer reduction: %v", err)
	}
	if got.Scalability != ScalabilityModeL1T3 {
		t.Fatalf("tiny input reduced to %v; want L1T3", got.Scalability)
	}
}

func TestSetWebRTCSVCConfigRejectsInvalidConfig(t *testing.T) {
	valid := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
	}
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{name: "wide", cfg: withResolution(valid, Resolution{Width: WebRTCMaxDimension + 1, Height: 360})},
		{name: "area", cfg: withResolution(valid, Resolution{Width: 65536, Height: 65536})},
		{name: "subfps", cfg: withFramerate(valid, Rational{Num: 1, Den: 2})},
		{name: "bitrate", cfg: withBitrate(valid, 0, WebRTCMaxBitrateKbps+1, 0)},
		{name: "target-outside", cfg: withBitrate(valid, 100, 500, 600)},
		{name: "layer-target-outside", cfg: withSpatialLayerBitrate(withScalability(valid, ScalabilityModeS2T2), 1, 400, 800, 900)},
		{name: "cbr-quantizer", cfg: withQuantizer(valid, RateControlCBR, 12)},
		{name: "cqp-quantizer", cfg: withQuantizer(valid, RateControlCQP, WebRTCMaxQuantizer+1)},
	} {
		if _, err := SetWebRTCSVCConfig(tc.cfg, 0, 0); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("%s err = %v; want ErrInvalidConfig", tc.name, err)
		}
	}
}

func withResolution(cfg Config, resolution Resolution) Config {
	cfg.Resolution = resolution
	return cfg
}

func withFramerate(cfg Config, framerate Rational) Config {
	cfg.MaxFramerate = framerate
	return cfg
}

func withBitrate(cfg Config, minBitrateKbps int32, maxBitrateKbps int32, targetBitrateKbps int32) Config {
	cfg.MinBitrateKbps = minBitrateKbps
	cfg.MaxBitrateKbps = maxBitrateKbps
	cfg.TargetBitrateKbps = targetBitrateKbps
	return cfg
}

func withScalability(cfg Config, mode ScalabilityMode) Config {
	cfg.Scalability = mode
	return cfg
}

func withSpatialLayerBitrate(cfg Config, layer int, minBitrateKbps int32, maxBitrateKbps int32, targetBitrateKbps int32) Config {
	cfg.SpatialLayers[layer].MinBitrateKbps = minBitrateKbps
	cfg.SpatialLayers[layer].MaxBitrateKbps = maxBitrateKbps
	cfg.SpatialLayers[layer].TargetBitrateKbps = targetBitrateKbps
	return cfg
}

func withQuantizer(cfg Config, rc RateControlMode, quantizer uint8) Config {
	cfg.RateControl = rc
	cfg.Quantizer = quantizer
	return cfg
}

func TestValidateTemporalUnitFrames(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}

	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	out, err := ValidateTemporalUnitFrames(frames[:], state, RateControlCBR)
	if err != nil {
		t.Fatalf("ValidateTemporalUnitFrames valid delta: %v", err)
	}
	if !out.Valid[1] || out.Resolutions[1] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("updated buffer state = %+v", out)
	}

	tu := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			SpatialID:       0,
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	out, err = ValidateTemporalUnitFrames(tu[:], ReferenceBufferState{}, RateControlCBR)
	if err != nil {
		t.Fatalf("ValidateTemporalUnitFrames key plus upper layer: %v", err)
	}
	if !out.Valid[0] || !out.Valid[1] {
		t.Fatalf("temporal unit updates missing: %+v", out)
	}
}

func TestValidateTemporalUnitFramesRejectsInvalidRefs(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}
	state.Valid[1] = true
	state.Resolutions[1] = Resolution{Width: 640, Height: 360}

	duplicate := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0, 0},
			ReferenceCount:   2,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(duplicate[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("duplicate refs err = %v; want ErrInvalidFrame", err)
	}

	badScale := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 1000, Height: 600},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(badScale[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad scaling err = %v; want ErrInvalidFrame", err)
	}

	reset := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(reset[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("stale ref after keyframe err = %v; want ErrInvalidFrame", err)
	}
}

func TestSupportedResolutionScaling(t *testing.T) {
	factor, ok := SupportedResolutionScaling(
		Resolution{Width: 320, Height: 180},
		Resolution{Width: 640, Height: 360},
	)
	if !ok || factor != (Rational{Num: 2, Den: 1}) {
		t.Fatalf("SupportedResolutionScaling = %+v,%v; want 2/1,true", factor, ok)
	}
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 960, Height: 540},
		Resolution{Width: 1440, Height: 810},
	)
	if !ok || factor != (Rational{Num: 3, Den: 2}) {
		t.Fatalf("SupportedResolutionScaling 3:2 = %+v,%v; want 3/2,true", factor, ok)
	}
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 1440, Height: 810},
		Resolution{Width: 960, Height: 540},
	)
	if !ok || factor != (Rational{Num: 2, Den: 3}) {
		t.Fatalf("SupportedResolutionScaling 2:3 = %+v,%v; want 2/3,true", factor, ok)
	}
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 853, Height: 480},
		Resolution{Width: 1280, Height: 720},
	)
	if !ok || factor != (Rational{Num: 3, Den: 2}) {
		t.Fatalf("SupportedResolutionScaling truncated 3:2 = %+v,%v; want 3/2,true", factor, ok)
	}
	if _, ok := SupportedResolutionScaling(
		Resolution{Width: 640, Height: 360},
		Resolution{Width: 1000, Height: 600},
	); ok {
		t.Fatal("SupportedResolutionScaling accepted unsupported ratio")
	}
}

func TestValidateTemporalUnitFramesAllocs(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(frames[:], state, RateControlCBR); err != nil {
		t.Fatalf("preflight failed: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = ValidateTemporalUnitFrames(frames[:], state, RateControlCBR)
	})
	if allocs != 0 {
		t.Fatalf("ValidateTemporalUnitFrames allocs = %f; want 0", allocs)
	}
}

func testCRRateControl(bestQ, worstQ uint8) *svtRateControl {
	var rc svtRateControl
	rc.init(1_000_000, 30, bestQ, worstQ, 1280, 720, rcBuffers{})
	return &rc
}

func TestCyclicComputeDeltaQ(t *testing.T) {
	rc := testCRRateControl(1, 255)
	// A higher rate target lowers q; the floor is -60% of q.
	for _, q := range []int{40, 100, 180, 250} {
		d := cyclicComputeDeltaQ(rc, q, 1.75, false)
		if d >= 0 {
			t.Fatalf("q=%d rate ratio 1.75 delta %d, want negative", q, d)
		}
		if d < -crMaxQDeltaPerc*q/100 {
			t.Fatalf("q=%d delta %d below the -60%% floor", q, d)
		}
	}
	if d := cyclicComputeDeltaQ(rc, 120, 1.0, false); d != 0 {
		t.Fatalf("unit rate ratio delta %d, want 0", d)
	}
}

func TestCyclicRefreshWindowAndGates(t *testing.T) {
	var cr cyclicRefresh
	rc := testCRRateControl(20, 200)
	rc.avgFrameQIndex[svtFrameInter] = 150
	percent := 20 + rc.percentRefreshAdjustment
	const w, h = 1280, 720 // 20x12 superblocks
	f := svtCRFrame{flat: true, hierLevels: 2, avgFrameLowMotion: 80}
	grid, stats := cr.plan(rc, f, 150, w, h, nil)
	if grid == nil || !stats.apply {
		t.Fatal("cyclic refresh did not apply")
	}
	boosted := 0
	for i, q := range grid {
		switch {
		case q < 150:
			boosted++
			if i >= 240*percent/100 {
				t.Fatalf("superblock %d boosted outside the first window", i)
			}
		case q != 150:
			t.Fatalf("superblock %d q=%d, want base 150", i, q)
		}
	}
	if want := 240 * percent / 100; boosted != want || stats.seg1 != want {
		t.Fatalf("boosted %d superblocks (seg1 %d), want %d", boosted, stats.seg1, want)
	}
	if cr.sbEnd != 240*percent/100 {
		t.Fatalf("window end %d", cr.sbEnd)
	}
	// avg_frame_low_motion under 50 turns it off.
	f.avgFrameLowMotion = 30
	if g, _ := cr.plan(rc, f, 150, w, h, nil); g != nil {
		t.Fatal("cyclic refresh applied on high-motion content")
	}
	// Non-base layers of a layered structure do not refresh.
	f = svtCRFrame{hierLevels: 2, temporalLayer: 1, avgFrameLowMotion: 80}
	if g, _ := cr.plan(rc, f, 150, w, h, nil); g != nil {
		t.Fatal("cyclic refresh applied on a non-base layer")
	}
}

// encodeCyclicOracle encodes a single-layer CBR sequence (the configuration
// cyclic refresh runs in) with the fused walk, the serial split walk or the
// split walk on wavefront lanes.
func encodeCyclicOracle(t *testing.T, frames []SourceFrame420, w, h int, split bool, lanes int) ([][]byte, int) {
	t.Helper()
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 300_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.forceSplit = split
	enc.wavefrontLanes = lanes
	enc.SetTileColumns(1)
	out := make([][]byte, 0, len(frames))
	deltaFrames := 0
	for i, f := range frames {
		data, _, err := enc.Encode(f, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if enc.pc.st.sbQIndex != nil {
			deltaFrames++
		}
		out = append(out, append([]byte(nil), data...))
	}
	return out, deltaFrames
}

func TestCyclicRefreshSplitMatchesFused(t *testing.T) {
	const w, h = 320, 192
	frames := syntheticMotionFrames(10, w, h)
	fused, deltaFrames := encodeCyclicOracle(t, frames, w, h, false, 0)
	if deltaFrames == 0 {
		t.Fatal("no frame signalled delta_q")
	}
	for _, cfg := range []struct {
		name  string
		lanes int
	}{{"split", 0}, {"wavefront2", 2}, {"wavefront3", 3}} {
		t.Run(cfg.name, func(t *testing.T) {
			got, _ := encodeCyclicOracle(t, frames, w, h, true, cfg.lanes)
			for i := range fused {
				if !bytes.Equal(fused[i], got[i]) {
					t.Fatalf("frame %d differs from fused (%d vs %d bytes)", i, len(fused[i]), len(got[i]))
				}
			}
		})
	}
}

func TestVideoEncoderDecisionStatsDisabledByDefault(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil {
		t.Fatal(err)
	}
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 1), false); err != nil {
		t.Fatal(err)
	}
	if got := enc.DecisionStats(); got != (EncoderDecisionStats{}) {
		t.Fatalf("disabled stats=%+v want zero", got)
	}
}

func TestVideoEncoderDecisionStatsAccumulatesKeyAndInter(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	enc.SetTileColumns(1)
	enc.SetDecisionStatsEnabled(true)
	if _, key, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil || !key {
		t.Fatalf("key encode key=%v err=%v", key, err)
	}
	if _, key, err := enc.Encode(decisionStatsFrame(64, 64, 1), false); err != nil || key {
		t.Fatalf("inter encode key=%v err=%v", key, err)
	}
	stats := enc.DecisionStats()
	if stats.Frames != 2 || stats.Keyframes != 1 || stats.InterFrames != 1 {
		t.Fatalf("frame counts=%+v", stats)
	}
	if stats.Tiles != 2 {
		t.Fatalf("tiles=%d want 2", stats.Tiles)
	}
	if stats.PartitionDecisions == 0 || stats.Blocks == 0 {
		t.Fatalf("missing decision counts=%+v", stats)
	}
	if stats.IntraBlocks == 0 || stats.InterBlocks == 0 {
		t.Fatalf("block kind counts=%+v", stats)
	}
	if stats.CodedBlocks+stats.SkipBlocks != stats.Blocks {
		t.Fatalf("coded+skip=%d blocks=%d", stats.CodedBlocks+stats.SkipBlocks, stats.Blocks)
	}
	if stats.LumaTXBs == 0 || stats.TXTypes[EncoderDecisionTransformDCTDCT] == 0 {
		t.Fatalf("missing tx counts=%+v", stats)
	}
}

func TestVideoEncoderDecisionStatsReset(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 90)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	enc.SetDecisionStatsEnabled(true)
	if _, _, err := enc.Encode(decisionStatsFrame(64, 64, 0), false); err != nil {
		t.Fatal(err)
	}
	if enc.DecisionStats().Frames == 0 {
		t.Fatal("expected stats before reset")
	}
	enc.ResetDecisionStats()
	if got := enc.DecisionStats(); got != (EncoderDecisionStats{}) {
		t.Fatalf("reset stats=%+v want zero", got)
	}
}

func decisionStatsFrame(w, h, n int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 48 + (x+y+n*3)%96
			if x >= 16+n && x < 32+n && y >= 20 && y < 36 {
				v = 220
			}
			f.Y[y*w+x] = byte(v)
		}
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 132
	}
	return f
}

func TestAppendDeltaParamsPayloadBaseQZero(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{}
	payload, parsed := appendAndParseDeltaParams(t, size, quant, DeltaParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.DeltaQPresent || parsed.BitsRead != 0 {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadQOnly(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	_, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 2 || parsed.DeltaLFPresent {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadLoopFilterMulti(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1, DeltaLFPresent: true, DeltaLFResLog2: 3, DeltaLFMulti: true}
	_, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 1 ||
		!parsed.DeltaLFPresent || parsed.DeltaLFResLog2 != 3 || !parsed.DeltaLFMulti {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadIntrabcSkipsLoopFilter(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff, AllowIntrabc: true}
	quant := QuantizationParams{BaseQIdx: 50}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	payload, parsed := appendAndParseDeltaParams(t, size, quant, delta)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 2 || parsed.DeltaLFPresent {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsInterPayload(t *testing.T) {
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	quant := QuantizationParams{BaseQIdx: 31}
	delta := DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1, DeltaLFPresent: true, DeltaLFResLog2: 2}
	var buf [2]byte
	payload, err := appendDeltaParamsInterPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("appendDeltaParamsInterPayload: %v", err)
	}
	parsed, err := parser.ParseDeltaParams(payload, parser.FrameSize{}, parser.QuantizationParams{BaseQIdx: 31}, parser.SegmentationParams{})
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	if !parsed.DeltaLFPresent || parsed.DeltaLFResLog2 != 2 || parsed.DeltaLFMulti {
		t.Fatalf("parsed inter delta=%+v", parsed)
	}
}

func TestAppendDeltaParamsPayloadRejectsInvalid(t *testing.T) {
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	quantZero := QuantizationParams{}
	quant := QuantizationParams{BaseQIdx: 50}
	cases := [...]struct {
		quant QuantizationParams
		delta DeltaParams
	}{
		{quant: quantZero, delta: DeltaParams{DeltaQPresent: true}},
		{quant: quant, delta: DeltaParams{DeltaQResLog2: 1}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaQResLog2: 4}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFPresent: true, DeltaLFResLog2: 4}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFResLog2: 1}},
		{quant: quant, delta: DeltaParams{DeltaQPresent: true, DeltaLFMulti: true}},
	}
	var buf [2]byte
	for _, tc := range cases {
		if _, err := appendDeltaParamsPayload(buf[:0], size, tc.quant, tc.delta); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendDeltaParamsPayload(%+v,%+v) err=%v want ErrInvalidFrame", tc.quant, tc.delta, err)
		}
	}
	intrabc := size
	intrabc.AllowIntrabc = true
	if _, err := appendDeltaParamsPayload(buf[:0], intrabc, quant, DeltaParams{DeltaQPresent: true, DeltaLFPresent: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intrabc delta lf err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseDeltaParams(t *testing.T, size IntraFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, parser.DeltaParams) {
	t.Helper()
	var buf [2]byte
	payload, err := appendDeltaParamsPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("appendDeltaParamsPayload: %v", err)
	}
	parsedSize := parser.FrameSize{AllowIntrabc: size.AllowIntrabc}
	parsedQuant := parser.QuantizationParams{BaseQIdx: quant.BaseQIdx}
	parsed, err := parser.ParseDeltaParams(payload, parsedSize, parsedQuant, parser.SegmentationParams{})
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	return payload, parsed
}

func TestAppendWebRTCDependencyDescriptorAttachedStructure(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:    10,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs

	size, err := WebRTCDependencyDescriptorSize(structure, info, true)
	if err != nil {
		t.Fatalf("WebRTCDependencyDescriptorSize: %v", err)
	}
	var buf [64]byte
	out, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, true)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(out) != size {
		t.Fatalf("len=%d want %d", len(out), size)
	}
	if out[0] != 0xc0 || out[1] != 0x00 || out[2] != 0x0a || out[3] != 0x80 {
		t.Fatalf("descriptor prefix=% x", out[:4])
	}
}

func TestAppendWebRTCDependencyDescriptorCustomFrameDiff(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 22
	info.DTIs = structure.Templates[5].DTIs

	var buf [16]byte
	out, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("descriptor=% x len=%d want 5", out, len(out))
	}
	if out[0] != 0xc5 || out[1] != 0x00 || out[2] != 0x17 || out[3] != 0x12 {
		t.Fatalf("descriptor prefix=% x", out[:4])
	}
}

func TestAppendWebRTCDependencyDescriptorSingleChainStructure(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:    77,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     1,
	}
	info.DTIs[0] = DecodeTargetSwitch
	var buf [32]byte
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, false, true); err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor single-chain: %v", err)
	}
}

func TestAppendWebRTCDependencyDescriptorL1T3TemplateDTIs(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T3,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	for _, tc := range [...]struct {
		temporalID uint8
		template   uint8
		frameID    uint64
		dependency uint64
		hint       bool
	}{
		{temporalID: 0, template: 0, frameID: 100},
		{temporalID: 0, template: 1, frameID: 104, dependency: 100},
		{temporalID: 1, template: 2, frameID: 106, dependency: 104},
		{temporalID: 2, template: 3, frameID: 107, dependency: 106},
		{temporalID: 2, template: 4, frameID: 109, dependency: 108, hint: true},
	} {
		info := WebRTCGenericFrameInfo{
			FrameID:    tc.frameID,
			TemporalID: tc.temporalID,
			DTINum:     structure.NumDecodeTargets,
		}
		info.DTIs = structure.Templates[tc.template].DTIs
		if tc.dependency != 0 {
			info.DependencyNum = 1
			info.Dependencies[0] = tc.dependency
		}
		if tc.hint {
			info.TemplateIndexHint = tc.template
			info.TemplateIndexHintSet = true
		}

		match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
		if err != nil {
			t.Fatalf("match template %d: %v", tc.template, err)
		}
		if match.templateIndex != tc.template || match.needCustomDTIs || match.needCustomDiffs {
			t.Fatalf("match template %d = %+v", tc.template, match)
		}
		var buf [16]byte
		if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); err != nil {
			t.Fatalf("AppendWebRTCDependencyDescriptor template %d: %v", tc.template, err)
		}
	}
}

func TestAppendWebRTCDependencyDescriptorL2T2KeyShiftDuplicateTemplates(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	for _, tc := range [...]struct {
		name          string
		frameID       uint64
		spatialID     uint8
		temporalID    uint8
		dependency    uint64
		templateIndex uint8
	}{
		{name: "s0t0-diff2", frameID: 102, spatialID: 0, temporalID: 0, dependency: 100, templateIndex: 1},
		{name: "s0t0-diff4", frameID: 106, spatialID: 0, temporalID: 0, dependency: 102, templateIndex: 2},
		{name: "s1t0-diff1", frameID: 101, spatialID: 1, temporalID: 0, dependency: 100, templateIndex: 4},
		{name: "s1t0-diff4", frameID: 105, spatialID: 1, temporalID: 0, dependency: 101, templateIndex: 5},
	} {
		info := WebRTCGenericFrameInfo{
			FrameID:       tc.frameID,
			SpatialID:     tc.spatialID,
			TemporalID:    tc.temporalID,
			DependencyNum: 1,
			DTINum:        structure.NumDecodeTargets,
		}
		info.Dependencies[0] = tc.dependency
		info.DTIs = structure.Templates[tc.templateIndex].DTIs
		match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
		if err != nil {
			t.Fatalf("%s match: %v", tc.name, err)
		}
		if match.templateIndex != tc.templateIndex || match.needCustomDTIs || match.needCustomDiffs {
			t.Fatalf("%s match=%+v", tc.name, match)
		}
		var buf [16]byte
		if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); err != nil {
			t.Fatalf("%s AppendWebRTCDependencyDescriptor: %v", tc.name, err)
		}
	}
}

func TestAppendWebRTCDependencyDescriptorRejectsInvalid(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 23
	info.DTIs = structure.Templates[3].DTIs
	var buf [16]byte
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("invalid dependency err=%v want ErrInvalidFrame", err)
	}

	info.Dependencies[0] = 22
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0:4], structure, info, true, true, false); !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
}

func TestAppendWebRTCDependencyDescriptorAllocs(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 22
	info.DTIs = structure.Templates[3].DTIs
	var buf [16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCDependencyDescriptorSize(structure, info, false)
		_, _ = AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	})
	if allocs != 0 {
		t.Fatalf("dependency descriptor allocs=%f want 0", allocs)
	}
}

func TestLibaomSVCRefFrameConfigForFrame(t *testing.T) {
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3, 1},
		ReferenceCount:   2,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}

	got, err := LibaomSVCRefFrameConfigForFrame(settings)
	if err != nil {
		t.Fatalf("LibaomSVCRefFrameConfigForFrame: %v", err)
	}
	if got.Reference[0] != 1 || got.Reference[1] != 1 || got.Reference[2] != 0 {
		t.Fatalf("reference flags = %+v", got.Reference)
	}
	if got.RefIdx[0] != 3 || got.RefIdx[1] != 1 || got.RefIdx[2] != 3 || got.RefIdx[6] != 3 {
		t.Fatalf("ref idx = %+v", got.RefIdx)
	}
	if got.Refresh[4] != 1 || got.Refresh[0] != 0 {
		t.Fatalf("refresh = %+v", got.Refresh)
	}
}

func TestLibaomSVCRefFrameConfigForFrameRejectsDuplicate(t *testing.T) {
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3, 3},
		ReferenceCount:   2,
	}
	if _, err := LibaomSVCRefFrameConfigForFrame(settings); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("duplicate reference err=%v want ErrInvalidFrame", err)
	}
}

func TestWebRTCGenericFrameInfoForFrame(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[3] = true
	state.FrameIDs[3] = 100
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		SpatialID:        1,
		TemporalID:       1,
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3},
		ReferenceCount:   1,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}

	info, out, err := WebRTCGenericFrameInfoForFrame(settings, 101, state, 2, 2)
	if err != nil {
		t.Fatalf("WebRTCGenericFrameInfoForFrame: %v", err)
	}
	if info.FrameID != 101 || info.SpatialID != 1 || info.TemporalID != 1 {
		t.Fatalf("info ids = %+v", info)
	}
	if info.DependencyNum != 1 || info.Dependencies[0] != 100 {
		t.Fatalf("dependencies = %+v count=%d", info.Dependencies, info.DependencyNum)
	}
	if info.DTINum != 4 ||
		info.DTIs[0] != DecodeTargetNotPresent ||
		info.DTIs[1] != DecodeTargetNotPresent ||
		info.DTIs[2] != DecodeTargetNotPresent ||
		info.DTIs[3] != DecodeTargetDiscardable {
		t.Fatalf("dtis = %+v count=%d", info.DTIs, info.DTINum)
	}
	if !out.Valid[4] || out.FrameIDs[4] != 101 || !out.Valid[3] {
		t.Fatalf("out state = %+v", out)
	}
}

func TestWebRTCGenericFrameInfoForFrameKeyResetsFrameIDs(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[7] = true
	state.FrameIDs[7] = 88
	settings := FrameEncodeSettings{
		Type:            FrameTypeKey,
		Resolution:      Resolution{Width: 320, Height: 180},
		SpatialID:       0,
		TemporalID:      0,
		UpdateBuffer:    0,
		UpdateBufferSet: true,
	}

	info, out, err := WebRTCGenericFrameInfoForFrame(settings, 99, state, 2, 2)
	if err != nil {
		t.Fatalf("WebRTCGenericFrameInfoForFrame key: %v", err)
	}
	if info.DependencyNum != 0 || info.DTINum != 4 ||
		info.DTIs[0] != DecodeTargetSwitch || info.DTIs[1] != DecodeTargetSwitch ||
		info.DTIs[2] != DecodeTargetSwitch || info.DTIs[3] != DecodeTargetSwitch {
		t.Fatalf("key info = %+v", info)
	}
	if out.Valid[7] || !out.Valid[0] || out.FrameIDs[0] != 99 {
		t.Fatalf("key reset state = %+v", out)
	}
}

func TestWebRTCFrameDependencyStructureForConfig(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 4 || got.NumChains != 2 || got.TemplateNum != 6 || got.ResolutionNum != 2 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 320, Height: 180}) || got.Resolutions[1] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	assertWebRTCTemplateSpecs(t, got, []webRTCTemplateSpec{
		{spatial: 0, temporal: 0, dtis: "SSSS", chainDiffs: []uint8{0, 0}},
		{spatial: 0, temporal: 0, dtis: "SSRR", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
		{spatial: 0, temporal: 1, dtis: "-D-R", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
		{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
		{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{4, 1}, chainDiffs: []uint8{1, 1}},
		{spatial: 1, temporal: 1, dtis: "---D", frameDiffs: []uint16{2, 1}, chainDiffs: []uint8{3, 2}},
	})
}

func TestWebRTCFrameDependencyStructureForConfigL2T2KeyShift(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2_KEY_SHIFT,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 4 || got.NumChains != 2 || got.TemplateNum != 7 || got.ResolutionNum != 2 {
		t.Fatalf("shape = %+v", got)
	}
	want := [...]struct {
		spatial    uint8
		temporal   uint8
		dtis       [4]DecodeTargetIndication
		frameDiffs [WebRTCMaxFrameReferences]uint16
		frameNum   uint8
		chainDiffs [2]uint8
	}{
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetSwitch}, chainDiffs: [2]uint8{0, 0}},
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{2, 1}},
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{4}, frameNum: 1, chainDiffs: [2]uint8{4, 1}},
		{spatial: 0, temporal: 1, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetDiscardable, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{2, 3}},
		{spatial: 1, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetSwitch, DecodeTargetSwitch}, frameDiffs: [WebRTCMaxFrameReferences]uint16{1}, frameNum: 1, chainDiffs: [2]uint8{1, 1}},
		{spatial: 1, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetSwitch, DecodeTargetSwitch}, frameDiffs: [WebRTCMaxFrameReferences]uint16{4}, frameNum: 1, chainDiffs: [2]uint8{3, 4}},
		{spatial: 1, temporal: 1, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetDiscardable}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{1, 2}},
	}
	for i, want := range want {
		template := got.Templates[i]
		if template.SpatialID != want.spatial || template.TemporalID != want.temporal ||
			template.DTINum != 4 || template.FrameDiffNum != want.frameNum ||
			template.ChainDiffNum != 2 {
			t.Fatalf("template[%d] shape = %+v", i, template)
		}
		for j := 0; j < 4; j++ {
			if template.DTIs[j] != want.dtis[j] {
				t.Fatalf("template[%d] dti[%d]=%v want %v", i, j, template.DTIs[j], want.dtis[j])
			}
		}
		for j := uint8(0); j < want.frameNum; j++ {
			if template.FrameDiffs[j] != want.frameDiffs[j] {
				t.Fatalf("template[%d] frame diff[%d]=%d want %d", i, j, template.FrameDiffs[j], want.frameDiffs[j])
			}
		}
		if template.ChainDiffs[0] != want.chainDiffs[0] || template.ChainDiffs[1] != want.chainDiffs[1] {
			t.Fatalf("template[%d] chain diffs=%+v want %+v", i, template.ChainDiffs, want.chainDiffs)
		}
	}
}

func TestWebRTCFrameDependencyStructureForConfigKeyReferenceTemplates(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode ScalabilityMode
		want []webRTCTemplateSpec
	}{
		{
			name: "L2T1_KEY",
			mode: ScalabilityModeL2T1_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "S-", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 0, temporal: 0, dtis: "SS", chainDiffs: []uint8{0, 0}},
				{spatial: 1, temporal: 0, dtis: "-S", frameDiffs: []uint16{2}, chainDiffs: []uint8{1, 2}},
				{spatial: 1, temporal: 0, dtis: "-S", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
			},
		},
		{
			name: "L2T2_KEY",
			mode: ScalabilityModeL2T2_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSS", chainDiffs: []uint8{0, 0}},
				{spatial: 0, temporal: 0, dtis: "SS--", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
				{spatial: 0, temporal: 1, dtis: "-D--", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{4}, chainDiffs: []uint8{1, 4}},
				{spatial: 1, temporal: 1, dtis: "---D", frameDiffs: []uint16{2}, chainDiffs: []uint8{3, 2}},
			},
		},
		{
			name: "L2T3_KEY",
			mode: ScalabilityModeL2T3_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSSSS", chainDiffs: []uint8{0, 0}},
				{spatial: 0, temporal: 0, dtis: "SSS---", frameDiffs: []uint16{8}, chainDiffs: []uint8{8, 7}},
				{spatial: 0, temporal: 1, dtis: "-DS---", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
				{spatial: 0, temporal: 2, dtis: "--D---", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 0, temporal: 2, dtis: "--D---", frameDiffs: []uint16{2}, chainDiffs: []uint8{6, 5}},
				{spatial: 1, temporal: 0, dtis: "---SSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
				{spatial: 1, temporal: 0, dtis: "---SSS", frameDiffs: []uint16{8}, chainDiffs: []uint8{1, 8}},
				{spatial: 1, temporal: 1, dtis: "----DS", frameDiffs: []uint16{4}, chainDiffs: []uint8{5, 4}},
				{spatial: 1, temporal: 2, dtis: "-----D", frameDiffs: []uint16{2}, chainDiffs: []uint8{3, 2}},
				{spatial: 1, temporal: 2, dtis: "-----D", frameDiffs: []uint16{2}, chainDiffs: []uint8{7, 6}},
			},
		},
		{
			name: "L3T1_KEY",
			mode: ScalabilityModeL3T1_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "S--", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 0, temporal: 0, dtis: "SSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 1, temporal: 0, dtis: "-S-", frameDiffs: []uint16{3}, chainDiffs: []uint8{1, 3, 2}},
				{spatial: 1, temporal: 0, dtis: "-SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 2, temporal: 0, dtis: "--S", frameDiffs: []uint16{3}, chainDiffs: []uint8{2, 1, 3}},
				{spatial: 2, temporal: 0, dtis: "--S", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
			},
		},
		{
			name: "L3T2_KEY",
			mode: ScalabilityModeL3T2_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SS----", frameDiffs: []uint16{6}, chainDiffs: []uint8{6, 5, 4}},
				{spatial: 0, temporal: 0, dtis: "SSSSSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 0, temporal: 1, dtis: "-D----", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS--", frameDiffs: []uint16{6}, chainDiffs: []uint8{1, 6, 5}},
				{spatial: 1, temporal: 0, dtis: "--SSSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 1, temporal: 1, dtis: "---D--", frameDiffs: []uint16{3}, chainDiffs: []uint8{4, 3, 2}},
				{spatial: 2, temporal: 0, dtis: "----SS", frameDiffs: []uint16{6}, chainDiffs: []uint8{2, 1, 6}},
				{spatial: 2, temporal: 0, dtis: "----SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
				{spatial: 2, temporal: 1, dtis: "-----D", frameDiffs: []uint16{3}, chainDiffs: []uint8{5, 4, 3}},
			},
		},
		{
			name: "L3T3_KEY",
			mode: ScalabilityModeL3T3_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSSSSSSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 0, temporal: 0, dtis: "SSS------", frameDiffs: []uint16{12}, chainDiffs: []uint8{12, 11, 10}},
				{spatial: 0, temporal: 1, dtis: "-DS------", frameDiffs: []uint16{6}, chainDiffs: []uint8{6, 5, 4}},
				{spatial: 0, temporal: 2, dtis: "--D------", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 0, temporal: 2, dtis: "--D------", frameDiffs: []uint16{3}, chainDiffs: []uint8{9, 8, 7}},
				{spatial: 1, temporal: 0, dtis: "---SSSSSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 1, temporal: 0, dtis: "---SSS---", frameDiffs: []uint16{12}, chainDiffs: []uint8{1, 12, 11}},
				{spatial: 1, temporal: 1, dtis: "----DS---", frameDiffs: []uint16{6}, chainDiffs: []uint8{7, 6, 5}},
				{spatial: 1, temporal: 2, dtis: "-----D---", frameDiffs: []uint16{3}, chainDiffs: []uint8{4, 3, 2}},
				{spatial: 1, temporal: 2, dtis: "-----D---", frameDiffs: []uint16{3}, chainDiffs: []uint8{10, 9, 8}},
				{spatial: 2, temporal: 0, dtis: "------SSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
				{spatial: 2, temporal: 0, dtis: "------SSS", frameDiffs: []uint16{12}, chainDiffs: []uint8{2, 1, 12}},
				{spatial: 2, temporal: 1, dtis: "-------DS", frameDiffs: []uint16{6}, chainDiffs: []uint8{8, 7, 6}},
				{spatial: 2, temporal: 2, dtis: "--------D", frameDiffs: []uint16{3}, chainDiffs: []uint8{5, 4, 3}},
				{spatial: 2, temporal: 2, dtis: "--------D", frameDiffs: []uint16{3}, chainDiffs: []uint8{11, 10, 9}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := WebRTCFrameDependencyStructureForConfig(Config{
				Resolution:  Resolution{Width: 1280, Height: 720},
				Scalability: tc.mode,
			})
			if err != nil {
				t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
			}
			assertWebRTCTemplateSpecs(t, got, tc.want)
		})
	}
}

func TestWebRTCFrameDependencyStructureForConfigL1T3(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T3,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 3 || got.NumChains != 1 || got.TemplateNum != 5 || got.ResolutionNum != 1 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	assertWebRTCTemplateSpecs(t, got, []webRTCTemplateSpec{
		{spatial: 0, temporal: 0, dtis: "SSS", chainDiffs: []uint8{0}},
		{spatial: 0, temporal: 0, dtis: "SSS", frameDiffs: []uint16{4}, chainDiffs: []uint8{4}},
		{spatial: 0, temporal: 1, dtis: "-DS", frameDiffs: []uint16{2}, chainDiffs: []uint8{2}},
		{spatial: 0, temporal: 2, dtis: "--D", frameDiffs: []uint16{1}, chainDiffs: []uint8{1}},
		{spatial: 0, temporal: 2, dtis: "--D", frameDiffs: []uint16{1}, chainDiffs: []uint8{3}},
	})
}

type webRTCTemplateSpec struct {
	spatial    uint8
	temporal   uint8
	dtis       string
	frameDiffs []uint16
	chainDiffs []uint8
}

func assertWebRTCTemplateSpecs(t *testing.T, structure WebRTCFrameDependencyStructure, want []webRTCTemplateSpec) {
	t.Helper()
	if int(structure.TemplateNum) != len(want) {
		t.Fatalf("template num=%d want %d", structure.TemplateNum, len(want))
	}
	for i, want := range want {
		template := structure.Templates[i]
		if template.SpatialID != want.spatial || template.TemporalID != want.temporal ||
			template.DTINum != uint8(len(want.dtis)) ||
			template.FrameDiffNum != uint8(len(want.frameDiffs)) ||
			template.ChainDiffNum != uint8(len(want.chainDiffs)) {
			t.Fatalf("template[%d] shape=%+v want spatial=%d temporal=%d dtis=%q frameDiffs=%v chainDiffs=%v",
				i, template, want.spatial, want.temporal, want.dtis, want.frameDiffs, want.chainDiffs)
		}
		for j := 0; j < len(want.dtis); j++ {
			if got, want := template.DTIs[j], webRTCTestDTI(want.dtis[j]); got != want {
				t.Fatalf("template[%d] dti[%d]=%v want %v", i, j, got, want)
			}
		}
		for j := 0; j < len(want.frameDiffs); j++ {
			if got := template.FrameDiffs[j]; got != want.frameDiffs[j] {
				t.Fatalf("template[%d] frame diff[%d]=%d want %d", i, j, got, want.frameDiffs[j])
			}
		}
		for j := 0; j < len(want.chainDiffs); j++ {
			if got := template.ChainDiffs[j]; got != want.chainDiffs[j] {
				t.Fatalf("template[%d] chain diff[%d]=%d want %d", i, j, got, want.chainDiffs[j])
			}
		}
	}
}

func webRTCTestDTI(symbol byte) DecodeTargetIndication {
	switch symbol {
	case 'D':
		return DecodeTargetDiscardable
	case 'S':
		return DecodeTargetSwitch
	case 'R':
		return DecodeTargetRequired
	default:
		return DecodeTargetNotPresent
	}
}

func TestWebRTCDependencyDescriptorMatchFrameTemplateID(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	structure.StructureID = 61
	info := WebRTCGenericFrameInfo{
		FrameID:       10,
		SpatialID:     1,
		TemporalID:    0,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 9
	info.DTIs = structure.Templates[3].DTIs
	match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
	if err != nil {
		t.Fatalf("webRTCDependencyDescriptorMatchFrame: %v", err)
	}
	if id := (structure.StructureID + match.templateIndex) % WebRTCRtpDependencyMaxTemplates; id != 0 {
		t.Fatalf("template id=%d want 0", id)
	}
	if _, err := webRTCDependencyDescriptorMatchFrame(structure, WebRTCGenericFrameInfo{SpatialID: 3}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing layer err=%v want ErrInvalidFrame", err)
	}
}

func TestWebRTCTemporalUnitControlForFramesKeyUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	frames := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			SpatialID:       0,
			TemporalID:      0,
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       0,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}

	got, err := WebRTCTemporalUnitControlForFrames(cfg, frames[:], ReferenceBufferState{}, FrameIDBufferState{}, 10)
	if err != nil {
		t.Fatalf("WebRTCTemporalUnitControlForFrames key unit: %v", err)
	}
	if got.FrameNum != 2 || !got.HasDependencyStructure || !got.Frames[0].AttachDependencyStructure || got.Frames[1].AttachDependencyStructure {
		t.Fatalf("control flags = %+v", got)
	}
	if got.Frames[1].GenericFrameInfo.DependencyNum != 1 || got.Frames[1].GenericFrameInfo.Dependencies[0] != 10 {
		t.Fatalf("upper-layer generic info = %+v", got.Frames[1].GenericFrameInfo)
	}
	if got.Frames[1].LibaomSVCRefFrameConfig.Reference[0] != 1 || got.Frames[1].LibaomSVCRefFrameConfig.RefIdx[0] != 0 {
		t.Fatalf("upper-layer ref config = %+v", got.Frames[1].LibaomSVCRefFrameConfig)
	}
	if !got.ReferenceState.Valid[0] || !got.ReferenceState.Valid[1] ||
		got.FrameIDState.FrameIDs[0] != 10 || got.FrameIDState.FrameIDs[1] != 11 {
		t.Fatalf("states = refs %+v ids %+v", got.ReferenceState, got.FrameIDState)
	}
	if got.DependencyStructure.TemplateNum != 6 || got.DependencyStructure.NumDecodeTargets != 4 {
		t.Fatalf("dependency structure = %+v", got.DependencyStructure)
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, structure, err := WebRTCDependencyStructureStateForTemporalUnit(key.Control, WebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("key structure state: %v", err)
	}
	if !state.Valid || structure != key.Control.DependencyStructure || state.Structure != key.Control.DependencyStructure {
		t.Fatalf("key state=%+v structure=%+v control=%+v", state, structure, key.Control.DependencyStructure)
	}

	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200, 0, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	next, carried, err := WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	if err != nil {
		t.Fatalf("delta structure state: %v", err)
	}
	if !next.Valid || next.Structure != state.Structure || carried != state.Structure {
		t.Fatalf("delta state=%+v carried=%+v previous=%+v", next, carried, state.Structure)
	}

	var descriptorBuf [16]byte
	descriptor, err := AppendWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		carried,
		delta.Control.Frames[1].GenericFrameInfo,
		true,
		true,
		delta.Control.Frames[1].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) == 0 {
		t.Fatal("empty dependency descriptor")
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnitRejectsMissingState(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T1,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 11, 0, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if _, _, err := WebRTCDependencyStructureStateForTemporalUnit(delta.Control, WebRTCDependencyStructureState{}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing carried structure err=%v want %v", err, ErrInvalidFrame)
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnitAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, _, err := WebRTCDependencyStructureStateForTemporalUnit(key.Control, WebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("key structure state: %v", err)
	}
	delta, err := webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200, 0, 0, false)
	if err != nil {
		t.Fatalf("webRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCDependencyStructureStateForTemporalUnit allocated: %f", allocs)
	}
}

func TestWebRTCTemporalUnitControlForFramesDeltaUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	refState := ReferenceBufferState{}
	refState.Valid[1] = true
	refState.Resolutions[1] = Resolution{Width: 640, Height: 360}
	idState := FrameIDBufferState{}
	idState.Valid[1] = true
	idState.FrameIDs[1] = 22
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}

	got, err := WebRTCTemporalUnitControlForFrames(cfg, frames[:], refState, idState, 23)
	if err != nil {
		t.Fatalf("WebRTCTemporalUnitControlForFrames delta unit: %v", err)
	}
	if got.HasDependencyStructure || got.Frames[0].AttachDependencyStructure {
		t.Fatalf("delta unexpectedly attached structure: %+v", got)
	}
	if got.Frames[0].GenericFrameInfo.DependencyNum != 1 || got.Frames[0].GenericFrameInfo.Dependencies[0] != 22 {
		t.Fatalf("delta generic info = %+v", got.Frames[0].GenericFrameInfo)
	}
	if got.FrameIDState.FrameIDs[1] != 23 {
		t.Fatalf("delta id state = %+v", got.FrameIDState)
	}
}

func TestWebRTCDependencyHelpersAllocs(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[3] = true
	state.FrameIDs[3] = 100
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		SpatialID:        1,
		TemporalID:       1,
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3},
		ReferenceCount:   1,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = LibaomSVCRefFrameConfigForFrame(settings)
		_, _, _ = WebRTCGenericFrameInfoForFrame(settings, 101, state, 2, 2)
	})
	if allocs != 0 {
		t.Fatalf("dependency helpers allocs = %f; want 0", allocs)
	}
}

func TestWebRTCTemporalUnitControlForFramesAllocs(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	refState := ReferenceBufferState{}
	refState.Valid[1] = true
	refState.Resolutions[1] = Resolution{Width: 640, Height: 360}
	idState := FrameIDBufferState{}
	idState.Valid[1] = true
	idState.FrameIDs[1] = 22
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCTemporalUnitControlForFrames(cfg, frames[:], refState, idState, 23)
	})
	if allocs != 0 {
		t.Fatalf("temporal-unit control allocs = %f; want 0", allocs)
	}
}

// TestEncoderSmallKernelsMatchScalar checks the architecture forms of the
// small SAD and downsampling kernels against their scalar references on
// random and extreme planes.
func TestEncoderSmallKernelsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const stride = 96
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	absSAD := func(a, b []byte, as, bs, w, h int) int {
		s := 0
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				s += max(int(a[r*as+c])-int(b[r*bs+c]), int(b[r*bs+c])-int(a[r*as+c]))
			}
		}
		return s
	}
	for pattern := range 3 {
		for i := range src {
			switch pattern {
			case 0:
				src[i], ref[i] = byte(rng.Intn(256)), byte(rng.Intn(256))
			case 1:
				src[i], ref[i] = 255, 0
			default:
				src[i], ref[i] = []byte{0, 255}[rng.Intn(2)], []byte{0, 255}[rng.Intn(2)]
			}
		}
		for range 50 {
			so := rng.Intn(10*stride) + rng.Intn(16)
			ro := rng.Intn(10*stride) + rng.Intn(16)
			got := [4]int{}
			for _, rows := range []int{4, 8, 16} {
				got[0], got[1], got[2], got[3] = sad16xNx4(src[so:], 2*stride, ref[ro:], 2*stride, rows)
				for k := range 4 {
					if want := absSAD(src[so:], ref[ro+k:], 2*stride, 2*stride, 16, rows); got[k] != want {
						t.Fatalf("pattern %d rows %d: sad16xNx4[%d] = %d, want %d", pattern, rows, k, got[k], want)
					}
				}
			}
		}
		for _, sz := range [][2]int{{16, 8}, {24, 5}, {40, 30}, {7, 3}} {
			w, h := sz[0], sz[1]
			want := make([]byte, 48*h)
			got := make([]byte, 48*h)
			svtDownsample2D(want, 48, src, stride, w, h)
			svtDownsample2DArch(got, 48, src, stride, w, h)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("pattern %d %dx%d: downsample byte %d = %d, want %d", pattern, w, h, i, got[i], want[i])
				}
			}
		}
	}
}

// SetCDEFDecisionTestHook installs (nil removes) the searched-picture CDEF
// decision rewrite for external tests.
func SetCDEFDecisionTestHook(h func(bits *uint8, y, uv *[MaxCDEFStrengths]uint8, fbIdx []uint8)) {
	if h == nil {
		cdefDecisionTestHook = nil
		return
	}
	cdefDecisionTestHook = func(p *CDEFParams, fbIdx []uint8) {
		h(&p.Bits, &p.YStrength, &p.UVStrength, fbIdx)
	}
}

// LayerRecon returns the reconstruction of the last frame the stream coded
// for spatial layer i.
func (s *WebRTCStream) LayerRecon(i uint8) SourceFrame420 {
	return s.encoders[i].Recon()
}

// EncodeKeyWithSequenceMax codes src as a keyframe whose sequence header
// advertises maxWidth x maxHeight, as upper spatial layers' encoders do.
func (e *VideoEncoder) EncodeKeyWithSequenceMax(src SourceFrame420, maxWidth, maxHeight int) ([]byte, error) {
	return e.encodeKeyWithSequenceMax(src, maxWidth, maxHeight)
}

// NewHighBitDepth420VideoEncoderWithColor builds the high-bit-depth 4:2:0
// encoder a WebRTC stream uses for its layers.
var NewHighBitDepth420VideoEncoderWithColor = newHighBitDepth420VideoEncoderWithColor

func TestAppendFilmGrainParamsPayloadSkippedBySequence(t *testing.T) {
	seq := filmGrainSequenceHeader()
	seq.FilmGrainParamsPresent = false
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	payload, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, FilmGrainParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.ParamsPresent || parsed.Apply || parsed.BitsRead != 0 || parsed.BitDepth != 8 {
		t.Fatalf("film grain=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadApplyFalse(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	payload, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, FilmGrainParams{})
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.ParamsPresent || parsed.Apply || parsed.BitsRead != 1 || parsed.BitDepth != 8 {
		t.Fatalf("film grain=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadUpdateMinimal(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	params := minimalFilmGrainUpdate()
	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, params)
	if !parsed.Apply || !parsed.Update || parsed.Seed != 0x1234 {
		t.Fatalf("film grain=%+v", parsed)
	}
	if parsed.NumYPoints != 1 || parsed.YPoints[0] != [2]uint8{10, 20} {
		t.Fatalf("y points=%+v", parsed.YPoints[:parsed.NumYPoints])
	}
	if parsed.ScalingShift != 9 || parsed.ARCoeffLag != 0 || parsed.ARCoeffShift != 8 || parsed.GrainScaleShift != 3 {
		t.Fatalf("shift fields=%+v", parsed)
	}
	if !parsed.Overlap || parsed.ClipToRestrictedRange {
		t.Fatalf("range flags=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadChromaAndCoeffs(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}
	params := minimalFilmGrainUpdate()
	params.NumCbPoints = 1
	params.CbPoints[0] = [2]uint8{4, 5}
	params.NumCrPoints = 1
	params.CrPoints[0] = [2]uint8{6, 7}
	params.ARCoeffLag = 1
	params.ARCoeffsY[0] = -2
	params.ARCoeffsY[1] = 3
	params.ARCoeffsY[2] = 4
	params.ARCoeffsY[3] = -5
	params.ARCoeffsCb[0] = 6
	params.ARCoeffsCb[1] = -7
	params.ARCoeffsCb[2] = 8
	params.ARCoeffsCb[3] = -9
	params.ARCoeffsCb[4] = 10
	params.ARCoeffsCr[0] = -11
	params.ARCoeffsCr[1] = 12
	params.ARCoeffsCr[2] = -13
	params.ARCoeffsCr[3] = 14
	params.ARCoeffsCr[4] = -15
	params.CbMult = 2
	params.CbLumaMult = 3
	params.CbOffset = 257
	params.CrMult = 4
	params.CrLumaMult = 5
	params.CrOffset = 258

	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, InterFrameSize{}, nil, params)
	if parsed.NumCbPoints != 1 || parsed.CbPoints[0] != [2]uint8{4, 5} || parsed.NumCrPoints != 1 || parsed.CrPoints[0] != [2]uint8{6, 7} {
		t.Fatalf("chroma points=%+v", parsed)
	}
	if parsed.ARCoeffLag != 1 || parsed.ARCoeffsY[0] != -2 || parsed.ARCoeffsCb[4] != 10 || parsed.ARCoeffsCr[4] != -15 {
		t.Fatalf("coeffs=%+v", parsed)
	}
	if parsed.CbOffset != 257 || parsed.CrOffset != 258 {
		t.Fatalf("offsets=%+v", parsed)
	}
}

func TestAppendFilmGrainParamsPayloadCopiesReference(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}
	var size InterFrameSize
	size.RefFrameIdx[0] = 2
	var refs parser.ReferenceState
	refs.Frames[2] = parser.ReferenceFrame{
		Valid: true,
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Update:        true,
			Seed:          0x1111,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{40, 50}},
			ScalingShift:  9,
		},
	}
	params := FilmGrainParams{Apply: true, Seed: 0x2244, RefIdx: 2}
	_, parsed := appendAndParseFilmGrainParams(t, seq, prefix, size, &refs, params)
	if !parsed.Apply || parsed.Update || parsed.RefIdx != 2 || parsed.Seed != 0x2244 {
		t.Fatalf("reference copy flags=%+v", parsed)
	}
	if parsed.NumYPoints != 1 || parsed.YPoints[0] != [2]uint8{40, 50} {
		t.Fatalf("copied y points=%+v", parsed.YPoints)
	}
}

func TestAppendFilmGrainParamsPayloadRejectsInvalid(t *testing.T) {
	seq := filmGrainSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	nonIncreasing := minimalFilmGrainUpdate()
	nonIncreasing.NumYPoints = 2
	nonIncreasing.YPoints[0] = [2]uint8{10, 20}
	nonIncreasing.YPoints[1] = [2]uint8{10, 30}
	badShift := minimalFilmGrainUpdate()
	badShift.ScalingShift = 12
	badChroma := minimalFilmGrainUpdate()
	badChroma.NumCbPoints = 1
	badChroma.CbPoints[0] = [2]uint8{1, 2}
	badReference := FilmGrainParams{Apply: true, RefIdx: 2}
	cases := [...]struct {
		prefix FrameHeaderPrefix
		size   InterFrameSize
		refs   *parser.ReferenceState
		params FilmGrainParams
	}{
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9), ShowFrame: true}, params: FilmGrainParams{}},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}, params: minimalFilmGrainUpdate()},
		{prefix: prefix, params: nonIncreasing},
		{prefix: prefix, params: badShift},
		{prefix: prefix, params: badChroma},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true}, params: badReference},
	}
	var buf [64]byte
	for _, tc := range cases {
		if _, err := appendFilmGrainParamsPayload(buf[:0], seq, tc.prefix, tc.size, tc.refs, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendFilmGrainParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.params, err)
		}
	}
}

func appendAndParseFilmGrainParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) ([]byte, parser.FilmGrainParams) {
	t.Helper()
	var buf [128]byte
	payload, err := appendFilmGrainParamsPayload(buf[:0], seq, prefix, size, refs, params)
	if err != nil {
		t.Fatalf("appendFilmGrainParamsPayload: %v", err)
	}
	parsed, err := parser.ParseFilmGrainParams(
		payload,
		parser.SequenceHeader{
			FilmGrainParamsPresent: seq.FilmGrainParamsPresent,
			ColorConfig: parser.ColorConfig{
				BitDepth:     seq.ColorConfig.BitDepth,
				MonoChrome:   seq.ColorConfig.MonoChrome,
				SubsamplingX: seq.ColorConfig.SubsamplingX,
				SubsamplingY: seq.ColorConfig.SubsamplingY,
			},
		},
		parser.FrameHeaderPrefix{
			ShowExistingFrame: prefix.ShowExistingFrame,
			FrameType:         parser.FrameType(prefix.FrameType),
			ShowFrame:         prefix.ShowFrame,
			ShowableFrame:     prefix.ShowableFrame,
		},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		refs,
		parser.GlobalMotionParams{},
	)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsed
}

func filmGrainSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:                Profile0,
		FilmGrainParamsPresent: true,
		OperatingPointsCount:   1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func minimalFilmGrainUpdate() FilmGrainParams {
	params := FilmGrainParams{
		Apply:           true,
		Update:          true,
		Seed:            0x1234,
		NumYPoints:      1,
		ScalingShift:    9,
		ARCoeffShift:    8,
		GrainScaleShift: 3,
		Overlap:         true,
	}
	params.YPoints[0] = [2]uint8{10, 20}
	return params
}

func TestAppendIntraFrameHeaderPayloadRoundTripKeyFrame(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	payload, parsed := appendAndParseIntraFrameHeader(t, seq, header)

	if len(payload) == 0 {
		t.Fatal("payload is empty")
	}
	if parsed.Prefix.FrameType != parser.FrameTypeKey || !parsed.Prefix.ShowFrame {
		t.Fatalf("prefix=%+v", parsed.Prefix)
	}
	if parsed.Size.UpscaledWidth != 64 || parsed.Size.Height != 64 || parsed.Size.CodedWidth != 64 {
		t.Fatalf("size=%+v", parsed.Size)
	}
	if parsed.Tile.Cols != 1 || parsed.Tile.Rows != 1 || !parsed.Tile.RefreshContext {
		t.Fatalf("tile=%+v", parsed.Tile)
	}
	if parsed.Quant.BaseQIdx != 50 || parsed.Seg.Enabled || parsed.Delta.DeltaQPresent {
		t.Fatalf("quant/seg/delta=%+v %+v %+v", parsed.Quant, parsed.Seg, parsed.Delta)
	}
	if parsed.LoopFilter.LevelY != [2]uint8{4, 4} || parsed.CDEF.Bits != 0 || parsed.Restoration.Type[0] != parser.RestorationNone {
		t.Fatalf("filters=%+v %+v %+v", parsed.LoopFilter, parsed.CDEF, parsed.Restoration)
	}
	if parsed.TransformRef.TransformMode != parser.TransformModeLargest || parsed.FrameMode.ReducedTxSet || parsed.GlobalMotion.Ref[0] != parser.DefaultWarpedMotionParams() {
		t.Fatalf("motion=%+v %+v %+v", parsed.TransformRef, parsed.FrameMode, parsed.GlobalMotion)
	}
	if parsed.FilmGrain.Apply {
		t.Fatalf("film grain=%+v", parsed.FilmGrain)
	}
}

func TestAppendInterFrameHeaderPayloadRoundTrip(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	payload, parsed := appendAndParseInterFrameHeader(t, seq, header, &refs)

	if len(payload) == 0 {
		t.Fatal("payload is empty")
	}
	if parsed.Prefix.FrameType != parser.FrameTypeInter || !parsed.Prefix.ShowFrame {
		t.Fatalf("prefix=%+v", parsed.Prefix)
	}
	if parsed.Size.RefreshFrameFlags != 0x02 || parsed.Size.RefFrameIdx[0] != 0 || parsed.Size.UpscaledWidth != 64 {
		t.Fatalf("size=%+v", parsed.Size)
	}
	if parsed.Tile.InterpolationFilter != parser.InterpolationEightTap || !parsed.Tile.RefreshContext {
		t.Fatalf("tile=%+v", parsed.Tile)
	}
	if parsed.TransformRef.ReferenceMode != parser.ReferenceModeSingle || parsed.SkipMode.Allowed {
		t.Fatalf("transform/skip=%+v %+v", parsed.TransformRef, parsed.SkipMode)
	}
	if parsed.GlobalMotion.Ref[0] != parser.DefaultWarpedMotionParams() || parsed.FilmGrain.Apply {
		t.Fatalf("tail=%+v %+v", parsed.GlobalMotion, parsed.FilmGrain)
	}
}

func TestAppendIntraFrameHeaderPayloadRejectsInvalid(t *testing.T) {
	seq := fullHeaderSequence()
	header := fullHeaderKeyFrame()
	header.Tile.SBCols = 2
	var buf [256]byte
	if _, err := appendIntraFrameHeaderPayload(buf[:0], seq, header); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("appendIntraFrameHeaderPayload err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendInterFrameHeaderPayloadRejectsInvalid(t *testing.T) {
	seq := fullHeaderSequence()
	header, refs := fullHeaderInterFrame(seq)
	header.References = &refs
	header.Size.RefFrameIdx[0] = parser.RefFrames
	var buf [256]byte
	if _, err := appendInterFrameHeaderPayload(buf[:0], seq, header); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("appendInterFrameHeaderPayload err=%v want ErrInvalidFrame", err)
	}
}

func TestAppendLowOverheadFrameHeaderOBURejectsInvalidExtension(t *testing.T) {
	seq := fullHeaderSequence()
	intra := fullHeaderKeyFrame()
	if _, err := LowOverheadIntraFrameHeaderOBUSize(seq, intra, 8, 0); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("LowOverheadIntraFrameHeaderOBUSize temporal err=%v want ErrInvalidFrame", err)
	}
	if _, err := LowOverheadIntraFrameHeaderOBUSize(seq, intra, 0, 4); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("LowOverheadIntraFrameHeaderOBUSize spatial err=%v want ErrInvalidFrame", err)
	}

	inter, refs := fullHeaderInterFrame(seq)
	inter.References = &refs
	if _, err := LowOverheadInterFrameHeaderOBUSize(seq, inter, 8, 0); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("LowOverheadInterFrameHeaderOBUSize temporal err=%v want ErrInvalidFrame", err)
	}
	if _, err := LowOverheadInterFrameHeaderOBUSize(seq, inter, 0, 4); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("LowOverheadInterFrameHeaderOBUSize spatial err=%v want ErrInvalidFrame", err)
	}
}

type parsedIntraFrameHeader struct {
	Prefix       parser.FrameHeaderPrefix
	Size         parser.FrameSize
	Tile         parser.TileInfo
	Quant        parser.QuantizationParams
	Seg          parser.SegmentationParams
	Delta        parser.DeltaParams
	LoopFilter   parser.LoopFilterParams
	CDEF         parser.CDEFParams
	Restoration  parser.RestorationParams
	TransformRef parser.TransformReferenceParams
	SkipMode     parser.SkipModeParams
	FrameMode    parser.FrameModeParams
	GlobalMotion parser.GlobalMotionParams
	FilmGrain    parser.FilmGrainParams
}

func appendAndParseIntraFrameHeader(t *testing.T, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, parsedIntraFrameHeader) {
	t.Helper()
	payloadSize, err := IntraFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("IntraFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := appendIntraFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("appendIntraFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := appendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	prefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := parser.ParseFrameSize(payload, parsedSeq, prefix, nil, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	tiles, err := parser.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	quant, err := parser.ParseQuantizationParams(payload, parsedSeq, tiles)
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	seg, err := parser.ParseSegmentationParams(payload, prefix, quant, nil)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	delta, err := parser.ParseDeltaParams(payload, size, quant, seg)
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	lf, err := parser.ParseLoopFilterParams(payload, parsedSeq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	cdef, err := parser.ParseCDEFParams(payload, parsedSeq, size, seg, lf)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	restoration, err := parser.ParseRestorationParams(payload, parsedSeq, size, seg, cdef)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	transformRef, err := parser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	skipMode, err := parser.ParseSkipModeParams(payload, parsedSeq, prefix, size, nil, transformRef)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	frameMode, err := parser.ParseFrameModeParams(payload, parsedSeq, prefix, skipMode)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	globalMotion, err := parser.ParseGlobalMotionParams(payload, prefix, size, tiles, nil, frameMode)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	filmGrain, err := parser.ParseFilmGrainParams(payload, parsedSeq, prefix, size, nil, globalMotion)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsedIntraFrameHeader{
		Prefix:       prefix,
		Size:         size,
		Tile:         tiles,
		Quant:        quant,
		Seg:          seg,
		Delta:        delta,
		LoopFilter:   lf,
		CDEF:         cdef,
		Restoration:  restoration,
		TransformRef: transformRef,
		SkipMode:     skipMode,
		FrameMode:    frameMode,
		GlobalMotion: globalMotion,
		FilmGrain:    filmGrain,
	}
}

func appendAndParseInterFrameHeader(t *testing.T, seq SequenceHeader, header InterFrameHeaderParams, refs *parser.ReferenceState) ([]byte, parsedIntraFrameHeader) {
	t.Helper()
	header.References = refs
	payloadSize, err := InterFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("InterFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := appendInterFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("appendInterFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := appendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	prefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := parser.ParseFrameSize(payload, parsedSeq, prefix, refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	tiles, err := parser.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	quant, err := parser.ParseQuantizationParams(payload, parsedSeq, tiles)
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	seg, err := parser.ParseSegmentationParams(payload, prefix, quant, nil)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	delta, err := parser.ParseDeltaParams(payload, size, quant, seg)
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	lf, err := parser.ParseLoopFilterParams(payload, parsedSeq, prefix, size, seg, delta, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	cdef, err := parser.ParseCDEFParams(payload, parsedSeq, size, seg, lf)
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	restoration, err := parser.ParseRestorationParams(payload, parsedSeq, size, seg, cdef)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	transformRef, err := parser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	skipMode, err := parser.ParseSkipModeParams(payload, parsedSeq, prefix, size, refs, transformRef)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	frameMode, err := parser.ParseFrameModeParams(payload, parsedSeq, prefix, skipMode)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	globalMotion, err := parser.ParseGlobalMotionParams(payload, prefix, size, tiles, refs, frameMode)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	filmGrain, err := parser.ParseFilmGrainParams(payload, parsedSeq, prefix, size, refs, globalMotion)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	return payload, parsedIntraFrameHeader{
		Prefix:       prefix,
		Size:         size,
		Tile:         tiles,
		Quant:        quant,
		Seg:          seg,
		Delta:        delta,
		LoopFilter:   lf,
		CDEF:         cdef,
		Restoration:  restoration,
		TransformRef: transformRef,
		SkipMode:     skipMode,
		FrameMode:    frameMode,
		GlobalMotion: globalMotion,
		FilmGrain:    filmGrain,
	}
}

func fullHeaderSequence() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:              64,
		MaxFrameHeight:             64,
		Use128x128Superblock:       false,
		SeqForceScreenContentTools: 1,
		SeqForceIntegerMV:          1,
		EnableCDEF:                 true,
		EnableRestoration:          true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func fullHeaderKeyFrame() IntraFrameHeaderParams {
	header := IntraFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:               FrameHeaderTypeKey,
			ShowFrame:               true,
			ErrorResilientMode:      true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         EncoderPrimaryRefNone,
		},
		Size: IntraFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile: TileInfo{
			RefreshContext: true,
			SBCols:         1,
			SBRows:         1,
			Cols:           1,
			Rows:           1,
			ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
			RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: QuantizationParams{
			BaseQIdx: 50,
		},
		LoopFilter: LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			Sharpness:           0,
			ModeRefDeltaEnabled: false,
			Deltas:              defaultLoopFilterDeltas(),
		},
		CDEF: CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		Restoration: RestorationParams{
			Type:       [3]RestorationType{RestorationNone, RestorationNone, RestorationNone},
			UnitSizeY:  0,
			UnitSizeUV: 0,
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformModeLargest,
			ReferenceMode: ReferenceModeSingle,
		},
		FrameMode: FrameModeParams{},
		FilmGrain: FilmGrainParams{},
	}
	return header
}

func fullHeaderInterFrame(seq SequenceHeader) (InterFrameHeaderParams, parser.ReferenceState) {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.RefFrames; i++ {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			OrderHint:    i,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}
	header := InterFrameHeaderParams{
		Prefix: FrameHeaderPrefix{
			FrameType:               FrameHeaderTypeInter,
			ShowFrame:               true,
			ShowableFrame:           true,
			ErrorResilientMode:      false,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         EncoderPrimaryRefNone,
		},
		Size: InterFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0x02,
			RefFrameIdx:         [7]uint8{0, 0, 0, 0, 0, 0, 0},
		},
		Tile: TileInfo{
			InterpolationFilter: InterpolationEightTap,
			RefreshContext:      true,
			SBCols:              1,
			SBRows:              1,
			Cols:                1,
			Rows:                1,
			ColStartSB:          [MaxTileCols + 1]uint16{0, 1},
			RowStartSB:          [MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: QuantizationParams{
			BaseQIdx: 50,
		},
		LoopFilter: LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			Sharpness:           0,
			ModeRefDeltaEnabled: false,
			Deltas:              defaultLoopFilterDeltas(),
		},
		CDEF: CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		Restoration: RestorationParams{
			Type:       [3]RestorationType{RestorationNone, RestorationNone, RestorationNone},
			UnitSizeY:  0,
			UnitSizeUV: 0,
		},
		TransformRef: TransformReferenceParams{
			TransformMode: TransformModeLargest,
			ReferenceMode: ReferenceModeSingle,
		},
		GlobalMotion: DefaultGlobalMotionParams(),
	}
	return header, refs
}

func TestAppendFrameModeParamsPayloadIntraReducedOnly(t *testing.T) {
	seq := frameModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true}
	params := FrameModeParams{ReducedTxSet: true}
	payload, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.AllowWarpedMotion || !parsed.ReducedTxSet || parsed.BitsRead != 1 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadInterWarpedAndReduced(t *testing.T) {
	seq := frameModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true}
	payload, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.AllowWarpedMotion || !parsed.ReducedTxSet || parsed.BitsRead != 2 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadSkipsWarpedWhenDisabled(t *testing.T) {
	seq := frameModeSequenceHeader()
	seq.EnableWarpedMotion = false
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := FrameModeParams{ReducedTxSet: false}
	_, parsed := appendAndParseFrameModeParams(t, seq, prefix, params)
	if parsed.AllowWarpedMotion || parsed.ReducedTxSet || parsed.BitsRead != 1 {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestAppendFrameModeParamsPayloadRejectsInvalid(t *testing.T) {
	seq := frameModeSequenceHeader()
	cases := [...]struct {
		seq    SequenceHeader
		prefix FrameHeaderPrefix
		params FrameModeParams
	}{
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ErrorResilientMode: true},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    frameModeSequenceHeaderWarpDisabled(),
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter},
			params: FrameModeParams{AllowWarpedMotion: true},
		},
		{
			seq:    seq,
			prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9)},
			params: FrameModeParams{},
		},
	}
	var buf [1]byte
	for _, tc := range cases {
		if _, err := appendFrameModeParamsPayload(buf[:0], tc.seq, tc.prefix, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendFrameModeParamsPayload(%+v,%+v) err=%v want ErrInvalidFrame", tc.prefix, tc.params, err)
		}
	}
}

func appendAndParseFrameModeParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) ([]byte, parser.FrameModeParams) {
	t.Helper()
	var buf [1]byte
	payload, err := appendFrameModeParamsPayload(buf[:0], seq, prefix, params)
	if err != nil {
		t.Fatalf("appendFrameModeParamsPayload: %v", err)
	}
	parsed, err := parser.ParseFrameModeParams(
		payload,
		parser.SequenceHeader{EnableWarpedMotion: seq.EnableWarpedMotion},
		parser.FrameHeaderPrefix{
			FrameType:          parser.FrameType(prefix.FrameType),
			ErrorResilientMode: prefix.ErrorResilientMode,
		},
		parser.SkipModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	return payload, parsed
}

func frameModeSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:              64,
		MaxFrameHeight:             64,
		Use128x128Superblock:       true,
		EnableFilterIntra:          true,
		EnableIntraEdgeFilter:      true,
		EnableInterIntraCompound:   true,
		EnableMaskedCompound:       true,
		EnableWarpedMotion:         true,
		EnableDualFilter:           true,
		EnableOrderHint:            true,
		EnableJNTComp:              true,
		EnableRefFrameMVS:          true,
		SeqForceScreenContentTools: SequenceSelectScreenContentTools,
		SeqForceIntegerMV:          SequenceSelectIntegerMV,
		OrderHintBits:              8,
		EnableSuperRes:             true,
		EnableCDEF:                 true,
		EnableRestoration:          true,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func frameModeSequenceHeaderWarpDisabled() SequenceHeader {
	seq := frameModeSequenceHeader()
	seq.EnableWarpedMotion = false
	return seq
}

func TestAppendFrameHeaderIntraPayloadShownKeyFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := shownKeyFrameHeaderPrefix(false)
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	_, parsedPrefix, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if parsedPrefix.BitsRead >= parsedSize.BitsRead {
		t.Fatalf("bits did not advance: prefix=%d size=%d", parsedPrefix.BitsRead, parsedSize.BitsRead)
	}
	if parsedSize.RefreshFrameFlags != 0xff ||
		parsedSize.UpscaledWidth != seq.MaxFrameWidth ||
		parsedSize.CodedWidth != seq.MaxFrameWidth ||
		parsedSize.Height != seq.MaxFrameHeight ||
		parsedSize.RenderWidth != seq.MaxFrameWidth ||
		parsedSize.RenderHeight != seq.MaxFrameHeight ||
		parsedSize.SuperResEnabled ||
		parsedSize.SuperResDenominator != 8 {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadOverrideSuperresRender(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.MaxFrameWidth = 64
	seq.MaxFrameHeight = 32
	prefix := shownKeyFrameHeaderPrefix(true)
	size := IntraFrameSize{
		UpscaledWidth:       48,
		Height:              24,
		RenderWidth:         40,
		RenderHeight:        20,
		SuperResDenominator: 12,
		HaveRenderSize:      true,
		RefreshFrameFlags:   0xff,
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if !parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 12 {
		t.Fatalf("superres=%+v", parsedSize)
	}
	if parsedSize.UpscaledWidth != 48 || parsedSize.CodedWidth != 32 || parsedSize.Height != 24 {
		t.Fatalf("dimensions=%+v", parsedSize)
	}
	if !parsedSize.HaveRenderSize || parsedSize.RenderWidth != 40 || parsedSize.RenderHeight != 20 {
		t.Fatalf("render dimensions=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadHiddenKeyRefreshAndOrderHints(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          4,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	for i := uint8(0); i < encoderRefFrames; i++ {
		size.RefOrderHints[i] = i
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if parsedSize.RefreshFrameFlags != 0x01 {
		t.Fatalf("RefreshFrameFlags=%02x", parsedSize.RefreshFrameFlags)
	}
	for i := uint8(0); i < encoderRefFrames; i++ {
		if parsedSize.RefOrderHints[i] != i {
			t.Fatalf("RefOrderHints[%d]=%d", i, parsedSize.RefOrderHints[i])
		}
	}
}

func TestAppendFrameHeaderIntraPayloadAllowIntrabc(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
	seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	prefix := shownKeyFrameHeaderPrefix(false)
	prefix.AllowScreenContentTools = true
	size := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		AllowIntrabc:        true,
		RefreshFrameFlags:   0xff,
	}
	_, _, parsedSize := appendAndParseFrameHeaderIntra(t, seq, prefix, size)
	if !parsedSize.AllowIntrabc {
		t.Fatalf("AllowIntrabc=false parsed size=%+v", parsedSize)
	}
}

func TestAppendFrameHeaderIntraPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	basePrefix := shownKeyFrameHeaderPrefix(false)
	baseSize := IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	tests := []struct {
		name string
		mut  func(*FrameHeaderPrefix, *IntraFrameSize)
	}{
		{name: "inter prefix", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			*prefix = FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, ShowFrame: true, PrimaryRefFrame: 0}
		}},
		{name: "bad dimensions without override", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.UpscaledWidth--
		}},
		{name: "shown key refresh flags", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.RefreshFrameFlags = 0x7f
		}},
		{name: "bad superres denom", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.SuperResDenominator = 17
		}},
		{name: "bad render size", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			size.HaveRenderSize = true
			size.RenderWidth = 0
			size.RenderHeight = 20
		}},
		{name: "intrabc with superres", mut: func(prefix *FrameHeaderPrefix, size *IntraFrameSize) {
			prefix.AllowScreenContentTools = true
			size.SuperResDenominator = 12
			size.AllowIntrabc = true
		}},
	}
	var buf [32]byte
	for _, tt := range tests {
		prefix := basePrefix
		size := baseSize
		tt.mut(&prefix, &size)
		if _, err := appendFrameHeaderIntraPayload(buf[:0], seq, prefix, size); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFrame)
		}
	}
}

func TestAppendFrameHeaderInterPayloadDirectSize(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(false)
	size := InterFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	_, parsedPrefix, parsedSize := appendAndParseFrameHeaderInter(t, seq, prefix, size)
	if parsedPrefix.FrameType != parser.FrameTypeInter || !parsedPrefix.ShowableFrame {
		t.Fatalf("prefix=%+v", parsedPrefix)
	}
	if parsedSize.RefreshFrameFlags != 0x01 || parsedSize.FrameRefsShortSignaling ||
		parsedSize.UpscaledWidth != seq.MaxFrameWidth || parsedSize.Height != seq.MaxFrameHeight ||
		parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 8 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if parsedSize.RefFrameIdx[i] != 0 {
			t.Fatalf("RefFrameIdx[%d]=%d want 0", i, parsedSize.RefFrameIdx[i])
		}
	}
}

func TestAppendFrameHeaderInterPayloadOverrideSuperresRender(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(true)
	size := InterFrameSize{
		UpscaledWidth:       12,
		Height:              6,
		RenderWidth:         10,
		RenderHeight:        5,
		SuperResDenominator: 12,
		HaveRenderSize:      true,
		RefreshFrameFlags:   0x02,
		RefFrameIdx:         [7]uint8{0, 1, 1, 0, 1, 0, 1},
	}
	_, _, parsedSize := appendAndParseFrameHeaderInter(t, seq, prefix, size)
	if parsedSize.RefreshFrameFlags != 0x02 ||
		parsedSize.UpscaledWidth != 12 || parsedSize.Height != 6 ||
		!parsedSize.SuperResEnabled || parsedSize.SuperResDenominator != 12 ||
		!parsedSize.HaveRenderSize || parsedSize.RenderWidth != 10 || parsedSize.RenderHeight != 5 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}
	for i := 0; i < parser.InterRefsPerFrame; i++ {
		if parsedSize.RefFrameIdx[i] != size.RefFrameIdx[i] {
			t.Fatalf("RefFrameIdx[%d]=%d want %d", i, parsedSize.RefFrameIdx[i], size.RefFrameIdx[i])
		}
	}
}

func TestAppendFrameHeaderInterPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := interFrameHeaderPrefix(false)
	size := InterFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	var buf [16]byte
	if _, err := appendFrameHeaderInterPayload(buf[:0], seq, shownKeyFrameHeaderPrefix(false), size); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intra prefix err=%v want %v", err, ErrInvalidFrame)
	}
	bad := size
	bad.RefFrameIdx[0] = 8
	if _, err := appendFrameHeaderInterPayload(buf[:0], seq, prefix, bad); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad ref idx err=%v want %v", err, ErrInvalidFrame)
	}
}

func appendAndParseFrameHeaderIntra(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	parsedSeq := parseEncoderSequenceHeader(t, seq)

	var buf [64]byte
	payload, err := appendFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("appendFrameHeaderIntraPayload: %v", err)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, nil, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	return payload, parsedPrefix, parsedSize
}

func appendAndParseFrameHeaderInter(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, parser.FrameHeaderPrefix, parser.FrameSize) {
	t.Helper()
	parsedSeq := parseEncoderSequenceHeader(t, seq)

	var buf [64]byte
	payload, err := appendFrameHeaderInterPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("appendFrameHeaderInterPayload: %v", err)
	}
	parsedPrefix, err := parser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := parserReferenceState(seq, size.RefFrameIdx[:])
	parsedSize, err := parser.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	return payload, parsedPrefix, parsedSize
}

func parseEncoderSequenceHeader(t *testing.T, seq SequenceHeader) parser.SequenceHeader {
	t.Helper()
	var seqBuf [128]byte
	seqPayload, err := appendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	return parsedSeq
}

func parserReferenceState(seq SequenceHeader, indices []uint8) parser.ReferenceState {
	var refs parser.ReferenceState
	for _, idx := range indices {
		refs.Frames[idx].Valid = true
		refs.Frames[idx].OrderHint = 1
		refs.Frames[idx].Size.UpscaledWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.CodedWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.Height = seq.MaxFrameHeight
		refs.Frames[idx].Size.RenderWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.RenderHeight = seq.MaxFrameHeight
		refs.Frames[idx].Size.SuperResDenominator = 8
	}
	return refs
}

func shownKeyFrameHeaderPrefix(frameSizeOverride bool) FrameHeaderPrefix {
	return FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		FrameSizeOverride:  frameSizeOverride,
		OrderHint:          3,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
}

func interFrameHeaderPrefix(frameSizeOverride bool) FrameHeaderPrefix {
	return FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		FrameSizeOverride:  frameSizeOverride,
		OrderHint:          4,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
}

func TestAppendFrameHeaderPrefixPayloadShownKeyFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		DisableCDFUpdate:   true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	out, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if len(out) != 2 {
		t.Fatalf("prefix bytes=% x len=%d want 2", out, len(out))
	}
	if parsed.ShowExistingFrame || parsed.FrameType != parser.FrameTypeKey || !parsed.ShowFrame ||
		!parsed.ErrorResilientMode || !parsed.DisableCDFUpdate || parsed.FrameSizeOverride ||
		parsed.OrderHint != 5 || parsed.PrimaryRefFrame != parser.PrimaryRefNone {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadHiddenInterFrame(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:         FrameHeaderTypeInter,
		ShowableFrame:     true,
		FrameSizeOverride: true,
		OrderHint:         9,
		PrimaryRefFrame:   3,
	}
	_, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if parsed.FrameType != parser.FrameTypeInter || parsed.ShowFrame || !parsed.ShowableFrame ||
		parsed.ErrorResilientMode || parsed.DisableCDFUpdate || !parsed.FrameSizeOverride ||
		parsed.OrderHint != 9 || parsed.PrimaryRefFrame != 3 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadShowExisting(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.FrameIDNumbersPresent = true
	seq.DeltaFrameIDLength = 4
	seq.AdditionalFrameIDLength = 2
	prefix := FrameHeaderPrefix{
		ShowExistingFrame: true,
		ExistingFrameIdx:  3,
		FrameID:           42,
		PrimaryRefFrame:   EncoderPrimaryRefNone,
	}
	_, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if !parsed.ShowExistingFrame || parsed.ExistingFrameIdx != 3 || parsed.FrameID != 42 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadPresentationDelay(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.TimingInfoPresent = true
	seq.TimingInfo = SequenceTimingInfo{
		NumUnitsInDisplayTick: 1001,
		TimeScale:             30000,
	}
	seq.DecoderModelInfoPresent = true
	seq.DecoderModelInfo = SequenceDecoderModelInfo{
		BufferDelayLength:           8,
		BufferRemovalTimeLength:     5,
		FramePresentationTimeLength: 6,
	}
	prefix := FrameHeaderPrefix{
		FrameType:              FrameHeaderTypeKey,
		ShowFrame:              true,
		ErrorResilientMode:     true,
		DisableCDFUpdate:       true,
		ForceIntegerMV:         true,
		FramePresentationDelay: 17,
		OrderHint:              5,
		PrimaryRefFrame:        EncoderPrimaryRefNone,
	}
	_, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if parsed.FramePresentationDelay != 17 {
		t.Fatalf("parsed presentation delay=%d want 17", parsed.FramePresentationDelay)
	}
}

func TestAppendFrameHeaderPrefixPayloadReducedStillPicture(t *testing.T) {
	seq := reducedStillEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	out, parsed := appendAndParseFrameHeaderPrefix(t, seq, prefix)
	if len(out) != 1 {
		t.Fatalf("prefix bytes=% x len=%d want 1", out, len(out))
	}
	if parsed.ShowExistingFrame || parsed.FrameType != parser.FrameTypeKey || !parsed.ShowFrame ||
		!parsed.ErrorResilientMode || parsed.DisableCDFUpdate || parsed.BitsRead != 2 {
		t.Fatalf("parsed prefix=%+v", parsed)
	}
}

func TestAppendFrameHeaderPrefixPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	tests := []struct {
		name   string
		prefix FrameHeaderPrefix
		want   error
	}{
		{
			name: "shown key without implicit error resilient",
			prefix: FrameHeaderPrefix{
				FrameType:       FrameHeaderTypeKey,
				ShowFrame:       true,
				ForceIntegerMV:  true,
				PrimaryRefFrame: EncoderPrimaryRefNone,
			},
			want: ErrInvalidFrame,
		},
		{
			name: "order hint overflow",
			prefix: FrameHeaderPrefix{
				FrameType:         FrameHeaderTypeInter,
				ShowableFrame:     true,
				FrameSizeOverride: true,
				OrderHint:         128,
				PrimaryRefFrame:   3,
			},
			want: ErrInvalidFrame,
		},
		{
			name: "presentation delay without decoder model",
			prefix: FrameHeaderPrefix{
				ShowExistingFrame:      true,
				ExistingFrameIdx:       1,
				FramePresentationDelay: 1,
				PrimaryRefFrame:        EncoderPrimaryRefNone,
			},
			want: ErrInvalidFrame,
		},
	}
	var buf [8]byte
	for _, tt := range tests {
		if _, err := appendFrameHeaderPrefixPayload(buf[:0], seq, tt.prefix); !errors.Is(err, tt.want) {
			t.Fatalf("%s err=%v want %v", tt.name, err, tt.want)
		}
	}
}

func appendAndParseFrameHeaderPrefix(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix) ([]byte, parser.FrameHeaderPrefix) {
	t.Helper()
	var seqBuf [96]byte
	seqPayload, err := appendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := parser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}

	var buf [16]byte
	out, err := appendFrameHeaderPrefixPayload(buf[:0], seq, prefix)
	if err != nil {
		t.Fatalf("appendFrameHeaderPrefixPayload: %v", err)
	}
	parsed, err := parser.ParseFrameHeaderPrefix(out, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	return out, parsed
}

func reducedStillEncoderSequenceHeader() SequenceHeader {
	seq := realtimeEncoderSequenceHeader()
	seq.StillPicture = true
	seq.ReducedStillPictureHeader = true
	seq.EnableInterIntraCompound = false
	seq.EnableMaskedCompound = false
	seq.EnableDualFilter = false
	seq.EnableOrderHint = false
	seq.EnableJNTComp = false
	seq.EnableRefFrameMVS = false
	seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
	seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	seq.OrderHintBits = 0
	seq.EnableSuperRes = false
	seq.EnableRestoration = false
	seq.FilmGrainParamsPresent = false
	return seq
}

func TestAppendGlobalMotionParamsPayloadIntraReadsNoBits(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	payload, parsed := appendAndParseGlobalMotionParams(t, prefix, InterFrameSize{}, parser.TileInfo{}, nil, DefaultGlobalMotionParams())
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.BitsRead != 0 {
		t.Fatalf("global motion bits=%d", parsed.BitsRead)
	}
	for i := range parser.InterRefsPerFrame {
		if parsed.Ref[i] != parser.DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, parsed.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadInterIdentity(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	payload, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{}, &refs, DefaultGlobalMotionParams())
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.BitsRead != parser.InterRefsPerFrame {
		t.Fatalf("BitsRead=%d want %d", parsed.BitsRead, parser.InterRefsPerFrame)
	}
	for i := range parser.InterRefsPerFrame {
		if parsed.Ref[i] != parser.DefaultWarpedMotionParams() {
			t.Fatalf("ref[%d]=%+v", i, parsed.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadTranslation(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	params.Ref[0] = DefaultWarpedMotionParams()
	params.Ref[0].Type = GlobalMotionTranslation
	params.Ref[0].Matrix[0] = 2 << gmTransOnlyPrecDiff
	params.Ref[0].Matrix[1] = -1 << gmTransOnlyPrecDiff

	_, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if parsed.Ref[0].Type != parser.GlobalMotionTranslation ||
		parsed.Ref[0].Matrix[0] != params.Ref[0].Matrix[0] ||
		parsed.Ref[0].Matrix[1] != params.Ref[0].Matrix[1] {
		t.Fatalf("ref[0]=%+v want %+v", parsed.Ref[0], params.Ref[0])
	}
}

func TestEstimateGlobalMotionTranslationRoundTrip(t *testing.T) {
	const w, h = 128, 128
	ref := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, (w/2)*(h/2)),
		V:            make([]byte, (w/2)*(h/2)),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	src := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, (w/2)*(h/2)),
		V:            make([]byte, (w/2)*(h/2)),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	// Textured reference and a pure global pan of (dx=4, dy=-2).
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ref.Y[y*w+x] = byte((x*7 + y*13 + (x^y)*3) & 0xff)
		}
	}
	const panDX, panDY = 4, -2
	for y := 0; y < h; y++ {
		ry := y + panDY
		if ry < 0 {
			ry = 0
		} else if ry >= h {
			ry = h - 1
		}
		for x := 0; x < w; x++ {
			rx := x + panDX
			if rx < 0 {
				rx = 0
			} else if rx >= w {
				rx = w - 1
			}
			src.Y[y*w+x] = ref.Y[ry*w+rx]
		}
	}
	gm := EstimateGlobalMotionTranslation(src, ref)
	if gm.Type != GlobalMotionTranslation {
		t.Fatalf("EstimateGlobalMotionTranslation type=%v want GlobalMotionTranslation", gm.Type)
	}
	if gm.Matrix[0] != int32(panDY)<<warpedModelPrecBits || gm.Matrix[1] != int32(panDX)<<warpedModelPrecBits {
		t.Fatalf("EstimateGlobalMotionTranslation matrix=%v want dy=%d dx=%d", gm.Matrix, panDY, panDX)
	}
	// Verify the estimated model round-trips through AV1 global_motion_params().
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	params.Ref[0] = gm
	_, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if parsed.Ref[0].Type != parser.GlobalMotionTranslation || parsed.Ref[0].Matrix != gm.Matrix {
		t.Fatalf("parsed ref[0]=%+v want %+v", parsed.Ref[0], gm)
	}
	// Identical frames must return Identity.
	if id := EstimateGlobalMotionTranslation(ref, ref); id.Type != GlobalMotionIdentity {
		t.Fatalf("identical frames returned %+v, want Identity", id)
	}
}

func TestAppendGlobalMotionParamsPayloadRotZoomAndAffine(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	params := DefaultGlobalMotionParams()
	params.Ref[0] = DefaultWarpedMotionParams()
	params.Ref[0].Type = GlobalMotionRotZoom
	params.Ref[0].Matrix[2] = (1 << warpedModelPrecBits) + 4
	params.Ref[0].Matrix[3] = 2
	params.Ref[0].Matrix[4] = -2
	params.Ref[0].Matrix[5] = (1 << warpedModelPrecBits) + 4
	params.Ref[1] = DefaultWarpedMotionParams()
	params.Ref[1].Type = GlobalMotionAffine
	params.Ref[1].Matrix[2] = (1 << warpedModelPrecBits) - 6
	params.Ref[1].Matrix[3] = 4
	params.Ref[1].Matrix[4] = 2
	params.Ref[1].Matrix[5] = (1 << warpedModelPrecBits) + 8

	_, parsed := appendAndParseGlobalMotionParams(t, prefix, size, parser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	for i := range 2 {
		if parsed.Ref[i].Type != parser.GlobalMotionType(params.Ref[i].Type) || parsed.Ref[i].Matrix != params.Ref[i].Matrix {
			t.Fatalf("ref[%d]=+%v want %+v", i, parsed.Ref[i], params.Ref[i])
		}
	}
}

func TestAppendGlobalMotionParamsPayloadRejectsInvalid(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: EncoderPrimaryRefNone}
	size, refs := encoderGlobalMotionRefs()
	invalidIntra := DefaultGlobalMotionParams()
	invalidIntra.Ref[0].Type = GlobalMotionTranslation
	invalidTranslation := DefaultGlobalMotionParams()
	invalidTranslation.Ref[0].Type = GlobalMotionTranslation
	invalidTranslation.Ref[0].Matrix[0] = 1
	invalidRotZoom := DefaultGlobalMotionParams()
	invalidRotZoom.Ref[0].Type = GlobalMotionRotZoom
	invalidRotZoom.Ref[0].Matrix[2] = 1 << warpedModelPrecBits
	invalidRotZoom.Ref[0].Matrix[3] = 2
	invalidRotZoom.Ref[0].Matrix[4] = 0
	invalidRotZoom.Ref[0].Matrix[5] = 1 << warpedModelPrecBits
	badRefSize := size
	badRefSize.RefFrameIdx[0] = parser.RefFrames
	cases := [...]struct {
		prefix FrameHeaderPrefix
		size   InterFrameSize
		refs   *parser.ReferenceState
		params GlobalMotionParams
	}{
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}, size: size, refs: nil, params: invalidIntra},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderType(9)}, size: size, refs: &refs, params: DefaultGlobalMotionParams()},
		{prefix: prefix, size: size, refs: &refs, params: invalidTranslation},
		{prefix: prefix, size: size, refs: &refs, params: invalidRotZoom},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: 0}, size: badRefSize, refs: &refs, params: DefaultGlobalMotionParams()},
		{prefix: FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, PrimaryRefFrame: 0}, size: size, refs: nil, params: DefaultGlobalMotionParams()},
	}
	var buf [8]byte
	for _, tc := range cases {
		if _, err := appendGlobalMotionParamsPayload(buf[:0], tc.prefix, tc.size, parser.TileInfo{}, tc.refs, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendGlobalMotionParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.prefix, err)
		}
	}
}

func appendAndParseGlobalMotionParams(t *testing.T, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) ([]byte, parser.GlobalMotionParams) {
	t.Helper()
	var buf [64]byte
	payload, err := appendGlobalMotionParamsPayload(buf[:0], prefix, size, tiles, refs, params)
	if err != nil {
		t.Fatalf("appendGlobalMotionParamsPayload: %v", err)
	}
	parsed, err := parser.ParseGlobalMotionParams(
		payload,
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType), PrimaryRefFrame: prefix.PrimaryRefFrame},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		tiles,
		refs,
		parser.FrameModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	return payload, parsed
}

func encoderGlobalMotionRefs() (InterFrameSize, parser.ReferenceState) {
	var size InterFrameSize
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
		}
	}
	return size, refs
}

func TestInterTXBResultMatchesScanWalk(t *testing.T) {
	var st lossyEncodeState
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	for _, e := range sharedScans.inverse {
		n := len(e.iscan)
		var sc []int16
		for _, cand := range [][]int16{sharedScans.scan4, sharedScans.scan8, sharedScans.scan16, sharedScans.scan32,
			sharedScans.scan16x8, sharedScans.scan8x16, sharedScans.scan8x4, sharedScans.scan4x8,
			sharedScans.scan32x16, sharedScans.scan16x32, sharedScans.scan4x16, sharedScans.scan16x4,
			sharedScans.scan8x32, sharedScans.scan32x8, sharedScans.scan16x64, sharedScans.scan64x16} {
			if &cand[0] == e.scan {
				sc = cand
			}
		}
		for trial := range 300 {
			q := make([]int16, n)
			for k := 0; k < 1+trial%7; k++ {
				q[rng.Intn(n)] = int16(rng.Intn(9) - 4)
			}
			want := 0
			for c := n - 1; c >= 0; c-- {
				if q[sc[c]] != 0 {
					want = c + 1
					break
				}
			}
			got := interTXBResult(q, sc, n)
			if want == 0 {
				if got != (tile.TXBDecodeResult{AllZero: true}) {
					t.Fatalf("n=%d: got %+v want all-zero", n, got)
				}
			} else if got != (tile.TXBDecodeResult{EOB: uint16(want)}) {
				t.Fatalf("n=%d trial %d: got %+v want eob %d", n, trial, got, want)
			}
		}
	}
}

func TestSelectIntraModeNMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x51ec7))
	const stride = 96
	src := make([]byte, stride*stride)
	recon := make([]byte, stride*stride)
	for i := range src {
		src[i] = byte(rng.Intn(256))
		recon[i] = byte(rng.Intn(256))
	}
	for _, n := range []int{4, 8, 16, 32} {
		for _, haveTop := range []bool{false, true} {
			for _, haveLeft := range []bool{false, true} {
				t.Run(fmt.Sprintf("n%d/top%v/left%v", n, haveTop, haveLeft), func(t *testing.T) {
					for i := 0; i < 64; i++ {
						px := 1 + rng.Intn(stride-n-1)
						py := 1 + rng.Intn(stride-n-1)
						gotPred := make([]byte, n*n)
						wantPred := make([]byte, n*n)
						got := selectIntraModeN(src, recon, stride, px, py, n, haveTop, haveLeft, gotPred)
						want := selectIntraModeNReference(src, recon, stride, px, py, n, haveTop, haveLeft, wantPred)
						if got != want {
							t.Fatalf("mode=%v want %v", got, want)
						}
						if string(gotPred) != string(wantPred) {
							t.Fatalf("prediction mismatch for mode %v", got)
						}
					}
				})
			}
		}
	}
}

func selectIntraModeNReference(srcPlane, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	sadDC, sadV, sadH := 0, 1<<30, 1<<30
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			d := int(srcPlane[row+c]) - int(dc)
			if d < 0 {
				d = -d
			}
			sadDC += d
		}
	}
	above := (py-1)*stride + px
	if haveTop {
		sadV = 0
		for r := range n {
			row := (py+r)*stride + px
			for c := range n {
				d := int(srcPlane[row+c]) - int(reconPlane[above+c])
				if d < 0 {
					d = -d
				}
				sadV += d
			}
		}
	}
	if haveLeft {
		sadH = 0
		for r := range n {
			row := (py+r)*stride + px
			left := int(reconPlane[row-1])
			for c := range n {
				d := int(srcPlane[row+c]) - left
				if d < 0 {
					d = -d
				}
				sadH += d
			}
		}
	}
	switch {
	case sadV+16 < sadDC && sadV <= sadH:
		for r := range n {
			copy(pred[r*n:r*n+n], reconPlane[above:above+n])
		}
		return tile.IntraModeVertical
	case sadH+16 < sadDC:
		for r := range n {
			v := reconPlane[(py+r)*stride+px-1]
			for c := range n {
				pred[r*n+c] = v
			}
		}
		return tile.IntraModeHorizontal
	default:
		for i := range n * n {
			pred[i] = dc
		}
		return tile.IntraModeDC
	}
}

func BenchmarkSelectIntraModeN(b *testing.B) {
	const stride = 128
	src := make([]byte, stride*stride)
	recon := make([]byte, stride*stride)
	for i := range src {
		src[i] = byte((i*37 + i>>3) & 0xff)
		recon[i] = byte((i*29 + i>>5 + 17) & 0xff)
	}
	var pred [32 * 32]byte
	for _, n := range []int{8, 16, 32} {
		b.Run(fmt.Sprintf("fused/n%d", n), func(b *testing.B) {
			mode := tile.IntraModeDC
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				px := 1 + (i*7)%(stride-n-1)
				py := 1 + (i*11)%(stride-n-1)
				mode = selectIntraModeN(src, recon, stride, px, py, n, true, true, pred[:n*n])
			}
			benchmarkIntraModeSink = mode
			benchmarkIntraPredSink = pred[0]
		})
		b.Run(fmt.Sprintf("reference/n%d", n), func(b *testing.B) {
			mode := tile.IntraModeDC
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				px := 1 + (i*7)%(stride-n-1)
				py := 1 + (i*11)%(stride-n-1)
				mode = selectIntraModeNReference(src, recon, stride, px, py, n, true, true, pred[:n*n])
			}
			benchmarkIntraModeSink = mode
			benchmarkIntraPredSink = pred[0]
		})
	}
}

var (
	benchmarkIntraModeSink tile.IntraMode
	benchmarkIntraPredSink byte
)

// loopfilter_mask_diff_test.go is the fast differential gate for the encoder's
// mask-driven single-tile loop-filter apply. It fills a loop-filter record map,
// then deblocks two independent copies of the same reconstruction -- (a) the
// edge-list applySerial, (b) the mask-driven applySerialMasks -- and asserts the
// filtered planes are byte-identical. It runs in milliseconds and is the
// iteration gate for routing the single-tile serial callers onto masks.

// lfDiffFillPlane writes smoothly varying content so the deblock filter engages
// and any filter-width/level divergence changes pixels (a flat plane would make
// every filter a no-op and hide mask-class bugs).
func lfDiffFillPlane(pix []byte, stride, w, h, salt int) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + ((x*3 + y*5 + salt*11) % 13) - 6
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			pix[y*stride+x] = byte(v)
		}
	}
}

func lfDiffFrame(w, h int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	lfDiffFillPlane(f.Y, w, w, h, 0)
	lfDiffFillPlane(f.U, cw, cw, ch, 1)
	lfDiffFillPlane(f.V, cw, cw, ch, 2)
	return f
}

func lfDiffCopy(src SourceFrame420) SourceFrame420 {
	dst := src
	dst.Y = append([]byte(nil), src.Y...)
	dst.U = append([]byte(nil), src.U...)
	dst.V = append([]byte(nil), src.V...)
	return dst
}

func lfDiffAssertEqual(t *testing.T, a, b SourceFrame420) {
	t.Helper()
	for _, p := range []struct {
		name         string
		pa, pb       []byte
		stride, w, h int
	}{
		{"Y", a.Y, b.Y, a.YStride, a.Width, a.Height},
		{"U", a.U, b.U, a.ChromaStride, a.Width / 2, a.Height / 2},
		{"V", a.V, b.V, a.ChromaStride, a.Width / 2, a.Height / 2},
	} {
		for y := 0; y < p.h; y++ {
			for x := 0; x < p.w; x++ {
				if p.pa[y*p.stride+x] != p.pb[y*p.stride+x] {
					t.Fatalf("plane %s differs at x=%d y=%d: edge-list=%d mask=%d",
						p.name, x, y, p.pa[y*p.stride+x], p.pb[y*p.stride+x])
				}
			}
		}
	}
}

// lfDiffFillGrid tiles the record map with inter blocks of the given MI size and
// transform tree, in raster (== single-tile decode) order.
func lfDiffFillGrid(m threading.FrameWorkLoopFilterMap, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult) {
	stride := int(m.Stride)
	rows := int(m.Rows)
	for row := 0; row < rows; row += bh4 {
		for col := 0; col < stride; col += bw4 {
			rec := threading.FrameWorkLoopFilterBlockRecord{
				Valid:         true,
				TransformTree: tree,
				RefFrame:      1,
				Mode:          loopfilter.ModeDeltaClassZero,
				Block: threading.FrameWorkLoopFilterBlock{
					MICol: uint16(col), MIRow: uint16(row),
					MIColEnd: uint16(col + bw4), MIRowEnd: uint16(row + bh4),
					X4: uint8(col), Y4: uint8(row), Size: size,
					VisibleW4: uint8(bw4), VisibleH4: uint8(bh4),
				},
			}
			for r := row; r < row+bh4; r++ {
				for c := col; c < col+bw4; c++ {
					m.Records[r*stride+c] = rec
				}
			}
		}
	}
}

func lfDiffRun(t *testing.T, w, h, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, lf parser.LoopFilterParams) {
	t.Helper()
	var a loopFilterApplier
	if err := a.init(w, h); err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.close()
	if err := a.reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	lfDiffFillGrid(a.filtMap, bw4, bh4, size, tree)

	src := lfDiffFrame(w, h)
	edgeList := lfDiffCopy(src)
	masks := lfDiffCopy(src)

	if err := a.applySerial(&edgeList, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.applySerialMasks(&masks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	// Guard against a silently empty apply masking a false pass.
	changed := false
	for i := range src.Y {
		if edgeList.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: edge-list apply changed no luma pixels")
	}
	lfDiffAssertEqual(t, edgeList, masks)
}

func TestLoopFilterMaskDiffUniform16x16(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	lfDiffRun(t, 128, 128, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}, lf)
}

func TestLoopFilterMaskDiff8x8(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, LevelU: 24, LevelV: 22, Sharpness: 0}
	lfDiffRun(t, 128, 96, 2, 2, tile.BlockSize8x8,
		tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}, lf)
}

func TestLoopFilterMaskDiff32x32(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 2}
	lfDiffRun(t, 192, 128, 8, 8, tile.BlockSize32x32,
		tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}, lf)
}

// loopfilter_parallel_mask_diff_test.go is the fast differential gate for the
// encoder's parallel mask-driven single-tile loop-filter apply
// (loopFilterApplier.applyParallelMasks). Its core assertion is that the parallel
// banded mask apply is byte-identical to the serial single-shot mask apply
// (applySerialMasks) -- which the existing loopfilter_mask_diff_test.go already
// pins byte-identical to the edge-list applySerial -- so any divergence introduced
// by the band fan-out / direction barrier shows up as a changed pixel. It sweeps
// several region-row counts (frame heights) so the luma fan-out spans 1..5 bands,
// and at frame sizes the dense-grid edge-list planner still fits it also runs the
// full four-way check (serial edge-list, parallel edge-list, serial mask, parallel
// mask all agree). It runs in milliseconds; run under -race to prove the fan-out
// has no data races on the shared masks / level cache / reconstruction planes.

// lfParallelMaskRun deblocks copies of the same reconstruction with the serial
// mask apply (ground truth) and the parallel mask apply, and asserts byte-identical
// planes. When fourWay is set it additionally runs the serial and parallel
// edge-list applies and asserts the whole family agrees (only valid at frame sizes
// whose fully dense grid stays within the edge-list planner's edge-counter cap).
func lfParallelMaskRun(t *testing.T, w, h, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, lf parser.LoopFilterParams, fourWay bool) {
	t.Helper()
	var a loopFilterApplier
	if err := a.init(w, h); err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.close()
	if err := a.reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	lfDiffFillGrid(a.filtMap, bw4, bh4, size, tree)

	src := lfDiffFrame(w, h)
	serialMasks := lfDiffCopy(src)
	parMasks := lfDiffCopy(src)

	if err := a.applySerialMasks(&serialMasks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	if err := a.applyParallelMasks(&parMasks, lf); err != nil {
		t.Fatalf("applyParallelMasks: %v", err)
	}

	// Guard against a silently empty apply masking a false pass.
	changed := false
	for i := range src.Y {
		if serialMasks.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: serial mask apply changed no luma pixels")
	}
	lfDiffAssertEqual(t, serialMasks, parMasks)

	if !fourWay {
		return
	}
	serial := lfDiffCopy(src)
	parEdge := lfDiffCopy(src)
	if err := a.applySerial(&serial, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.apply(&parEdge, lf); err != nil {
		t.Fatalf("apply (parallel edge-list): %v", err)
	}
	lfDiffAssertEqual(t, serial, serialMasks)
	lfDiffAssertEqual(t, serial, parEdge)
	lfDiffAssertEqual(t, parEdge, parMasks)
}

func TestLoopFilterParallelMaskDiffUniform16x16(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	tree := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	// A 128x128 mask region covers 128px, so h=128,256,384,640 => 1,2,3,5 luma
	// region-row bands. fourWay only where the dense edge-list planner fits (<=256).
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{128, 128, true}, {192, 256, true}, {256, 384, false}, {192, 640, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 4, 4, tile.BlockSize16x16, tree, lf, c.fourWay)
		})
	}
}

func TestLoopFilterParallelMaskDiff8x8(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, LevelU: 24, LevelV: 22, Sharpness: 0}
	tree := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{128, 96, true}, {256, 256, true}, {192, 512, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 2, 2, tile.BlockSize8x8, tree, lf, c.fourWay)
		})
	}
}

func TestLoopFilterParallelMaskDiff32x32(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 2}
	tree := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	for _, c := range []struct {
		w, h    int
		fourWay bool
	}{{192, 128, true}, {256, 256, true}, {256, 640, false}} {
		t.Run(fmt.Sprintf("%dx%d", c.w, c.h), func(t *testing.T) {
			lfParallelMaskRun(t, c.w, c.h, 8, 8, tile.BlockSize32x32, tree, lf, c.fourWay)
		})
	}
}

// TestLoopFilterParallelMaskDiffChromaGated checks the chroma gating: a frame with
// a zero chroma base level must deblock luma only, and the parallel mask apply must
// skip the chroma planes exactly as the serial apply does (chroma == source).
func TestLoopFilterParallelMaskDiffChromaGated(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{24, 24}, LevelU: 0, LevelV: 0, Sharpness: 1}
	tree := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	lfParallelMaskRun(t, 256, 256, 4, 4, tile.BlockSize16x16, tree, lf, true)
}

func TestLoopFilterParallelMaskDiffHorizontalMixedWidthBoundary(t *testing.T) {
	for _, w := range []int{128, 136, 384} {
		t.Run(fmt.Sprintf("width-%d", w), func(t *testing.T) {
			const h = 256
			lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, Sharpness: 0}
			wideTree := tile.TransformTreeResult{Y: tile.TransformSize8x16, UV: tile.TransformSize4x4, HasUV: true}
			narrowTree := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}

			var a loopFilterApplier
			if err := a.init(w, h); err != nil {
				t.Fatalf("init: %v", err)
			}
			defer a.close()
			if err := a.reset(); err != nil {
				t.Fatalf("reset: %v", err)
			}

			for col := 0; col < int(a.filtMap.Stride); col += 4 {
				wideCols := min(2, int(a.filtMap.Stride)-col)
				for row := 0; row < int(a.filtMap.Rows); row += 4 {
					lfTilePutBlock(a.filtMap, col, row, wideCols, 4, tile.BlockSize8x16, wideTree, false)
				}
				narrowCol := col + wideCols
				if narrowCol >= int(a.filtMap.Stride) {
					continue
				}
				narrowCols := min(2, int(a.filtMap.Stride)-narrowCol)
				for row := 0; row < int(a.filtMap.Rows); row += 2 {
					lfTilePutBlock(a.filtMap, narrowCol, row, narrowCols, 2, tile.BlockSize8x8, narrowTree, false)
				}
			}

			src := lfDiffFrame(w, h)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					switch {
					case x%16 < 8 && y >= 128:
						src.Y[y*w+x] = 101
					case x%16 >= 8 && y >= 120:
						src.Y[y*w+x] = 108
					default:
						src.Y[y*w+x] = 100
					}
				}
			}
			serialMasks := lfDiffCopy(src)
			parallelMasks := lfDiffCopy(src)

			for iteration := range 100 {
				copy(serialMasks.Y, src.Y)
				copy(serialMasks.U, src.U)
				copy(serialMasks.V, src.V)
				copy(parallelMasks.Y, src.Y)
				copy(parallelMasks.U, src.U)
				copy(parallelMasks.V, src.V)

				if err := a.applySerialMasks(&serialMasks, lf); err != nil {
					t.Fatalf("iteration %d applySerialMasks: %v", iteration, err)
				}
				if got := serialMasks.Y[122*w+8]; got == 108 {
					t.Fatalf("iteration %d serial narrow q2=%d was not filtered", iteration, got)
				}
				if err := a.applyParallelMasks(&parallelMasks, lf); err != nil {
					t.Fatalf("iteration %d applyParallelMasks: %v", iteration, err)
				}
				lfDiffAssertEqual(t, serialMasks, parallelMasks)
			}
		})
	}
}

var _ = loopfilter.PlaneY

func TestAppendLoopFilterParamsPayloadLosslessDefaults(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas:              defaultLoopFilterDeltas(),
	}
	payload, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, true, lf, nil)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if !parsed.ModeRefDeltaEnabled || !parsed.ModeRefDeltaUpdate || parsed.Deltas != parserDefaultLoopFilterDeltas() {
		t.Fatalf("parsed lossless loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadLevelsAndUpdates(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	deltas := defaultLoopFilterDeltas()
	deltas.Ref[0] = -2
	deltas.Mode[1] = 3
	lf := LoopFilterParams{
		LevelY:              [2]uint8{10, 0},
		LevelU:              5,
		LevelV:              6,
		Sharpness:           3,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas:              deltas,
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, nil)
	if parsed.LevelY[0] != 10 || parsed.LevelY[1] != 0 || parsed.LevelU != 5 || parsed.LevelV != 6 || parsed.Sharpness != 3 {
		t.Fatalf("parsed levels=%+v", parsed)
	}
	if !parsed.ModeRefDeltaEnabled || !parsed.ModeRefDeltaUpdate ||
		parsed.Deltas.Ref[0] != -2 || parsed.Deltas.Ref[4] != -1 || parsed.Deltas.Mode[1] != 3 {
		t.Fatalf("parsed deltas=%+v", parsed.Deltas)
	}
}

func TestAppendLoopFilterParamsPayloadMonochromeSkipsUVLevels(t *testing.T) {
	seq := realtimeEncoderMonochromeSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	lf := LoopFilterParams{
		LevelY:    [2]uint8{12, 0},
		Sharpness: 1,
		Deltas:    defaultLoopFilterDeltas(),
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, nil)
	if parsed.LevelY[0] != 12 || parsed.LevelU != 0 || parsed.LevelV != 0 || parsed.Sharpness != 1 {
		t.Fatalf("parsed monochrome loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadCopiesPreviousDeltas(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	previous := defaultLoopFilterDeltas()
	previous.Ref[2] = 4
	lf := LoopFilterParams{
		Sharpness:           1,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  false,
		Deltas:              previous,
	}
	_, parsed := appendAndParseLoopFilterParams(t, seq, prefix, size, false, lf, &previous)
	if parsed.Deltas.Ref[2] != 4 || parsed.Deltas.Ref[4] != -1 || parsed.Sharpness != 1 {
		t.Fatalf("parsed previous loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	lf := LoopFilterParams{
		LevelY:              [2]uint8{8, 7},
		LevelU:              6,
		LevelV:              5,
		Sharpness:           2,
		ModeRefDeltaEnabled: false,
		Deltas:              defaultLoopFilterDeltas(),
	}
	var buf [8]byte
	payload, err := appendLoopFilterParamsInterPayload(buf[:0], seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("appendLoopFilterParamsInterPayload: %v", err)
	}
	parsed, err := parser.ParseLoopFilterParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, parser.FrameSize{}, parser.SegmentationParams{}, parser.DeltaParams{}, nil)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	if parsed.LevelY != [2]uint8{8, 7} || parsed.LevelU != 6 || parsed.LevelV != 5 || parsed.Sharpness != 2 || parsed.ModeRefDeltaEnabled {
		t.Fatalf("parsed inter loopfilter=%+v", parsed)
	}
}

func TestAppendLoopFilterParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	base := defaultLoopFilterDeltas()
	cases := [...]LoopFilterParams{
		{LevelY: [2]uint8{64, 0}, Deltas: base},
		{LevelU: 1, Deltas: base},
		{ModeRefDeltaEnabled: false, ModeRefDeltaUpdate: true, Deltas: base},
		{ModeRefDeltaEnabled: false, Deltas: LoopFilterDeltas{}},
	}
	var buf [8]byte
	for _, lf := range cases {
		if _, err := appendLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendLoopFilterParamsPayload(%+v) err=%v want ErrInvalidFrame", lf, err)
		}
	}
	lossless := LoopFilterParams{Deltas: base}
	if _, err := appendLoopFilterParamsPayload(buf[:0], seq, prefix, size, true, lossless, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless missing default flags err=%v want ErrInvalidFrame", err)
	}
	nonNone := FrameHeaderPrefix{PrimaryRefFrame: 0}
	if _, err := appendLoopFilterParamsPayload(buf[:0], seq, nonNone, size, false, LoopFilterParams{}, nil); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing previous err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseLoopFilterParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, parser.LoopFilterParams) {
	t.Helper()
	var buf [16]byte
	payload, err := appendLoopFilterParamsPayload(buf[:0], seq, prefix, size, allLossless, lf, previous)
	if err != nil {
		t.Fatalf("appendLoopFilterParamsPayload: %v", err)
	}
	parsedPrevious := parserLoopFilterDeltas(previous)
	parsed, err := parser.ParseLoopFilterParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameHeaderPrefix{PrimaryRefFrame: prefix.PrimaryRefFrame},
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.DeltaParams{},
		parsedPrevious,
	)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	return payload, parsed
}

func parserLoopFilterDeltas(previous *LoopFilterDeltas) *parser.LoopFilterDeltas {
	if previous == nil {
		return nil
	}
	var out parser.LoopFilterDeltas
	for i := uint8(0); i < encoderRefFrames; i++ {
		out.Ref[i] = previous.Ref[i]
	}
	for i := uint8(0); i < LoopFilterModeDeltas; i++ {
		out.Mode[i] = previous.Mode[i]
	}
	return &out
}

func parserDefaultLoopFilterDeltas() parser.LoopFilterDeltas {
	return *parserLoopFilterDeltasPtr(defaultLoopFilterDeltas())
}

func parserLoopFilterDeltasPtr(deltas LoopFilterDeltas) *parser.LoopFilterDeltas {
	return parserLoopFilterDeltas(&deltas)
}

// loopfilter_tile_diff_test.go is the CI-portable reproduction of the mask-build
// order bug the RealC recon oracle exposed: raster-by-origin order does not
// reproduce dav1d's Z-order left/above tx-context when a taller block on the
// right of a superblock boundary has a smaller origin row than the shorter
// block on its left (the "staircase"). The mask build must derive neighbour
// context from the shared record map (== what the edge-list sweep reads), so the
// two agree byte-for-byte. These synthetic maps engineer the staircase and a
// variable-transform neighbour so the boundary vertical edge width depends on the
// left neighbour, and assert sweep == mask.

func lfTilePutBlock(m threading.FrameWorkLoopFilterMap, col, row, bw4, bh4 int, size tile.BlockSize, tree tile.TransformTreeResult, intra bool) {
	stride := int(m.Stride)
	rec := threading.FrameWorkLoopFilterBlockRecord{
		Valid:         true,
		TransformTree: tree,
		RefFrame:      1,
		Intra:         intra,
		Mode:          loopfilter.ModeDeltaClassZero,
		Block: threading.FrameWorkLoopFilterBlock{
			MICol: uint16(col), MIRow: uint16(row),
			MIColEnd: uint16(col + bw4), MIRowEnd: uint16(row + bh4),
			X4: uint8(col), Y4: uint8(row), Size: size,
			VisibleW4: uint8(bw4), VisibleH4: uint8(bh4),
		},
	}
	if intra {
		rec.RefFrame = 0
	}
	for r := row; r < row+bh4; r++ {
		for c := col; c < col+bw4; c++ {
			m.Records[r*stride+c] = rec
		}
	}
}

func lfTileFirstDiff(t *testing.T, a, b SourceFrame420) bool {
	t.Helper()
	for _, p := range []struct {
		name         string
		pa, pb       []byte
		stride, w, h int
	}{
		{"Y", a.Y, b.Y, a.YStride, a.Width, a.Height},
		{"U", a.U, b.U, a.ChromaStride, a.Width / 2, a.Height / 2},
		{"V", a.V, b.V, a.ChromaStride, a.Width / 2, a.Height / 2},
	} {
		for y := 0; y < p.h; y++ {
			for x := 0; x < p.w; x++ {
				if p.pa[y*p.stride+x] != p.pb[y*p.stride+x] {
					t.Logf("plane %s first diff at x=%d y=%d: sweep=%d mask=%d", p.name, x, y, p.pa[y*p.stride+x], p.pb[y*p.stride+x])
					return true
				}
			}
		}
	}
	return false
}

// lfTileCompareSweepMask fills the record map via fill, then deblocks copies of
// the same reconstruction with the edge-list sweep (ground truth) and the mask
// apply and asserts byte-identical planes.
func lfTileCompareSweepMask(t *testing.T, w, h int, lf parser.LoopFilterParams, fill func(m threading.FrameWorkLoopFilterMap)) {
	t.Helper()
	var a loopFilterApplier
	if err := a.init(w, h); err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.close()
	if err := a.reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	fill(a.filtMap)

	src := lfDiffFrame(w, h)
	sweep := lfDiffCopy(src)
	masks := lfDiffCopy(src)
	if err := a.applySerial(&sweep, lf); err != nil {
		t.Fatalf("applySerial: %v", err)
	}
	if err := a.applySerialMasks(&masks, lf); err != nil {
		t.Fatalf("applySerialMasks: %v", err)
	}
	changed := false
	for i := range src.Y {
		if sweep.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("degenerate: sweep apply changed no luma pixels")
	}
	if lfTileFirstDiff(t, sweep, masks) {
		lfDiffAssertEqual(t, sweep, masks)
	}
}

// TestLoopFilterMaskDiffStaircaseSBBoundary is an end-to-end positive check of
// the staircase configuration (a 16x16 column right of a 64px superblock
// boundary abutting shorter 8x8s whose origin rows exceed it): the mask apply
// must equal the edge-list sweep. The content-independent bit-level proof of the
// order fix lives in threading TestBuildFromMapMatchesZOrderStaircase.
func TestLoopFilterMaskDiffStaircaseSBBoundary(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{34, 30}, LevelU: 28, LevelV: 26, Sharpness: 1}
	tree8 := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	tree16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	tree32 := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	const w, h = 128, 128
	lfTileCompareSweepMask(t, w, h, lf, func(m threading.FrameWorkLoopFilterMap) {
		stride := int(m.Stride)
		rows := int(m.Rows)
		// Left superblock [cols 0..15]: 32x32 to the left, then a column of 8x8s
		// filling the two cell-columns just left of the SB boundary (col 16). The
		// 8x8 origin rows (0,2,4,...) exceed the 16x16 right-block origin rows on
		// odd 16x16 rows, so raster-by-origin order reads a stale class-2 left
		// context (the reset value), while the true 8x8 right tx is class 1.
		for row := 0; row < rows; row += 8 {
			for col := 0; col < 8; col += 8 {
				lfTilePutBlock(m, col, row, 8, 8, tile.BlockSize32x32, tree32, true)
			}
		}
		for row := 0; row < rows; row += 2 {
			for col := 8; col < 16; col += 2 {
				lfTilePutBlock(m, col, row, 2, 2, tile.BlockSize8x8, tree8, true)
			}
		}
		// Right superblock [cols 16..31]: 16x16 blocks (origin rows 0,4,8,...),
		// whose left edge at rows 2,3,6,7,... abuts an 8x8 whose origin row is
		// LARGER, so raster order has not yet processed it.
		for row := 0; row < rows; row += 4 {
			for col := 16; col < 32; col += 4 {
				lfTilePutBlock(m, col, row, 4, 4, tile.BlockSize16x16, tree16, true)
			}
		}
		// Far right: 32x32 context.
		for row := 0; row < rows; row += 8 {
			for col := 32; col < stride; col += 8 {
				lfTilePutBlock(m, col, row, 8, 8, tile.BlockSize32x32, tree32, true)
			}
		}
	})
}

// TestLoopFilterMaskDiffVariableTxNeighbor engineers a variable-transform inter
// block whose right column carries mixed tx sizes, abutting a fixed-tx block, so
// the boundary edge width derivation depends on decomposing the variable
// neighbour's right-edge tx per row.
func TestLoopFilterMaskDiffVariableTxNeighbor(t *testing.T) {
	lf := parser.LoopFilterParams{LevelY: [2]uint8{40, 40}, LevelU: 32, LevelV: 30, Sharpness: 0}
	// A 16x16 inter block coded with a split transform tree: depth-0 split of the
	// 16x16 into 8x8 sub-transforms. Split[0] bit for yOff=0,xOff=0.
	varTree := tile.TransformTreeResult{
		Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true,
		Variable: true, Split: [2]uint16{0xffff, 0xffff},
	}
	fixed16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	const w, h = 128, 128
	lfTileCompareSweepMask(t, w, h, lf, func(m threading.FrameWorkLoopFilterMap) {
		stride := int(m.Stride)
		rows := int(m.Rows)
		// Alternate 16x16 blocks between a split (variable) tree and a fixed
		// 16x16 tx, so each vertical block boundary derives its width from a
		// variable neighbour's right-column tx on one side.
		for row := 0; row < rows; row += 4 {
			for col := 0; col < stride; col += 4 {
				tree := fixed16
				if (col/4+row/4)%2 == 0 {
					tree = varTree
				}
				lfTilePutBlock(m, col, row, 4, 4, tile.BlockSize16x16, tree, false)
			}
		}
	})
}

// TestLPD1SampledSADMatchesPureGo checks the dispatched sampled SAD against
// its reference over the detector's widths, row counts and steps.
func TestLPD1SampledSADMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const stride = 80
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 4000; trial++ {
		for i := range src {
			src[i], ref[i] = byte(rng.Intn(256)), byte(rng.Intn(256))
		}
		w := []int{4, 8, 16, 32}[rng.Intn(4)]
		rows := 1 + rng.Intn(8)
		step := 1 << rng.Intn(3)
		maxOff := stride*40 - ((rows-1)*step*stride + w)
		so, ro := rng.Intn(maxOff), rng.Intn(maxOff)
		got := lpd1SampledSAD(src, ref, so, ro, stride, stride, w, rows, step)
		want := lpd1SampledSADPureGo(src, ref, so, ro, stride, stride, w, rows, step)
		if got != want {
			t.Fatalf("w=%d rows=%d step=%d: got %d want %d", w, rows, step, got, want)
		}
	}
}

func benchPixelStats(b *testing.B, fn func([]byte, int, []byte, int) (uint32, int32)) {
	rng := rand.New(rand.NewSource(6103))
	const (
		srcStride = 1920
		refStride = 1920
		height    = 80
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	src = src[srcStride+3:]
	ref = ref[2*refStride+5:]
	var sse uint32
	var sum int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sse, sum = fn(src, srcStride, ref, refStride)
	}
	if sse == 0 && sum == 0 {
		b.Fatal("unexpected zero metric")
	}
}

func BenchmarkPixelStats8x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x8PureGo)
}

func BenchmarkPixelStats16x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x16PureGo)
}

func BenchmarkPixelStats32x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x32PureGo)
}

func TestPixelStatsImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(6101))
	const (
		srcStride = 151
		refStride = 157
		height    = 144
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, sh := range []struct {
		w, h int
		pure func([]byte, int, []byte, int) (uint32, int32)
		impl func([]byte, int, []byte, int) (uint32, int32)
	}{
		{w: 4, h: 4, pure: pixelStats4x4PureGo, impl: pixelStats4x4Impl},
		{w: 8, h: 8, pure: pixelStats8x8PureGo, impl: pixelStats8x8Impl},
		{w: 16, h: 8, pure: pixelStats16x8PureGo, impl: pixelStats16x8Impl},
		{w: 8, h: 16, pure: pixelStats8x16PureGo, impl: pixelStats8x16Impl},
		{w: 16, h: 16, pure: pixelStats16x16PureGo, impl: pixelStats16x16Impl},
		{w: 32, h: 16, pure: pixelStats32x16PureGo, impl: pixelStats32x16Impl},
		{w: 16, h: 32, pure: pixelStats16x32PureGo, impl: pixelStats16x32Impl},
		{w: 32, h: 32, pure: pixelStats32x32PureGo, impl: pixelStats32x32Impl},
		{w: 64, h: 32, pure: pixelStats64x32PureGo, impl: pixelStats64x32Impl},
		{w: 32, h: 64, pure: pixelStats32x64PureGo, impl: pixelStats32x64Impl},
	} {
		for range 1000 {
			srow := rng.Intn(height - sh.h)
			scol := rng.Intn(srcStride - sh.w)
			rrow := rng.Intn(height - sh.h)
			rcol := rng.Intn(refStride - sh.w)
			srcOff := srow*srcStride + scol
			refOff := rrow*refStride + rcol
			wantSSE, wantSum := sh.pure(src[srcOff:], srcStride, ref[refOff:], refStride)
			gotSSE, gotSum := sh.impl(src[srcOff:], srcStride, ref[refOff:], refStride)
			if gotSSE != wantSSE || gotSum != wantSum {
				t.Fatalf("%dx%d srcOff=%d refOff=%d: got sse=%d sum=%d want sse=%d sum=%d",
					sh.w, sh.h, srcOff, refOff, gotSSE, gotSum, wantSSE, wantSum)
			}
		}
	}
}

func TestSSEVarianceMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6102))
	const (
		srcStride = 192
		refStride = 208
		height    = 128
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, sh := range []struct {
		w, h  int
		fn    func([]byte, int, []byte, int) (uint32, uint32)
		shift uint
	}{
		{w: 4, h: 4, fn: sseVariance4x4, shift: 4},
		{w: 8, h: 8, fn: sseVariance8x8, shift: 6},
		{w: 16, h: 8, fn: sseVariance16x8, shift: 7},
		{w: 8, h: 16, fn: sseVariance8x16, shift: 7},
		{w: 16, h: 16, fn: sseVariance16x16, shift: 8},
		{w: 32, h: 16, fn: sseVariance32x16, shift: 9},
		{w: 16, h: 32, fn: sseVariance16x32, shift: 9},
		{w: 32, h: 32, fn: sseVariance32x32, shift: 10},
		{w: 64, h: 64, fn: sseVariance64x64, shift: 12},
	} {
		for range 500 {
			srow := rng.Intn(height - sh.h)
			scol := rng.Intn(srcStride - sh.w)
			rrow := rng.Intn(height - sh.h)
			rcol := rng.Intn(refStride - sh.w)
			srcOff := srow*srcStride + scol
			refOff := rrow*refStride + rcol
			wantSSE, wantSum := pixelStatsPureGo(src[srcOff:], srcStride, ref[refOff:], refStride, sh.w, sh.h)
			wantVar := varianceFromStats(wantSSE, wantSum, sh.shift)
			gotSSE, gotVar := sh.fn(src[srcOff:], srcStride, ref[refOff:], refStride)
			if gotSSE != wantSSE || gotVar != wantVar {
				t.Fatalf("%dx%d srcOff=%d refOff=%d: got sse=%d var=%d want sse=%d var=%d",
					sh.w, sh.h, srcOff, refOff, gotSSE, gotVar, wantSSE, wantVar)
			}
		}
	}
}

type payloadWrite func(*bitWriter) error

func payloadSize(write payloadWrite) (int, error) {
	w := newSizingBitWriter()
	if err := write(&w); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func appendPayload(dst []byte, write payloadWrite) ([]byte, error) {
	n, err := payloadSize(write)
	if err != nil {
		return dst, err
	}
	out := append(dst, make([]byte, n)...)
	w := newBitWriter(out[len(dst):])
	if err := write(&w); err != nil {
		return dst, err
	}
	if w.bytesWritten() != n {
		return dst, fmt.Errorf("payload wrote %d bytes, sized %d", w.bytesWritten(), n)
	}
	return out, nil
}

func cdefWrite(seq SequenceHeader, allowIntrabc, allLossless bool, cdef CDEFParams) payloadWrite {
	return func(w *bitWriter) error { return writeCDEFParamsPayload(w, seq, allowIntrabc, allLossless, cdef) }
}

func appendCDEFParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	return appendPayload(dst, cdefWrite(seq, size.AllowIntrabc, allLossless, cdef))
}

func appendCDEFParamsInterPayload(dst []byte, seq SequenceHeader, _ InterFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	return appendPayload(dst, cdefWrite(seq, false, allLossless, cdef))
}

func deltaWrite(allowIntrabc bool, quant QuantizationParams, delta DeltaParams) payloadWrite {
	return func(w *bitWriter) error { return writeDeltaParamsPayload(w, allowIntrabc, quant, delta) }
}

func appendDeltaParamsPayload(dst []byte, size IntraFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	return appendPayload(dst, deltaWrite(size.AllowIntrabc, quant, delta))
}

func appendDeltaParamsInterPayload(dst []byte, _ InterFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	return appendPayload(dst, deltaWrite(false, quant, delta))
}

func appendFilmGrainParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeFilmGrainParamsPayload(w, seq, prefix, size, refs, params)
	})
}

func appendFrameHeaderPrefixPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeFrameHeaderPrefixPayload(w, seq, prefix) })
}

func appendIntraFrameHeaderPayload(dst []byte, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeIntraFrameHeaderPayload(w, seq, header) })
}

func appendInterFrameHeaderPayload(dst []byte, seq SequenceHeader, header InterFrameHeaderParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeInterFrameHeaderPayload(w, seq, header) })
}

func appendFrameModeParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeFrameModeParamsPayload(w, seq, prefix, params) })
}

// appendFrameHeaderIntraPayload writes the frame header through frame_size()
// on the intra path.
func appendFrameHeaderIntraPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		if err := writeFrameHeaderPrefixPayload(w, seq, prefix); err != nil {
			return err
		}
		return writeIntraFrameSizePayload(w, seq, prefix, size)
	})
}

// appendFrameHeaderInterPayload writes the frame header through frame_size()
// on the inter/switch path.
func appendFrameHeaderInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		if err := writeFrameHeaderPrefixPayload(w, seq, prefix); err != nil {
			return err
		}
		return writeInterFrameSizePayload(w, seq, prefix, size)
	})
}

func appendGlobalMotionParamsPayload(dst []byte, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeGlobalMotionParamsPayload(w, prefix, size, tiles, refs, params)
	})
}

func loopFilterWrite(seq SequenceHeader, prefix FrameHeaderPrefix, allowIntrabc, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) payloadWrite {
	return func(w *bitWriter) error {
		return writeLoopFilterParamsPayload(w, seq, prefix, allowIntrabc, allLossless, lf, previous)
	}
}

func appendLoopFilterParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	return appendPayload(dst, loopFilterWrite(seq, prefix, size.AllowIntrabc, allLossless, lf, previous))
}

func appendLoopFilterParamsInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, _ InterFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	return appendPayload(dst, loopFilterWrite(seq, prefix, false, allLossless, lf, previous))
}

func appendQuantizationParamsPayload(dst []byte, seq SequenceHeader, quant QuantizationParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeQuantizationParamsPayload(w, seq, quant) })
}

func restorationWrite(seq SequenceHeader, allowIntrabc, superRes, allLossless bool, restoration RestorationParams) payloadWrite {
	return func(w *bitWriter) error {
		return writeRestorationParamsPayload(w, seq, allowIntrabc, superRes, allLossless, restoration)
	}
}

func appendRestorationParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	return appendPayload(dst, restorationWrite(seq, size.AllowIntrabc, size.SuperResDenominator != 8, allLossless, restoration))
}

func appendRestorationParamsInterPayload(dst []byte, seq SequenceHeader, size InterFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	return appendPayload(dst, restorationWrite(seq, false, size.SuperResDenominator != 8, allLossless, restoration))
}

func appendSegmentationParamsPayload(dst []byte, prefix FrameHeaderPrefix, seg SegmentationParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeSegmentationParamsPayload(w, prefix, seg) })
}

func appendSequenceHeaderPayload(dst []byte, seq SequenceHeader) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeSequenceHeaderPayload(w, seq) })
}

func appendSkipModeParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeSkipModeParamsPayload(w, seq, prefix, size, refs, transformRef, params)
	})
}

func appendTileInfoPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeTileInfoPayload(w, seq, prefix, codedWidth, height, tiles)
	})
}

func appendTransformReferenceParamsPayload(dst []byte, prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeTransformReferenceParamsPayload(w, prefix, allLossless, params)
	})
}

func TestRealtimeSADPerBitMatchesLibaomMELUT(t *testing.T) {
	for _, tc := range []struct {
		qIndex   uint8
		bitDepth uint8
		scale    float64
	}{
		{qIndex: 37, bitDepth: 8, scale: 4},
		{qIndex: 160, bitDepth: 8, scale: 4},
		{qIndex: 255, bitDepth: 8, scale: 4},
		{qIndex: 160, bitDepth: 10, scale: 16},
		{qIndex: 160, bitDepth: 12, scale: 64},
	} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, tc.qIndex, tc.bitDepth, quantize.PlaneY)
		if err != nil {
			t.Fatalf("PlaneQuantizer(q=%d bd=%d): %v", tc.qIndex, tc.bitDepth, err)
		}
		got := realtimeSADPerBit(q, tc.bitDepth)
		want := int(0.0418*(float64(q.AC)/tc.scale) + 2.4107)
		if got != want {
			t.Fatalf("q=%d bd=%d sadPerBit=%d want %d", tc.qIndex, tc.bitDepth, got, want)
		}
		dcBased := int(0.0418*(float64(q.DC)/tc.scale) + 2.4107)
		if q.DC != q.AC && dcBased != want && got == dcBased {
			t.Fatalf("q=%d bd=%d used DC quantizer: got %d want AC-based %d", tc.qIndex, tc.bitDepth, got, want)
		}
	}
}

func TestRealtimeInterTXLeafSizeMatchesLibaomCalculateTXSize(t *testing.T) {
	const (
		qAC     = int32(88)
		txLevel = 2
	)
	for _, tc := range []struct {
		name      string
		block     tile.BlockSize
		qIndex    uint8
		sse       uint32
		variance  uint32
		wantLeaf  int
		wantSplit [2]uint16
	}{
		{name: "high variance keeps 8x8", block: tile.BlockSize16x16, qIndex: 72, sse: 1000, variance: 1000, wantLeaf: 8, wantSplit: [2]uint16{1, 0}},
		{name: "dc dominated uses max square", block: tile.BlockSize16x16, qIndex: 72, sse: 2000, variance: 1000, wantLeaf: 16},
		{name: "low ac variance uses max square", block: tile.BlockSize16x16, qIndex: 72, sse: 200, variance: 100, wantLeaf: 16},
		{name: "rect max square is 8x8", block: tile.BlockSize16x8, qIndex: 72, sse: 2000, variance: 1000, wantLeaf: 8, wantSplit: [2]uint16{1, 0}},
		{name: "32x32 max square capped to 16x16", block: tile.BlockSize32x32, qIndex: 72, sse: 5000, variance: 1000, wantLeaf: 16, wantSplit: [2]uint16{1, 0}},
		{name: "larger than 32 forced to 16x16", block: tile.BlockSize64x64, qIndex: 72, sse: 1000, variance: 1000, wantLeaf: 16, wantSplit: [2]uint16{1, 0x0033}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotLeaf, err := realtimeInterTXLeafSizeForBlock(tc.block, tc.qIndex, qAC, txLevel, tc.sse, tc.variance)
			if err != nil {
				t.Fatal(err)
			}
			if gotLeaf != tc.wantLeaf {
				t.Fatalf("leaf=%d want %d", gotLeaf, tc.wantLeaf)
			}
			plan, err := realtimeInterTXPlanForBlock(tc.block, tc.qIndex, qAC, txLevel, tc.sse, tc.variance)
			if err != nil {
				t.Fatal(err)
			}
			if plan.leafSize != tc.wantLeaf {
				t.Fatalf("plan leaf=%d want %d", plan.leafSize, tc.wantLeaf)
			}
			if plan.tree.Split != tc.wantSplit {
				t.Fatalf("split=%#v want %#v", plan.tree.Split, tc.wantSplit)
			}
		})
	}
}

func TestRealtimeTXSizeLevelBasedOnQstepMatchesSpeedFeatures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		w, h   int
		effort int8
		speed  int
		level  int
	}{
		{name: "1080p default speed8 level2", w: 1920, h: 1080, effort: 0, speed: 8, level: 2},
		{name: "1080p fastest speed10 disables qstep level", w: 1920, h: 1080, effort: WebRTCMinEffortLevel, speed: 10, level: 0},
		{name: "1080p max effort speed4 disables qstep level", w: 1920, h: 1080, effort: WebRTCMaxEffortLevel, speed: 4, level: 0},
		{name: "360p default speed8 level2", w: 640, h: 360, effort: 0, speed: 8, level: 2},
		{name: "below 360p default speed8 level1", w: 320, h: 180, effort: 0, speed: 8, level: 1},
		{name: "below 360p fastest speed10 keeps level1", w: 320, h: 180, effort: WebRTCMinEffortLevel, speed: 10, level: 1},
		{name: "720p speed7 disables qstep level", w: 1280, h: 720, effort: 1, speed: 7, level: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := realtimeLibaomSpeedForEffort(tc.effort); got != tc.speed {
				t.Fatalf("speed=%d want %d", got, tc.speed)
			}
			if got := realtimeTXSizeLevelBasedOnQstep(tc.w, tc.h, tc.effort); got != tc.level {
				t.Fatalf("level=%d want %d", got, tc.level)
			}
		})
	}
}

func TestRealtimeInterTXLeafSizeHonorsQstepSpeedFeatureLevel(t *testing.T) {
	const (
		qIndex   = uint8(72)
		qAC      = int32(88)
		sse      = uint32(100)
		variance = uint32(100)
	)
	level0, err := realtimeInterTXLeafSizeForBlock(tile.BlockSize16x16, qIndex, qAC, 0, sse, variance)
	if err != nil {
		t.Fatal(err)
	}
	if level0 != 8 {
		t.Fatalf("level0 leaf=%d want 8", level0)
	}
	level2, err := realtimeInterTXLeafSizeForBlock(tile.BlockSize16x16, qIndex, qAC, 2, sse, variance)
	if err != nil {
		t.Fatal(err)
	}
	if level2 != 16 {
		t.Fatalf("level2 leaf=%d want 16", level2)
	}
}

func TestRealtimeInterTXPlanReplayMatchesTransformTree(t *testing.T) {
	for _, tc := range []struct {
		name       string
		size       tile.BlockSize
		leaf       int
		wantSplit  [2]uint16
		wantLeaves int
	}{
		{name: "8x8 leaf8", size: tile.BlockSize8x8, leaf: 8, wantLeaves: 1},
		{name: "16x8 leaf8", size: tile.BlockSize16x8, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "8x16 leaf8", size: tile.BlockSize8x16, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "16x16 leaf8", size: tile.BlockSize16x16, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 4},
		{name: "32x16 leaf8", size: tile.BlockSize32x16, leaf: 8, wantSplit: [2]uint16{1, 0x0003}, wantLeaves: 8},
		{name: "16x32 leaf8", size: tile.BlockSize16x32, leaf: 8, wantSplit: [2]uint16{1, 0x0011}, wantLeaves: 8},
		{name: "32x32 leaf8", size: tile.BlockSize32x32, leaf: 8, wantSplit: [2]uint16{1, 0x0033}, wantLeaves: 16},
		{name: "32x16 leaf16", size: tile.BlockSize32x16, leaf: 16, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "32x32 leaf16", size: tile.BlockSize32x32, leaf: 16, wantSplit: [2]uint16{1, 0}, wantLeaves: 4},
		{name: "64x16 leaf16", size: tile.BlockSize64x16, leaf: 16, wantSplit: [2]uint16{1, 0x0003}, wantLeaves: 4},
		{name: "16x64 leaf16", size: tile.BlockSize16x64, leaf: 16, wantSplit: [2]uint16{1, 0x0011}, wantLeaves: 4},
		{name: "64x64 leaf16", size: tile.BlockSize64x64, leaf: 16, wantSplit: [2]uint16{1, 0x0033}, wantLeaves: 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := realtimeInterTXPlanForLeafSize(tc.size, tc.leaf)
			if err != nil {
				t.Fatal(err)
			}
			if plan.tree.Split != tc.wantSplit {
				t.Fatalf("split=%#v want %#v", plan.tree.Split, tc.wantSplit)
			}

			var helperLeaves []tile.TransformBlock
			err = plan.ForEachLeaf(func(i, dx, dy int) error {
				helperLeaves = append(helperLeaves, tile.TransformBlock{
					X4:        uint8(dx / 4),
					Y4:        uint8(dy / 4),
					Size:      plan.leafTX,
					VisibleW4: uint8(tc.leaf / 4),
					VisibleH4: uint8(tc.leaf / 4),
				})
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(helperLeaves) != tc.wantLeaves {
				t.Fatalf("helper leaves=%d want %d", len(helperLeaves), tc.wantLeaves)
			}

			dims, ok := tc.size.Dimensions()
			if !ok {
				t.Fatal("bad block size")
			}
			maxY, err := tile.MaxTransformSize(tc.size, parser.ColorConfig{}, 0)
			if err != nil {
				t.Fatal(err)
			}
			tree := plan.tree
			tree.Y = maxY
			tree.Variable = plan.Variable()
			var replayLeaves []tile.TransformBlock
			err = tree.ForEachLumaTXB(tile.TransformTreeRequest{
				Size:          tc.size,
				VisibleW4:     dims.W4,
				VisibleH4:     dims.H4,
				TransformMode: parser.TransformModeSwitchable,
				Inter:         true,
			}, func(block tile.TransformBlock) error {
				replayLeaves = append(replayLeaves, block)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(replayLeaves) != len(helperLeaves) {
				t.Fatalf("replay leaves=%d helper=%d replay=%+v helper=%+v", len(replayLeaves), len(helperLeaves), replayLeaves, helperLeaves)
			}
			for i := range replayLeaves {
				if replayLeaves[i] != helperLeaves[i] {
					t.Fatalf("leaf[%d]=%+v helper %+v", i, replayLeaves[i], helperLeaves[i])
				}
			}
		})
	}
}

func TestEncodePBlockCompoundLastGolden8x8(t *testing.T) {
	const w, h = 16, 16
	solid := func(y, u, v byte) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = y
		}
		for i := range f.U {
			f.U[i] = u
			f.V[i] = v
		}
		return f
	}
	src := solid(128, 128, 128)
	ref := solid(200, 180, 90)
	golden := solid(56, 76, 166)
	recon := solid(0, 0, 0)

	var pc pframeCoder
	if err := pc.reset(72, 1, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      72,
	}
	st.afterSkipInter = func() error {
		return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, transform.TypeDCTDCT)
	}
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	st.sadPerBit = realtimeSADPerBit(st.yQuant, 8)

	block := tile.BlockVisit{
		MIColEnd:  2,
		MIRowEnd:  2,
		Size:      tile.BlockSize8x8,
		VisibleW4: 2,
		VisibleH4: 2,
	}
	walkReq := tile.BlockWalkRequest{MIColEnd: 4, MIRowEnd: 4}
	if err := st.encodePBlock(src, ref, &golden, &recon, block, &pc.scratch, &pc.refCDFs, &pc.modeCDFs, &pc.interpCDFs, parser.ReferenceModeSelect, walkReq, 4, 4); err != nil {
		t.Fatal(err)
	}

	got := pc.scratch.Mode.AboveInterMotion[0]
	if !got.References.Compound ||
		got.References.Ref0 != tile.ReferenceFrameLast ||
		got.References.Ref1 != tile.ReferenceFrameGolden ||
		got.Mode.CompoundMode != tile.CompoundInterModeGlobalGlobal {
		t.Fatalf("motion = %+v, want GLOBAL_GLOBAL LAST+GOLDEN compound", got)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if recon.Y[y*w+x] != src.Y[y*w+x] {
				t.Fatalf("recon Y(%d,%d)=%d want %d", x, y, recon.Y[y*w+x], src.Y[y*w+x])
			}
		}
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if recon.U[y*(w/2)+x] != src.U[y*(w/2)+x] || recon.V[y*(w/2)+x] != src.V[y*(w/2)+x] {
				t.Fatalf("recon chroma(%d,%d)=(%d,%d) want (%d,%d)", x, y, recon.U[y*(w/2)+x], recon.V[y*(w/2)+x], src.U[y*(w/2)+x], src.V[y*(w/2)+x])
			}
		}
	}
}

func TestEncodePBlockGoldenSingleLarge(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
		size tile.BlockSize
		tx   tile.TransformSize
	}{
		{name: "16x16", w: 16, h: 16, size: tile.BlockSize16x16, tx: tile.TransformSize16x16},
		{name: "32x32", w: 32, h: 32, size: tile.BlockSize32x32, tx: tile.TransformSize32x32},
		{name: "16x8", w: 16, h: 8, size: tile.BlockSize16x8, tx: tile.TransformSize16x8},
		{name: "8x16", w: 8, h: 16, size: tile.BlockSize8x16, tx: tile.TransformSize8x16},
		{name: "32x16", w: 32, h: 16, size: tile.BlockSize32x16, tx: tile.TransformSize32x16},
		{name: "16x32", w: 16, h: 32, size: tile.BlockSize16x32, tx: tile.TransformSize16x32},
		{name: "64x16", w: 64, h: 16, size: tile.BlockSize64x16, tx: tile.TransformSize64x16},
		{name: "16x64", w: 16, h: 64, size: tile.BlockSize16x64, tx: tile.TransformSize16x64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, h := tc.w, tc.h
			solid := func(y, u, v byte) SourceFrame420 {
				f := SourceFrame420{
					Y:            make([]byte, w*h),
					U:            make([]byte, w*h/4),
					V:            make([]byte, w*h/4),
					YStride:      w,
					ChromaStride: w / 2,
					Width:        w,
					Height:       h,
				}
				for i := range f.Y {
					f.Y[i] = y
				}
				for i := range f.U {
					f.U[i] = u
					f.V[i] = v
				}
				return f
			}
			src := solid(56, 76, 166)
			ref := solid(200, 180, 90)
			golden := solid(56, 76, 166)
			recon := solid(0, 0, 0)

			var pc pframeCoder
			if err := pc.reset(72, 1, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
				t.Fatal(err)
			}
			st := &pc.st
			pc.writer.Reset(pc.writerBuf[:0])
			st.w = &pc.writer
			st.interTxTypeReq = tile.InterTransformTypeRequest{
				Size:        tc.tx,
				QIndexKnown: true,
				QIndex:      72,
			}
			st.afterSkipInter = func() error {
				return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, transform.TypeDCTDCT)
			}
			dcq := float64(st.yQuant.DC)
			st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
			st.sadPerBit = realtimeSADPerBit(st.yQuant, 8)
			miW, miH := uint16(w/4), uint16(h/4)
			block := tile.BlockVisit{
				MIColEnd:  miW,
				MIRowEnd:  miH,
				Size:      tc.size,
				VisibleW4: uint8(miW),
				VisibleH4: uint8(miH),
			}
			walkReq := tile.BlockWalkRequest{MIColEnd: miW, MIRowEnd: miH}
			if err := st.encodePBlock(src, ref, &golden, &recon, block, &pc.scratch, &pc.refCDFs, &pc.modeCDFs, &pc.interpCDFs, parser.ReferenceModeSingle, walkReq, miW, miH); err != nil {
				t.Fatal(err)
			}

			got := pc.scratch.Mode.AboveInterMotion[0]
			if got.References.Compound ||
				got.References.Ref0 != tile.ReferenceFrameGolden ||
				got.References.Ref1 != tile.ReferenceFrameNone ||
				got.Mode.Mode != tile.InterModeGlobalMV {
				t.Fatalf("motion = %+v, want GLOBALMV single GOLDEN", got)
			}
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if recon.Y[y*w+x] != src.Y[y*w+x] {
						t.Fatalf("recon Y(%d,%d)=%d want %d", x, y, recon.Y[y*w+x], src.Y[y*w+x])
					}
				}
			}
		})
	}
}

func TestCompoundGoldenLikely(t *testing.T) {
	const w, h = 16, 16
	solid := func(y byte) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = y
		}
		for i := range f.U {
			f.U[i] = 128
			f.V[i] = 128
		}
		return f
	}

	ref := solid(200)
	golden := solid(56)
	average := solid(128)
	var st lossyEncodeState
	if !compoundGoldenLikely(&st, average, ref, &golden) {
		t.Fatal("compoundGoldenLikely returned false for a LAST/GOLDEN average")
	}
	if compoundGoldenLikely(&st, ref, ref, &golden) {
		t.Fatal("compoundGoldenLikely returned true when LAST already predicts the frame")
	}
}

// bruteSAD is the scalar reference for SUB_SAD_SEARCH's n x n SAD between
// arbitrary plane positions: even rows only, doubled, with the reference
// edge-replicated (SVT's padded picture).
func bruteSAD(src []byte, srcStride, sx, sy int, ref []byte, refStride, rx, ry, n, w, h int) uint32 {
	var total uint32
	for r := 0; r < n; r += 2 {
		sr := min(sy+r, h-1)
		qy := min(max(ry+r, 0), h-1)
		for c := 0; c < n; c++ {
			qx := min(max(rx+c, 0), w-1)
			d := int(src[sr*srcStride+min(sx+c, w-1)]) - int(ref[qy*refStride+qx])
			if d < 0 {
				d = -d
			}
			total += uint32(d)
		}
	}
	return total << 1
}

// sweepPUGeometry returns the SB-relative (x, y, size) of tier-zero PU index
// pu, mirroring the realtimeSBMultiSizeME layout.
func sweepPUGeometry(pu int) (x, y, n int) {
	switch {
	case pu == 0:
		return 0, 0, 64
	case pu < 5:
		q := pu - 1
		return (q & 1) * 32, (q >> 1) * 32, 32
	case pu < 21:
		i := pu - 5
		q, w := i>>2, i&3
		return (q&1)*32 + (w&1)*16, (q>>1)*32 + (w>>1)*16, 16
	default:
		i := pu - 21
		n16, c := i>>2, i&3
		q, w := n16>>2, n16&3
		return (q&1)*32 + (w&1)*16 + (c&1)*8, (q>>1)*32 + (w>>1)*16 + (c>>1)*8, 8
	}
}

// TestRealtimeSBMultiSizeMESweepMatchesBruteForce checks the hierarchical
// per-PU argmins of the sweep against an independent brute-force search over
// the same window: every PU must report the minimum full SAD over the swept
// positions with the earliest position (scan order) winning ties.
func TestRealtimeSBMultiSizeMESweepMatchesBruteForce(t *testing.T) {
	// 192x160 has only full superblocks; 200x150 adds partial ones at the
	// right and bottom edges, which read the padded pictures.
	checkSweepBruteForce(t, 192, 160, [][2]int{{0, 0}, {64, 64}, {128, 96}, {64, 0}})
	checkSweepBruteForce(t, 200, 150, [][2]int{{0, 0}, {192, 0}, {64, 128}, {192, 128}, {128, 64}})
}

func checkSweepBruteForce(t *testing.T, w, h int, sbs [][2]int) {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	mk := func() SourceFrame420 {
		y := make([]byte, w*h)
		for i := range y {
			y[i] = byte(rng.Intn(256))
		}
		return SourceFrame420{Y: y, YStride: w, Width: w, Height: h}
	}
	src, ref := mk(), mk()

	hs := &hmeState{}
	padLuma(&hs.refY[0].pl, ref)
	hs.refY[0].valid = true
	hs.selectRef(0)
	hs.setCur(src)

	for _, sb := range sbs {
		px, py := sb[0], sb[1]
		var me realtimeSBMultiSizeME
		hs.motionEstimateSB(&me, w, h, px, py)
		if !me.valid {
			t.Fatalf("sweep at (%d,%d) not valid", px, py)
		}

		// The swept 8x3 preset-12 rtc window around a (0, 0) center (no
		// pyramid, so no HME), edge-replicated where it leaves the frame.
		nx, ny := 8, 3
		x0 := -(nx >> 1)
		y0 := -(ny >> 1)

		for pu := 0; pu < 85; pu++ {
			bx, by, n := sweepPUGeometry(pu)
			bestSAD := ^uint32(0)
			bestMV := uint32(0)
			for yi := 0; yi < ny; yi++ {
				for xi := 0; xi < nx; xi++ {
					mvX, mvY := x0+xi, y0+yi
					s := bruteSAD(src.Y, w, px+bx, py+by, ref.Y, w, px+bx+mvX, py+by+mvY, n, w, h)
					if s < bestSAD {
						bestSAD = s
						bestMV = svtPackMV(mvX, mvY)
					}
				}
			}
			if me.bestSAD[pu] != bestSAD || me.bestMV[pu] != bestMV {
				t.Fatalf("SB (%d,%d) pu %d (%dx%d at +%d+%d): sweep sad=%d mv=%v, brute sad=%d mv=%v",
					px, py, pu, n, n, bx, by, me.bestSAD[pu], me.bestMV[pu], bestSAD, bestMV)
			}
		}

		if px+64 > w || py+64 > h {
			continue // partial: the sums are area-normalized
		}
		// compute_distortion sums.
		var d32, d16, d8 uint32
		for i := 1; i < 5; i++ {
			d32 += me.bestSAD[i]
		}
		for i := 5; i < 21; i++ {
			d16 += me.bestSAD[i]
		}
		for i := 21; i < 85; i++ {
			d8 += me.bestSAD[i]
		}
		if me.dist64 != me.bestSAD[0] || me.dist32 != d32 || me.dist16 != d16 || me.dist8 != d8 {
			t.Fatalf("SB (%d,%d) distortion sums mismatch", px, py)
		}
		if me.dist16 < me.dist8 {
			t.Fatalf("SB (%d,%d): dist16 %d < dist8 %d (independent argmins cannot make the parent cheaper)", px, py, me.dist16, me.dist8)
		}
	}
}

// TestRealtimeDepthRemovalBelow16Arms exercises the ported threshold algebra
// on synthetic distortion profiles.
func TestRealtimeDepthRemovalBelow16Arms(t *testing.T) {
	me := realtimeSBMultiSizeME{valid: true}
	// Near-zero 16x16 cost: the absolute cost arm must fire at level 9. Keep
	// dist32 large so the Phase-2 arms stay out of the picture.
	me.dist32 = 1 << 24
	me.dist16, me.dist8 = 100, 90
	me.me8x8CostVariance = 1 << 30 // noise guard zeroes the dev threshold
	dec := realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if !dec.cost16Arm || !dec.below16 {
		t.Fatalf("cost arm should fire on tiny dist16: %+v", dec)
	}
	if dec.dev16Arm {
		t.Fatalf("dev arm must be silenced by the high-variance noise guard: %+v", dec)
	}

	// Large distortion with tight 16->8 deviation: only the dev arm fires.
	me.dist16, me.dist8 = 900000, 880000 // dev16 = 22
	me.me8x8CostVariance = 0             // LOW guard: dev th <<= 2
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if dec.cost16Arm {
		t.Fatalf("cost arm must not fire on large dist16: %+v", dec)
	}
	if !dec.dev16Arm || !dec.below16 {
		t.Fatalf("dev arm should fire on tight deviation: %+v", dec)
	}
	if dec.dev16 != 22 {
		t.Fatalf("dev16 = %d, want 22", dec.dev16)
	}

	// Wide deviation: nothing fires.
	me.dist16, me.dist8 = 900000, 200000
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if dec.below16 {
		t.Fatalf("no arm should fire on wide deviation: %+v", dec)
	}

	// Level 0 disables the gate.
	me.dist16, me.dist8 = 100, 90
	dec = realtimeDepthRemovalDecide(0, testDepthRemovalInputs(120), &me)
	if dec.below16 || dec.below32 || dec.below64 {
		t.Fatalf("level 0 must disable the gate: %+v", dec)
	}
}

// TestRealtimeDepthRemovalBelow32And64Arms exercises the Phase-2 threshold
// algebra (enc_mode_config.c:3196-3232) on synthetic distortion profiles.
// At level 9 / qIndex 120: fastLambda = 2664, costTh32 = costTh64 =
// RDCOST(2664, 8192, 512*8) = 566912, so the cost arms fire below
// dist ~4429; picture_qp = 30 gives qpFactor = 3, so dev32To16Th = 50*3 =
// 150 (LOW noise band) and dev32To8Th = 150*5/4 = 187.
func TestRealtimeDepthRemovalBelow32And64Arms(t *testing.T) {
	me := realtimeSBMultiSizeME{valid: true}

	// Tiny 32x32 cost: the 32 cost arm fires; wide 32->16 deviation keeps the
	// dev arm silent (dist16 tiny makes dev32To16 huge).
	me.dist64 = 1 << 24
	me.dist32, me.dist16, me.dist8 = 4000, 100, 90
	me.me8x8CostVariance = 1 << 30 // HIGH noise band zeroes the dev thresholds
	dec := realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if !dec.cost32Arm || !dec.below32 {
		t.Fatalf("32 cost arm should fire on tiny dist32: %+v", dec)
	}
	if dec.dev32Arm {
		t.Fatalf("32 dev arm must be silenced by the high-variance noise guard: %+v", dec)
	}
	if dec.below64 {
		t.Fatalf("64 arm must not fire on large dist64: %+v", dec)
	}

	// Large distortions with tight 32->16 and 32->8 deviations: only the dual
	// dev arm fires. dev32To16 = 22 < 150, dev32To8 = 46 < 187.
	me.dist32, me.dist16, me.dist8 = 900000, 880000, 860000
	me.me8x8CostVariance = 0 // LOW band leaves dev32To16Th untouched
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if dec.cost32Arm {
		t.Fatalf("32 cost arm must not fire on large dist32: %+v", dec)
	}
	if !dec.dev32Arm || !dec.below32 {
		t.Fatalf("32 dev arm should fire on tight dual deviation: %+v", dec)
	}
	if dec.dev32To16 != 22 || dec.dev32To8 != 46 {
		t.Fatalf("dev32To16 = %d dev32To8 = %d, want 22/46", dec.dev32To16, dec.dev32To8)
	}

	// The middle noise band halves dev32To16Th (75, 32->8 th 93): still fires
	// at 22/46. picture_qp 30 divides the raw variance by 23.
	me.me8x8CostVariance = 30000 * 23 // 30000 after the qp division: middle band
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if !dec.dev32Arm {
		t.Fatalf("32 dev arm should survive the middle noise band: %+v", dec)
	}

	// The dual condition: tight 32->16 but wide 32->8 must NOT fire.
	me.dist32, me.dist16, me.dist8 = 900000, 880000, 200000
	me.me8x8CostVariance = 0
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if dec.dev32Arm || dec.below32 {
		t.Fatalf("32 dev arm requires BOTH deviations tight: %+v", dec)
	}

	// Tiny 64x64 cost: the (cost-only) 64 arm fires.
	me.dist64 = 4000
	dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me)
	if !dec.below64 {
		t.Fatalf("64 cost arm should fire on tiny dist64: %+v", dec)
	}

	// Level 14 (leaf ladder) raises the 32/64 multipliers to 16: a dist that
	// misses at level 9 (>= 4429) fires at level 14 (th 1091200 ->
	// dist < 8525).
	me = realtimeSBMultiSizeME{valid: true}
	me.dist64 = 1 << 24
	me.dist32, me.dist16, me.dist8 = 6000, 100, 90
	me.me8x8CostVariance = 1 << 30
	if dec = realtimeDepthRemovalDecide(9, testDepthRemovalInputs(120), &me); dec.cost32Arm {
		t.Fatalf("32 cost arm must miss at level 9 with dist32=6000: %+v", dec)
	}
	if dec = realtimeDepthRemovalDecide(14, testDepthRemovalInputs(120), &me); !dec.cost32Arm {
		t.Fatalf("32 cost arm should fire at level 14 with dist32=6000: %+v", dec)
	}
}

// testDepthRemovalInputs is a frame-uniform SB at qindex q with no reference
// feedback.
func testDepthRemovalInputs(q uint8) svtDepthRemovalInputs {
	return svtDepthRemovalInputs{pictureQP: svtPictureQP(q), fastLambda: int64(av1LambdaModeDecision8BitSAD[q]), refMinSq: 0xFF}
}

func TestSVTDepthRemovalLevelForSB(t *testing.T) {
	qp := svtPictureQP(120) // 30 -> quantizer_to_qindex 120
	cases := []struct {
		sbQ  uint8
		want uint8
	}{{120, 5}, {119, 4}, {117, 3}, {114, 2}, {108, 1}, {100, 1}, {130, 5}}
	for _, c := range cases {
		if got := svtDepthRemovalLevelForSB(5, true, c.sbQ, qp); got != c.want {
			t.Errorf("sbQ %d: level %d, want %d", c.sbQ, got, c.want)
		}
	}
	if got := svtDepthRemovalLevelForSB(5, false, 100, qp); got != 5 {
		t.Errorf("no delta q: level %d, want 5", got)
	}
}

func TestSVTPictureQP(t *testing.T) {
	for _, c := range []struct {
		q    uint8
		want int64
	}{{0, 4}, {120, 30}, {121, 30}, {122, 31}, {255, 63}} {
		if got := svtPictureQP(c.q); got != c.want {
			t.Errorf("svtPictureQP(%d) = %d, want %d", c.q, got, c.want)
		}
	}
}

// TestSVTCorrectSearchArea pins integer_search_b64's search-area correction
// (1280x720, 8x3 area).
func TestSVTCorrectSearchArea(t *testing.T) {
	for _, c := range []struct{ px, py, x0, y0, nx, ny, wx, wy, wnx, wny int }{
		{64, 64, -4, -1, 8, 3, -4, -1, 8, 3},   // inside
		{0, 0, -80, -70, 8, 3, -63, -63, 8, 3}, // left/top: origin moves, size kept
		{1216, 0, 60, 0, 8, 3, 60, 0, 4, 3},    // right: 1276+8 > 1280 -> 4, below 8 kept
		{1216, 0, 70, 0, 8, 3, 63, 0, 1, 3},    // origin past the edge moves to width-1
		{1216, 0, 50, 0, 16, 3, 50, 0, 8, 3},   // 14 rounds down to 8
		{0, 704, 0, 14, 8, 3, 0, 14, 8, 2},     // bottom: 718+3 > 720 -> 2
		{0, 704, 0, 30, 8, 3, 0, 15, 8, 1},     // origin past the bottom
		{0, 0, -4, -1, 1, 1, -4, -1, 1, 1},     // early-exit point
	} {
		x0, y0, nx, ny := svtCorrectSearchArea(c.px, c.py, c.x0, c.y0, c.nx, c.ny, 1280, 720)
		if x0 != c.wx || y0 != c.wy || nx != c.wnx || ny != c.wny {
			t.Errorf("%+v: got %d %d %d %d", c, x0, y0, nx, ny)
		}
	}
}

// sceneCutToSmoothFrames is dense noise followed by a slowly moving smooth
// gradient: the cut makes the light-PD1 DC_PRED arm code whole 16x16/32x32
// blocks as intra inside the inter frames.
func sceneCutToSmoothFrames(n, w, h int) []SourceFrame420 {
	rng := rand.New(rand.NewSource(5))
	cw, ch := w/2, h/2
	frames := make([]SourceFrame420, n)
	for k := range frames {
		f := SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		for y := range h {
			for x := range w {
				if k == 0 {
					f.Y[y*w+x] = uint8(40 + rng.Intn(170))
				} else {
					f.Y[y*w+x] = uint8(60 + (x+y+2*k)/4)
				}
			}
		}
		frames[k] = f
	}
	return frames
}

func TestPFrameIntraDCSplitMatchesFused(t *testing.T) {
	const w, h = 256, 192
	frames := sceneCutToSmoothFrames(6, w, h)
	// The oracle below proves byte identity; this fused run proves the DC
	// arm actually fires in the oracle's configuration.
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 600_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	enc.SetDecisionStatsEnabled(true)
	for i, f := range frames {
		if _, _, err := enc.Encode(f, i == 0); err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
	}
	stats := enc.DecisionStats()
	if stats.IntraBlockSizes[tile.BlockSize16x16]+stats.IntraBlockSizes[tile.BlockSize32x32] == 0 {
		t.Fatal("oracle configuration coded no 16x16/32x32 DC intra blocks")
	}
	assertSplitMatchesFused(t, frames, w, h, 30, 600_000, []int{1, 2})
	assertWavefrontMatchesSerial(t, frames, w, h, 30, 600_000, []int{2, 3})
}

func TestMDS0RetainedPredictionMatchesWinner(t *testing.T) {
	zero := motion.Vector{}
	onePixel := motion.Vector{Col: 8}
	farMV := motion.Vector{Col: 64}
	tests := []struct {
		name     string
		sourceMV motion.Vector
		meMV     motion.Vector
		nearest  motion.Vector
		near     motion.Vector
		dcQ      int32 // luma DC quantizer: sets full_lambda (lpd1FullLambda)
		rates    tile.InterModeRateTables
		wantMode tile.InterMode
		wantMV   motion.Vector
	}{
		{
			name:     "restore-overwritten-me-winner",
			sourceMV: farMV, meMV: farMV, nearest: zero, near: onePixel,
			dcQ: 12, wantMode: tile.InterModeNewMV, wantMV: farMV,
		},
		{
			name:     "copy-new-near-winner",
			sourceMV: onePixel, meMV: farMV, nearest: zero, near: onePixel,
			dcQ: 12, wantMode: tile.InterModeNearMV, wantMV: onePixel,
		},
		{
			// No GLOBALMV candidate in light-PD1: the zero vector wins as
			// NEARESTMV once rate dominates.
			name:     "copy-nearest-rate-winner",
			sourceMV: onePixel, meMV: farMV, nearest: zero, near: onePixel,
			dcQ: 30000,
			rates: tile.InterModeRateTables{
				NewMV: [tile.NewMVModeContexts][2]uint32{{50_000, 0}},
				RefMV: [tile.RefMVModeContexts][2]uint32{{10_000, 20_000}},
			},
			wantMode: tile.InterModeNearestMV, wantMV: zero,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const width, height, block, px, py = 64, 64, 8, 24, 24
			rng := rand.New(rand.NewSource(91))
			refPlane := make([]byte, width*height)
			for i := range refPlane {
				refPlane[i] = byte(rng.Intn(256))
			}
			var sourcePrediction [block * block]byte
			var sourceScratch motion.ScaledConvolveScratch
			if err := predictIntoScaled(sourcePrediction[:], refPlane, width, width, height, width, height,
				px, py, block, block, tc.sourceMV, false, false, &sourceScratch); err != nil {
				t.Fatalf("source prediction: %v", err)
			}
			srcY := make([]byte, width*height)
			for row := range block {
				copy(srcY[(py+row)*width+px:(py+row)*width+px+block], sourcePrediction[row*block:(row+1)*block])
			}
			src := SourceFrame420{Y: srcY, YStride: width, Width: width, Height: height}
			stack := &tile.ReferenceMVStackResult{ModeContext: 0}
			stack.Stack.Count = 2
			stack.Stack.Candidates[0].This = tc.nearest
			stack.Stack.Candidates[1].This = tc.near
			st := &lossyEncodeState{mds0Rates: tc.rates}
			st.yQuant.DC = tc.dcQ

			winner, ok, retained := st.mds0PickInterMode(src, refPlane, width, stack, px, py, block, block, tc.meMV)
			if !ok {
				t.Fatal("MDS0 did not return a winner")
			}
			if winner.mode != tc.wantMode || winner.mv != tc.wantMV {
				t.Fatalf("winner=(%v,%v), want (%v,%v)", winner.mode, winner.mv, tc.wantMode, tc.wantMV)
			}
			if !retained {
				t.Fatal("winner prediction was not retained")
			}

			var want [block * block]byte
			var wantScratch motion.ScaledConvolveScratch
			if err := predictIntoScaled(want[:], refPlane, width, width, height, width, height,
				px, py, block, block, winner.mv, false, false, &wantScratch); err != nil {
				t.Fatalf("winner prediction: %v", err)
			}
			if !bytes.Equal(st.predY[:block*block], want[:]) {
				t.Fatal("retained predY does not match the selected winner")
			}
		})
	}
}

func TestMDS0PredictorMatchesCodedFilterGate(t *testing.T) {
	fullPel := motion.Vector{Row: 8, Col: 16}
	subPel := motion.Vector{Row: 1, Col: 9}
	regular := motion.RegularFilters
	smooth := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpEightTapRegular}
	dual := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	tests := []struct {
		name     string
		retained bool
		mv       motion.Vector
		filters  motion.InterpFilters
		want     bool
	}{
		{"invalid", false, subPel, regular, false},
		{"full-pel ignores filter", true, fullPel, smooth, true},
		{"sub-pel regular", true, subPel, regular, true},
		{"sub-pel smooth", true, subPel, smooth, false},
		{"sub-pel dual", true, subPel, dual, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := mds0PredictorMatchesCodedFilters(tc.retained, tc.mv, tc.filters); got != tc.want {
				t.Fatalf("reusable=%t, want %t", got, tc.want)
			}
		})
	}
}

// TestPFrameReuseOptimizationsPreserveEncodedOutput pins the AV1 bytes and
// reconstructed planes produced by the pre-optimization encoder. Its serial
// and multi-thread cases cover both fused and split P-frame residual paths.
// Re-pinned for the SVT md_subpel_me level-6 sub-pel search (subpel_svt.go),
// which replaced the libaom sub-pel tree, and for the light-PD1 MDS0
// candidate set without GLOBALMV (pframe_mds0.go); both change the coded
// vectors. Re-pinned again for SVT's open-loop per-SB ME (HME levels 0/1,
// sub-sampled 8x3 sweep) as the leaves' full-pel ME, and for light-PD1's
// inter path (no interpolation filter search, md_subpel_me level 8, then
// the per-SB sub-pel level with its zero-subpel override and full lambda),
// and for SVT's per-picture CDEF strength search (cdef_search.go).
func TestPFrameReuseOptimizationsPreserveEncodedOutput(t *testing.T) {
	const width, height, frameCount = 640, 360, 48
	tests := []struct {
		threads int
		stream  string
		recon   string
	}{
		{1, "e134ed44bcb0773070ef0db5b6aab00a09e1f233671bb1a99ec7b182e2557de1", "fa6ce8f537bba1af7221d1c54a4ffcbd0a6e5aea317d6197371e0b502159cd38"},
		// One tile at every thread count: the SB-row wavefront (keyframes and
		// inter frames) reproduces the serial stream.
		{4, "e134ed44bcb0773070ef0db5b6aab00a09e1f233671bb1a99ec7b182e2557de1", "fa6ce8f537bba1af7221d1c54a4ffcbd0a6e5aea317d6197371e0b502159cd38"},
	}

	rng := rand.New(rand.NewSource(3))
	background := make([]byte, width*height)
	for i := range background {
		background[i] = uint8(60 + rng.Intn(60))
	}
	makeFrame := func(i int) SourceFrame420 {
		f := SourceFrame420{
			Y:            append([]byte(nil), background...),
			U:            make([]byte, width/2*(height/2)),
			V:            make([]byte, width/2*(height/2)),
			YStride:      width,
			ChromaStride: width / 2,
			Width:        width,
			Height:       height,
		}
		for i := range f.U {
			f.U[i], f.V[i] = 120, 130
		}
		sx, sy := (i*4)%(width-32), (i*2)%(height-32)
		for y := sy; y < sy+32; y++ {
			for x := sx; x < sx+32; x++ {
				f.Y[y*width+x] = 220
			}
		}
		return f
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("threads-%d", tc.threads), func(t *testing.T) {
			e, err := NewVideoEncoder(width, height, 60)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := e.Close(); err != nil {
					t.Errorf("close encoder: %v", err)
				}
			})
			e.SetMaxThreads(tc.threads)
			if err := e.Prewarm(); err != nil {
				t.Fatal(err)
			}

			streamHash, reconHash := sha256.New(), sha256.New()
			for i := range frameCount {
				tu, _, err := e.Encode(makeFrame(i), false)
				if err != nil {
					t.Fatalf("frame %d encode: %v", i, err)
				}
				var length [8]byte
				binary.LittleEndian.PutUint64(length[:], uint64(len(tu)))
				_, _ = streamHash.Write(length[:])
				_, _ = streamHash.Write(tu)
				recon := e.Recon()
				_, _ = reconHash.Write(recon.Y)
				_, _ = reconHash.Write(recon.U)
				_, _ = reconHash.Write(recon.V)
			}
			if got := fmt.Sprintf("%x", streamHash.Sum(nil)); got != tc.stream {
				t.Errorf("bitstream SHA-256 = %s, want baseline %s", got, tc.stream)
			}
			if got := fmt.Sprintf("%x", reconHash.Sum(nil)); got != tc.recon {
				t.Errorf("reconstruction SHA-256 = %s, want baseline %s", got, tc.recon)
			}
		})
	}
}

func TestPredictIntoScaledReferenceMatchesMotionConvolver(t *testing.T) {
	tests := []struct {
		name                         string
		refWidth, refHeight          int
		curWidth, curHeight          int
		px, py, blockWidth, blockHgt int
		mv                           motion.Vector
		subsamplingX, subsamplingY   bool
	}{
		{
			name:       "luma upsample reference",
			refWidth:   64,
			refHeight:  48,
			curWidth:   128,
			curHeight:  96,
			px:         80,
			py:         40,
			blockWidth: 16,
			blockHgt:   16,
			mv:         motion.Vector{Col: 5, Row: -3},
		},
		{
			name:         "chroma upsample reference",
			refWidth:     32,
			refHeight:    24,
			curWidth:     64,
			curHeight:    48,
			px:           30,
			py:           18,
			blockWidth:   8,
			blockHgt:     8,
			mv:           motion.Vector{Col: 5, Row: -3},
			subsamplingX: true,
			subsamplingY: true,
		},
		{
			name:       "luma downscale right edge",
			refWidth:   128,
			refHeight:  96,
			curWidth:   64,
			curHeight:  48,
			px:         56,
			py:         32,
			blockWidth: 8,
			blockHgt:   16,
			mv:         motion.Vector{Col: -4, Row: 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ref := scaledPredictionPattern(tt.refWidth, tt.refHeight)
			got := make([]byte, tt.blockWidth*tt.blockHgt)
			var scratch motion.ScaledConvolveScratch
			if err := predictIntoScaled(got, ref, tt.refWidth, tt.refWidth, tt.refHeight, tt.curWidth, tt.curHeight,
				tt.px, tt.py, tt.blockWidth, tt.blockHgt, tt.mv, tt.subsamplingX, tt.subsamplingY, &scratch); err != nil {
				t.Fatalf("predictIntoScaled: %v", err)
			}

			want := make([]byte, tt.blockWidth*tt.blockHgt)
			if err := directScaledPredict(want, ref, tt.refWidth, tt.refHeight, tt.curWidth, tt.curHeight,
				tt.px, tt.py, tt.blockWidth, tt.blockHgt, tt.mv, tt.subsamplingX, tt.subsamplingY); err != nil {
				t.Fatalf("direct scaled predict: %v", err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("prediction mismatch\n got %v\nwant %v", got, want)
			}
		})
	}
}

func TestPredictIntoScaledReferenceIdentityMatchesSameSizePredictor(t *testing.T) {
	const (
		width      = 64
		height     = 48
		px         = 24
		py         = 16
		blockWidth = 16
		blockHgt   = 8
	)
	ref := scaledPredictionPattern(width, height)
	mv := motion.Vector{Col: 5, Row: -3}

	got := make([]byte, blockWidth*blockHgt)
	var scratch motion.ScaledConvolveScratch
	if err := predictIntoScaled(got, ref, width, width, height, width, height, px, py, blockWidth, blockHgt, mv, false, false, &scratch); err != nil {
		t.Fatalf("predictIntoScaled: %v", err)
	}

	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, false, false)
	if err != nil {
		t.Fatalf("ReferenceOriginSubsampled: %v", err)
	}
	want := make([]byte, blockWidth*blockHgt)
	dstPlane := frame.Plane{Pix: want, Stride: blockWidth, Width: blockWidth, Height: blockHgt}
	refPlane := frame.Plane{Pix: ref, Stride: width, Width: width, Height: height}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstPlane, refPlane, 1, 8, 0, 0,
		refX, refY, blockWidth, blockHgt, subX, subY, motion.RegularFilters); err != nil {
		t.Fatalf("same-size predictor: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("identity prediction mismatch\n got %v\nwant %v", got, want)
	}
}

func TestPredictIntoScaledReferenceRejectsInvalidScale(t *testing.T) {
	ref := scaledPredictionPattern(4, 4)
	err := predictIntoScaled(make([]byte, 64), ref, 4, 4, 4, 128, 128, 0, 0, 8, 8, motion.Vector{}, false, false, nil)
	if !errors.Is(err, motion.ErrInvalidMotion) {
		t.Fatalf("err=%v, want ErrInvalidMotion", err)
	}
}

func scaledPredictionPattern(width, height int) []byte {
	pix := make([]byte, width*height)
	for y := range height {
		for x := range width {
			pix[y*width+x] = byte(17*x + 29*y + 3*x*y + 19)
		}
	}
	return pix
}

func directScaledPredict(dst, ref []byte, refWidth, refHeight, curWidth, curHeight, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool) error {
	sf, err := motion.NewScaleFactors(refWidth, refHeight, curWidth, curHeight)
	if err != nil {
		return err
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(px, py, mv, ssX, ssY)
	if err != nil {
		return err
	}
	filters := motion.RegularFilters
	xTable, err := motion.SubpelKernelTableFor(filters.X, bw)
	if err != nil {
		return err
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, bh)
	if err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	refPlane := frame.Plane{Pix: ref, Stride: refWidth, Width: refWidth, Height: refHeight}
	var scratch motion.ScaledConvolveScratch
	return motion.ConvolveScale2D8ClampedWithScratch(dstPlane, refPlane, 0, 0, bw, bh, startX, xStep, startY, yStep, xTable, yTable, &scratch)
}

// The split (decision pass + serial entropy pass) P-frame pipeline must be
// byte-identical to the fused single-pass coder at every tile count: the
// write pass replays the recorded decisions through the same adaptive-CDF
// writers in the same order, so any divergence is a bug in the split. These
// tests are the correctness oracle for PERF_PLAN §6 E1 stage (a).

func encodeSplitOracleSequence(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate, tileCols int, fused bool) [][]byte {
	return encodeSplitOracleSequenceLanes(t, frames, w, h, fps, bitrate, tileCols, fused, 0)
}

func encodeSplitOracleSequenceLanes(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate, tileCols int, fused bool, lanes int) [][]byte {
	t.Helper()
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: bitrate, FramesPerSecond: fps, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.forceSplit = !fused
	enc.wavefrontLanes = lanes
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	if tileCols > 0 {
		enc.SetTileColumns(tileCols)
	}
	out := make([][]byte, 0, len(frames))
	for i, f := range frames {
		data, _, err := enc.Encode(f, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		out = append(out, append([]byte(nil), data...))
	}
	return out
}

func assertSplitMatchesFused(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate int, tileCounts []int) {
	t.Helper()
	for _, tiles := range tileCounts {
		t.Run(fmt.Sprintf("tiles=%d", tiles), func(t *testing.T) {
			fused := encodeSplitOracleSequence(t, frames, w, h, fps, bitrate, tiles, true)
			split := encodeSplitOracleSequence(t, frames, w, h, fps, bitrate, tiles, false)
			if len(fused) != len(split) {
				t.Fatalf("frame count mismatch: fused=%d split=%d", len(fused), len(split))
			}
			for i := range fused {
				if !bytes.Equal(fused[i], split[i]) {
					t.Fatalf("frame %d: split bitstream differs from fused (fused=%d bytes, split=%d bytes)", i, len(fused[i]), len(split[i]))
				}
			}
		})
	}
}

// assertWavefrontMatchesSerial encodes single-tile with the serial decision
// walk and with 2..n wavefront lanes; every lane count must produce
// byte-identical bitstreams (the stage-b oracle: the SB-row wavefront reads
// exactly the serial order's neighbor state).
func assertWavefrontMatchesSerial(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate int, laneCounts []int) {
	t.Helper()
	serial := encodeSplitOracleSequenceLanes(t, frames, w, h, fps, bitrate, 1, false, 0)
	for _, lanes := range laneCounts {
		t.Run(fmt.Sprintf("lanes=%d", lanes), func(t *testing.T) {
			wave := encodeSplitOracleSequenceLanes(t, frames, w, h, fps, bitrate, 1, false, lanes)
			if len(serial) != len(wave) {
				t.Fatalf("frame count mismatch: serial=%d wavefront=%d", len(serial), len(wave))
			}
			for i := range serial {
				if !bytes.Equal(serial[i], wave[i]) {
					t.Fatalf("frame %d: wavefront bitstream differs from serial (serial=%d bytes, wavefront=%d bytes)", i, len(serial[i]), len(wave[i]))
				}
			}
		})
	}
}

func TestPFrameWavefrontMatchesSerialSynthetic(t *testing.T) {
	const w, h = 640, 384
	frames := syntheticMotionFrames(12, w, h)
	assertWavefrontMatchesSerial(t, frames, w, h, 30, 1_200_000, []int{2, 3, 6})
}

func TestPFrameWavefrontMatchesSerialRealC(t *testing.T) {
	frames, w, h, ok := loadOracleCorpusSegment(t, "/tmp/corpus/realC.yuv", 16)
	if !ok {
		return
	}
	assertWavefrontMatchesSerial(t, frames, w, h, 30, 5_000_000, []int{2, 4, 8, 16})
}

// loadOracleCorpusSegment reads up to maxFrames 1080p I420 frames from the
// local corpus, skipping the test when the corpus is absent.
func loadOracleCorpusSegment(t *testing.T, path string, maxFrames int) ([]SourceFrame420, int, int, bool) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("local corpus unavailable: %v", err)
		return nil, 0, 0, false
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
	if nf > maxFrames {
		nf = maxFrames
	}
	if nf < 3 {
		t.Skipf("%s too short", path)
		return nil, 0, 0, false
	}
	frames := make([]SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	return frames, w, h, true
}

// syntheticMotionFrames builds a deterministic moving-texture clip that
// exercises skip blocks, coded residuals, golden probes, and the 8x8 intra
// fallback (via the injected scene cut at the midpoint).
func syntheticMotionFrames(n, w, h int) []SourceFrame420 {
	frames := make([]SourceFrame420, n)
	for i := range frames {
		y := make([]byte, w*h)
		u := make([]byte, w*h/4)
		v := make([]byte, w*h/4)
		phase := i * 3
		flip := 0
		if i >= n/2 {
			flip = 97 // scene change: shifts every value, forcing intra rescue
		}
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				val := (c+phase)&255 ^ (r&63)*2
				if (c/48+r/32)%2 == 0 {
					val += ((c * r) >> 6) & 31
				}
				y[r*w+c] = byte(val + flip)
			}
		}
		for r := 0; r < h/2; r++ {
			for c := 0; c < w/2; c++ {
				u[r*w/2+c] = byte(128 + ((c - phase) & 31) + flip/2)
				v[r*w/2+c] = byte(128 - (r & 31) - flip/3)
			}
		}
		frames[i] = SourceFrame420{
			Y: y, U: u, V: v,
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	return frames
}

func TestPFrameSplitMatchesFusedSynthetic(t *testing.T) {
	const w, h = 640, 384
	frames := syntheticMotionFrames(12, w, h)
	assertSplitMatchesFused(t, frames, w, h, 30, 1_200_000, []int{1, 2, 4})
}

// TestPFrameSplitMatchesFusedRealC runs the oracle on a real 1080p camera
// segment when the local corpus is present (complex motion: compound probes,
// realtime TX splits, interpolation filter search all engage).
func TestPFrameSplitMatchesFusedRealC(t *testing.T) {
	const path = "/tmp/corpus/realC.yuv"
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("local corpus unavailable: %v", err)
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
	if nf > 16 {
		nf = 16
	}
	if nf < 3 {
		t.Skipf("%s too short", path)
	}
	frames := make([]SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	assertSplitMatchesFused(t, frames, w, h, 30, 5_000_000, []int{1, 4, 32})
}

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

func TestChooseInter8x8TXTypeCanSelectHybrid(t *testing.T) {
	st := newTXTypeTestState(t, 96)
	src := SourceFrame420{
		Y:            make([]byte, 64),
		U:            make([]byte, 16),
		V:            make([]byte, 16),
		YStride:      8,
		ChromaStride: 4,
		Width:        8,
		Height:       8,
	}
	for i := range st.predY[:64] {
		st.predY[i] = 128
	}
	for i := range st.predU[:16] {
		st.predU[i] = 128
		st.predV[i] = 128
		src.U[i] = 128
		src.V[i] = 128
	}

	rng := rand.New(rand.NewSource(37))
	for attempt := range 2048 {
		for i := range src.Y {
			src.Y[i] = uint8(rng.Intn(256))
		}
		dctDcode := prepareTXTypeDCTCandidate(st, src)
		typ := st.chooseInter8x8TXType(src, 0, 0, dctDcode)
		if typ != transform.TypeDCTDCT {
			t.Logf("attempt %d selected %v", attempt, typ)
			return
		}
	}
	t.Fatal("selector never chose a hybrid tx_type")
}

func TestChooseInter8x8TXTypeSeededMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for attempt := range 128 {
		seeded := newTXTypeTestState(t, 96)
		reference := newTXTypeTestState(t, 96)
		src := SourceFrame420{
			Y:            make([]byte, 64),
			U:            make([]byte, 16),
			V:            make([]byte, 16),
			YStride:      8,
			ChromaStride: 4,
			Width:        8,
			Height:       8,
		}
		for i := range src.Y {
			src.Y[i] = uint8(rng.Intn(256))
			seeded.predY[i] = uint8(rng.Intn(256))
			reference.predY[i] = seeded.predY[i]
		}
		for i := range src.U {
			src.U[i] = uint8(rng.Intn(256))
			src.V[i] = uint8(rng.Intn(256))
			seeded.predU[i] = uint8(rng.Intn(256))
			seeded.predV[i] = uint8(rng.Intn(256))
			reference.predU[i] = seeded.predU[i]
			reference.predV[i] = seeded.predV[i]
		}

		dctDcode := prepareTXTypeDCTCandidate(seeded, src)
		got := seeded.chooseInter8x8TXType(src, 0, 0, dctDcode)
		want := chooseInter8x8TXTypeReference(reference, src, 0, 0)
		if got != want ||
			!slices.Equal(seeded.lumaQ[:64], reference.lumaQ[:64]) ||
			!slices.Equal(seeded.uQ[:16], reference.uQ[:16]) ||
			!slices.Equal(seeded.vQ[:16], reference.vQ[:16]) {
			t.Fatalf("attempt %d: seeded type=%v reference type=%v qcoeffs equal Y=%t U=%t V=%t",
				attempt, got, want,
				slices.Equal(seeded.lumaQ[:64], reference.lumaQ[:64]),
				slices.Equal(seeded.uQ[:16], reference.uQ[:16]),
				slices.Equal(seeded.vQ[:16], reference.vQ[:16]))
		}
	}
}

func TestTrialTXBBitsInterMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	cases := []struct {
		n     int
		size  tile.TransformSize
		types []transform.Type
	}{
		{
			n:    8,
			size: tile.TransformSize8x8,
			types: []transform.Type{
				transform.TypeDCTDCT,
				transform.TypeADSTDCT,
				transform.TypeDCTADST,
				transform.TypeADSTADST,
				transform.TypeIDTX,
			},
		},
	}
	for _, tc := range cases {
		for attempt := range 128 {
			qcoeff := make([]int16, tc.n*tc.n)
			qcoeff[0] = 1
			for i := range qcoeff {
				if i == 0 || rng.Intn(4) != 0 {
					continue
				}
				v := int16(1 + rng.Intn(7))
				if rng.Intn(2) == 0 {
					v = -v
				}
				qcoeff[i] = v
			}
			for _, typ := range tc.types {
				fast := newTXTypeTestState(t, 96)
				reference := newTXTypeTestState(t, 96)
				txBefore := fast.txCDFs
				got := fast.trialTXBBitsInter8x8((*[64]int16)(qcoeff), typ)
				want := trialTXBBitsInterReference(reference, qcoeff, tc.n, tc.size, typ)
				if got != want {
					t.Fatalf("%dx%d attempt %d typ %v: trial bits=%d want %d", tc.n, tc.n, attempt, typ, got, want)
				}
				if fast.txCDFs != txBefore {
					t.Fatalf("%dx%d attempt %d typ %v: trial mutated transform cdfs", tc.n, tc.n, attempt, typ)
				}
			}
		}
	}
}

func BenchmarkChooseInter8x8TXType(b *testing.B) {
	st := newTXTypeTestState(b, 96)
	src := SourceFrame420{
		Y:            make([]byte, 64),
		U:            make([]byte, 16),
		V:            make([]byte, 16),
		YStride:      8,
		ChromaStride: 4,
		Width:        8,
		Height:       8,
	}
	rng := rand.New(rand.NewSource(109))
	for i := range src.Y {
		src.Y[i] = uint8(rng.Intn(256))
		st.predY[i] = uint8(rng.Intn(256))
	}
	for i := range src.U {
		src.U[i] = uint8(rng.Intn(256))
		src.V[i] = uint8(rng.Intn(256))
		st.predU[i] = uint8(rng.Intn(256))
		st.predV[i] = uint8(rng.Intn(256))
	}
	b.ReportAllocs()
	b.ResetTimer()
	var typ transform.Type
	for range b.N {
		dctDcode := prepareTXTypeDCTCandidate(st, src)
		typ = st.chooseInter8x8TXType(src, 0, 0, dctDcode)
	}
	if typ == transform.TypeCount {
		b.Fatal(typ)
	}
}

func prepareTXTypeDCTCandidate(st *lossyEncodeState, src SourceFrame420) int64 {
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, st.predY[:64], 8, src.YStride, 0, 0, 8, 8, st.yQuant, st.lumaQ[:64])
	st.prepareInterTXB(src.U, st.predU[:16], 4, src.ChromaStride, 0, 0, 4, 4, st.uQuant, st.uQ[:16])
	st.prepareInterTXB(src.V, st.predV[:16], 4, src.ChromaStride, 0, 0, 4, 4, st.vQuant, st.vQ[:16])
	return st.rdDcode
}

func chooseInter8x8TXTypeReference(st *lossyEncodeState, src SourceFrame420, lumaPX, lumaPY int) transform.Type {
	if !st.armTrial() {
		return transform.TypeDCTDCT
	}
	bestType := transform.TypeDCTDCT
	bestCost := int64(1 << 62)
	saveDcode, saveDskip, saveRcode := st.rdDcode, st.rdDskip, st.rdRcode
	baseTrialCDFs := st.trialCDFs
	tmpY := st.lumaQ2[:64]
	tmpU := st.lumaQ2[64:80]
	tmpV := st.lumaQ2[80:96]
	for _, typ := range [...]transform.Type{
		transform.TypeDCTDCT,
		transform.TypeADSTDCT,
		transform.TypeDCTADST,
		transform.TypeADSTADST,
		transform.TypeIDTX,
	} {
		st.trialCDFs = baseTrialCDFs
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := st.prepareInterTXBTyped(src.Y, st.predY[:64], 8, src.YStride, lumaPX, lumaPY, 8, 8, st.yQuant, tmpY, typ)
		if typ != transform.TypeDCTDCT && lumaZero {
			continue
		}
		st.prepareInterTXBTyped(src.U, st.predU[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.uQuant, tmpU, typ)
		st.prepareInterTXBTyped(src.V, st.predV[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.vQuant, tmpV, typ)
		cost := st.rdDcode << 7
		cost += st.trialTXBBitsInter8x8((*[64]int16)(tmpY), typ)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpU, 4)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpV, 4)
		if cost < bestCost {
			bestCost = cost
			bestType = typ
			copy(st.lumaQ[:64], tmpY)
			copy(st.uQ[:16], tmpU)
			copy(st.vQ[:16], tmpV)
		}
	}
	st.rdDcode, st.rdDskip, st.rdRcode = saveDcode, saveDskip, saveRcode
	st.trialCDFs = baseTrialCDFs
	return bestType
}

func trialTXBBitsInterReference(st *lossyEncodeState, qcoeff []int16, n int, size tile.TransformSize, typ transform.Type) int64 {
	scan := st.scan4
	switch n {
	case 8:
		scan = st.scan8
	case 16:
		scan = st.scan16
	case 32:
		scan = st.scan32
	}
	buf := make([]byte, 0, 1<<14)
	tw := entropy.NewWriter(buf)
	txCDFs := st.txCDFs
	txReq := tile.InterTransformTypeRequest{
		Size:        size,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteInterTransformType(&tw, &txCDFs, txReq, typ)
	}
	base := tw.Tell()
	if _, err := tile.WriteCoefficientsTXB(&tw, &st.trialCDFs, tile.TXBEncodeRequest{
		Size: size, Plane: tile.CoeffPlaneY, Class: transform.Class2D, AfterSkip: afterSkip,
	}, qcoeff[:n*n], scan, st.levels); err != nil {
		return 1 << 59
	}
	bits := int64(tw.Tell() - base)
	return ((bits<<9)*st.rdMult + 256) >> 9
}

func newTXTypeTestState(t interface {
	Helper()
	Fatal(args ...any)
}, qIndex uint8) *lossyEncodeState {
	t.Helper()
	var st lossyEncodeState
	st.qIndex = qIndex
	st.color = parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			t.Fatal(err)
		}
		*dst = q
	}
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	if err := st.txCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := st.trialCDFs.InitDefault(qIndex); err != nil {
		t.Fatal(err)
	}
	st.trialReady = true
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	return &st
}

func TestSVTDimensionsRequire8x8(t *testing.T) {
	cases := []struct {
		w, h int
		want bool
	}{
		{64, 64, false},
		{64, 16, false}, // 720p bottom row: 16x16 covers it
		{64, 56, false}, // 1080p bottom row: the 32 half fits
		{64, 8, true},
		{8, 64, true},
		{40, 64, false},
		{24, 64, false},
	}
	for _, c := range cases {
		if got := svtDimensionsRequire8x8(c.w, c.h); got != c.want {
			t.Errorf("svtDimensionsRequire8x8(%d, %d) = %v, want %v", c.w, c.h, got, c.want)
		}
	}
}

func TestSVTInputResolution(t *testing.T) {
	cases := []struct {
		w, h, want int
	}{
		{320, 240, 0},
		{640, 360, 1}, // 0x38400: >= the 240p and < the 360p threshold
		{1280, 720, 3},
		{1920, 1080, 4},
		{3840, 2160, 5},
	}
	for _, c := range cases {
		if got := svtInputResolution(c.w, c.h); got != c.want {
			t.Errorf("svtInputResolution(%d, %d) = %d, want %d", c.w, c.h, got, c.want)
		}
	}
}

// TestSVTPD0PicksLargeBlocksOnStaticContent checks the VLPD0 tree end to end:
// a source identical to its reference has zero variance everywhere, so every
// split only adds partition rate and the 64x64 block stays whole.
func TestSVTPD0PicksLargeBlocksOnStaticContent(t *testing.T) {
	const w, h = 128, 128
	mk := func(fill func(i int) byte) SourceFrame420 {
		f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h}
		for i := range f.Y {
			f.Y[i] = fill(i)
		}
		return f
	}
	flat := mk(func(int) byte { return 100 })
	var pc pframeCoder
	if err := pc.reset(120, 2, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	st.prepareSVTPD0(2, 2)
	st.pd0Rates.estimate(&pc.partCDFs)
	st.setInterLambda()
	sb, err := st.svtPD0ForSB(flat, flat, 0, 0)
	if err != nil || sb == nil {
		t.Fatalf("svtPD0ForSB: %v %v", sb, err)
	}
	if sb.part64 != tile.PartitionNone {
		t.Fatalf("static SB partition %v, want None", sb.part64)
	}
}

// TestSVTPD0EdgeShapes checks the incomplete-SB shapes: a 16-row bottom SB
// cannot code the 64x32 PART_H block (it crosses the frame edge), so the
// 64x64 node splits and each 32x32 node, missing its lower half, codes the
// 32x16 PART_H block of a static source.
func TestSVTPD0EdgeShapes(t *testing.T) {
	const w, h = 128, 80
	f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	for i := range f.Y {
		f.Y[i] = 100
	}
	var pc pframeCoder
	if err := pc.reset(120, 2, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	st.prepareSVTPD0(2, 2)
	st.pd0Rates.estimate(&pc.partCDFs)
	st.setInterLambda()
	sb, err := st.svtPD0ForSB(f, f, 0, 64)
	if err != nil || sb == nil {
		t.Fatalf("svtPD0ForSB: %v %v", sb, err)
	}
	if sb.part64 != tile.PartitionSplit || sb.part32[0] != tile.PartitionH || sb.part32[1] != tile.PartitionH {
		t.Fatalf("edge SB tree %v %v, want Split [H H ..]", sb.part64, sb.part32)
	}
}

func TestPFrameDecideWorkerAdoptKeepsLaneSubpelCache(t *testing.T) {
	ownerCache := &subpelProbeCache{}
	src := &lossyEncodeState{subpelCache: ownerCache}
	var workers [3]pframeDecideWorker
	for i := range workers {
		workers[i].st.subpelCache = &subpelProbeCache{}
		workers[i].adopt(src)
	}

	for i := range workers {
		cache := workers[i].st.subpelCache
		if cache == nil || cache == ownerCache {
			t.Fatalf("lane %d did not keep a private cache", i)
		}
		for j := 0; j < i; j++ {
			if cache == workers[j].st.subpelCache {
				t.Fatalf("lanes %d and %d share a subpel cache", j, i)
			}
		}
	}

	workers[0].st.subpelCache.scratch[0] = 1
	workers[0].adopt(src)
	if workers[0].st.subpelCache.scratch[0] != 1 {
		t.Fatal("repeated adopt discarded lane-owned subpel cache state")
	}
	if ownerCache.scratch[0] != 0 {
		t.Fatal("lane cache mutation changed the owner cache")
	}
}

func overflowingStride(width int) (height, stride int, ok bool) {
	maxInt := int(^uint(0) >> 1)
	for height = 8; height <= 1<<20; height += 8 {
		rows := height - 1
		stride = (maxInt - (width - 1)) / rows
		if stride > (maxInt-width)/rows {
			return height, stride, true
		}
	}
	return 0, 0, false
}

func TestCheckedPlaneLengthBoundaries(t *testing.T) {
	if got, ok := checkedPlaneLength(8, 8, 8); !ok || got != 64 {
		t.Fatalf("minimal plane: got (%d, %v), want (64, true)", got, ok)
	}
	if got, ok := checkedPlaneLength(8, 8, 10); !ok || got != 78 {
		t.Fatalf("padded plane: got (%d, %v), want (78, true)", got, ok)
	}
	if got, ok := checkedPlaneLength(8, 1, 10); !ok || got != 8 {
		t.Fatalf("single-row plane: got (%d, %v), want (8, true)", got, ok)
	}
	maxInt := int(^uint(0) >> 1)
	if got, ok := checkedPlaneLength(8, 2, maxInt-8); !ok || got != maxInt {
		t.Fatalf("exact MaxInt plane: got (%d, %v), want (%d, true)", got, ok, maxInt)
	}

	height, stride, found := overflowingStride(8)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}
	if got, ok := checkedPlaneLength(8, height, stride); ok {
		t.Fatalf("overflowing plane accepted with length %d (height %d, stride %d)", got, height, stride)
	}
}

func TestMonochromeRenderValidatorsRejectOverflowingStride(t *testing.T) {
	const width = 8
	height, stride, found := overflowingStride(width)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}

	mono := &MonochromeVideoEncoder{renderWidth: width, renderHeight: height}
	if err := mono.validateRenderSource(SourceFrameMono{
		YStride: stride,
		Width:   width,
		Height:  height,
	}); err == nil || !strings.Contains(err.Error(), "dimensions overflow int") {
		t.Fatalf("8-bit render validator error = %v, want plane-length overflow", err)
	}

	mono16 := &HighBitDepthMonochromeVideoEncoder{
		renderWidth:  width,
		renderHeight: height,
		bitDepth:     10,
	}
	if err := mono16.validateRenderSource(SourceFrameMono16{
		YStride:  stride,
		Width:    width,
		Height:   height,
		BitDepth: 10,
	}); err == nil || !strings.Contains(err.Error(), "dimensions overflow int") {
		t.Fatalf("high-bit-depth render validator error = %v, want plane-length overflow", err)
	}
}

func TestMonochromeRenderValidatorsAcceptSingleRow(t *testing.T) {
	const width, height, stride = 8, 1, 10
	mono := &MonochromeVideoEncoder{renderWidth: width, renderHeight: height}
	if err := mono.validateRenderSource(SourceFrameMono{
		Y:       make([]byte, width),
		YStride: stride,
		Width:   width,
		Height:  height,
	}); err != nil {
		t.Fatalf("8-bit single-row render: %v", err)
	}

	mono16 := &HighBitDepthMonochromeVideoEncoder{
		renderWidth:  width,
		renderHeight: height,
		bitDepth:     10,
	}
	if err := mono16.validateRenderSource(SourceFrameMono16{
		Y:        make([]uint16, width),
		YStride:  stride,
		Width:    width,
		Height:   height,
		BitDepth: 10,
	}); err != nil {
		t.Fatalf("high-bit-depth single-row render: %v", err)
	}
}

func TestColor16ValidatorChecksChromaPlaneLength(t *testing.T) {
	const width = 8
	height, stride, found := overflowingStride(width)
	if !found {
		t.Fatal("could not find a portable overflowing stride case")
	}
	color := parser.ColorConfig{BitDepth: 10}
	src := SourceFrame42016{
		Y:            make([]uint16, (height-1)*width+width),
		YStride:      width,
		ChromaStride: stride,
		Width:        width,
		Height:       height,
		BitDepth:     10,
	}
	if err := validateSourceFrameColor16(src, color, "4:4:4 test"); err == nil || !strings.Contains(err.Error(), "chroma plane dimensions overflow int") {
		t.Fatalf("color validator error = %v, want chroma-plane length overflow", err)
	}
}

func TestColor16RenderValidatorAcceptsSingleRowSubsampledGeometry(t *testing.T) {
	src := SourceFrame42016{
		Y:            []uint16{0, 0},
		U:            []uint16{0},
		V:            []uint16{0},
		YStride:      2,
		ChromaStride: 1,
		Width:        2,
		Height:       1,
		BitDepth:     10,
	}
	color := parser.ColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
	if err := validateSourceRenderFrameColor16(src, color, "4:2:0 test"); err != nil {
		t.Fatalf("single-row subsampled render: %v", err)
	}
}

func BenchmarkPred8(b *testing.B) {
	if pred8Impl == nil {
		b.Skip("no vector predictor in this build")
	}
	const w, h = 256, 128
	rng := rand.New(rand.NewSource(1))
	pix := make([]byte, w*h)
	rng.Read(pix)
	ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
	var im [pred8IMLen]int16
	var scratch motion.ConvolveScratch
	dst := make([]byte, 64*64)
	for _, tc := range []struct {
		name       string
		subX, subY int
	}{{"H", 8, 0}, {"V", 0, 8}, {"2D", 8, 8}, {"FP", 0, 0}} {
		b.Run("fast16/"+tc.name, func(b *testing.B) {
			for range b.N {
				pred8Predict(dst, 16, &ref, 64, 32, 16, 16, tc.subX, tc.subY, motion.RegularFilters, &im)
			}
		})
		b.Run("motion16/"+tc.name, func(b *testing.B) {
			d := frame.Plane{Pix: dst, Stride: 16, Width: 16, Height: 16}
			for range b.N {
				_ = motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(d, ref, 1, 8, 0, 0, 64, 32, 16, 16, 16, 16, tc.subX, tc.subY, motion.RegularFilters, &scratch)
			}
		})
	}
}

// TestPred8MatchesMotionPredictor checks the fast luma predictor against the
// motion package predictor for every Q4 phase, filter
// pair and block shape, over random, extreme and alternating planes.
func TestPred8MatchesMotionPredictor(t *testing.T) {
	if pred8Impl == nil {
		t.Skip("no vector predictor in this build")
	}
	const w, h = 160, 120
	rng := rand.New(rand.NewSource(17))
	pix := make([]byte, w*h)
	ref := frame.Plane{Pix: pix, Stride: w, Width: w, Height: h}
	var scratch motion.ConvolveScratch
	var im [pred8IMLen]int16
	want := make([]byte, 64*64)
	got := make([]byte, 64*64)
	for pattern := range 4 {
		for i := range pix {
			switch pattern {
			case 0:
				pix[i] = byte(rng.Intn(256))
			case 1:
				pix[i] = 255 * byte((i/w+i%w)&1)
			case 2:
				pix[i] = []byte{0, 255}[rng.Intn(2)]
			default:
				pix[i] = byte(128 + rng.Intn(9) - 4)
			}
		}
		for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}, {8, 16}, {32, 16}, {16, 32}, {64, 16}, {16, 64}, {64, 32}, {32, 64}} {
			bw, bh := sz[0], sz[1]
			for trial := range 40 {
				refX := 3 + rng.Intn(w-bw-20)
				refY := 3 + rng.Intn(h-bh-8)
				subX := rng.Intn(16)
				subY := rng.Intn(16)
				if trial < 4 {
					subX, subY = []int{0, 8, 0, 4}[trial], []int{0, 0, 8, 12}[trial]
				}
				filters := motion.InterpFilters{X: motion.InterpFilter(rng.Intn(3)), Y: motion.InterpFilter(rng.Intn(3))}
				dst := frame.Plane{Pix: want, Stride: bw, Width: bw, Height: bh}
				if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 1, 8, 0, 0, refX, refY, bw, bh, bw, bh, subX, subY, filters, &scratch); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					got[i] = 0xa5
				}
				if pred8FullPelOnly && subX|subY != 0 {
					if pred8Predict(got, bw, &ref, refX, refY, bw, bh, subX, subY, filters, &im) {
						t.Fatalf("%dx%d sub (%d,%d): full-pel-only fast path took a sub-pel block", bw, bh, subX, subY)
					}
					continue
				}
				if !pred8Predict(got, bw, &ref, refX, refY, bw, bh, subX, subY, filters, &im) {
					t.Fatalf("%dx%d at (%d,%d): fast path declined", bw, bh, refX, refY)
				}
				if !bytes.Equal(got[:bw*bh], want[:bw*bh]) {
					t.Fatalf("pattern %d %dx%d at (%d,%d) sub (%d,%d) filters %+v: prediction differs", pattern, bw, bh, refX, refY, subX, subY, filters)
				}
			}
		}
	}
}

// Frozen c07ac861 prepare path. The old quantizer and following dispatched
// statistics remain unchanged production functions, independent of the new
// fused helper. Compare all touched scratch, levels, state and panic behavior.
func (st *lossyEncodeState) prepareInterTXBDecideOriginalQuantRD(srcPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16) bool {
	var geo *txBlockGeometry
	switch {
	case w == 8 && h == 8:
		geo = &txBlockGeometry8x8
	case w == 16 && h == 16:
		geo = &txBlockGeometry16x16
	case w == 32 && h == 32:
		geo = &txBlockGeometry32x32
	}
	if geo == nil || len(qcoeff) < geo.coeffCount {
		return st.prepareInterTXB(srcPlane, pred, predStride, stride, px, py, w, h, q, qcoeff)
	}
	residualBlockImpl(st.resScratch[:geo.sampleCount], srcPlane, py*stride+px, stride, pred, predStride, w, h)
	dcZero, acZero := quantize.ZeroBinB(q, geo.txScale)
	thr0, thr := min(dcZero, acZero), acZero
	var mask uint8
	switch w {
	case 8:
		mask = transform.ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(st.tranScratch[:64]), (*[64]int16)(st.resScratch[:64]), thr0, thr)
	case 16:
		mask = transform.ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(st.tranScratch[:256]), (*[256]int16)(st.resScratch[:256]), thr0, thr)
	default:
		mask = transform.ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(st.tranScratch[:1024]), (*[1024]int16)(st.resScratch[:1024]), thr0, thr)
	}
	if mask == 0 {
		clear(qcoeff[:geo.coeffCount])
		return true
	}
	// Quantize only the live rows; when every level is zero the block adds
	// nothing to the skip comparison.
	if !quantize.QuantizeBRowGroupsTrusted(qcoeff[:geo.coeffCount], st.tranScratch[:geo.coeffCount], w, mask, q, geo.txScale) {
		return true
	}
	return st.accumulateInterTXBStatsDiff(geo, q, qcoeff)
}

func TestPrepareInterTXBQuantRDOriginalState(t *testing.T) {
	rng := rand.New(rand.NewSource(8145))
	for _, n := range []int{8, 16, 32} {
		for trial := 0; trial < 40; trial++ {
			q := []quantize.Quantizer{{DC: 4, AC: 4}, {DC: 120, AC: 140}, {DC: 1828, AC: 4}, {DC: 4, AC: 1828}, {DC: 147, AC: 148}, {DC: 148, AC: 147}, {DC: 65536, AC: 65536}}[trial%7]
			got, want := &lossyEncodeState{}, &lossyEncodeState{}
			for i := range got.resScratch {
				got.resScratch[i] = int16(i*101 - 32768)
			}
			for i := range got.tranScratch {
				got.tranScratch[i] = int32(i*733 - 39000)
			}
			for i := range got.lumaQ {
				got.lumaQ[i] = int16(i*11 + 0x1200)
			}
			got.rdDskip, got.rdDcode, got.rdRcode = -317, 117, 913
			*want = *got
			stride, predStride := n+7, n+3
			src, pred := make([]byte, (n+3)*stride), make([]byte, n*predStride)
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			for i := range pred {
				pred[i] = byte(rng.Intn(256))
			}
			if trial%5 == 0 {
				clear(src)
				clear(pred)
			}
			if trial%5 == 1 {
				for i := range src {
					src[i] = 255
				}
				clear(pred)
			}
			gq, wq := got.lumaQ[:n*n], want.lumaQ[:n*n]
			if trial%10 == 3 {
				gq, wq = gq[:len(gq)-1], wq[:len(wq)-1]
			}
			if trial%10 == 4 {
				src = src[:7]
			}
			if trial%10 == 9 {
				src = src[:6*stride]
			}
			var gv, wv bool
			var gp, wp any
			geob := eobUnknown
			func() {
				defer func() { gp = recover() }()
				geob = got.prepareInterTXBDecide(src, pred, predStride, stride, 3, 2, n, n, q, gq)
				gv = geob == 0
			}()
			if g, ok := rdoqSquareGeometry(n, 0); gp == nil && geob > 0 && ok {
				if want := rdoqEOBFromIScan(gq[:n*n], g.iscan); geob != want {
					t.Fatalf("trial %d n=%d: eob %d, levels give %d", trial, n, geob, want)
				}
			}
			func() {
				defer func() { wp = recover() }()
				wv = want.prepareInterTXBDecideOriginalQuantRD(src, pred, predStride, stride, 3, 2, n, n, q, wq)
			}()
			if gv != wv || reflect.TypeOf(gp) != reflect.TypeOf(wp) ||
				got.rdDskip != want.rdDskip || got.rdDcode != want.rdDcode || got.rdRcode != want.rdRcode ||
				got.resScratch != want.resScratch || got.tranScratch != want.tranScratch ||
				got.dqScratch != want.dqScratch || got.lumaQ != want.lumaQ {
				t.Fatalf("n=%d trial=%d: allzero=%v/%v panic=%v/%v rd=(%d,%d,%d)/(%d,%d,%d)", n, trial, gv, wv, gp, wp,
					got.rdDskip, got.rdDcode, got.rdRcode, want.rdDskip, want.rdDcode, want.rdRcode)
			}
		}
	}
}

func TestAppendQuantizationParamsPayloadZeroDeltas(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	payload, parsed := appendAndParseQuantizationParams(t, seq, QuantizationParams{})
	if len(payload) != 2 {
		t.Fatalf("payload len=%d want 2", len(payload))
	}
	if parsed.BaseQIdx != 0 || parsed.YDCDelta != 0 || parsed.UDCDelta != 0 || parsed.UACDelta != 0 {
		t.Fatalf("parsed quant=%+v", parsed)
	}
	if parsed.UsingQMatrix {
		t.Fatalf("qmatrix set in parsed quant=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadSeparateUVAndQMatrix(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.SeparateUVDeltaQ = true
	quant := QuantizationParams{
		BaseQIdx:      96,
		YDCDelta:      -2,
		UDCDelta:      5,
		UACDelta:      -3,
		VDCDelta:      7,
		VACDelta:      -9,
		DiffUVDeltas:  true,
		UsingQMatrix:  true,
		QMatrixLevelY: 2,
		QMatrixLevelU: 3,
		QMatrixLevelV: 4,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.BaseQIdx != quant.BaseQIdx || parsed.YDCDelta != quant.YDCDelta ||
		parsed.UDCDelta != quant.UDCDelta || parsed.UACDelta != quant.UACDelta ||
		parsed.VDCDelta != quant.VDCDelta || parsed.VACDelta != quant.VACDelta {
		t.Fatalf("parsed deltas=%+v want %+v", parsed, quant)
	}
	if !parsed.DiffUVDeltas || !parsed.UsingQMatrix ||
		parsed.QMatrixLevelY != 2 || parsed.QMatrixLevelU != 3 || parsed.QMatrixLevelV != 4 {
		t.Fatalf("parsed qmatrix=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadSharedUV(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.ColorConfig.SeparateUVDeltaQ = false
	quant := QuantizationParams{
		BaseQIdx:      10,
		UDCDelta:      4,
		VDCDelta:      4,
		UsingQMatrix:  true,
		QMatrixLevelY: 1,
		QMatrixLevelU: 6,
		QMatrixLevelV: 6,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.DiffUVDeltas || parsed.UDCDelta != 4 || parsed.VDCDelta != 4 || parsed.VACDelta != 0 {
		t.Fatalf("parsed shared uv=%+v", parsed)
	}
	if parsed.QMatrixLevelY != 1 || parsed.QMatrixLevelU != 6 || parsed.QMatrixLevelV != 6 {
		t.Fatalf("parsed qmatrix=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadMonoChrome(t *testing.T) {
	seq := realtimeEncoderMonochromeSequenceHeader()
	quant := QuantizationParams{
		BaseQIdx: 3,
		YDCDelta: -1,
	}
	_, parsed := appendAndParseQuantizationParams(t, seq, quant)
	if parsed.BaseQIdx != 3 || parsed.YDCDelta != -1 {
		t.Fatalf("parsed mono quant=%+v", parsed)
	}
	if parsed.UDCDelta != 0 || parsed.UACDelta != 0 || parsed.VDCDelta != 0 || parsed.VACDelta != 0 {
		t.Fatalf("parsed mono uv=%+v", parsed)
	}
}

func TestAppendQuantizationParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	cases := [...]QuantizationParams{
		{YDCDelta: -65},
		{YDCDelta: 64},
		{DiffUVDeltas: true},
		{UDCDelta: 1},
		{UsingQMatrix: true, QMatrixLevelY: 16},
		{UsingQMatrix: false, QMatrixLevelY: 1},
		{UsingQMatrix: true, QMatrixLevelU: 3, QMatrixLevelV: 4},
	}
	var buf [8]byte
	for _, quant := range cases {
		if _, err := appendQuantizationParamsPayload(buf[:0], seq, quant); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendQuantizationParamsPayload(%+v) err=%v want ErrInvalidFrame", quant, err)
		}
	}

	mono := seq
	mono.ColorConfig.MonoChrome = true
	if _, err := appendQuantizationParamsPayload(buf[:0], mono, QuantizationParams{UDCDelta: 1}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono uv delta err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseQuantizationParams(t *testing.T, seq SequenceHeader, quant QuantizationParams) ([]byte, parser.QuantizationParams) {
	t.Helper()
	var buf [16]byte
	out, err := appendQuantizationParamsPayload(buf[:0], seq, quant)
	if err != nil {
		t.Fatalf("appendQuantizationParamsPayload: %v", err)
	}
	parsedSeq := parseEncoderSequenceHeader(t, seq)
	parsed, err := parser.ParseQuantizationParams(out, parsedSeq, parser.TileInfo{})
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	return out, parsed
}

func TestRDOQEOBMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{16, 64, 256, 1024} {
		iscan := rng.Perm(n)
		is := make([]int16, n)
		for i, v := range iscan {
			is[i] = int16(v)
		}
		q := make([]int16, n)
		for trial := range 500 {
			clear(q)
			for k := rng.Intn(6); k > 0; k-- {
				q[rng.Intn(n)] = int16(rng.Intn(9) - 4)
			}
			want := 0
			for ci, v := range q {
				if v != 0 {
					want = max(want, int(is[ci])+1)
				}
			}
			if got := rdoqEOBFromIScan(q, is); got != want {
				t.Fatalf("n=%d trial %d: eob %d want %d", n, trial, got, want)
			}
			if got := rdoqEOBPureGo(q, is); got != want {
				t.Fatalf("n=%d trial %d: pure eob %d want %d", n, trial, got, want)
			}
		}
	}
}

func TestLevelsAllZeroFindsEveryLevel(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 7, 16, 24, 64, 100, 256, 1024} {
		q := make([]int16, n)
		if !levelsAllZero(q) || !levelsAllZeroPureGo(q) {
			t.Fatalf("n=%d: zero levels reported non-zero", n)
		}
		for i := range q {
			for _, v := range []int16{1, -1, 256, -32768} {
				q[i] = v
				if levelsAllZero(q) || levelsAllZeroPureGo(q) {
					t.Fatalf("n=%d: level %d at %d reported zero", n, v, i)
				}
			}
			q[i] = 0
		}
		if n >= 2 {
			q[0], q[n-1] = 1, -1
			if levelsAllZero(q) || levelsAllZeroPureGo(q) {
				t.Fatalf("n=%d: cancelling levels reported zero", n)
			}
		}
	}
}

// TestRDStatsDiffMatchesBlockStats checks the difference-only statistics
// against rdStatsBlockPureGo's dskip - dcode, rate and all-zero flag.
func TestRDStatsDiffMatchesBlockStats(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for trial := range 4000 {
		count := []int{16, 64, 256, 1024, 24, 7}[trial%6]
		tran := make([]int32, count)
		qc := make([]int16, count)
		step := int32(rng.Intn(3000) + 1)
		ts := uint8(rng.Intn(3))
		density := []int{0, 1, 4, 16, 64}[trial%5]
		for i := range tran {
			tran[i] = rng.Int31n(1<<18) - 1<<17
			if density > 0 && rng.Intn(64) < density {
				qc[i] = int16(rng.Intn(65536) - 32768)
			}
		}
		if trial%7 == 0 {
			qc[0] = -32768
		}
		dskip, dcode, wantRate, wantZero := rdStatsBlockPureGo(tran, qc, count, step, ts)
		for name, fn := range map[string]func([]int32, []int16, int, int32, uint8) (int64, int64, bool){
			"scalar":   rdStatsDiffPureGo,
			"dispatch": rdStatsDiffImpl,
		} {
			diff, rate, zero := fn(tran, qc, count, step, ts)
			if diff != dskip-dcode || rate != wantRate || zero != wantZero {
				t.Fatalf("%s trial %d: diff %d rate %d zero %v, want %d %d %v", name, trial, diff, rate, zero, dskip-dcode, wantRate, wantZero)
			}
		}
	}
}

func TestReconKernelsMatchPureGoExactWindows(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const stride = 80
	for _, sz := range [][2]int{{4, 4}, {8, 8}, {8, 16}, {16, 8}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 16}, {64, 64}} {
		w, h := sz[0], sz[1]
		pred := make([]byte, w*h)
		rng.Read(pred)
		for trial := range 40 {
			px, py := rng.Intn(stride-w+1), rng.Intn(4)
			want := make([]byte, (py+h-1)*stride+px+w)
			rng.Read(want)
			got := bytes.Clone(want)
			for _, r := range []int16{int16(rng.Intn(801) - 400), 0, 1, -1, 255, -255, 256, -256, 32767, -32768} {
				rng.Read(want)
				copy(got, want)
				addConstantResidualToPredReconBlockPureGo(want, pred, w, stride, px, py, w, h, r)
				addConstantResidualToPredReconBlock(got, pred, w, stride, px, py, w, h, r)
				if !bytes.Equal(got, want) {
					t.Fatalf("add-constant %dx%d r=%d trial %d differs", w, h, r, trial)
				}
			}
			rng.Read(want)
			copy(got, want)
			copyPredToReconBlockPureGo(want, pred, w, stride, px, py, w, h)
			copyPredToReconBlock(got, pred, w, stride, px, py, w, h)
			if !bytes.Equal(got, want) {
				t.Fatalf("copy %dx%d trial %d differs", w, h, trial)
			}
		}
	}
}

// rawPredReconOriginal retains the original complete copy followed by an
// independent scalar expression of the DSP numeric contract. The copy kernel
// itself is unchanged. pred is dst itself or disjoint from it (the
// addRawResidualToPredReconBlock contract).
func rawPredReconOriginal(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	copyPredToReconBlock(dst, pred, predStride, stride, 0, 0, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := (int64(raw[y*w+x]) + 8) >> 4
			if r < -32768 {
				r = -32768
			}
			if r > 32767 {
				r = 32767
			}
			v := int64(dst[y*stride+x]) + r
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			dst[y*stride+x] = byte(v)
		}
	}
}

var rawPredReconBoundaries = []int32{
	-1 << 31, -1<<31 + 7, -524305, -524304, -524297, -524296,
	-4097, -4096, -4089, -4088, -25, -24, -17, -16, -9, -8, -7, -1,
	0, 1, 7, 8, 9, 15, 16, 23, 24, 4071, 4072, 4087, 4088,
	524279, 524280, 524287, 524288, 524295, 524296, 1<<31 - 9, 1<<31 - 8, 1<<31 - 1,
}

func TestRawPredReconNumericBoundaries(t *testing.T) {
	const w = 64
	h := 256 * len(rawPredReconBoundaries) / w
	for _, kernel := range []struct {
		name string
		fn   func([]byte, int, []byte, int, []int32, int, int)
	}{{"scalar", addRawResidualToPredReconBlockPureGo}, {"dispatch", addRawResidualToPredReconBlock}} {
		t.Run(kernel.name, func(t *testing.T) {
			pred := make([]byte, w*h)
			raw := make([]int32, w*h)
			for i := range raw {
				pred[i] = byte(i)
				raw[i] = rawPredReconBoundaries[i/256]
			}
			want, got := make([]byte, w*h), make([]byte, w*h)
			rawPredReconOriginal(want, w, pred, w, raw, w, h)
			kernel.fn(got, w, pred, w, raw, w, h)
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("pred=%d raw=%d: got %d want %d", pred[i], raw[i], got[i], want[i])
					}
				}
			}
		})
	}
}

func TestRawPredReconExactWindowsAndAliasing(t *testing.T) {
	for _, kernel := range []struct {
		name string
		fn   func([]byte, int, []byte, int, []int32, int, int)
	}{{"scalar", addRawResidualToPredReconBlockPureGo}, {"dispatch", addRawResidualToPredReconBlock}} {
		for _, w := range []int{4, 5, 8, 12, 16, 24, 32, 64} {
			for _, h := range []int{1, 2, 3, 8} {
				stride := w + 7
				span := (h-1)*stride + w
				for _, layout := range []struct {
					name         string
					d, p, ds, ps int
				}{
					{"disjoint", 16, 32 + span, stride, stride},
					{"inplace", 16, 16, stride, stride},
					{"differentstride", 16, 32 + span, w + 9, w + 4},
				} {
					t.Run(fmt.Sprintf("%s/%dx%d/%s", kernel.name, w, h, layout.name), func(t *testing.T) {
						dlen := (h-1)*layout.ds + w
						plen := (h-1)*layout.ps + w
						n := max(layout.d+dlen, layout.p+plen) + 19
						got := make([]byte, n)
						for i := range got {
							got[i] = byte(i*73 + 17)
						}
						want := slices.Clone(got)
						raw := make([]int32, w*h)
						for i := range raw {
							raw[i] = rawPredReconBoundaries[(i*13+w+h)%len(rawPredReconBoundaries)]
						}
						beforeRaw := slices.Clone(raw)
						rawPredReconOriginal(want[layout.d:layout.d+dlen:layout.d+dlen], layout.ds, want[layout.p:layout.p+plen:layout.p+plen], layout.ps, raw, w, h)
						kernel.fn(got[layout.d:layout.d+dlen:layout.d+dlen], layout.ds, got[layout.p:layout.p+plen:layout.p+plen], layout.ps, raw, w, h)
						if !bytes.Equal(got, want) {
							t.Fatal("output, padding, or in-place prediction differs from original")
						}
						if !slices.Equal(raw, beforeRaw) {
							t.Fatal("raw input changed")
						}
					})
				}
			}
		}
	}
}

func TestReconInterTXBResultOriginalState(t *testing.T) {
	for _, dims := range [][2]int{{4, 4}, {4, 16}, {16, 4}, {8, 8}, {8, 16}, {16, 8}, {16, 16}, {16, 32}, {32, 16}, {32, 32}} {
		w, h := dims[0], dims[1]
		geo, err := txBlockGeometryForDimensions(w, h)
		if err != nil {
			t.Fatal(err)
		}
		scan, err := transform.DefaultScan(geo.size, transform.Class2D)
		if err != nil {
			t.Fatal(err)
		}
		for _, txType := range []transform.Type{transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeIDTX, transform.Type(255)} {
			for _, eob := range []int{0, 1, 2, 3, geo.coeffCount / 2, geo.coeffCount} {
				for _, allZero := range []bool{false, true} {
					t.Run(fmt.Sprintf("%dx%d/type%d/eob%d/zero%t", w, h, txType, eob, allZero), func(t *testing.T) {
						var got, want lossyEncodeState
						got.invScratch = make([]int32, 1024)
						for i := range got.invScratch {
							got.invScratch[i] = int32(i*911 + 17)
							got.dqScratch[i] = int32(i*131 - 7777)
							got.invResidual[i] = int16(i*31 - 12345)
						}
						want.dqScratch, want.invResidual = got.dqScratch, got.invResidual
						want.invScratch = slices.Clone(got.invScratch)
						qcoeff := make([]int16, geo.coeffCount)
						for i := range qcoeff {
							qcoeff[i] = int16((i*23+1)%97 - 48)
						}
						qBefore, scanBefore := slices.Clone(qcoeff), slices.Clone(scan)
						stride, predStride, px, py := w+13, w+3, 5, 2
						recon := make([]byte, (py+h+2)*stride)
						for i := range recon {
							recon[i] = byte(i*53 + 121)
						}
						reference := slices.Clone(recon)
						pred := make([]byte, (h-1)*predStride+w)
						for i := range pred {
							pred[i] = byte(i*71 + 29)
						}
						predBefore := slices.Clone(pred)
						q := quantize.Quantizer{DC: 1217, AC: 2039}
						result := tile.TXBDecodeResult{AllZero: allZero, EOB: uint16(eob)}
						wantErr := want.reconInterTXBResultOriginal(reference, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, result)
						gotErr := got.reconInterTXBResult(recon, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, result)
						if gotErr != wantErr {
							t.Fatalf("error %v want %v", gotErr, wantErr)
						}
						if !bytes.Equal(recon, reference) {
							t.Fatal("reconstruction or guard bytes differ")
						}
						if got.dqScratch != want.dqScratch || got.invResidual != want.invResidual || !slices.Equal(got.invScratch, want.invScratch) {
							t.Fatal("scratch state differs")
						}
						if !slices.Equal(qcoeff, qBefore) || !slices.Equal(scan, scanBefore) || !bytes.Equal(pred, predBefore) {
							t.Fatal("input changed")
						}
					})
				}
			}
		}
	}
}

// Frozen from b9e260f470c33a5369b5357d74a54b302044ea74. Keep the two-pass
// output and separate scan traversal independent of optimized reconstruction.
func (st *lossyEncodeState) reconInterTXBResultOriginal(reconPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16,
	geo txBlockGeometry, scan []int16, txType transform.Type, txbResult tile.TXBDecodeResult) error {
	n := geo.sampleCount
	if txbResult.AllZero {
		copyPredToReconBlock(reconPlane, pred, predStride, stride, px, py, w, h)
		return nil
	}
	if txbResult.EOB == 1 && txType == transform.TypeDCTDCT {
		dc := quantize.DequantizeDCCoeffBitDepthTrusted(qcoeff[0], q, geo.txScale, 8)
		sample, err := transform.InverseDCTDCOnlySampleBitDepth(dc, st.invScratch[:n], geo.size, 8)
		if err != nil {
			return err
		}
		addConstantResidualToPredReconBlock(reconPlane, pred, predStride, stride, px, py, w, h, sample)
		return nil
	}

	dq := &st.dqScratch
	eob := int(txbResult.EOB)
	quantize.DequantizeBlockScaledBitDepthEOBTrusted(dq[:geo.coeffCount], geo.coeffHeight, qcoeff, geo.coeffHeight, scan, eob, geo.coeffWidth, geo.coeffHeight, q, geo.txScale, 8)
	if txType != transform.TypeIDTX {
		raw := st.invScratch[:n]
		// Every coefficient past the EOB is zero, so the row pass only needs
		// the transform rows the scan prefix reaches.
		if err := transform.InverseBlockBitDepthRaw(raw, dq[:geo.coeffCount], geo.coeffHeight, geo.size, txType, 8, scanActiveRowsOriginal(scan, eob, geo.coeffHeight)); err != nil {
			return err
		}
		copyPredToReconBlock(reconPlane, pred, predStride, stride, px, py, w, h)
		dst := reconPlane[py*stride+px:]
		dsp.AddRawTransformPlaneBlockTrusted(dst, stride, 1, 0xff, w, h, raw, w)
		return nil
	}

	res := &st.invResidual
	if err := transform.InverseBlock(res[:n], w, dq[:geo.coeffCount], geo.coeffHeight, st.invScratch[:n], geo.size, txType); err != nil {
		return err
	}
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*predStride+c]) + int(res[r*w+c])
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			reconPlane[row+c] = uint8(v)
		}
	}
	return nil

}
func scanActiveRowsOriginal(scan []int16, eob, coeffHeight int) int {
	if eob > len(scan) {
		return 0
	}
	mask := coeffHeight - 1
	rows := 0
	for _, p := range scan[:eob] {
		if r := int(p) & mask; r >= rows {
			rows = r + 1
			if rows == coeffHeight {
				break
			}
		}
	}
	return rows
}

func TestLossyEncodeStateScansCoverThinChromaMaxTransformSizes(t *testing.T) {
	var st lossyEncodeState
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	cases := []struct {
		name  string
		block tile.BlockSize
		color parser.ColorConfig
		plane int
		want  tile.TransformSize
	}{
		{name: "420_8x32", block: tile.BlockSize8x32, color: color420, plane: 1, want: tile.TransformSize4x16},
		{name: "420_32x8", block: tile.BlockSize32x8, color: color420, plane: 1, want: tile.TransformSize16x4},
		{name: "420_16x64", block: tile.BlockSize16x64, color: color420, plane: 1, want: tile.TransformSize8x32},
		{name: "420_64x16", block: tile.BlockSize64x16, color: color420, plane: 1, want: tile.TransformSize32x8},
		{name: "luma_16x64", block: tile.BlockSize16x64, plane: 0, want: tile.TransformSize16x64},
		{name: "luma_64x16", block: tile.BlockSize64x16, plane: 0, want: tile.TransformSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tile.MaxTransformSize(tc.block, tc.color, tc.plane)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("MaxTransformSize=%d want %d", got, tc.want)
			}
			dims, ok := got.Dimensions()
			if !ok {
				t.Fatalf("invalid transform size %d", got)
			}
			scan, ok := st.scanForTransformSize(got)
			if !ok {
				scan, ok = st.scanForThinTransformSize(got)
			}
			if !ok {
				t.Fatalf("missing scan for transform %d", got)
			}
			txSize, err := got.TransformSize()
			if err != nil {
				t.Fatal(err)
			}
			scanSize, err := transform.ScanSize(txSize)
			if err != nil {
				t.Fatal(err)
			}
			wantLen := int(scanSize.Width) * int(scanSize.Height)
			if len(scan) != wantLen {
				t.Fatalf("scan len=%d want %d", len(scan), wantLen)
			}
			if wantLen > int(dims.W4)*4*int(dims.H4)*4 {
				t.Fatalf("adjusted scan len=%d larger than transform area", wantLen)
			}
		})
	}
}

func TestForwardDCTBlockSupportsThinRectangles(t *testing.T) {
	cases := []struct {
		name string
		w    int
		h    int
	}{
		{name: "4x16", w: 4, h: 16},
		{name: "16x4", w: 16, h: 4},
		{name: "8x32", w: 8, h: 32},
		{name: "32x8", w: 32, h: 8},
		{name: "16x64", w: 16, h: 64},
		{name: "64x16", w: 64, h: 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			size, err := transform.SizeFromDimensions(tc.w, tc.h)
			if err != nil {
				t.Fatal(err)
			}
			scanSize, err := transform.ScanSize(size)
			if err != nil {
				t.Fatal(err)
			}
			n := tc.w * tc.h
			residual := make([]int16, n)
			for i := range residual {
				residual[i] = int16((i*17)%511 - 255)
			}
			coeff := make([]int32, int(scanSize.Width)*int(scanSize.Height))
			if err := forwardDCTBlock(coeff, residual, tc.w, tc.h); err != nil {
				t.Fatalf("forwardDCTBlock(%dx%d): %v", tc.w, tc.h, err)
			}
			if coeff[0] == 0 {
				t.Fatalf("forwardDCTBlock(%dx%d) produced zero DC", tc.w, tc.h)
			}
		})
	}
}

func TestTXBlockGeometryUsesAdjusted64ExtentCoeffPlane(t *testing.T) {
	cases := []struct {
		name        string
		size        tile.TransformSize
		wantW       int
		wantH       int
		wantCoeffW  int
		wantCoeffH  int
		wantScale   uint8
		wantSamples int
		wantCoeffs  int
	}{
		{name: "16x64", size: tile.TransformSize16x64, wantW: 16, wantH: 64, wantCoeffW: 16, wantCoeffH: 32, wantScale: 1, wantSamples: 1024, wantCoeffs: 512},
		{name: "64x16", size: tile.TransformSize64x16, wantW: 64, wantH: 16, wantCoeffW: 32, wantCoeffH: 16, wantScale: 1, wantSamples: 1024, wantCoeffs: 512},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			geo, err := txBlockGeometryForTransformSize(tc.size)
			if err != nil {
				t.Fatal(err)
			}
			if geo.width != tc.wantW || geo.height != tc.wantH ||
				geo.coeffWidth != tc.wantCoeffW || geo.coeffHeight != tc.wantCoeffH ||
				geo.sampleCount != tc.wantSamples || geo.coeffCount != tc.wantCoeffs ||
				geo.txScale != tc.wantScale {
				t.Fatalf("geo=%+v", geo)
			}
			if scratchLen, err := tile.CoeffLevelsScratchLen(tc.size); err != nil {
				t.Fatal(err)
			} else if scratchLen > len(newReducedTXBTestState(t, 80).levels) {
				t.Fatalf("scratch len=%d exceeds encoder levels len=%d", scratchLen, len(newReducedTXBTestState(t, 80).levels))
			}
		})
	}
}

func TestReduced64ExtentTXBEncoderHelpers(t *testing.T) {
	cases := []struct {
		name  string
		size  tile.TransformSize
		w     int
		h     int
		block tile.BlockSize
	}{
		{name: "16x64", size: tile.TransformSize16x64, w: 16, h: 64, block: tile.BlockSize16x64},
		{name: "64x16", size: tile.TransformSize64x16, w: 64, h: 16, block: tile.BlockSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := newReducedTXBTestState(t, 80)
			scan, ok := st.scanForThinTransformSize(tc.size)
			if !ok {
				t.Fatalf("missing scan for %s", tc.name)
			}
			geo, err := txBlockGeometryForTransformSize(tc.size)
			if err != nil {
				t.Fatal(err)
			}
			if len(scan) != geo.coeffCount {
				t.Fatalf("scan len=%d coeffCount=%d", len(scan), geo.coeffCount)
			}
			src := make([]byte, tc.w*tc.h)
			recon := make([]byte, tc.w*tc.h)
			pred := make([]byte, tc.w*tc.h)
			for i := range src {
				src[i] = uint8((i*29 + 17) & 0xff)
				pred[i] = uint8((i*7 + 91) & 0xff)
			}
			ctxReq := tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: tc.block,
				Size:       tc.size,
			}
			var coeffCtx tile.CoeffEntropyContext
			if err := st.encodeTXBPredRect(recon, src, tc.w, 0, 0, tc.w, tc.h, st.yQuant, ctxReq, &coeffCtx, scan, nil, pred); err != nil {
				t.Fatalf("encodeTXBPredRect: %v", err)
			}

			st = newReducedTXBTestState(t, 80)
			qcoeff := st.lumaQ[:geo.coeffCount]
			if st.prepareInterTXB(src, pred, tc.w, tc.w, 0, 0, tc.w, tc.h, st.yQuant, qcoeff) {
				t.Fatal("prepareInterTXB unexpectedly produced all-zero coefficients")
			}
			for i := geo.coeffCount; i < geo.sampleCount; i++ {
				st.lumaQ[i] = int16(12345)
			}
			st.interTxTypeReq.Size = tc.size
			st.interTxType = transform.TypeDCTDCT
			st.afterSkipInter = func() error {
				return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, st.interTxType)
			}
			var interCtx tile.CoeffEntropyContext
			recon = make([]byte, tc.w*tc.h)
			if err := st.finishInterTXB(recon, pred, tc.w, tc.w, 0, 0, tc.w, tc.h, st.yQuant, qcoeff, ctxReq, &interCtx, scan, st.afterSkipInter, tile.EOBUnknown); err != nil {
				t.Fatalf("finishInterTXB: %v", err)
			}
		})
	}
}

func TestReduced64ExtentHighBDTXBEncoderHelper(t *testing.T) {
	cases := []struct {
		name  string
		size  tile.TransformSize
		w     int
		h     int
		block tile.BlockSize
	}{
		{name: "16x64", size: tile.TransformSize16x64, w: 16, h: 64, block: tile.BlockSize16x64},
		{name: "64x16", size: tile.TransformSize64x16, w: 64, h: 16, block: tile.BlockSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const bitDepth = 10
			st := newReducedTXBTestState(t, 80)
			q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, 80, bitDepth, quantize.PlaneY)
			if err != nil {
				t.Fatal(err)
			}
			st.yQuant = q
			scan, ok := st.scanForThinTransformSize(tc.size)
			if !ok {
				t.Fatalf("missing scan for %s", tc.name)
			}
			src := make([]uint16, tc.w*tc.h)
			recon := make([]uint16, tc.w*tc.h)
			pred := make([]uint16, tc.w*tc.h)
			for i := range src {
				src[i] = uint16((i*41 + 77) & ((1 << bitDepth) - 1))
				pred[i] = uint16((i*11 + 303) & ((1 << bitDepth) - 1))
			}
			ctxReq := tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: tc.block,
				Size:       tc.size,
			}
			var coeffCtx tile.CoeffEntropyContext
			if err := st.encodeTXBPredRect16(recon, src, tc.w, 0, 0, tc.w, tc.h, bitDepth, st.yQuant, ctxReq, &coeffCtx, scan, nil, pred); err != nil {
				t.Fatalf("encodeTXBPredRect16: %v", err)
			}
		})
	}
}

func newReducedTXBTestState(t *testing.T, qIndex uint8) *lossyEncodeState {
	t.Helper()
	st := newTXTypeTestState(t, qIndex)
	if err := st.coeffCDFs.InitDefault(qIndex); err != nil {
		t.Fatal(err)
	}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			t.Fatal(err)
		}
		*dst = q
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<16))
	st.w = &w
	return st
}

// TestRefPadPlaneMatchesClampedReads: every sample of the padded copy is the
// source sample at the clamped position, the decoder's reference edge rule.
func TestRefPadPlaneMatchesClampedReads(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const w, h, stride, border = 37, 23, 41, 9
	src := make([]byte, stride*h)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	var p refPadPlane
	p.build(src, stride, w, h, border)
	for y := -border; y < h+border; y++ {
		for x := -border; x < w+border; x++ {
			want := src[min(max(y, 0), h-1)*stride+min(max(x, 0), w-1)]
			if got := p.buf[(y+border)*p.stride+x+border]; got != want {
				t.Fatalf("(%d,%d) = %d, want %d", x, y, got, want)
			}
		}
	}
}

// TestPredictPlanePaddedMatchesClamped: edge and outside-picture predictions
// through the padded copy equal the clamped predictor's.
func TestPredictPlanePaddedMatchesClamped(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const w, h = 128, 96
	ref := make([]byte, w*h)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	var plain, padded lossyEncodeState
	var cache refPadCache
	frame := SourceFrame420{Y: ref, U: ref[:64*48], V: ref[64*48 : 2*64*48], YStride: w, ChromaStride: 64, Width: w, Height: h}
	padded.refPads = cache.padsFor(frame, nil, 64, 48)
	got := make([]byte, 64*64)
	want := make([]byte, 64*64)
	for range 400 {
		bw := 8 << rng.Intn(4)
		bh := bw
		px := rng.Intn(w/8) * 8
		py := rng.Intn(h/8) * 8
		mv := motion.Vector{Row: int16(rng.Intn(161) - 80), Col: int16(rng.Intn(161) - 80)}
		if err := plain.predictPlane(want[:bw*bh], ref, w, w, h, px, py, bw, bh, mv, false, false, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
		if err := padded.predictPlane(got[:bw*bh], ref, w, w, h, px, py, bw, bh, mv, false, false, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
		for i := range bw * bh {
			if got[i] != want[i] {
				t.Fatalf("block %dx%d at (%d,%d) mv %+v: sample %d = %d, want %d", bw, bh, px, py, mv, i, got[i], want[i])
			}
		}
	}
}

func TestAppendRestorationParamsPayloadSkipsLosslessWithoutSuperres(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	payload, parsed := appendAndParseRestorationParams(t, seq, size, true, RestorationParams{})
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.BitsRead != 0 || parsed.UnitSizeY != 0 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadAllNone(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		UnitSizeYLog2:  8,
		UnitSizeUVLog2: 8,
		UnitSizeY:      RestorationUnitMax,
		UnitSizeUV:     RestorationUnitMax,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationNone || parsed.Type[1] != parser.RestorationNone || parsed.Type[2] != parser.RestorationNone {
		t.Fatalf("parsed types=%+v", parsed.Type)
	}
	if parsed.UnitSizeY != parser.RestorationUnitMax || parsed.UnitSizeUV != parser.RestorationUnitMax ||
		parsed.UnitSizeYLog2 != 8 || parsed.UnitSizeUVLog2 != 8 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadTypesAndUnitSizes(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{Resolution: Resolution{Width: 64, Height: 64}})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	seq.EnableRestoration = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationWiener, RestorationSGRProj, RestorationNone},
		UnitSizeYLog2:  7,
		UnitSizeUVLog2: 6,
		UnitSizeY:      128,
		UnitSizeUV:     64,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationWiener || parsed.Type[1] != parser.RestorationSGRProj || parsed.Type[2] != parser.RestorationNone {
		t.Fatalf("parsed types=%+v", parsed.Type)
	}
	if parsed.UnitSizeY != 128 || parsed.UnitSizeUV != 64 ||
		parsed.UnitSizeYLog2 != 7 || parsed.UnitSizeUVLog2 != 6 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayload128SuperblockUnitSize(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.Use128x128Superblock = true
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationSGRProj, RestorationNone, RestorationNone},
		UnitSizeYLog2:  8,
		UnitSizeUVLog2: 8,
		UnitSizeY:      256,
		UnitSizeUV:     256,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.UnitSizeY != 256 || parsed.UnitSizeUV != 256 || parsed.UnitSizeYLog2 != 8 {
		t.Fatalf("parsed unit sizes=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadMonochromeReadsYOnly(t *testing.T) {
	seq := realtimeEncoderMonochromeSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationSwitchable, RestorationNone, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	_, parsed := appendAndParseRestorationParams(t, seq, size, false, restoration)
	if parsed.Type[0] != parser.RestorationSwitchable || parsed.Type[1] != parser.RestorationNone || parsed.UnitSizeY != 64 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsInterPayload(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := InterFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 1}
	restoration := RestorationParams{
		Type:           [3]RestorationType{RestorationNone, RestorationWiener, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	var buf [4]byte
	payload, err := appendRestorationParamsInterPayload(buf[:0], seq, size, false, restoration)
	if err != nil {
		t.Fatalf("appendRestorationParamsInterPayload: %v", err)
	}
	parsed, err := parser.ParseRestorationParams(payload, parseEncoderSequenceHeader(t, seq), parser.FrameSize{}, parser.SegmentationParams{}, parser.CDEFParams{})
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	if parsed.Type[1] != parser.RestorationWiener || parsed.UnitSizeY != 64 || parsed.UnitSizeUV != 64 {
		t.Fatalf("parsed inter restoration=%+v", parsed)
	}
}

func TestAppendRestorationParamsPayloadRejectsInvalid(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size := IntraFrameSize{UpscaledWidth: 64, Height: 64, SuperResDenominator: 8, RefreshFrameFlags: 0xff}
	active := RestorationParams{
		Type:           [3]RestorationType{RestorationWiener, RestorationNone, RestorationNone},
		UnitSizeYLog2:  6,
		UnitSizeUVLog2: 6,
		UnitSizeY:      64,
		UnitSizeUV:     64,
	}
	cases := [...]RestorationParams{
		{Type: [3]RestorationType{4, 0, 0}},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 5},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 9},
		{Type: [3]RestorationType{RestorationWiener, 0, 0}, UnitSizeYLog2: 6, UnitSizeY: 128},
		{UnitSizeYLog2: 7},
	}
	var buf [8]byte
	for _, restoration := range cases {
		if _, err := appendRestorationParamsPayload(buf[:0], seq, size, false, restoration); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendRestorationParamsPayload(%+v) err=%v want ErrInvalidFrame", restoration, err)
		}
	}
	mono := seq
	mono.ColorConfig.MonoChrome = true
	if _, err := appendRestorationParamsPayload(buf[:0], mono, size, false, RestorationParams{Type: [3]RestorationType{0, RestorationWiener, 0}}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("mono chroma restoration err=%v want ErrInvalidFrame", err)
	}
	if _, err := appendRestorationParamsPayload(buf[:0], seq, size, true, active); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("lossless disabled restoration err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseRestorationParams(t *testing.T, seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) ([]byte, parser.RestorationParams) {
	t.Helper()
	var buf [8]byte
	payload, err := appendRestorationParamsPayload(buf[:0], seq, size, allLossless, restoration)
	if err != nil {
		t.Fatalf("appendRestorationParamsPayload: %v", err)
	}
	parsed, err := parser.ParseRestorationParams(
		payload,
		parseEncoderSequenceHeader(t, seq),
		parser.FrameSize{AllowIntrabc: size.AllowIntrabc, SuperResEnabled: size.SuperResDenominator != 8},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.CDEFParams{},
	)
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	return payload, parsed
}

// TestSAD8x8ImplMatchesPureGo proves the dispatched SAD kernel is bit-exact
// with the portable reference across random data, strides, and offsets
// (including totals far above any early-exit threshold).
func TestSAD8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const stride = 73 // deliberately odd
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*48) + rng.Intn(stride-8)
		want := sad8x8PureGo(src[off:], ref[off:], stride, 1<<30)
		got := sad8x8Impl(src[off:], ref[off:], stride, 1<<30)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD8x8x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const stride = 77
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*48) + rng.Intn(stride-27)
		w0, w1, w2, w3 := sad8x8x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad8x8x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD8x8x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(424))
	const stride = 79
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 8
		col := rng.Intn(stride-24) + 8
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad8x8x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad8x8x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD16x16x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(425))
	const stride = 91
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 16
		col := rng.Intn(stride-40) + 16
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad16x16x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad16x16x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD32x32x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(426))
	const stride = 137
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 32
		col := rng.Intn(stride-72) + 32
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad32x32x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad32x32x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

// TestSAD16x16ImplMatchesPureGo proves the 16x16 kernel bit-exact with the
// portable reference.
func TestSAD16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 91
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*64) + rng.Intn(stride-16)
		want := sad16x16PureGo(src[off:], ref[off:], stride)
		got := sad16x16Impl(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD16x16x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	const stride = 93
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*64) + rng.Intn(stride-35)
		w0, w1, w2, w3 := sad16x16x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad16x16x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

// TestSAD32x32ImplMatchesPureGo proves the 32x32 kernel bit-exact with the
// portable reference.
func TestSAD32x32ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(45))
	const stride = 113
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*80) + rng.Intn(stride-32)
		want := sad32x32PureGo(src[off:], ref[off:], stride)
		got := sad32x32Impl(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD64x64MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4564))
	const stride = 151
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*96) + rng.Intn(stride-64)
		want := sadRectBlockReference(src, ref, off, off, stride, 64, 64)
		got := sad64x64(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD64x64ComposedMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(45640))
	const stride = 151
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*96) + rng.Intn(stride-64)
		want := sadRectBlockReference(src, ref, off, off, stride, 64, 64)
		got := sad64x64Composed(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD64x64x4Step4MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4565))
	const (
		stride = 151
		height = 160
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(height - 64)
		col := rng.Intn(stride - 76)
		off := row*stride + col
		w0 := sadRectBlockReference(src, ref, off, off, stride, 64, 64)
		w1 := sadRectBlockReference(src, ref, off, off+4, stride, 64, 64)
		w2 := sadRectBlockReference(src, ref, off, off+8, stride, 64, 64)
		w3 := sadRectBlockReference(src, ref, off, off+12, stride, 64, 64)
		g0, g1, g2, g3 := sad64x64x4Step4(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestFullPelSearchSAD4Step4MatchesSingles(t *testing.T) {
	rng := rand.New(rand.NewSource(4567))
	const (
		stride = 160
		height = 160
		px     = 64
		py     = 64
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	shapes := [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}}
	base := py*stride + px
	srcBlock := src[base:]
	for _, shape := range shapes {
		bw, bh := shape[0], shape[1]
		for range 500 {
			dx := rng.Intn(17) - 8
			dy := rng.Intn(17) - 8
			ref0 := ref[base+dy*stride+dx:]
			w0 := fullPelSearchSAD(srcBlock, ref0, stride, bw, bh, 1<<30)
			w1 := fullPelSearchSAD(srcBlock, ref0[4:], stride, bw, bh, 1<<30)
			w2 := fullPelSearchSAD(srcBlock, ref0[8:], stride, bw, bh, 1<<30)
			w3 := fullPelSearchSAD(srcBlock, ref0[12:], stride, bw, bh, 1<<30)
			g0, g1, g2, g3 := fullPelSearchSAD4Step4(srcBlock, ref, base, stride, bw, bh, dx, dy)
			if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
				t.Fatalf("%dx%d dx=%d dy=%d: got (%d,%d,%d,%d) want (%d,%d,%d,%d)",
					bw, bh, dx, dy, g0, g1, g2, g3, w0, w1, w2, w3)
			}
		}
	}
}

func TestSAD64x64x4MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4566))
	const (
		stride = 160
		height = 192
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 500 {
		row := rng.Intn(56) + 64
		col := rng.Intn(stride-136) + 64
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0 := sadRectBlockReference(src, ref, off, ref0, stride, 64, 64)
		w1 := sadRectBlockReference(src, ref, off, ref1, stride, 64, 64)
		w2 := sadRectBlockReference(src, ref, off, ref2, stride, 64, 64)
		w3 := sadRectBlockReference(src, ref, off, ref3, stride, 64, 64)
		g0, g1, g2, g3 := sad64x64x4(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD32x32x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(46))
	const stride = 117
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*80) + rng.Intn(stride-51)
		w0, w1, w2, w3 := sad32x32x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad32x32x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

// TestSAD8x8DualImplMatchesPureGo proves the two-stride kernel bit-exact
// with the portable reference.
func TestSAD8x8DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const srcStride, refStride = 83, 19
	src := make([]byte, srcStride*64)
	ref := make([]byte, refStride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*48) + rng.Intn(srcStride-8)
		ro := rng.Intn(refStride*48) + rng.Intn(refStride-8)
		want := sad8x8DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad8x8DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

func TestSAD16x16DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(4716))
	const srcStride, refStride = 83, 29
	src := make([]byte, srcStride*96)
	ref := make([]byte, refStride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*64) + rng.Intn(srcStride-16)
		ro := rng.Intn(refStride*64) + rng.Intn(refStride-16)
		want := sad16x16DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad16x16DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

func TestSAD32x32DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(4732))
	const srcStride, refStride = 117, 41
	src := make([]byte, srcStride*128)
	ref := make([]byte, refStride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*80) + rng.Intn(srcStride-32)
		ro := rng.Intn(refStride*80) + rng.Intn(refStride-32)
		want := sad32x32DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad32x32DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

func TestSAD64x64DualMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(4764))
	const srcStride, refStride = 149, 83
	src := make([]byte, srcStride*160)
	ref := make([]byte, refStride*160)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*80) + rng.Intn(srcStride-64)
		ro := rng.Intn(refStride*80) + rng.Intn(refStride-64)
		want := sad64x64DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad64x64Dual(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

// TestSAD8x8CompoundAvgBlockImplMatchesPureGo proves the compound average SAD
// kernel is bit-exact with the portable rounded-average reference.
func TestSAD8x8CompoundAvgBlockImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(48))
	const srcStride, ref0Stride, ref1Stride = 83, 31, 47
	src := make([]byte, srcStride*64)
	ref0 := make([]byte, ref0Stride*64)
	ref1 := make([]byte, ref1Stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref0 {
		ref0[i] = uint8(rng.Intn(256))
	}
	for i := range ref1 {
		ref1[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*48) + rng.Intn(srcStride-8)
		r0o := rng.Intn(ref0Stride*48) + rng.Intn(ref0Stride-8)
		r1o := rng.Intn(ref1Stride*48) + rng.Intn(ref1Stride-8)
		want := sad8x8CompoundAvgBlockPureGo(src[so:], srcStride, ref0[r0o:], ref0Stride, ref1[r1o:], ref1Stride)
		got := sad8x8CompoundAvgBlockImpl(src[so:], srcStride, ref0[r0o:], ref0Stride, ref1[r1o:], ref1Stride)
		if got != want {
			t.Fatalf("so %d r0o %d r1o %d: impl %d want %d", so, r0o, r1o, got, want)
		}
	}
}

func TestSADRectBlockMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(49))
	const stride = 97
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for _, size := range [][2]int{{16, 8}, {8, 16}, {32, 16}, {16, 32}, {24, 8}} {
		bw, bh := size[0], size[1]
		for range 500 {
			base := rng.Intn(96-bh)*stride + rng.Intn(stride-bw)
			refBase := rng.Intn(96-bh)*stride + rng.Intn(stride-bw)
			want := sadRectBlockReference(src, ref, base, refBase, stride, bw, bh)
			got := sadRectBlock(src, ref, base, refBase, stride, bw, bh, 1<<30)
			if got != want {
				t.Fatalf("%dx%d base=%d refBase=%d: got %d want %d", bw, bh, base, refBase, got, want)
			}
		}
	}
}

func TestSADRectDualBlockMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4901))
	const srcStride, refStride = 97, 67
	src := make([]byte, srcStride*128)
	ref := make([]byte, refStride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, size := range [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 64}, {64, 32}, {32, 64}, {64, 16}, {16, 64}, {24, 8}} {
		bw, bh := size[0], size[1]
		for range 500 {
			so := rng.Intn(128-bh)*srcStride + rng.Intn(srcStride-bw)
			ro := rng.Intn(128-bh)*refStride + rng.Intn(refStride-bw)
			want := sadRectDualBlockReference(src[so:], srcStride, ref[ro:], refStride, bw, bh)
			got := sadRectDualBlock(src[so:], srcStride, ref[ro:], refStride, bw, bh)
			if got != want {
				t.Fatalf("%dx%d so=%d ro=%d: got %d want %d", bw, bh, so, ro, got, want)
			}
		}
	}
}

func sadRectBlockReference(src, ref []byte, base, refBase, stride, bw, bh int) int {
	total := 0
	for r := range bh {
		row := base + r*stride
		refRow := refBase + r*stride
		for c := range bw {
			d := int(src[row+c]) - int(ref[refRow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sadRectDualBlockReference(src []byte, srcStride int, ref []byte, refStride int, bw, bh int) int {
	total := 0
	for r := range bh {
		srow := r * srcStride
		rrow := r * refStride
		for c := range bw {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func TestFullPelDiamondSearchSeededMatchesLibaomReference(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const (
		width  = 96
		height = 80
		stride = 112
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for _, n := range []int{8, 12, 16, 32, 64} {
		for _, reach := range []int{4, 8} {
			for range 200 {
				px := rng.Intn((width-n)/8+1) * 8
				py := rng.Intn((height-n)/8+1) * 8
				seedDX := rng.Intn(41) - 20
				seedDY := rng.Intn(41) - 20
				wantDX, wantDY, wantSAD := fullPelDiamondSearchSeededReference(src, ref, stride, width, height, px, py, n, seedDX, seedDY, reach)
				gotDX, gotDY, gotSAD := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, n, seedDX, seedDY, reach)
				if gotDX != wantDX || gotDY != wantDY || gotSAD != wantSAD {
					t.Fatalf("n=%d reach=%d px=%d py=%d seed=(%d,%d): got (%d,%d,%d) want (%d,%d,%d)",
						n, reach, px, py, seedDX, seedDY, gotDX, gotDY, gotSAD, wantDX, wantDY, wantSAD)
				}
			}
		}
	}
}

func fullPelDiamondSearchSeededReference(src, ref []byte, stride, width, height, px, py, n int, seedDX, seedDY, reach int) (int, int, int) {
	sad := func(dx, dy, limit int) int {
		base := py*stride + px
		refBase := (py+dy)*stride + px + dx
		switch n {
		case 8:
			return sad8x8Impl(src[base:], ref[refBase:], stride, limit)
		case 16:
			return sad16x16Impl(src[base:], ref[refBase:], stride)
		case 32:
			return sad16x16Impl(src[base:], ref[refBase:], stride) +
				sad16x16Impl(src[base+16:], ref[refBase+16:], stride) +
				sad16x16Impl(src[base+16*stride:], ref[refBase+16*stride:], stride) +
				sad16x16Impl(src[base+16*stride+16:], ref[refBase+16*stride+16:], stride)
		}
		total := 0
		for r := range n {
			row := base + r*stride
			refRow := refBase + r*stride
			for c := range n {
				d := int(src[row+c]) - int(ref[refRow+c])
				if d < 0 {
					d = -d
				}
				total += d
			}
			if total >= limit {
				return total
			}
		}
		return total
	}
	seedDX = minInt(maxInt(seedDX, -px), width-n-px)
	seedDY = minInt(maxInt(seedDY, -py), height-n-py)

	mesh := func(startDX, startDY, searchRange, step int) (int, int, int) {
		if step < 1 {
			step = 1
		}
		startDX = minInt(maxInt(startDX, -px), width-n-px)
		startDY = minInt(maxInt(startDY, -py), height-n-py)
		startCol := maxInt(-searchRange, -px-startDX)
		endCol := minInt(searchRange, width-n-px-startDX)
		startRow := maxInt(-searchRange, -py-startDY)
		endRow := minInt(searchRange, height-n-py-startDY)
		bestDX, bestDY := startDX, startDY
		bestSAD := sad(bestDX, bestDY, 1<<30)
		colStep := step
		if step <= 1 {
			colStep = 4
		}
		for row := startRow; row <= endRow; row += step {
			for col := startCol; col <= endCol; col += colStep {
				dx := startDX + col
				dy := startDY + row
				if step <= 1 && col+3 <= endCol {
					for i := 0; i < 4; i++ {
						if s := sad(dx+i, dy, bestSAD); s < bestSAD {
							bestSAD, bestDX, bestDY = s, dx+i, dy
						}
					}
					continue
				}
				if s := sad(dx, dy, bestSAD); s < bestSAD {
					bestSAD, bestDX, bestDY = s, dx, dy
				}
			}
		}
		return bestDX, bestDY, bestSAD
	}
	bestDX, bestDY, _ := mesh(seedDX, seedDY, reach, 4)
	return mesh(bestDX, bestDY, 2, 2)
}

func BenchmarkSAD8x8(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8Impl(src, ref, 64, 1<<30)
	}
}

func BenchmarkSAD8x8x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad8x8x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD16x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad16x16Impl(src, ref, 64)
	}
}

func BenchmarkSAD16x16x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad32x32Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32x4(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*96 + 32
	b.ReportAllocs()
	for b.Loop() {
		sad32x32x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*96:], ref[off-2*96:], 96)
	}
}

func BenchmarkSAD32x32x4ScalarReference(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*96 + 32
	b.ReportAllocs()
	for b.Loop() {
		sad32x32x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*96:], ref[off-2*96:], 96)
	}
}

func BenchmarkSAD64x64(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad64x64(src, ref, 96)
	}
}

func BenchmarkSAD64x64Composed(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad64x64Composed(src, ref, 96)
	}
}

func BenchmarkSAD64x64x4Step4(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad64x64x4Step4(src, ref, 96)
	}
}

func BenchmarkSAD64x64x4(b *testing.B) {
	const stride = 160
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 64*stride + 64
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad64x64x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*stride:], ref[off-2*stride:], stride)
	}
}

func BenchmarkSAD16x16Dual(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 16*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad16x16Dual(src, 64, ref, 16)
	}
}

func BenchmarkSAD32x32Dual(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 32*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad32x32Dual(src, 64, ref, 32)
	}
}

func BenchmarkSAD64x64Dual(b *testing.B) {
	const srcStride, refStride = 96, 64
	src := make([]byte, srcStride*96)
	ref := make([]byte, refStride*96)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad64x64Dual(src, srcStride, ref, refStride)
	}
}

func BenchmarkSAD64x64DualComposed(b *testing.B) {
	const srcStride, refStride = 96, 64
	src := make([]byte, srcStride*96)
	ref := make([]byte, refStride*96)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad64x64DualComposed(src, srcStride, ref, refStride)
	}
}

func BenchmarkSADRectDual32x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 32*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sadRectDualBlock(src, 64, ref, 32, 32, 16)
	}
}

func BenchmarkSADRect32x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sadRectBlock(src, ref, 0, 0, 64, 32, 16, 1<<30)
	}
}

func BenchmarkSAD8x8CompoundAvgBlock(b *testing.B) {
	src := make([]byte, 64*64)
	ref0 := make([]byte, 64*64)
	ref1 := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref0[i] = uint8(i * 13)
		ref1[i] = uint8(i*17 + 3)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8CompoundAvgBlock(src, 64, ref0, 64, ref1, 64)
	}
}

func BenchmarkSAD8x8CompoundAvgBlockScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref0 := make([]byte, 64*64)
	ref1 := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref0[i] = uint8(i * 13)
		ref1[i] = uint8(i*17 + 3)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8CompoundAvgBlockPureGo(src, 64, ref0, 64, ref1, 64)
	}
}

func BenchmarkSAD8x8x4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad8x8x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD8x8x4ScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad8x8x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD16x16x4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad16x16x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD16x16x4ScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad16x16x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

var fullPelDiamondBenchSink int

func BenchmarkFullPelDiamondSearch8(b *testing.B) {
	const (
		width  = 96
		height = 80
		stride = 112
		px     = 32
		py     = 24
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 8, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch16(b *testing.B) {
	const (
		width  = 112
		height = 96
		stride = 128
		px     = 40
		py     = 32
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 16, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch32(b *testing.B) {
	const (
		width  = 128
		height = 112
		stride = 160
		px     = 48
		py     = 40
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 32, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch64(b *testing.B) {
	const (
		width  = 176
		height = 160
		stride = 192
		px     = 64
		py     = 48
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 64, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

// TestMemsetBytes checks the memset helpers at every length up to 4096.
func TestMemsetBytes(t *testing.T) {
	buf := make([]byte, 4096+32)
	for n := 0; n <= 4096; n++ {
		for _, f := range []func([]byte, byte){memsetBytes, memsetBytesScalar} {
			for i := range buf {
				buf[i] = 0xEE
			}
			v := byte(n*7 + 1)
			f(buf[3:3+n], v)
			for i, b := range buf {
				want := byte(0xEE)
				if i >= 3 && i < 3+n {
					want = v
				}
				if b != want {
					t.Fatalf("n=%d: byte %d = %#x, want %#x", n, i, b, want)
				}
			}
		}
	}
}

func boxTestPlane(w, h, stride int) []byte {
	p := make([]byte, stride*h)
	for y := 0; y < h; y++ {
		for x := 0; x < stride; x++ {
			p[y*stride+x] = byte((x*17 + y*31 + x*y*3 + 11) & 0xff)
		}
	}
	return p
}

// The layer steps of every scalability mode: exact 1/2 (luma and even
// chroma), odd-sized 1/2 (luma floor -> ScalePlaneBox, chroma ceil ->
// bilinear) and 2/3 (bilinear).
var boxTestSteps = []struct{ sw, sh, dw, dh int }{
	{64, 32, 32, 16},
	{38, 26, 19, 13},
	{135, 135, 67, 67},
	{135, 135, 68, 68},
	{270, 135, 135, 68},
	{1008, 576, 672, 384},
	{853, 480, 568, 320},
	{1, 9, 1, 4},
}

func TestScalePlaneBoxExactHalfIsPyramidDownsample(t *testing.T) {
	const sw, sh, pad = 70, 46, 9
	src := boxTestPlane(sw, sh, sw+pad)
	got := make([]byte, (sw/2)*(sh/2))
	want := make([]byte, len(got))
	scalePlaneBox8(got, sw/2, sw/2, sh/2, src, sw+pad, sw, sh, &boxScaleScratch{})
	svtDownsample2D(want, sw/2, src, sw+pad, sw/2, sh/2)
	if !bytes.Equal(got, want) {
		t.Fatal("exact 1/2 box step differs from svt_aom_downsample_2d")
	}
}

func TestScalePlaneBoxKeepsFlatPlanesWithinTruncation(t *testing.T) {
	var scratch boxScaleScratch
	for _, st := range boxTestSteps {
		src := make([]byte, st.sw*st.sh)
		src16 := make([]uint16, st.sw*st.sh)
		for i := range src {
			src[i] = 77
			src16[i] = 777
		}
		dst := make([]byte, st.dw*st.dh)
		dst16 := make([]uint16, st.dw*st.dh)
		scalePlaneBox8(dst, st.dw, st.dw, st.dh, src, st.sw, st.sw, st.sh, &scratch)
		scalePlaneBox16(dst16, st.dw, st.dw, st.dh, src16, st.sw, st.sw, st.sh, &scratch)
		for i := range dst {
			// ScaleAddCols truncates: a box of 3 rows loses up to one step.
			if dst[i] < 76 || dst[i] > 77 || dst16[i] < 776 || dst16[i] > 777 {
				t.Fatalf("%+v: sample %d = %d / %d", st, i, dst[i], dst16[i])
			}
		}
	}
}

// A box/bilinear downscale of a horizontal ramp stays monotone and inside
// the source range (catches mis-stepped or mis-weighted taps).
func TestScalePlaneBoxRampMonotone(t *testing.T) {
	var scratch boxScaleScratch
	for _, st := range boxTestSteps {
		src := make([]byte, st.sw*st.sh)
		for y := 0; y < st.sh; y++ {
			for x := 0; x < st.sw; x++ {
				src[y*st.sw+x] = byte(x * 255 / max(st.sw-1, 1))
			}
		}
		dst := make([]byte, st.dw*st.dh)
		scalePlaneBox8(dst, st.dw, st.dw, st.dh, src, st.sw, st.sw, st.sh, &scratch)
		for y := 0; y < st.dh; y++ {
			for x := 1; x < st.dw; x++ {
				if dst[y*st.dw+x] < dst[y*st.dw+x-1] {
					t.Fatalf("%+v: row %d not monotone at %d", st, y, x)
				}
			}
		}
	}
}

// TestWebRTCPyramidHintMatchesFilteredPyramid checks that the S3 stream's
// lower-layer luma is the top layer's HME pyramid, that buildPyramid takes
// it, and that a hint for another picture is ignored.
func TestWebRTCPyramidHintMatchesFilteredPyramid(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1008, Height: 576},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS3T3,
	}
	s, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	const w, h = 1008, 576
	src := SourceFrame420{Y: boxTestPlane(w, h, w), U: boxTestPlane(w/2, h/2, w/2), V: boxTestPlane(w/2, h/2, w/2), YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	s.pictureGen++
	if _, err := s.sourceForLayer(src, 0, s.config.SpatialLayers[0].Resolution); err != nil {
		t.Fatal(err)
	}
	hint := s.webRTCPyramidHint(src, 2)
	if hint.half.Width != w/2 || hint.six.Width != w/4 {
		t.Fatalf("hint half %d six %d", hint.half.Width, hint.six.Width)
	}
	var want, got svtPyramid
	want.build(src)
	got.setHint(hint)
	got.build(src)
	if !bytes.Equal(got.curHalf.pix, want.curHalf.pix) || !bytes.Equal(got.curSix.pix, want.curSix.pix) {
		t.Fatal("hinted pyramid differs from the filtered one")
	}
	// The hint is taken: a planted sample shows up in the half plane.
	hint.half.Y[0] ^= 0xff
	got.setHint(hint)
	got.build(src)
	if got.curHalf.pix[got.curHalf.org] == want.curHalf.pix[want.curHalf.org] {
		t.Fatal("buildPyramid ignored a matching hint")
	}
	// A hint made from another picture is not.
	other := src
	other.Y = append([]byte(nil), src.Y...)
	got.setHint(hint)
	got.build(other)
	if !bytes.Equal(got.curHalf.pix, want.curHalf.pix) {
		t.Fatal("buildPyramid took another picture's hint")
	}
	hint.half.Y[0] ^= 0xff
	// The next picture invalidates the cached layers.
	s.pictureGen++
	if hint := s.webRTCPyramidHint(src, 2); hint.parent != nil {
		t.Fatal("stale layers produced a hint")
	}
}

func seedTestFrame(w, h, n int) SourceFrame420 {
	f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4), YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			f.Y[y*w+x] = byte(((x+3*n)*7 + (y+n)*13 + ((x+3*n)>>3)*((y+n)>>3)*5) & 0xff)
		}
	}
	for i := range f.U {
		f.U[i], f.V[i] = 128, 128
	}
	return f
}

// TestWebRTCMESeedFiresOnSimulcastInterFrames checks the cross-layer ME
// seed: it centers the upper layer's HME on simulcast inter pictures, never
// on the keyframe (the lower layer ran no ME), never for a lower layer not
// coded in the picture, and not in an SVC mode.
func TestWebRTCMESeedFiresOnSimulcastInterFrames(t *testing.T) {
	for _, tc := range []struct {
		mode ScalabilityMode
		want bool
	}{
		{ScalabilityModeS2T3, true},
		{ScalabilityModeL2T3_KEY, false},
	} {
		cfg := Config{
			Resolution:        Resolution{Width: 640, Height: 360},
			MaxFramerate:      Rational{Num: 30, Den: 1},
			MinBitrateKbps:    100,
			MaxBitrateKbps:    900,
			TargetBitrateKbps: 500,
			Scalability:       tc.mode,
		}
		s, err := NewWebRTCStreamConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		for n := 0; n < 4; n++ {
			if _, err := s.EncodePicture(seedTestFrame(640, 360, n), false); err != nil {
				t.Fatalf("%s picture %d: %v", tc.mode, n, err)
			}
			got := s.encoders[1].hme.seedOn
			if want := tc.want && n > 0; got != want {
				t.Fatalf("%s picture %d: seedOn = %v, want %v", tc.mode, n, got, want)
			}
			if s.encoders[1].hme.seed != nil {
				t.Fatalf("%s picture %d: seed left set after the encode", tc.mode, n)
			}
		}
		s.pictureGen++
		if s.webRTCMESeed(1) != nil {
			t.Fatalf("%s: a lower layer not coded in the picture seeded", tc.mode)
		}
		s.Close()
	}
}

func TestBoxInterpolateRow8MatchesScalar(t *testing.T) {
	const n = 71
	s0, s1 := boxTestPlane(n, 1, n), boxTestPlane(n, 2, n)[n:]
	for _, f := range []int{1, 64, 127, 128, 129, 192, 255} {
		got, want := make([]byte, n), make([]byte, n)
		boxInterpolateRow8(got, s0, s1, f)
		boxInterpolateRow(want, s0, s1, f, 128)
		if !bytes.Equal(got, want) {
			t.Fatalf("f=%d: SIMD row differs", f)
		}
	}
}

// The 2/3 column fast path computes ScaleFilterCols' BLENDER values.
func TestBoxFilterColsTwoThirds(t *testing.T) {
	for _, dw := range []int{8, 9, 672} {
		sw := dw * 3 / 2
		s := boxTestPlane(sw, 1, sw)
		got := make([]byte, dw)
		boxFilterCols(got, s, 0x4000, 0x18000)
		for i, x := 0, 0x4000; i < dw; i, x = i+1, x+0x18000 {
			a, b := int(s[x>>16]), int(s[min(x>>16+1, sw-1)])
			if want := byte(a + (((x&0xffff)*(b-a) + 0x8000) >> 16)); got[i] != want {
				t.Fatalf("dw=%d i=%d: %d want %d", dw, i, got[i], want)
			}
		}
	}
}

func BenchmarkBoxFilterCols8(b *testing.B) {
	for _, tc := range []struct {
		name   string
		sw, dw int
		x, dx  int
	}{
		{"2/3 672", 1008, 672, 0x4000, 0x18000},
		{"3/2 640->427", 640, 427, 0, 0},
		{"853->568", 853, 568, 0, 0},
	} {
		x, dx := tc.x, tc.dx
		if dx == 0 {
			dx = boxFixedDiv(tc.sw, tc.dw)
			x = dx>>1 - 32768
		}
		s := boxTestPlane(tc.sw, 1, tc.sw)
		d := make([]byte, tc.dw)
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(tc.dw))
			for b.Loop() {
				boxFilterCols(d, s, x, dx)
			}
		})
	}
}

func TestAppendSegmentationParamsPayloadDisabled(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	payload, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, SegmentationParams{}, nil)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.Enabled || parsed.UpdateMap || parsed.UpdateData || parsed.AllLossless {
		t.Fatalf("parsed segmentation=%+v", parsed)
	}
}

func TestAppendSegmentationParamsPayloadDisabledInactiveSentinel(t *testing.T) {
	// A caller bridging decoder SegmentationData carries the SEG_LVL_REF_FRAME
	// disabled sentinel (RefFrame == -1) on inactive segments. With segmentation
	// disabled the feature data is never written, so this must encode to the same
	// single 'false' bit as a zero-value SegmentationParams rather than being
	// rejected as active data.
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	seg := SegmentationParams{Data: inactiveEncoderSegmentationData()}
	payload, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, seg, nil)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.Enabled {
		t.Fatalf("parsed segmentation=%+v", parsed)
	}
}

func TestAppendSegmentationParamsPayloadPrimaryRefNoneUpdateData(t *testing.T) {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaQ = -8
	data.Segments[1].DeltaLFYV = 3
	data.Segments[2].RefFrame = 4
	data.Segments[3].Skip = true
	data.Segments[4].GlobalMV = true
	seg := SegmentationParams{
		Enabled:    true,
		UpdateMap:  true,
		UpdateData: true,
		Data:       data,
	}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	_, parsed := appendAndParseSegmentationParams(t, prefix, parser.FrameHeaderPrefix{PrimaryRefFrame: parser.PrimaryRefNone}, seg, nil)
	if !parsed.Enabled || !parsed.UpdateMap || !parsed.UpdateData {
		t.Fatalf("parsed segmentation flags=%+v", parsed)
	}
	if parsed.Data.Segments[0].DeltaQ != -8 || parsed.QIndex[0] != 12 {
		t.Fatalf("parsed segment0=%+v q=%d", parsed.Data.Segments[0], parsed.QIndex[0])
	}
	if parsed.Data.Segments[1].DeltaLFYV != 3 || parsed.Data.Segments[2].RefFrame != 4 ||
		!parsed.Data.Segments[3].Skip || !parsed.Data.Segments[4].GlobalMV {
		t.Fatalf("parsed data=%+v", parsed.Data)
	}
	if !parsed.Data.Preskip || parsed.Data.LastActiveID != 4 {
		t.Fatalf("parsed derived data=%+v", parsed.Data)
	}
}

func TestAppendSegmentationParamsPayloadCopiesPreviousData(t *testing.T) {
	previous := inactiveParserSegmentationData()
	previous.Segments[2].DeltaQ = 9
	previous.LastActiveID = 2
	seg := SegmentationParams{
		Enabled:    true,
		UpdateMap:  false,
		UpdateData: false,
		Data:       inactiveEncoderSegmentationData(),
	}
	prefix := FrameHeaderPrefix{PrimaryRefFrame: 0}
	parsedPrefix := parser.FrameHeaderPrefix{PrimaryRefFrame: 0}
	_, parsed := appendAndParseSegmentationParams(t, prefix, parsedPrefix, seg, &previous)
	if !parsed.Enabled || parsed.UpdateMap || parsed.UpdateData {
		t.Fatalf("parsed flags=%+v", parsed)
	}
	if parsed.Data.Segments[2].DeltaQ != 9 || parsed.Data.LastActiveID != 2 {
		t.Fatalf("parsed previous data=%+v", parsed.Data)
	}
}

func TestAppendSegmentationParamsPayloadRejectsInvalid(t *testing.T) {
	prefix := FrameHeaderPrefix{PrimaryRefFrame: EncoderPrimaryRefNone}
	active := inactiveEncoderSegmentationData()
	active.Segments[0].RefFrame = 8
	cases := [...]SegmentationParams{
		{UpdateMap: true},
		{Enabled: true, UpdateMap: false, UpdateData: true},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: active},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: segmentationDataWithDeltaQ(256)},
		{Enabled: true, UpdateMap: true, UpdateData: true, Data: segmentationDataWithDeltaLF(64)},
	}
	var buf [64]byte
	for _, seg := range cases {
		if _, err := appendSegmentationParamsPayload(buf[:0], prefix, seg); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendSegmentationParamsPayload(%+v) err=%v want ErrInvalidFrame", seg, err)
		}
	}
	nonNone := FrameHeaderPrefix{PrimaryRefFrame: 0}
	if _, err := appendSegmentationParamsPayload(buf[:0], nonNone, SegmentationParams{Enabled: true, TemporalUpdate: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("temporal without update map err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseSegmentationParams(t *testing.T, prefix FrameHeaderPrefix, parsedPrefix parser.FrameHeaderPrefix, seg SegmentationParams, previous *parser.SegmentationData) ([]byte, parser.SegmentationParams) {
	t.Helper()
	var buf [96]byte
	out, err := appendSegmentationParamsPayload(buf[:0], prefix, seg)
	if err != nil {
		t.Fatalf("appendSegmentationParamsPayload: %v", err)
	}
	parsed, err := parser.ParseSegmentationParams(out, parsedPrefix, parser.QuantizationParams{BaseQIdx: 20}, previous)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	return out, parsed
}

func inactiveEncoderSegmentationData() SegmentationData {
	var data SegmentationData
	for i := uint8(0); i < MaxSegments; i++ {
		data.Segments[i].RefFrame = -1
	}
	return data
}

func inactiveParserSegmentationData() parser.SegmentationData {
	var data parser.SegmentationData
	data.LastActiveID = -1
	for i := uint8(0); i < parser.MaxSegments; i++ {
		data.Segments[i].RefFrame = -1
	}
	return data
}

func segmentationDataWithDeltaQ(delta int16) SegmentationData {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaQ = delta
	return data
}

func segmentationDataWithDeltaLF(delta int16) SegmentationData {
	data := inactiveEncoderSegmentationData()
	data.Segments[0].DeltaLFYV = delta
	return data
}

// TestSequenceHeaderForConfigSRGBProfile keeps the encoder from writing the
// sRGB colour description (BT.709 / sRGB / identity matrix) where dav1d and
// libaom reject it: the triple implies 4:4:4, which only profile 1 and
// 12-bit profile 2 carry. Every accepted header must parse back.
func TestSequenceHeaderForConfigSRGBProfile(t *testing.T) {
	srgb := func(bitDepth uint8) SequenceColorConfig {
		return SequenceColorConfig{
			BitDepth:                bitDepth,
			ColorDescriptionPresent: true,
			ColorPrimaries:          SequenceColorPrimariesBT709,
			TransferCharacteristics: SequenceTransferCharacteristicsSRGB,
			MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
			ColorRange:              true,
		}
	}
	mono := srgb(8)
	mono.MonoChrome = true
	identity420 := SequenceColorConfig{
		BitDepth:                8,
		ColorDescriptionPresent: true,
		ColorPrimaries:          SequenceColorPrimariesBT709,
		TransferCharacteristics: 1,
		MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
		SubsamplingX:            true,
		SubsamplingY:            true,
	}
	for _, tc := range []struct {
		name    string
		profile Profile
		color   SequenceColorConfig
		ok      bool
	}{
		{"profile0-8bit", Profile0, srgb(8), false},
		{"profile0-10bit", Profile0, srgb(10), false},
		{"profile2-8bit", Profile2, srgb(8), false},
		{"profile2-10bit", Profile2, srgb(10), false},
		{"profile1-8bit", Profile1, srgb(8), true},
		{"profile1-10bit", Profile1, srgb(10), true},
		{"profile2-12bit", Profile2, srgb(12), true},
		{"profile0-mono", Profile0, mono, true},
		{"profile0-identity-420", Profile0, identity420, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seq, err := SequenceHeaderForConfig(Config{
				Resolution:     Resolution{Width: 64, Height: 64},
				Profile:        tc.profile,
				BitDepth:       tc.color.BitDepth,
				ColorConfigSet: true,
				ColorConfig:    tc.color,
			})
			if !tc.ok {
				if !errors.Is(err, ErrInvalidConfig) {
					t.Fatalf("SequenceHeaderForConfig err=%v want %v", err, ErrInvalidConfig)
				}
				return
			}
			if err != nil {
				t.Fatalf("SequenceHeaderForConfig: %v", err)
			}
			var buf [160]byte
			out, err := appendSequenceHeaderPayload(buf[:0], seq)
			if err != nil {
				t.Fatalf("appendSequenceHeaderPayload: %v", err)
			}
			if _, err := parser.ParseSequenceHeader(out); err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
		})
	}
}

type sequenceTestBitWriter struct {
	buf [128]byte
	bit uint16
}

func (w *sequenceTestBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *sequenceTestBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *sequenceTestBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func TestBitWriterMatchesMSBFirstShape(t *testing.T) {
	var buf [4]byte
	w := newBitWriter(buf[:])
	for _, op := range []struct {
		value uint64
		n     uint8
	}{
		{0b101, 3},
		{0, 1},
		{0b1110_0011, 8},
	} {
		if err := w.writeBits(op.value, op.n); err != nil {
			t.Fatalf("writeBits: %v", err)
		}
	}
	if err := w.writeTrailingBits(); err != nil {
		t.Fatalf("writeTrailingBits: %v", err)
	}
	want := []byte{0xae, 0x38}
	if got := buf[:w.bytesWritten()]; !bytes.Equal(got, want) {
		t.Fatalf("bit writer = %08b want %08b", got, want)
	}
	if w.bytesWritten() != 2 || !w.byteAligned() {
		t.Fatalf("writer position bytes=%d aligned=%v", w.bytesWritten(), w.byteAligned())
	}
}

func TestAppendSequenceHeaderPayloadRealtime(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	want := realtimeEncoderSequenceHeaderPayload()

	size, err := SequenceHeaderPayloadSize(seq)
	if err != nil {
		t.Fatalf("SequenceHeaderPayloadSize: %v", err)
	}
	if size != len(want) {
		t.Fatalf("payload size=%d want %d", size, len(want))
	}
	var buf [64]byte
	out, err := appendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatalf("payload = % x; want % x", out, want)
	}

	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsed.SeqProfile != 1 || parsed.OperatingPointsCount != 1 || parsed.OperatingPoints[0].SeqLevelIdx != 5 {
		t.Fatalf("parsed operating point/profile: %+v", parsed)
	}
	if parsed.MaxFrameWidth != 16 || parsed.MaxFrameHeight != 9 {
		t.Fatalf("parsed dimensions=%dx%d", parsed.MaxFrameWidth, parsed.MaxFrameHeight)
	}
	if !parsed.EnableOrderHint || parsed.OrderHintBits != 7 || !parsed.EnableSuperRes || !parsed.EnableCDEF || !parsed.EnableRestoration {
		t.Fatalf("parsed tools: %+v", parsed)
	}
	if !parsed.ColorConfig.ColorRange || parsed.ColorConfig.SubsamplingX || parsed.ColorConfig.SubsamplingY {
		t.Fatalf("parsed color config: %+v", parsed.ColorConfig)
	}
	if !parsed.FilmGrainParamsPresent {
		t.Fatal("film_grain_params_present=false")
	}
}

func TestAppendLowOverheadSequenceHeaderOBU(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	size, err := LowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		t.Fatalf("LowOverheadSequenceHeaderOBUSize: %v", err)
	}
	var buf [80]byte
	out, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendLowOverheadSequenceHeaderOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("obu size=%d want %d", len(out), size)
	}

	unit, consumed, err := obu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverhead: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != obu.TypeSequenceHeader || !unit.Header.HasSizeField {
		t.Fatalf("parsed obu header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := parser.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}

func TestAppendSequenceHeaderRejectsInvalid(t *testing.T) {
	base := realtimeEncoderSequenceHeader()
	tests := []struct {
		name string
		mut  func(*SequenceHeader)
		want error
	}{
		{name: "dimension", mut: func(seq *SequenceHeader) { seq.MaxFrameWidth = 0 }, want: ErrInvalidConfig},
		{name: "decoder model without timing", mut: func(seq *SequenceHeader) { seq.DecoderModelInfoPresent = true }, want: ErrInvalidConfig},
		{name: "decoder model bad length", mut: func(seq *SequenceHeader) {
			seq.TimingInfoPresent = true
			seq.DecoderModelInfoPresent = true
			seq.DecoderModelInfo = SequenceDecoderModelInfo{
				BufferDelayLength:           0,
				BufferRemovalTimeLength:     4,
				FramePresentationTimeLength: 4,
			}
		}, want: ErrInvalidConfig},
		{name: "operating point display delay without header flag", mut: func(seq *SequenceHeader) {
			seq.OperatingPoints[0].InitialDisplayDelayPresent = true
		}, want: ErrInvalidConfig},
		{name: "level", mut: func(seq *SequenceHeader) { seq.OperatingPoints[0].SeqLevelIdx = 10 }, want: ErrInvalidConfig},
		{name: "profile-bitdepth", mut: func(seq *SequenceHeader) {
			seq.Profile = Profile0
			seq.ColorConfig.BitDepth = 12
		}, want: ErrInvalidConfig},
		{name: "order-hint", mut: func(seq *SequenceHeader) {
			seq.EnableOrderHint = false
			seq.EnableJNTComp = true
		}, want: ErrInvalidConfig},
		{name: "color", mut: func(seq *SequenceHeader) { seq.ColorConfig.SubsamplingX = true }, want: ErrInvalidConfig},
	}
	var buf [80]byte
	for _, tt := range tests {
		seq := base
		tt.mut(&seq)
		if _, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq); !errors.Is(err, tt.want) {
			t.Fatalf("%s err=%v want %v", tt.name, err, tt.want)
		}
	}
}

func TestAppendSequenceHeaderTimingAndDisplayDelay(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	seq.TimingInfoPresent = true
	seq.TimingInfo = SequenceTimingInfo{
		NumUnitsInDisplayTick:    1001,
		TimeScale:                30000,
		EqualPictureInterval:     false,
		NumTicksPerPictureMinus1: 0,
	}
	seq.DecoderModelInfoPresent = true
	seq.DecoderModelInfo = SequenceDecoderModelInfo{
		BufferDelayLength:           8,
		NumUnitsInDecodingTick:      1001,
		BufferRemovalTimeLength:     5,
		FramePresentationTimeLength: 6,
	}
	seq.InitialDisplayDelayPresent = true
	seq.OperatingPoints[0].DecoderModelPresent = true
	seq.OperatingPoints[0].DecoderBufferDelay = 17
	seq.OperatingPoints[0].EncoderBufferDelay = 19
	seq.OperatingPoints[0].LowDelayMode = true
	seq.OperatingPoints[0].InitialDisplayDelayPresent = true
	seq.OperatingPoints[0].InitialDisplayDelayMinus1 = 3

	var buf [128]byte
	out, err := appendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if !parsed.TimingInfoPresent || !parsed.DecoderModelInfoPresent || !parsed.InitialDisplayDelayPresent {
		t.Fatalf("parsed timing flags: %+v", parsed)
	}
	if parsed.TimingInfo.NumUnitsInDisplayTick != 1001 || parsed.TimingInfo.TimeScale != 30000 ||
		parsed.TimingInfo.EqualPictureInterval {
		t.Fatalf("parsed timing info: %+v", parsed.TimingInfo)
	}
	if parsed.DecoderModelInfo.BufferDelayLength != 8 || parsed.DecoderModelInfo.NumUnitsInDecodingTick != 1001 ||
		parsed.DecoderModelInfo.BufferRemovalTimeLength != 5 || parsed.DecoderModelInfo.FramePresentationTimeLength != 6 {
		t.Fatalf("parsed decoder model info: %+v", parsed.DecoderModelInfo)
	}
	op := parsed.OperatingPoints[0]
	if !op.DecoderModelPresent || op.DecoderBufferDelay != 17 || op.EncoderBufferDelay != 19 || !op.LowDelayMode ||
		!op.InitialDisplayDelayPresent || op.InitialDisplayDelayMinus1 != 3 {
		t.Fatalf("parsed operating point: %+v", op)
	}
}

func TestAppendSequenceHeaderAllocs(t *testing.T) {
	seq := realtimeEncoderSequenceHeader()
	var buf [80]byte
	if _, err := AppendLowOverheadSequenceHeaderOBU(buf[:0], seq); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	})
	if allocs != 0 {
		t.Fatalf("AppendLowOverheadSequenceHeaderOBU allocated: %f", allocs)
	}
}

func TestSequenceHeaderForConfigSingleLayer(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.Profile != Profile0 || seq.MaxFrameWidth != 640 || seq.MaxFrameHeight != 360 {
		t.Fatalf("sequence basics: %+v", seq)
	}
	if seq.OperatingPointsCount != 1 || seq.OperatingPoints[0].IDC != 0 || seq.OperatingPoints[0].SeqLevelIdx != SequenceLevelMax {
		t.Fatalf("single-layer operating point: count=%d op=%+v", seq.OperatingPointsCount, seq.OperatingPoints[0])
	}
	if !seq.EnableOrderHint || seq.OrderHintBits != webRTCDefaultOrderHintBits || !seq.EnableCDEF {
		t.Fatalf("realtime tools: %+v", seq)
	}

	var buf [128]byte
	out, err := appendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsed.MaxFrameWidth != 640 || parsed.MaxFrameHeight != 360 || parsed.OperatingPoints[0].SeqLevelIdx != parser.SeqLevelMax {
		t.Fatalf("parsed sequence: %+v", parsed)
	}
	if parsed.ColorConfig.BitDepth != 8 || !parsed.ColorConfig.SubsamplingX || !parsed.ColorConfig.SubsamplingY {
		t.Fatalf("parsed color config: %+v", parsed.ColorConfig)
	}
}

func TestSequenceHeaderForConfigSVCOperatingPoints(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 4 {
		t.Fatalf("operating point count=%d want 4", seq.OperatingPointsCount)
	}
	wantIDC := [...]uint16{0x303, 0x301, 0x103, 0x101}
	for i := range wantIDC {
		if seq.OperatingPoints[i].IDC != wantIDC[i] || seq.OperatingPoints[i].SeqLevelIdx != SequenceLevelMax {
			t.Fatalf("op[%d]=%+v want idc=%#x level=%d", i, seq.OperatingPoints[i], wantIDC[i], SequenceLevelMax)
		}
	}

	var buf [160]byte
	out, err := appendSequenceHeaderPayload(buf[:0], seq)
	if err != nil {
		t.Fatalf("appendSequenceHeaderPayload: %v", err)
	}
	parsed, err := parser.ParseSequenceHeader(out)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	for i := range wantIDC {
		if parsed.OperatingPoints[i].IDC != wantIDC[i] {
			t.Fatalf("parsed op[%d]=%+v want idc=%#x", i, parsed.OperatingPoints[i], wantIDC[i])
		}
	}
}

func TestSequenceHeaderForConfigRequestedLayers(t *testing.T) {
	seq, err := SequenceHeaderForConfig(Config{
		Resolution:         Resolution{Width: 640, Height: 360},
		SpatialLayerCount:  2,
		TemporalLayerCount: 3,
	})
	if err != nil {
		t.Fatalf("SequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 6 || seq.OperatingPoints[0].IDC != 0x307 || seq.OperatingPoints[5].IDC != 0x101 {
		t.Fatalf("requested-layer operating points: count=%d first=%#x last=%#x",
			seq.OperatingPointsCount, seq.OperatingPoints[0].IDC, seq.OperatingPoints[5].IDC)
	}
}

func TestSequenceHeaderForConfigExplicitColorConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want SequenceColorConfig
	}{
		{
			name: "profile2-420-12bit",
			cfg: Config{
				Resolution:     Resolution{Width: 640, Height: 360},
				Profile:        Profile2,
				BitDepth:       12,
				ColorConfigSet: true,
				ColorConfig: SequenceColorConfig{
					BitDepth:             12,
					SubsamplingX:         true,
					SubsamplingY:         true,
					ChromaSamplePosition: 1,
				},
			},
			want: SequenceColorConfig{
				BitDepth:             12,
				SubsamplingX:         true,
				SubsamplingY:         true,
				ChromaSamplePosition: 1,
			},
		},
		{
			name: "profile2-444-12bit-color-bitdepth-only",
			cfg: Config{
				Resolution:     Resolution{Width: 640, Height: 360},
				Profile:        Profile2,
				ColorConfigSet: true,
				ColorConfig:    SequenceColorConfig{BitDepth: 12},
			},
			want: SequenceColorConfig{BitDepth: 12},
		},
		{
			name: "profile0-mono-10bit",
			cfg: Config{
				Resolution:     Resolution{Width: 64, Height: 64},
				Profile:        Profile0,
				BitDepth:       10,
				ColorConfigSet: true,
				ColorConfig: SequenceColorConfig{
					BitDepth:   10,
					MonoChrome: true,
				},
			},
			want: SequenceColorConfig{
				BitDepth:     10,
				MonoChrome:   true,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			seq, err := SequenceHeaderForConfig(tc.cfg)
			if err != nil {
				t.Fatalf("SequenceHeaderForConfig: %v", err)
			}
			if seq.ColorConfig != tc.want {
				t.Fatalf("sequence color=%+v want %+v", seq.ColorConfig, tc.want)
			}
			var buf [160]byte
			out, err := appendSequenceHeaderPayload(buf[:0], seq)
			if err != nil {
				t.Fatalf("appendSequenceHeaderPayload: %v", err)
			}
			parsed, err := parser.ParseSequenceHeader(out)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			got := parsed.ColorConfig
			if got.BitDepth != tc.want.BitDepth ||
				got.MonoChrome != tc.want.MonoChrome ||
				got.SubsamplingX != tc.want.SubsamplingX ||
				got.SubsamplingY != tc.want.SubsamplingY ||
				got.ChromaSamplePosition != tc.want.ChromaSamplePosition {
				t.Fatalf("parsed color=%+v want %+v", got, tc.want)
			}
		})
	}
}

func TestSequenceHeaderForConfigRejectsInvalidSequenceCombination(t *testing.T) {
	_, err := SequenceHeaderForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
		Profile:    Profile0,
		BitDepth:   12,
	})
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("SequenceHeaderForConfig err=%v want %v", err, ErrInvalidConfig)
	}
}

func TestSequenceHeaderForConfigAllocs(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	if _, err := SequenceHeaderForConfig(cfg); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = SequenceHeaderForConfig(cfg)
	})
	if allocs != 0 {
		t.Fatalf("SequenceHeaderForConfig allocated: %f", allocs)
	}
}

func realtimeEncoderSequenceHeader() SequenceHeader {
	var seq SequenceHeader
	seq.Profile = Profile1 // sRGB is 4:4:4
	seq.OperatingPointsCount = 1
	seq.OperatingPoints[0].SeqLevelIdx = 5
	seq.MaxFrameWidth = 16
	seq.MaxFrameHeight = 9
	seq.EnableFilterIntra = true
	seq.EnableIntraEdgeFilter = true
	seq.EnableInterIntraCompound = true
	seq.EnableMaskedCompound = true
	seq.EnableDualFilter = true
	seq.EnableOrderHint = true
	seq.EnableJNTComp = true
	seq.EnableRefFrameMVS = true
	seq.OrderHintBits = 7
	seq.EnableSuperRes = true
	seq.EnableCDEF = true
	seq.EnableRestoration = true
	seq.ColorConfig = SequenceColorConfig{
		BitDepth:                8,
		ColorDescriptionPresent: true,
		ColorPrimaries:          SequenceColorPrimariesBT709,
		TransferCharacteristics: SequenceTransferCharacteristicsSRGB,
		MatrixCoefficients:      SequenceMatrixCoefficientsIdentity,
		ColorRange:              true,
	}
	seq.FilmGrainParamsPresent = true
	return seq
}

// realtimeEncoderMonochromeSequenceHeader is realtimeEncoderSequenceHeader
// with monochrome colour, which profile 1 cannot carry.
func realtimeEncoderMonochromeSequenceHeader() SequenceHeader {
	seq := realtimeEncoderSequenceHeader()
	seq.Profile = Profile0
	seq.ColorConfig.MonoChrome = true
	return seq
}

func realtimeEncoderSequenceHeaderPayload() []byte {
	var w sequenceTestBitWriter
	w.writeBits(1, 3) // seq_profile
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, 5)
	w.writeBits(0, 12)
	w.writeBits(5, 5)
	w.writeBits(3, 4)
	w.writeBits(3, 4)
	w.writeBits(15, 4)
	w.writeBits(8, 4)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(true)
	w.writeBool(false)
	w.writeBits(0, 1)
	w.writeBits(6, 3)
	w.writeBool(true)  // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(true)  // enable_restoration
	w.writeBool(false) // high_bitdepth; profile 1 codes no mono_chrome
	w.writeBool(true)  // color_description_present_flag
	w.writeBits(SequenceColorPrimariesBT709, 8)
	w.writeBits(SequenceTransferCharacteristicsSRGB, 8)
	w.writeBits(SequenceMatrixCoefficientsIdentity, 8)
	w.writeBool(false)
	w.writeBool(true)
	return w.trailingBits()
}

func TestAppendSkipModeParamsPayloadDisabledForSingleReference(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 8}
	params := SkipModeParams{}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, InterFrameSize{}, nil, TransformReferenceParams{ReferenceMode: ReferenceModeSingle}, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.Allowed || parsed.Enabled || parsed.BitsRead != 0 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadBeforeAfterRefs(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	params := SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.Allowed || !parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} || parsed.BitsRead != 1 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadSecondBeforeRef(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{12, 14, 11, 10, 9, 8, 7})
	params := SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}
	_, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if !parsed.Allowed || parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} || parsed.BitsRead != 1 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadNoLegalPair(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{16, 16, 16, 16, 16, 16, 16})
	params := SkipModeParams{}
	payload, parsed := appendAndParseSkipModeParams(t, seq, prefix, size, &refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.Allowed || parsed.Enabled || parsed.BitsRead != 0 {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestAppendSkipModeParamsPayloadRejectsInvalid(t *testing.T) {
	seq := skipModeSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter, OrderHint: 16}
	size, refs := encoderSkipModeRefs(seq, [7]uint8{15, 17, 14, 18, 13, 19, 12})
	cases := [...]struct {
		size   InterFrameSize
		refs   *parser.ReferenceState
		params SkipModeParams
	}{
		{size: size, refs: nil, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}},
		{size: size, refs: &refs, params: SkipModeParams{Enabled: true}},
		{size: size, refs: &refs, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{1, 2}}},
		{size: invalidSkipModeRefSize(size), refs: &refs, params: SkipModeParams{Allowed: true, RefFrameIdx: [2]uint8{0, 1}}},
	}
	var buf [1]byte
	for _, tc := range cases {
		if _, err := appendSkipModeParamsPayload(buf[:0], seq, prefix, tc.size, tc.refs, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendSkipModeParamsPayload(%+v) err=%v want ErrInvalidFrame", tc.params, err)
		}
	}

	intraPrefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ErrorResilientMode: true}
	if _, err := appendSkipModeParamsPayload(buf[:0], seq, intraPrefix, InterFrameSize{}, nil, TransformReferenceParams{ReferenceMode: ReferenceModeSelect}, SkipModeParams{Allowed: true}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("intra skip err=%v want ErrInvalidFrame", err)
	}
}

func appendAndParseSkipModeParams(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) ([]byte, parser.SkipModeParams) {
	t.Helper()
	var buf [1]byte
	payload, err := appendSkipModeParamsPayload(buf[:0], seq, prefix, size, refs, transformRef, params)
	if err != nil {
		t.Fatalf("appendSkipModeParamsPayload: %v", err)
	}
	parsed, err := parser.ParseSkipModeParams(
		payload,
		parser.SequenceHeader{EnableOrderHint: seq.EnableOrderHint, OrderHintBits: seq.OrderHintBits},
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType), OrderHint: prefix.OrderHint},
		parser.FrameSize{RefFrameIdx: size.RefFrameIdx},
		refs,
		parser.TransformReferenceParams{ReferenceMode: parser.ReferenceMode(transformRef.ReferenceMode)},
	)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	return payload, parsed
}

func skipModeSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: true,
		EnableOrderHint:      true,
		OrderHintBits:        5,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func encoderSkipModeRefs(seq SequenceHeader, orderHints [7]uint8) (InterFrameSize, parser.ReferenceState) {
	var size InterFrameSize
	var refs parser.ReferenceState
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:     true,
			OrderHint: orderHints[i],
			Size: parser.FrameSize{
				CodedWidth:          seq.MaxFrameWidth,
				UpscaledWidth:       seq.MaxFrameWidth,
				Height:              seq.MaxFrameHeight,
				RenderWidth:         seq.MaxFrameWidth,
				RenderHeight:        seq.MaxFrameHeight,
				SuperResDenominator: 8,
			},
		}
	}
	return size, refs
}

func invalidSkipModeRefSize(size InterFrameSize) InterFrameSize {
	size.RefFrameIdx[0] = parser.RefFrames
	return size
}

// svtBilinearReference is variance.c SUBPIX_VAR's two passes, transliterated.
func svtBilinearReference(ref []byte, refStride, w, h, xoff, yoff int) []byte {
	fdata := make([]uint16, (h+1)*w)
	for i := 0; i < h+1; i++ {
		for j := 0; j < w; j++ {
			a := ref[i*refStride+j:]
			f := svtBilinearFilters2T[xoff]
			fdata[i*w+j] = uint16((uint32(a[0])*f[0] + uint32(a[1])*f[1] + 64) >> 7)
		}
	}
	out := make([]byte, h*w)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			f := svtBilinearFilters2T[yoff]
			out[i*w+j] = uint8((uint32(fdata[i*w+j])*f[0] + uint32(fdata[(i+1)*w+j])*f[1] + 64) >> 7)
		}
	}
	return out
}

func TestSVTBilinearPredictMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const stride = 160
	ref := make([]byte, stride*110)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	var scratch svtSubpelScratch
	sizes := [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 32}, {32, 64}, {64, 64}, {64, 16}, {16, 64}}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for xoff := 0; xoff < 8; xoff++ {
			for yoff := 0; yoff < 8; yoff++ {
				off := 3*stride + 5
				want := svtBilinearReference(ref[off:], stride, w, h, xoff, yoff)
				src := ref[40*stride+70:]
				var sse, sum int64
				for r := 0; r < h; r++ {
					for c := 0; c < w; c++ {
						d := int64(want[r*w+c]) - int64(src[r*stride+c])
						sse += d * d
						sum += d
					}
				}
				wantVar := uint32(sse - sum*sum/int64(w*h))
				if got := svtSubpelVarianceArch(&scratch, ref[off:], stride, src, stride, w, h, xoff, yoff); got != wantVar {
					t.Fatalf("svf %dx%d off (%d,%d) = %d, want %d", w, h, xoff, yoff, got, wantVar)
				}
				svtBilinearPredict(&scratch, ref[off:], stride, w, h, xoff, yoff)
				for i := range want {
					if scratch.pred[i] != want[i] {
						t.Fatalf("scalar %dx%d off (%d,%d): sample %d = %d, want %d", w, h, xoff, yoff, i, scratch.pred[i], want[i])
					}
				}
			}
		}
	}
}

// svtPrunedTreeModel transliterates mcomp.c svt_av1_find_best_sub_pixel_tree_pruned
// (with first_level_check_fast / second_level_check_fast / svt_check_better_fast)
// over scalar reference kernels, for the sub-pel search's differential test.
type svtPrunedTreeModel struct {
	src, ref         []byte
	stride           int
	width, height    int
	px, py, bw, bh   int
	refMV            motion.Vector
	epb              int64
	qp               uint64
	earlyExitTh      uint64
	itersPerStep     int
	ctrls            svtSubpelCtrls
	besterr, distort uint64
	best             motion.Vector
}

func (m *svtPrunedTreeModel) refAt(x, y int) byte {
	x = min(max(x, 0), m.width-1)
	y = min(max(y, 0), m.height-1)
	return m.ref[y*m.stride+x]
}

func (m *svtPrunedTreeModel) variance(pred []byte) uint64 {
	var sse, sum int64
	for r := 0; r < m.bh; r++ {
		for c := 0; c < m.bw; c++ {
			d := int64(pred[r*m.bw+c]) - int64(m.src[(m.py+r)*m.stride+m.px+c])
			sse += d * d
			sum += d
		}
	}
	return uint64(sse - sum*sum/int64(m.bw*m.bh))
}

func (m *svtPrunedTreeModel) svf(mv motion.Vector) uint64 {
	x, y := m.px+int(mv.Col>>3), m.py+int(mv.Row>>3)
	win := make([]byte, (m.bw+1)*(m.bh+1))
	for r := 0; r <= m.bh; r++ {
		for c := 0; c <= m.bw; c++ {
			win[r*(m.bw+1)+c] = m.refAt(x+c, y+r)
		}
	}
	return m.variance(svtBilinearReference(win, m.bw+1, m.bw, m.bh, int(mv.Col&7), int(mv.Row&7)))
}

func (m *svtPrunedTreeModel) mvCost(mv motion.Vector) uint64 {
	d := absI(int(mv.Row)-int(m.refMV.Row)) + absI(int(mv.Col)-int(m.refMV.Col))
	return uint64((int64(d<<8)*m.epb + 8192) >> 14)
}

func absI(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func (m *svtPrunedTreeModel) checkBetterFast(this motion.Vector) uint64 {
	if !svtSubpelLimits(m.px, m.py, m.bw, m.bh, m.width, m.height, m.refMV).contains(this) {
		return 1<<31 - 1
	}
	cost := m.mvCost(this)
	if bestcost := m.distort + cost; bestcost > m.besterr*m.earlyExitTh/1000 {
		return bestcost
	}
	thismse := m.svf(this)
	cost += thismse
	weight := uint64(100)
	if m.ctrls.biasFP != 0 && m.best.Col%8 == 0 && m.best.Row%8 == 0 {
		weight = m.ctrls.biasFP
	}
	if cost*weight/100 < m.besterr {
		m.besterr, m.best, m.distort = cost, this, thismse
	}
	return cost
}

func (m *svtPrunedTreeModel) firstLevelCheckFast(this motion.Vector, hstep int16, orgerr uint64) motion.Vector {
	left := m.checkBetterFast(motion.Vector{Row: this.Row, Col: this.Col - hstep})
	right := m.checkBetterFast(motion.Vector{Row: this.Row, Col: this.Col + hstep})
	up := m.checkBetterFast(motion.Vector{Row: this.Row - hstep, Col: this.Col})
	down := m.checkBetterFast(motion.Vector{Row: this.Row + hstep, Col: this.Col})
	diag := motion.Vector{Row: hstep, Col: hstep}
	if left <= right {
		diag.Col = -hstep
	}
	if up <= down {
		diag.Row = -hstep
	}
	if m.besterr >= orgerr {
		return diag
	}
	m.checkBetterFast(motion.Vector{Row: this.Row + diag.Row, Col: this.Col + diag.Col})
	return diag
}

func (m *svtPrunedTreeModel) secondLevelCheckFast(this, diag motion.Vector, hstep int16) {
	tr, tc, br, bc := this.Row, this.Col, m.best.Row, m.best.Col
	switch {
	case tr != br && tc != bc:
		m.checkBetterFast(motion.Vector{Row: br, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc})
	case tr == br && tc != bc:
		m.checkBetterFast(motion.Vector{Row: br + hstep, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br - hstep, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br - diag.Row, Col: bc})
	case tr != br && tc == bc:
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc + hstep})
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc - hstep})
		m.checkBetterFast(motion.Vector{Row: br, Col: bc - diag.Col})
	}
}

func (m *svtPrunedTreeModel) run(start motion.Vector) motion.Vector {
	m.best = start
	x, y := m.px+int(start.Col>>3), m.py+int(start.Row>>3)
	pred := make([]byte, m.bw*m.bh)
	for r := 0; r < m.bh; r++ {
		for c := 0; c < m.bw; c++ {
			pred[r*m.bw+c] = m.refAt(x+c, y+r)
		}
	}
	m.distort = m.variance(pred)
	m.besterr = m.distort + m.mvCost(start)
	if m.qp*m.besterr < uint64((m.bw*m.bh)<<5)*m.ctrls.absThMult {
		return m.best
	}
	if m.ctrls.predVarianceTh != 0 {
		var sse, sum int64
		for _, v := range pred {
			sse += int64(v) * int64(v)
			sum += int64(v)
		}
		n := int64(m.bw * m.bh)
		v := uint64(sse - sum*sum/n)
		if (v+uint64(n/2))/uint64(n) < uint64(m.ctrls.predVarianceTh) {
			return m.best
		}
	}
	var orgerr uint64
	if m.ctrls.skipDiag >= 4 {
		orgerr = 0
	} else {
		demo := uint64(1)
		if m.ctrls.skipDiag >= 2 && (m.bw >= 64 || m.bh >= 64) {
			demo = 2
		}
		orgerr = 1<<31 - 1
		if m.ctrls.skipDiag != 0 {
			orgerr = m.besterr / demo
		}
	}
	hstep := int16(4)
	for iter := 0; iter < m.ctrls.rounds; iter++ {
		prev := m.besterr
		diag := m.firstLevelCheckFast(start, hstep, orgerr)
		if m.besterr < orgerr && m.itersPerStep > 1 {
			m.secondLevelCheckFast(start, diag, hstep)
		}
		hstep >>= 1
		start = m.best
		if m.ctrls.skipDiag != 0 && iter < 1 {
			orgerr = min(orgerr, m.besterr)
		}
		dev := (int64(max(m.besterr, 1)) - int64(max(prev, 1))) * 100 / int64(max(prev, 1))
		if dev >= m.ctrls.roundDevTh {
			return m.best
		}
	}
	return m.best
}

func TestSVTSubpelSearchMatchesPrunedTreeModel(t *testing.T) {
	const width, height, stride = 176, 144, 192
	rng := rand.New(rand.NewSource(7))
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	level6 := svtSubpelCtrls{rounds: 2, roundDevTh: 1<<31 - 1, skipDiag: 3, biasFP: 110}
	sizes := [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}, {8, 16}, {32, 16}, {16, 32}, {64, 16}, {16, 64}}
	for trial := 0; trial < 300; trial++ {
		// Smooth content with a sub-pel shift so the tree actually moves.
		fx, fy := rng.Float64()*0.3, rng.Float64()*0.3
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				v := 128 + 60*math.Sin(float64(x)*fx+float64(y)*fy) + float64(rng.Intn(9)-4)
				src[y*stride+x] = byte(min(max(v, 0), 255))
				w := 128 + 60*math.Sin((float64(x)+0.4)*fx+(float64(y)-0.3)*fy) + float64(rng.Intn(9)-4)
				ref[y*stride+x] = byte(min(max(w, 0), 255))
			}
		}
		sz := sizes[trial%len(sizes)]
		bw, bh := sz[0], sz[1]
		px := (rng.Intn((width-bw)/8 + 1)) * 8
		py := (rng.Intn((height-bh)/8 + 1)) * 8
		start := motion.Vector{Row: int16(rng.Intn(5)-2) * 8, Col: int16(rng.Intn(5)-2) * 8}
		refMV := motion.Vector{Row: int16(rng.Intn(33) - 16), Col: int16(rng.Intn(33) - 16)}
		ctrls := svtSubpelLevel8
		if trial&1 == 1 {
			ctrls = level6
		}
		var st lossyEncodeState
		st.qIndex = uint8(60 + rng.Intn(180))
		st.rdMult = int64(2000 + rng.Intn(60000))
		got, _ := st.svtSubpelSearch(src, ref, stride, width, height, px, py, bw, bh, start, refMV, &ctrls, st.rdMult)
		sq := max(bw, bh)
		m := svtPrunedTreeModel{
			src: src, ref: ref, stride: stride, width: width, height: height,
			px: px, py: py, bw: bw, bh: bh, refMV: refMV,
			epb: max(st.rdMult>>6, 1), qp: uint64(st.qIndex) >> 2,
			earlyExitTh: uint64(1020 - (sq >> 2)), itersPerStep: 1, ctrls: ctrls,
		}
		if want := m.run(start); got != want {
			t.Fatalf("trial %d %dx%d at (%d,%d) start %+v: got %+v, want %+v", trial, bw, bh, px, py, start, got, want)
		}
	}
}

// randomSVTPlane is a padded w x h plane of random samples, or of a smooth
// ramp with a little noise, so the searches see both ties and distinct
// minima.
func randomSVTPlane(rng *rand.Rand, w, h, border int, smooth bool) *svtPlane {
	p := &svtPlane{}
	p.size(w, h, border)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := rng.Intn(256)
			if smooth {
				v = (x*3+y*5)&255 ^ rng.Intn(4)
			}
			p.pix[p.org+y*p.stride+x] = byte(v)
		}
	}
	p.pad()
	return p
}

func TestSVTColTotalsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 200; iter++ {
		ref := randomSVTPlane(rng, 160, 90, 32, iter%2 == 0)
		if iter%9 == 0 {
			for i := range ref.pix {
				ref.pix[i] = 255
			}
		}
		rows := 1 + rng.Intn(16)
		span := 16 * (1 + rng.Intn(svtHMEMaxCols/16))
		s2 := 2 * ref.stride
		off := rng.Intn(len(ref.pix) - (rows+1)*s2 - span)
		var got, want [svtHMEMaxCols]uint16
		svtColTotalsArch(&got, ref.pix[off:], s2, rows, span)
		svtColTotals(&want, ref.pix[off:], s2, rows, span)
		if got != want {
			t.Fatalf("iter %d: totals differ", iter)
		}
		// Slide two rows down.
		out, in := ref.pix[off:], ref.pix[off+rows*s2:]
		svtColTotalsSlideArch(&got, out, in, span)
		svtColTotals(&want, ref.pix[off+s2:], s2, rows, span)
		if got != want {
			t.Fatalf("iter %d: slid totals differ", iter)
		}
	}
}

func TestSVTHMEColSumRowMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for iter := 0; iter < 400; iter++ {
		smooth := iter%3 == 0
		ref := randomSVTPlane(rng, 160, 90, 32, smooth)
		cur := randomSVTPlane(rng, 160, 90, 32, !smooth)
		if iter%7 == 0 {
			for i := range ref.pix {
				ref.pix[i] = 255
			}
			for i := range cur.pix {
				cur.pix[i] = byte(255 * (iter & 1))
			}
		}
		bw := []int{16, 32}[iter&1]
		rows := 1 + rng.Intn(16)
		w := 8 * (1 + rng.Intn(4))
		s2 := 2 * ref.stride
		r0 := rng.Intn(ref.h + 2*ref.border - 2*rows)
		c0 := rng.Intn(ref.stride - bw - w - 16)
		s := cur.pix[rng.Intn(4)*cur.stride+rng.Intn(40):]
		refPix := ref.pix[r0*ref.stride+c0:]
		var cw [svtHMEMaxCols]uint16
		svtColTotals(&cw, refPix, s2, rows, bw+w+8)
		srcSum := svtBlockSumSub(s, s2, bw, rows)
		if got := svtBlockSumSubArch(s, s2, bw, rows); got != srcSum {
			t.Fatalf("iter %d: block sum %d, want %d", iter, got, srcSum)
		}
		var got, want [svtHMEMaxCols / 8][8]uint32
		key := svtHMEColSumRow(&got, s, s2, refPix, &cw, bw, rows, w, srcSum)
		wantKey := svtHMEColSumRowScalar(&want, s, s2, refPix, cw[:], bw, rows, w, srcSum)
		if nilKey := svtHMEColSumRow(nil, s, s2, refPix, &cw, bw, rows, w, srcSum); nilKey != key {
			t.Fatalf("iter %d: key %#x without SADs, %#x with", iter, nilKey, key)
		}
		best := ^uint32(0)
		for x := 0; x < w; x++ {
			var direct uint32
			for r := 0; r < rows; r++ {
				for c := 0; c < bw; c++ {
					a, b := int(s[r*s2+c]), int(refPix[r*s2+x+c])
					direct += uint32(max(a-b, b-a))
				}
			}
			if g, wv := got[x/8][x%8], want[x/8][x%8]; g != wv || g != direct {
				t.Fatalf("iter %d bw %d rows %d x %d: got %d scalar %d direct %d", iter, bw, rows, x, g, wv, direct)
			}
			// The key is the minimum SAD and its first position.
			if direct<<7|uint32(x) < best {
				best = direct<<7 | uint32(x)
			}
		}
		if key != wantKey || key != best {
			t.Fatalf("iter %d bw %d rows %d: key %#x scalar %#x direct %#x", iter, bw, rows, key, wantKey, best)
		}
	}
}

func TestSVTHMESearchColSumMatchesSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	tested := 0
	for iter := 0; iter < 400; iter++ {
		smooth := iter%2 == 0
		w, h := 160, 96
		border := 32
		cur := randomSVTPlane(rng, w, h, border, smooth)
		ref := randomSVTPlane(rng, w, h, border, smooth)
		bw := []int{16, 32}[iter%2]
		bh := bw
		bx := rng.Intn(w/bw) * bw
		by := rng.Intn(h/bh) * bh
		sw := 8 * (1 + rng.Intn(6))
		sh := 1 + rng.Intn(12)
		ox, oy, sw, sh := svtSearchAreaCorrect(bx, by, -sw/2, -sh/2, sw, sh, w, h, border-1, true)
		if sw%8 != 0 || ref.org+(by+oy+sh-1+bh-2)*ref.stride+bx+ox+svtHMEColSpan(sw, bw) > len(ref.pix) {
			continue
		}
		wantSAD, wantX, wantY := svtHMESearchMaxMin(cur, ref, bx, by, bw, bh, ox, oy, sw, sh)
		gotSAD, gotX, gotY := svtHMESearchColSum(cur, ref, bx, by, bw, bh, ox, oy, sw, sh)
		if gotSAD != wantSAD || gotX != wantX || gotY != wantY {
			t.Fatalf("iter %d: colsum (%d,%d,%d) != search (%d,%d,%d)", iter, gotSAD, gotX, gotY, wantSAD, wantX, wantY)
		}
		tested++
	}
	if tested < 100 {
		t.Fatalf("only %d searches compared", tested)
	}
}

func benchmarkSVTHMESearch(b *testing.B, bw, w, h int, colSum bool) {
	rng := rand.New(rand.NewSource(4))
	cur := randomSVTPlane(rng, 320, 180, 64, false)
	ref := randomSVTPlane(rng, 320, 180, 64, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if colSum {
			svtHMESearchColSum(cur, ref, 64, 64, bw, bw, -w/2, -h/2, w, h)
		} else {
			svtHMESearchMaxMin(cur, ref, 64, 64, bw, bw, -w/2, -h/2, w, h)
		}
	}
}

func BenchmarkSVTHMESearchL0(b *testing.B) {
	b.Run("maxmin", func(b *testing.B) { benchmarkSVTHMESearch(b, 16, 16, 8, false) })
	b.Run("colsum", func(b *testing.B) { benchmarkSVTHMESearch(b, 16, 16, 8, true) })
}

func BenchmarkSVTHMESearchL1(b *testing.B) {
	b.Run("maxmin", func(b *testing.B) { benchmarkSVTHMESearch(b, 32, 8, 3, false) })
	b.Run("colsum", func(b *testing.B) { benchmarkSVTHMESearch(b, 32, 8, 3, true) })
}

// fillMEPattern fills b with random, flat (tie-heavy) or extreme content.
func fillMEPattern(rng *rand.Rand, b []byte, mode int) {
	for i := range b {
		switch mode % 4 {
		case 0:
			b[i] = byte(rng.Intn(256))
		case 1:
			b[i] = 128 // flat: every position ties
		case 2:
			if rng.Intn(2) == 0 {
				b[i] = 0
			} else {
				b[i] = 255
			}
		default:
			b[i] = byte(100 + rng.Intn(4))
		}
	}
}

func TestSVTSAD16xhx8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const stride = 96
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 64; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		for rows := 1; rows <= 16; rows++ {
			so := rng.Intn(16)
			ro := rng.Intn(16)
			// Exact-length slices: the kernel's last read is the slice end.
			s := src[so : so+(rows-1)*2*stride+16]
			r := ref[ro : ro+(rows-1)*stride+23]
			var got [8]uint32
			svtSAD16xhx8(&got, s, 2*stride, r, stride, rows)
			want := svtSAD16xhx8Scalar(s, 2*stride, r, stride, rows)
			if got != want {
				t.Fatalf("trial %d rows %d: got %v want %v", trial, rows, got, want)
			}
		}
	}
}

func BenchmarkSVTSAD16xhx8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const stride = 416
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	fillMEPattern(rng, src, 0)
	fillMEPattern(rng, ref, 0)
	for _, rows := range []int{8, 16} {
		b.Run(map[int]string{8: "L0rows8", 16: "L1rows16"}[rows], func(b *testing.B) {
			var sink uint32
			for b.Loop() {
				var a [8]uint32
				svtSAD16xhx8(&a, src, 2*stride, ref, 2*stride, rows)
				sink += a[3]
			}
			_ = sink
		})
		b.Run(map[int]string{8: "old_x4x2_L0rows8", 16: "old_x4x2_L1rows16"}[rows], func(b *testing.B) {
			var sink int
			for b.Loop() {
				a0, _, _, _ := sad16xNx4(src, 2*stride, ref, 2*stride, rows)
				a4, _, _, _ := sad16xNx4(src, 2*stride, ref[4:], 2*stride, rows)
				sink += a0 + a4
			}
			_ = sink
		})
	}
}

func BenchmarkSVTHMESearch(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	var cur, ref svtPlane
	cur.size(320, 180, svtSixBorder)
	ref.size(320, 180, svtSixBorder)
	fillMEPattern(rng, cur.pix, 0)
	fillMEPattern(rng, ref.pix, 0)
	var half, halfRef svtPlane
	half.size(640, 360, svtHalfBorder)
	halfRef.size(640, 360, svtHalfBorder)
	fillMEPattern(rng, half.pix, 0)
	fillMEPattern(rng, halfRef.pix, 0)
	b.Run("L0", func(b *testing.B) {
		for b.Loop() {
			svtHMESearch(&cur, &ref, 160, 96, 16, 16, -16, -4, 16, 4)
		}
	})
	b.Run("L1", func(b *testing.B) {
		for b.Loop() {
			svtHMESearch(&half, &halfRef, 320, 192, 32, 32, -4, -1, svtHMEL1W, svtHMEL1H)
		}
	})
}

// TestSVTHMECorrectArea pins hme_level_0 / hme_level_1's search-area
// correction on the 1280x720 pyramid (borders 32 and 64).
func TestSVTHMECorrectArea(t *testing.T) {
	for _, c := range []struct{ orgX, orgY, ox, oy, w, h, picW, picH, pad, wx, wy, ww, wh int }{
		{160, 96, -16, -4, 16, 4, 320, 180, 31, -16, -4, 16, 4}, // L0 inside
		{608, 0, 26, -1, 8, 3, 640, 360, 63, 26, -1, 6, 3},      // L1 right edge: 608+26+8 > 640
		{608, 0, 40, -1, 8, 3, 640, 360, 63, 31, -1, 1, 3},      // origin past the right edge
		{0, 0, -70, -70, 8, 3, 640, 360, 63, -63, -63, 8, 3},    // left/top: origin moves, size kept
		{0, 352, 0, 7, 8, 3, 640, 360, 63, 0, 7, 8, 1},          // bottom shrink
		{304, 0, 0, 0, 16, 4, 320, 180, 31, 0, 0, 16, 4},        // L0 last column, exact fit
		{304, 176, 0, 2, 16, 4, 320, 180, 31, 0, 2, 16, 2},      // L0 bottom shrink
	} {
		ox, oy, w, h := svtSearchAreaCorrect(c.orgX, c.orgY, c.ox, c.oy, c.w, c.h, c.picW, c.picH, c.pad, true)
		if ox != c.wx || oy != c.wy || w != c.ww || h != c.wh {
			t.Errorf("%+v: got %d %d %d %d", c, ox, oy, w, h)
		}
	}
}

// TestSVTHMEL0SearchArea pins get_hme_l0_search_area's per-region size at
// preset 12 rtc (CBR sequence qp 35) for the low-delay reference distances,
// camera and screen content.
func TestSVTHMEL0SearchArea(t *testing.T) {
	for _, c := range []struct {
		screen     bool
		dist, w, h int
	}{
		{false, 1, 16, 4}, {false, 2, 16, 8}, {false, 4, 16, 12}, {false, 8, 32, 20}, {false, 16, 48, 40}, {false, 32, 48, 48},
		{true, 1, 16, 9}, {true, 2, 32, 18}, {true, 4, 32, 27},
	} {
		ctrl := svtMEControls(c.screen, 1280, 720, 35)
		var p svtPyramid
		p.setDistance(c.dist, ctrl.l0Min, ctrl.l0Max)
		if p.l0W != c.w || p.l0H != c.h {
			t.Errorf("screen %v dist %d: %dx%d, want %dx%d", c.screen, c.dist, p.l0W, p.l0H, c.w, c.h)
		}
	}
}

func TestSVTSAD32xhx8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	const stride = 96
	src := make([]byte, stride*40)
	ref := make([]byte, stride*40)
	for trial := 0; trial < 64; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		if trial%5 == 0 { // the largest SADs: every difference 255
			for i := range src {
				src[i], ref[i] = 0, 255
			}
		}
		for rows := 1; rows <= 16; rows++ {
			so := rng.Intn(16)
			ro := rng.Intn(16)
			s := src[so : so+(rows-1)*2*stride+32]
			r := ref[ro : ro+(rows-1)*stride+39]
			var got [8]uint32
			got[3] = 7 // accumulates
			svtSAD32xhx8(&got, s, 2*stride, r, stride, rows)
			a := svtSAD16xhx8Scalar(s, 2*stride, r, stride, rows)
			b := svtSAD16xhx8Scalar(s[16:], 2*stride, r[16:], stride, rows)
			for k := range a {
				want := a[k] + b[k]
				if k == 3 {
					want += 7
				}
				if got[k] != want {
					t.Fatalf("trial %d rows %d pos %d: got %d want %d", trial, rows, k, got[k], want)
				}
			}
		}
	}
}

// TestSVTMEControls pins the qp-scaled preset-12 rtc ME/HME areas
// (set_hme_search_params, set_me_search_params with
// svt_aom_get_qp_based_th_scaling_factors at the CBR sequence qp 35) and the
// integer-search area per reference distance.
func TestSVTMEControls(t *testing.T) {
	cam := svtMEControls(false, 1280, 720, 35)
	if cam.l0Min != (svtSearchArea{8, 8}) || cam.l0Max != (svtSearchArea{96, 96}) ||
		cam.meMin != (svtSearchArea{8, 3}) || cam.meMax != (svtSearchArea{8, 3}) || cam.hmeL2 || cam.prehme {
		t.Fatalf("camera controls %+v", cam)
	}
	sc := svtMEControls(true, 1280, 720, 35)
	if sc.l0Min != (svtSearchArea{18, 18}) || sc.l0Max != (svtSearchArea{107, 107}) ||
		sc.meMin != (svtSearchArea{13, 9}) || sc.meMax != (svtSearchArea{18, 9}) || !sc.hmeL2 || !sc.prehme {
		t.Fatalf("screen controls %+v", sc)
	}
	for _, c := range []struct {
		ctrl         svtMECtrl
		f            int
		sad, zz      uint32
		wantW, wantH int
	}{
		{cam, 1, 1 << 20, 1 << 20, 8, 3},
		{cam, 3, 0, 1 << 20, 8, 3},
		{cam, 1, 0, 100, 1, 1},
		{sc, 1, 1 << 20, 1 << 20, 16, 9}, // 13x9 rounded up to 16
		{sc, 3, 1 << 20, 1 << 20, 24, 9}, // capped at 18x9, rounded up
		{sc, 3, 1000, 1 << 20, 8, 3},     // low hme_sad: divisor 8
		{sc, 3, 1 << 20, 600, 1, 1},      // below me_early_exit_th / 6
	} {
		if w, h := c.ctrl.meArea(c.f, 64, 64, c.sad, c.zz); w != c.wantW || h != c.wantH {
			t.Errorf("screen %v f %d sad %d zz %d: %dx%d, want %dx%d", c.ctrl.screen, c.f, c.sad, c.zz, w, h, c.wantW, c.wantH)
		}
	}
}

// TestSVTRateControlReplaysSVT410Traces drives svtRateControl with the
// per-frame inputs an instrumented SVT-AV1 v4.1.0 fed its own one-pass CBR
// controller and requires the same qindex and frame target on every picture.
//
// The traces in testdata/svtrc are SvtAv1EncApp runs (preset 13, --rtc 1,
// --pred-struct 1, --rc 2, --buf-initial-sz 500) on the first 120 frames of the
// Xiph 1280x720 60 fps FourPeople and Johnny clips at 3 Mbps, once with the
// default three-layer hierarchy and once flat (--hierarchical-levels 0).
// svt410_rc_trace.patch is the logging added to rc_vbr_cbr.c: per picture the
// state svt_av1_rc_calc_qindex_rate_control chose q from (ME distortion,
// reference q / layer / slice type, rate-correction factors) and, after the
// frame, its coded size and cyclic-refresh segment statistics. The replay feeds
// the recorded ME distortion, reference, size and cyclic-refresh inputs, so a
// pass proves the controller arithmetic (buffer model, active-quality
// bounds, q regulation, flat and layered adjust_q_cbr, correction factors) is
// SVT's; it says nothing about the sizes the encoder produces.
func TestSVTRateControlReplaysSVT410Traces(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "svtrc", "*.csv"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no SVT rate-control traces: %v", err)
	}
	for _, file := range files {
		name := filepath.Base(file)
		t.Run(name, func(t *testing.T) {
			replaySVTRCTrace(t, file, strings.HasPrefix(name, "flat_"))
		})
	}
}

func replaySVTRCTrace(t *testing.T, file string, flat bool) {
	t.Helper()
	fh, err := os.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	var rows [][]int
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, ",")
		row := make([]int, len(fields))
		for i, f := range fields {
			v, err := strconv.Atoi(f)
			if err != nil {
				t.Fatalf("%s: bad field %q in %q", file, f, line)
			}
			row[i] = v
		}
		if len(row) != 18 {
			t.Fatalf("%s: %d fields in %q, want 18", file, len(row), line)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 120 {
		t.Fatalf("%s: %d pictures, want 120", file, len(rows))
	}

	const (
		colPic = iota
		colKey
		colTL
		colQ
		colTarget
		colBits
		colME
		colPrevME
		colRefs
		colRefQ
		colRefTL
		colRefSlice
		colCRApply
		colCRSeg1
		colCRSeg2
		colCRDelta1
		colCRDelta2
		colCRCount
	)
	const iSlice = 1 // SVT-AV1 SliceType I_SLICE
	var rc svtRateControl
	rc.init(3_000_000, 60, 16, 255, 1280, 720, rcBuffers{initialMs: 500})
	for _, r := range rows {
		pic := r[colPic]
		f := svtRCFrame{
			key:         r[colKey] == 1,
			flat:        flat,
			hierLevels:  2,
			frameOffset: pic,
			width:       1280,
			height:      720,
			gfUpdate:    pic%lpd1MaxGFInterval == 0,
			refValid:    r[colRefs] > 0,
			refBaseQ:    r[colRefQ],
			refLayer:    r[colRefTL],
			refIsI:      r[colRefSlice] == iSlice,
		}
		if !flat {
			f.temporalLayer = r[colTL]
		}
		rc.curAvgBaseMEDist = uint32(r[colME])
		rc.prevAvgBaseMEDist = uint32(r[colPrevME])
		rc.target(&f)
		q := int(rc.pickQ(&f))
		if f.thisFrame != r[colTarget] || q != r[colQ] {
			t.Fatalf("picture %d: target %d q %d, SVT-AV1 v4.1.0 target %d q %d", pic, f.thisFrame, q, r[colTarget], r[colQ])
		}
		var cr svtCRStats
		if r[colCRApply] == 1 {
			cr = svtCRStats{apply: true, seg1: r[colCRSeg1], seg2: r[colCRSeg2], delta1: r[colCRDelta1], delta2: r[colCRDelta2], sbCount: r[colCRCount]}
		}
		rc.postencode(&f, r[colQ], r[colBits], cr)
	}
}

func TestRCBuffersResolve(t *testing.T) {
	got, err := rcBuffers{}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 600, optimalMs: 600, maxMs: 1000}) {
		t.Fatalf("zero value resolves to %+v, %v; want SVT-AV1's 600 / 600 / 1000", got, err)
	}
	got, err = rcBuffers{initialMs: 500, optimalMs: 600, maxMs: 1000}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 500, optimalMs: 600, maxMs: 1000}) {
		t.Fatalf("500 / 600 / 1000 resolves to %+v, %v", got, err)
	}
	// enc_settings.c clamps an initial or optimal size that is not below the
	// maximum to maximum-1.
	got, err = rcBuffers{initialMs: 800, optimalMs: 900, maxMs: 700}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 699, optimalMs: 699, maxMs: 700}) {
		t.Fatalf("800 / 900 / 700 resolves to %+v, %v; want 699 / 699 / 700", got, err)
	}
	// The default initial size is clamped the same way under a short maximum.
	got, err = rcBuffers{maxMs: 500}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 499, optimalMs: 499, maxMs: 500}) {
		t.Fatalf("max 500 resolves to %+v, %v; want 499 / 499 / 500", got, err)
	}
	for _, bad := range []rcBuffers{{initialMs: -1}, {initialMs: 19}, {optimalMs: 10001}, {maxMs: 5}, {maxMs: 20000}} {
		if _, err := bad.resolve(); err == nil {
			t.Fatalf("%+v resolved without error", bad)
		}
	}
}

func TestRateControlConfigBuffersReachController(t *testing.T) {
	rc := RateControlConfig{TargetBitsPerSecond: 3_000_000, FramesPerSecond: 60, MinQIndex: 16, MaxQIndex: 255}
	def, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer def.Close()
	if got := def.svtRC.startingBufferLevel; got != 1_800_000 {
		t.Fatalf("default starting buffer %d, want 600 ms of 3 Mbps", got)
	}
	rc.BufferInitialMs, rc.BufferOptimalMs, rc.BufferMaxMs = 500, 600, 1000
	enc, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if got, want := enc.svtRC.startingBufferLevel, int64(1_500_000); got != want {
		t.Fatalf("starting buffer %d, want %d", got, want)
	}
	if enc.svtRC.optimalBufferLevel != 1_800_000 || enc.svtRC.maximumBufferSize != 3_000_000 {
		t.Fatalf("optimal %d maximum %d", enc.svtRC.optimalBufferLevel, enc.svtRC.maximumBufferSize)
	}
	rc.BufferInitialMs = 5
	if _, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig()); err == nil {
		t.Fatal("an initial buffer below 20 ms was accepted")
	}
}

// TestSVTScreenContentDetectionMatchesSVT compares svt_aom_is_screen_content
// counters and classes with SVT-AV1 4.1.0's own (an instrumented
// SvtAv1EncApp --scm 2 on frame 0 of each 1280x720 real-corpus encoder input).
// Set GOAV1_REALCORPUS_YUV to the directory of <clip>_720p.yuv files.
func TestSVTScreenContentDetectionMatchesSVT(t *testing.T) {
	dir := os.Getenv("GOAV1_REALCORPUS_YUV")
	if dir == "" {
		t.Skip("GOAV1_REALCORPUS_YUV not set")
	}
	const w, h = 1280, 720
	cases := []struct {
		clip string
		cnt  svtSCCounts
		cls  svtSCClass
	}{
		{"wikipedia", svtSCCounts{402, 304, 4424, 1274}, svtSCClass{true, true, true, true, true}},
		{"debugging", svtSCCounts{320, 220, 1938, 1706}, svtSCClass{class4: true}},
		{"fourpeople", svtSCCounts{2, 1, 393, 38}, svtSCClass{}},
		{"johnny", svtSCCounts{17, 17, 607, 77}, svtSCClass{}},
		{"mtbike", svtSCCounts{20, 12, 540, 60}, svtSCClass{}},
		{"parkjoy", svtSCCounts{0, 0, 0, 0}, svtSCClass{}},
	}
	var d svtSCDetector
	buf := make([]byte, w*h)
	for _, c := range cases {
		f, err := os.Open(filepath.Join(dir, c.clip+"_720p.yuv"))
		if err != nil {
			t.Skipf("%s: %v", c.clip, err)
		}
		_, err = f.Read(buf)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		cls, cnt := d.detect(buf, w, w, h)
		if cnt != c.cnt || cls != c.cls {
			t.Errorf("%s: counts %+v classes %+v, SVT %+v %+v", c.clip, cnt, cls, c.cnt, c.cls)
		}
	}
}

// TestSVTScreenContentDetectionSynthetic pins the classes on two synthetic
// pictures: flat two-color text-like blocks are screen content (sc_class1),
// smooth noise is not.
func TestSVTScreenContentDetectionSynthetic(t *testing.T) {
	const w, h = 320, 192
	text := make([]byte, w*h)
	noise := make([]byte, w*h)
	seed := uint32(7)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			text[y*w+x] = 255
			if (x/3+y/5)%4 == 0 {
				text[y*w+x] = 20
			}
			seed = seed*1664525 + 1013904223
			noise[y*w+x] = uint8(100 + x/4 + int(seed>>28))
		}
	}
	var d svtSCDetector
	if cls := d.classes(ContentAuto, text, w, w, h); !cls.class1 {
		t.Fatalf("text picture classes %+v, want sc_class1", cls)
	}
	if cls := d.classes(ContentAuto, noise, w, w, h); cls != (svtSCClass{}) {
		t.Fatalf("noise picture classes %+v, want none", cls)
	}
	if cls := d.classes(ContentCamera, text, w, w, h); cls != (svtSCClass{}) {
		t.Fatalf("camera classes %+v, want none", cls)
	}
	if cls := d.classes(ContentScreen, noise, w, w, h); !cls.class1 || !cls.class4 {
		t.Fatalf("screen classes %+v, want all", cls)
	}
	if n := testing.AllocsPerRun(5, func() { d.detect(text, w, w, h) }); n != 0 {
		t.Fatalf("detect allocates %.0f", n)
	}
}

// TestSweepSearchPoint8MatchesPoint1 checks the eight-position batch
// (open_loop_me_get_eight_search_point_results_block) against eight
// single-position searches (open_loop_me_get_search_point_results_block).
func TestSweepSearchPoint8MatchesPoint1(t *testing.T) {
	const stride = 160
	rng := rand.New(rand.NewSource(5))
	src := make([]byte, stride*100)
	ref := make([]byte, stride*100)
	for trial := 0; trial < 24; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		if trial%2 == 1 { // structured: ref is src shifted, so argmins move
			for i := range ref {
				ref[i] = src[(i+3)%len(src)] + byte(rng.Intn(5))
			}
		}
		var a, b realtimeSBMultiSizeME
		a.reset()
		b.reset()
		srcBase, refBase := 10*stride+20, 9*stride+14
		for y := 0; y < 3; y++ {
			rb := refBase + y*stride
			if !a.sweepSearchPoint8(src, ref, stride, srcBase, rb, -6, y-1) {
				t.Fatal("batch declined in-bounds reads")
			}
			for k := range 8 {
				b.sweepSearchPoint1(src, stride, ref, stride, srcBase, rb+k, -6+k, y-1)
			}
		}
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: batch differs: %v %v vs %v %v", trial, a.bestSAD[:6], a.bestMV[:6], b.bestSAD[:6], b.bestMV[:6])
		}
	}
}

// TestSVTAllSAD8x8_16x16SubMatchesScalar is the differential test of the
// svt_ext_all_sad_calculation_8x8_16x16 kernel against its scalar
// definition, on exact-length slices (the kernel's last read is the slice
// end), with random running bests and ties.
func TestSVTAllSAD8x8_16x16SubMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const stride = 96
	src := make([]byte, 64*stride)
	ref := make([]byte, 64*stride)
	for trial := 0; trial < 160; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		s := src[:63*stride+64]
		ro := rng.Intn(stride - 71)
		r := ref[ro : ro+63*stride+71]
		var a realtimeSBMultiSizeME
		a.reset()
		for i := range a.bestSAD {
			// Random running bests, including ties with the new SADs.
			if trial%3 != 0 {
				a.bestSAD[i] = uint32(rng.Intn(20000))
				a.bestMV[i] = svtPackMV(rng.Intn(9)-4, 1)
			}
		}
		b := a
		mvX := rng.Intn(40) - 20
		if trial%7 == 0 {
			mvX = -3 // positions cross zero: the x half wraps in 16 bits
		}
		mvY := rng.Intn(9) - 4
		var gA, gB [16][8]uint32
		svtAllSAD8x8_16x16Sub(&a, s, r, stride, mvX, mvY, &gA)
		svtAllSAD8x8_16x16SubScalar(&b, s, r, stride, mvX, mvY, &gB)
		if gA != gB || a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: kernel and scalar differ", trial)
		}
		a.eightSAD32x32_64x64(&gA, mvX, mvY)
		b.eightSAD32x32_64x64Scalar(&gB, mvX, mvY)
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: 32x32/64x64 differ", trial)
		}
	}
}

// TestSVTAllSAD8x8_16x16SubExtremes runs the kernel on all-0 against
// all-255 (every 8x8 SUB_SAD at its 2*32*255 maximum, every position tied)
// both ways, and on ref columns that alternate so the positions alternate
// between the extremes.
func TestSVTAllSAD8x8_16x16SubExtremes(t *testing.T) {
	const stride = 80
	n := 63*stride + 71
	for trial := range 4 {
		src := make([]byte, n)
		ref := make([]byte, n)
		for i := range ref {
			switch trial {
			case 0:
				ref[i] = 255
			case 1:
				src[i] = 255
			case 2:
				ref[i] = byte(255 * (i & 1))
			default:
				src[i] = byte(255 * (i & 1))
				ref[i] = byte(255 * ((i >> 1) & 1))
			}
		}
		var a realtimeSBMultiSizeME
		a.reset()
		b := a
		var gA, gB [16][8]uint32
		svtAllSAD8x8_16x16Sub(&a, src[:63*stride+64], ref, stride, -3, 2, &gA)
		svtAllSAD8x8_16x16SubScalar(&b, src[:63*stride+64], ref, stride, -3, 2, &gB)
		if gA != gB || a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: kernel and scalar differ: %v vs %v", trial, gA[0], gB[0])
		}
	}
}

// eightSAD32x32_64x64Scalar is svt_ext_eight_sad_calculation_32x32_64x64_c.
func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64Scalar(sad16 *[16][8]uint32, mvX, mvY int) {
	for k := range 8 {
		var s64 uint32
		for q := range 4 {
			s := sad16[4*q][k] + sad16[4*q+1][k] + sad16[4*q+2][k] + sad16[4*q+3][k]
			me.update(1+q, s, mvX+k, mvY)
			s64 += s
		}
		me.update(0, s64, mvX+k, mvY)
	}
}

func BenchmarkSweepSearchPoint8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	const stride = 1440
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	fillMEPattern(rng, src, 0)
	fillMEPattern(rng, ref, 0)
	var me realtimeSBMultiSizeME
	for b.Loop() {
		me.reset()
		me.sweepSearchPoint8(src, ref, stride, 4*stride+100, 3*stride+96, -4, -1)
	}
}

// TestSweepPoint1ArchMatchesGeneric checks the single-position sweep
// dispatch (svtSweepPoint1 where the build has it) against the generic
// open_loop_me_get_search_point_results_block, on exact-length windows,
// with random running bests and ties, and on windows that leave the plane.
func TestSweepPoint1ArchMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const stride = 96
	src := make([]byte, 70*stride)
	ref := make([]byte, 70*stride)
	for trial := 0; trial < 200; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		if trial%2 == 1 {
			for i := range ref {
				ref[i] = src[(i+5)%len(src)] + byte(rng.Intn(3))
			}
		}
		var a realtimeSBMultiSizeME
		a.reset()
		for i := range a.bestSAD {
			if trial%3 != 0 {
				a.bestSAD[i] = uint32(rng.Intn(20000))
				a.bestMV[i] = svtPackMV(rng.Intn(9)-4, 1)
			}
		}
		b := a
		srcBase := rng.Intn(4*stride + 1)
		refBase := rng.Intn(6*stride + 1)
		s := src
		r := ref[:refBase+63*stride+64]
		if trial%5 == 4 {
			r = ref[:refBase+63*stride+60] // too short: the generic path
		}
		mvX, mvY := rng.Intn(17)-8, rng.Intn(5)-2
		if trial%7 == 0 {
			mvX = -1
		}
		func() {
			defer func() { _ = recover() }()
			b.sweepSearchPoint1(s, stride, r, stride, srcBase, refBase, mvX, mvY)
		}()
		func() {
			defer func() { _ = recover() }()
			a.sweepPoint1Arch(s, stride, r, srcBase, refBase, mvX, mvY)
		}()
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: arch and generic differ", trial)
		}
	}
}

func TestAppendTileInfoPayloadSingleTileExplicit(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	payload, parsed := appendAndParseTileInfo(t, seq, prefix, 64, 64, tiles)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if !parsed.RefreshContext || parsed.UniformSpacing || parsed.Cols != 1 || parsed.Rows != 1 {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadUniformSplit(t *testing.T) {
	seq := tileInfoSequenceHeader()
	seq.MaxFrameWidth = 256
	seq.MaxFrameHeight = 128
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := TileInfo{
		RefreshContext:      true,
		UniformSpacing:      true,
		SBCols:              4,
		SBRows:              2,
		MinLog2Cols:         0,
		MaxLog2Cols:         2,
		MinLog2Rows:         0,
		MaxLog2Rows:         1,
		MinLog2Tiles:        0,
		Log2Cols:            1,
		Log2Rows:            1,
		Cols:                2,
		Rows:                2,
		ContextUpdateTileID: 1,
		TileSizeBytes:       3,
	}
	tiles.ColStartSB[0], tiles.ColStartSB[1], tiles.ColStartSB[2] = 0, 2, 4
	tiles.RowStartSB[0], tiles.RowStartSB[1], tiles.RowStartSB[2] = 0, 1, 2
	_, parsed := appendAndParseTileInfo(t, seq, prefix, 256, 128, tiles)
	if !parsed.UniformSpacing || parsed.Log2Cols != 1 || parsed.Log2Rows != 1 || parsed.ContextUpdateTileID != 1 || parsed.TileSizeBytes != 3 {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadInterMotionControls(t *testing.T) {
	seq := tileInfoSequenceHeader()
	seq.EnableOrderHint = true
	seq.EnableRefFrameMVS = true
	seq.OrderHintBits = 8
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	tiles := TileInfo{
		AllowHighPrecisionMV: true,
		InterpolationFilter:  InterpolationSharp,
		SwitchableMotionMode: true,
		UseRefFrameMVS:       true,
		RefreshContext:       true,
		SBCols:               1,
		SBRows:               1,
		Cols:                 1,
		Rows:                 1,
		ColStartSB:           [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:           [MaxTileRows + 1]uint16{0, 1},
	}
	_, parsed := appendAndParseTileInfo(t, seq, prefix, 64, 64, tiles)
	if !parsed.AllowHighPrecisionMV || parsed.InterpolationFilter != parser.InterpolationSharp ||
		!parsed.SwitchableMotionMode || !parsed.UseRefFrameMVS || !parsed.RefreshContext {
		t.Fatalf("tile info=%+v", parsed)
	}
}

func TestAppendTileInfoPayloadRejectsInvalid(t *testing.T) {
	seq := tileInfoSequenceHeader()
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	valid := TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [MaxTileRows + 1]uint16{0, 1},
	}
	badContext := valid
	badContext.ContextUpdateTileID = 1
	badMotion := valid
	badMotion.InterpolationFilter = InterpolationSharp
	badStarts := valid
	badStarts.ColStartSB[1] = 0
	cases := [...]TileInfo{badContext, badMotion, badStarts}
	var buf [4]byte
	for _, tiles := range cases {
		if _, err := appendTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendTileInfoPayload err=%v want ErrInvalidFrame", err)
		}
	}
}

func TestAppendTileGroupPayloadSingleTile(t *testing.T) {
	tiles := TileInfo{
		Cols: 1,
		Rows: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}}
	var buf [8]byte
	size, err := TileGroupPayloadSize(tiles, 0, 0, payloads[:])
	if err != nil {
		t.Fatalf("TileGroupPayloadSize: %v", err)
	}
	out, err := AppendTileGroupPayload(buf[:0], tiles, 0, 0, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	if len(out) != size || len(out) != 1 || out[0] != 0x80 {
		t.Fatalf("payload=% x size=%d", out, size)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 0, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	var spans [1]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 1 || spans[0].Offset != 0 || spans[0].Size != 1 {
		t.Fatalf("spans=%+v n=%d group=%+v", spans, n, group)
	}
}

func TestAppendTileGroupPayloadMultiTileRoundTrip(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 2,
	}
	payloads := [...]TilePayload{
		{Data: []byte{0x11, 0x12}},
		{Data: []byte{0x21, 0x22, 0x23}},
		{Data: []byte{0x31}},
		{Data: []byte{0x41, 0x42, 0x43, 0x44}},
	}
	var buf [32]byte
	size, err := TileGroupPayloadSize(tiles, 0, 3, payloads[:])
	if err != nil {
		t.Fatalf("TileGroupPayloadSize: %v", err)
	}
	out, err := AppendTileGroupPayload(buf[:0], tiles, 0, 3, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 0, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if group.TileStartAndEndPresent || group.DataOffset != 1 || group.TileCount != 4 || !group.Final {
		t.Fatalf("group=%+v", group)
	}
	var spans [4]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != len(payloads) {
		t.Fatalf("span count=%d want %d", n, len(payloads))
	}
	for i := range payloads {
		span := spans[i]
		got := out[span.Offset : span.Offset+span.Size]
		if string(got) != string(payloads[i].Data) || span.Tile != uint16(i) {
			t.Fatalf("span %d=%+v data=% x want % x", i, span, got, payloads[i].Data)
		}
	}
}

func TestAppendTileGroupPayloadSubsetRoundTrip(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          2,
		Log2Cols:      1,
		Log2Rows:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{
		{Data: []byte{0x51, 0x52}},
		{Data: []byte{0x61}},
	}
	var buf [16]byte
	out, err := AppendTileGroupPayload(buf[:0], tiles, 1, 2, payloads[:])
	if err != nil {
		t.Fatalf("AppendTileGroupPayload: %v", err)
	}
	group, err := parser.ParseTileGroupHeader(out, tiles, 0, 1, false)
	if err != nil {
		t.Fatalf("ParseTileGroupHeader: %v", err)
	}
	if !group.TileStartAndEndPresent || group.StartTile != 1 || group.EndTile != 2 || group.DataOffset != 1 {
		t.Fatalf("group=%+v", group)
	}
	var spans [2]parser.TileSpan
	n, err := parser.SplitTileGroup(out, tiles, group, spans[:])
	if err != nil {
		t.Fatalf("SplitTileGroup: %v", err)
	}
	if n != 2 || spans[0].Tile != 1 || spans[1].Tile != 2 || spans[0].Size != 2 || spans[1].Size != 1 {
		t.Fatalf("spans=%+v n=%d", spans, n)
	}
}

func TestAppendTileGroupPayloadRejectsInvalid(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	valid := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	empty := [...]TilePayload{{Data: []byte{}}, {Data: []byte{0x81}}}
	tooLargeForSizeByte := [...]TilePayload{{Data: make([]byte, 257)}, {Data: []byte{0x81}}}
	cases := []struct {
		name     string
		tiles    TileInfo
		start    uint16
		end      uint16
		payloads []TilePayload
	}{
		{name: "range", tiles: tiles, start: 1, end: 0, payloads: valid[:]},
		{name: "count", tiles: tiles, start: 0, end: 1, payloads: valid[:1]},
		{name: "empty", tiles: tiles, start: 0, end: 1, payloads: empty[:]},
		{name: "size", tiles: tiles, start: 0, end: 1, payloads: tooLargeForSizeByte[:]},
	}
	var buf [8]byte
	for _, tc := range cases {
		if _, err := AppendTileGroupPayload(buf[:0], tc.tiles, tc.start, tc.end, tc.payloads); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("%s err=%v want ErrInvalidFrame", tc.name, err)
		}
	}
}

func TestAppendTileGroupPayloadShortBuffer(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	var buf [1]byte
	dst := buf[:1]
	dst[0] = 0xee
	out, err := AppendTileGroupPayload(dst, tiles, 0, 1, payloads[:])
	if !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
	if len(out) != len(dst) || out[0] != 0xee {
		t.Fatalf("short buffer mutated output=% x", out)
	}
}

func TestAppendTileGroupPayloadAllocs(t *testing.T) {
	tiles := TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]TilePayload{{Data: []byte{0x80}}, {Data: []byte{0x81}}}
	var buf [8]byte
	if _, err := AppendTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:]); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = TileGroupPayloadSize(tiles, 0, 1, payloads[:])
		_, _ = AppendTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:])
	})
	if allocs != 0 {
		t.Fatalf("AppendTileGroupPayload allocated: %f", allocs)
	}
}

func appendAndParseTileInfo(t *testing.T, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) ([]byte, parser.TileInfo) {
	t.Helper()
	var buf [16]byte
	payload, err := appendTileInfoPayload(buf[:0], seq, prefix, codedWidth, height, tiles)
	if err != nil {
		t.Fatalf("appendTileInfoPayload: %v", err)
	}
	parsed, err := parser.ParseTileInfo(
		payload,
		parser.SequenceHeader{
			Use128x128Superblock: seq.Use128x128Superblock,
			EnableOrderHint:      seq.EnableOrderHint,
			EnableRefFrameMVS:    seq.EnableRefFrameMVS,
		},
		parser.FrameHeaderPrefix{
			FrameType:          parser.FrameType(prefix.FrameType),
			ForceIntegerMV:     prefix.ForceIntegerMV,
			ErrorResilientMode: prefix.ErrorResilientMode,
			DisableCDFUpdate:   prefix.DisableCDFUpdate,
		},
		parser.FrameSize{CodedWidth: codedWidth, Height: height},
	)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	return payload, parsed
}

func tileInfoSequenceHeader() SequenceHeader {
	return SequenceHeader{
		Profile:              Profile0,
		OperatingPointsCount: 1,
		OperatingPoints: [32]SequenceOperatingPoint{
			{SeqLevelIdx: SequenceLevelMax},
		},
		MaxFrameWidth:        64,
		MaxFrameHeight:       64,
		Use128x128Superblock: false,
		ColorConfig: SequenceColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func TestAppendTransformReferenceParamsPayloadLosslessIntraReadsNoBits(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	params := TransformReferenceParams{TransformMode: TransformMode4x4Only, ReferenceMode: ReferenceModeSingle}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, true, params)
	if len(payload) != 0 {
		t.Fatalf("payload len=%d want 0", len(payload))
	}
	if parsed.TransformMode != parser.TransformMode4x4Only || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 0 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadLargestAndSwitchable(t *testing.T) {
	cases := [...]struct {
		name string
		mode TransformMode
		want parser.TransformMode
	}{
		{name: "largest", mode: TransformModeLargest, want: parser.TransformModeLargest},
		{name: "switchable", mode: TransformModeSwitchable, want: parser.TransformModeSwitchable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
			params := TransformReferenceParams{TransformMode: tc.mode, ReferenceMode: ReferenceModeSingle}
			payload, parsed := appendAndParseTransformReferenceParams(t, prefix, false, params)
			if len(payload) != 1 {
				t.Fatalf("payload len=%d want 1", len(payload))
			}
			if parsed.TransformMode != tc.want || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 1 {
				t.Fatalf("parsed transform/reference=%+v want mode=%d bits=1", parsed, tc.want)
			}
		})
	}
}

func TestAppendTransformReferenceParamsPayloadInterReferenceMode(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := TransformReferenceParams{TransformMode: TransformMode4x4Only, ReferenceMode: ReferenceModeSelect}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, true, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.TransformMode != parser.TransformMode4x4Only || parsed.ReferenceMode != parser.ReferenceModeSelect || parsed.BitsRead != 1 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadInterSwitchableSingleReference(t *testing.T) {
	prefix := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	params := TransformReferenceParams{TransformMode: TransformModeSwitchable, ReferenceMode: ReferenceModeSingle}
	payload, parsed := appendAndParseTransformReferenceParams(t, prefix, false, params)
	if len(payload) != 1 {
		t.Fatalf("payload len=%d want 1", len(payload))
	}
	if parsed.TransformMode != parser.TransformModeSwitchable || parsed.ReferenceMode != parser.ReferenceModeSingle || parsed.BitsRead != 2 {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestAppendTransformReferenceParamsPayloadRejectsInvalid(t *testing.T) {
	intra := FrameHeaderPrefix{FrameType: FrameHeaderTypeKey}
	inter := FrameHeaderPrefix{FrameType: FrameHeaderTypeInter}
	cases := [...]struct {
		prefix      FrameHeaderPrefix
		allLossless bool
		params      TransformReferenceParams
	}{
		{prefix: intra, allLossless: true, params: TransformReferenceParams{TransformMode: TransformModeLargest}},
		{prefix: intra, allLossless: false, params: TransformReferenceParams{TransformMode: TransformMode4x4Only}},
		{prefix: intra, allLossless: false, params: TransformReferenceParams{TransformMode: TransformModeLargest, ReferenceMode: ReferenceModeSelect}},
		{prefix: inter, allLossless: false, params: TransformReferenceParams{TransformMode: TransformModeSwitchable, ReferenceMode: ReferenceModeCompound}},
		{prefix: inter, allLossless: false, params: TransformReferenceParams{TransformMode: TransformMode(9), ReferenceMode: ReferenceModeSingle}},
	}
	var buf [2]byte
	for _, tc := range cases {
		if _, err := appendTransformReferenceParamsPayload(buf[:0], tc.prefix, tc.allLossless, tc.params); !errors.Is(err, ErrInvalidFrame) {
			t.Fatalf("appendTransformReferenceParamsPayload(%+v,%v,%+v) err=%v want ErrInvalidFrame", tc.prefix, tc.allLossless, tc.params, err)
		}
	}
}

func appendAndParseTransformReferenceParams(t *testing.T, prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) ([]byte, parser.TransformReferenceParams) {
	t.Helper()
	var buf [2]byte
	payload, err := appendTransformReferenceParamsPayload(buf[:0], prefix, allLossless, params)
	if err != nil {
		t.Fatalf("appendTransformReferenceParamsPayload: %v", err)
	}
	parsed, err := parser.ParseTransformReferenceParams(
		payload,
		parser.FrameHeaderPrefix{FrameType: parser.FrameType(prefix.FrameType)},
		parser.SegmentationParams{AllLossless: allLossless},
		parser.RestorationParams{},
	)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	return payload, parsed
}

// TestVideoEncoderOverlapDeterminism proves the backgrounded in-loop filter
// pass cannot change the emitted stream: an encoder whose reconstruction is
// polled after every frame (forcing the filter join before the next encode
// begins) and one that is never polled (maximal overlap) must produce
// byte-identical temporal units, across P-frames, a mid-stream keyframe,
// and golden refreshes.
func TestVideoEncoderOverlapDeterminism(t *testing.T) {
	const w, h, n = 320, 192, 24
	rng := rand.New(rand.NewSource(7))
	frames := make([]SourceFrame420, n)
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = byte(rng.Intn(256))
	}
	for i := range frames {
		f := SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		dx := (i * 2) % 32
		for y := 0; y < h; y++ {
			copy(f.Y[y*w:y*w+w-dx], bg[y*w+dx:])
		}
		for j := range f.U {
			f.U[j] = 110
			f.V[j] = 140
		}
		frames[i] = f
	}
	newEnc := func() *VideoEncoder {
		e, err := NewVideoEncoder(w, h, 160)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = e.Close() })
		e.SetGoldenInterval(8)
		return e
	}
	polled, free := newEnc(), newEnc()
	for i, f := range frames {
		forceKey := i == 13
		tuA, keyA, errA := polled.Encode(f, forceKey)
		// Polling the reconstruction joins the filter pass immediately.
		ra := polled.Recon()
		if ra.Y == nil {
			t.Fatalf("frame %d: polled recon unavailable", i)
		}
		tuB, keyB, errB := free.Encode(f, forceKey)
		if errA != nil || errB != nil {
			t.Fatalf("frame %d: encode errors %v / %v", i, errA, errB)
		}
		if keyA != keyB || !bytes.Equal(tuA, tuB) {
			t.Fatalf("frame %d: temporal units diverge (key %v/%v, %d/%d bytes)", i, keyA, keyB, len(tuA), len(tuB))
		}
	}
	// Both end states expose identical reconstructions.
	ra, rb := polled.Recon(), free.Recon()
	if !bytes.Equal(ra.Y, rb.Y) || !bytes.Equal(ra.U, rb.U) || !bytes.Equal(ra.V, rb.V) {
		t.Fatal("final reconstructions diverge")
	}
}

func TestVideoEncoderSingleThreadFiltersStayInline(t *testing.T) {
	const w, h = 320, 192
	makeFrame := func(tick int) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f.Y[y*w+x] = uint8(48 + (x/5+y/7+tick)%96)
			}
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 800_000,
		FramesPerSecond:     30,
		MinQIndex:           160,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	enc.SetMaxThreads(1)
	if err := enc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	if _, key, err := enc.Encode(makeFrame(0), true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	if _, key, err := enc.Encode(makeFrame(1), false); err != nil {
		t.Fatal(err)
	} else if key {
		t.Fatal("P-frame was coded as keyframe")
	}
	if !enc.lf.bound || !enc.cdefApp.bound {
		t.Fatal("filters did not bind during single-thread encode")
	}
	if enc.filterStarted || enc.tileWorkers != 0 || enc.lf.started || enc.cdefApp.started || enc.hme.started {
		t.Fatalf("single-thread mode started workers: filter=%v tile=%d lf=%v cdef=%v hme=%v",
			enc.filterStarted, enc.tileWorkers, enc.lf.started, enc.cdefApp.started, enc.hme.started)
	}
}

func TestVideoEncoderCloseWaitsForWavefrontWorkers(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 80)
	if err != nil {
		t.Fatal(err)
	}
	enc.wavefront.ensureWorkers(3, &enc.pc)
	leaf := enc.ensureLeaf()
	leaf.wavefront.ensureWorkers(2, &leaf.pc)
	baseStarted := enc.wavefront.started
	leafStarted := leaf.wavefront.started
	if baseStarted != 3 || leafStarted != 2 {
		t.Fatalf("wavefront helpers did not start: base=%d leaf=%d", baseStarted, leafStarted)
	}
	t.Cleanup(func() {
		enc.wavefront.close()
		leaf.wavefront.close()
	})

	waitForHelpers := func(wg *sync.WaitGroup) <-chan struct{} {
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()
		return done
	}
	baseDone := waitForHelpers(&enc.wavefront.workerWG)
	leafDone := waitForHelpers(&leaf.wavefront.workerWG)
	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	for _, pool := range []struct {
		name    string
		done    <-chan struct{}
		wf      *pframeWavefront
		started int
	}{
		{name: "base", done: baseDone, wf: &enc.wavefront, started: baseStarted},
		{name: "leaf", done: leafDone, wf: &leaf.wavefront, started: leafStarted},
	} {
		select {
		case <-pool.done:
		case <-time.After(5 * time.Second):
			t.Fatalf("%s wavefront helpers did not exit after Close", pool.name)
		}
		if pool.wf.work != nil || pool.wf.started != 0 || len(pool.wf.workers) != 0 {
			t.Fatalf("%s wavefront retained workers: work=%v started=%d workers=%d", pool.name, pool.wf.work != nil, pool.wf.started, len(pool.wf.workers))
		}
		if pool.wf.coder != nil || pool.wf.src.Y != nil || pool.wf.ref.Y != nil || pool.wf.golden != nil || pool.wf.recon != nil {
			t.Fatalf("%s wavefront retained frame state", pool.name)
		}
		if pool.started == 0 {
			t.Fatalf("%s wavefront test did not exercise helpers", pool.name)
		}
	}

	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVideoEncoderCloseWaitsForHMEWorkers(t *testing.T) {
	enc, err := NewVideoEncoder(64, 64, 80)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = enc.Close() })
	enc.hme.startWorkers()
	if !enc.hme.started {
		t.Fatal("HME workers did not start")
	}

	exited := make(chan struct{})
	go func() {
		enc.hme.workerWG.Wait()
		close(exited)
	}()

	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("HME workers did not exit after Close")
	}
	if enc.hme.work != nil || enc.hme.done != nil || enc.hme.started {
		t.Fatalf("Close left HME workers active: work=%v done=%v started=%v",
			enc.hme.work != nil, enc.hme.done != nil, enc.hme.started)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestVideoEncoderCloseWaitsForOuterWorkers(t *testing.T) {
	for _, wantFilterErr := range []bool{false, true} {
		name := "success"
		if wantFilterErr {
			name = "filter error"
		}
		t.Run(name, func(t *testing.T) {
			enc, err := NewVideoEncoder(64, 64, 80)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = enc.Close() })

			enc.startTileWorkers(2)
			enc.startFilterWorker()
			enc.startLeafWorker()

			helpersDone := make(chan struct{})
			go func() {
				enc.tileWorkerWG.Wait()
				enc.filterWorkerWG.Wait()
				enc.leaf.workerWG.Wait()
				close(helpersDone)
			}()

			var filterErr error
			if wantFilterErr {
				filterErr = errors.New("test filter error")
				enc.filterErr = filterErr
			}
			if err := enc.Close(); err != filterErr {
				t.Fatalf("Close error = %v, want %v", err, filterErr)
			}
			select {
			case <-helpersDone:
			case <-time.After(5 * time.Second):
				t.Fatal("outer workers did not exit after Close")
			}
			if enc.tileWork != nil || enc.tileWorkers != 0 ||
				enc.filterWork != nil || enc.filterDone != nil || enc.filterStarted ||
				enc.leaf.work != nil || enc.leaf.done != nil || enc.leaf.started {
				t.Fatal("Close left outer worker state active")
			}
			if err := enc.Close(); err != nil {
				t.Fatalf("second Close: %v", err)
			}
		})
	}
}

func TestVideoEncoderCloseReleasesPersistentWorkers(t *testing.T) {
	const w, h = 320, 192
	f0 := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	f1 := SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, w*h/4),
		V:            make([]byte, w*h/4),
		YStride:      w,
		ChromaStride: w / 2,
		Width:        w,
		Height:       h,
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			f0.Y[y*w+x] = uint8(48 + (x/5+y/7)%96)
			f1.Y[y*w+x] = uint8(48 + ((x+3)/5+y/7)%96)
		}
	}
	for i := range f0.U {
		f0.U[i], f0.V[i] = 120, 130
		f1.U[i], f1.V[i] = 120, 130
	}
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: 800_000,
		FramesPerSecond:     30,
		MinQIndex:           160,
		MaxQIndex:           200,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, key, err := enc.Encode(f0, true); err != nil {
		t.Fatal(err)
	} else if !key {
		t.Fatal("initial frame was not a keyframe")
	}
	if _, key, err := enc.Encode(f1, false); err != nil {
		t.Fatal(err)
	} else if key {
		t.Fatal("P-frame was coded as keyframe")
	}
	if err := enc.Flush(); err != nil {
		t.Fatal(err)
	}
	// No deblocking at SVT preset 12 rtc: the loop-filter workers stay
	// parked, CDEF runs.
	if !enc.hme.started || !enc.cdefApp.started {
		t.Fatalf("expected persistent workers to start: hme=%v cdef=%v", enc.hme.started, enc.cdefApp.started)
	}
	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	if enc.hme.started || enc.lf.started || enc.cdefApp.started || enc.hme.work != nil || enc.lf.work != nil || enc.cdefApp.work != nil {
		t.Fatalf("close left workers active: hme=%v/%v lf=%v/%v cdef=%v/%v",
			enc.hme.started, enc.hme.work != nil,
			enc.lf.started, enc.lf.work != nil,
			enc.cdefApp.started, enc.cdefApp.work != nil)
	}
}

// TestPipelineOverlapFires confirms the overlap-eligible configuration actually
// exercises the concurrent leaf/base path (rather than silently falling back to
// the serial FIFO), so the external byte-identity oracle is testing real
// concurrency. It also checks that golden-enabled streams do NOT
// overlap.
func TestPipelineOverlapFires(t *testing.T) {
	const w, h = 192, 128
	newEnc := func(golden int) *VideoEncoder {
		// Fixed q: SVT's shared CBR rate control needs each frame's
		// feedback before the next q, so CBR streams code serially.
		enc, err := NewVideoEncoder(w, h, 120)
		if err != nil {
			t.Fatal(err)
		}
		if err := enc.SetTemporalLayers(2); err != nil {
			t.Fatal(err)
		}
		enc.SetGoldenInterval(golden)
		return enc
	}
	run := func(enc *VideoEncoder, frames int) {
		defer enc.Close()
		if err := enc.SetThroughputPipelining(true); err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(3))
		bg := make([]byte, w*h)
		for i := range bg {
			bg[i] = uint8(70 + rng.Intn(50))
		}
		for i := 0; i < frames; i++ {
			f := SourceFrame420{
				Y: append([]byte(nil), bg...), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
				YStride: w, ChromaStride: w / 2, Width: w, Height: h,
			}
			for j := range f.U {
				f.U[j] = 121
				f.V[j] = 129
			}
			for y := 10 + i*3; y < 10+i*3+20 && y < h; y++ {
				for x := 8 + i*2; x < 8+i*2+20 && x < w; x++ {
					f.Y[y*w+x] = 210
				}
			}
			if _, _, _, err := enc.EncodeThroughput(f, false); err != nil {
				t.Fatal(err)
			}
		}
		if _, _, _, err := enc.Drain(); err != nil {
			t.Fatal(err)
		}
	}

	const frames = 12
	eligible := newEnc(0)
	run(eligible, frames)
	if eligible.pipeOverlaps == 0 {
		t.Fatalf("overlap-eligible stream never overlapped (pipeOverlaps=0)")
	}

	golden := newEnc(16)
	run(golden, frames)
	if golden.pipeOverlaps != 0 {
		t.Fatalf("golden-enabled stream overlapped (pipeOverlaps=%d), want 0", golden.pipeOverlaps)
	}
}

// This body is frozen from main 10f4a6ae956e881e8d9e85a4797c55a441cd27c3.
// It is independent of the shorter candidate walk.
func frozenPrewarmOracle(e *VideoEncoder) error {
	color := e.parserColorConfig()
	cw := chromaWidthForColor(e.width, color)
	ch := chromaHeightForColor(e.height, color)
	src := SourceFrame420{
		Y:            make([]byte, e.width*e.height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      e.width,
		ChromaStride: cw,
		Width:        e.renderWidth,
		Height:       e.renderHeight,
	}
	savedQ := e.qIndex
	savedCREnd := e.cr.sbEnd
	savedRC, savedRef := e.svtRC, e.svtRef
	// Size the assembly buffers for real content up front; the throwaway
	// frame below is black and would leave them near-empty.
	if bound := e.width * e.height; cap(e.tuScratch) < bound {
		e.tuScratch = make([]byte, 0, bound)
		e.tuGroup = make([]byte, 0, bound)
	}
	if _, _, err := e.Encode(src, true); err != nil {
		return err
	}
	// Exercise every reconstruction buffer and the golden-refresh cadence so
	// delayed scratch growth happens before the first externally visible frame.
	frames := e.temporalLayers
	if frames < 2 {
		frames = 2
	} else {
		frames *= 2
	}
	if e.goldenEvery > 0 && frames < e.goldenEvery+1 {
		frames = e.goldenEvery + 1
	}
	for i := 0; i < frames; i++ {
		if _, _, err := e.Encode(src, false); err != nil {
			return err
		}
	}
	if err := e.joinFilter(); err != nil {
		return err
	}
	e.prewarmPipeline(src)
	e.haveKey = false
	e.haveCtx = false
	e.haveCtxT1 = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.cr.sbEnd = savedCREnd
	e.svtRC, e.svtRef = savedRC, savedRef
	e.svtCurOn = false
	e.sinceGoldenFresh = 0
	e.hme.havePrev = false
	e.hme.fme.valid = false
	e.hme.svt.haveCur, e.hme.svt.havePrev = false, false
	e.lastRecon = SourceFrame420{}
	return nil
}

func prewarmStateForOracle(e *VideoEncoder) any {
	return struct {
		HaveKey, HaveCtx, HaveCtxT1, HMEArmed        bool
		FrameIndex, ReconIdx, SinceGolden, LowMotion int
		LastTemporalID, QIndex                       uint8
		RC                                           svtRateControl
		Ref                                          [2]svtRCRef
		Stats                                        EncoderDecisionStats
	}{e.haveKey, e.haveCtx, e.haveCtxT1, e.hme.havePrev,
		e.frameIndex, e.reconIdx, e.sinceGoldenFresh, e.avgFrameLowMotion,
		e.lastTemporalID, e.qIndex, e.svtRC, e.svtRef, e.decisionStats}
}

func equalPrewarmRecon(a, b SourceFrame420) bool {
	return a.Width == b.Width && a.Height == b.Height && a.YStride == b.YStride &&
		a.ChromaStride == b.ChromaStride && bytes.Equal(a.Y, b.Y) && bytes.Equal(a.U, b.U) && bytes.Equal(a.V, b.V)
}

func TestVideoEncoderPrewarmLegacyOracle(t *testing.T) {
	cases := []struct {
		name               string
		q                  uint8
		golden, layers     int
		cbr, reused, stats bool
	}{
		{name: "fresh_cbr_default_golden", q: 100, golden: 16, cbr: true},
		{name: "fresh_fixed_high_q", q: 200, golden: 16},
		{name: "fresh_even_recon_parity", q: 80, golden: 17},
		{name: "golden_disabled_fallback", q: 100, golden: 0, cbr: true},
		{name: "temporal_state_fallback", q: 100, golden: 16, layers: 2, cbr: true},
		{name: "reused_state_fallback", q: 100, golden: 16, cbr: true, reused: true},
		{name: "decision_stats_fallback", q: 100, golden: 16, cbr: true, stats: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			makeEnc := func() *VideoEncoder {
				e, err := NewVideoEncoder(192, 128, tc.q)
				if err != nil {
					t.Fatal(err)
				}
				e.SetMaxThreads(1)
				e.SetGoldenInterval(tc.golden)
				if tc.cbr {
					if err = e.SetRateControlConfig(RateControlConfig{TargetBitsPerSecond: 500000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200}); err != nil {
						t.Fatal(err)
					}
				}
				if tc.layers > 1 {
					if err = e.SetTemporalLayers(tc.layers); err != nil {
						t.Fatal(err)
					}
				}
				e.SetDecisionStatsEnabled(tc.stats)
				t.Cleanup(func() { _ = e.Close() })
				return e
			}
			got, want := makeEnc(), makeEnc()
			if tc.reused {
				for n := 0; n < 3; n++ {
					f := lfDiffFrame(192, 128)
					for _, e := range []*VideoEncoder{got, want} {
						if _, _, err := e.Encode(f, false); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			if err := got.Prewarm(); err != nil {
				t.Fatal(err)
			}
			if err := frozenPrewarmOracle(want); err != nil {
				t.Fatal(err)
			}
			gs, ws := prewarmStateForOracle(got), prewarmStateForOracle(want)
			if !reflect.DeepEqual(gs, ws) {
				t.Fatalf("retained state differs: got %+v; want %+v", gs, ws)
			}
			if !equalPrewarmRecon(got.Recon(), want.Recon()) {
				t.Fatal("returned reconstruction differs before first real frame")
			}
			for n := 0; n < 20; n++ {
				f := lfDiffFrame(192, 128)
				for y := 0; y < 128; y++ {
					for x := 0; x < 192; x++ {
						f.Y[y*192+x] = byte(40 + ((x+n*3)/5+y/3)%150)
					}
				}
				gb, gk, ge := got.Encode(f, n == 11)
				wb, wk, we := want.Encode(f, n == 11)
				if ge != nil || we != nil {
					t.Fatalf("frame %d errors %v/%v", n, ge, we)
				}
				if gk != wk || !bytes.Equal(gb, wb) {
					t.Fatalf("frame %d encoded bytes differ, sizes %d/%d", n, len(gb), len(wb))
				}
				if !equalPrewarmRecon(got.Recon(), want.Recon()) {
					t.Fatalf("frame %d reconstruction differs", n)
				}
				if !reflect.DeepEqual(prewarmStateForOracle(got), prewarmStateForOracle(want)) {
					t.Fatalf("frame %d retained state differs", n)
				}
			}
		})
	}
}

func TestVideoEncoderPrewarmFirstFramesAllocs(t *testing.T) {
	// AllocsPerRun calls the closure once before measuring. Each invocation uses
	// a different already-prewarmed encoder so the measured frames are still its
	// first externally visible keyframe and both P reconstruction buffers.
	encs := make([]*VideoEncoder, 4)
	for i := range encs {
		e, err := NewVideoEncoderCBR(192, 128, RateControlConfig{TargetBitsPerSecond: 500000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200})
		if err != nil {
			t.Fatal(err)
		}
		e.SetMaxThreads(1)
		if err = e.Prewarm(); err != nil {
			t.Fatal(err)
		}
		encs[i] = e
		t.Cleanup(func() { _ = e.Close() })
	}
	f := lfDiffFrame(192, 128)
	i := 0
	if n := testing.AllocsPerRun(3, func() {
		e := encs[i]
		i++
		for frame := 0; frame < 3; frame++ {
			if _, _, err := e.Encode(f, false); err != nil {
				t.Fatal(err)
			}
		}
	}); n != 0 {
		t.Fatalf("first three frames allocated %g times", n)
	}
}

func TestRateControlTemporalLayerPerFrameBitsMatchesLibaomRTC(t *testing.T) {
	const targetBPS = 1_000_000
	const fps = 30
	twoLayer := [...]int{40_000, 26_667}
	for temporalID, want := range twoLayer {
		got := rateControlTemporalLayerPerFrameBits(targetBPS, fps, 2, uint8(temporalID), 0)
		if got != want {
			t.Fatalf("L1T2 temporal layer %d per-frame bits=%d want %d", temporalID, got, want)
		}
	}
	threeLayer := [...]int{66_667, 26_667, 20_000}
	for temporalID, want := range threeLayer {
		got := rateControlTemporalLayerPerFrameBits(targetBPS, fps, 3, uint8(temporalID), 0)
		if got != want {
			t.Fatalf("L1T3 temporal layer %d per-frame bits=%d want %d", temporalID, got, want)
		}
	}
}

type rcControllerSnapshot struct {
	qIndex              uint8
	buffer              int
	recent              [2]int
	temporalQ           [WebRTCMaxTemporalLayers]uint8
	temporalBuffer      [WebRTCMaxTemporalLayers]int
	temporalRecentBits  [WebRTCMaxTemporalLayers][2]int
	temporalPerFrameBit [WebRTCMaxTemporalLayers]int
	perFrameBits        int
}

func TestUpdateRateControlConfigPreservesControllerState(t *testing.T) {
	oldRC := RateControlConfig{
		TargetBitsPerSecond: 1_200_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}
	nextRC := RateControlConfig{
		TargetBitsPerSecond: 900_000,
		FramesPerSecond:     60,
		MinQIndex:           30,
		MaxQIndex:           180,
	}
	for _, tc := range []struct {
		name string
		run  func(*testing.T)
	}{
		{
			name: "i400",
			run: func(t *testing.T) {
				enc, err := NewMonochromeVideoEncoderCBR(64, 64, oldRC)
				if err != nil {
					t.Fatalf("NewMonochromeVideoEncoderCBR: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
		{
			name: "i400-10",
			run: func(t *testing.T) {
				enc, err := NewHighBitDepthMonochromeVideoEncoderCBR(64, 64, 10, oldRC)
				if err != nil {
					t.Fatalf("NewHighBitDepthMonochromeVideoEncoderCBR: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
		{
			name: "i420-10",
			run: func(t *testing.T) {
				enc, err := newHighBitDepth420VideoEncoderCBRWithColor(64, 64, Profile0, SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}, oldRC)
				if err != nil {
					t.Fatalf("newHighBitDepth420VideoEncoderCBRWithColor: %v", err)
				}
				if err := enc.SetTemporalLayers(3); err != nil {
					t.Fatalf("SetTemporalLayers: %v", err)
				}
				seedVideoRCState(&enc.qIndex, &enc.rcBuffer, &enc.rcRecentBits, &enc.rcTemporalQ, &enc.rcTemporalBuffer, &enc.rcTemporalRecentBits)
				before := snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits)
				if err := enc.UpdateRateControlConfig(nextRC); err != nil {
					t.Fatalf("UpdateRateControlConfig: %v", err)
				}
				assertPreservedRCUpdate(t, before, snapshotVideoRC(enc.qIndex, enc.rcBuffer, enc.rcRecentBits, enc.rcTemporalQ, enc.rcTemporalBuffer, enc.rcTemporalRecentBits, enc.rcTemporalPerFrameBits, enc.rcPerFrameBits), nextRC, enc.temporalLayers)
			},
		},
	} {
		t.Run(tc.name, tc.run)
	}
}

func TestRateControlTemporalLayerQClampMatchesAdjustQCBR(t *testing.T) {
	qs := [WebRTCMaxTemporalLayers]uint8{80, 70, 60}
	for _, temporalID := range []uint8{1, 2} {
		got := rateControlTemporalLayerQIndex(qs, 20, 200, 3, temporalID)
		if got != 76 {
			t.Fatalf("temporal layer %d q=%d want TL0-4 clamp 76", temporalID, got)
		}
	}
	qs[0] = 22
	qs[1] = 1
	if got := rateControlTemporalLayerQIndex(qs, 20, 200, 2, 1); got != 20 {
		t.Fatalf("min-clamped temporal q=%d want 20", got)
	}
}

func seedVideoRCState(q *uint8, buffer *int, recent *[2]int, temporalQ *[WebRTCMaxTemporalLayers]uint8, temporalBuffer *[WebRTCMaxTemporalLayers]int, temporalRecent *[WebRTCMaxTemporalLayers][2]int) {
	*q = 92
	*buffer = -1234
	*recent = [2]int{111, 222}
	*temporalQ = [WebRTCMaxTemporalLayers]uint8{92, 96, 104}
	*temporalBuffer = [WebRTCMaxTemporalLayers]int{1400, -900, 300}
	*temporalRecent = [WebRTCMaxTemporalLayers][2]int{{11, 12}, {21, 22}, {31, 32}}
}

func snapshotVideoRC(q uint8, buffer int, recent [2]int, temporalQ [WebRTCMaxTemporalLayers]uint8, temporalBuffer [WebRTCMaxTemporalLayers]int, temporalRecent [WebRTCMaxTemporalLayers][2]int, temporalPerFrameBits [WebRTCMaxTemporalLayers]int, perFrameBits int) rcControllerSnapshot {
	return rcControllerSnapshot{
		qIndex:              q,
		buffer:              buffer,
		recent:              recent,
		temporalQ:           temporalQ,
		temporalBuffer:      temporalBuffer,
		temporalRecentBits:  temporalRecent,
		temporalPerFrameBit: temporalPerFrameBits,
		perFrameBits:        perFrameBits,
	}
}

func assertPreservedRCUpdate(t *testing.T, before rcControllerSnapshot, after rcControllerSnapshot, rc RateControlConfig, temporalLayers int) {
	t.Helper()
	wantPerFrame, err := rateControlPerFrameBits(rc)
	if err != nil {
		t.Fatalf("rateControlPerFrameBits: %v", err)
	}
	if after.qIndex != before.qIndex || after.buffer != before.buffer || after.recent != before.recent ||
		after.temporalQ != before.temporalQ || after.temporalBuffer != before.temporalBuffer ||
		after.temporalRecentBits != before.temporalRecentBits {
		t.Fatalf("controller state not preserved: before=%+v after=%+v", before, after)
	}
	if after.perFrameBits != wantPerFrame {
		t.Fatalf("per-frame bits=%d want %d", after.perFrameBits, wantPerFrame)
	}
	for temporalID := 0; temporalID < temporalLayers; temporalID++ {
		wantLayerBits := rateControlTemporalLayerPerFrameBits(rc.TargetBitsPerSecond, rc.FramesPerSecond, temporalLayers, uint8(temporalID), wantPerFrame)
		if got := after.temporalPerFrameBit[temporalID]; got != wantLayerBits {
			t.Fatalf("temporal layer %d per-frame bits=%d want %d", temporalID, got, wantLayerBits)
		}
	}
}

func TestUpdateRateControlConfigReinitializesSVTRateControl(t *testing.T) {
	// SVT-AV1 restarts its rate control on a rate change event
	// (seq_param_changed -> set_rc_buffer_sizes + svt_av1_rc_init).
	enc, err := NewVideoEncoderCBR(64, 64, RateControlConfig{
		TargetBitsPerSecond: 1_200_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoderCBR: %v", err)
	}
	enc.svtRC.bufferLevel = -1234
	enc.svtRC.rateCorrection[svtRCInterNormal] = 3
	next := RateControlConfig{TargetBitsPerSecond: 900_000, FramesPerSecond: 60, MinQIndex: 30, MaxQIndex: 180}
	if err := enc.UpdateRateControlConfig(next); err != nil {
		t.Fatalf("UpdateRateControlConfig: %v", err)
	}
	rc := &enc.svtRC
	if rc.avgFrameBandwidth != 15_000 || rc.bestQuality != 30 || rc.worstQuality != 180 {
		t.Fatalf("rc after update: bandwidth=%d q=[%d,%d]", rc.avgFrameBandwidth, rc.bestQuality, rc.worstQuality)
	}
	if rc.bufferLevel != rc.startingBufferLevel || rc.startingBufferLevel != 900_000*600/1000 {
		t.Fatalf("buffer %d starting %d", rc.bufferLevel, rc.startingBufferLevel)
	}
	if rc.rateCorrection[svtRCInterNormal] != 0.7 || enc.rcPerFrameBits != 15_000 {
		t.Fatalf("rate correction %v per-frame %d", rc.rateCorrection[svtRCInterNormal], enc.rcPerFrameBits)
	}
}

func TestSVTRateControlTargetsAndQ(t *testing.T) {
	var rc svtRateControl
	rc.init(1_000_000, 30, 16, 255, 1280, 720, rcBuffers{})
	// av1_calc_iframe_target_size_one_pass_cbr: the first key takes half
	// the starting buffer, clamped to max_intra_bitrate_pct.
	key := svtRCFrame{key: true, width: 1280, height: 720}
	rc.target(&key)
	if want := min(int(rc.startingBufferLevel/2), rc.avgFrameBandwidth*3); key.thisFrame != want {
		t.Fatalf("key target %d want %d", key.thisFrame, want)
	}
	// A buffer above optimal raises the inter target by at most 25%/2.
	rc.bufferLevel = rc.optimalBufferLevel * 2
	p := svtRCFrame{flat: true, hierLevels: 2, width: 1280, height: 720}
	rc.target(&p)
	if want := rc.avgFrameBandwidth + rc.avgFrameBandwidth*25/200; p.thisFrame != want {
		t.Fatalf("inter target %d want %d", p.thisFrame, want)
	}
	// More bits per macroblock never pick a higher q.
	lo := svtRCFrame{flat: true, hierLevels: 2, width: 1280, height: 720, thisFrame: 10_000}
	hi := lo
	hi.thisFrame = 200_000
	if qLo, qHi := rc.regulateQ(&lo, 16, 255), rc.regulateQ(&hi, 16, 255); qHi > qLo {
		t.Fatalf("q at 200k bits %d above q at 10k bits %d", qHi, qLo)
	}
	// Non-base layers never go below their reference's q.
	layered := svtRCFrame{hierLevels: 2, temporalLayer: 2, refValid: true, refBaseQ: 120, refLayer: 1, width: 1280, height: 720}
	rc.target(&layered)
	if q := rc.pickQ(&layered); q < 120 {
		t.Fatalf("TL2 q %d below the TL1 reference q 120", q)
	}
	// Feedback: an overshoot drains the buffer and raises the correction.
	before := rc.rateCorrection[svtRCInterNormal]
	buf := rc.bufferLevel
	rc.postencode(&p, 100, p.thisFrame*12, svtCRStats{})
	if rc.bufferLevel >= buf || rc.rateCorrection[svtRCInterNormal] <= before || rc.rc1Frame != -1 {
		t.Fatalf("overshoot feedback: buffer %d->%d rcf %v->%v rc1 %d", buf, rc.bufferLevel, before,
			rc.rateCorrection[svtRCInterNormal], rc.rc1Frame)
	}
}

func TestRCInterQWithoutRateControlKeepsFixedQ(t *testing.T) {
	enc := &VideoEncoder{qIndex: 90, temporalLayers: 3}
	for temporalID := uint8(0); temporalID < 3; temporalID++ {
		if got := enc.rcInterQ(temporalID, svtRefLast, 1, 64, 64, false); got != 90 {
			t.Fatalf("temporal layer %d q=%d want fixed CQP q 90", temporalID, got)
		}
	}
}

func TestSetQIndexDisablesRateControlState(t *testing.T) {
	enc := &VideoEncoder{qIndex: 120, rcEnabled: true, rcPerFrameBits: 1000, rcMinQ: 20, rcMaxQ: 200}
	enc.svtRC.enabled = true
	if err := enc.SetQIndex(37); err != nil {
		t.Fatalf("SetQIndex: %v", err)
	}
	if enc.rcEnabled || enc.qIndex != 37 || enc.rcPerFrameBits != 0 || enc.svtRC.enabled {
		t.Fatalf("encoder state after SetQIndex: rc=%v q=%d", enc.rcEnabled, enc.qIndex)
	}
	if err := enc.SetQIndex(0); err == nil {
		t.Fatal("SetQIndex(0) succeeded")
	}
	if enc.qIndex != 37 || enc.rcEnabled {
		t.Fatalf("invalid SetQIndex mutated state: rc=%v q=%d", enc.rcEnabled, enc.qIndex)
	}
}

// TestWebRTCStreamQRangeReachesSVTRC: a stream's MinQIndex/MaxQIndex must
// bound the qindex SVT's rate control picks (best/worst quality), not only
// the encoder's legacy rcMinQ/rcMaxQ fields.
func TestWebRTCStreamQRangeReachesSVTRC(t *testing.T) {
	const w, h = 192, 128
	const minQ, maxQ = 120, 140
	stream, err := NewWebRTCStreamLayers(w, h, RateControlConfig{
		TargetBitsPerSecond: 5_000_000, // far more than needed: q would fall to the floor
		FramesPerSecond:     30,
		MinQIndex:           minQ,
		MaxQIndex:           maxQ,
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	enc := stream.encoders[0]
	if enc == nil {
		t.Fatal("no 8-bit encoder")
	}
	if got := [2]int{enc.svtRC.bestQuality, enc.svtRC.worstQuality}; got != [2]int{minQ, maxQ} {
		t.Fatalf("svtRC q range = %v, want [%d %d]", got, minQ, maxQ)
	}
	rng := rand.New(rand.NewSource(1))
	cw, ch := w/2, h/2
	for i := range 8 {
		f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch), YStride: w, ChromaStride: cw, Width: w, Height: h}
		for j := range f.Y {
			f.Y[j] = byte(rng.Intn(256))
		}
		if _, err := stream.Encode(f, false); err != nil {
			t.Fatal(err)
		}
		if q := enc.svtCurQ; q < minQ || q > maxQ {
			t.Fatalf("frame %d qindex %d outside [%d, %d]", i, q, minQ, maxQ)
		}
	}
}

func TestWebRTCStreamEncoderOptions(t *testing.T) {
	stream, err := NewWebRTCStreamLayers(192, 128, RateControlConfig{
		TargetBitsPerSecond: 500_000,
		FramesPerSecond:     30,
		MinQIndex:           20,
		MaxQIndex:           200,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	stream.SetGoldenInterval(0)
	if got := stream.encoders[0].goldenEvery; got != 0 {
		t.Fatalf("golden interval=%d want 0", got)
	}
	stream.SetTileColumns(4)
	if got := stream.encoders[0].tileColsLog2; got != 2 {
		t.Fatalf("tileColsLog2=%d want 2", got)
	}
	if err := stream.Prewarm(); err != nil {
		t.Fatalf("prewarm: %v", err)
	}
	if stream.state.NextFrameID != 0 || stream.state.DeltaPictureIndex != 0 || stream.encoders[0].haveKey {
		t.Fatalf("prewarm advanced stream state: frameID=%d deltaPictureIndex=%d haveKey=%v", stream.state.NextFrameID, stream.state.DeltaPictureIndex, stream.encoders[0].haveKey)
	}
}

func TestWebRTCStreamEncoderOptionsSurviveSetConfig(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	stream.SetGoldenInterval(0)
	stream.SetTileColumns(4)

	change := stream.Config()
	change.Scalability = ScalabilityModeS2T2
	change.MinBitrateKbps = 120
	change.MaxBitrateKbps = 1200
	change.TargetBitrateKbps = 760
	change.SpatialLayers[0].MinBitrateKbps = 120
	change.SpatialLayers[0].MaxBitrateKbps = 480
	change.SpatialLayers[0].TargetBitrateKbps = 260
	change.SpatialLayers[1].MinBitrateKbps = 300
	change.SpatialLayers[1].MaxBitrateKbps = 1200
	change.SpatialLayers[1].TargetBitrateKbps = 760
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig simulcast: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		if got := stream.encoders[i].goldenEvery; got != 0 {
			t.Fatalf("simulcast layer %d golden interval=%d want 0", i, got)
		}
		if got := stream.encoders[i].tileColsLog2; got != 2 {
			t.Fatalf("simulcast layer %d tileColsLog2=%d want 2", i, got)
		}
	}

	change = stream.Config()
	change.Scalability = ScalabilityModeL2T2
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig shared SVC: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		if got := stream.encoders[i].goldenEvery; got != 0 {
			t.Fatalf("shared SVC layer %d golden interval=%d want 0", i, got)
		}
		if got := stream.encoders[i].tileColsLog2; got != 2 {
			t.Fatalf("shared SVC layer %d tileColsLog2=%d want 2", i, got)
		}
	}
}

func TestWebRTCStreamConfigMaxThreadsControlsTileColumns(t *testing.T) {
	const w, h = 640, 360
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		MaxThreads:        1,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()
	if got := stream.encoders[0].tileColsLog2; got != 0 {
		t.Fatalf("initial tileColsLog2=%d want 0", got)
	}
	if !stream.encoders[0].singleThread {
		t.Fatal("initial MaxThreads=1 did not enable single-thread mode")
	}
	picture, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("initial EncodePicture: %v", err)
	}
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 1 {
		t.Fatalf("initial encoded tile columns=%d want 1", got)
	}

	change := stream.Config()
	change.MaxThreads = 4
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig MaxThreads=4: %v", err)
	}
	// MaxThreads>1 now drives the intra-tile SB-row wavefront (single tile
	// column, N decision lanes) rather than tile-column parallelism.
	if got := stream.encoders[0].tileColsLog2; got != 0 {
		t.Fatalf("updated tileColsLog2=%d want 0", got)
	}
	if got := stream.encoders[0].wavefrontLanes; got != 4 {
		t.Fatalf("updated wavefrontLanes=%d want 4", got)
	}
	if stream.encoders[0].singleThread {
		t.Fatal("updated MaxThreads=4 left single-thread mode enabled")
	}
	picture, err = stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("updated EncodePicture: %v", err)
	}
	// Keyframes code one tile too, as SVT-AV1's default.
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 1 {
		t.Fatalf("updated encoded tile columns=%d want 1", got)
	}

	change = stream.Config()
	change.MaxThreads = 0
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig MaxThreads=0: %v", err)
	}
	if got, want := stream.encoders[0].tileColsLog2, uint8(0); got != want {
		t.Fatalf("defaulted tileColsLog2=%d want %d", got, want)
	}
	if stream.encoders[0].singleThread {
		t.Fatal("defaulted MaxThreads=0 left single-thread mode enabled")
	}
	picture, err = stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
	if err != nil {
		t.Fatalf("defaulted EncodePicture: %v", err)
	}
	// The default codes one tile, inter frames on a GOMAXPROCS-lane wavefront.
	if procs := runtime.GOMAXPROCS(0); procs > 1 {
		if got := stream.encoders[0].wavefrontLanes; got != procs {
			t.Fatalf("defaulted wavefrontLanes=%d want %d", got, procs)
		}
	}
	if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 1 {
		t.Fatalf("defaulted encoded tile columns=%d want 1", got)
	}
}

func TestWebRTCStreamGoldenIntervalSurvivesMonochromeReplacement(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	stream.SetGoldenInterval(0)

	mono := stream.Config()
	mono.ColorConfigSet = true
	mono.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
	if err := stream.SetConfig(mono); err != nil {
		t.Fatalf("SetConfig monochrome: %v", err)
	}
	if stream.monoEncoders[0] == nil {
		t.Fatal("missing monochrome encoder")
	}
	if got := stream.monoEncoders[0].goldenEvery; got != 0 {
		t.Fatalf("monochrome replacement golden interval=%d want 0", got)
	}

	stream.SetGoldenInterval(7)
	if got := stream.monoEncoders[0].goldenEvery; got != 7 {
		t.Fatalf("monochrome live golden interval=%d want 7", got)
	}
}

func TestWebRTCStreamGoldenIntervalAppliesToHighBitDepthPixelEncoders(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
	}
	for _, tc := range []struct {
		name   string
		config func(Config) Config
		assert func(*testing.T, *WebRTCStream, int)
	}{
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream, want int) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].goldenEvery; got != want {
					t.Fatalf("mono16 golden interval=%d want %d", got, want)
				}
			},
		},
		{
			name: "i400-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream, want int) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].goldenEvery; got != want {
					t.Fatalf("mono16 golden interval=%d want %d", got, want)
				}
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream, want int) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].goldenEvery; got != want {
					t.Fatalf("color16 golden interval=%d want %d", got, want)
				}
			},
		},
		{
			name: "i420-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream, want int) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].goldenEvery; got != want {
					t.Fatalf("color16 golden interval=%d want %d", got, want)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			live, err := NewWebRTCStreamConfig(tc.config(base))
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig live: %v", err)
			}
			defer live.Close()
			live.SetGoldenInterval(7)
			tc.assert(t, live, 7)
			live.SetGoldenInterval(0)
			tc.assert(t, live, 0)

			replacement, err := NewWebRTCStreamConfig(base)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig replacement base: %v", err)
			}
			defer replacement.Close()
			replacement.SetGoldenInterval(0)
			if err := replacement.SetConfig(tc.config(base)); err != nil {
				t.Fatalf("SetConfig replacement: %v", err)
			}
			tc.assert(t, replacement, 0)
		})
	}
}

func TestWebRTCStreamConfigMaxThreadsAppliesToPixelEncoders(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		MaxThreads:        4,
	}
	for _, tc := range []struct {
		name   string
		config func(Config) Config
		assert func(*testing.T, *WebRTCStream)
		encode func(*testing.T, *WebRTCStream) WebRTCEncodedPicture
	}{
		{
			name: "i420-8",
			config: func(cfg Config) Config {
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.encoders[0] == nil {
					t.Fatal("missing i420 encoder")
				}
				if got := stream.encoders[0].tileColsLog2; got != 0 {
					t.Fatalf("i420 encoder tileColsLog2=%d want 0", got)
				}
				if got := stream.encoders[0].wavefrontLanes; got != 4 {
					t.Fatalf("i420 encoder wavefrontLanes=%d want 4", got)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), true)
				if err != nil {
					t.Fatalf("EncodePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.monoEncoders[0] == nil {
					t.Fatal("missing mono encoder")
				}
				if got := stream.monoEncoders[0].tileColsLog2; got != 2 {
					t.Fatalf("mono encoder tileColsLog2=%d want 2", got)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeMonochromePicture(testWebRTCStreamMonoFrame(w, h), true)
				if err != nil {
					t.Fatalf("EncodeMonochromePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].tileColsLog2; got != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", got)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 10), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].tileColsLog2; got != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", got)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 10), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(tc.config(base))
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			tc.assert(t, stream)
			picture := tc.encode(t, stream)
			// The 8-bit 4:2:0 encoder codes one tile (SVT-AV1) on the SB-row
			// wavefront; the others have no wavefront and parallelize on tiles.
			want := 4
			if tc.name == "i420-8" {
				want = 1
			}
			if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != want {
				t.Fatalf("encoded tile columns=%d want %d", got, want)
			}

			change := stream.Config()
			change.MaxThreads = 1
			if err := stream.SetConfig(change); err != nil {
				t.Fatalf("SetConfig MaxThreads=1: %v", err)
			}
			picture = tc.encode(t, stream)
			if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 1 {
				t.Fatalf("updated encoded tile columns=%d want 1", got)
			}
		})
	}
}

func TestWebRTCStreamMaxThreadsSurvivePixelFormatReplacement(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		MaxThreads:        4,
	}

	for _, tc := range []struct {
		name   string
		config func(Config) Config
		encode func(*testing.T, *WebRTCStream) WebRTCEncodedPicture
	}{
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeMonochromePicture(testWebRTCStreamMonoFrame(w, h), true)
				if err != nil {
					t.Fatalf("EncodeMonochromePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 10), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i400-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, MonoChrome: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 12), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 10), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
		},
		{
			name: "i420-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 12), true)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(base)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()

			if err := stream.SetConfig(tc.config(base)); err != nil {
				t.Fatalf("SetConfig: %v", err)
			}
			picture := tc.encode(t, stream)
			if !picture.Keyframe || picture.FrameNum != 1 {
				t.Fatalf("picture key=%v frameNum=%d want key single frame", picture.Keyframe, picture.FrameNum)
			}
			if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != 4 {
				t.Fatalf("encoded tile columns=%d want 4", got)
			}
		})
	}
}

func TestWebRTCStreamConfigSpeedAppliesToPixelEncoders(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
		Speed:             WebRTCMinEffortLevel,
	}

	for _, tc := range []struct {
		name         string
		config       func(Config) Config
		assertEffort func(*testing.T, *WebRTCStream, int8)
	}{
		{
			name: "i420-8",
			config: func(cfg Config) Config {
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.encoders[0] == nil {
					t.Fatal("missing i420 encoder")
				}
				if got := stream.encoders[0].effortLevel; got != want {
					t.Fatalf("i420 effort=%d want %d", got, want)
				}
			},
		},
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.monoEncoders[0] == nil {
					t.Fatal("missing mono encoder")
				}
				if got := stream.monoEncoders[0].effortLevel; got != want {
					t.Fatalf("mono effort=%d want %d", got, want)
				}
			},
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].effortLevel; got != want {
					t.Fatalf("mono16 effort=%d want %d", got, want)
				}
			},
		},
		{
			name: "i400-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, MonoChrome: true}
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.mono16Encoders[0] == nil {
					t.Fatal("missing mono16 encoder")
				}
				if got := stream.mono16Encoders[0].effortLevel; got != want {
					t.Fatalf("mono16 effort=%d want %d", got, want)
				}
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].effortLevel; got != want {
					t.Fatalf("color16 effort=%d want %d", got, want)
				}
			},
		},
		{
			name: "i420-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assertEffort: func(t *testing.T, stream *WebRTCStream, want int8) {
				t.Helper()
				if stream.color16Encoders[0] == nil {
					t.Fatal("missing color16 encoder")
				}
				if got := stream.color16Encoders[0].effortLevel; got != want {
					t.Fatalf("color16 effort=%d want %d", got, want)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(tc.config(base))
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			tc.assertEffort(t, stream, WebRTCMinEffortLevel)
			change := stream.Config()
			change.Speed = WebRTCMaxEffortLevel
			if err := stream.SetConfig(change); err != nil {
				t.Fatalf("SetConfig speed: %v", err)
			}
			tc.assertEffort(t, stream, WebRTCMaxEffortLevel)
		})
	}
}

func TestWebRTCStreamTileColumnsSurvivePixelFormatReplacement(t *testing.T) {
	const w, h = 512, 288
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T1,
	}

	for _, tc := range []struct {
		name      string
		config    func(Config) Config
		assert    func(*testing.T, *WebRTCStream)
		encode    func(*testing.T, *WebRTCStream) WebRTCEncodedPicture
		wantTiles int
	}{
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.monoEncoders[0]
				if enc == nil {
					t.Fatal("missing mono encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeMonochromePicture(testWebRTCStreamMonoFrame(w, h), false)
				if err != nil {
					t.Fatalf("EncodeMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.mono16Encoders[0]
				if enc == nil {
					t.Fatal("missing mono16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i400-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, MonoChrome: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.mono16Encoders[0]
				if enc == nil {
					t.Fatal("missing mono16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("mono16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 12), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.color16Encoders[0]
				if enc == nil {
					t.Fatal("missing color16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
		{
			name: "i420-12",
			config: func(cfg Config) Config {
				cfg.Profile = Profile2
				cfg.BitDepth = 12
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 12, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			assert: func(t *testing.T, stream *WebRTCStream) {
				t.Helper()
				enc := stream.color16Encoders[0]
				if enc == nil {
					t.Fatal("missing color16 encoder")
				}
				if enc.tileColsLog2 != 2 {
					t.Fatalf("color16 encoder tileColsLog2=%d want 2", enc.tileColsLog2)
				}
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 12), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
			wantTiles: 4,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(base)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			stream.SetTileColumns(4)

			if err := stream.SetConfig(tc.config(base)); err != nil {
				t.Fatalf("SetConfig: %v", err)
			}
			tc.assert(t, stream)
			picture := tc.encode(t, stream)
			if !picture.Keyframe || picture.FrameNum != 1 {
				t.Fatalf("picture key=%v frameNum=%d want key single frame", picture.Keyframe, picture.FrameNum)
			}
			if got := parseWebRTCTileColumns(t, picture.Frames[0].TU); got != tc.wantTiles {
				t.Fatalf("encoded tile columns=%d want %d", got, tc.wantTiles)
			}
		})
	}
}

func TestWebRTCStreamConfigAcceptsFullSVCPixelModes(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig L2T2: %v", err)
	}
	if stream.config.Scalability != ScalabilityModeL2T2 || stream.config.SpatialLayerCount != 2 {
		t.Fatalf("config mode=%s spatial=%d", stream.config.Scalability, stream.config.SpatialLayerCount)
	}
}

func TestWebRTCStreamConfigAcceptsSimulcastPixelModes(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig S2T2: %v", err)
	}
	if stream.config.Scalability != ScalabilityModeS2T2 || stream.config.SpatialLayerCount != 2 || stream.config.TemporalLayerCount != 2 {
		t.Fatalf("config mode=%s spatial=%d temporal=%d", stream.config.Scalability, stream.config.SpatialLayerCount, stream.config.TemporalLayerCount)
	}
}

func TestWebRTCStreamEncodeRejectsMultiSpatialWithoutMutating(t *testing.T) {
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig L2T2: %v", err)
	}
	if _, err := stream.Encode(testWebRTCStreamFrame(640, 360), false); err != ErrUnsupported {
		t.Fatalf("Encode multi-spatial err=%v want %v", err, ErrUnsupported)
	}
	if stream.state.NextFrameID != 0 || stream.state.DependencyStructureState.Valid {
		t.Fatalf("unsupported Encode mutated state: %+v", stream.state)
	}
	picture, err := stream.EncodePicture(testWebRTCStreamFrame(640, 360), false)
	if err != nil {
		t.Fatalf("EncodePicture after rejected Encode: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != 0 ||
		picture.Frames[1].Info.FrameID != 1 {
		t.Fatalf("picture after rejected Encode=%+v", picture)
	}
}

func TestWebRTCStreamConfigPixelScalabilityModeMatrix(t *testing.T) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		t.Run(mode.String(), func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(Config{
				Resolution:        Resolution{Width: 1296, Height: 720},
				MaxFramerate:      Rational{Num: 30, Den: 1},
				MinBitrateKbps:    100,
				MaxBitrateKbps:    1200,
				TargetBitrateKbps: 800,
				Scalability:       mode,
			})
			if _, _, _, ok := mode.Layers(); !ok {
				t.Fatalf("mode %d invalid in matrix", mode)
			}
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			if stream.config.Scalability != mode {
				t.Fatalf("NewWebRTCStreamConfig(%s) normalized to %s", mode, stream.config.Scalability)
			}
		})
	}
}

func TestWebRTCStreamTracksReferenceSurfaces(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	if _, err := stream.EncodePicture(testWebRTCStreamFrame(640, 360), false); err != nil {
		t.Fatalf("EncodePicture key: %v", err)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		ref := stream.refSize[i]
		enc := stream.encoders[i]
		if enc == nil {
			t.Fatalf("missing layer encoder %d", i)
		}
		if !stream.referenceValid[i] || int(ref[0]) != enc.width || int(ref[1]) != enc.height {
			t.Fatalf("reference buffer %d geometry=%dx%d valid=%v want %dx%d",
				i, ref[0], ref[1], stream.referenceValid[i], enc.width, enc.height)
		}
	}

	controlChange := stream.config
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.TargetBitrateKbps += 50
	if err := stream.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	if !stream.referenceValid[0] || !stream.referenceValid[1] {
		t.Fatal("control-only change cleared reference surfaces")
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T3
	if err := stream.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	for i, valid := range stream.referenceValid {
		if valid {
			t.Fatalf("reference buffer %d survived structure reset", i)
		}
	}
}

func TestWebRTCStreamDescriptorStateMatrix(t *testing.T) {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		t.Run(mode.String(), func(t *testing.T) {
			cfg, supported := webRTCStreamDescriptorMatrixConfig(mode)
			if !supported {
				t.Skip("pixel stream intentionally rejects delta inter-layer SVC")
			}
			stream, err := NewWebRTCStreamConfig(cfg)
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			src := testWebRTCStreamFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height))
			var receiver internalrtp.DependencyDescriptorState

			key, err := stream.EncodePicture(src, false)
			if err != nil {
				t.Fatalf("key EncodePicture: %v", err)
			}
			if !key.Keyframe {
				t.Fatalf("first picture is not key: %+v", key)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, key)

			controlChange := stream.Config()
			controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps += 10
			controlChange.MaxBitrateKbps += 200
			controlChange.TargetBitrateKbps = (controlChange.MinBitrateKbps + controlChange.MaxBitrateKbps) / 2
			beforeFrameID := stream.state.NextFrameID
			if err := stream.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig control change: %v", err)
			}
			delta, err := stream.EncodePicture(src, false)
			if err != nil {
				t.Fatalf("delta EncodePicture after control change: %v", err)
			}
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("control change forced key or skipped id: picture=%+v before=%d", delta, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)
		})
	}
}

func TestWebRTCStreamSetConfigReconfigure(t *testing.T) {
	const w, h = 640, 360
	cfg := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	src := testWebRTCStreamFrame(w, h)
	key, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	delta, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}
	if !key.Keyframe || delta.Keyframe || key.FrameNum != 1 || delta.FrameNum != 1 || stream.state.NextFrameID != 2 {
		t.Fatalf("warm key=%+v delta=%+v state=%+v", key, delta, stream.state)
	}

	controlChange := cfg
	controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1200
	controlChange.TargetBitrateKbps = 900
	if err := stream.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	wantPerFrameBits := int(controlChange.TargetBitrateKbps) * 1000 / webRTCStreamFramesPerSecond(controlChange.MaxFramerate)
	if got, want := stream.encoders[0].rcPerFrameBits, wantPerFrameBits; got != want {
		t.Fatalf("control change rcPerFrameBits=%d want %d", got, want)
	}
	beforeFrameID := stream.state.NextFrameID
	delta, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after control change: %v", err)
	}
	if delta.Keyframe || delta.FrameNum != 1 || delta.Frames[0].Info.FrameID != beforeFrameID ||
		stream.config.TargetBitrateKbps != 900 || stream.state.NextFrameID != beforeFrameID+1 {
		t.Fatalf("control change delta=%+v state=%+v config=%+v", delta, stream.state, stream.config)
	}

	structureChange := controlChange
	structureChange.Scalability = ScalabilityModeS2T2
	structureChange.SpatialLayers[0].MinBitrateKbps = 120
	structureChange.SpatialLayers[0].MaxBitrateKbps = 480
	structureChange.SpatialLayers[0].TargetBitrateKbps = 240
	structureChange.SpatialLayers[1].MinBitrateKbps = 300
	structureChange.SpatialLayers[1].MaxBitrateKbps = 1400
	structureChange.SpatialLayers[1].TargetBitrateKbps = 840
	if err := stream.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	for i, wantTarget := range [...]int32{240, 840} {
		if got := stream.config.SpatialLayers[i].TargetBitrateKbps; got != wantTarget {
			t.Fatalf("layer %d target bitrate=%d want %d", i, got, wantTarget)
		}
		wantBits := int(wantTarget) * 1000 / webRTCStreamFramesPerSecond(structureChange.MaxFramerate)
		if got := stream.encoders[i].rcPerFrameBits; got != wantBits {
			t.Fatalf("layer %d rcPerFrameBits=%d want %d", i, got, wantBits)
		}
	}
	beforeFrameID = stream.state.NextFrameID
	picture, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key after structure change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeFrameID ||
		picture.Frames[1].Info.FrameID != beforeFrameID+1 ||
		stream.state.NextFrameID != beforeFrameID+2 {
		t.Fatalf("structure change picture=%+v state=%+v", picture, stream.state)
	}

	fullSVC := structureChange
	fullSVC.Scalability = ScalabilityModeL2T2
	if err := stream.SetConfig(fullSVC); err != nil {
		t.Fatalf("SetConfig full SVC: %v", err)
	}
	beforeFrameID = stream.state.NextFrameID
	picture, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("key after full SVC change: %v", err)
	}
	if !picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeFrameID ||
		picture.Frames[1].Info.FrameID != beforeFrameID+1 ||
		stream.state.NextFrameID != beforeFrameID+2 {
		t.Fatalf("full SVC change picture=%+v state=%+v", picture, stream.state)
	}
}

func TestWebRTCStreamSetConfigRateControlTransitions(t *testing.T) {
	const w, h = 640, 360
	cfg := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL2T2,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()

	src := testWebRTCStreamFrame(w, h)
	if _, err := stream.EncodePicture(src, false); err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}
	if _, err := stream.EncodePicture(src, false); err != nil {
		t.Fatalf("delta EncodePicture: %v", err)
	}

	toCQP := stream.Config()
	toCQP.RateControl = RateControlCQP
	toCQP.Quantizer = 37
	beforeState := stream.state
	if err := stream.SetConfig(toCQP); err != nil {
		t.Fatalf("SetConfig CQP: %v", err)
	}
	if stream.state != beforeState {
		t.Fatalf("CQP control transition reset state: before=%+v after=%+v", beforeState, stream.state)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		enc := stream.encoders[i]
		if enc == nil || enc.rcEnabled || enc.rcPerFrameBits != 0 || enc.qIndex != 37 {
			t.Fatalf("layer %d after CQP transition encoder=%+v", i, enc)
		}
	}
	picture, err := stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after CQP transition: %v", err)
	}
	if picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeState.NextFrameID ||
		picture.Frames[1].Info.FrameID != beforeState.NextFrameID+1 {
		t.Fatalf("CQP transition picture=%+v before=%+v state=%+v", picture, beforeState, stream.state)
	}

	toCBR := stream.Config()
	toCBR.RateControl = RateControlCBR
	toCBR.Quantizer = 0
	toCBR.MaxFramerate = Rational{Num: 60, Den: 1}
	for i := uint8(0); i < toCBR.SpatialLayerCount; i++ {
		layer := &toCBR.SpatialLayers[i]
		layer.MinBitrateKbps = 120 + int32(i)*180
		layer.MaxBitrateKbps = 620 + int32(i)*420
		layer.TargetBitrateKbps = layer.MinBitrateKbps + (layer.MaxBitrateKbps-layer.MinBitrateKbps)/2
	}
	beforeState = stream.state
	if err := stream.SetConfig(toCBR); err != nil {
		t.Fatalf("SetConfig CBR: %v", err)
	}
	if stream.state != beforeState {
		t.Fatalf("CBR control transition reset state: before=%+v after=%+v", beforeState, stream.state)
	}
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		enc := stream.encoders[i]
		wantBits := int(stream.config.SpatialLayers[i].TargetBitrateKbps) * 1000 / webRTCStreamFramesPerSecond(stream.config.MaxFramerate)
		if enc == nil || !enc.rcEnabled || enc.rcPerFrameBits != wantBits {
			t.Fatalf("layer %d after CBR transition rc=%v bits=%d want %d", i, enc != nil && enc.rcEnabled, encoderPerFrameBits(enc), wantBits)
		}
	}
	picture, err = stream.EncodePicture(src, false)
	if err != nil {
		t.Fatalf("delta after CBR transition: %v", err)
	}
	if picture.Keyframe || picture.FrameNum != 2 ||
		picture.Frames[0].Info.FrameID != beforeState.NextFrameID ||
		picture.Frames[1].Info.FrameID != beforeState.NextFrameID+1 {
		t.Fatalf("CBR transition picture=%+v before=%+v state=%+v", picture, beforeState, stream.state)
	}
}

func TestWebRTCStreamSetConfigControlsAllPixelEncoders(t *testing.T) {
	const w, h = 320, 180
	base := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    120,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 480,
		Scalability:       ScalabilityModeL1T2,
		RateControl:       RateControlCBR,
	}
	type controlSnapshot struct {
		rcEnabled              bool
		rcTargetBits           int
		rcFramesPerSec         int
		rcPerFrameBits         int
		qIndex                 uint8
		temporalLayers         int
		rcTemporalPerFrameBits [WebRTCMaxTemporalLayers]int
		// sharedRC marks SVT-AV1's one rate control for every temporal
		// layer (the 8-bit encoder): no per-layer budgets.
		sharedRC bool
	}
	type pixelCase struct {
		name     string
		config   func(Config) Config
		encode   func(*testing.T, *WebRTCStream) WebRTCEncodedPicture
		snapshot func(*WebRTCStream) controlSnapshot
	}

	cases := []pixelCase{
		{
			name: "i420-8",
			config: func(cfg Config) Config {
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), false)
				if err != nil {
					t.Fatalf("EncodePicture: %v", err)
				}
				return picture
			},
			snapshot: func(stream *WebRTCStream) controlSnapshot {
				enc := stream.encoders[0]
				if enc == nil {
					return controlSnapshot{}
				}
				return controlSnapshot{
					rcEnabled:      enc.rcEnabled,
					rcTargetBits:   enc.rcTargetBits,
					rcFramesPerSec: enc.rcFramesPerSec,
					rcPerFrameBits: enc.rcPerFrameBits,
					qIndex:         enc.qIndex,
					temporalLayers: enc.temporalLayers,
					sharedRC:       true,
				}
			},
		},
		{
			name: "i400-8",
			config: func(cfg Config) Config {
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 8, MonoChrome: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeMonochromePicture(testWebRTCStreamMonoFrame(w, h), false)
				if err != nil {
					t.Fatalf("EncodeMonochromePicture: %v", err)
				}
				return picture
			},
			snapshot: func(stream *WebRTCStream) controlSnapshot {
				enc := stream.monoEncoders[0]
				if enc == nil {
					return controlSnapshot{}
				}
				return controlSnapshot{
					rcEnabled:              enc.rcEnabled,
					rcTargetBits:           enc.rcTargetBits,
					rcFramesPerSec:         enc.rcFramesPerSec,
					rcPerFrameBits:         enc.rcPerFrameBits,
					qIndex:                 enc.qIndex,
					temporalLayers:         enc.temporalLayers,
					rcTemporalPerFrameBits: enc.rcTemporalPerFrameBits,
				}
			},
		},
		{
			name: "i400-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, MonoChrome: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepthMonochromePicture(testWebRTCStreamMono16Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepthMonochromePicture: %v", err)
				}
				return picture
			},
			snapshot: func(stream *WebRTCStream) controlSnapshot {
				enc := stream.mono16Encoders[0]
				if enc == nil {
					return controlSnapshot{}
				}
				return controlSnapshot{
					rcEnabled:              enc.rcEnabled,
					rcTargetBits:           enc.rcTargetBits,
					rcFramesPerSec:         enc.rcFramesPerSec,
					rcPerFrameBits:         enc.rcPerFrameBits,
					qIndex:                 enc.qIndex,
					temporalLayers:         enc.temporalLayers,
					rcTemporalPerFrameBits: enc.rcTemporalPerFrameBits,
				}
			},
		},
		{
			name: "i420-10",
			config: func(cfg Config) Config {
				cfg.BitDepth = 10
				cfg.ColorConfigSet = true
				cfg.ColorConfig = SequenceColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
				return cfg
			},
			encode: func(t *testing.T, stream *WebRTCStream) WebRTCEncodedPicture {
				t.Helper()
				picture, err := stream.EncodeHighBitDepth420Picture(testWebRTCStream42016Frame(w, h, 10), false)
				if err != nil {
					t.Fatalf("EncodeHighBitDepth420Picture: %v", err)
				}
				return picture
			},
			snapshot: func(stream *WebRTCStream) controlSnapshot {
				enc := stream.color16Encoders[0]
				if enc == nil {
					return controlSnapshot{}
				}
				return controlSnapshot{
					rcEnabled:              enc.rcEnabled,
					rcTargetBits:           enc.rcTargetBits,
					rcFramesPerSec:         enc.rcFramesPerSec,
					rcPerFrameBits:         enc.rcPerFrameBits,
					qIndex:                 enc.qIndex,
					temporalLayers:         enc.temporalLayers,
					rcTemporalPerFrameBits: enc.rcTemporalPerFrameBits,
				}
			},
		},
	}

	assertCBR := func(t *testing.T, stream *WebRTCStream, tc pixelCase) {
		t.Helper()
		cfg := stream.Config()
		snap := tc.snapshot(stream)
		if !snap.rcEnabled {
			t.Fatalf("%s CBR disabled: %+v", tc.name, snap)
		}
		fps := webRTCStreamFramesPerSecond(cfg.MaxFramerate)
		targetBits := int(webRTCStreamLayerTargetKbps(cfg, 0)) * 1000
		wantPerFrame, err := rateControlPerFrameBits(RateControlConfig{
			TargetBitsPerSecond: targetBits,
			FramesPerSecond:     fps,
			MinQIndex:           stream.rcMinQ,
			MaxQIndex:           stream.rcMaxQ,
		})
		if err != nil {
			t.Fatalf("%s rateControlPerFrameBits: %v", tc.name, err)
		}
		if snap.rcTargetBits != targetBits || snap.rcFramesPerSec != fps ||
			snap.rcPerFrameBits != wantPerFrame || snap.temporalLayers != int(cfg.TemporalLayerCount) {
			t.Fatalf("%s CBR snapshot=%+v target=%d fps=%d perFrame=%d temporal=%d",
				tc.name, snap, targetBits, fps, wantPerFrame, cfg.TemporalLayerCount)
		}
		for temporalID := uint8(0); temporalID < cfg.TemporalLayerCount && !snap.sharedRC; temporalID++ {
			want := rateControlTemporalLayerPerFrameBits(targetBits, fps, int(cfg.TemporalLayerCount), temporalID, wantPerFrame)
			if got := snap.rcTemporalPerFrameBits[temporalID]; got != want {
				t.Fatalf("%s T%d per-frame bits=%d want %d snapshot=%+v", tc.name, temporalID, got, want, snap)
			}
		}
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewWebRTCStreamConfig(tc.config(base))
			if err != nil {
				t.Fatalf("NewWebRTCStreamConfig: %v", err)
			}
			defer stream.Close()
			var receiver internalrtp.DependencyDescriptorState

			key := tc.encode(t, stream)
			if !key.Keyframe || key.FrameNum != 1 || key.Frames[0].Info.TemporalID != 0 {
				t.Fatalf("initial key picture=%+v", key)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, key)
			delta := tc.encode(t, stream)
			if delta.Keyframe || delta.FrameNum != 1 || delta.Frames[0].Info.TemporalID != 1 {
				t.Fatalf("initial delta picture=%+v", delta)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)
			assertCBR(t, stream, tc)

			controlChange := stream.Config()
			controlChange.MaxFramerate = Rational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps = 160
			controlChange.MaxBitrateKbps = 1200
			controlChange.TargetBitrateKbps = 720
			beforeState := stream.state
			if err := stream.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig CBR control change: %v", err)
			}
			if stream.state != beforeState {
				t.Fatalf("CBR control change reset state: before=%+v after=%+v", beforeState, stream.state)
			}
			assertCBR(t, stream, tc)
			beforeFrameID := stream.state.NextFrameID
			delta = tc.encode(t, stream)
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("CBR control change picture=%+v beforeFrameID=%d", delta, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)

			toCQP := stream.Config()
			toCQP.RateControl = RateControlCQP
			toCQP.Quantizer = 37
			beforeState = stream.state
			if err := stream.SetConfig(toCQP); err != nil {
				t.Fatalf("SetConfig CQP: %v", err)
			}
			if stream.state != beforeState {
				t.Fatalf("CQP transition reset state: before=%+v after=%+v", beforeState, stream.state)
			}
			snap := tc.snapshot(stream)
			if snap.rcEnabled || snap.rcTargetBits != 0 || snap.rcFramesPerSec != 0 ||
				snap.rcPerFrameBits != 0 || snap.qIndex != toCQP.Quantizer {
				t.Fatalf("CQP snapshot=%+v want q=%d", snap, toCQP.Quantizer)
			}
			beforeFrameID = stream.state.NextFrameID
			delta = tc.encode(t, stream)
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID {
				t.Fatalf("CQP transition picture=%+v beforeFrameID=%d", delta, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)

			structureChange := stream.Config()
			structureChange.Scalability = ScalabilityModeL1T3
			structureChange.RateControl = RateControlCBR
			structureChange.Quantizer = 0
			structureChange.MaxFramerate = Rational{Num: 30, Den: 1}
			structureChange.MinBitrateKbps = 180
			structureChange.MaxBitrateKbps = 1500
			structureChange.TargetBitrateKbps = 900
			beforeFrameID = stream.state.NextFrameID
			if err := stream.SetConfig(structureChange); err != nil {
				t.Fatalf("SetConfig structure change: %v", err)
			}
			assertCBR(t, stream, tc)
			key = tc.encode(t, stream)
			if !key.Keyframe || key.Frames[0].Info.FrameID != beforeFrameID ||
				!key.Frames[0].AttachDependencyStructure || key.Frames[0].Info.TemporalID != 0 {
				t.Fatalf("structure change key=%+v beforeFrameID=%d", key, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, key)
			beforeFrameID = stream.state.NextFrameID
			delta = tc.encode(t, stream)
			if delta.Keyframe || delta.Frames[0].Info.FrameID != beforeFrameID ||
				delta.Frames[0].Info.TemporalID != 2 {
				t.Fatalf("L1T3 delta=%+v beforeFrameID=%d", delta, beforeFrameID)
			}
			assertWebRTCStreamPictureDependencyDescriptors(t, &receiver, delta)
		})
	}
}

func TestWebRTCStreamSetConfigCBRRestartsSVTRateControl(t *testing.T) {
	const w, h = 640, 360
	cfg := Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeL1T3,
		RateControl:       RateControlCBR,
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()
	if _, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), false); err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}

	enc := stream.encoders[0]
	if enc == nil {
		t.Fatal("missing video encoder")
	}
	// SVT-AV1 restarts its rate control on a rate change event
	// (seq_param_changed): the buffer returns to the starting level.
	enc.svtRC.bufferLevel = -1234
	beforeState := stream.state

	change := stream.Config()
	change.MaxFramerate = Rational{Num: 60, Den: 1}
	change.MinBitrateKbps = 200
	change.MaxBitrateKbps = 1200
	change.TargetBitrateKbps = 720
	if err := stream.SetConfig(change); err != nil {
		t.Fatalf("SetConfig CBR soft update: %v", err)
	}
	if stream.state != beforeState {
		t.Fatalf("CBR soft update reset stream state: before=%+v after=%+v", beforeState, stream.state)
	}
	if enc.svtRC.bufferLevel != enc.svtRC.startingBufferLevel {
		t.Fatalf("CBR update kept the buffer: %d (starting %d)", enc.svtRC.bufferLevel, enc.svtRC.startingBufferLevel)
	}
	wantPerFrame, err := rateControlPerFrameBits(RateControlConfig{
		TargetBitsPerSecond: int(change.TargetBitrateKbps) * 1000,
		FramesPerSecond:     60,
		MinQIndex:           stream.rcMinQ,
		MaxQIndex:           stream.rcMaxQ,
	})
	if err != nil {
		t.Fatalf("rateControlPerFrameBits: %v", err)
	}
	if enc.rcPerFrameBits != wantPerFrame {
		t.Fatalf("rcPerFrameBits=%d want %d", enc.rcPerFrameBits, wantPerFrame)
	}
	picture, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), false)
	if err != nil {
		t.Fatalf("delta after CBR soft update: %v", err)
	}
	if picture.Keyframe || picture.Frames[0].Info.FrameID != beforeState.NextFrameID {
		t.Fatalf("CBR soft update picture=%+v before=%+v state=%+v", picture, beforeState, stream.state)
	}
}

func TestWebRTCStreamSetConfigRejectsInvalidWithoutMutation(t *testing.T) {
	const w, h = 640, 360
	stream, err := NewWebRTCStreamConfig(Config{
		Resolution:        Resolution{Width: w, Height: h},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS2T2,
	})
	if err != nil {
		t.Fatalf("NewWebRTCStreamConfig: %v", err)
	}
	defer stream.Close()
	if _, err := stream.EncodePicture(testWebRTCStreamFrame(w, h), false); err != nil {
		t.Fatalf("key EncodePicture: %v", err)
	}

	type layerSnapshot struct {
		rcEnabled      bool
		rcPerFrameBits int
		qIndex         uint8
		temporalLayers int
		haveKey        bool
	}
	snapshotLayers := func() [WebRTCMaxSpatialLayers]layerSnapshot {
		var out [WebRTCMaxSpatialLayers]layerSnapshot
		for i := uint8(0); i < WebRTCMaxSpatialLayers; i++ {
			enc := stream.encoders[i]
			if enc == nil {
				continue
			}
			out[i] = layerSnapshot{
				rcEnabled:      enc.rcEnabled,
				rcPerFrameBits: enc.rcPerFrameBits,
				qIndex:         enc.qIndex,
				temporalLayers: enc.temporalLayers,
				haveKey:        enc.haveKey,
			}
		}
		return out
	}

	keepConfig := stream.config
	keepState := stream.state
	keepLayers := snapshotLayers()
	for _, tc := range []struct {
		name string
		cfg  Config
		want error
	}{
		{
			name: "invalid-target",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.TargetBitrateKbps = cfg.MaxBitrateKbps + 1
				return cfg
			}(),
			want: ErrInvalidConfig,
		},
		{
			name: "invalid-fps",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.MaxFramerate = Rational{Num: 0, Den: 1}
				return cfg
			}(),
			want: ErrInvalidConfig,
		},
		{
			name: "unsupported-cqp-zero",
			cfg: func() Config {
				cfg := stream.Config()
				cfg.RateControl = RateControlCQP
				cfg.Quantizer = 0
				return cfg
			}(),
			want: ErrUnsupported,
		},
	} {
		if err := stream.SetConfig(tc.cfg); !errors.Is(err, tc.want) {
			t.Fatalf("%s SetConfig err=%v want %v", tc.name, err, tc.want)
		}
		if stream.config != keepConfig || stream.state != keepState || snapshotLayers() != keepLayers {
			t.Fatalf("%s mutated stream config=%+v state=%+v layers=%+v", tc.name, stream.config, stream.state, snapshotLayers())
		}
	}
}

func encoderPerFrameBits(enc *VideoEncoder) int {
	if enc == nil {
		return 0
	}
	return enc.rcPerFrameBits
}

func webRTCStreamDescriptorMatrixConfig(mode ScalabilityMode) (Config, bool) {
	spatial, _, _, ok := mode.Layers()
	if !ok {
		return Config{}, false
	}
	cfg := Config{
		Resolution:        Resolution{Width: 192, Height: 128},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    80,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
		Scalability:       mode,
	}
	switch spatial {
	case 2:
		cfg.Resolution = Resolution{Width: 640, Height: 360}
		cfg.MinBitrateKbps = 100
		cfg.MaxBitrateKbps = 1000
		cfg.TargetBitrateKbps = 650
	case 3:
		cfg.Resolution = Resolution{Width: 1008, Height: 576}
		cfg.MinBitrateKbps = 150
		cfg.MaxBitrateKbps = 1800
		cfg.TargetBitrateKbps = 1100
	}
	return cfg, true
}

func assertWebRTCStreamPictureDependencyDescriptors(t *testing.T, receiver *internalrtp.DependencyDescriptorState, picture WebRTCEncodedPicture) {
	t.Helper()
	if picture.FrameNum == 0 || picture.FrameNum > WebRTCMaxSpatialLayers {
		t.Fatalf("invalid picture frame num=%d", picture.FrameNum)
	}
	for i := uint8(0); i < picture.FrameNum; i++ {
		frame := picture.Frames[i]
		parsed, consumed, err := receiver.Parse(frame.Descriptor)
		if err != nil {
			t.Fatalf("frame %d dependency descriptor parse: %v", i, err)
		}
		if consumed != len(frame.Descriptor) ||
			parsed.Mandatory.FrameNumber != uint16(frame.Info.FrameID) ||
			!parsed.Mandatory.FirstPacketInFrame ||
			!parsed.Mandatory.LastPacketInFrame {
			t.Fatalf("frame %d parsed mandatory=%+v consumed=%d len=%d info=%+v", i, parsed.Mandatory, consumed, len(frame.Descriptor), frame.Info)
		}
		if parsed.HasAttachedStructure != frame.AttachDependencyStructure {
			t.Fatalf("frame %d attached=%v want %v", i, parsed.HasAttachedStructure, frame.AttachDependencyStructure)
		}
		if frame.AttachDependencyStructure &&
			(!receiver.Valid ||
				parsed.AttachedStructure.NumDecodeTargets != frame.Structure.NumDecodeTargets ||
				parsed.AttachedStructure.NumChains != frame.Structure.NumChains ||
				parsed.AttachedStructure.TemplateNum != frame.Structure.TemplateNum ||
				parsed.AttachedStructure.ResolutionNum != frame.Structure.ResolutionNum) {
			t.Fatalf("frame %d attached structure parsed=%+v encoder=%+v receiver=%+v", i, parsed.AttachedStructure, frame.Structure, receiver)
		}
		deps := parsed.FrameDependencies
		if deps.SpatialID != frame.Info.SpatialID || deps.TemporalID != frame.Info.TemporalID ||
			deps.DTINum != frame.Info.DTINum || deps.FrameDiffNum != frame.Info.DependencyNum {
			t.Fatalf("frame %d deps=%+v info=%+v", i, deps, frame.Info)
		}
		for target := uint8(0); target < frame.Info.DTINum; target++ {
			if deps.DTIs[target] != internalrtp.DependencyDescriptorDecodeTargetIndication(frame.Info.DTIs[target]) {
				t.Fatalf("frame %d target %d dti=%v want %v", i, target, deps.DTIs[target], frame.Info.DTIs[target])
			}
		}
		for dep := uint8(0); dep < frame.Info.DependencyNum; dep++ {
			wantDiff := uint16(frame.Info.FrameID - frame.Info.Dependencies[dep])
			if deps.FrameDiffs[dep] != wantDiff {
				t.Fatalf("frame %d dep %d diff=%d want %d info=%+v", i, dep, deps.FrameDiffs[dep], wantDiff, frame.Info)
			}
		}
		if parsed.HasResolution {
			resolution := frame.Structure.Resolutions[frame.Info.SpatialID]
			if parsed.Resolution.Width != uint16(resolution.Width) ||
				parsed.Resolution.Height != uint16(resolution.Height) {
				t.Fatalf("frame %d resolution=%+v want %+v", i, parsed.Resolution, resolution)
			}
		}
	}
}

func testWebRTCStreamFrame(width int, height int) SourceFrame420 {
	cw, ch := width/2, height/2
	f := SourceFrame420{
		Y: make([]byte, width*height), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: width, ChromaStride: cw, Width: width, Height: height,
	}
	for i := range f.Y {
		f.Y[i] = uint8(40 + i%170)
	}
	for i := range f.U {
		f.U[i] = 121
		f.V[i] = 133
	}
	return f
}

func testWebRTCStreamMonoFrame(width int, height int) SourceFrameMono {
	f := SourceFrameMono{
		Y:       make([]byte, width*height),
		YStride: width,
		Width:   width,
		Height:  height,
	}
	for i := range f.Y {
		f.Y[i] = uint8(40 + i%170)
	}
	return f
}

func testWebRTCStreamMono16Frame(width int, height int, bitDepth uint8) SourceFrameMono16 {
	f := SourceFrameMono16{
		Y:        make([]uint16, width*height),
		YStride:  width,
		Width:    width,
		Height:   height,
		BitDepth: bitDepth,
	}
	base := uint16(1 << max(bitDepth-4, 0))
	span := uint16((1 << bitDepth) - 1)
	for i := range f.Y {
		f.Y[i] = (base + uint16((i*17)%int(span/2))) & span
	}
	return f
}

func testWebRTCStream42016Frame(width int, height int, bitDepth uint8) SourceFrame42016 {
	cw, ch := width/2, height/2
	f := SourceFrame42016{
		Y:            make([]uint16, width*height),
		U:            make([]uint16, cw*ch),
		V:            make([]uint16, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
		BitDepth:     bitDepth,
	}
	maxSample := uint16((1 << bitDepth) - 1)
	for i := range f.Y {
		f.Y[i] = uint16((64 + i%512) & int(maxSample))
	}
	for i := range f.U {
		f.U[i] = maxSample / 3
		f.V[i] = maxSample * 2 / 3
	}
	return f
}

func parseWebRTCTileColumns(t *testing.T, tu []byte) int {
	t.Helper()
	var seq parser.SequenceHeader
	haveSeq := false
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		switch unit.Header.Type {
		case obu.TypeSequenceHeader:
			seq, err = parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			haveSeq = true
		case obu.TypeFrameHeader, obu.TypeFrame:
			if !haveSeq {
				t.Fatal("frame header before sequence header")
			}
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			var size parser.FrameSize
			if prefix.UsesIntraFrameSizePath() {
				size, err = parser.ParseFrameSize(unit.Payload, seq, prefix, nil, unit.Header.TemporalID, unit.Header.SpatialID)
			} else {
				size, err = parser.ParseFrameSize(unit.Payload, seq, prefix, nil, unit.Header.TemporalID, unit.Header.SpatialID)
			}
			if err != nil {
				t.Fatalf("ParseFrameSize: %v", err)
			}
			tiles, err := parser.ParseTileInfo(unit.Payload, seq, prefix, size)
			if err != nil {
				t.Fatalf("ParseTileInfo: %v", err)
			}
			return int(tiles.Cols)
		}
	}
	t.Fatal("missing frame header")
	return 0
}

// TestWebRTCStreamSharedSlotFlipResetsReferences guards the invariant the
// reference pool relies on: a SetConfig that switches between shared-slot
// coding (slot pixels live in the pool) and any other mode (slots record
// only geometry) must drop every reference slot, so a pixel-less slot is never
// read and the pool's reference counts stay exact.
func TestWebRTCStreamSharedSlotFlipResetsReferences(t *testing.T) {
	checkPool := func(t *testing.T, s *WebRTCStream, when string) {
		t.Helper()
		if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
			return
		}
		var want [webRTCRefPoolSize]uint8
		for slot, valid := range s.referenceValid {
			if valid {
				want[s.refSlot[slot]]++
			}
		}
		if want != s.refPoolRefs {
			t.Fatalf("%s: pool refs %v, slots imply %v", when, s.refPoolRefs, want)
		}
	}
	var flips atomic.Int64
	// Each starting mode runs as a parallel subtest; the group returns once
	// every pair has run, so the flip count below covers all of them.
	t.Run("pairs", func(t *testing.T) {
		for a := ScalabilityMode(0); a < scalabilityModeCount; a++ {
			cfgA, ok := webRTCStreamDescriptorMatrixConfig(a)
			if !ok {
				continue
			}
			t.Run(a.String(), func(t *testing.T) {
				t.Parallel()
				for b := ScalabilityMode(0); b < scalabilityModeCount; b++ {
					cfgB, ok := webRTCStreamDescriptorMatrixConfig(b)
					// Same resolution keeps SetConfig on its in-place path; only a
					// simulcast flip can flip shared-slot coding there.
					if !ok || a == b || cfgA.Resolution != cfgB.Resolution || a.IsSimulcast() == b.IsSimulcast() {
						continue
					}
					stream, err := NewWebRTCStreamConfig(cfgA)
					if err != nil {
						continue
					}
					src := testWebRTCStreamFrame(int(cfgA.Resolution.Width), int(cfgA.Resolution.Height))
					for i := 0; i < 1; i++ {
						if _, err := stream.EncodePicture(src, false); err != nil {
							t.Fatalf("%s picture %d: %v", a, i, err)
						}
					}
					checkPool(t, stream, a.String())
					sharedA := webRTCStreamUsesSharedReferenceSlotCoding(stream.config)
					if err := stream.SetConfig(cfgB); err != nil {
						_ = stream.Close()
						continue
					}
					if sharedA != webRTCStreamUsesSharedReferenceSlotCoding(stream.config) {
						flips.Add(1)
						for slot, valid := range stream.referenceValid {
							if valid {
								t.Fatalf("%s -> %s: slot %d survived a shared-slot mode flip", a, b, slot)
							}
						}
					}
					for i := 0; i < 2; i++ {
						if _, err := stream.EncodePicture(src, false); err != nil {
							t.Fatalf("%s -> %s picture %d: %v", a, b, i, err)
						}
					}
					checkPool(t, stream, a.String()+"->"+b.String())
					_ = stream.Close()
				}
			})
		}
	})
	if flips.Load() == 0 {
		t.Fatal("no mode pair flipped shared-slot coding")
	}
	t.Logf("%d shared-slot flips checked", flips.Load())
}
