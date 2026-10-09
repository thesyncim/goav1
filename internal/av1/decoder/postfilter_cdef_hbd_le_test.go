//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package decoder

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

func TestCDEFHBDInPlaceMatchesSnapshot(t *testing.T) {
	if !frameWorkCDEFHBDInPlaceAvailable() {
		t.Fatal("little-endian HBD in-place path is unavailable in its build-tagged test")
	}
	tests := []struct {
		name             string
		width, height    int
		bitDepth         uint8
		subsamplingX     bool
		subsamplingY     bool
		mono             bool
		secondaryOnly    bool
		unalignedY       bool
		oddSampleStrideY bool
	}{
		{name: "10bit_420_odd_crop", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true},
		{name: "12bit_422_odd_crop", width: 137, height: 131, bitDepth: 12, subsamplingX: true},
		{name: "10bit_444_partial_units", width: 73, height: 69, bitDepth: 10},
		{name: "12bit_monochrome", width: 137, height: 131, bitDepth: 12, mono: true},
		{name: "10bit_422_secondary_only", width: 137, height: 131, bitDepth: 10, subsamplingX: true, secondaryOnly: true},
		{name: "12bit_444_secondary_only", width: 137, height: 131, bitDepth: 12, secondaryOnly: true},
		{name: "12bit_monochrome_secondary_only", width: 137, height: 131, bitDepth: 12, mono: true, secondaryOnly: true},
		{name: "10bit_420_unaligned_luma_fallback", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true, unalignedY: true},
		{name: "10bit_420_odd_sample_stride", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true, oddSampleStrideY: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCDEFHBDInPlaceDifferential(t, tt.width, tt.height, tt.bitDepth, tt.subsamplingX, tt.subsamplingY, tt.mono, tt.secondaryOnly, tt.unalignedY, tt.oddSampleStrideY)
		})
	}
}

func runCDEFHBDInPlaceDifferential(t *testing.T, width, height int, bitDepth uint8, subsamplingX, subsamplingY, mono, secondaryOnly, unalignedY, oddSampleStrideY bool) {
	t.Helper()
	seq := testSequence()
	seq.EnableCDEF = true
	strengths := parser.CDEFParams{
		Damping: 5, StrengthCount: 2,
		YStrength:  [parser.MaxCDEFStrengths]uint8{9<<2 | 1, 15<<2 | 3},
		UVStrength: [parser.MaxCDEFStrengths]uint8{5<<2 | 2, 11<<2 | 1},
	}
	if secondaryOnly {
		strengths.YStrength = [parser.MaxCDEFStrengths]uint8{3, 1}
		strengths.UVStrength = [parser.MaxCDEFStrengths]uint8{2, 3}
	}
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: uint32(width), UpscaledWidth: uint32(width), Height: uint32(height),
			SuperResDenominator: 8,
		},
		CDEF: strengths,
	}
	format := frame.Format{
		Width: width, Height: height, BitDepth: bitDepth,
		SubsamplingX: subsamplingX, SubsamplingY: subsamplingY,
		MonoChrome: mono, Align: 64,
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		fillCDEFHBDPlaneAllocation(out.Y, int(bitDepth), 1)
		if !mono {
			fillCDEFHBDPlaneAllocation(out.U, int(bitDepth), 2)
			fillCDEFHBDPlaneAllocation(out.V, int(bitDepth), 3)
		}
		if unalignedY {
			out.Y = cdefHBDPlaneWithUnalignedPix(out.Y)
		}
		if oddSampleStrideY {
			out.Y = cdefHBDPlaneWithOddSampleStride(out.Y)
		}
		if unalignedY && frameWorkCDEFHBDPlaneViewAvailable(out.Y) {
			t.Fatal("unaligned luma unexpectedly eligible for the in-place uint16 view")
		}
		if oddSampleStrideY {
			if out.Y.Stride/2%2 == 0 {
				t.Fatalf("luma stride has an even sample count: byte stride=%d", out.Y.Stride)
			}
			if !frameWorkCDEFHBDPlaneViewAvailable(out.Y) {
				t.Fatal("odd sample stride should remain eligible for the in-place uint16 view")
			}
		}
		ctx := (FrameWorkPostFilterContext{Event: event, Output: out}).WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
		for i := range req.IndexMap.Index {
			req.IndexMap.Index[i] = uint8(i % 2)
			req.IndexMap.Read[i] = i%9 != 4
		}
		if secondaryOnly {
			poisonCDEFHBDGrids(req)
		}
		for i := range req.SampleScratch {
			for j := range req.SampleScratch[i] {
				req.SampleScratch[i][j] = 0x6bad
			}
		}
		return out, ctx, req
	}

	streamOut, streamCtx, streamReq := build()
	snapshotOut, snapshotCtx, snapshotReq := build()
	initialPlanes := [3][]byte{
		append([]byte(nil), streamOut.Y.Pix...),
		append([]byte(nil), streamOut.U.Pix...),
		append([]byte(nil), streamOut.V.Pix...),
	}
	var skipMap threading.FrameWorkLoopFilterMap
	miCols := (width + 3) / 4
	miRows := (height + 3) / 4
	skipMap.Stride = uint16(miCols)
	skipMap.Rows = uint16(miRows)
	skipMap.Records = make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
	for i := range skipMap.Records {
		skipMap.Records[i].Valid = true
		skipMap.Records[i].SkipTransform = i%13 == 5 || i%17 == 3
	}
	streamCtx.LoopFilterMap = &skipMap
	snapshotCtx.LoopFilterMap = &skipMap

	applyStream := func() (FrameWorkCDEFPostFilterResult, error) {
		copy(streamOut.Y.Pix, initialPlanes[0])
		copy(streamOut.U.Pix, initialPlanes[1])
		copy(streamOut.V.Pix, initialPlanes[2])
		return streamCtx.ApplyCDEFPostFilter(streamReq)
	}
	streamResult, err := applyStream()
	if err != nil {
		t.Fatalf("in-place ApplyCDEFPostFilter: %v", err)
	}
	if streamResult.Blocks == 0 || streamResult.Units == 0 {
		t.Fatalf("in-place path did no filtering: %+v", streamResult)
	}
	allocs := testing.AllocsPerRun(5, func() {
		result, err := applyStream()
		if err != nil || result != streamResult {
			t.Fatalf("warmed in-place apply result=%+v err=%v, want %+v", result, err, streamResult)
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed HBD in-place apply allocated %g times", allocs)
	}
	if err := snapshotCtx.LoadCDEFPostFilterSamples(snapshotReq); err != nil {
		t.Fatalf("snapshot load: %v", err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	var snapshotResult FrameWorkCDEFPostFilterResult
	for row := 0; row < rows; row++ {
		bandResult, err := snapshotCtx.ApplyCDEFPostFilterUnitRows(snapshotReq, row, row+1)
		if err != nil {
			t.Fatalf("snapshot row %d: %v", row, err)
		}
		snapshotResult.Units += bandResult.Units
		snapshotResult.Blocks += bandResult.Blocks
		if bandResult.Planes > snapshotResult.Planes {
			snapshotResult.Planes = bandResult.Planes
		}
	}
	if streamResult != snapshotResult {
		t.Fatalf("in-place counts=%+v snapshot counts=%+v", streamResult, snapshotResult)
	}
	if secondaryOnly {
		assertCDEFHBDGridsRemainPoisoned(t, streamReq)
		assertCDEFHBDGridsRemainPoisoned(t, snapshotReq)
	}
	for plane, pair := range [][2]frame.Plane{{streamOut.Y, snapshotOut.Y}, {streamOut.U, snapshotOut.U}, {streamOut.V, snapshotOut.V}} {
		if !bytes.Equal(pair[0].Pix, pair[1].Pix) {
			i := firstCDEFHBDByteDiff(pair[0].Pix, pair[1].Pix)
			if i >= len(pair[0].Pix) || i >= len(pair[1].Pix) {
				t.Fatalf("plane %d byte lengths differ: in-place=%d snapshot=%d", plane, len(pair[0].Pix), len(pair[1].Pix))
			}
			t.Fatalf("plane %d differs at byte %d: in-place=%d snapshot=%d", plane, i, pair[0].Pix[i], pair[1].Pix[i])
		}
	}
	if len(streamReq.DirectionGrid) != len(snapshotReq.DirectionGrid) ||
		len(streamReq.VarianceGrid) != len(snapshotReq.VarianceGrid) {
		t.Fatalf("grid lengths differ: stream=%d/%d snapshot=%d/%d", len(streamReq.DirectionGrid), len(streamReq.VarianceGrid), len(snapshotReq.DirectionGrid), len(snapshotReq.VarianceGrid))
	}
	for i := range streamReq.DirectionGrid {
		if streamReq.DirectionGrid[i] != snapshotReq.DirectionGrid[i] || streamReq.VarianceGrid[i] != snapshotReq.VarianceGrid[i] {
			t.Fatalf("direction/variance grids differ at unit %d", i)
		}
	}
}

func cdefHBDPlaneWithUnalignedPix(plane frame.Plane) frame.Plane {
	backing := make([]byte, len(plane.Pix)+1)
	copy(backing[1:], plane.Pix)
	plane.Pix = backing[1:]
	return plane
}

func cdefHBDPlaneWithOddSampleStride(plane frame.Plane) frame.Plane {
	oldStride := plane.Stride
	strideSamples := oldStride / 2
	if strideSamples&1 == 0 {
		strideSamples++
	}
	plane.Stride = strideSamples * 2
	allocatedRows := len(plane.Pix) / oldStride
	tailBytes := len(plane.Pix) - allocatedRows*oldStride
	pix := make([]byte, allocatedRows*plane.Stride+tailBytes)
	rowBytes := minInt(oldStride, plane.Stride)
	for y := 0; y < allocatedRows; y++ {
		oldStart := y * oldStride
		newStart := y * plane.Stride
		copy(pix[newStart:newStart+rowBytes], plane.Pix[oldStart:oldStart+rowBytes])
	}
	copy(pix[allocatedRows*plane.Stride:], plane.Pix[allocatedRows*oldStride:])
	plane.Pix = pix
	return plane
}

const cdefHBDDirectionPoison uint8 = 0xa7
const cdefHBDVariancePoison int32 = 0x1234567

func poisonCDEFHBDGrids(req FrameWorkCDEFPostFilterRequest) {
	for i := range req.DirectionGrid {
		for by := range req.DirectionGrid[i] {
			for bx := range req.DirectionGrid[i][by] {
				req.DirectionGrid[i][by][bx] = cdefHBDDirectionPoison
				req.VarianceGrid[i][by][bx] = cdefHBDVariancePoison
			}
		}
	}
}

func assertCDEFHBDGridsRemainPoisoned(t *testing.T, req FrameWorkCDEFPostFilterRequest) {
	t.Helper()
	for i := range req.DirectionGrid {
		for by := range req.DirectionGrid[i] {
			for bx := range req.DirectionGrid[i][by] {
				if req.DirectionGrid[i][by][bx] != cdefHBDDirectionPoison || req.VarianceGrid[i][by][bx] != cdefHBDVariancePoison {
					t.Fatalf("secondary-only CDEF changed grid unit=%d block=(%d,%d): dir=%d variance=%d", i, by, bx, req.DirectionGrid[i][by][bx], req.VarianceGrid[i][by][bx])
				}
			}
		}
	}
}

func fillCDEFHBDPlaneAllocation(plane frame.Plane, bitDepth, seed int) {
	stride := plane.Stride / 2
	maxSample := (1 << bitDepth) - 1
	for y := 0; y < len(plane.Pix)/plane.Stride; y++ {
		for x := 0; x < stride; x++ {
			value := (x*37 + y*53 + (x^y)*17 + seed*101) & maxSample
			binary.LittleEndian.PutUint16(plane.Pix[y*plane.Stride+x*2:], uint16(value))
		}
	}
}

func firstCDEFHBDByteDiff(a, b []byte) int {
	for i := range minInt(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return minInt(len(a), len(b))
}
