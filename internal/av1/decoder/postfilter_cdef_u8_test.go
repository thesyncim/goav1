package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// Differential proof for the in-place 8-bit CDEF frame walk
// (postfilter_cdef_u8.go): on identical planes, index maps, skip maps, and
// strengths, frameWorkApplyCDEFPlaneRowsU8 must leave every frame byte
// identical to the uint16 snapshot walk (frameWorkApplyCDEFPlaneRows over a
// LoadSamplePlaneFull snapshot plus storeCDEFUnit), with identical
// direction/variance grids and unit/block counts, across full/partial units,
// skip patterns, direction-only luma passes, and chroma geometries.

type cdefU8WalkCase struct {
	name        string
	lumaW       int // MI-aligned luma extent (multiple of 8)
	lumaH       int
	xDecC       int // chroma decimation (420: 1,1; 422: 1,0; 444: 0,0)
	yDecC       int
	mono        bool
	params      parser.CDEFParams
	readProb    uint32   // percent of units with Read=true
	withSkip    bool     // attach a skip map with random SkipTransform blocks
	noPrimary   bool     // all entries use secondary-only strengths; direction grids must stay untouched
	unitIndex   []uint8  // optional exact per-unit strength indexes, row-major
	unitRead    []bool   // optional exact per-unit read mask, row-major
	activeMasks []uint64 // optional explicit active 8x8 masks, row-major
	poisonInput bool     // poison the shared tap scratch before each U8 walk
	lowRange    bool     // use low-range samples so secondary taps produce visible deltas
}

func cdefU8WalkCases() []cdefU8WalkCase {
	mk := func(damping uint8, strengths ...[2]uint8) parser.CDEFParams {
		var p parser.CDEFParams
		p.Damping = damping
		p.StrengthCount = uint8(len(strengths))
		for i, s := range strengths {
			p.YStrength[i] = s[0]
			p.UVStrength[i] = s[1]
		}
		return p
	}
	return []cdefU8WalkCase{
		{name: "multi_unit_420", lumaW: 192, lumaH: 136, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{9<<2 | 2, 5<<2 | 1}, [2]uint8{0, 0}, [2]uint8{15<<2 | 3, 4<<2 | 2}, [2]uint8{1 << 2, 2 << 2}), readProb: 100},
		{name: "partial_units_420", lumaW: 136, lumaH: 72, xDecC: 1, yDecC: 1,
			params: mk(5, [2]uint8{7<<2 | 1, 3<<2 | 2}), readProb: 100},
		{name: "sparse_read_420", lumaW: 256, lumaH: 192, xDecC: 1, yDecC: 1,
			params: mk(3, [2]uint8{12 << 2, 6<<2 | 3}, [2]uint8{2<<2 | 2, 0}), readProb: 45, withSkip: true},
		{name: "direction_only_luma", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{0, 5<<2 | 2}, [2]uint8{0, 3 << 2}), readProb: 100},
		{name: "secondary_only_no_direction_420", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}, [2]uint8{0, 0}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_420", lumaW: 192, lumaH: 192, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_420_partial_crop_skip", lumaW: 200, lumaH: 200, xDecC: 1, yDecC: 1,
			params: mk(5, [2]uint8{1, 2}), readProb: 100, withSkip: true, noPrimary: true},
		{name: "secondary_only_byte_interior_422", lumaW: 192, lumaH: 192, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_444", lumaW: 192, lumaH: 192, xDecC: 0, yDecC: 0,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_mono", lumaW: 192, lumaH: 192, mono: true,
			params: mk(4, [2]uint8{2, 0}), readProb: 100, noPrimary: true},
		{name: "secondary_byte_mixed_skip_transitions", lumaW: 384, lumaH: 256, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}, [2]uint8{5<<2 | 2, 5<<2 | 2}, [2]uint8{4, 4}), readProb: 100, withSkip: true, poisonInput: true, lowRange: true,
			unitIndex: []uint8{
				0, 0, 0, 0, 0, 0,
				0, 1, 0, 2, 0, 1,
				0, 0, 1, 0, 2, 0,
				0, 0, 0, 0, 0, 0,
			},
			unitRead: []bool{
				true, true, true, true, true, true,
				true, true, true, false, true, true,
				true, true, true, true, false, true,
				true, true, true, true, true, true,
			}},
		{name: "secondary_only_no_luma_direction_422", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{0, 2}, [2]uint8{0, 2}), readProb: 100, noPrimary: true},
		{name: "luma_only", lumaW: 128, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(6, [2]uint8{9<<2 | 3, 0}), readProb: 80, withSkip: true},
		{name: "chroma_422", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{8<<2 | 2, 6<<2 | 1}), readProb: 90},
		{name: "chroma_444", lumaW: 128, lumaH: 72, xDecC: 0, yDecC: 0,
			params: mk(5, [2]uint8{10<<2 | 1, 7<<2 | 2}), readProb: 100, withSkip: true},
		{name: "single_unit", lumaW: 40, lumaH: 24, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{13<<2 | 2, 9<<2 | 3}), readProb: 100},
		{name: "single_unit_row", lumaW: 320, lumaH: 56, xDecC: 1, yDecC: 1,
			params: mk(3, [2]uint8{6<<2 | 1, 2<<2 | 2}, [2]uint8{0, 4 << 2}), readProb: 100, withSkip: true},
		{name: "monochrome", lumaW: 192, lumaH: 136, mono: true,
			params: mk(4, [2]uint8{11<<2 | 2, 0}), readProb: 100},
	}
}

// cdefU8TestRand is a tiny deterministic LCG (libaom test-style).
type cdefU8TestRand struct{ state uint32 }

func (r *cdefU8TestRand) next(n uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % n
}

func TestCDEFPlaneWalkU8MatchesSnapshot(t *testing.T) {
	for _, tc := range cdefU8WalkCases() {
		t.Run(tc.name, func(t *testing.T) {
			runCDEFU8WalkDifferential(t, tc)
		})
	}
}

func runCDEFU8WalkDifferential(t *testing.T, tc cdefU8WalkCase) {
	t.Helper()
	rnd := &cdefU8TestRand{state: 0x5538 ^ uint32(tc.lumaW*31+tc.lumaH)}

	cols := (tc.lumaW + cdef.BlockSize - 1) / cdef.BlockSize
	rows := (tc.lumaH + cdef.BlockSize - 1) / cdef.BlockSize
	unitCount := cols * rows

	indexMap := threading.FrameWorkCDEFIndexMap{
		Stride: uint16(cols),
		Rows:   uint16(rows),
		Index:  make([]uint8, unitCount),
		Read:   make([]bool, unitCount),
	}
	if len(tc.unitIndex) != 0 && len(tc.unitIndex) != unitCount {
		t.Fatalf("exact unit index map length=%d want=%d", len(tc.unitIndex), unitCount)
	}
	if len(tc.unitRead) != 0 && len(tc.unitRead) != unitCount {
		t.Fatalf("exact unit read map length=%d want=%d", len(tc.unitRead), unitCount)
	}
	for i := range unitCount {
		if len(tc.unitIndex) != 0 {
			indexMap.Index[i] = tc.unitIndex[i]
		} else {
			indexMap.Index[i] = uint8(rnd.next(uint32(tc.params.StrengthCount)))
		}
		if len(tc.unitRead) != 0 {
			indexMap.Read[i] = tc.unitRead[i]
		} else {
			indexMap.Read[i] = rnd.next(100) < tc.readProb
		}
	}

	var skipMap *FrameWorkLoopFilterMap
	if tc.withSkip {
		miCols := tc.lumaW / 4
		miRows := tc.lumaH / 4
		m := threading.FrameWorkLoopFilterMap{
			Stride:  uint16(miCols),
			Rows:    uint16(miRows),
			Records: make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows),
		}
		for i := range m.Records {
			m.Records[i].Valid = true
			m.Records[i].SkipTransform = rnd.next(3) == 0
		}
		if len(tc.activeMasks) != 0 {
			if len(tc.activeMasks) != unitCount {
				t.Fatal("active mask count does not match unit grid")
			}
			for y := 0; y < miRows; y++ {
				for x := 0; x < miCols; x++ {
					unit := (y/16)*cols + x/16
					bit := ((y%16)/2)*8 + (x%16)/2
					m.Records[y*miCols+x].SkipTransform = tc.activeMasks[unit]&(uint64(1)<<uint(bit)) == 0
				}
			}
		}
		skipMap = &m
	}

	planes := 1
	if !tc.mono {
		planes = 3
	}
	chromaFiltering := !tc.mono && frameWorkCDEFChromaHasFiltering(tc.params)

	// Shared direction/variance grids, like the production walk.
	wantDirGrid := make([]cdef.DirectionGrid, unitCount)
	wantVarGrid := make([]cdef.VarianceGrid, unitCount)
	gotDirGrid := make([]cdef.DirectionGrid, unitCount)
	gotVarGrid := make([]cdef.VarianceGrid, unitCount)
	var wantDirs, gotDirs cdef.DirectionGrid
	var wantVars, gotVars cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	input := make([]uint16, cdef.InputBufferSize)
	unitDst := make([]uint16, cdef.InputBufferSize)
	for by := range cdef.NBlocks {
		for bx := range cdef.NBlocks {
			wantDirs[by][bx] = 5
			gotDirs[by][bx] = 5
			wantVars[by][bx] = 0x12345
			gotVars[by][bx] = 0x12345
		}
	}
	for i := range unitCount {
		for by := range cdef.NBlocks {
			for bx := range cdef.NBlocks {
				wantDirGrid[i][by][bx] = 5
				gotDirGrid[i][by][bx] = 5
				wantVarGrid[i][by][bx] = 0x12345
				gotVarGrid[i][by][bx] = 0x12345
			}
		}
	}

	var wantUnits, gotUnits uint32
	var wantBlocks, gotBlocks uint32
	for plane := 0; plane < planes; plane++ {
		xDec, yDec := 0, 0
		if plane != 0 {
			xDec, yDec = tc.xDecC, tc.yDecC
		}
		width := tc.lumaW >> xDec
		height := tc.lumaH >> yDec
		stride := width + 16 // stride padding with garbage to pin clamping
		pix := make([]byte, stride*height)
		pixelRange := uint32(256)
		if tc.lowRange {
			pixelRange = 16
		}
		for i := range pix {
			pix[i] = byte(rnd.next(pixelRange))
		}
		processPlane := frameWorkCDEFPlaneHasFiltering(tc.params, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !processPlane {
			continue
		}

		wantPlane := frame.Plane{Pix: append([]byte(nil), pix...), Stride: stride, Width: width, Height: height}
		scratch := make([]uint16, stride*height)
		src, _, err := frame.LoadSamplePlaneFull(scratch, wantPlane, 1)
		if err != nil {
			t.Fatalf("snapshot load: %v", err)
		}
		wu, wb, err := frameWorkApplyCDEFPlaneRows(tc.params, indexMap, skipMap, cols, rows, 0, rows, src, wantPlane, 1, input, unitDst, blockStorage[:], &wantDirs, &wantVars, wantDirGrid, wantVarGrid, plane, xDec, yDec, 0, chromaFiltering, nil)
		if err != nil {
			t.Fatalf("u16 walk: %v", err)
		}
		wantUnits += wu
		wantBlocks += wb

		gotPlane := frame.Plane{Pix: append([]byte(nil), pix...), Stride: stride, Width: width, Height: height}
		byteScratch := make([]uint16, stride*height)
		if tc.poisonInput {
			for i := range input {
				input[i] = 0xdeed
			}
		}
		gu, gb, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, skipMap, cols, rows, 0, rows, gotPlane, byteScratch, FrameWorkCDEFPostFilterU8BandBoundary{}, input, blockStorage[:], &gotDirs, &gotVars, gotDirGrid, gotVarGrid, plane, xDec, yDec, chromaFiltering, nil)
		if err != nil {
			t.Fatalf("u8 walk: %v", err)
		}
		gotUnits += gu
		gotBlocks += gb

		for i := range wantPlane.Pix {
			if gotPlane.Pix[i] != wantPlane.Pix[i] {
				t.Fatalf("plane %d byte mismatch at row=%d col=%d: got=%d want=%d",
					plane, i/stride, i%stride, gotPlane.Pix[i], wantPlane.Pix[i])
			}
		}
	}
	if gotUnits != wantUnits || gotBlocks != wantBlocks {
		t.Fatalf("count mismatch: got units=%d blocks=%d want units=%d blocks=%d", gotUnits, gotBlocks, wantUnits, wantBlocks)
	}
	for i := range wantDirGrid {
		if gotDirGrid[i] != wantDirGrid[i] {
			t.Fatalf("direction grid %d differs", i)
		}
		if gotVarGrid[i] != wantVarGrid[i] {
			t.Fatalf("variance grid %d differs", i)
		}
	}
	if tc.noPrimary {
		for by := range cdef.NBlocks {
			for bx := range cdef.NBlocks {
				if gotDirs[by][bx] != 5 || gotVars[by][bx] != 0x12345 {
					t.Fatalf("primary-zero luma changed standalone grids at (%d,%d): dir=%d variance=%d", bx, by, gotDirs[by][bx], gotVars[by][bx])
				}
			}
		}
		for i := range unitCount {
			for by := range cdef.NBlocks {
				for bx := range cdef.NBlocks {
					if gotDirGrid[i][by][bx] != 5 || gotVarGrid[i][by][bx] != 0x12345 {
						t.Fatalf("primary-zero unit %d changed grids at (%d,%d): dir=%d variance=%d", i, bx, by, gotDirGrid[i][by][bx], gotVarGrid[i][by][bx])
					}
				}
			}
		}
	}
}

// TestCDEFPlaneWalkU8IsZeroAlloc protects the in-place walk's steady-state
// allocation budget.
func TestCDEFPlaneWalkU8IsZeroAlloc(t *testing.T) {
	tc := cdefU8WalkCases()[0]
	if allocs := cdefPlaneWalkU8Allocs(t, tc); allocs != 0 {
		t.Fatalf("u8 CDEF walk allocates: %v allocs/op", allocs)
	}
}

func TestCDEFPlaneWalkU8SecondaryByteIsZeroAlloc(t *testing.T) {
	if !cdef.U8SecondaryOnlyByteInteriorAvailable() {
		t.Skip("secondary-only byte kernel is unavailable in this build")
	}
	var tc cdefU8WalkCase
	for _, candidate := range cdefU8WalkCases() {
		if candidate.name == "secondary_only_byte_interior_420" {
			tc = candidate
			break
		}
	}
	if tc.name == "" {
		t.Fatal("secondary-only byte walk case is missing")
	}
	if allocs := cdefPlaneWalkU8Allocs(t, tc); allocs != 0 {
		t.Fatalf("secondary-only byte u8 CDEF walk allocates: %v allocs/op", allocs)
	}
}

func cdefPlaneWalkU8Allocs(t *testing.T, tc cdefU8WalkCase) float64 {
	t.Helper()
	rnd := &cdefU8TestRand{state: 0xa110c}
	cols := (tc.lumaW + cdef.BlockSize - 1) / cdef.BlockSize
	rows := (tc.lumaH + cdef.BlockSize - 1) / cdef.BlockSize
	unitCount := cols * rows
	indexMap := threading.FrameWorkCDEFIndexMap{
		Stride: uint16(cols),
		Rows:   uint16(rows),
		Index:  make([]uint8, unitCount),
		Read:   make([]bool, unitCount),
	}
	for i := range unitCount {
		indexMap.Read[i] = true
	}
	width := tc.lumaW
	height := tc.lumaH
	stride := width + 8
	plane := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	for i := range plane.Pix {
		plane.Pix[i] = byte(rnd.next(256))
	}
	byteScratch := make([]uint16, stride*height)
	input := make([]uint16, cdef.InputBufferSize)
	dirGrid := make([]cdef.DirectionGrid, unitCount)
	varGrid := make([]cdef.VarianceGrid, unitCount)
	var dirs cdef.DirectionGrid
	var vars cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	var skipMap *FrameWorkLoopFilterMap
	if len(tc.activeMasks) != 0 {
		miCols, miRows := width/4, height/4
		records := make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
		for y := 0; y < miRows; y++ {
			for x := 0; x < miCols; x++ {
				unit, bit := (y/16)*cols+x/16, ((y%16)/2)*8+(x%16)/2
				records[y*miCols+x].Valid = true
				records[y*miCols+x].SkipTransform = tc.activeMasks[unit]&(uint64(1)<<uint(bit)) == 0
			}
		}
		skipMap = &FrameWorkLoopFilterMap{Records: records, Stride: uint16(miCols), Rows: uint16(miRows)}
	}
	return testing.AllocsPerRun(8, func() {
		if _, _, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, skipMap, cols, rows, 0, rows, plane, byteScratch, FrameWorkCDEFPostFilterU8BandBoundary{}, input, blockStorage[:], &dirs, &vars, dirGrid, varGrid, 0, 0, 0, true, nil); err != nil {
			t.Fatal(err)
		}
	})
}
