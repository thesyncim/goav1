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
	name     string
	lumaW    int // MI-aligned luma extent (multiple of 8)
	lumaH    int
	xDecC    int // chroma decimation (420: 1,1; 422: 1,0; 444: 0,0)
	yDecC    int
	mono     bool
	params   parser.CDEFParams
	readProb uint32 // percent of units with Read=true
	withSkip bool   // attach a skip map with random SkipTransform blocks
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
	for i := range unitCount {
		indexMap.Index[i] = uint8(rnd.next(uint32(tc.params.StrengthCount)))
		indexMap.Read[i] = rnd.next(100) < tc.readProb
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
		for i := range pix {
			pix[i] = byte(rnd.next(256))
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
		wu, wb, err := frameWorkApplyCDEFPlaneRows(tc.params, indexMap, skipMap, cols, rows, 0, rows, src, wantPlane, 1, input, unitDst, blockStorage[:], &wantDirs, &wantVars, wantDirGrid, wantVarGrid, plane, xDec, yDec, 0, chromaFiltering)
		if err != nil {
			t.Fatalf("u16 walk: %v", err)
		}
		wantUnits += wu
		wantBlocks += wb

		gotPlane := frame.Plane{Pix: append([]byte(nil), pix...), Stride: stride, Width: width, Height: height}
		byteScratch := make([]uint16, stride*height)
		gu, gb, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, skipMap, cols, rows, gotPlane, byteScratch, input, blockStorage[:], &gotDirs, &gotVars, gotDirGrid, gotVarGrid, plane, xDec, yDec, chromaFiltering)
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
}

// TestCDEFPlaneWalkU8IsZeroAlloc protects the in-place walk's steady-state
// allocation budget.
func TestCDEFPlaneWalkU8IsZeroAlloc(t *testing.T) {
	tc := cdefU8WalkCases()[0]
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
	allocs := testing.AllocsPerRun(8, func() {
		if _, _, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, nil, cols, rows, plane, byteScratch, input, blockStorage[:], &dirs, &vars, dirGrid, varGrid, 0, 0, 0, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("u8 CDEF walk allocates: %v allocs/op", allocs)
	}
}
