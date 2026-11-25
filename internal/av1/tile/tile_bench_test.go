package tile

import (
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"testing"
)

func coeffBenchClassName(class transform.Class) string {
	switch class {
	case transform.Class2D:
		return "2d"
	case transform.ClassHoriz:
		return "horiz"
	case transform.ClassVert:
		return "vert"
	default:
		return fmt.Sprintf("class%d", class)
	}
}

func BenchmarkCoeffInitLevels(b *testing.B) {
	sizes := [...]TransformSize{
		TransformSize4x4,
		TransformSize8x8,
		TransformSize16x16,
		TransformSize32x32,
		TransformSize64x64,
		TransformSize4x16,
		TransformSize16x4,
		TransformSize8x32,
		TransformSize32x8,
	}
	for _, size := range sizes {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB := int(geo.maxEOB)
		scratchLen := int(geo.scratchLen)
		coeffs := make([]int16, maxEOB)
		levels := make([]uint8, scratchLen)
		for i := range coeffs {
			switch i & 7 {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(i%9 - 4)
			default:
				coeffs[i] = int16((i*53)%4095 - 2047)
			}
		}
		dims, _ := size.Dimensions()
		name := fmt.Sprintf("%dx%d_scan%dx%d", int(dims.W4)*4, int(dims.H4)*4, geo.scanWidth, geo.scanHeight)
		b.Run(name+"/impl", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := CoeffInitLevels(coeffs, size, levels); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/scalar", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				coeffInitLevelsPureGo(coeffs, int(geo.scanWidth), int(geo.scanHeight), levels, scratchLen)
			}
		})
	}
}

var motionFieldBenchmarkSink uint64

func BenchmarkTemporalMotionFieldProjectReferenceFrame(b *testing.B) {
	for _, distribution := range []string{"mixed", "zero", "extreme"} {
		b.Run(distribution, func(b *testing.B) {
			benchmarkTemporalMotionFieldProjection(b, distribution, false)
		})
	}
}

func BenchmarkTemporalMotionFieldProjectReferenceFrameScalarReference(b *testing.B) {
	for _, distribution := range []string{"mixed", "zero", "extreme"} {
		b.Run(distribution, func(b *testing.B) {
			benchmarkTemporalMotionFieldProjection(b, distribution, true)
		})
	}
}

func benchmarkTemporalMotionFieldProjection(b *testing.B, distribution string, scalarReference bool) {
	start, field, req := motionFieldBenchmarkInputs(b, distribution)
	req.StartFrame = start
	b.ReportAllocs()
	for b.Loop() {
		if scalarReference {
			if applied, err := projectTemporalMotionFieldScalarReference(field, req); err != nil || !applied {
				b.Fatalf("projection applied=%v err=%v", applied, err)
			}
		} else if applied, err := field.ProjectReferenceFrame(req); err != nil || !applied {
			b.Fatalf("projection applied=%v err=%v", applied, err)
		}
	}
	motionFieldBenchmarkSink ^= temporalMotionFieldBenchmarkChecksum(field)
}

func motionFieldBenchmarkInputs(b *testing.B, distribution string) (*ReferenceMVFrame, *TemporalMotionField, TemporalMotionProjectionRequest) {
	b.Helper()
	const rows, cols = 90, 160
	startEntries := make([]ReferenceMVEntry, rows*cols)
	for row := range rows {
		for col := range cols {
			var mv motion.Vector
			switch distribution {
			case "mixed":
				mv = motion.Vector{Row: int16((row*37+col*13)%8193 - 4096), Col: int16((col*17-row*7)%8193 - 4096)}
			case "zero":
				mv = motion.Vector{}
			case "extreme":
				switch (row + col) & 3 {
				case 0:
					mv = motion.Vector{Row: 16383, Col: -16383}
				case 1:
					mv = motion.Vector{Row: -16383, Col: 16383}
				case 2:
					mv = motion.Vector{Row: 8191, Col: -8191}
				default:
					mv = motion.Vector{Row: -8191, Col: 8191}
				}
			default:
				b.Fatalf("unknown motion distribution %q", distribution)
			}
			startEntries[row*cols+col] = ReferenceMVEntry{
				Ref:   ReferenceFrameLast,
				MV:    mv,
				Valid: true,
			}
		}
	}
	start := &ReferenceMVFrame{Rows: rows, Cols: cols, Stride: cols, Entries: startEntries}
	field := &TemporalMotionField{
		Rows:    rows,
		Cols:    cols,
		Stride:  cols,
		Entries: make([]TemporalMotionEntry, rows*cols),
	}
	var refOrderHints [referenceFrameCount]uint8
	refOrderHints[ReferenceFrameLast] = 255
	return start, field, TemporalMotionProjectionRequest{
		OrderHintBits:      8,
		CurrentOrderHint:   31,
		StartOrderHint:     30,
		StartRefOrderHints: refOrderHints,
	}
}

func temporalMotionFieldBenchmarkChecksum(field *TemporalMotionField) uint64 {
	var checksum uint64
	for i, entry := range field.Entries {
		if entry.Valid {
			checksum += uint64(uint16(entry.MV.Row)) + uint64(uint16(entry.MV.Col)) + uint64(entry.RefFrameOffset)
		}
		checksum ^= uint64(i) * 0x9e3779b97f4a7c15
	}
	return checksum
}

// BenchmarkEncoderReferenceMVStack times dav1d_refmvs_find through the
// encoder entry on a 720p frame of 64x64 LAST blocks with varied vectors
// (64x64 is three quarters of the realtime encoder's light-PD1 queries).
// The blocks of the third superblock row are queried; every neighbour is
// splatted, so each query finds about four candidates.
func BenchmarkEncoderReferenceMVStack(b *testing.B) {
	const cols, rows = 320, 180
	var s BlockLoopScratch
	BindEncoderRefmvs(&s, EncoderRefmvsFrame{
		FrameMICols: cols, FrameMIRows: rows,
		Tile: BlockWalkRequest{MIColEnd: cols, MIRowEnd: rows},
	})
	refs := InterReferencesResult{Ref0: ReferenceFrameLast, Ref1: ReferenceFrameNone}
	var blocks []BlockVisit
	for sby := int32(0); sby < 3; sby++ {
		s.Refmvs.sbrowInit(sby, 16)
		for bx := uint16(0); bx < cols; bx += 16 {
			blk := BlockVisit{MICol: bx, MIRow: uint16(sby * 16), Size: BlockSize64x64,
				HaveTop: sby > 0, HaveLeft: bx > 0, Edge: edgeI444TopHasRight}
			if sby == 2 {
				blocks = append(blocks, blk)
			}
			mv := motion.Vector{Row: int16(bx%48) - 8, Col: int16(sby*4) - int16(bx%32)}
			s.Refmvs.splatInterBlock(blk.Size, int32(blk.MIRow), int32(blk.MICol),
				InterMotionResult{MV: [2]motion.Vector{mv}, References: refs, Mode: InterModeResult{Mode: InterModeNewMV}})
		}
	}
	var dst ReferenceMVStackResult
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blk := &blocks[i%len(blocks)]
		s.EncoderReferenceMVStack(&dst, refs, blk)
	}
}
