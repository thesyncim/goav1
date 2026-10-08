package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

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
