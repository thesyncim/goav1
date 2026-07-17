package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestReferenceMVFrameInitMatchesLibaomHalfMIGrid(t *testing.T) {
	need, err := ReferenceMVFrameEntries(5, 7)
	if err != nil {
		t.Fatal(err)
	}
	if need != 12 {
		t.Fatalf("entries=%d want 12", need)
	}
	entries := make([]ReferenceMVEntry, need+1)
	for i := range entries {
		entries[i] = ReferenceMVEntry{Ref: ReferenceFrameGolden, MV: motion.Vector{Row: 3, Col: -5}, Valid: true}
	}

	var frame ReferenceMVFrame
	if err := frame.Init(5, 7, entries); err != nil {
		t.Fatal(err)
	}
	if frame.Rows != 3 || frame.Cols != 4 || frame.Stride != 4 || len(frame.Entries) != need {
		t.Fatalf("frame=%+v want rows=3 cols=4 stride=4 entries=%d", frame, need)
	}
	for i, entry := range frame.Entries {
		if entry != (ReferenceMVEntry{Ref: ReferenceFrameNone}) {
			t.Fatalf("entry %d=%+v want NONE", i, entry)
		}
	}
	if !entries[need].Valid || entries[need].Ref != ReferenceFrameGolden {
		t.Fatalf("caller storage past MV_REF grid was modified: %+v", entries[need])
	}
}

func TestReferenceMVFrameEntriesRejectsOversizedHalfMIGrid(t *testing.T) {
	if _, err := ReferenceMVFrameEntries(1<<17, 8); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("ReferenceMVFrameEntries oversized rows err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ReferenceMVFrameEntries(^uint32(0), 8); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("ReferenceMVFrameEntries wrapping rows err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReferenceMVFrameMarkBlockIntraClearsLibaomMVREFCells(t *testing.T) {
	frame := newReferenceMVFrameForTest(t, 6, 8)
	for i := range frame.Entries {
		frame.Entries[i] = ReferenceMVEntry{
			Ref:   ReferenceFrameLast,
			MV:    motion.Vector{Row: 9, Col: -11},
			Valid: true,
		}
	}

	err := frame.MarkBlock(ReferenceMVFrameBlockRequest{
		MICol:     2,
		MIRow:     2,
		VisibleW4: 3,
		VisibleH4: 4,
		Prediction: BlockPredictionModeResult{
			Valid: true,
			Intra: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for row := 0; row < int(frame.Rows); row++ {
		for col := 0; col < int(frame.Cols); col++ {
			entry := frame.Entries[row*int(frame.Stride)+col]
			inClearedBlock := row >= 1 && row <= 2 && col >= 1 && col <= 2
			if inClearedBlock {
				if entry != (ReferenceMVEntry{Ref: ReferenceFrameNone}) {
					t.Fatalf("entry (%d,%d)=%+v want NONE", row, col, entry)
				}
				continue
			}
			if !entry.Valid || entry.Ref != ReferenceFrameLast {
				t.Fatalf("entry (%d,%d)=%+v want preserved LAST", row, col, entry)
			}
		}
	}
}

func TestReferenceMVFrameMarkBlockInterCopiesLibaomMVREFCells(t *testing.T) {
	frame := newReferenceMVFrameForTest(t, 8, 8)
	want := ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 12, Col: -34},
		Valid: true,
	}

	err := frame.MarkBlock(ReferenceMVFrameBlockRequest{
		MICol:     2,
		MIRow:     2,
		VisibleW4: 4,
		VisibleH4: 4,
		Prediction: BlockPredictionModeResult{
			Valid:            true,
			InterMotionValid: true,
			InterMotion: InterMotionResult{
				References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
				MV:         [2]motion.Vector{want.MV},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for row := 0; row < int(frame.Rows); row++ {
		for col := 0; col < int(frame.Cols); col++ {
			entry := frame.Entries[row*int(frame.Stride)+col]
			if row >= 1 && row <= 2 && col >= 1 && col <= 2 {
				if entry != want {
					t.Fatalf("entry (%d,%d)=%+v want %+v", row, col, entry, want)
				}
				continue
			}
			if entry != (ReferenceMVEntry{Ref: ReferenceFrameNone}) {
				t.Fatalf("entry (%d,%d)=%+v want NONE", row, col, entry)
			}
		}
	}
}

func TestReferenceMVFrameMarkBlockInterFiltersSideAndLimitLikeLibaom(t *testing.T) {
	t.Run("second eligible ref overwrites first", func(t *testing.T) {
		frame := newReferenceMVFrameForTest(t, 4, 4)
		err := frame.MarkBlock(ReferenceMVFrameBlockRequest{
			VisibleW4: 4,
			VisibleH4: 4,
			Prediction: compoundPredictionForRefMVFrameTest(
				motion.Vector{Row: 1, Col: 2},
				motion.Vector{Row: 3, Col: 4},
			),
		})
		if err != nil {
			t.Fatal(err)
		}
		want := ReferenceMVEntry{Ref: ReferenceFrameBWD, MV: motion.Vector{Row: 3, Col: 4}, Valid: true}
		if got := frame.Entries[0]; got != want {
			t.Fatalf("entry=%+v want %+v", got, want)
		}
	})

	t.Run("future or same-side refs are skipped", func(t *testing.T) {
		frame := newReferenceMVFrameForTest(t, 4, 4)
		var side [referenceFrameCount]int8
		side[ReferenceFrameLast] = 1
		err := frame.MarkBlock(ReferenceMVFrameBlockRequest{
			VisibleW4:    4,
			VisibleH4:    4,
			RefFrameSide: side,
			Prediction: compoundPredictionForRefMVFrameTest(
				motion.Vector{Row: 1, Col: 2},
				motion.Vector{Row: 3, Col: 4},
			),
		})
		if err != nil {
			t.Fatal(err)
		}
		want := ReferenceMVEntry{Ref: ReferenceFrameBWD, MV: motion.Vector{Row: 3, Col: 4}, Valid: true}
		if got := frame.Entries[0]; got != want {
			t.Fatalf("entry=%+v want %+v", got, want)
		}
	})

	t.Run("out of range motion vectors clear to none", func(t *testing.T) {
		frame := newReferenceMVFrameForTest(t, 4, 4)
		var side [referenceFrameCount]int8
		side[ReferenceFrameLast] = 1
		err := frame.MarkBlock(ReferenceMVFrameBlockRequest{
			VisibleW4:    4,
			VisibleH4:    4,
			RefFrameSide: side,
			Prediction: compoundPredictionForRefMVFrameTest(
				motion.Vector{Row: 1, Col: 2},
				motion.Vector{Row: refMVSLimit + 1, Col: 4},
			),
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := frame.Entries[0]; got != (ReferenceMVEntry{Ref: ReferenceFrameNone}) {
			t.Fatalf("entry=%+v want NONE", got)
		}
	})
}

func BenchmarkReferenceMVFrameMarkBlock720p(b *testing.B) {
	const (
		miRows = 180
		miCols = 320
	)
	need, err := ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		b.Fatal(err)
	}
	prediction := BlockPredictionModeResult{
		Valid:            true,
		InterMotionValid: true,
		InterMotion: InterMotionResult{
			References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
			MV:         [2]motion.Vector{{Row: 7, Col: -11}},
		},
	}
	for _, tracked := range []bool{false, true} {
		name := "untracked"
		if tracked {
			name = "tracked"
		}
		b.Run(name, func(b *testing.B) {
			entries := make([]ReferenceMVEntry, need)
			var frame ReferenceMVFrame
			if tracked {
				err = frame.InitTracked(miRows, miCols, entries)
			} else {
				err = frame.Init(miRows, miCols, entries)
			}
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				for row := uint16(0); row < miRows; row += 4 {
					for col := uint16(0); col < miCols; col += 4 {
						if err := frame.MarkBlockPtr(col, row, 4, min(uint8(4), uint8(miRows-row)), &prediction, [referenceFrameCount]int8{}); err != nil {
							b.Fatal(err)
						}
					}
				}
			}
		})
	}
}

func newReferenceMVFrameForTest(t *testing.T, miRows uint32, miCols uint32) *ReferenceMVFrame {
	t.Helper()
	need, err := ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		t.Fatal(err)
	}
	frame := &ReferenceMVFrame{}
	if err := frame.Init(miRows, miCols, make([]ReferenceMVEntry, need)); err != nil {
		t.Fatal(err)
	}
	return frame
}

func compoundPredictionForRefMVFrameTest(first motion.Vector, second motion.Vector) BlockPredictionModeResult {
	return BlockPredictionModeResult{
		Valid:            true,
		InterMotionValid: true,
		InterMotion: InterMotionResult{
			References: InterReferencesResult{
				Ref:      [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameBWD},
				Compound: true,
			},
			MV: [2]motion.Vector{first, second},
		},
	}
}
