package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestTemporalMotionFieldProjectMVMatchesLibaom(t *testing.T) {
	tests := []struct {
		name string
		ref  motion.Vector
		num  int
		den  int
		want motion.Vector
	}{
		{name: "forward scale", ref: motion.Vector{Row: 64, Col: -32}, num: 2, den: 4, want: motion.Vector{Row: 32, Col: -16}},
		{name: "negative numerator", ref: motion.Vector{Row: 65, Col: -65}, num: -3, den: 5, want: motion.Vector{Row: -39, Col: 39}},
		{name: "clamps distance and mv range", ref: motion.Vector{Row: 20000, Col: -20000}, num: 99, den: 1, want: motion.Vector{Row: 16383, Col: -16383}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := motionFieldProjectMV(tt.ref, tt.num, tt.den)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("projected=%+v want %+v", got, tt.want)
			}
		})
	}
	if _, err := motionFieldProjectMV(motion.Vector{}, 1, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("zero denominator err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTemporalMotionFieldProjectReferenceFrameMatchesLibaomPlacement(t *testing.T) {
	start := newReferenceMVFrameForTest(t, 16, 16)
	start.Entries[4*start.Stride+4] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 64, Col: 128},
		Valid: true,
	}
	field := newTemporalMotionFieldForTest(t, 16, 16)
	applied, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
		StartFrame:         start,
		OrderHintBits:      5,
		CurrentOrderHint:   8,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 0},
		Backward:           true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !applied {
		t.Fatal("projection was not applied")
	}
	for row := 0; row < field.Rows; row++ {
		for col := 0; col < field.Cols; col++ {
			got := field.Entries[row*field.Stride+col]
			if row == 3 && col == 2 {
				want := TemporalMotionEntry{
					MV:             motion.Vector{Row: 64, Col: 128},
					RefFrameOffset: 4,
					Valid:          true,
				}
				if got != want {
					t.Fatalf("entry (%d,%d)=%+v want %+v", row, col, got, want)
				}
				continue
			}
			if got.Valid {
				t.Fatalf("unexpected entry (%d,%d)=%+v", row, col, got)
			}
		}
	}
}

func TestTemporalMotionFieldProjectReferenceFrameFiltersLikeLibaom(t *testing.T) {
	t.Run("dimension mismatch skips", func(t *testing.T) {
		start := newReferenceMVFrameForTest(t, 8, 8)
		field := newTemporalMotionFieldForTest(t, 16, 16)
		applied, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
			StartFrame:         start,
			OrderHintBits:      5,
			CurrentOrderHint:   8,
			StartOrderHint:     4,
			StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 0},
		})
		if err != nil {
			t.Fatal(err)
		}
		if applied {
			t.Fatal("dimension mismatch should skip projection")
		}
	})

	t.Run("nonpositive ref offset skips", func(t *testing.T) {
		start := newReferenceMVFrameForTest(t, 16, 16)
		start.Entries[0] = ReferenceMVEntry{
			Ref:   ReferenceFrameLast,
			MV:    motion.Vector{Row: 64, Col: 0},
			Valid: true,
		}
		field := newTemporalMotionFieldForTest(t, 16, 16)
		applied, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
			StartFrame:         start,
			OrderHintBits:      5,
			CurrentOrderHint:   8,
			StartOrderHint:     4,
			StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 6},
		})
		if err != nil {
			t.Fatal(err)
		}
		if !applied {
			t.Fatal("valid dimensions should still apply projection pass")
		}
		if field.Entries[0].Valid {
			t.Fatalf("nonpositive offset projected entry=%+v", field.Entries[0])
		}
	})

	t.Run("block position outside window skips", func(t *testing.T) {
		start := newReferenceMVFrameForTest(t, 16, 16)
		start.Entries[4*start.Stride+4] = ReferenceMVEntry{
			Ref:   ReferenceFrameLast,
			MV:    motion.Vector{Row: 0, Col: 2048},
			Valid: true,
		}
		field := newTemporalMotionFieldForTest(t, 16, 16)
		_, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
			StartFrame:         start,
			OrderHintBits:      5,
			CurrentOrderHint:   8,
			StartOrderHint:     4,
			StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 0},
			Backward:           true,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i, entry := range field.Entries {
			if entry.Valid {
				t.Fatalf("entry %d=%+v want invalid", i, entry)
			}
		}
	})

	if _, err := newTemporalMotionFieldForTest(t, 16, 16).ProjectReferenceFrame(TemporalMotionProjectionRequest{
		StartFrame:         newReferenceMVFrameForTest(t, 16, 16),
		OrderHintBits:      0,
		CurrentOrderHint:   8,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 0},
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad order hint bits err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTemporalMotionFieldProjectReferenceFrameAllocs(t *testing.T) {
	start := newReferenceMVFrameForTest(t, 16, 16)
	start.Entries[4*start.Stride+4] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 64, Col: 128},
		Valid: true,
	}
	field := newTemporalMotionFieldForTest(t, 16, 16)
	req := TemporalMotionProjectionRequest{
		StartFrame:         start,
		OrderHintBits:      5,
		CurrentOrderHint:   8,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint32{ReferenceFrameLast: 0},
		Backward:           true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		field.Clear()
		if _, err := field.ProjectReferenceFrame(req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ProjectReferenceFrame allocated: %f", allocs)
	}
}

func newTemporalMotionFieldForTest(t *testing.T, miRows uint32, miCols uint32) *TemporalMotionField {
	t.Helper()
	need, err := ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		t.Fatal(err)
	}
	field := &TemporalMotionField{}
	if err := field.Init(miRows, miCols, make([]TemporalMotionEntry, need)); err != nil {
		t.Fatal(err)
	}
	return field
}
