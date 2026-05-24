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

func TestTemporalMotionFieldSetupMatchesLibaomOverlayAndLast2(t *testing.T) {
	field := newTemporalMotionFieldForTest(t, 16, 16)
	field.Entries[0] = TemporalMotionEntry{
		MV:    motion.Vector{Row: 99, Col: 99},
		Valid: true,
	}
	last := newReferenceMVFrameForTest(t, 16, 16)
	last.Entries[4*last.Stride+4] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 64, Col: 128},
		Valid: true,
	}
	last2 := newReferenceMVFrameForTest(t, 16, 16)
	last2.Entries[4*last2.Stride+4] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 128, Col: 64},
		Valid: true,
	}

	var refs [referenceFrameCount]TemporalMotionReferenceFrame
	refs[ReferenceFrameLast] = temporalReferenceFrameForSetupTest(last, 4, ReferenceFrameLast, 0)
	refs[ReferenceFrameLast].RefOrderHints[ReferenceFrameAltref] = 12
	refs[ReferenceFrameLast2] = temporalReferenceFrameForSetupTest(last2, 4, ReferenceFrameLast, 0)
	refs[ReferenceFrameGolden].OrderHint = 12

	stats, err := field.Setup(TemporalMotionSetupRequest{
		EnableOrderHint:  true,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		References:       refs,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (TemporalMotionSetupStats{Projections: 1, LastOverlay: true, RefStamp: 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if field.Entries[0].Valid {
		t.Fatalf("stale entry was not cleared: %+v", field.Entries[0])
	}
	if got := field.Entries[3*field.Stride+2]; got.Valid {
		t.Fatalf("overlay LAST projection wrote entry: %+v", got)
	}
	want := TemporalMotionEntry{
		MV:             motion.Vector{Row: 128, Col: 64},
		RefFrameOffset: 4,
		Valid:          true,
	}
	if got := field.Entries[2*field.Stride+3]; got != want {
		t.Fatalf("LAST2 entry=%+v want %+v", got, want)
	}
}

func TestTemporalMotionFieldSetupPortsLibaomProjectionBudget(t *testing.T) {
	field := newTemporalMotionFieldForTest(t, 16, 16)
	last := newReferenceMVFrameForTest(t, 16, 16)
	last.Entries[7*last.Stride+7] = ReferenceMVEntry{Ref: ReferenceFrameLast, Valid: true}
	bwd := newReferenceMVFrameForTest(t, 16, 16)
	bwd.Entries[0] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Row: 8}, Valid: true}
	altref2 := newReferenceMVFrameForTest(t, 16, 16)
	altref2.Entries[1] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Row: 16}, Valid: true}
	altref := newReferenceMVFrameForTest(t, 16, 16)
	altref.Entries[2] = ReferenceMVEntry{Ref: ReferenceFrameLast, MV: motion.Vector{Row: 24}, Valid: true}

	var refs [referenceFrameCount]TemporalMotionReferenceFrame
	refs[ReferenceFrameLast] = temporalReferenceFrameForSetupTest(last, 4, ReferenceFrameLast, 0)
	refs[ReferenceFrameLast].RefOrderHints[ReferenceFrameAltref] = 3
	refs[ReferenceFrameBWD] = temporalReferenceFrameForSetupTest(bwd, 12, ReferenceFrameLast, 8)
	refs[ReferenceFrameAltref2] = temporalReferenceFrameForSetupTest(altref2, 13, ReferenceFrameLast, 9)
	refs[ReferenceFrameAltref] = temporalReferenceFrameForSetupTest(altref, 14, ReferenceFrameLast, 10)

	stats, err := field.Setup(TemporalMotionSetupRequest{
		EnableOrderHint:  true,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		References:       refs,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (TemporalMotionSetupStats{Projections: 3, RefStamp: -1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if got := field.Entries[7*field.Stride+7]; !got.Valid {
		t.Fatalf("LAST projection missing: %+v", got)
	}
	if got := field.Entries[0]; got != (TemporalMotionEntry{MV: motion.Vector{Row: 8}, RefFrameOffset: 4, Valid: true}) {
		t.Fatalf("BWD projection=%+v", got)
	}
	if got := field.Entries[1]; got != (TemporalMotionEntry{MV: motion.Vector{Row: 16}, RefFrameOffset: 4, Valid: true}) {
		t.Fatalf("ALTREF2 projection=%+v", got)
	}
	if got := field.Entries[2]; got.Valid {
		t.Fatalf("ALTREF should be skipped after projection budget: %+v", got)
	}
}

func TestTemporalMotionFieldSetupDisabledMatchesLibaomEarlyReturn(t *testing.T) {
	field := newTemporalMotionFieldForTest(t, 16, 16)
	field.Entries[0] = TemporalMotionEntry{
		MV:             motion.Vector{Row: 5, Col: 7},
		RefFrameOffset: 3,
		Valid:          true,
	}
	stats, err := field.Setup(TemporalMotionSetupRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (TemporalMotionSetupStats{RefStamp: motionFieldMFMVStackSize - 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if got := field.Entries[0]; got != (TemporalMotionEntry{MV: motion.Vector{Row: 5, Col: 7}, RefFrameOffset: 3, Valid: true}) {
		t.Fatalf("disabled setup changed field entry=%+v", got)
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
	var setupRefs [referenceFrameCount]TemporalMotionReferenceFrame
	setupRefs[ReferenceFrameLast] = temporalReferenceFrameForSetupTest(start, 4, ReferenceFrameLast, 0)
	setup := TemporalMotionSetupRequest{
		EnableOrderHint:  true,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		References:       setupRefs,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		field.Clear()
		if _, err := field.ProjectReferenceFrame(req); err != nil {
			t.Fatal(err)
		}
		if _, err := field.Setup(setup); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("motion field projection allocated: %f", allocs)
	}
}

func temporalReferenceFrameForSetupTest(frame *ReferenceMVFrame, orderHint uint32, ref ReferenceFrame, refOrderHint uint32) TemporalMotionReferenceFrame {
	var hints [referenceFrameCount]uint32
	hints[ref] = refOrderHint
	return TemporalMotionReferenceFrame{
		Frame:         frame,
		OrderHint:     orderHint,
		RefOrderHints: hints,
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
