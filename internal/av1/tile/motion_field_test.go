package tile

import (
	"errors"
	"math/rand"
	"reflect"
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
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
		Backward:           true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !applied {
		t.Fatal("projection was not applied")
	}
	for row := 0; row < int(field.Rows); row++ {
		for col := 0; col < int(field.Cols); col++ {
			got := field.Entries[row*int(field.Stride)+col]
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
			StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
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
			StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 6},
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
			StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
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
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad order hint bits err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTemporalMotionFieldProjectReferenceFrameRunsMatchScalar(t *testing.T) {
	const (
		miRows = 34
		miCols = 38
	)
	start := newReferenceMVFrameForTest(t, miRows, miCols)
	rnd := rand.New(rand.NewSource(0x5eed))
	refs := [...]ReferenceFrame{ReferenceFrameLast, ReferenceFrameGolden, ReferenceFrameBWD}
	for row := 0; row < int(start.Rows); row++ {
		line := start.Entries[row*int(start.Stride) : row*int(start.Stride)+int(start.Cols)]
		for col := 0; col < len(line); {
			runEnd := min(len(line), col+1+rnd.Intn(12))
			entry := ReferenceMVEntry{Ref: ReferenceFrameNone}
			if rnd.Intn(4) != 0 {
				entry = ReferenceMVEntry{
					Ref:   refs[rnd.Intn(len(refs))],
					MV:    motion.Vector{Row: int16(rnd.Intn(4096) - 2048), Col: int16(rnd.Intn(4096) - 2048)},
					Valid: true,
				}
			}
			for i := col; i < runEnd; i++ {
				line[i] = entry
			}
			setReferenceMVEntryRun(&line[col], uint8(runEnd-col))
			col = runEnd
		}
	}
	start.runsValid = true
	copiedEntries := make([]ReferenceMVEntry, len(start.Entries))
	copy(copiedEntries, start.Entries)
	for i := range copiedEntries {
		if got, want := referenceMVEntryRun(&copiedEntries[i]), referenceMVEntryRun(&start.Entries[i]); got != want {
			t.Fatalf("copied run metadata entry %d=%d want %d", i, got, want)
		}
	}
	copiedStart := *start
	copiedStart.Entries = copiedEntries
	start = &copiedStart

	for _, backward := range []bool{false, true} {
		got := newTemporalMotionFieldForTest(t, miRows, miCols)
		want := newTemporalMotionFieldForTest(t, miRows, miCols)
		for i := range got.Entries {
			stale := TemporalMotionEntry{MV: motion.Vector{Row: int16(i), Col: -int16(i)}, RefFrameOffset: 1, Valid: i%3 == 0}
			got.Entries[i] = stale
			want.Entries[i] = stale
		}
		req := TemporalMotionProjectionRequest{
			StartFrame:       start,
			OrderHintBits:    5,
			CurrentOrderHint: 24,
			StartOrderHint:   20,
			StartRefOrderHints: [referenceFrameCount]uint8{
				ReferenceFrameLast:   10,
				ReferenceFrameGolden: 16,
				ReferenceFrameBWD:    22,
			},
			Backward: backward,
		}
		applied, err := got.ProjectReferenceFrame(req)
		if err != nil {
			t.Fatalf("backward=%v optimized: %v", backward, err)
		}
		wantApplied, err := projectReferenceFrameScalar(want, req)
		if err != nil {
			t.Fatalf("backward=%v scalar: %v", backward, err)
		}
		if applied != wantApplied || !reflect.DeepEqual(got, want) {
			t.Fatalf("backward=%v run projection differs from scalar", backward)
		}
	}
}

func projectReferenceFrameScalar(f *TemporalMotionField, req TemporalMotionProjectionRequest) (bool, error) {
	startToCurrent, err := motionFieldRelativeOrderHint(req.OrderHintBits, req.StartOrderHint, req.CurrentOrderHint)
	if err != nil {
		return false, err
	}
	if req.Backward {
		startToCurrent = -startToCurrent
	}
	refOffsets, err := motionFieldRefOffsets(req.OrderHintBits, req.StartOrderHint, req.StartRefOrderHints)
	if err != nil {
		return false, err
	}
	start := req.StartFrame
	for row := 0; row < int(start.Rows); row++ {
		for col := 0; col < int(start.Cols); col++ {
			mvRef := start.Entries[row*int(start.Stride)+col]
			if !mvRef.Valid || !mvRef.Ref.Valid() {
				continue
			}
			refOffset := refOffsets[mvRef.Ref]
			if absInt(refOffset) > motionFieldMaxFrameDistance || refOffset <= 0 || absInt(startToCurrent) > motionFieldMaxFrameDistance {
				continue
			}
			projected, err := motionFieldProjectMV(mvRef.MV, startToCurrent, refOffset)
			if err != nil {
				return false, err
			}
			dstRow, dstCol, ok := motionFieldBlockPosition(int(f.Rows), int(f.Cols), row, col, projected, req.Backward)
			if ok {
				f.Entries[dstRow*int(f.Stride)+dstCol] = TemporalMotionEntry{MV: mvRef.MV, RefFrameOffset: uint8(refOffset), Valid: true}
			}
		}
	}
	return true, nil
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

// TestMotionFieldBlockOffsetMatchesLibaomGetBlockPosition guards the exact
// rounding direction of the get_block_position offset formula in
// /tmp/libaom-v3.13.1/av1/common/mvref_common.c (lines 886-890):
//
//	row_offset = (mv.row >= 0) ? (mv.row >> (4+MI_SIZE_LOG2))
//	                           : -((-mv.row) >> (4+MI_SIZE_LOG2));
//
// Positive values floor toward zero; negative values also floor toward zero
// (i.e. ceil toward zero in magnitude terms). The libaom MI_SIZE_LOG2=2 so the
// shift is by 6.
func TestMotionFieldBlockOffsetMatchesLibaomGetBlockPosition(t *testing.T) {
	cases := []struct {
		name string
		v    int32
		want int
	}{
		{"zero", 0, 0},
		{"plus_one_floors_to_zero", 1, 0},
		{"minus_one_floors_to_zero", -1, 0},
		{"plus_63_floors_to_zero", 63, 0},
		{"minus_63_floors_to_zero", -63, 0},
		{"plus_64_one_block", 64, 1},
		{"minus_64_one_block", -64, -1},
		{"plus_65_one_block", 65, 1},
		{"minus_65_one_block", -65, -1},
		{"plus_128_two_blocks", 128, 2},
		{"minus_128_two_blocks", -128, -2},
		{"large_positive", 16383, 16383 >> 6},
		{"large_negative", -16383, -(16383 >> 6)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := motionFieldBlockOffset(int16(tc.v)); got != tc.want {
				t.Fatalf("motionFieldBlockOffset(%d)=%d want %d", tc.v, got, tc.want)
			}
		})
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
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
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

func BenchmarkTemporalMotionFieldProjectReferenceFrame720p(b *testing.B) {
	const miRows = 180
	const miCols = 320
	need, err := ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		b.Fatal(err)
	}
	entries := make([]ReferenceMVEntry, need)
	var start ReferenceMVFrame
	if err := start.InitTracked(miRows, miCols, entries); err != nil {
		b.Fatal(err)
	}
	for row := uint16(0); row < miRows; row += 4 {
		for col := uint16(0); col < miCols; col += 4 {
			prediction := BlockPredictionModeResult{
				Valid:            true,
				InterMotionValid: true,
				InterMotion: InterMotionResult{
					MV:         [2]motion.Vector{{Row: int16(row&31) - 16, Col: int16(col&31) - 16}},
					References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
				},
			}
			if err := start.MarkBlockPtr(col, row, 4, min(uint8(4), uint8(miRows-row)), &prediction, [referenceFrameCount]int8{}); err != nil {
				b.Fatal(err)
			}
		}
	}
	fieldEntries := make([]TemporalMotionEntry, need)
	var field TemporalMotionField
	if err := field.Init(miRows, miCols, fieldEntries); err != nil {
		b.Fatal(err)
	}
	req := TemporalMotionProjectionRequest{
		StartFrame:         &start,
		OrderHintBits:      5,
		CurrentOrderHint:   8,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := field.ProjectReferenceFrame(req); err != nil {
			b.Fatal(err)
		}
	}
}

func temporalReferenceFrameForSetupTest(frame *ReferenceMVFrame, orderHint uint8, ref ReferenceFrame, refOrderHint uint8) TemporalMotionReferenceFrame {
	var hints [referenceFrameCount]uint8
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
