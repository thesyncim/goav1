package tile

import (
	"errors"
	"math/rand"
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

func TestTemporalMotionFieldProjectionMatchesScalarOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(0x7e6d5c4b))
	for _, dims := range []struct{ rows, cols int }{{9, 13}, {13, 17}, {17, 11}} {
		for trial := range 80 {
			startStride := dims.cols + 3
			startEntries := make([]ReferenceMVEntry, dims.rows*startStride+2)
			start := &ReferenceMVFrame{
				Rows:    uint16(dims.rows),
				Cols:    uint16(dims.cols),
				Stride:  uint16(startStride),
				Entries: startEntries,
			}
			for row := 0; row < dims.rows; row++ {
				for col := 0; col < startStride; col++ {
					idx := row*startStride + col
					start.Entries[idx] = ReferenceMVEntry{
						Ref:   ReferenceFrame(rng.Intn(int(referenceFrameCount) + 2)),
						MV:    motion.Vector{Row: int16(rng.Intn(1<<16) - (1 << 15)), Col: int16(rng.Intn(1<<16) - (1 << 15))},
						Valid: rng.Intn(5) != 0,
					}
				}
			}

			fieldStride := dims.cols + 5
			fieldEntries := make([]TemporalMotionEntry, dims.rows*fieldStride+4)
			for i := range fieldEntries {
				fieldEntries[i] = TemporalMotionEntry{
					MV:             motion.Vector{Row: int16(rng.Intn(1<<16) - (1 << 15)), Col: int16(rng.Intn(1<<16) - (1 << 15))},
					RefFrameOffset: uint8(rng.Intn(256)),
					Valid:          rng.Intn(2) != 0,
				}
			}
			got := &TemporalMotionField{Rows: uint16(dims.rows), Cols: uint16(dims.cols), Stride: uint16(fieldStride), Entries: append([]TemporalMotionEntry(nil), fieldEntries...)}
			want := &TemporalMotionField{Rows: uint16(dims.rows), Cols: uint16(dims.cols), Stride: uint16(fieldStride), Entries: append([]TemporalMotionEntry(nil), fieldEntries...)}

			bits := uint8(5)
			if trial&1 != 0 {
				bits = 8
			}
			orderMask := (1 << bits) - 1
			var refOrderHints [referenceFrameCount]uint8
			for i := range refOrderHints {
				refOrderHints[i] = uint8(rng.Intn(orderMask + 1))
			}
			req := TemporalMotionProjectionRequest{
				StartFrame:         start,
				OrderHintBits:      bits,
				CurrentOrderHint:   uint8(rng.Intn(orderMask + 1)),
				StartOrderHint:     uint8(rng.Intn(orderMask + 1)),
				StartRefOrderHints: refOrderHints,
				Backward:           rng.Intn(2) != 0,
			}

			gotApplied, gotErr := got.ProjectReferenceFrame(req)
			wantApplied, wantErr := projectTemporalMotionFieldScalarReference(want, req)
			if gotApplied != wantApplied || (gotErr == nil) != (wantErr == nil) {
				t.Fatalf("dims=%+v trial=%d: got applied=%v err=%v, want applied=%v err=%v", dims, trial, gotApplied, gotErr, wantApplied, wantErr)
			}
			for i := range got.Entries {
				if got.Entries[i] != want.Entries[i] {
					t.Fatalf("dims=%+v trial=%d entry %d: got %+v, want %+v", dims, trial, i, got.Entries[i], want.Entries[i])
				}
			}
		}
	}
}

func TestTemporalMotionFieldProjectionCollisionKeepsLastWriter(t *testing.T) {
	start := newReferenceMVFrameForTest(t, 32, 32)
	start.Entries[4*start.Stride+4] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 12, Col: 34},
		Valid: true,
	}
	start.Entries[4*start.Stride+5] = ReferenceMVEntry{
		Ref:   ReferenceFrameLast,
		MV:    motion.Vector{Row: 56, Col: -64},
		Valid: true,
	}
	field := newTemporalMotionFieldForTest(t, 32, 32)
	_, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
		StartFrame:         start,
		OrderHintBits:      5,
		CurrentOrderHint:   0,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := TemporalMotionEntry{MV: motion.Vector{Row: 56, Col: -64}, RefFrameOffset: 4, Valid: true}
	if got := field.Entries[4*field.Stride+4]; got != want {
		t.Fatalf("collision result=%+v want later row-major entry %+v", got, want)
	}
}

func TestTemporalMotionFieldProjectionZeroMVUsesOrigin(t *testing.T) {
	const rows, cols = 9, 13
	const startStride, fieldStride = cols + 4, cols + 6
	startEntries := make([]ReferenceMVEntry, rows*startStride)
	for row := 0; row < rows; row++ {
		for col := 0; col < startStride; col++ {
			startEntries[row*startStride+col] = ReferenceMVEntry{
				Ref:   ReferenceFrameLast,
				MV:    motion.Vector{Row: 64, Col: -128},
				Valid: true,
			}
		}
		for col := 0; col < cols; col++ {
			startEntries[row*startStride+col].MV = motion.Vector{}
		}
	}
	start := &ReferenceMVFrame{Rows: rows, Cols: cols, Stride: startStride, Entries: startEntries}
	fieldEntries := make([]TemporalMotionEntry, rows*fieldStride+3)
	for i := range fieldEntries {
		fieldEntries[i] = TemporalMotionEntry{
			MV:             motion.Vector{Row: int16(i + 1), Col: int16(-i - 1)},
			RefFrameOffset: uint8(i),
			Valid:          i&1 == 0,
		}
	}
	field := &TemporalMotionField{Rows: rows, Cols: cols, Stride: fieldStride, Entries: fieldEntries}

	if _, err := field.ProjectReferenceFrame(TemporalMotionProjectionRequest{
		StartFrame:         start,
		OrderHintBits:      5,
		CurrentOrderHint:   8,
		StartOrderHint:     4,
		StartRefOrderHints: [referenceFrameCount]uint8{ReferenceFrameLast: 0},
	}); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			want := TemporalMotionEntry{MV: motion.Vector{}, RefFrameOffset: 4, Valid: true}
			if got := field.Entries[row*fieldStride+col]; got != want {
				t.Fatalf("origin (%d,%d)=%+v want %+v", row, col, got, want)
			}
		}
		for col := cols; col < fieldStride; col++ {
			idx := row*fieldStride + col
			want := TemporalMotionEntry{
				MV:             motion.Vector{Row: int16(idx + 1), Col: int16(-idx - 1)},
				RefFrameOffset: uint8(idx),
				Valid:          idx&1 == 0,
			}
			if got := field.Entries[idx]; got != want {
				t.Fatalf("padding entry %d=%+v want unchanged %+v", idx, got, want)
			}
		}
	}
	for i := rows * fieldStride; i < len(field.Entries); i++ {
		want := TemporalMotionEntry{
			MV:             motion.Vector{Row: int16(i + 1), Col: int16(-i - 1)},
			RefFrameOffset: uint8(i),
			Valid:          i&1 == 0,
		}
		if got := field.Entries[i]; got != want {
			t.Fatalf("tail entry %d=%+v want unchanged %+v", i, got, want)
		}
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

// projectTemporalMotionFieldScalarReference preserves the pre-optimization
// row-major projection loop as an oracle for the precomputed path.
func projectTemporalMotionFieldScalarReference(f *TemporalMotionField, req TemporalMotionProjectionRequest) (bool, error) {
	if err := f.validate(); err != nil {
		return false, err
	}
	start := req.StartFrame
	if start == nil {
		return false, nil
	}
	if err := validateReferenceMVFrame(start); err != nil {
		return false, err
	}
	if start.Rows != f.Rows || start.Cols != f.Cols {
		return false, nil
	}

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

	startRows := int(start.Rows)
	startCols := int(start.Cols)
	startStride := int(start.Stride)
	rows := int(f.Rows)
	cols := int(f.Cols)
	stride := int(f.Stride)
	for blkRow := 0; blkRow < startRows; blkRow++ {
		for blkCol := 0; blkCol < startCols; blkCol++ {
			mvRef := start.Entries[blkRow*startStride+blkCol]
			if !mvRef.Valid || !mvRef.Ref.Valid() {
				continue
			}
			refFrameOffset := refOffsets[mvRef.Ref]
			if absInt(refFrameOffset) > motionFieldMaxFrameDistance ||
				refFrameOffset <= 0 ||
				absInt(startToCurrent) > motionFieldMaxFrameDistance {
				continue
			}
			projected := motionFieldProjectMVScalarReference(mvRef.MV, startToCurrent, refFrameOffset)
			row, col, ok := motionFieldBlockPosition(rows, cols, blkRow, blkCol, projected, req.Backward)
			if !ok {
				continue
			}
			f.Entries[row*stride+col] = TemporalMotionEntry{MV: mvRef.MV, RefFrameOffset: uint8(refFrameOffset), Valid: true}
		}
	}
	return true, nil
}

func motionFieldProjectMVScalarReference(ref motion.Vector, num int, den int) motion.Vector {
	if den > motionFieldMaxFrameDistance {
		den = motionFieldMaxFrameDistance
	}
	if num > motionFieldMaxFrameDistance {
		num = motionFieldMaxFrameDistance
	} else if num < -motionFieldMaxFrameDistance {
		num = -motionFieldMaxFrameDistance
	}
	row := roundMotionFieldScalarReference(int64(ref.Row)*int64(num)*int64(motionFieldDivMult[den]), 14)
	col := roundMotionFieldScalarReference(int64(ref.Col)*int64(num)*int64(motionFieldDivMult[den]), 14)
	return motion.Vector{
		Row: int16(clampInt64(row, motionFieldMVLower+1, motionFieldMVUpper-1)),
		Col: int16(clampInt64(col, motionFieldMVLower+1, motionFieldMVUpper-1)),
	}
}

func roundMotionFieldScalarReference(value int64, bits uint) int64 {
	if value < 0 {
		return -((-value + (1 << (bits - 1))) >> bits)
	}
	return (value + (1 << (bits - 1))) >> bits
}
